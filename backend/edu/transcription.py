"""
backend/edu/transcription.py
────────────────────────────────────────────────────────────
Audio transcription service using faster-whisper.

Responsible for:
    1. Extracting audio from video files (via ffmpeg subprocess)
    2. Running faster-whisper for speech-to-text
    3. Returning timestamped segments for chunk alignment
    4. Cleaning up after itself (delete temp audio)

VRAM lifecycle:
    - Whisper is loaded, used, then explicitly unloaded
    - torch.cuda.empty_cache() called after unloading
    - Must NOT coexist with Qwen in VRAM

Public API:
    transcribe_video(video_path, model_size) -> TranscriptResult
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptResult:
    """Output of the transcription pipeline."""
    full_text: str
    segments: list[dict]   # [{"start": float, "end": float, "text": str}, ...]
    language: str
    duration_sec: float

    @property
    def num_segments(self) -> int:
        return len(self.segments)


def _extract_audio(
    video_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Extract audio from video using ffmpeg.
    Outputs 16kHz mono WAV (optimal for Whisper).

    Args:
        video_path:  Path to source video
        output_path: Where to save the .wav file

    Returns:
        Path to the extracted audio file

    Raises:
        RuntimeError: If ffmpeg fails or is not installed
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz (Whisper's native rate)
        "-ac", "1",               # mono
        "-y",                     # overwrite
        str(output_path),
    ]

    logger.debug("Extracting audio: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}): {result.stderr[:500]}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg:\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced empty or missing audio file")

    logger.info("Audio extracted: %s (%.1f MB)", output_path.name, output_path.stat().st_size / 1e6)
    return output_path


def transcribe_video(
    video_path: str | Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
) -> TranscriptResult:
    """
    Transcribe a video file using faster-whisper.

    Full pipeline:
        1. Extract audio to temp WAV
        2. Load faster-whisper model
        3. Transcribe
        4. Unload model + free VRAM
        5. Delete temp audio
        6. Return segments with timestamps

    Args:
        video_path:    Path to the source video
        model_size:    Whisper model size ("tiny", "base", "small", "medium")
        device:        "auto", "cuda", or "cpu"
        compute_type:  "auto", "float16", "int8", "float32"

    Returns:
        TranscriptResult with full text and timestamped segments
    """
    import torch

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "float32"

    # Step 1: Extract audio to temp file
    temp_dir = video_path.parent
    audio_path = temp_dir / "_temp_audio.wav"

    try:
        _extract_audio(video_path, audio_path)

        # Step 2: Load Whisper
        logger.info("Loading faster-whisper model: %s (device=%s, compute=%s)", model_size, device, compute_type)

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for transcription. "
                "Install with: pip install faster-whisper"
            )

        whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        # Step 3: Transcribe
        logger.info("Transcribing audio...")
        segments_iter, info = whisper_model.transcribe(
            str(audio_path),
            beam_size=5,
            language=None,       # auto-detect
            vad_filter=True,     # filter silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # Collect segments (iterator is consumed once)
        segments = []
        for seg in segments_iter:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })

        full_text = " ".join(s["text"] for s in segments)

        logger.info(
            "Transcription complete: %d segments, %d words, lang=%s",
            len(segments), len(full_text.split()), info.language,
        )

        result = TranscriptResult(
            full_text=full_text,
            segments=segments,
            language=info.language or "unknown",
            duration_sec=info.duration or 0.0,
        )

        # Step 4: Unload Whisper and free VRAM
        del whisper_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Whisper unloaded, VRAM freed")

        return result

    finally:
        # Step 5: Clean up temp audio
        if audio_path.exists():
            audio_path.unlink()
            logger.debug("Temp audio deleted: %s", audio_path)


def transcribe_audio_file(
    audio_path: str | Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
) -> TranscriptResult:
    """
    Transcribe an audio file directly (skip ffmpeg extraction).
    Useful when audio is already extracted.
    """
    import torch

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "float32"

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper required. Install: pip install faster-whisper"
        )

    logger.info("Loading faster-whisper: %s", model_size)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        str(audio_path), beam_size=5, vad_filter=True,
    )

    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })

    full_text = " ".join(s["text"] for s in segments)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return TranscriptResult(
        full_text=full_text,
        segments=segments,
        language=info.language or "unknown",
        duration_sec=info.duration or 0.0,
    )
