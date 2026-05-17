from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.utils.logger import get_logger
from config import EDU_WHISPER_MODEL

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def transcribe_video(video_path: Path) -> list[TranscriptSegment]:
    """Run faster-whisper when installed. Returns [] when unavailable."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster-whisper is not installed; auto transcript skipped.")
        return []

    model = WhisperModel(EDU_WHISPER_MODEL, device="auto", compute_type="auto")
    segments, _ = model.transcribe(str(video_path))
    return [
        TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
        for seg in segments
        if seg.text and seg.text.strip()
    ]


def manual_transcript_to_segments(transcript: str, duration_sec: float) -> list[TranscriptSegment]:
    """Split plain transcript text evenly over the video duration."""
    text = transcript.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    span = max(duration_sec / max(len(paragraphs), 1), 1.0)
    segments = []
    for idx, paragraph in enumerate(paragraphs):
        start = idx * span
        end = duration_sec if idx == len(paragraphs) - 1 else min((idx + 1) * span, duration_sec)
        segments.append(TranscriptSegment(start=start, end=end, text=paragraph))
    return segments


def align_transcript_to_chunks(
    segments: list[TranscriptSegment],
    chunks: list[dict],
) -> dict[str, str]:
    aligned: dict[str, list[str]] = {chunk["chunk_id"]: [] for chunk in chunks}
    for chunk in chunks:
        chunk_start = float(chunk["start_time"])
        chunk_end = float(chunk["end_time"])
        for segment in segments:
            overlaps = segment.start < chunk_end and segment.end > chunk_start
            if overlaps:
                aligned[chunk["chunk_id"]].append(segment.text)
    return {chunk_id: " ".join(parts).strip() for chunk_id, parts in aligned.items()}
