"""
backend/edu/chunking.py
────────────────────────────────────────────────────────────
Video chunking and frame sampling for Edu-VQAGuider.

Chunking is metadata-only — we do NOT physically split the video file.
Frames are sampled by seeking to timestamps using cv2.

Public API:
    create_chunks(video_path, chunk_duration_sec) → list[ChunkMeta]
    sample_chunk_frames(video_path, chunk, num_frames, output_dir) → list[str]
"""
from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkMeta:
    """Metadata for a single video chunk. No physical file created."""
    chunk_id: str
    video_id: str
    start_time: float     # seconds
    end_time: float       # seconds
    transcript_text: str = ""
    visual_summary: Optional[str] = None
    frame_paths: list[str] = field(default_factory=list)
    embedding_path: Optional[str] = None

    def to_db_row(self) -> dict:
        """Convert to a dict matching the video_chunks table schema."""
        return {
            "id": self.chunk_id,
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "transcript_text": self.transcript_text,
            "visual_summary": self.visual_summary,
            "frame_paths_json": json.dumps(self.frame_paths),
            "embedding_path": self.embedding_path,
        }


def _run_ffprobe(video_path: str | Path) -> dict:
    """Run ffprobe to get video metadata."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-of", "json",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Please install ffmpeg (which includes ffprobe).")
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No video streams found in {video_path}")
    return data["streams"][0]


def _run_ffprobe_format(video_path: str | Path) -> dict:
    """Run ffprobe to get format-level metadata (like duration)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found. Please install ffmpeg (which includes ffprobe).")
        
    if result.returncode != 0:
        return {}
    data = json.loads(result.stdout)
    return data.get("format", {})


def get_video_duration(video_path: str | Path) -> float:
    """
    Get video duration in seconds using ffprobe.
    """
    stream_info = _run_ffprobe(video_path)
    if "duration" in stream_info:
        return float(stream_info["duration"])
    
    fmt_info = _run_ffprobe_format(video_path)
    if "duration" in fmt_info:
        return float(fmt_info["duration"])
        
    raise RuntimeError(f"Could not determine duration for {video_path}")


def get_video_info(video_path: str | Path) -> dict:
    """
    Get basic video metadata using ffprobe.

    Returns:
        Dict with keys: fps, total_frames, duration_sec, width, height
    """
    stream_info = _run_ffprobe(video_path)
    
    r_frame_rate = stream_info.get("r_frame_rate", "30/1")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(r_frame_rate)
        
    total_frames = int(stream_info.get("nb_frames", 0))
    
    if "duration" in stream_info:
        duration_sec = float(stream_info["duration"])
    else:
        fmt_info = _run_ffprobe_format(video_path)
        duration_sec = float(fmt_info.get("duration", 0.0))
        
    if total_frames == 0 and duration_sec > 0 and fps > 0:
        total_frames = int(duration_sec * fps)
        
    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "width": int(stream_info.get("width", 0)),
        "height": int(stream_info.get("height", 0)),
    }


def create_chunks(
    video_id: str,
    duration_sec: float,
    chunk_duration_sec: float = 60.0,
) -> list[ChunkMeta]:
    """
    Create non-overlapping chunk metadata for a video.
    No physical splitting — just metadata boundaries.

    Args:
        video_id:           UUID of the uploaded video
        duration_sec:       Total video duration in seconds
        chunk_duration_sec: Target chunk length (default 60s)

    Returns:
        List of ChunkMeta objects with start/end times
    """
    if duration_sec <= 0:
        raise ValueError(f"Invalid duration: {duration_sec}")

    chunks: list[ChunkMeta] = []
    start = 0.0

    while start < duration_sec:
        end = min(start + chunk_duration_sec, duration_sec)
        chunk = ChunkMeta(
            chunk_id=str(uuid.uuid4()),
            video_id=video_id,
            start_time=round(start, 2),
            end_time=round(end, 2),
        )
        chunks.append(chunk)
        start = end

    logger.info(
        "Created %d chunks for video %s (%.1fs each, %.1fs total)",
        len(chunks), video_id, chunk_duration_sec, duration_sec,
    )
    return chunks


def sample_chunk_frames(
    video_path: str | Path,
    chunk: ChunkMeta,
    num_frames: int = 4,
    output_dir: str | Path = ".",
) -> list[str]:
    """
    Sample frames from a specific chunk using FFmpeg.
    Saves frames as JPEG files and returns their paths.

    Args:
        video_path: Path to the source video file
        chunk:      ChunkMeta with start_time and end_time
        num_frames: Number of frames to sample from this chunk
        output_dir: Directory to save frame JPEGs

    Returns:
        List of absolute paths to saved frame files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = chunk.end_time - chunk.start_time
    if duration <= 0:
        return []

    timestamps_sec = np.linspace(
        chunk.start_time,
        chunk.end_time,
        num_frames + 2,  # exclude very start and end
    )[1:-1]  # trim boundary frames

    frame_paths: list[str] = []

    for i, ts in enumerate(timestamps_sec):
        fname = f"frame_{chunk.chunk_id[:8]}_{i:02d}.jpg"
        fpath = output_dir / fname
        
        cmd = [
            "ffmpeg",
            "-y",               # overwrite
            "-ss", str(ts),     # fast seek
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",        # high quality jpeg
            str(fpath)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and fpath.exists():
                frame_paths.append(str(fpath.resolve()))
            else:
                logger.warning(
                    "Failed to read frame at %.1fs for chunk %s. err: %s",
                    ts, chunk.chunk_id[:8], result.stderr.decode('utf-8', errors='ignore')[-200:]
                )
        except FileNotFoundError:
            logger.error("ffmpeg not found. Cannot extract frames.")
            break

    logger.debug(
        "Sampled %d frames for chunk %s (%.1f-%.1fs)",
        len(frame_paths), chunk.chunk_id[:8],
        chunk.start_time, chunk.end_time,
    )
    return frame_paths


def sample_all_chunks(
    video_path: str | Path,
    chunks: list[ChunkMeta],
    frames_per_chunk: int = 4,
    base_output_dir: str | Path = ".",
) -> list[ChunkMeta]:
    """
    Sample frames for all chunks and update their frame_paths in place.

    Args:
        video_path:       Path to the source video
        chunks:           List of ChunkMeta objects
        frames_per_chunk: Frames to sample per chunk
        base_output_dir:  Base directory; frames saved in {base}/{chunk_id}/

    Returns:
        Same list of ChunkMeta objects with frame_paths populated
    """
    base = Path(base_output_dir)
    for chunk in chunks:
        chunk_dir = base / chunk.chunk_id[:8]
        paths = sample_chunk_frames(
            video_path, chunk, frames_per_chunk, chunk_dir,
        )
        chunk.frame_paths = paths

    total_frames = sum(len(c.frame_paths) for c in chunks)
    logger.info(
        "Sampled %d total frames across %d chunks",
        total_frames, len(chunks),
    )
    return chunks
