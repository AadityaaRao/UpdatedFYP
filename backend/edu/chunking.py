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
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
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


def get_video_duration(video_path: str | Path) -> float:
    """
    Get video duration in seconds using cv2.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If video cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata: frames={total_frames}, fps={fps}")

    duration = total_frames / fps
    logger.debug("Video duration: %.1f sec (%.0f fps, %d frames)", duration, fps, total_frames)
    return duration


def get_video_info(video_path: str | Path) -> dict:
    """
    Get basic video metadata without decoding frames.

    Returns:
        Dict with keys: fps, total_frames, duration_sec, width, height
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()

    info["duration_sec"] = round(
        info["total_frames"] / info["fps"], 2
    ) if info["fps"] > 0 else 0.0

    return info


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
    Sample frames from a specific chunk by seeking to timestamps.
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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Calculate timestamps to sample at (uniform within chunk)
    duration = chunk.end_time - chunk.start_time
    if duration <= 0:
        cap.release()
        return []

    timestamps_sec = np.linspace(
        chunk.start_time,
        chunk.end_time,
        num_frames + 2,  # exclude very start and end
    )[1:-1]  # trim boundary frames

    frame_paths: list[str] = []

    for i, ts in enumerate(timestamps_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None:
            fname = f"frame_{chunk.chunk_id[:8]}_{i:02d}.jpg"
            fpath = output_dir / fname
            cv2.imwrite(str(fpath), frame)
            frame_paths.append(str(fpath.resolve()))
        else:
            logger.warning(
                "Failed to read frame at %.1fs for chunk %s",
                ts, chunk.chunk_id[:8],
            )

    cap.release()
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
