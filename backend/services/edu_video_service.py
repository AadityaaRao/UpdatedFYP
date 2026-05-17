from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from backend.database import db
from backend.services.edu_transcript_service import (
    align_transcript_to_chunks,
    manual_transcript_to_segments,
    transcribe_video,
)
from backend.utils.logger import get_logger
from config import EDU_CHUNK_SECONDS, EDU_FRAMES_PER_CHUNK, UPLOADS_DIR

logger = get_logger(__name__)


def get_video_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = round(total_frames / fps, 2) if fps else 0.0
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration,
        "width": width,
        "height": height,
    }


def build_chunk_ranges(duration_sec: float, chunk_seconds: int = EDU_CHUNK_SECONDS) -> list[dict]:
    if duration_sec <= 0:
        return []
    chunks = []
    start = 0.0
    while start < duration_sec:
        end = min(start + chunk_seconds, duration_sec)
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "start_time": round(start, 2),
                "end_time": round(end, 2),
            }
        )
        start = end
    return chunks


def sample_chunk_frames(
    video_path: Path,
    video_id: str,
    chunk: dict,
    frames_per_chunk: int = EDU_FRAMES_PER_CHUNK,
) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_dir = UPLOADS_DIR / video_id / "frames" / chunk["chunk_id"]
    frame_dir.mkdir(parents=True, exist_ok=True)
    start = float(chunk["start_time"])
    end = float(chunk["end_time"])
    if frames_per_chunk <= 1:
        times = [(start + end) / 2]
    else:
        step = (end - start) / (frames_per_chunk + 1)
        times = [start + step * (idx + 1) for idx in range(frames_per_chunk)]
    paths = []
    for idx, timestamp in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = frame_dir / f"frame_{idx + 1}.jpg"
        cv2.imwrite(str(out_path), frame)
        paths.append(str(out_path))
    cap.release()
    return paths


def process_edu_video(
    video_id: str,
    video_path: Path,
    manual_transcript: str | None = None,
    use_auto_transcript: bool = True,
    chunk_seconds: int = EDU_CHUNK_SECONDS,
) -> None:
    try:
        db.update_video_processing(video_id, "processing", processing_error=None)
        metadata = get_video_metadata(video_path)
        duration = float(metadata["duration_sec"])
        chunks = build_chunk_ranges(duration, chunk_seconds=chunk_seconds)
        db.delete_video_chunks(video_id)

        transcript_segments = []
        if manual_transcript and manual_transcript.strip():
            transcript_segments = manual_transcript_to_segments(manual_transcript, duration)
        elif use_auto_transcript:
            transcript_segments = transcribe_video(video_path)

        aligned_text = align_transcript_to_chunks(transcript_segments, chunks)
        for chunk in chunks:
            frame_paths = sample_chunk_frames(video_path, video_id, chunk)
            db.insert_video_chunk(
                chunk_id=chunk["chunk_id"],
                video_id=video_id,
                start_time=float(chunk["start_time"]),
                end_time=float(chunk["end_time"]),
                transcript_text=aligned_text.get(chunk["chunk_id"], ""),
                visual_summary=None,
                frame_paths=frame_paths,
            )
        db.update_video_processing(video_id, "ready", duration_sec=duration, processing_error=None)
        logger.info("Edu video processed | video_id=%s chunks=%d", video_id, len(chunks))
    except Exception as exc:
        logger.exception("Edu video processing failed | video_id=%s | %s", video_id, exc)
        db.update_video_processing(video_id, "failed", processing_error=str(exc))
