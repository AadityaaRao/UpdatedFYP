"""
backend/utils/video_utils.py
────────────────────────────────────────────────────────────
Low-level video helpers.
Only responsibility: read frames from a video file.
No ML code here.
"""
from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from backend.utils.logger import get_logger
logger = get_logger(__name__)
def sample_frames(video_path: str | Path, num_frames: int = 16) -> list[np.ndarray]:
    """
    Sample `num_frames` evenly spaced RGB frames from a video file.
    Args:
        video_path: Path to the video file (.mp4, .avi, .webm, …)
        num_frames: Number of frames to extract
    Returns:
        List of (H, W, 3) uint8 RGB arrays
    Raises:
        RuntimeError: if the file cannot be opened or has no readable frames
    """
    import os
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    # cv2.setLogLevel(0) # Not available in all cv2 versions
    
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    
    # Clamp num_frames to what is actually available
    n = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n).astype(int).tolist()
    frames: list[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    
    if not frames:
        raise RuntimeError(f"Frame extraction produced no frames: {video_path}")
    
    logger.debug("Sampled %d / %d frames from %s", len(frames), total_frames, video_path)
    return frames

def get_video_metadata(video_path: str | Path) -> dict:
    """
    Return basic metadata for a video file without decoding frames.
    Returns:
        {
            "fps": float,
            "total_frames": int,
            "duration_sec": float,
            "width": int,
            "height": int,
        }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if fps else 0.0,
        "width": width,
        "height": height,
    }