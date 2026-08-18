"""
backend/edu/frame_captioning.py
────────────────────────────────────────────────────────────
Visual grounding via VLM frame captioning.

During video processing, this module uses Qwen2.5-VL to generate
text descriptions of sampled lecture frames. These descriptions
are stored in ChunkMeta.visual_summary and make visual content
searchable alongside transcript text.

Public API:
    caption_chunk_frames()    → caption frames for a single chunk
    caption_all_chunks()      → caption frames for all chunks
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Captioning prompt ─────────────────────────────────────────
FRAME_CAPTION_PROMPT = (
    "You are analyzing a frame from an educational lecture video. "
    "Describe what you see in detail, focusing on:\n"
    "- Any text, titles, or headings visible on slides or the board\n"
    "- Mathematical equations, formulas, or expressions\n"
    "- Diagrams, charts, graphs, or visual aids\n"
    "- Code snippets or pseudocode\n"
    "- Key structural elements (bullet points, numbered lists, tables)\n\n"
    "Be concise but thorough. Only describe what is actually visible. "
    "Do NOT speculate about what is not shown. "
    "Respond in 2-4 sentences."
)


def caption_chunk_frames(
    chunk_frame_paths: list[str],
    generate_fn: Callable,
    max_frames: int = 1,
) -> str:
    """
    Generate a visual description of a chunk's keyframes.

    Picks the first available frame (CLIP-selected best frame if
    available, otherwise the first frame) and asks the VLM to
    describe it.

    Args:
        chunk_frame_paths: List of frame file paths for this chunk
        generate_fn:       VLM generate function (prompt, image_paths) -> str
        max_frames:        Max frames to caption per chunk (default 1
                           for processing speed; increase for richer
                           descriptions at the cost of latency)

    Returns:
        Visual description string, or empty string if captioning fails
    """
    if not chunk_frame_paths:
        return ""

    # Filter to existing frames only
    valid_frames = [p for p in chunk_frame_paths if os.path.exists(p)]
    if not valid_frames:
        logger.warning("No valid frame files found for captioning")
        return ""

    # Use only the first N frames to keep processing fast
    frames_to_caption = valid_frames[:max_frames]

    try:
        caption = generate_fn(
            FRAME_CAPTION_PROMPT,
            image_paths=frames_to_caption,
        )
        logger.debug(
            "Frame caption generated: %d chars from %d frames",
            len(caption), len(frames_to_caption),
        )
        return caption.strip()
    except Exception as e:
        logger.warning("Frame captioning failed: %s", e)
        return ""


def caption_all_chunks(
    chunks,
    generate_fn: Callable,
    max_frames_per_chunk: int = 1,
) -> list:
    """
    Generate visual summaries for all chunks in a video.

    Updates each chunk's visual_summary field in place.

    Args:
        chunks:               List of ChunkMeta objects with frame_paths
        generate_fn:          VLM generate function
        max_frames_per_chunk: Max frames to caption per chunk

    Returns:
        Same list of chunks with visual_summary populated
    """
    total = len(chunks)
    captioned = 0

    for i, chunk in enumerate(chunks):
        if not chunk.frame_paths:
            logger.debug("Chunk %d/%d: no frames, skipping", i + 1, total)
            continue

        logger.info(
            "Captioning chunk %d/%d (%.0fs-%.0fs) ...",
            i + 1, total, chunk.start_time, chunk.end_time,
        )

        caption = caption_chunk_frames(
            chunk.frame_paths,
            generate_fn,
            max_frames=max_frames_per_chunk,
        )

        if caption:
            chunk.visual_summary = caption
            captioned += 1

    logger.info(
        "Frame captioning complete: %d / %d chunks captioned",
        captioned, total,
    )
    return chunks
