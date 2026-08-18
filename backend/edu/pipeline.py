"""
backend/edu/pipeline.py
────────────────────────────────────────────────────────────
Edu-VQAGuider end-to-end pipeline orchestrator.

This module wires together all v2 components:
    chunking → transcription → embedding → planner → retrieval → generation

It provides two main functions:
    process_video()  — async-compatible video processing (upload → ready)
    answer_question() — synchronous QA pipeline (question → answer)

All heavy model loading is deferred — this module imports cleanly
without any ML dependencies, making routes and tests importable.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from backend.edu.chunking import (
    ChunkMeta,
    create_chunks,
    get_video_info,
    sample_all_chunks,
)
from backend.edu.planner import (
    PlannerResult,
    classify_with_fallback,
    EduPlanner,
)
from backend.edu.prompts import build_prompt, build_direct_answer_prompt
from backend.edu.retrieval import (
    ChunkIndex,
    RetrievedChunk,
    build_chunk_index,
    retrieve_chunks,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ── Answer Confidence Thresholds ─────────────────────────────
LOW_ROUTE_CONFIDENCE = 0.4      # Planner softmax confidence threshold
LOW_EVIDENCE_SCORE = 0.3        # Combined retrieval score threshold
DISCLAIMER_LOW_CONFIDENCE = (
    "⚠️ **Note:** I'm not fully confident about this answer based on the "
    "lecture content. The question may not be well covered in the video, "
    "or the relevant section may not have been clearly transcribed. "
    "Please verify with the original lecture.\n\n"
)
DISCLAIMER_NO_EVIDENCE = (
    "⚠️ **Note:** I could not find strong evidence in the lecture transcript "
    "for this question. The answer below is my best attempt, but it may not "
    "accurately reflect what was discussed in the video.\n\n"
)


# ══════════════════════════════════════════════════════════════
# Video Processing Pipeline
# ══════════════════════════════════════════════════════════════

class VideoProcessingResult:
    """Result of processing a video through the ingestion pipeline."""

    def __init__(
        self,
        video_id: str,
        video_path: str,
        duration_sec: float,
        chunks: list[ChunkMeta],
        chunk_index: Optional[ChunkIndex] = None,
    ):
        self.video_id = video_id
        self.video_path = video_path
        self.duration_sec = duration_sec
        self.chunks = chunks
        self.chunk_index = chunk_index


def process_video(
    video_id: str,
    video_path: str | Path,
    frames_dir: str | Path,
    chunk_duration_sec: float = 60.0,
    frames_per_chunk: int = 4,
    transcript_text: Optional[str] = None,
) -> VideoProcessingResult:
    """
    Process a video through the full ingestion pipeline.

    Steps:
        1. Get video metadata (duration, fps, resolution)
        2. Create chunk metadata (non-overlapping, metadata-only)
        3. Sample frames for each chunk
        4. If transcript provided, align text to chunks
        5. Build embedding index over chunk transcripts

    Args:
        video_id:           UUID of the video
        video_path:         Path to the source video file
        frames_dir:         Directory to save extracted frames
        chunk_duration_sec: Target chunk length in seconds
        frames_per_chunk:   Number of frames to sample per chunk
        transcript_text:    Optional pre-existing transcript text

    Returns:
        VideoProcessingResult with chunks and (optionally) index

    Note:
        Whisper transcription is NOT called here. It should be
        called separately (it's a heavy, async operation).
        Pass the transcript via transcript_text after Whisper runs.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Step 1: Get metadata
    info = get_video_info(video_path)
    duration = info["duration_sec"]
    logger.info(
        "Processing video %s: %.1fs, %dx%d, %.0f fps",
        video_id, duration, info["width"], info["height"], info["fps"],
    )

    # Step 2: Create chunk metadata
    chunks = create_chunks(video_id, duration, chunk_duration_sec)

    # Step 3: Sample frames
    chunks = sample_all_chunks(
        video_path, chunks, frames_per_chunk, frames_dir,
    )

    # Step 4: Align transcript to chunks (if provided)
    if transcript_text:
        chunks = align_transcript_to_chunks(chunks, transcript_text)

    # Step 5: Build index (only if we have transcripts)
    chunk_index = None
    has_text = any(c.transcript_text.strip() for c in chunks)
    if has_text:
        chunk_index = build_chunk_index(chunks)

    result = VideoProcessingResult(
        video_id=video_id,
        video_path=str(video_path),
        duration_sec=duration,
        chunks=chunks,
        chunk_index=chunk_index,
    )

    logger.info(
        "Video processing complete: %d chunks, index=%s",
        len(chunks), "built" if chunk_index else "not built (no transcript)",
    )
    return result


# ══════════════════════════════════════════════════════════════
# Transcript Alignment
# ══════════════════════════════════════════════════════════════

def align_transcript_to_chunks(
    chunks: list[ChunkMeta],
    transcript_text: str,
) -> list[ChunkMeta]:
    """
    Distribute transcript text across chunks proportionally.

    For MVP, this uses a simple proportional split by duration.
    When Whisper word-level timestamps are available, use
    align_whisper_segments() instead for accurate alignment.

    Args:
        chunks:          List of ChunkMeta with start/end times
        transcript_text: Full transcript as a single string

    Returns:
        Same chunks with transcript_text populated
    """
    words = transcript_text.split()
    if not words:
        logger.warning("Empty transcript — chunks will have no text")
        return chunks

    total_duration = sum(c.end_time - c.start_time for c in chunks)
    if total_duration <= 0:
        return chunks

    word_idx = 0
    for chunk in chunks:
        chunk_duration = chunk.end_time - chunk.start_time
        proportion = chunk_duration / total_duration
        num_words = max(1, int(len(words) * proportion))

        chunk_words = words[word_idx : word_idx + num_words]
        chunk.transcript_text = " ".join(chunk_words)
        word_idx += num_words

    # Assign any remaining words to the last chunk
    if word_idx < len(words):
        remaining = " ".join(words[word_idx:])
        chunks[-1].transcript_text += " " + remaining

    assigned = sum(1 for c in chunks if c.transcript_text.strip())
    logger.info(
        "Transcript aligned: %d / %d chunks have text (%d words total)",
        assigned, len(chunks), len(words),
    )
    return chunks


def align_whisper_segments(
    chunks: list[ChunkMeta],
    segments: list[dict],
) -> list[ChunkMeta]:
    """
    Align Whisper segments (with timestamps) to chunks.

    Each Whisper segment has: {"start": float, "end": float, "text": str}
    A segment is assigned to the chunk whose time range it overlaps most.

    Args:
        chunks:   List of ChunkMeta with start/end times
        segments: List of Whisper segment dicts

    Returns:
        Same chunks with transcript_text populated from Whisper output
    """
    # Build a mapping: chunk_idx → list of segment texts
    chunk_texts: dict[int, list[str]] = {i: [] for i in range(len(chunks))}

    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start)
        seg_mid = (seg_start + seg_end) / 2.0
        seg_text = seg.get("text", "").strip()

        if not seg_text:
            continue

        # Find the chunk that contains this segment's midpoint
        best_chunk_idx = 0
        for i, chunk in enumerate(chunks):
            if chunk.start_time <= seg_mid < chunk.end_time:
                best_chunk_idx = i
                break
        else:
            # Segment is past the last chunk — assign to last
            best_chunk_idx = len(chunks) - 1

        chunk_texts[best_chunk_idx].append(seg_text)

    # Assign concatenated texts to chunks
    for i, chunk in enumerate(chunks):
        chunk.transcript_text = " ".join(chunk_texts[i])

    assigned = sum(1 for c in chunks if c.transcript_text.strip())
    logger.info(
        "Whisper segments aligned: %d / %d chunks have text (%d segments total)",
        assigned, len(chunks), len(segments),
    )
    return chunks


# ══════════════════════════════════════════════════════════════
# Question Answering Pipeline
# ══════════════════════════════════════════════════════════════

class AnswerResult:
    """Full result of the QA pipeline."""

    def __init__(
        self,
        result_id: str,
        question: str,
        direct_answer: str,
        detailed_answer: str,
        route: PlannerResult,
        evidence: list[RetrievedChunk],
        confidence_level: str = "high",
    ):
        self.result_id = result_id
        self.question = question
        self.direct_answer = direct_answer
        self.detailed_answer = detailed_answer
        self.route = route
        self.evidence = evidence
        self.confidence_level = confidence_level  # "high", "medium", or "low"

    def to_evidence_json(self) -> str:
        """Serialize evidence chunks for DB storage."""
        evidence_data = []
        for e in self.evidence:
            evidence_data.append({
                "chunk_id": e.chunk.chunk_id,
                "start_time": e.chunk.start_time,
                "end_time": e.chunk.end_time,
                "transcript_snippet": e.chunk.transcript_text[:300],
                "transcript_score": round(e.transcript_score, 4),
                "clip_score": round(e.clip_score, 4) if e.clip_score else None,
                "combined_score": round(e.combined_score, 4),
                "selected_frame": e.selected_frame,
            })
        return json.dumps(evidence_data)

    def to_route_json(self) -> str:
        """Serialize route info for DB storage."""
        return json.dumps({
            "route": self.route.route,
            "confidence": round(self.route.confidence, 4),
            "source": self.route.source,
            "all_scores": self.route.all_scores,
        })


def answer_question(
    question: str,
    chunk_index: ChunkIndex,
    chunks: list[ChunkMeta],
    planner: Optional[EduPlanner] = None,
    question_embedding=None,
    clip_model=None,
    clip_preprocess=None,
    generate_fn=None,
    device: str = "cpu",
) -> AnswerResult:
    """
    Run the full Edu-VQAGuider QA pipeline.

    Steps:
        1. Classify question intent with Edu-VQAGuider planner
        2. Retrieve relevant chunks (route-aware)
        3. Build route-specific prompt from evidence
        4. Generate detailed answer with LLM
        5. Generate direct answer (condensed)

    Args:
        question:           User's question text
        chunk_index:        Built ChunkIndex for the video
        chunks:             List of all ChunkMeta for the video
        planner:            Trained EduPlanner (None → fallback only)
        question_embedding: (768,) DistilBERT CLS embedding for planner
        clip_model:         CLIP model for frame selection (optional)
        clip_preprocess:    CLIP preprocessing (optional)
        generate_fn:        Callable(prompt: str) → str for LLM generation
        device:             Device string for CLIP inference

    Returns:
        AnswerResult with all pipeline outputs
    """
    result_id = str(uuid.uuid4())
    logger.info("QA pipeline start | question='%s'", question[:60])

    # Step 1: Classify question intent
    route_result = classify_with_fallback(
        planner, question_embedding, question,
    )
    logger.info(
        "Route: %s (confidence=%.3f, source=%s)",
        route_result.route, route_result.confidence, route_result.source,
    )

    # Step 2: Route-aware retrieval
    evidence = retrieve_chunks(
        index=chunk_index,
        question=question,
        route=route_result.route,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=device,
    )

    # Step 3: Build prompt from evidence
    evidence_text = "\n\n".join(
        f"[{e.chunk.start_time:.0f}s - {e.chunk.end_time:.0f}s]: {e.chunk.transcript_text}"
        for e in evidence
        if e.chunk.transcript_text.strip()
    )

    prompt = build_prompt(
        route=route_result.route,
        question=question,
        evidence=evidence_text,
    )

    # Step 4: Generate detailed answer
    if generate_fn is not None:
        image_paths = [
            e.selected_frame for e in evidence
            if e.selected_frame and Path(e.selected_frame).exists()
        ]
        detailed_answer = generate_fn(prompt, image_paths=image_paths)
    else:
        # Placeholder when no LLM is loaded
        detailed_answer = (
            f"[LLM not loaded] Route: {route_result.route}. "
            f"Retrieved {len(evidence)} chunks. "
            f"Evidence preview: {evidence_text[:200]}..."
        )
        logger.warning("No generate_fn provided — using placeholder answer")

    # Step 5: Generate direct answer
    if generate_fn is not None:
        direct_prompt = build_direct_answer_prompt(question, detailed_answer)
        direct_answer = generate_fn(direct_prompt)
    else:
        # Extract first sentence as direct answer placeholder
        direct_answer = detailed_answer.split(".")[0] + "." if detailed_answer else ""

    # Step 6: Assess answer confidence
    # Summary route intentionally retrieves broad, lower-scoring chunks,
    # so we skip the evidence threshold for summaries to avoid false alarms.
    top_evidence_score = evidence[0].combined_score if evidence else 0.0
    is_summary = route_result.route == "summary"

    if route_result.confidence < LOW_ROUTE_CONFIDENCE and top_evidence_score < LOW_EVIDENCE_SCORE:
        confidence_level = "low"
        detailed_answer = DISCLAIMER_LOW_CONFIDENCE + detailed_answer
        logger.info(
            "Answer confidence: LOW (route=%.3f, evidence=%.3f)",
            route_result.confidence, top_evidence_score,
        )
    elif not is_summary and (route_result.confidence < LOW_ROUTE_CONFIDENCE or top_evidence_score < LOW_EVIDENCE_SCORE):
        confidence_level = "medium"
        if top_evidence_score < LOW_EVIDENCE_SCORE:
            detailed_answer = DISCLAIMER_NO_EVIDENCE + detailed_answer
        logger.info(
            "Answer confidence: MEDIUM (route=%.3f, evidence=%.3f)",
            route_result.confidence, top_evidence_score,
        )
    else:
        confidence_level = "high"

    result = AnswerResult(
        result_id=result_id,
        question=question,
        direct_answer=direct_answer,
        detailed_answer=detailed_answer,
        route=route_result,
        evidence=evidence,
        confidence_level=confidence_level,
    )

    logger.info(
        "QA pipeline complete | result_id=%s | route=%s | chunks=%d | answer_len=%d | confidence=%s",
        result_id, route_result.route, len(evidence), len(detailed_answer), confidence_level,
    )
    return result
