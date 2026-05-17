"""
backend/edu/routes/edu_video.py
────────────────────────────────────────────────────────────
POST /api/v2/videos              — upload + begin processing
POST /api/v2/videos/{id}/transcript — manual transcript fallback
GET  /api/v2/videos/{id}/status  — check processing status
GET  /api/v2/videos/{id}/history — question history

Responsibilities (this file only):
    ✓ Receive video upload
    ✓ Trigger chunk creation + frame sampling
    ✓ Accept manual transcript
    ✓ Report processing status
    ✓ Return question history

NOT here:
    ✗ QA inference        → edu_query.py
    ✗ Model loading       → pipeline.py
    ✗ Heavy processing    → pipeline.py (called from here)
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from backend.database import db
from backend.edu import db_edu
from backend.edu.chunking import create_chunks, get_video_info, sample_all_chunks
from backend.edu.pipeline import align_transcript_to_chunks, align_whisper_segments
from backend.edu.retrieval import build_chunk_index
from backend.edu.schemas import (
    EduUploadResponse,
    HistoryItem,
    HistoryResponse,
    ManualTranscriptRequest,
    TranscriptResponse,
    VideoStatus,
    VideoStatusResponse,
)
from backend.utils.logger import get_logger
from config import MAX_UPLOAD_SIZE_MB, UPLOADS_DIR

logger = get_logger(__name__)

router = APIRouter(tags=["Edu-VQAGuider Video"])

ALLOWED_EXTENSIONS = frozenset({".mp4", ".avi", ".webm", ".mov", ".mkv"})
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# In-memory store for processed video data (chunk indices, etc.)
# In production this would be Redis or similar; for MVP, a dict is fine.
_video_store: dict[str, dict] = {}


def get_video_store() -> dict:
    """Access the in-memory video store. Used by edu_query.py too."""
    return _video_store


# ── POST /api/v2/videos ──────────────────────────────────────

@router.post(
    "/videos",
    response_model=EduUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an educational video",
    response_description="Video ID and initial processing status",
)
async def upload_edu_video(
    request: Request,
    file: UploadFile,
) -> EduUploadResponse:
    """
    Upload a long educational video and begin processing.

    Processing steps (synchronous for MVP):
        1. Save file to uploads/{video_id}/source{ext}
        2. Extract metadata (duration, fps, resolution)
        3. Create 60-second non-overlapping chunk metadata
        4. Sample 4 frames per chunk

    Transcript must be provided separately via:
        - POST /api/v2/videos/{id}/transcript (manual paste)
        - Future: automatic Whisper transcription
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    video_id = str(uuid.uuid4())
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"source{ext}"

    logger.info("Receiving upload: '%s' → %s", file.filename, video_path)

    # Save file
    try:
        with video_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
            size_bytes = out.tell()
    except Exception as exc:
        logger.exception("Failed to save upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save video file.",
        ) from exc
    finally:
        await file.close()

    if size_bytes == 0:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size_bytes > MAX_UPLOAD_BYTES:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_UPLOAD_SIZE_MB} MB.")

    # Register in DB (reuse v1 table with v2 extensions)
    try:
        db.insert_video(video_id, str(video_path.resolve()), file.filename)
    except Exception as exc:
        logger.error("DB insert failed (non-fatal): %s", exc)

    # Extract metadata + create chunks + sample frames
    try:
        info = get_video_info(video_path)
        duration = info["duration_sec"]

        db_edu.ensure_video_v2_columns()
        db_edu.update_video_status(video_id, "indexing", duration_sec=duration)

        chunks = create_chunks(video_id, duration, chunk_duration_sec=60.0)

        frames_dir = video_dir / "frames"
        chunks = sample_all_chunks(video_path, chunks, frames_per_chunk=4, base_output_dir=frames_dir)

        # Persist chunks to DB
        chunk_rows = [c.to_db_row() for c in chunks]
        db_edu.insert_chunks(chunk_rows)

        # Store in memory for query-time access
        _video_store[video_id] = {
            "video_path": str(video_path),
            "duration_sec": duration,
            "chunks": chunks,
            "chunk_index": None,  # Built after transcript
            "info": info,
        }

        # Status depends on whether we have transcript
        db_edu.update_video_status(video_id, "pending")  # Waiting for transcript

        logger.info(
            "Video %s processed: %.1fs, %d chunks, frames saved",
            video_id, duration, len(chunks),
        )

    except Exception as exc:
        logger.exception("Video processing failed: %s", exc)
        db_edu.update_video_status(video_id, "error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video processing failed: {exc}",
        ) from exc

    return EduUploadResponse(
        video_id=video_id,
        original_filename=file.filename,
        duration_sec=duration,
        num_chunks=len(chunks),
        status=VideoStatus.pending,
    )


# ── POST /api/v2/videos/{id}/transcript ───────────────────────

@router.post(
    "/videos/{video_id}/transcript",
    response_model=TranscriptResponse,
    summary="Provide manual transcript for a video",
)
async def upload_transcript(
    video_id: str,
    body: ManualTranscriptRequest,
) -> TranscriptResponse:
    """
    Accept a manual transcript and align it to video chunks.

    After this call, the video status becomes 'ready' and
    questions can be asked.
    """
    # Verify video exists
    store = _video_store.get(video_id)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found. Upload it first.",
        )

    chunks = store["chunks"]
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Video has no chunks. Re-upload the video.",
        )

    # Align transcript to chunks
    db_edu.update_video_status(video_id, "transcribing")

    try:
        chunks = align_transcript_to_chunks(chunks, body.transcript_text)
        store["chunks"] = chunks

        # Update DB
        for chunk in chunks:
            db_edu.update_chunk_transcript(chunk.chunk_id, chunk.transcript_text)

        # Build search index
        db_edu.update_video_status(video_id, "indexing")
        chunk_index = build_chunk_index(chunks)
        store["chunk_index"] = chunk_index

        db_edu.update_video_status(video_id, "ready")

    except Exception as exc:
        logger.exception("Transcript processing failed: %s", exc)
        db_edu.update_video_status(video_id, "error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcript processing failed: {exc}",
        ) from exc

    chunks_with_text = sum(1 for c in chunks if c.transcript_text.strip())

    return TranscriptResponse(
        video_id=video_id,
        num_chunks_with_text=chunks_with_text,
        status=VideoStatus.ready,
    )


# ── POST /api/v2/videos/{id}/transcribe ───────────────────────

@router.post(
    "/videos/{video_id}/transcribe",
    response_model=TranscriptResponse,
    summary="Auto-transcribe video with Whisper",
)
async def auto_transcribe(
    video_id: str,
    request: Request,
) -> TranscriptResponse:
    """
    Automatically transcribe a video using faster-whisper.

    Steps:
        1. Extract audio from video via ffmpeg
        2. Run faster-whisper (model loaded, used, then unloaded)
        3. Align timestamped segments to chunks
        4. Build embedding index
        5. Set status to 'ready'

    VRAM note: Whisper is loaded and unloaded within this call.
    """
    store = _video_store.get(video_id)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found. Upload it first.",
        )

    chunks = store["chunks"]
    video_path = store["video_path"]

    db_edu.update_video_status(video_id, "transcribing")

    try:
        from backend.edu.transcription import transcribe_video
        from config import EDU_WHISPER_MODEL

        # This loads Whisper, transcribes, then unloads + frees VRAM
        result = transcribe_video(
            video_path=video_path,
            model_size=EDU_WHISPER_MODEL,
        )

        logger.info(
            "Transcription complete: %d segments, lang=%s",
            result.num_segments, result.language,
        )

        # Align Whisper segments to chunks
        chunks = align_whisper_segments(chunks, result.segments)
        store["chunks"] = chunks

        # Update DB
        for chunk in chunks:
            db_edu.update_chunk_transcript(chunk.chunk_id, chunk.transcript_text)

        # Build search index
        db_edu.update_video_status(video_id, "indexing")
        chunk_index = build_chunk_index(chunks)
        store["chunk_index"] = chunk_index

        db_edu.update_video_status(video_id, "ready")

    except Exception as exc:
        logger.exception("Auto-transcription failed: %s", exc)
        db_edu.update_video_status(video_id, "error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc

    chunks_with_text = sum(1 for c in chunks if c.transcript_text.strip())

    return TranscriptResponse(
        video_id=video_id,
        num_chunks_with_text=chunks_with_text,
        status=VideoStatus.ready,
    )


# ── GET /api/v2/videos/{id}/status ────────────────────────────

@router.get(
    "/videos/{video_id}/status",
    response_model=VideoStatusResponse,
    summary="Check video processing status",
)
async def video_status(video_id: str) -> VideoStatusResponse:
    """
    Check whether a video is ready for questions.
    """
    record = db_edu.get_video_status(video_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found.",
        )

    store = _video_store.get(video_id, {})
    num_chunks = db_edu.count_chunks(video_id)
    has_transcript = any(
        c.transcript_text.strip()
        for c in store.get("chunks", [])
    )
    has_embeddings = store.get("chunk_index") is not None

    return VideoStatusResponse(
        video_id=video_id,
        status=VideoStatus(record.get("status", "pending")),
        duration_sec=record.get("duration_sec", 0.0),
        num_chunks=num_chunks,
        has_transcript=has_transcript,
        has_embeddings=has_embeddings,
        processing_error=record.get("processing_error"),
    )


# ── GET /api/v2/videos/{id}/history ───────────────────────────

@router.get(
    "/videos/{video_id}/history",
    response_model=HistoryResponse,
    summary="Get question history for a video",
)
async def video_history(video_id: str) -> HistoryResponse:
    """
    Fetch past questions and answers for a video.
    """
    records = db_edu.get_edu_results_for_video(video_id)

    items = []
    for r in records:
        route_data = r.get("route", {})
        items.append(HistoryItem(
            result_id=r["id"],
            question=r["question"],
            direct_answer=r["direct_answer"],
            route=route_data.get("route", "concept"),
            created_at=r["created_at"],
        ))

    return HistoryResponse(video_id=video_id, items=items)
