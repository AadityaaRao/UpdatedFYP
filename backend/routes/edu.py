from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from backend.database import db
from backend.models.schemas import (
    EduAskRequest,
    EduAskResponse,
    EduEvidenceSchema,
    EduResultResponse,
    EduRouteSchema,
    EduVideoCreateResponse,
    EduVideoStatusResponse,
)
from backend.routes.video import ALLOWED_EXTENSIONS
from backend.services.edu_answer_service import build_grounded_answer
from backend.services.edu_planner import planner
from backend.services.edu_retrieval_service import retrieve_chunks
from backend.services.edu_video_service import process_edu_video
from config import EDU_CHUNK_SECONDS, UPLOADS_DIR

router = APIRouter(tags=["Edu-VQAGuider"])


@router.post("/videos", response_model=EduVideoCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_edu_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcript_text: str | None = Form(default=None),
    use_auto_transcript: bool = Form(default=True),
    chunk_seconds: int = Form(default=EDU_CHUNK_SECONDS),
) -> EduVideoCreateResponse:
    if file.filename is None or not file.filename.strip():
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    video_id = str(uuid.uuid4())
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"source{ext}"
    try:
        with dest.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    finally:
        await file.close()

    db.insert_video(
        video_id=video_id,
        path=str(dest.resolve()),
        original_filename=file.filename,
        duration_sec=None,
        status="processing",
    )
    background_tasks.add_task(
        process_edu_video,
        video_id=video_id,
        video_path=dest,
        manual_transcript=transcript_text,
        use_auto_transcript=use_auto_transcript,
        chunk_seconds=chunk_seconds,
    )
    return EduVideoCreateResponse(
        video_id=video_id,
        status="processing",
        message="Video accepted. Processing has started.",
    )


@router.get("/videos/{video_id}/status", response_model=EduVideoStatusResponse)
async def get_edu_video_status(video_id: str) -> EduVideoStatusResponse:
    video = db.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"No video found for video_id='{video_id}'.")
    chunks = db.get_video_chunks(video_id)
    status_value = video.get("status") or "unknown"
    chunks_total = len(chunks)
    chunks_processed = chunks_total if status_value == "ready" else 0
    progress = 1.0 if status_value == "ready" else (0.0 if status_value == "processing" else 0.0)
    return EduVideoStatusResponse(
        video_id=video_id,
        status=status_value,
        progress=progress,
        chunks_total=chunks_total,
        chunks_processed=chunks_processed,
        duration_sec=video.get("duration_sec"),
        error=video.get("processing_error"),
    )


@router.post("/videos/{video_id}/ask", response_model=EduAskResponse)
async def ask_edu_video(video_id: str, body: EduAskRequest) -> EduAskResponse:
    video = db.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"No video found for video_id='{video_id}'.")
    if video.get("status") != "ready":
        raise HTTPException(status_code=409, detail=f"Video is not ready. Current status: {video.get('status')}")

    route = planner.route(body.question)
    evidence = retrieve_chunks(video_id, body.question, route, top_k=body.top_k)
    if not evidence:
        raise HTTPException(status_code=422, detail="No processed chunks are available for this video.")

    direct_answer, answer = build_grounded_answer(body.question, route, evidence)
    result_id = str(uuid.uuid4())
    db.insert_edu_result(
        result_id=result_id,
        video_id=video_id,
        question=body.question,
        direct_answer=direct_answer,
        answer=answer,
        route=route.to_dict(),
        evidence=evidence,
        planner_source=route.planner_source,
    )
    return _build_response(result_id, video_id, body.question, direct_answer, answer, route.to_dict(), evidence)


@router.get("/results/{result_id}", response_model=EduResultResponse)
async def get_edu_result(result_id: str) -> EduResultResponse:
    record = db.get_edu_result(result_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No result found for result_id='{result_id}'.")
    response = _build_response(
        record["id"],
        record["video_id"],
        record["question"],
        record["direct_answer"],
        record["answer"],
        record["route"],
        record["evidence"],
    )
    return EduResultResponse(**response.model_dump(), created_at=record["created_at"])


def _build_response(
    result_id: str,
    video_id: str,
    question: str,
    direct_answer: str,
    answer: str,
    route: dict,
    evidence: list[dict],
) -> EduAskResponse:
    return EduAskResponse(
        result_id=result_id,
        video_id=video_id,
        question=question,
        direct_answer=direct_answer,
        answer=answer,
        route=EduRouteSchema(**route),
        evidence=[EduEvidenceSchema(**item) for item in evidence],
    )
