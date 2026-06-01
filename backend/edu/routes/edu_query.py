"""
backend/edu/routes/edu_query.py
────────────────────────────────────────────────────────────
POST /api/v2/videos/{id}/ask     — ask a question
GET  /api/v2/results/{id}        — fetch a stored result

This is the core QA endpoint for Edu-VQAGuider.

Pipeline per request:
    1. Validate video is ready (status = 'ready')
    2. Classify question with Edu-VQAGuider planner
    3. Retrieve relevant chunks (route-aware)
    4. Generate answer with local LLM
    5. Persist result to DB
    6. Return grounded answer with evidence
"""
from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException, Request, status

from backend.edu import db_edu
from backend.edu.pipeline import answer_question
from backend.edu.routes.edu_video import get_video_store
from backend.edu.schemas import (
    ChunkInfo,
    EduAskRequest,
    EduAskResponse,
    EduResultResponse,
    EduRoute,
    PlannerSource,
    RouteInfo,
    BaselineAskRequest,
    BaselineAskResponse,
    DirectGenerateRequest,
    DirectGenerateResponse,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Edu-VQAGuider Query"])


# ── POST /api/v2/videos/{id}/ask ──────────────────────────────

@router.post(
    "/videos/{video_id}/ask",
    response_model=EduAskResponse,
    summary="Ask a question about an educational video",
    response_description="Grounded answer with route, evidence, and timestamps",
)
async def ask_edu_question(
    video_id: str,
    body: EduAskRequest,
    request: Request,
) -> EduAskResponse:
    """
    Run the full Edu-VQAGuider pipeline:
        question → planner → retrieval → generation → grounded answer

    Requires:
        - Video is uploaded and status = 'ready'
        - Transcript has been provided (auto or manual)
        - Chunk index is built

    Returns:
        EduAskResponse with direct_answer, detailed_answer, route,
        evidence chunks with timestamps and transcript snippets.
    """
    question = body.question
    logger.info("ask_edu | video=%s | question='%s'", video_id, question[:60])

    # ── Validate video is ready ──────────────────────────────
    store = get_video_store().get(video_id)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found. Upload it first.",
        )

    chunk_index = store.get("chunk_index")
    if chunk_index is None or not chunk_index.is_built:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Video is not ready for questions. "
                "Please provide a transcript first via POST /api/v2/videos/{id}/transcript"
            ),
        )

    chunks = store["chunks"]

    # ── Get models from app.state (if loaded) ────────────────
    edu_state = getattr(request.app.state, "edu_models", None)

    planner = None
    question_embedding = None
    clip_model = None
    clip_preprocess = None
    generate_fn = None
    device = "cpu"

    if edu_state is not None:
        planner = edu_state.get("planner")
        clip_model = edu_state.get("clip_model")
        clip_preprocess = edu_state.get("clip_preprocess")
        generate_fn = edu_state.get("generate_fn")
        device = edu_state.get("device", "cpu")

        # Encode question with DistilBERT for planner
        distilbert_encode = edu_state.get("distilbert_encode")
        if distilbert_encode is not None and planner is not None:
            question_embedding = distilbert_encode(question)

    # ── Run QA pipeline ──────────────────────────────────────
    try:
        result = answer_question(
            question=question,
            chunk_index=chunk_index,
            chunks=chunks,
            planner=planner,
            question_embedding=question_embedding,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            generate_fn=generate_fn,
            device=device,
        )
    except Exception as exc:
        logger.exception("QA pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {exc}",
        ) from exc

    # ── Persist to DB ────────────────────────────────────────
    try:
        db_edu.insert_edu_result(
            result_id=result.result_id,
            video_id=video_id,
            question=question,
            direct_answer=result.direct_answer,
            detailed_answer=result.detailed_answer,
            route_json=result.to_route_json(),
            evidence_json=result.to_evidence_json(),
            planner_source=result.route.source,
        )
    except Exception as exc:
        logger.error("DB persist failed (non-fatal): %s", exc)

    # ── Build response ───────────────────────────────────────
    evidence_chunks = []
    for e in result.evidence:
        evidence_chunks.append(ChunkInfo(
            chunk_id=e.chunk.chunk_id,
            start_time=e.chunk.start_time,
            end_time=e.chunk.end_time,
            transcript_text=e.chunk.transcript_text[:500],  # Truncate for response
            selected_frame_path=e.selected_frame,
            relevance_score=round(e.combined_score, 4),
        ))

    route_info = RouteInfo(
        route=EduRoute(result.route.route),
        confidence=round(result.route.confidence, 4),
        planner_source=PlannerSource(result.route.source),
        all_scores=result.route.all_scores,
    )

    logger.info(
        "ask_edu complete | result_id=%s | route=%s | chunks=%d",
        result.result_id, result.route.route, len(evidence_chunks),
    )

    return EduAskResponse(
        result_id=result.result_id,
        video_id=video_id,
        question=question,
        direct_answer=result.direct_answer,
        detailed_answer=result.detailed_answer,
        route=route_info,
        evidence_chunks=evidence_chunks,
    )


# ── GET /api/v2/results/{id} ─────────────────────────────────

@router.get(
    "/results/{result_id}",
    response_model=EduResultResponse,
    summary="Fetch a stored Edu-VQAGuider result",
)
async def get_edu_result(result_id: str) -> EduResultResponse:
    """
    Retrieve a previously computed result by its result_id.
    """
    if not result_id.strip():
        raise HTTPException(status_code=400, detail="result_id must not be blank.")

    record = db_edu.get_edu_result(result_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result found for result_id='{result_id}'.",
        )

    route_data = record.get("route", {})
    evidence_data = record.get("evidence", [])

    evidence_chunks = [
        ChunkInfo(
            chunk_id=e.get("chunk_id", ""),
            start_time=e.get("start_time", 0),
            end_time=e.get("end_time", 0),
            transcript_text=e.get("transcript_snippet", ""),
            selected_frame_path=e.get("selected_frame"),
            relevance_score=e.get("combined_score", 0),
        )
        for e in evidence_data
    ]

    return EduResultResponse(
        result_id=record["id"],
        video_id=record["video_id"],
        question=record["question"],
        direct_answer=record["direct_answer"],
        detailed_answer=record["answer"],
        route=RouteInfo(
            route=EduRoute(route_data.get("route", "concept")),
            confidence=route_data.get("confidence", 0),
            planner_source=PlannerSource(route_data.get("source", "fallback")),
            all_scores=route_data.get("all_scores", {}),
        ),
        evidence_chunks=evidence_chunks,
        created_at=record["created_at"],
    )


# ── POST /api/v2/baseline_ask ────────────────────────────────

@router.post(
    "/baseline_ask",
    response_model=BaselineAskResponse,
    summary="Ask a question without video context (baseline)",
)
async def baseline_ask(
    body: BaselineAskRequest,
    request: Request,
) -> BaselineAskResponse:
    """
    Directly query Qwen without any video transcript or retrieval context.
    Acts as the baseline (zero-shot) for comparison.
    """
    question = body.question
    logger.info("baseline_ask | question='%s'", question[:60])
    
    edu_state = getattr(request.app.state, "edu_models", None)
    generate_fn = edu_state.get("generate_fn") if edu_state else None
    
    if generate_fn is None:
        logger.warning("No Qwen generate_fn available for baseline_ask.")
        ans = (
            f"[Baseline Placeholder] Qwen is not loaded (likely CPU-only or skipped). "
            f"Unable to generate direct answer to: '{question}'"
        )
    else:
        try:
            # Construct a clean instruction prompt for direct general-knowledge answering
            prompt = (
                f"You are an educational AI assistant. Answer the following question "
                f"based on your general knowledge. Be precise, accurate, and direct.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            ans = generate_fn(prompt)
        except Exception as exc:
            logger.exception("Baseline generation failed: %s", exc)
            ans = f"Error during baseline generation: {exc}"
            
    return BaselineAskResponse(answer=ans)


# ── POST /api/v2/direct_generate ─────────────────────────────

@router.post(
    "/direct_generate",
    response_model=DirectGenerateResponse,
    summary="Run direct text generation on a raw prompt",
)
async def direct_generate(
    body: DirectGenerateRequest,
    request: Request,
) -> DirectGenerateResponse:
    """
    Run direct prompt generation on Qwen without any framing or schemas.
    """
    prompt = body.prompt
    logger.info("direct_generate | prompt='%s'", prompt[:60])
    
    edu_state = getattr(request.app.state, "edu_models", None)
    generate_fn = edu_state.get("generate_fn") if edu_state else None
    
    if generate_fn is None:
        logger.warning("No Qwen generate_fn available for direct_generate.")
        ans = "[Generation Placeholder] Qwen is not loaded."
    else:
        try:
            ans = generate_fn(prompt)
        except Exception as exc:
            logger.exception("Direct generation failed: %s", exc)
            ans = f"Error during direct generation: {exc}"
            
    return DirectGenerateResponse(answer=ans)

