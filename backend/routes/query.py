"""
backend/routes/query.py
────────────────────────────────────────────────────────────
POST /api/v1/ask_question
GET  /api/v1/result/{result_id}
Responsibilities (this file only):
  ✓ Validate request inputs (via Pydantic schemas)
  ✓ Resolve video_id → file path
  ✓ Call vqa_service step-by-step (not monolithic)
  ✓ Intercept with cache (get/set) at each step
  ✓ Persist results to SQLite via db.py
  ✓ Return typed JSON responses
  ✓ Handle model + I/O errors gracefully
NOT here:
  ✗ Any model forward pass     → vqa_service.py
  ✗ Pickle read/write          → cache_service.py
  ✗ SQLite queries             → db.py
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request, status
from backend.database import db
from backend.models.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    ResultResponse,
    TaskRoutingSchema,
)
from backend.routes.video import resolve_video_path
from backend.services.cache_service import cache
from backend.services.vqa_service import (
    assemble_response,
    generate_answer,
    get_question_embedding,
    load_video_features,
    run_vqa_model,
)
from backend.utils.logger import get_logger
logger = get_logger(__name__)
router = APIRouter(tags=["Query"])
# ── POST /ask_question ────────────────────────────────────────
@router.post(
    "/ask_question",
    response_model=AskQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about an uploaded video",
    response_description="Generated answer with task routing probabilities",
)
async def ask_question(
    request: Request,
    body: AskQuestionRequest,
) -> AskQuestionResponse:
    """
    Run the full VQA inference pipeline for a given video and question.
    **Request flow:**
    1. Validate `video_id` exists on disk.
    2. Check full-answer cache → early return if HIT.
    3. Encode video (with per-video feature cache).
    4. Encode question (with per-question embedding cache).
    5. Run VQAGuiderCore → fusion vector + task routing.
    6. Generate answer with Phi-2.
    7. Persist query + result to SQLite.
    8. Populate answer cache.
    9. Return response.
    """
    video_id: str = body.video_id
    question: str = body.question
    logger.info("ask_question | video_id=%s | question='%s'", video_id, question[:60])
    # ── Guard: model must be loaded ───────────────────────────
    model = getattr(request.app.state, "model", None)
    if model is None or not getattr(request.app.state, "model_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Server may still be initializing.",
        )
    # ── Step 0: Resolve video path ────────────────────────────
    video_path = resolve_video_path(video_id)
    logger.debug("Resolved video path: %s", video_path)
    # ════════════════════════════════════════════════════════
    # EARLY RETURN: Full answer cache check
    # If we've answered this exact (video, question) pair before,
    # return immediately — no encoding, no model calls.
    # ════════════════════════════════════════════════════════
    cached_answer = cache.get_answer(video_id, question)
    if cached_answer is not None:
        logger.info("Answer cache HIT | video_id=%s question='%s...'", video_id, question[:40])
        routing = cached_answer["task_routing"]
        return AskQuestionResponse(
            result_id=cached_answer["result_id"],
            video_id=video_id,
            question=question,
            answer=cached_answer["answer"],
            task_routing=TaskRoutingSchema(
                action=routing["action"],
                tracking=routing["tracking"],
                scene=routing["scene"],
            ),
            from_cache=True,
        )
    # ════════════════════════════════════════════════════════
    # Step 1 — Video feature encoding (with cache)
    # ════════════════════════════════════════════════════════
    video_feat = cache.get_video_feature(video_id)
    if video_feat is None:
        logger.debug("Video feature cache MISS — encoding video...")
        try:
            video_feat = load_video_features(model, video_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        except RuntimeError as exc:
            logger.error("Video encoding failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not encode video: {exc}",
            ) from exc
        cache.set_video_feature(video_id, video_feat)
    else:
        logger.debug("Video feature cache HIT | video_id=%s", video_id)
    # Move cached CPU tensor to inference device
    video_feat = video_feat.to(model.device)
    # ════════════════════════════════════════════════════════
    # Step 2 — Question embedding (with cache)
    # ════════════════════════════════════════════════════════
    question_feat = cache.get_embedding(question)
    if question_feat is None:
        logger.debug("Embedding cache MISS — encoding question...")
        try:
            question_feat = get_question_embedding(model, question)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.error("Question encoding failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode question.",
            ) from exc
        cache.set_embedding(question, question_feat)
    else:
        logger.debug("Embedding cache HIT | question='%s...'", question[:40])
    # Move cached CPU tensor to inference device
    question_feat = question_feat.to(model.device)
    # ════════════════════════════════════════════════════════
    # Step 3 — VQA core model (routing + fusion)
    # ════════════════════════════════════════════════════════
    try:
        fusion_vec, task_probs = run_vqa_model(model, video_feat, question_feat)
    except Exception as exc:
        logger.error("VQA model forward pass failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="VQA model inference failed. Check server logs.",
        ) from exc
    # ════════════════════════════════════════════════════════
    # Step 4 — Generative answer (Phi-2)
    # ════════════════════════════════════════════════════════
    try:
        answer = generate_answer(model, fusion_vec, question)
    except Exception as exc:
        logger.error("Phi-2 generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Answer generation failed. Check server logs.",
        ) from exc
    # ════════════════════════════════════════════════════════
    # Step 5 — Assemble response object
    # ════════════════════════════════════════════════════════
    result_id = str(uuid.uuid4())
    vqa_response = assemble_response(answer, task_probs, result_id=result_id)
    routing = vqa_response.task_routing
    # ════════════════════════════════════════════════════════
    # Step 6 — Persist to SQLite
    # ════════════════════════════════════════════════════════
    try:
        # Ensure video is registered (idempotent INSERT OR IGNORE)
        db.insert_video(
            video_id=video_id,
            path=str(video_path.resolve()),
            original_filename=video_path.name,
        )
        # Insert query record, get its UUID back
        query_id = db.insert_query(question=question)
        # Insert result linked to video + query
        db.insert_result(
            result_id=result_id,
            video_id=video_id,
            query_id=query_id,
            answer=vqa_response.answer,
            routing_dict=routing.to_dict(),
            from_cache=False,
        )
        logger.info("Result persisted to DB | result_id=%s", result_id)
    except Exception as exc:
        # DB failure should not kill the HTTP response.
        # The answer is already computed — log and continue.
        logger.error("DB persistence failed (non-fatal): %s", exc)
    # ════════════════════════════════════════════════════════
    # Step 7 — Populate answer cache for future requests
    # ════════════════════════════════════════════════════════
    cache.set_answer(
        video_id=video_id,
        question=question,
        data={
            "result_id": result_id,
            "answer": vqa_response.answer,
            "task_routing": routing.to_dict(),
        },
    )
    logger.info(
        "ask_question complete | result_id=%s | answer='%s...'",
        result_id,
        answer[:40],
    )
    return AskQuestionResponse(
        result_id=result_id,
        video_id=video_id,
        question=question,
        answer=vqa_response.answer,
        task_routing=TaskRoutingSchema(
            action=routing.action,
            tracking=routing.tracking,
            scene=routing.scene,
        ),
        from_cache=False,
    )
# ── GET /result/{result_id} ───────────────────────────────────
@router.get(
    "/result/{result_id}",
    response_model=ResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch a stored VQA result by ID",
    response_description="Full result including answer and routing",
)
async def get_result(result_id: str, request: Request) -> ResultResponse:
    """
    Retrieve a previously computed result by its `result_id`.
    Results are persisted to SQLite by `POST /ask_question` and
    survive server restarts.
    """
    if not result_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="result_id must not be blank.",
        )
    logger.debug("get_result | result_id=%s", result_id)
    try:
        record = db.get_result(result_id)
    except Exception as exc:
        logger.error("DB get_result failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching result.",
        ) from exc
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result found for result_id='{result_id}'.",
        )
    routing_dict = record["task_routing"]
    return ResultResponse(
        result_id=record["result_id"],
        video_id=record["video_id"],
        question=record["question"],
        answer=record["answer"],
        task_routing=TaskRoutingSchema(
            action=routing_dict["action"],
            tracking=routing_dict["tracking"],
            scene=routing_dict["scene"],
        ),
        created_at=record["created_at"],
    )