"""
backend/main.py
────────────────────────────────────────────────────────────
FastAPI application factory.
Key design decisions:
  • create_app() returns a configured FastAPI instance — testable,
    importable, and decoupled from the module-level run command.
  • Model loading happens inside the @asynccontextmanager lifespan,
    which FastAPI calls once at startup and once at shutdown.
  • The loaded ModelBundle is stored on app.state.model so every
    request handler can access it via request.app.state.model.
  • Routers registered: /api/v1/upload_video, /api/v1/ask_question,
    /api/v1/result/{id}, /health
Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# ── Internal imports ──────────────────────────────────────────
# sys.path is expected to include the project root so `config`
# and `backend` are both importable as top-level packages.
from backend.services.model_loader import load_model_bundle
from backend.utils.logger import get_logger
logger = get_logger(__name__)
# ── Lifespan (startup + shutdown) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager executed by FastAPI around the app lifecycle.
    Startup: load all model components once and store on app.state.
    Shutdown: clean up GPU memory.
    """
    # ── STARTUP ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("VQA Guider — startup")
    logger.info("Python %s", sys.version)
    logger.info("PyTorch %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("=" * 60)
    t0 = time.perf_counter()
    from config import (
        DEVICE_PREFERENCE, MODEL_CHECKPOINT,
        LEGACY_VQA_ENABLED, EDU_QWEN_MODEL, EDU_MAX_NEW_TOKENS,
        MODELS_DIR,
    )
    # ── Database initialisation (always runs, model-independent) ──
    try:
        from backend.database.db import init_db
        init_db()
        # Ensure v2 columns exist on the videos table
        from backend.edu.db_edu import ensure_video_v2_columns
        ensure_video_v2_columns()
    except Exception as exc:
        logger.exception("FATAL: database initialisation failed -- %s", exc)
        # Don't prevent server startup -- routes will surface DB errors per-request
    # ── v1 Legacy model (only if explicitly enabled) ──────────
    if LEGACY_VQA_ENABLED:
        try:
            app.state.model = load_model_bundle(
                checkpoint_path=MODEL_CHECKPOINT,
                device_preference=DEVICE_PREFERENCE,
            )
            app.state.model_ready = True
            logger.info("Legacy v1 model loaded in %.1f s", time.perf_counter() - t0)
        except Exception as exc:
            logger.exception("Legacy model loading failed -- %s", exc)
            app.state.model_ready = False
            app.state.model = None
    else:
        logger.info("Legacy v1 model SKIPPED (set LEGACY_VQA_ENABLED=true to load)")
        app.state.model_ready = False
        app.state.model = None
    # ── Edu-VQAGuider v2 models ───────────────────────────────
    try:
        from backend.edu.model_manager import EduModelManager
        edu_manager = EduModelManager(device=DEVICE_PREFERENCE)
        planner_path = MODELS_DIR / "edu_planner.pt"
        edu_manager.load_query_models(
            planner_checkpoint=planner_path,
            qwen_model=EDU_QWEN_MODEL,
            max_new_tokens=EDU_MAX_NEW_TOKENS,
            skip_qwen=not torch.cuda.is_available(),  # skip Qwen on CPU-only machines
            skip_clip=False,
        )
        app.state.edu_models = edu_manager.get_state()
        logger.info("Edu-VQAGuider v2 models loaded in %.1f s", time.perf_counter() - t0)
    except Exception as exc:
        logger.exception("Edu-VQAGuider model loading failed -- %s", exc)
        app.state.edu_models = None
    logger.info("Server ready. v1 + v2 endpoints available.")
    yield  # ← application runs here
    # ── SHUTDOWN ──────────────────────────────────────────────
    logger.info("VQA Guider — shutdown")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared.")
# ── App factory ───────────────────────────────────────────────
def create_app() -> FastAPI:
    """
    Build and return a fully configured FastAPI application.
    Steps:
      1. Create FastAPI instance with metadata and lifespan hook.
      2. Add CORS middleware.
      3. Add global exception handler.
      4. Register /health endpoint.
      5. Register routers (Step 4 — placeholders for now).
    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="Edu-VQAGuider API",
        description=(
            "Edu-VQAGuider: Retrieval-Augmented Multimodal VQA for "
            "Long Educational Video Understanding.\n\n"
            "**v1** — Legacy VQAGuider (short-video, NExT-QA)\n"
            "**v2** — Edu-VQAGuider (long educational video, RAG pipeline)"
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    # ── CORS ──────────────────────────────────────────────────
    # Allow the Streamlit frontend (default port 8501) and any
    # localhost origin during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",   # Streamlit
            "http://127.0.0.1:8501",
            "http://localhost:3000",   # any local dev UI
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # ── Global exception handler ──────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check server logs."},
        )
    # ── Health endpoint ───────────────────────────────────────
    @app.get(
        "/health",
        summary="Health check",
        tags=["System"],
        response_description="Server and model status",
    )
    async def health(request: Request):
        """
        Returns 200 when the server is up.
        Indicates whether models loaded successfully.
        """
        model_ready: bool = getattr(request.app.state, "model_ready", False)
        edu_models = getattr(request.app.state, "edu_models", None)
        edu_ready = edu_models is not None
        bundle = getattr(request.app.state, "model", None)
        device_str = "unknown"
        if bundle is not None:
            device_str = str(bundle.device)
        elif edu_models is not None:
            device_str = edu_models.get("device", "unknown")
        return {
            "status": "ok",
            "v1_model_ready": model_ready,
            "v2_edu_ready": edu_ready,
            "v2_has_planner": edu_ready and edu_models.get("planner") is not None,
            "v2_has_qwen": edu_ready and edu_models.get("generate_fn") is not None,
            "v2_has_clip": edu_ready and edu_models.get("clip_model") is not None,
            "device": device_str,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    # ── v1 Routers (legacy VQAGuider) ─────────────────────────
    from backend.routes.video import router as video_router
    from backend.routes.query import router as query_router
    app.include_router(video_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    # ── v2 Routers (Edu-VQAGuider) ───────────────────────────
    from backend.edu.routes.edu_video import router as edu_video_router
    from backend.edu.routes.edu_query import router as edu_query_router
    app.include_router(edu_video_router, prefix="/api/v2")
    app.include_router(edu_query_router, prefix="/api/v2")
    logger.info("FastAPI app created (v1 + v2 routes registered).")
    return app
# ── Module-level app instance ─────────────────────────────────
# Uvicorn needs a module-level attribute: `uvicorn backend.main:app`
app = create_app()