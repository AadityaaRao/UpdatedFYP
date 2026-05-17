"""
config.py
─────────────────────────────────────────────────────────────
Central configuration for VQA Guider.
All paths, flags, and hyper-parameters live here.
No hardcoded values anywhere else in the codebase.
"""
import os
from pathlib import Path
# ── Root paths ────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent          # project root
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR  = BASE_DIR / "cache"
DB_DIR     = BASE_DIR / "database"
UPLOADS_DIR = BASE_DIR / "uploads"
EDU_CACHE_DIR = CACHE_DIR / "edu_chunks"
# ── Model checkpoint ──────────────────────────────────────────
MODEL_CHECKPOINT = MODELS_DIR / "vqa_model_generative.pt"
# ── Database ──────────────────────────────────────────────────
SQLITE_PATH = DB_DIR / "vqaguider.db"
# ── Cache (file-based pickle) ─────────────────────────────────
VIDEO_FEATURE_CACHE = CACHE_DIR / "video_features.pkl"
EMBEDDING_CACHE     = CACHE_DIR / "embeddings.pkl"
ANSWER_CACHE        = CACHE_DIR / "answers.pkl"
# ── Model architecture parameters ────────────────────────────
VIDEO_DIM    = 512
QUESTION_DIM = 768
HIDDEN_DIM   = 512
NUM_TASKS    = 3       # Action, Tracking, Scene
LLM_DIM      = 2560   # Phi-2 hidden size
NUM_TOKENS   = 10     # LLMProjector prefix tokens
# ── Phi-2 ─────────────────────────────────────────────────────
PHI2_MODEL_NAME = "microsoft/phi-2"
MAX_NEW_TOKENS  = 50
# ── Video encoding ────────────────────────────────────────────
NUM_FRAMES = 16
# ── Firebase (optional — set to True to enable) ───────────────
FIREBASE_ENABLED         = False
FIREBASE_CREDENTIALS_PATH = BASE_DIR / "firebase_credentials.json"
# ── Upload limits ─────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 500
# ── Device ────────────────────────────────────────────────────
# 'auto' → CUDA if available, else CPU
# 'cuda' → force GPU  |  'cpu' → force CPU
DEVICE_PREFERENCE = "auto"
LEGACY_VQA_ENABLED = os.getenv("LEGACY_VQA_ENABLED", "false").lower() in {"1", "true", "yes"}

# Edu-VQAGuider defaults
EDU_CHUNK_SECONDS = int(os.getenv("EDU_CHUNK_SECONDS", "60"))
EDU_FRAMES_PER_CHUNK = int(os.getenv("EDU_FRAMES_PER_CHUNK", "4"))
EDU_TOP_K = int(os.getenv("EDU_TOP_K", "3"))
EDU_SUMMARY_TOP_K = int(os.getenv("EDU_SUMMARY_TOP_K", "5"))
EDU_MAX_NEW_TOKENS = int(os.getenv("EDU_MAX_NEW_TOKENS", "600"))
EDU_WHISPER_MODEL = os.getenv("EDU_WHISPER_MODEL", "small")
EDU_QWEN_MODEL = os.getenv("EDU_QWEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")
# ── Ensure directories exist at import time ───────────────────
for _dir in [MODELS_DIR, CACHE_DIR, EDU_CACHE_DIR, DB_DIR, UPLOADS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
