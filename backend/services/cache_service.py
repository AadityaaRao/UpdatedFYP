"""
backend/services/cache_service.py
────────────────────────────────────────────────────────────
File-based pickle cache with in-memory backing, lazy loading,
and per-cache-file thread locking.
Architecture:
  PickleCache          — generic key/value store backed by a .pkl file
  CacheService         — typed facade with domain-specific methods
  cache                — module-level singleton (import and use directly)
Cache files:
  cache/video_features.pkl  →  video_id         → torch.Tensor (512,) CPU
  cache/embeddings.pkl      →  question str      → torch.Tensor (768,) CPU
  cache/answers.pkl         →  "video_id::q"     → dict {answer, task_routing, ...}
Thread safety:
  Each PickleCache instance owns a threading.Lock.
  All reads and writes acquire the lock before touching _store or disk.
  Disk writes are atomic: write to .tmp → os.replace() (rename on POSIX,
  MoveFileEx on Windows) — no partial-write corruption.
Lazy loading:
  _store is None until the first get/set call. On first access the .pkl
  is deserialized into memory once. Subsequent calls hit the in-memory
  dict — zero disk I/O per request after the first.
"""
from __future__ import annotations
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Optional
import torch
from backend.utils.logger import get_logger
from config import ANSWER_CACHE, EMBEDDING_CACHE, VIDEO_FEATURE_CACHE
logger = get_logger(__name__)
# ══════════════════════════════════════════════════════════════
# Generic pickle cache — not domain-aware
# ══════════════════════════════════════════════════════════════
class PickleCache:
    """
    Generic file-backed key/value cache.
    • In-memory dict (_store) acts as an L1 cache.
    • Lazy loading: disk is read at most once per process lifetime.
    • Atomic writes: data goes to a .tmp file first, then os.replace()
      so a crash never leaves a corrupted .pkl on disk.
    • Thread safety: a per-instance Lock guards all _store access.
    """
    def __init__(self, cache_path: Path) -> None:
        self._path: Path = cache_path
        self._lock: threading.Lock = threading.Lock()
        self._store: Optional[dict] = None   # None = "not yet loaded"
        # Ensure the cache directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
    # ── Internal: lazy load ───────────────────────────────────
    def _load(self) -> dict:
        """
        Load the pickle file from disk on first call.
        Must be called with self._lock already held.
        """
        if self._store is not None:
            return self._store   # already in memory
        if self._path.exists():
            try:
                with self._path.open("rb") as fh:
                    self._store = pickle.load(fh)
                logger.info(
                    "Cache loaded: %s  (%d entries)",
                    self._path.name, len(self._store),
                )
            except (pickle.UnpicklingError, EOFError, Exception) as exc:
                logger.warning(
                    "Cache file '%s' is corrupted — starting fresh. Error: %s",
                    self._path.name, exc,
                )
                self._store = {}
        else:
            logger.debug("Cache file '%s' not found — starting fresh.", self._path.name)
            self._store = {}
        return self._store
    # ── Internal: atomic flush to disk ────────────────────────
    def _flush(self) -> None:
        """
        Persist _store to disk atomically.
        Must be called with self._lock already held.
        """
        tmp_path = self._path.with_suffix(".pkl.tmp")
        try:
            with tmp_path.open("wb") as fh:
                pickle.dump(self._store, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # os.replace is atomic on POSIX and Windows (MoveFileEx)
            os.replace(tmp_path, self._path)
        except Exception as exc:
            logger.error("Cache flush failed for '%s': %s", self._path.name, exc)
            tmp_path.unlink(missing_ok=True)   # clean up failed temp
    # ── Public API ────────────────────────────────────────────
    def get(self, key: Any) -> Optional[Any]:
        """Return cached value or None if key is absent."""
        with self._lock:
            store = self._load()
            value = store.get(key)
            if value is None:
                logger.debug("Cache MISS: %s | key=%r", self._path.name, _trunc(key))
            else:
                logger.debug("Cache HIT:  %s | key=%r", self._path.name, _trunc(key))
            return value
    def set(self, key: Any, value: Any) -> None:
        """Store value under key and flush to disk."""
        with self._lock:
            store = self._load()
            store[key] = value
            self._flush()
            logger.debug(
                "Cache SET:  %s | key=%r  (total=%d)",
                self._path.name, _trunc(key), len(store),
            )
    def has(self, key: Any) -> bool:
        """Return True if key exists in the cache."""
        with self._lock:
            return key in self._load()
    def delete(self, key: Any) -> bool:
        """Remove a key. Returns True if it existed."""
        with self._lock:
            store = self._load()
            existed = key in store
            if existed:
                del store[key]
                self._flush()
                logger.debug("Cache DEL: %s | key=%r", self._path.name, _trunc(key))
            return existed
    def size(self) -> int:
        """Number of entries currently in the cache."""
        with self._lock:
            return len(self._load())
    def clear(self) -> None:
        """Wipe all entries and persist the empty state to disk."""
        with self._lock:
            self._store = {}
            self._flush()
            logger.info("Cache CLEARED: %s", self._path.name)
    def stats(self) -> dict:
        """Return basic statistics for monitoring / /health endpoint."""
        with self._lock:
            size = len(self._load())
        disk_bytes = self._path.stat().st_size if self._path.exists() else 0
        return {
            "file": self._path.name,
            "entries": size,
            "disk_bytes": disk_bytes,
        }
# ══════════════════════════════════════════════════════════════
# Typed facade — domain-aware cache service
# ══════════════════════════════════════════════════════════════
class CacheService:
    """
    Domain-specific cache service wrapping three PickleCache instances.
    Typed API:
        get_video_feature / set_video_feature   → torch.Tensor (512,) CPU
        get_embedding     / set_embedding        → torch.Tensor (768,) CPU
        get_answer        / set_answer           → dict
    Key conventions:
        video features  →  video_id                         (UUID string)
        embeddings      →  _normalize(question)             (lowercased + stripped)
        answers         →  video_id + "::" + _normalize(q)  (composite string)
    Tensors are always stored on CPU and returned on CPU.
    The calling code (query.py) moves them to the correct device after retrieval.
    """
    def __init__(self) -> None:
        self._video_cache: PickleCache = PickleCache(VIDEO_FEATURE_CACHE)
        self._embed_cache: PickleCache = PickleCache(EMBEDDING_CACHE)
        self._answer_cache: PickleCache = PickleCache(ANSWER_CACHE)
        logger.info(
            "CacheService initialized | video=%s | embed=%s | answer=%s",
            VIDEO_FEATURE_CACHE.name,
            EMBEDDING_CACHE.name,
            ANSWER_CACHE.name,
        )
    # ── Key builders ──────────────────────────────────────────
    @staticmethod
    def _video_key(video_id: str) -> str:
        return video_id.strip()
    @staticmethod
    def _embed_key(question: str) -> str:
        return question.strip().lower()
    @staticmethod
    def _answer_key(video_id: str, question: str) -> str:
        return f"{video_id.strip()}::{question.strip().lower()}"
    # ── Video feature cache ───────────────────────────────────
    def get_video_feature(self, video_id: str) -> Optional[torch.Tensor]:
        """
        Return cached (512,) video feature tensor on CPU,
        or None on cache miss.
        """
        key = self._video_key(video_id)
        value = self._video_cache.get(key)
        if value is not None and not isinstance(value, torch.Tensor):
            logger.warning("Unexpected type in video cache: %s — evicting.", type(value))
            self._video_cache.delete(key)
            return None
        return value  # CPU tensor or None
    def set_video_feature(self, video_id: str, feat: torch.Tensor) -> None:
        """
        Cache a video feature tensor.
        Tensor is moved to CPU and detached before pickling.
        """
        key = self._video_key(video_id)
        cpu_feat = feat.detach().cpu()
        self._video_cache.set(key, cpu_feat)
    def has_video_feature(self, video_id: str) -> bool:
        return self._video_cache.has(self._video_key(video_id))
    # ── Question embedding cache ──────────────────────────────
    def get_embedding(self, question: str) -> Optional[torch.Tensor]:
        """
        Return cached (768,) question embedding on CPU,
        or None on cache miss.
        """
        key = self._embed_key(question)
        value = self._embed_cache.get(key)
        if value is not None and not isinstance(value, torch.Tensor):
            logger.warning("Unexpected type in embed cache: %s — evicting.", type(value))
            self._embed_cache.delete(key)
            return None
        return value  # CPU tensor or None
    def set_embedding(self, question: str, emb: torch.Tensor) -> None:
        """
        Cache a question embedding tensor.
        Tensor is moved to CPU and detached before pickling.
        """
        key = self._embed_key(question)
        cpu_emb = emb.detach().cpu()
        self._embed_cache.set(key, cpu_emb)
    def has_embedding(self, question: str) -> bool:
        return self._embed_cache.has(self._embed_key(question))
    # ── Answer cache ──────────────────────────────────────────
    def get_answer(self, video_id: str, question: str) -> Optional[dict]:
        """
        Return cached answer dict or None on miss.
        Expected structure:
            {
                "answer": str,
                "task_routing": {"action": float, "tracking": float, "scene": float},
                "result_id": str,
                "created_at": str,
            }
        """
        key = self._answer_key(video_id, question)
        value = self._answer_cache.get(key)
        if value is not None and not isinstance(value, dict):
            logger.warning("Unexpected type in answer cache: %s — evicting.", type(value))
            self._answer_cache.delete(key)
            return None
        return value
    def set_answer(self, video_id: str, question: str, data: dict) -> None:
        """
        Cache an answer dict.
        Ensures no tensors are stored (only JSON-serializable types).
        """
        key = self._answer_key(video_id, question)
        # Defensive: convert any stray tensors to Python scalars
        safe_data = _sanitize_for_cache(data)
        self._answer_cache.set(key, safe_data)
    def has_answer(self, video_id: str, question: str) -> bool:
        return self._answer_cache.has(self._answer_key(video_id, question))
    # ── Utility ───────────────────────────────────────────────
    def stats(self) -> dict:
        """Aggregate stats for all three caches — used by /health."""
        return {
            "video_features": self._video_cache.stats(),
            "embeddings": self._embed_cache.stats(),
            "answers": self._answer_cache.stats(),
        }
    def clear_all(self) -> None:
        """Wipe all three caches. Use with caution."""
        self._video_cache.clear()
        self._embed_cache.clear()
        self._answer_cache.clear()
        logger.info("All caches cleared.")
# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def _trunc(key: Any, limit: int = 60) -> str:
    """Truncate key repr for readable log lines."""
    s = repr(key)
    return s if len(s) <= limit else s[:limit] + "…"
def _sanitize_for_cache(data: dict) -> dict:
    """
    Recursively convert any torch.Tensor values to Python lists/floats
    so the dict is fully picklable without PyTorch tensors.
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().tolist()
        elif isinstance(v, dict):
            out[k] = _sanitize_for_cache(v)
        else:
            out[k] = v
    return out
# ══════════════════════════════════════════════════════════════
# Module-level singleton
# ══════════════════════════════════════════════════════════════
# Import and use this directly in query.py:
#   from backend.services.cache_service import cache
#
# The singleton is created once at module import time — Python's
# import system guarantees this is thread-safe.
cache: CacheService = CacheService()