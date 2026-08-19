"""
backend/edu/model_manager.py
────────────────────────────────────────────────────────────
VRAM lifecycle manager for Edu-VQAGuider.

Manages loading/unloading of heavy models to fit within
24GB RTX 3090 VRAM budget.

Two phases:
    PROCESSING: Whisper (1 GB) -- transcribe then unload
    QUERY:      CLIP (0.4 GB) + DistilBERT+Planner (0.3 GB) + Qwen VL (5-6 GB)

Public API:
    EduModelManager  — singleton that manages all model loading
    init_query_models()    — load models needed for question answering
    get_edu_state()        — return dict for app.state.edu_models
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EduModelManager:
    """
    Manages the lifecycle of all Edu-VQAGuider models.

    Usage:
        manager = EduModelManager(device="cuda")
        manager.load_query_models()   # loads CLIP, planner, Qwen
        state = manager.get_state()   # dict for app.state.edu_models
        manager.unload_all()          # frees everything
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model references
        self.clip_model = None
        self.clip_preprocess = None
        self.planner = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.generate_fn: Optional[Callable] = None

        self._query_ready = False
        logger.info("EduModelManager initialized (device=%s)", self.device)

    # ══════════════════════════════════════════════════════════
    # CLIP
    # ══════════════════════════════════════════════════════════

    def load_clip(self) -> None:
        """Load CLIP ViT-B/32 for frame selection."""
        import clip

        logger.info("Loading CLIP ViT-B/32...")
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=self.device,
        )
        self.clip_model.eval()
        self._log_vram("CLIP loaded")

    # ══════════════════════════════════════════════════════════
    # DistilBERT + EduPlanner
    # ══════════════════════════════════════════════════════════

    def load_planner(self, checkpoint_path: str | Path) -> None:
        """
        Load DistilBERT encoder + trained EduPlanner classifier.

        Args:
            checkpoint_path: Path to edu_planner.pt
        """
        from transformers import DistilBertModel, DistilBertTokenizer
        from backend.edu.planner import load_planner

        logger.info("Loading DistilBERT + EduPlanner...")

        # DistilBERT for question encoding
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.distilbert_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        ).to(self.device)
        self.distilbert_model.eval()

        # Trained planner head
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            self.planner = load_planner(checkpoint_path, device=self.device)
            logger.info("Learned planner loaded from %s", checkpoint_path)
        else:
            logger.warning(
                "Planner checkpoint not found at %s -- will use fallback only",
                checkpoint_path,
            )
            self.planner = None

        self._log_vram("DistilBERT + Planner loaded")

    def encode_question_for_planner(self, question: str) -> torch.Tensor:
        """
        Encode a question with DistilBERT for the planner.

        Args:
            question: Raw question text

        Returns:
            (768,) tensor — CLS embedding
        """
        if self.distilbert_model is None or self.distilbert_tokenizer is None:
            raise RuntimeError("DistilBERT not loaded. Call load_planner() first.")

        inputs = self.distilbert_tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)

        return cls_embedding.squeeze(0)  # (768,)

    # ══════════════════════════════════════════════════════════
    # Qwen (answer generation)
    # ══════════════════════════════════════════════════════════

    def load_qwen(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 350,
    ) -> None:
        """Load Qwen and create the generate function."""
        from backend.edu.generation import load_qwen, create_generate_fn

        model, processor = load_qwen(
            model_name=model_name,
            device=self.device,
        )

        self.generate_fn = create_generate_fn(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        self._log_vram("Qwen VL loaded")

    # ══════════════════════════════════════════════════════════
    # Composite loading
    # ══════════════════════════════════════════════════════════

    def load_query_models(
        self,
        planner_checkpoint: str | Path = "models/edu_planner.pt",
        qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 350,
        skip_qwen: bool = False,
        skip_clip: bool = False,
    ) -> None:
        """
        Load all models needed for the query phase.

        Order matters for VRAM:
            1. CLIP (small, 0.4 GB)
            2. DistilBERT + Planner (small, 0.3 GB)
            3. Qwen VL (largest, 5-6 GB in 4-bit)

        Args:
            planner_checkpoint: Path to trained planner .pt file
            qwen_model:         HuggingFace model ID for Qwen
            max_new_tokens:     Max generation length
            skip_qwen:          Skip Qwen loading (for testing without LLM)
            skip_clip:          Skip CLIP loading
        """
        logger.info("Loading query-phase models...")

        if not skip_clip:
            try:
                self.load_clip()
            except Exception as e:
                logger.error("CLIP loading failed (non-fatal): %s", e)

        try:
            self.load_planner(planner_checkpoint)
        except Exception as e:
            logger.error("Planner loading failed (non-fatal): %s", e)

        if not skip_qwen:
            try:
                self.load_qwen(qwen_model, max_new_tokens)
            except Exception as e:
                logger.error("Qwen loading failed: %s", e)

        self._query_ready = True
        self._log_vram("All query models loaded")

    # ══════════════════════════════════════════════════════════
    # State export (for app.state.edu_models)
    # ══════════════════════════════════════════════════════════

    def get_state(self) -> dict:
        """
        Return a dict suitable for FastAPI app.state.edu_models.

        This dict is consumed by edu_query.py routes to access models.
        """
        return {
            "planner": self.planner,
            "clip_model": self.clip_model,
            "clip_preprocess": self.clip_preprocess,
            "generate_fn": self.generate_fn,
            "device": self.device,
            "distilbert_encode": self.encode_question_for_planner,
            "manager": self,  # for lifecycle management
        }

    @property
    def is_ready(self) -> bool:
        return self._query_ready

    # ══════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════

    def unload_all(self) -> None:
        """Free all models and VRAM."""
        from backend.edu.generation import unload_qwen

        if self.clip_model is not None:
            del self.clip_model, self.clip_preprocess
            self.clip_model = None
            self.clip_preprocess = None

        if self.distilbert_model is not None:
            del self.distilbert_model, self.distilbert_tokenizer
            self.distilbert_model = None
            self.distilbert_tokenizer = None

        if self.planner is not None:
            del self.planner
            self.planner = None

        try:
            unload_qwen()
        except Exception:
            pass

        self.generate_fn = None
        self._query_ready = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("All edu models unloaded")

    # ── Helpers ───────────────────────────────────────────────

    def _log_vram(self, label: str) -> None:
        """Log current VRAM usage."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(
                "%s | VRAM: %.2f GB allocated, %.2f GB reserved",
                label, alloc, reserved,
            )
