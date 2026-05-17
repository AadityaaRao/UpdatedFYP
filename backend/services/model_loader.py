"""
backend/services/model_loader.py
────────────────────────────────────────────────────────────
Single-responsibility: load ALL model components exactly once
at application startup and expose them as a typed bundle.
Rules enforced here:
  • All models → eval() mode immediately after loading
  • Phi-2 and DistilBERT parameters → requires_grad = False
  • Device is resolved once and stored on the bundle
  • The checkpoint keys match the training save format exactly:
      {"vqaguider": ..., "projector": ..., "scorer": ...}
Usage (in main.py lifespan):
    bundle = load_model_bundle()
    app.state.model = bundle
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
# Architecture definitions (inference only)
from backend.models.architectures import (
    LLMProjector,
    OptionScorer,
    TemporalAttentionPooling,
    VQAGuiderCore,
)
from backend.utils.logger import get_logger
logger = get_logger(__name__)
# ── Typed bundle ──────────────────────────────────────────────
@dataclass
class ModelBundle:
    """
    Immutable container for all inference components.
    Passed via app.state.model to every request handler.
    """
    device: torch.device
    # Core VQA components (trained weights)
    vqaguider: VQAGuiderCore
    projector: LLMProjector
    scorer: OptionScorer
    # Frozen encoders
    video_encoder: "VideoEncoder"          # defined below
    question_encoder: "QuestionEncoder"    # defined below
    # Generative LLM (frozen)
    phi2: nn.Module
    phi2_tokenizer: object  # transformers tokenizer (not an nn.Module)
# ── Video encoder ─────────────────────────────────────────────
class VideoEncoder:
    """
    CLIP ViT-B/32 frame encoder + learned temporal attention pooling.
    Inference-only; no gradient tracking.
    """
    def __init__(self, device: torch.device):
        import clip  # openai/CLIP
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        self.pooling = TemporalAttentionPooling(dim=512).to(device)
        self.pooling.eval()
        logger.info("VideoEncoder (CLIP ViT-B/32 + TemporalAttentionPooling) loaded.")
    @torch.no_grad()
    def encode(self, video_path: str | Path, num_frames: int = 16) -> torch.Tensor:
        """
        Encode a video file → (512,) tensor on self.device.
        """
        from backend.utils.video_utils import sample_frames
        from PIL import Image
        frames = sample_frames(str(video_path), num_frames)
        images = torch.stack(
            [self.preprocess(Image.fromarray(f)) for f in frames]
        ).to(self.device)
        frame_embeddings = self.clip_model.encode_image(images).float()
        frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
        video_feat, _ = self.pooling(frame_embeddings)  # (512,)
        return video_feat
# ── Question encoder ──────────────────────────────────────────
class QuestionEncoder:
    """
    DistilBERT-based encoder for question (and question+option) text.
    Weights are frozen; no gradient tracking.
    """
    def __init__(self, device: torch.device):
        from transformers import DistilBertModel, DistilBertTokenizer
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        logger.info("QuestionEncoder (DistilBERT) loaded and frozen.")
    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode arbitrary text → (768,) CLS embedding on self.device.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
    @torch.no_grad()
    def encode_question_option(self, question: str, option: str) -> torch.Tensor:
        """
        Encode "question [SEP] option" → (768,) — matches training exactly.
        """
        return self.encode(f"{question} [SEP] {option}")
    @torch.no_grad()
    def encode_question(self, question: str) -> torch.Tensor:
        """
        Encode question alone → (768,) — used for generative inference.
        """
        return self.encode(question)
# ── Main loader ───────────────────────────────────────────────
def _resolve_device(preferred: str = "auto") -> torch.device:
    """
    Resolve device from config or environment.
    'auto'  → CUDA if available, else CPU
    'cuda'  → force GPU (raises if unavailable)
    'cpu'   → force CPU
    """
    if preferred == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(preferred)
    logger.info("Using device: %s", device)
    return device
def load_model_bundle(checkpoint_path: Path, device_preference: str = "auto") -> ModelBundle:
    """
    Load all model components from a single checkpoint file.
    Call this ONCE at application startup.
    Args:
        checkpoint_path:   Path to vqa_model_final.pt
        device_preference: 'auto' | 'cuda' | 'cpu'
    Returns:
        ModelBundle — all components in eval mode
    """
    device = _resolve_device(device_preference)
    # ── 1. Verify checkpoint exists ───────────────────────────
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        logger.warning(
            "Starting WITHOUT trained weights. "
            "Place vqa_model_final.pt in the models/ directory."
        )
        ckpt = None
    else:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)
        logger.info(
            "Checkpoint loaded (epoch=%s, best_acc=%s)",
            ckpt.get("epoch", "?"),
            ckpt.get("best_acc", "?"),
        )
    # ── 2. Instantiate core VQA models ────────────────────────
    from config import HIDDEN_DIM, LLM_DIM, NUM_TASKS, NUM_TOKENS, QUESTION_DIM, VIDEO_DIM
    vqaguider = VQAGuiderCore(
        video_dim=VIDEO_DIM,
        question_dim=QUESTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_tasks=NUM_TASKS,
    ).to(device)
    projector = LLMProjector(
        input_dim=VIDEO_DIM,
        llm_dim=LLM_DIM,
        num_tokens=NUM_TOKENS,
    ).to(device)
    scorer = OptionScorer(dim=VIDEO_DIM).to(device)
    # ── 3. Load trained weights (if checkpoint available) ──────
    if ckpt is not None:
        vqaguider.load_state_dict(ckpt["vqaguider"])
        logger.info("VQAGuiderCore weights loaded.")
        projector.load_state_dict(ckpt["projector"])
        logger.info("LLMProjector weights loaded.")
        scorer.load_state_dict(ckpt["scorer"])
        logger.info("OptionScorer weights loaded.")
    else:
        logger.warning("Running with randomly-initialized VQA weights.")
    # ── 4. Set core models to eval mode ───────────────────────
    vqaguider.eval()
    projector.eval()
    scorer.eval()
    # Freeze gradients (belt-and-suspenders: eval() doesn't disable grad)
    for model in [vqaguider, projector, scorer]:
        for p in model.parameters():
            p.requires_grad = False
    logger.info("VQAGuiderCore, LLMProjector, OptionScorer → eval + frozen.")
    # ── 5. Load Phi-2 ─────────────────────────────────────────
    from config import MAX_NEW_TOKENS, PHI2_MODEL_NAME
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading Phi-2 (%s) — this may take a moment...", PHI2_MODEL_NAME)
    phi2_tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_NAME, trust_remote_code=True)
    phi2_tokenizer.pad_token = phi2_tokenizer.eos_token
    # float16 on CPU is emulated and causes massive slowdowns/hangs during shard loading.
    # bfloat16 is natively supported on modern CPUs and avoids the hang.
    dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    
    phi2 = AutoModelForCausalLM.from_pretrained(
        PHI2_MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    phi2.eval()
    for p in phi2.parameters():
        p.requires_grad = False
    logger.info("Phi-2 loaded and frozen.")
    # ── 6. Load encoders ──────────────────────────────────────
    logger.info("Loading VideoEncoder (CLIP + TemporalAttentionPooling)...")
    video_encoder = VideoEncoder(device=device)
    # Load trained TemporalAttentionPooling weights if checkpoint contains them
    if ckpt is not None and "temporal_pooling" in ckpt:
        video_encoder.pooling.load_state_dict(ckpt["temporal_pooling"])
        logger.info("TemporalAttentionPooling trained weights loaded.")
    else:
        logger.info("TemporalAttentionPooling using default weights (no trained weights in checkpoint).")
    video_encoder.pooling.eval()
    for p in video_encoder.pooling.parameters():
        p.requires_grad = False
    logger.info("Loading QuestionEncoder (DistilBERT)...")
    question_encoder = QuestionEncoder(device=device)
    # ── 7. Assemble and return bundle ─────────────────────────
    bundle = ModelBundle(
        device=device,
        vqaguider=vqaguider,
        projector=projector,
        scorer=scorer,
        video_encoder=video_encoder,
        question_encoder=question_encoder,
        phi2=phi2,
        phi2_tokenizer=phi2_tokenizer,
    )
    logger.info("ModelBundle ready. All components loaded successfully.")
    return bundle