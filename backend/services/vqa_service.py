"""
backend/services/vqa_service.py
────────────────────────────────────────────────────────────
Core inference pipeline for VQA Guider.
This module contains ONLY inference logic — no HTTP, no cache,
no database. Each function has a single clear responsibility
and can be called independently or composed in sequence.
Inference data flow:
  video_path ──► load_video_features()  ──► video_feat  (512,)
  question   ──► get_question_embedding() ─► q_feat     (768,)
                                                │
               run_vqa_model(video_feat, q_feat)│
                  ├──► VQAGuiderCore ───────────┤
                  │      fusion_vec  (512,)      │
                  │      task_probs  (3,)  ──────┤
                  └──► softmax(task_probs) ──────►─────────────────────┐
                                                 │                     │
               generate_answer(fusion_vec, q)   │                     │
                  ├──► LLMProjector ────── prefix (1,10,2560)          │
                  ├──► phi2_tokenizer ─── token embeds                 │
                  ├──► phi2.generate() ── output tokens                │
                  └──► decode + clean ─── answer str ──────────────────┤
                                                                        │
               assemble_response(answer, task_probs) ◄─────────────────┘
                  └──► VQAResponse dataclass
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from backend.services.model_loader import ModelBundle
from backend.utils.logger import get_logger
logger = get_logger(__name__)
# ── Response schema ───────────────────────────────────────────
@dataclass
class TaskRouting:
    """
    Normalized task probabilities (sum to 1.0 after softmax).
    Maps to the three specialist heads in VQAGuiderCore.
    """
    action: float
    tracking: float
    scene: float
    def to_dict(self) -> dict:
        return {
            "action": round(self.action, 4),
            "tracking": round(self.tracking, 4),
            "scene": round(self.scene, 4),
        }
@dataclass
class VQAResponse:
    """
    Final inference output returned by assemble_response().
    This is the contract between the service layer and the API layer.
    """
    answer: str
    task_routing: TaskRouting
    result_id: Optional[str] = None   # filled in by the API layer after DB insert
    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "task_routing": self.task_routing.to_dict(),
            "result_id": self.result_id,
        }
# ── Step 1 — Video encoding ───────────────────────────────────
@torch.no_grad()
def load_video_features(
    model: ModelBundle,
    video_path: str | Path,
    num_frames: int = 16,
) -> torch.Tensor:
    """
    Encode a video file into a fixed-size feature vector.
    Pipeline:
        video_path
          → sample_frames()          (N RGB frames)
          → CLIP.encode_image()      (N, 512) — unit-normed
          → TemporalAttentionPooling (512,)   — weighted sum
    Args:
        model:      ModelBundle holding VideoEncoder
        video_path: Absolute path to the video file
        num_frames: Number of frames to sample (default 16)
    Returns:
        Tensor of shape (512,) on model.device
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    logger.debug("Encoding video: %s", video_path.name)
    video_feat = model.video_encoder.encode(video_path, num_frames=num_frames)
    # Guarantee correct device
    video_feat = video_feat.to(model.device)
    logger.debug("video_feat shape: %s, device: %s", video_feat.shape, video_feat.device)
    return video_feat  # (512,)
# ── Step 2 — Question encoding ────────────────────────────────
@torch.no_grad()
def get_question_embedding(
    model: ModelBundle,
    question: str,
) -> torch.Tensor:
    """
    Encode the question into a fixed-size embedding.
    Uses question-only encoding (no MCQ options) — matches the
    generative training pipeline exactly.
    Args:
        model:        ModelBundle holding QuestionEncoder
        question:     Natural language question string
    Returns:
        Tensor of shape (768,) on model.device — CLS embedding
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    logger.debug("Encoding question: '%s'", question[:60])
    q_feat = model.question_encoder.encode_question(question)
    # Guarantee correct device
    q_feat = q_feat.to(model.device)
    logger.debug("q_feat shape: %s, device: %s", q_feat.shape, q_feat.device)
    return q_feat  # (768,)
# ── Step 3 — VQA core model ───────────────────────────────────
@torch.no_grad()
def run_vqa_model(
    model: ModelBundle,
    video_feat: torch.Tensor,
    question_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run VQAGuiderCore to produce a fusion vector and task routing probs.
    Shapes expected:
        video_feat    : (512,)   → batched to (1, 512) internally
        question_feat : (768,)   → batched to (1, 768) internally
    Returns:
        fusion_vec  : (512,)  — multimodal representation
        task_probs  : (3,)    — softmax-normalized routing weights
                                [action, tracking, scene]
    Note on task_probs normalization:
        VQAGuiderCore outputs sigmoid(logits), which does NOT sum to 1.
        We apply softmax here to produce a proper probability simplex
        suitable for visualization and interpretation.
    """
    # Add batch dimension for the model
    vf = video_feat.unsqueeze(0).to(model.device)    # (1, 512)
    qf = question_feat.unsqueeze(0).to(model.device)  # (1, 768)
    # Forward pass through VQAGuiderCore
    fusion_vec, task_probs_raw = model.vqaguider(vf, qf)
    # fusion_vec  : (1, 512)
    # task_probs_raw : (1, 3)  — sigmoid-activated
    # Remove batch dim
    fusion_vec = fusion_vec.squeeze(0)         # (512,)
    task_probs_raw = task_probs_raw.squeeze(0) # (3,)
    # Apply softmax so probabilities sum to 1 for display
    task_probs = F.softmax(task_probs_raw, dim=0)  # (3,)
    logger.debug(
        "VQAGuider output | fusion: %s | task_probs (softmax): action=%.3f "
        "tracking=%.3f scene=%.3f",
        fusion_vec.shape,
        task_probs[0].item(),
        task_probs[1].item(),
        task_probs[2].item(),
    )
    return fusion_vec, task_probs  # both on model.device
# ── Step 4 — Generative answer ────────────────────────────────
@torch.no_grad()
def generate_answer(
    model: ModelBundle,
    fusion_vec: torch.Tensor,
    question: str,
    max_new_tokens: int = 50,
) -> str:
    """
    Generate a free-text answer using LLMProjector + Phi-2.
    Pipeline:
        fusion_vec (512,)
          → LLMProjector   → prefix (1, 10, 2560)   [visual context]
          → phi2_tokenizer → token_embeds (1, T, 2560) [question prompt]
          → concatenate    → inputs_embeds (1, 10+T, 2560)
          → phi2.generate  → output token ids
          → decode + clean → answer string
    The prefix injects the video+question context directly into
    Phi-2's embedding space, bypassing the text encoder for visual info.
    Args:
        model:          ModelBundle
        fusion_vec:     (512,) tensor from run_vqa_model()
        question:       Original question string
        max_new_tokens: Cap on generated token count
    Returns:
        Cleaned answer string (single sentence, no prompt prefix)
    """
    from config import NUM_TOKENS
    device = model.device
    # ── 1. Build prefix tokens from fusion vector ─────────────
    fv_batch = fusion_vec.unsqueeze(0).to(device)          # (1, 512)
    prefix = model.projector(fv_batch).to(dtype=torch.float16)  # (1, 10, 2560)
    # ── 2. Build the text prompt ──────────────────────────────
    prompt = (
        "You are a video understanding AI. "
        "Watch the video carefully and answer the question.\n"
        f"Question: {question}\n"
        "Detailed Answer:"
    )
    enc = model.phi2_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    # ── 3. Get Phi-2 token embeddings for the prompt ──────────
    token_embeds = model.phi2.get_input_embeddings()(enc.input_ids)
    token_embeds = token_embeds.to(dtype=torch.float16)    # (1, T, 2560)
    # ── 4. Concatenate: [visual prefix | text tokens] ─────────
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)  # (1, 10+T, 2560)
    # ── 5. Build attention mask (prefix is always attended to) ─
    prefix_mask = torch.ones(
        (1, NUM_TOKENS),
        dtype=enc.attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([prefix_mask, enc.attention_mask], dim=1)
    # ── 6. Generate ───────────────────────────────────────────
    logger.debug("Generating answer with Phi-2 (max_new_tokens=%d)…", max_new_tokens)
    output_ids = model.phi2.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,                              # deterministic greedy
        pad_token_id=model.phi2_tokenizer.eos_token_id,
    )
    # ── 7. Decode and clean ───────────────────────────────────
    raw_text = model.phi2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = _clean_generated_answer(raw_text)
    logger.debug("Generated answer: '%s'", answer)
    return answer
def _clean_generated_answer(raw_text: str) -> str:
    """
    Extract the answer portion from Phi-2's full decoded output.
    Phi-2 echoes the prompt, so we split on the sentinel and take
    only what follows. Multiple cleanup passes handle edge cases.
    """
    # Split on our prompt sentinel
    if "Detailed Answer:" in raw_text:
        answer = raw_text.split("Detailed Answer:")[-1].strip()
    else:
        answer = raw_text.strip()
    # Take only the first sentence / line
    answer = answer.split("\n")[0].strip()
    # Remove any trailing prompt artifacts
    for artifact in ["Question:", "Answer:", "You are"]:
        if artifact in answer:
            answer = answer.split(artifact)[0].strip()
    # Collapse multiple spaces
    answer = re.sub(r"\s+", " ", answer).strip()
    # Fallback if cleaning left nothing
    if not answer:
        answer = "The model could not generate a clear answer."
    return answer
# ── Step 5 — Assemble response ────────────────────────────────
def assemble_response(
    answer: str,
    task_probs: torch.Tensor,
    result_id: Optional[str] = None,
) -> VQAResponse:
    """
    Package the answer and routing probabilities into a typed response.
    Args:
        answer:      Generated answer string
        task_probs:  (3,) softmax-normalized tensor [action, tracking, scene]
        result_id:   Optional DB result ID (filled in by API layer)
    Returns:
        VQAResponse dataclass — call .to_dict() for JSON serialization
    """
    if task_probs.shape != torch.Size([3]):
        raise ValueError(f"task_probs must be shape (3,), got {task_probs.shape}")
    # Move to CPU for Python-side handling
    probs = task_probs.detach().cpu().tolist()
    routing = TaskRouting(
        action=probs[0],
        tracking=probs[1],
        scene=probs[2],
    )
    response = VQAResponse(
        answer=answer,
        task_routing=routing,
        result_id=result_id,
    )
    logger.debug(
        "Response assembled | answer='%s...' | routing=%s",
        answer[:40],
        routing.to_dict(),
    )
    return response
# ── Full pipeline convenience function ────────────────────────
@torch.no_grad()
def run_full_pipeline(
    model: ModelBundle,
    video_path: str | Path,
    question: str,
    num_frames: int = 16,
    max_new_tokens: int = 50,
) -> VQAResponse:
    """
    End-to-end convenience wrapper — runs all steps in sequence.
    Useful for:
      • Direct testing without the cache/DB layer
      • Integration tests
    The API route handlers do NOT call this directly; they call
    each step individually so the cache layer can intercept between
    steps (e.g., return cached video_feat without re-encoding).
    Args:
        model:          ModelBundle from app.state
        video_path:     Path to the uploaded video file
        question:       Natural language question
        num_frames:     Frames to sample from the video
        max_new_tokens: Max tokens for Phi-2 to generate
    Returns:
        VQAResponse — ready for JSON serialization
    """
    logger.info("run_full_pipeline | video=%s | question='%s'",
                Path(video_path).name, question[:60])
    # Step 1 — Video encoding
    video_feat = load_video_features(model, video_path, num_frames)
    # Step 2 — Question encoding
    question_feat = get_question_embedding(model, question)
    # Step 3 — VQA core (routing + fusion)
    fusion_vec, task_probs = run_vqa_model(model, video_feat, question_feat)
    # Step 4 — Generative answer
    answer = generate_answer(model, fusion_vec, question, max_new_tokens)
    # Step 5 — Assemble
    response = assemble_response(answer, task_probs)
    logger.info(
        "Pipeline complete | answer='%s...' | action=%.3f tracking=%.3f scene=%.3f",
        answer[:40],
        task_probs[0].item(),
        task_probs[1].item(),
        task_probs[2].item(),
    )
    return response