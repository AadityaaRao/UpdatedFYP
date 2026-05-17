"""
backend/edu/planner.py
────────────────────────────────────────────────────────────
Edu-VQAGuider Planner — learned question-intent classifier.

Architecture (inspired by VQAGuiderCore.task_planner):
    DistilBERT CLS embedding (768-dim)
        → Linear(768, 256) → ReLU → Dropout(0.2)
        → Linear(256, 5)
        → softmax → route probabilities

The planner is a STANDALONE module, not embedded in VQAGuiderCore.
It can be loaded and used independently.

Public API:
    EduPlanner          — nn.Module for route classification
    load_planner()      — load trained planner from checkpoint
    classify_question() — full pipeline: text → DistilBERT → route
    fallback_classify() — rule-based fallback for low-confidence cases

Training data format (CSV):
    question,route
    "Why does entropy increase?",concept
    "What are the steps to solve this?",procedure
    ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ── Route labels (order matches Linear output) ───────────────
ROUTE_LABELS: list[str] = ["concept", "procedure", "temporal", "visual", "summary"]
ROUTE_TO_IDX: dict[str, int] = {r: i for i, r in enumerate(ROUTE_LABELS)}
NUM_ROUTES: int = len(ROUTE_LABELS)

# ── Fallback confidence threshold ─────────────────────────────
FALLBACK_CONFIDENCE_THRESHOLD: float = 0.35


@dataclass
class PlannerResult:
    """Output of the planner: route + confidence + source."""
    route: str               # one of ROUTE_LABELS
    confidence: float        # softmax probability of chosen route
    source: str              # "learned" or "fallback"
    all_scores: dict[str, float]  # scores for all 5 routes


# ══════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════

class EduPlanner(nn.Module):
    """
    Lightweight question-intent classifier for educational questions.

    Architecture mirrors VQAGuiderCore.task_planner but with 5 outputs
    instead of 3, and its own dropout/hidden-dim configuration.

    Input:  (B, 768) — DistilBERT CLS embedding
    Output: (B, 5)   — logits for [concept, procedure, temporal, visual, summary]
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_routes: int = NUM_ROUTES,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_routes),
        )

    def forward(self, question_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            question_embedding: (B, 768) DistilBERT CLS features

        Returns:
            (B, num_routes) raw logits — apply softmax externally
        """
        return self.classifier(question_embedding)


# ══════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════

def load_planner(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> EduPlanner:
    """
    Load a trained EduPlanner from a .pt checkpoint.

    The checkpoint should contain at minimum:
        {"planner_state_dict": OrderedDict(...)}

    Args:
        checkpoint_path: Path to the .pt file
        device:          Target device

    Returns:
        EduPlanner in eval mode on the specified device
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Planner checkpoint not found: {checkpoint_path}")

    planner = EduPlanner()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    planner.load_state_dict(ckpt["planner_state_dict"])
    planner.to(device)
    planner.eval()

    logger.info("EduPlanner loaded from %s", checkpoint_path)
    return planner


# ══════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def classify_question(
    planner: EduPlanner,
    question_embedding: torch.Tensor,
    use_fallback: bool = True,
) -> PlannerResult:
    """
    Classify a question into one of 5 education routes.

    Uses the learned planner first. Falls back to rule-based
    classification if confidence is below threshold.

    Args:
        planner:            Trained EduPlanner module
        question_embedding: (768,) DistilBERT CLS embedding
        use_fallback:       Whether to use rule-based fallback on low confidence

    Returns:
        PlannerResult with route, confidence, source, and all scores
    """
    # Add batch dim if needed
    if question_embedding.dim() == 1:
        question_embedding = question_embedding.unsqueeze(0)

    logits = planner(question_embedding)           # (1, 5)
    probs = F.softmax(logits, dim=-1).squeeze(0)   # (5,)

    top_idx = probs.argmax().item()
    top_conf = probs[top_idx].item()
    route = ROUTE_LABELS[top_idx]

    all_scores = {
        label: round(probs[i].item(), 4)
        for i, label in enumerate(ROUTE_LABELS)
    }

    # Check confidence threshold
    if use_fallback and top_conf < FALLBACK_CONFIDENCE_THRESHOLD:
        logger.info(
            "Planner confidence %.3f < threshold %.3f — using fallback",
            top_conf, FALLBACK_CONFIDENCE_THRESHOLD,
        )
        # We don't have the question text here, so return a special marker
        # The caller should call fallback_classify() with the text
        return PlannerResult(
            route=route,
            confidence=top_conf,
            source="_needs_fallback",
            all_scores=all_scores,
        )

    logger.debug(
        "Planner: route=%s confidence=%.3f scores=%s",
        route, top_conf, all_scores,
    )
    return PlannerResult(
        route=route,
        confidence=top_conf,
        source="learned",
        all_scores=all_scores,
    )


# ══════════════════════════════════════════════════════════════
# Rule-Based Fallback
# ══════════════════════════════════════════════════════════════

# Keyword patterns per route (compiled once)
_ROUTE_PATTERNS: dict[str, re.Pattern] = {
    "concept":   re.compile(
        r"\b(why|explain|what\s+is|what\s+are|define|meaning|concept|theory|principle)\b",
        re.IGNORECASE,
    ),
    "procedure": re.compile(
        r"\b(how\s+to|how\s+do|steps|process|method|procedure|algorithm|implement|solve)\b",
        re.IGNORECASE,
    ),
    "temporal":  re.compile(
        r"\b(when|before|after|first|then|next|order|sequence|previous|following|during)\b",
        re.IGNORECASE,
    ),
    "visual":    re.compile(
        r"\b(shown|visible|diagram|graph|chart|image|figure|slide|board|draw|display|screen)\b",
        re.IGNORECASE,
    ),
    "summary":   re.compile(
        r"\b(summarize|summary|overview|main\s+points|key\s+takeaways|recap|outline|overall)\b",
        re.IGNORECASE,
    ),
}


def fallback_classify(question: str) -> PlannerResult:
    """
    Rule-based question classification as a fallback.

    Counts keyword matches per route and returns the highest-scoring one.
    Returns "concept" as default if no keywords match.

    Args:
        question: Raw question text

    Returns:
        PlannerResult with source="fallback"
    """
    scores: dict[str, int] = {}
    for route_name, pattern in _ROUTE_PATTERNS.items():
        matches = pattern.findall(question)
        scores[route_name] = len(matches)

    total = sum(scores.values())
    if total == 0:
        # No keyword matches — default to concept
        return PlannerResult(
            route="concept",
            confidence=0.2,
            source="fallback",
            all_scores={r: 0.2 for r in ROUTE_LABELS},
        )

    # Normalize scores
    all_scores = {r: round(scores[r] / total, 4) for r in ROUTE_LABELS}
    best_route = max(scores, key=scores.get)  # type: ignore

    logger.debug(
        "Fallback planner: route=%s scores=%s",
        best_route, all_scores,
    )
    return PlannerResult(
        route=best_route,
        confidence=all_scores[best_route],
        source="fallback",
        all_scores=all_scores,
    )


def classify_with_fallback(
    planner: Optional[EduPlanner],
    question_embedding: Optional[torch.Tensor],
    question_text: str,
) -> PlannerResult:
    """
    Convenience function: try learned planner, fall back to rules.

    This is the main entry point for route classification.

    Args:
        planner:            Trained EduPlanner (None if not loaded)
        question_embedding: (768,) DistilBERT embedding (None if planner is None)
        question_text:      Raw question string (for fallback)

    Returns:
        PlannerResult with appropriate source
    """
    # If no planner is loaded, go straight to fallback
    if planner is None or question_embedding is None:
        logger.info("No learned planner available — using fallback")
        return fallback_classify(question_text)

    result = classify_question(planner, question_embedding, use_fallback=True)

    # If learned planner was low-confidence, use fallback
    if result.source == "_needs_fallback":
        fb = fallback_classify(question_text)
        # Keep learned scores in the result for logging
        fb.all_scores = {
            f"learned_{k}": v for k, v in result.all_scores.items()
        } | fb.all_scores
        return fb

    return result
