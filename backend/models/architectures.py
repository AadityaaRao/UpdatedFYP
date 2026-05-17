"""
backend/models/architectures.py
────────────────────────────────────────────────────────────
All PyTorch model class definitions — exactly matching the
training pipeline. No training code here; pure architecture.
Components:
  • TemporalAttentionPooling  — pools frame embeddings → single vector
  • VQAGuiderCore             — task planner + tool heads + fusion
  • LLMProjector              — projects fusion vec → Phi-2 prefix tokens
  • OptionScorer              — scores each MCQ option
  • VideoEncoder              — CLIP frames → (512,) via attention pooling
  • QuestionEncoder           — DistilBERT → (768,) CLS embedding
"""
import torch
import torch.nn as nn
# ── 1. Temporal Attention Pooling ─────────────────────────────
class TemporalAttentionPooling(nn.Module):
    """
    Learnable attention over N frame embeddings → single (dim,) vector.
    Matches training: same projection + softmax attention.
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 4, 1),
        )
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (N, dim) — N frame embeddings
        Returns:
            pooled: (dim,)
            weights: (N,)
        """
        x = x.float()
        proj_x = self.proj(x)
        attn_logits = self.attn_fc(proj_x)
        weights = torch.softmax(attn_logits, dim=0).squeeze(-1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=0)
        return pooled, weights
# ── 2. VQAGuiderCore ──────────────────────────────────────────
class VQAGuiderCore(nn.Module):
    """
    Task planner + question-aware tool heads + fusion.
    Architecture is UNCHANGED from training pipeline.
    Inputs:
        video_feat    : (B, 512)
        question_feat : (B, 768)
    Outputs:
        fusion_vec  : (B, 512)
        task_probs  : (B, num_tasks)  — sigmoid-activated
    """
    def __init__(
        self,
        video_dim: int = 512,
        question_dim: int = 768,
        hidden_dim: int = 512,
        num_tasks: int = 3,
    ):
        super().__init__()
        # Task planner — question alone drives routing decision
        self.task_planner = nn.Sequential(
            nn.Linear(question_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
        )
        joint_dim = video_dim + question_dim  # 1280
        # Question-aware tool heads
        self.action_head = nn.Sequential(
            nn.Linear(joint_dim, 256), nn.ReLU(), nn.LayerNorm(256)
        )
        self.tracking_head = nn.Sequential(
            nn.Linear(joint_dim, 256), nn.ReLU(), nn.LayerNorm(256)
        )
        self.scene_head = nn.Sequential(
            nn.Linear(joint_dim, 256), nn.ReLU(), nn.LayerNorm(256)
        )
        # Fusion
        self.pre_fusion = nn.Sequential(
            nn.Linear(256 * num_tasks + question_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        # Planning refinement
        self.planning_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, 512)
    def forward(self, video_feat: torch.Tensor, question_feat: torch.Tensor):
        task_logits = self.task_planner(question_feat)
        task_probs = torch.sigmoid(task_logits)
        joint = torch.cat([video_feat, question_feat], dim=-1)
        action_out = self.action_head(joint)
        tracking_out = self.tracking_head(joint)
        scene_out = self.scene_head(joint)
        tool_outputs = torch.stack(
            [action_out, tracking_out, scene_out], dim=1
        )  # (B, 3, 256)
        weighted_tools = tool_outputs * task_probs.unsqueeze(-1)
        weighted_tools = weighted_tools.view(video_feat.size(0), -1)  # (B, 768)
        combined = torch.cat([weighted_tools, question_feat], dim=-1)
        fused = self.pre_fusion(combined)
        refined = self.planning_refine(fused)
        fusion_vec = self.output_layer(refined)
        return fusion_vec, task_probs
# ── 3. LLM Projector ──────────────────────────────────────────
class LLMProjector(nn.Module):
    """
    Projects fusion vector → num_tokens prefix tokens for Phi-2.
    Matches training Phase 2 architecture exactly.
    Input:  (B, 512)
    Output: (B, num_tokens, llm_dim)  — normalized
    """
    def __init__(
        self,
        input_dim: int = 512,
        llm_dim: int = 2560,
        num_tokens: int = 10,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(input_dim, num_tokens * llm_dim)
        self.norm = nn.LayerNorm(llm_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = self.proj(x)                       # (B, num_tokens * llm_dim)
        out = out.view(B, self.num_tokens, -1)   # (B, num_tokens, llm_dim)
        return self.norm(out)
# ── 4. Option Scorer ──────────────────────────────────────────
class OptionScorer(nn.Module):
    """
    Scores a single answer option given a fusion vector.
    Used at inference for MCQ ranking (pick highest-scored option).
    Input:  (B, 512)
    Output: (B,)
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (B,)