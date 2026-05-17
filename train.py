"""
train.py
────────────────────────────────────────────────────────────
VQA Guider — End-to-End Generative Training Script.

Fixes over the original notebook:
  1. Question-only encoding (no MCQ options) — matches inference exactly
  2. Full dataset (no head(200) / Subset(100) limits)
  3. TemporalAttentionPooling included in training graph
  4. Task routing entropy regularization to prevent collapse
  5. Single-phase generative training through Phi-2 LM loss

Trainable:  VQAGuiderCore, LLMProjector, TemporalAttentionPooling
Frozen:     CLIP, DistilBERT, Phi-2

Usage (Google Colab):
    !python train.py \\
        --csv_path /content/drive/MyDrive/dataset/train.csv \\
        --video_root /content/drive/MyDrive/dataset/Training \\
        --save_path ./models/vqa_model_generative.pt \\
        --epochs 20 --batch_size 2 --lr 1e-4

Usage (local):
    python train.py \\
        --csv_path ./data/train.csv \\
        --video_root ./data/Training \\
        --save_path ./models/vqa_model_generative.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="VQA Guider Generative Training")
    # Paths
    p.add_argument("--csv_path", type=str, required=True, help="Path to NExT-QA train.csv")
    p.add_argument("--video_root", type=str, required=True, help="Root directory of video files")
    p.add_argument("--frame_cache_path", type=str, default=None,
                   help="Path to per-frame CLIP embeddings cache (.pt). Created if missing.")
    p.add_argument("--save_path", type=str, default="./models/vqa_model_generative.pt")
    # Hyperparameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4, help="LR for VQAGuiderCore + TemporalAttentionPooling")
    p.add_argument("--proj_lr", type=float, default=5e-5, help="LR for LLMProjector")
    p.add_argument("--max_seq_len", type=int, default=64)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--val_every", type=int, default=2, help="Validate every N epochs")
    p.add_argument("--entropy_weight", type=float, default=0.1, help="Weight for task routing entropy loss")
    p.add_argument("--max_samples", type=int, default=0, help="Limit samples (0 = use all)")
    return p.parse_args()

# ══════════════════════════════════════════════════════════════
# Model Architectures (identical to backend/models/architectures.py)
# ══════════════════════════════════════════════════════════════

class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        proj_x = self.proj(x)
        attn_logits = self.attn_fc(proj_x)
        weights = torch.softmax(attn_logits, dim=0).squeeze(-1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=0)
        return pooled, weights


class VQAGuiderCore(nn.Module):
    def __init__(self, video_dim=512, question_dim=768, hidden_dim=512, num_tasks=3):
        super().__init__()
        self.task_planner = nn.Sequential(
            nn.Linear(question_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
        )
        joint_dim = video_dim + question_dim
        self.action_head = nn.Sequential(nn.Linear(joint_dim, 256), nn.ReLU(), nn.LayerNorm(256))
        self.tracking_head = nn.Sequential(nn.Linear(joint_dim, 256), nn.ReLU(), nn.LayerNorm(256))
        self.scene_head = nn.Sequential(nn.Linear(joint_dim, 256), nn.ReLU(), nn.LayerNorm(256))
        self.pre_fusion = nn.Sequential(
            nn.Linear(256 * num_tasks + question_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.planning_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.output_layer = nn.Linear(hidden_dim, 512)

    def forward(self, video_feat, question_feat):
        task_logits = self.task_planner(question_feat)
        task_probs = torch.sigmoid(task_logits)
        joint = torch.cat([video_feat, question_feat], dim=-1)
        action_out = self.action_head(joint)
        tracking_out = self.tracking_head(joint)
        scene_out = self.scene_head(joint)
        tool_outputs = torch.stack([action_out, tracking_out, scene_out], dim=1)
        weighted_tools = tool_outputs * task_probs.unsqueeze(-1)
        weighted_tools = weighted_tools.view(video_feat.size(0), -1)
        combined = torch.cat([weighted_tools, question_feat], dim=-1)
        fused = self.pre_fusion(combined)
        refined = self.planning_refine(fused)
        fusion_vec = self.output_layer(refined)
        return fusion_vec, task_probs


class LLMProjector(nn.Module):
    def __init__(self, input_dim=512, llm_dim=2560, num_tokens=10):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(input_dim, num_tokens * llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        B = x.size(0)
        out = self.proj(x)
        out = out.view(B, self.num_tokens, -1)
        return self.norm(out)


class OptionScorer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x):
        return self.fc(x).squeeze(-1)

# ══════════════════════════════════════════════════════════════
# Video Frame Sampling
# ══════════════════════════════════════════════════════════════

def sample_frames(video_path: str, num_frames: int = 16) -> list[np.ndarray]:
    import cv2
    # Suppress OpenCV warnings (like the swscaler invalid slice warnings)
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    cv2.setLogLevel(0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames: {video_path}")
    
    n = min(num_frames, total)
    indices = np.linspace(0, total - 1, n).astype(int).tolist()
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted: {video_path}")
    return frames

# ══════════════════════════════════════════════════════════════
# Per-Frame CLIP Cache Builder
# ══════════════════════════════════════════════════════════════

def build_frame_cache(video_map: dict, clip_model, preprocess, device, num_frames=16) -> dict:
    """
    Encode all videos into per-frame CLIP embeddings: {video_id: (N, 512) tensor}.
    These are NOT pooled yet — TemporalAttentionPooling is applied during training
    with gradients enabled.
    """
    cache = {}
    print(f"Building per-frame CLIP cache for {len(video_map)} videos...")
    for vid_id, vid_path in tqdm(video_map.items(), desc="CLIP encoding"):
        try:
            frames = sample_frames(vid_path, num_frames)
            images = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
            with torch.no_grad():
                embs = clip_model.encode_image(images).float()
                embs = embs / embs.norm(dim=-1, keepdim=True)
            cache[vid_id] = embs.cpu()  # (N, 512)
        except Exception as e:
            print(f"  ⚠️ Skipping {vid_id}: {e}")
    print(f"Frame cache built: {len(cache)} videos")
    return cache

# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

# ── Question type → Task routing label mapping ──
# Maps NExT-QA question types to task indices for direct supervision
TASK_TYPE_MAP = {
    "CW": 0,  # Causal Why     → Action
    "CH": 0,  # Causal How     → Action
    "TN": 1,  # Temporal Next  → Tracking
    "TP": 1,  # Temporal Prev  → Tracking
    "TC": 1,  # Temporal Conc  → Tracking
    "DC": 2,  # Descriptive Count    → Scene
    "DL": 2,  # Descriptive Location → Scene
    "DO": 2,  # Descriptive Other    → Scene
}

class NExTQAGenerativeDataset(Dataset):
    """
    NExT-QA dataset for generative training.
    Returns: (video_id, question_text, answer_text, task_label)
    task_label: 0=Action, 1=Tracking, 2=Scene
    """
    def __init__(self, csv_path: str, available_ids: set, max_samples: int = 0):
        full_data = pd.read_csv(csv_path)
        # Filter to available videos
        self.data = full_data[
            full_data["video"].astype(int).astype(str).isin(available_ids)
        ].reset_index(drop=True)
        if max_samples > 0:
            self.data = self.data.head(max_samples)
        print(f"Dataset: {len(self.data)} rows (from {len(full_data)} total, "
              f"{len(available_ids)} videos available)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = str(int(row["video"]))
        question = str(row["question"])
        answer_idx = int(row["answer"])
        answer_text = str(row[f"a{answer_idx}"])
        q_type = str(row.get("type", "DO"))
        task_label = TASK_TYPE_MAP.get(q_type, 2)  # Default to Scene
        return video_id, question, answer_text, task_label

# ══════════════════════════════════════════════════════════════
# Collate Function
# ══════════════════════════════════════════════════════════════

def make_collate_fn(frame_cache, q_tokenizer, q_model, phi2_tokenizer, device, max_seq_len, pooling):
    """
    Build a collate function that:
      1. Loads per-frame CLIP embeddings from cache
      2. Applies TemporalAttentionPooling WITH gradients
      3. Encodes question with DistilBERT (question-only, no options)
      4. Tokenizes target text for Phi-2 LM loss
    """
    def collate_fn(batch):
        video_ids, questions, answers, task_labels = zip(*batch)
        B = len(batch)

        # ── 1. Video features: per-frame → pooled (with gradients) ──
        video_feats = []
        for vid_id in video_ids:
            frame_embs = frame_cache[vid_id].to(device)  # (N, 512)
            pooled, _ = pooling(frame_embs)               # (512,) — WITH gradients
            video_feats.append(pooled)
        video_feat = torch.stack(video_feats)              # (B, 512)

        # ── 2. Question encoding: question-only (NO options) ──
        with torch.no_grad():
            q_inputs = q_tokenizer(
                list(questions), return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            ).to(device)
            q_outputs = q_model(**q_inputs)
            q_feat = q_outputs.last_hidden_state[:, 0, :]  # (B, 768) CLS

        # ── 3. Target text for Phi-2 LM loss ──
        target_texts = [
            f"You are a video understanding AI. "
            f"Watch the video carefully and answer the question.\n"
            f"Question: {q}\n"
            f"Detailed Answer: {a}."
            for q, a in zip(questions, answers)
        ]
        enc = phi2_tokenizer(
            target_texts, return_tensors="pt",
            truncation=True, max_length=max_seq_len, padding="max_length",
        )

        # ── 4. Task routing labels (direct supervision) ──
        task_labels_tensor = torch.tensor(list(task_labels), dtype=torch.long)

        return {
            "video_feat": video_feat,       # (B, 512) — has gradients via pooling
            "q_feat": q_feat,               # (B, 768) — detached (DistilBERT frozen)
            "input_ids": enc.input_ids,     # (B, T)
            "attn_mask": enc.attention_mask, # (B, T)
            "task_labels": task_labels_tensor, # (B,) — 0=Action, 1=Tracking, 2=Scene
        }

    return collate_fn

# ══════════════════════════════════════════════════════════════
# Training Step
# ══════════════════════════════════════════════════════════════

def train_step(batch, vqaguider, projector, pooling, phi2, phi2_tokenizer,
               optimizer, device, routing_weight):
    """
    Single training step with generative LM loss + direct task routing supervision.
    Gradient flow: loss → Phi-2 (frozen) → prefix → projector → fusion_vec → vqaguider
                   Also: routing_loss → task_planner (direct supervision)
                   And video_feat has gradients from pooling
    """
    video_feat = batch["video_feat"].to(device)   # (B, 512) — has grad from pooling
    q_feat     = batch["q_feat"].to(device)        # (B, 768)
    input_ids  = batch["input_ids"].to(device)     # (B, T)
    attn_mask  = batch["attn_mask"].to(device)     # (B, T)
    task_labels = batch["task_labels"].to(device)   # (B,) — 0/1/2
    B = video_feat.size(0)

    # ── Forward: VQAGuiderCore ──
    fusion_vec, task_probs = vqaguider(video_feat, q_feat)  # (B, 512), (B, 3)

    # ── Forward: LLMProjector → prefix tokens ──
    prefix = projector(fusion_vec).to(dtype=torch.float16)  # (B, 10, 2560)

    # ── Build Phi-2 inputs ──
    with torch.no_grad():
        token_embeds = phi2.get_input_embeddings()(input_ids)  # (B, T, 2560)
        token_embeds = token_embeds.to(dtype=torch.float16)

    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)  # (B, 10+T, 2560)

    prefix_mask = torch.ones((B, projector.num_tokens), dtype=attn_mask.dtype, device=device)
    attention_mask = torch.cat([prefix_mask, attn_mask], dim=1)

    # Labels: mask prefix tokens with -100
    prefix_labels = torch.full((B, projector.num_tokens), -100, dtype=input_ids.dtype, device=device)
    labels = torch.cat([prefix_labels, input_ids], dim=1)

    # ── Phi-2 forward (frozen, but gradients flow to inputs_embeds) ──
    outputs = phi2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    lm_loss = outputs.loss

    # ── Direct task routing supervision ──
    # Use the question type labels from NExT-QA to directly train the task planner
    # task_probs are sigmoid-activated, so we need logits for cross-entropy
    task_logits = vqaguider.task_planner(q_feat)  # (B, 3) — raw logits
    routing_loss = F.cross_entropy(task_logits, task_labels)

    total_loss = lm_loss + routing_weight * routing_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(vqaguider.parameters()) + list(projector.parameters()) + list(pooling.parameters()),
        max_norm=1.0,
    )
    optimizer.step()

    return lm_loss.item(), routing_loss.item()

# ══════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(val_loader, vqaguider, projector, pooling, phi2, phi2_tokenizer, device):
    vqaguider.eval()
    projector.eval()
    pooling.eval()

    total_loss = 0.0
    count = 0

    for batch in val_loader:
        video_feat = batch["video_feat"].to(device)
        q_feat     = batch["q_feat"].to(device)
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attn_mask"].to(device)
        B = video_feat.size(0)

        fusion_vec, _ = vqaguider(video_feat, q_feat)
        prefix = projector(fusion_vec).to(dtype=torch.float16)

        token_embeds = phi2.get_input_embeddings()(input_ids).to(dtype=torch.float16)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        prefix_mask = torch.ones((B, projector.num_tokens), dtype=attn_mask.dtype, device=device)
        attention_mask = torch.cat([prefix_mask, attn_mask], dim=1)

        prefix_labels = torch.full((B, projector.num_tokens), -100, dtype=input_ids.dtype, device=device)
        labels = torch.cat([prefix_labels, input_ids], dim=1)

        outputs = phi2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        count += 1

    vqaguider.train()
    projector.train()
    pooling.train()

    return total_loss / max(count, 1)

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  VQA Guider — Generative Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.csv_path}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("-" * 60)

    # ── 1. Load CLIP ──
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    print("✅ CLIP loaded (frozen)")

    # ── 2. Load DistilBERT ──
    from transformers import DistilBertModel, DistilBertTokenizer
    q_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    q_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    q_model.eval()
    for p in q_model.parameters():
        p.requires_grad = False
    print("✅ DistilBERT loaded (frozen)")

    # ── 3. Load Phi-2 ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    phi2_name = "microsoft/phi-2"
    phi2_tokenizer = AutoTokenizer.from_pretrained(phi2_name, trust_remote_code=True)
    phi2_tokenizer.pad_token = phi2_tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    phi2 = AutoModelForCausalLM.from_pretrained(
        phi2_name, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)
    phi2.eval()
    for p in phi2.parameters():
        p.requires_grad = False
    print("✅ Phi-2 loaded (frozen)")

    # ── 4. Initialize trainable models ──
    pooling = TemporalAttentionPooling(dim=512).to(device)
    vqaguider = VQAGuiderCore().to(device)
    projector = LLMProjector(512, 2560).to(device)
    scorer = OptionScorer(512).to(device)  # Kept for checkpoint compatibility

    pooling.train()
    vqaguider.train()
    projector.train()
    scorer.eval()  # Not used during generative training

    total_params = (
        sum(p.numel() for p in vqaguider.parameters()) +
        sum(p.numel() for p in projector.parameters()) +
        sum(p.numel() for p in pooling.parameters())
    )
    print(f"✅ Trainable parameters: {total_params:,}")

    # ── 5. Build video index + frame cache ──
    def build_video_index(video_root):
        vmap = {}
        for folder in os.listdir(video_root):
            folder_path = os.path.join(video_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for f in os.listdir(folder_path):
                if f.endswith((".mp4", ".avi", ".webm", ".mov")):
                    vid_id = os.path.splitext(f)[0]
                    vmap[vid_id] = os.path.join(folder_path, f)
        return vmap

    video_map = build_video_index(args.video_root)
    print(f"Found {len(video_map)} videos in {args.video_root}")

    # Build or load per-frame CLIP cache
    cache_path = args.frame_cache_path or os.path.join(
        os.path.dirname(args.save_path), "frame_features.pt"
    )
    if os.path.exists(cache_path):
        print(f"Loading frame cache from {cache_path}...")
        frame_cache = torch.load(cache_path, map_location="cpu")
        print(f"Frame cache loaded: {len(frame_cache)} videos")
    else:
        frame_cache = build_frame_cache(
            video_map, clip_model, clip_preprocess, device, args.num_frames,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(frame_cache, cache_path)
        print(f"Frame cache saved to {cache_path}")

    # ── 6. Create dataset + splits ──
    available_ids = set(frame_cache.keys())
    dataset = NExTQAGenerativeDataset(args.csv_path, available_ids, args.max_samples)

    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size} | Val: {val_size}")

    collate = make_collate_fn(
        frame_cache, q_tokenizer, q_model, phi2_tokenizer, device, args.max_seq_len, pooling,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ── 7. Optimizer ──
    optimizer = torch.optim.Adam([
        {"params": vqaguider.parameters(), "lr": args.lr},
        {"params": projector.parameters(), "lr": args.proj_lr},
        {"params": pooling.parameters(), "lr": args.lr},
    ], weight_decay=1e-5)

    # ── 8. Training loop ──
    best_val_loss = float("inf")
    print("=" * 60)
    print("  Training Start")
    print("=" * 60)

    for epoch in range(args.epochs):
        vqaguider.train()
        projector.train()
        pooling.train()

        epoch_lm_loss = 0.0
        epoch_entropy = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            lm_loss, routing_loss = train_step(
                batch, vqaguider, projector, pooling, phi2, phi2_tokenizer,
                optimizer, device, args.entropy_weight,
            )
            epoch_lm_loss += lm_loss
            epoch_entropy += routing_loss
            n_batches += 1
            pbar.set_postfix(lm_loss=f"{lm_loss:.4f}", routing=f"{routing_loss:.3f}")

        avg_lm = epoch_lm_loss / max(n_batches, 1)
        avg_route = epoch_entropy / max(n_batches, 1)
        print(f"Epoch {epoch+1:3d}/{args.epochs} │ LM Loss: {avg_lm:.4f} │ Routing Loss: {avg_route:.3f}")

        # ── Validation ──
        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs:
            val_loss = validate(val_loader, vqaguider, projector, pooling, phi2, phi2_tokenizer, device)
            print(f"          ↳ Val Loss: {val_loss:.4f}", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "vqaguider": vqaguider.state_dict(),
                    "projector": projector.state_dict(),
                    "scorer": scorer.state_dict(),
                    "temporal_pooling": pooling.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                }
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                torch.save(checkpoint, args.save_path)
                print(f" ✅ Saved → {args.save_path}")
            else:
                print()

    print("-" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
