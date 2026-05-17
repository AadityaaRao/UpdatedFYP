"""
train_planner.py
────────────────────────────────────────────────────────────
Edu-VQAGuider Planner Training Script.

Trains the lightweight question-intent classifier (198K params)
on a labeled CSV of educational questions.

No heavy GPU training — this runs in minutes on CPU.

Pipeline:
    1. Load CSV (question, route columns)
    2. Encode all questions with DistilBERT (frozen, CLS token)
    3. Split into train/val (80/20 stratified)
    4. Train EduPlanner classifier
    5. Evaluate (accuracy, per-class metrics, confusion matrix)
    6. Save best checkpoint

Usage:
    python train_planner.py --csv data/planner_training/seed_questions.csv
    python train_planner.py --csv data/planner_training/full_dataset.csv --epochs 30

CSV format:
    question,route,source
    "Why does entropy increase?",concept,seed
    "What are the steps?",procedure,manual
    ...

The 'source' column is optional (for tracking where questions came from).
Only 'question' and 'route' are required.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.edu.planner import EduPlanner, ROUTE_LABELS, ROUTE_TO_IDX, NUM_ROUTES


# ══════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train Edu-VQAGuider Planner")
    p.add_argument(
        "--csv", type=str, required=True,
        help="Path to labeled questions CSV (columns: question, route)",
    )
    p.add_argument(
        "--save_path", type=str,
        default="./models/edu_planner.pt",
        help="Where to save the trained checkpoint",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument(
        "--device", type=str, default="auto",
        help="'auto', 'cuda', or 'cpu'",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════

def load_dataset(csv_path: str) -> tuple[list[str], list[int]]:
    """
    Load and validate the training CSV.

    Returns:
        questions: list of question strings
        labels:    list of integer route labels (0-4)
    """
    df = pd.read_csv(csv_path)

    # Validate columns
    if "question" not in df.columns or "route" not in df.columns:
        raise ValueError(
            f"CSV must have 'question' and 'route' columns. "
            f"Found: {list(df.columns)}"
        )

    # Clean
    df = df.dropna(subset=["question", "route"])
    df["question"] = df["question"].str.strip()
    df["route"] = df["route"].str.strip().str.lower()

    # Validate routes
    invalid = df[~df["route"].isin(ROUTE_LABELS)]
    if len(invalid) > 0:
        print(f"WARNING: {len(invalid)} rows have invalid routes:")
        print(invalid[["question", "route"]].head(5))
        df = df[df["route"].isin(ROUTE_LABELS)]

    questions = df["question"].tolist()
    labels = [ROUTE_TO_IDX[r] for r in df["route"]]

    # Print distribution
    print(f"\nDataset: {len(questions)} questions")
    print("Route distribution:")
    for route in ROUTE_LABELS:
        count = sum(1 for r in df["route"] if r == route)
        bar = "#" * count
        print(f"  {route:12s}: {count:3d} {bar}")

    return questions, labels


# ══════════════════════════════════════════════════════════════
# DistilBERT Encoding
# ══════════════════════════════════════════════════════════════

def encode_questions(
    questions: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Encode all questions with DistilBERT (frozen).
    Returns (N, 768) tensor of CLS embeddings.
    """
    from transformers import DistilBertModel, DistilBertTokenizer

    print("\nLoading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    model.eval()

    print(f"Encoding {len(questions)} questions...")
    all_embeddings = []

    # Process in batches to avoid OOM
    batch_size = 32
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            all_embeddings.append(cls_embeddings.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, 768)
    print(f"Encoded: {embeddings.shape}")

    # Free DistilBERT from memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def train(
    planner: EduPlanner,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: str,
) -> dict:
    """
    Train the EduPlanner and save the best checkpoint.

    Returns:
        dict with training stats
    """
    planner.to(device)
    planner.train()

    optimizer = torch.optim.Adam(planner.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5,
    )

    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  Training EduPlanner")
    print(f"  Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────
        planner.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = planner(embeddings)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # ── Validate ──────────────────────────────────────────
        planner.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)

                logits = planner(embeddings)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "planner_state_dict": planner.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "route_labels": ROUTE_LABELS,
                },
                save_path,
            )
            marker = " << saved"

        # Print every 5 epochs or on save
        if (epoch + 1) % 5 == 0 or marker:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"train_acc={train_acc:.3f} | "
                f"val_acc={val_acc:.3f}{marker}"
            )

    print(f"\nBest val accuracy: {best_val_acc:.3f} (epoch {best_epoch})")
    print(f"Checkpoint saved: {save_path}")

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }


# ══════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate(
    planner: EduPlanner,
    val_embeddings: torch.Tensor,
    val_labels: list[int],
    device: torch.device,
):
    """
    Full evaluation: accuracy, per-class metrics, confusion matrix.
    """
    planner.to(device)
    planner.eval()

    with torch.no_grad():
        logits = planner(val_embeddings.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()

    y_true = np.array(val_labels)
    y_pred = preds

    print(f"\n{'='*60}")
    print("  Evaluation Results")
    print(f"{'='*60}\n")

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.3f} ({sum(y_true == y_pred)}/{len(y_true)})\n")

    # Per-class report
    print("Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=ROUTE_LABELS,
        digits=3,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"{'':>12s}", end="")
    for label in ROUTE_LABELS:
        print(f" {label[:6]:>6s}", end="")
    print()

    for i, label in enumerate(ROUTE_LABELS):
        print(f"  {label:>10s}", end="")
        for j in range(len(ROUTE_LABELS)):
            print(f" {cm[i][j]:6d}", end="")
        print()

    return acc


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Dataset: {args.csv}")

    # Step 1: Load dataset
    questions, labels = load_dataset(args.csv)

    if len(questions) < 10:
        print(f"\nERROR: Need at least 10 questions, got {len(questions)}.")
        print("Add more questions to your CSV and re-run.")
        sys.exit(1)

    # Step 2: Encode with DistilBERT
    embeddings = encode_questions(questions, device)

    # Step 3: Train/val split (stratified)
    train_emb, val_emb, train_labels, val_labels = train_test_split(
        embeddings.numpy(),
        labels,
        test_size=args.val_split,
        stratify=labels,
        random_state=42,
    )

    print(f"\nSplit: {len(train_labels)} train / {len(val_labels)} val")

    # Convert to tensors
    train_emb = torch.tensor(train_emb, dtype=torch.float32)
    val_emb = torch.tensor(val_emb, dtype=torch.float32)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_emb, train_labels_t)
    val_dataset = TensorDataset(val_emb, val_labels_t)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
    )

    # Step 4: Create planner
    planner = EduPlanner()
    total_params = sum(p.numel() for p in planner.parameters())
    print(f"EduPlanner parameters: {total_params:,}")

    # Step 5: Train
    t0 = time.time()
    stats = train(
        planner=planner,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_path,
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")

    # Step 6: Load best checkpoint and evaluate
    print("\nLoading best checkpoint for final evaluation...")
    ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
    planner.load_state_dict(ckpt["planner_state_dict"])

    acc = evaluate(planner, val_emb, val_labels, device)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Dataset:     {len(questions)} questions")
    print(f"  Train/Val:   {len(train_labels)} / {len(val_labels)}")
    print(f"  Best epoch:  {stats['best_epoch']}")
    print(f"  Val accuracy: {acc:.3f}")
    print(f"  Checkpoint:  {args.save_path}")
    print(f"  Time:        {elapsed:.1f}s")

    if acc < 0.8:
        print(f"\n  WARNING: Accuracy {acc:.3f} < 0.80 target.")
        print(f"  Consider adding more labeled questions (currently {len(questions)}).")
    else:
        print(f"\n  Target accuracy (>0.80) ACHIEVED.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
