"""
evaluate.py
Evaluate the trained model and display a confusion matrix.

The model outputs a continuous score in [-1, 1].
We convert it to 3 classes:
  score >  0.3  → predicted White wins
  score < -0.3  → predicted Black wins
  otherwise     → predicted Draw

Usage:
  python evaluate.py --model models/chess_eval.pt --csv data/games.csv
"""

import argparse
import random

import chess
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from board_encoder import board_to_tensor
from model import load_model


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

WINNER_TO_LABEL = {"white": 1.0, "black": -1.0, "draw": 0.0}
CLASSES = ["Black wins", "Draw", "White wins"]

# Threshold for converting continuous score → class
THRESHOLD = 0.3


def score_to_class(score: float) -> int:
    """Convert continuous model score to class index (0=Black, 1=Draw, 2=White)."""
    if score > THRESHOLD:
        return 2   # White wins
    elif score < -THRESHOLD:
        return 0   # Black wins
    else:
        return 1   # Draw


def label_to_class(label: float) -> int:
    """Convert ground truth label to class index."""
    if label > 0.5:
        return 2
    elif label < -0.5:
        return 0
    else:
        return 1


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_eval_data(csv_path: str, max_games: int = 1000, positions_per_game: int = 4):
    """Load positions from CSV for evaluation."""
    df = pd.read_csv(csv_path)
    df = df[df["winner"].isin(WINNER_TO_LABEL.keys())]
    df = df.sample(frac=1, random_state=42).head(max_games)  # shuffle + cap

    tensors, labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading positions"):
        label = WINNER_TO_LABEL[row["winner"]]
        move_list = str(row["moves"]).strip().split()

        if len(move_list) < 10:
            continue

        board = chess.Board()
        positions = []

        for san in move_list:
            try:
                move = board.parse_san(san)
                board.push(move)
                if board.fullmove_number > 5:
                    positions.append(board.copy())
            except ValueError:
                break

        if not positions:
            continue

        sampled = random.sample(positions, min(positions_per_game, len(positions)))
        for pos in sampled:
            tensors.append(board_to_tensor(pos))
            labels.append(label)

    X = torch.stack(tensors)
    y = torch.tensor(labels, dtype=torch.float32)
    print(f"Loaded {len(X):,} positions for evaluation.")
    return X, y


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model, device=str(device))
    print(f"Loaded model from {args.model}")

    # Load data
    X, y = load_eval_data(args.csv, max_games=args.max_games)

    # Run inference in batches
    all_scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = X[i:i+256].to(device)
            scores = model(batch).cpu().numpy()
            all_scores.extend(scores.tolist())

    # Convert to class predictions
    y_true = [label_to_class(l.item()) for l in y]
    y_pred = [score_to_class(s) for s in all_scores]

    # ── Confusion matrix ──
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chess AI — Model Evaluation", fontsize=14, fontweight="bold", y=1.01)

    # Left: raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[0],
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Actual", fontsize=11)
    axes[0].set_title("Confusion Matrix (counts)")

    # Right: normalised percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[1],
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "format": ticker.FuncFormatter(lambda x, _: f"{x:.0f}%")},
    )
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("Actual", fontsize=11)
    axes[1].set_title("Confusion Matrix (% per actual class)")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("\nSaved → confusion_matrix.png")
    plt.show()

    # ── Classification report ──
    print("\nClassification Report:")
    print("─" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    # ── Score distribution ──
    fig2, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c", "#95a5a6", "#3498db"]
    class_names = ["Black wins (actual)", "Draw (actual)", "White wins (actual)"]

    for cls_idx, (color, name) in enumerate(zip(colors, class_names)):
        cls_scores = [s for s, t in zip(all_scores, y_true) if t == cls_idx]
        if cls_scores:
            ax.hist(cls_scores, bins=40, alpha=0.6, color=color, label=name)

    ax.axvline(x=THRESHOLD,  color="black", linestyle="--", linewidth=1, label=f"Threshold ±{THRESHOLD}")
    ax.axvline(x=-THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Model score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Score distribution by actual outcome")
    ax.legend()
    plt.tight_layout()
    plt.savefig("score_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved → score_distribution.png")
    plt.show()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate chess model with confusion matrix")
    p.add_argument("--model",     default="models/chess_eval.pt", help="Path to .pt model")
    p.add_argument("--csv",       required=True,                   help="Path to Kaggle CSV")
    p.add_argument("--max-games", type=int, default=1000,          help="Games to evaluate on")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
