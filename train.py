"""
train.py
Train ChessEvaluator on positions from a PGN file or Kaggle CSV.

Kaggle CSV usage:
  python train.py --csv data/games.csv --epochs 20

PGN usage:
  python train.py --pgn data/games.pgn --epochs 20

Kaggle dataset: https://www.kaggle.com/datasets/datasnaek/chess
The CSV has columns: id, rated, turns, victory_status, winner, moves, ...
"""

import argparse
import random
from pathlib import Path

import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from board_encoder import board_to_tensor
from model import ChessEvaluator


# ──────────────────────────────────────────────
# Data extraction — CSV (Kaggle)
# ──────────────────────────────────────────────

WINNER_TO_LABEL = {"white": 1.0, "black": -1.0, "draw": 0.0}


def extract_positions_csv(csv_path: str, max_games: int = 5000, positions_per_game: int = 8):
    """
    Read the Kaggle chess CSV and extract (tensor, label) pairs.

    The CSV contains a 'moves' column with space-separated UCI moves
    and a 'winner' column with 'white', 'black', or 'draw'.
    """
    import pandas as pd

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    df = df[df["winner"].isin(WINNER_TO_LABEL.keys())]
    df = df.head(max_games)

    print(f"Processing {len(df):,} games from CSV...")
    tensors, labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing games"):
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

    if not tensors:
        raise ValueError("No positions extracted — check your CSV file path.")

    X = torch.stack(tensors)
    y = torch.tensor(labels, dtype=torch.float32)
    print(f"Extracted {len(X):,} positions from CSV.")
    return X, y


# ──────────────────────────────────────────────
# Data extraction — PGN
# ──────────────────────────────────────────────

RESULT_TO_LABEL = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def extract_positions_pgn(pgn_path: str, max_games: int = 5000, positions_per_game: int = 8):
    """Parse a PGN file and return (tensors, labels)."""
    tensors, labels = [], []

    with open(pgn_path) as f:
        for _ in tqdm(range(max_games), desc="Parsing games"):
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            label = RESULT_TO_LABEL.get(result)
            if label is None:
                continue

            board = game.board()
            positions = []
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number > 5:
                    positions.append(board.copy())

            if not positions:
                continue

            sampled = random.sample(positions, min(positions_per_game, len(positions)))
            for pos in sampled:
                tensors.append(board_to_tensor(pos))
                labels.append(label)

    if not tensors:
        raise ValueError("No positions extracted — check your PGN file path.")

    X = torch.stack(tensors)
    y = torch.tensor(labels, dtype=torch.float32)
    print(f"Extracted {len(X):,} positions from PGN.")
    return X, y


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if args.csv:
        X, y = extract_positions_csv(args.csv, max_games=args.max_games)
    else:
        X, y = extract_positions_pgn(args.pgn, max_games=args.max_games)

    dataset = TensorDataset(X, y)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = ChessEvaluator(num_res_blocks=args.res_blocks).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        train_loss = total_loss / train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
        val_loss /= val_size

        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train chess position evaluator")
    p.add_argument("--csv",        default=None,  help="Path to Kaggle CSV file")
    p.add_argument("--pgn",        default=None,  help="Path to PGN file")
    p.add_argument("--save",       default="models/chess_eval.pt")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--max-games",  type=int,   default=5000)
    p.add_argument("--res-blocks", type=int,   default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.csv and not args.pgn:
        print("Error: provide either --csv or --pgn")
    else:
        train(args)