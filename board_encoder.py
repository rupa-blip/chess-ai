"""
board_encoder.py
Converts a chess.Board into a (12, 8, 8) float32 tensor.

Plane layout (index → piece):
  0-5  : White  P N B R Q K
  6-11 : Black  P N B R Q K
"""

import chess
import numpy as np
import torch

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode a chess.Board as a (12, 8, 8) float32 tensor.
    The board is always oriented from White's perspective (rank 0 = rank 1).
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        row = chess.square_rank(square)   # 0-7, rank 1-8
        col = chess.square_file(square)   # 0-7, file a-h
        planes[plane, row, col] = 1.0

    return torch.from_numpy(planes)


def batch_boards_to_tensor(boards: list[chess.Board]) -> torch.Tensor:
    """Stack multiple boards into a (N, 12, 8, 8) tensor for batched inference."""
    return torch.stack([board_to_tensor(b) for b in boards])


if __name__ == "__main__":
    board = chess.Board()
    t = board_to_tensor(board)
    print(f"Tensor shape: {t.shape}")          # (12, 8, 8)
    print(f"White pawns plane:\n{t[0]}")        # Row 1 should be all ones
