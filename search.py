"""
search.py
Alpha-beta minimax search using the neural network as a leaf evaluator.

The AI searches the game tree to a given depth.
At leaf nodes it calls ChessEvaluator instead of a hand-coded heuristic.
Alpha-beta pruning cuts branches that cannot affect the final decision,
making the search ~10× faster than plain minimax.

Positive scores favour White; negative scores favour Black.
"""

import chess
import torch

from board_encoder import board_to_tensor
from model import ChessEvaluator

# Small material fallback used when model is None (for testing without weights)
PIECE_VALUE = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}


def _material_score(board: chess.Board) -> float:
    """Simple material count — fallback evaluator."""
    score = 0
    for piece in board.piece_map().values():
        val = PIECE_VALUE[piece.piece_type]
        score += val if piece.color == chess.WHITE else -val
    return score / 39.0  # normalise to approx [-1, 1]


def _nn_score(board: chess.Board, model: ChessEvaluator, device: torch.device) -> float:
    """Neural-network leaf evaluation."""
    t = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).item()


class AlphaBetaEngine:
    """
    Chess engine backed by alpha-beta search + neural network evaluation.

    Args:
        model:      Trained ChessEvaluator (or None to use material fallback)
        depth:      Search depth in half-moves (plies). 3–4 is playable, 5+ is strong.
        device:     Torch device for model inference
    """

    def __init__(self, model: ChessEvaluator | None = None, depth: int = 3, device: str = "cpu"):
        self.model = model
        self.depth = depth
        self.device = torch.device(device)
        if model is not None:
            self.model.to(self.device)
            self.model.eval()

    def evaluate(self, board: chess.Board) -> float:
        """Score the current position from White's perspective."""
        if self.model is not None:
            return _nn_score(board, self.model, self.device)
        return _material_score(board)

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximising: bool,
    ) -> float:
        """Recursive alpha-beta search. Returns score from White's perspective."""

        # Terminal / horizon checks
        if board.is_checkmate():
            return -10.0 if maximising else 10.0   # current side lost
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if depth == 0:
            return self.evaluate(board)

        if maximising:
            value = -float("inf")
            for move in board.legal_moves:
                board.push(move)
                value = max(value, self._alpha_beta(board, depth - 1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # β cut-off
            return value
        else:
            value = float("inf")
            for move in board.legal_moves:
                board.push(move)
                value = min(value, self._alpha_beta(board, depth - 1, alpha, beta, True))
                board.pop()
                beta = min(beta, value)
                if beta <= alpha:
                    break  # α cut-off
            return value

    def best_move(self, board: chess.Board) -> chess.Move | None:
        """
        Return the best move from the current position.

        Returns None if there are no legal moves (game over).
        """
        if not board.legal_moves:
            return None

        maximising = board.turn == chess.WHITE
        best = None
        best_score = -float("inf") if maximising else float("inf")

        for move in board.legal_moves:
            board.push(move)
            score = self._alpha_beta(
                board,
                self.depth - 1,
                -float("inf"),
                float("inf"),
                not maximising,
            )
            board.pop()

            if maximising:
                if score > best_score:
                    best_score, best = score, move
            else:
                if score < best_score:
                    best_score, best = score, move

        return best


if __name__ == "__main__":
    # Quick smoke-test with material fallback (no model needed)
    engine = AlphaBetaEngine(model=None, depth=3)
    board = chess.Board()

    print("Starting position, AI (White) thinking at depth 3…")
    move = engine.best_move(board)
    print(f"Best move: {board.san(move)}")
