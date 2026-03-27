"""
play.py
Play chess against the AI in your terminal.

Usage:
  # With trained model:
  python play.py --model models/chess_eval.pt --depth 3

  # Without model (material heuristic fallback):
  python play.py --depth 3

  # Play as Black (AI goes first):
  python play.py --color black

Controls:
  Enter moves in Standard Algebraic Notation (SAN): e4, Nf3, O-O, etc.
  Type 'quit' or 'exit' to stop.
  Type 'undo' to take back your last move.
  Type 'board' to reprint the board.
"""

import argparse
import sys
import time

import chess

from model import ChessEvaluator, load_model
from search import AlphaBetaEngine

# ANSI colours for terminal board rendering
RESET  = "\033[0m"
WHITE_SQ = "\033[48;5;230m"   # light square
BLACK_SQ = "\033[48;5;94m"    # dark square
WHITE_PC = "\033[1;37m"
BLACK_PC = "\033[1;30m"

PIECE_UNICODE = {
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.KING,   chess.WHITE): "♔",
    (chess.PAWN,   chess.BLACK): "♟",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.KING,   chess.BLACK): "♚",
}


def print_board(board: chess.Board, player_color: chess.Color):
    """Render the board with ANSI colours, oriented to the player's side."""
    ranks = range(7, -1, -1) if player_color == chess.WHITE else range(8)
    files = range(8)        if player_color == chess.WHITE else range(7, -1, -1)

    print()
    for rank in ranks:
        print(f" {rank + 1} ", end="")
        for file in files:
            square = chess.square(file, rank)
            piece  = board.piece_at(square)
            is_light = (rank + file) % 2 == 1
            bg = WHITE_SQ if is_light else BLACK_SQ

            if piece:
                fg = WHITE_PC if piece.color == chess.WHITE else BLACK_PC
                sym = PIECE_UNICODE[(piece.piece_type, piece.color)]
                print(f"{bg}{fg} {sym} {RESET}", end="")
            else:
                print(f"{bg}   {RESET}", end="")
        print()

    file_labels = "abcdefgh" if player_color == chess.WHITE else "hgfedcba"
    print("    " + "  ".join(file_labels))
    print()


def get_game_result(board: chess.Board) -> str | None:
    """Return a result string if the game is over, else None."""
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate! {winner} wins."
    if board.is_stalemate():
        return "Stalemate — it's a draw."
    if board.is_insufficient_material():
        return "Draw by insufficient material."
    if board.is_seventyfive_moves():
        return "Draw by 75-move rule."
    if board.is_fivefold_repetition():
        return "Draw by fivefold repetition."
    return None


def play(args):
    player_color = chess.WHITE if args.color == "white" else chess.BLACK
    ai_color     = not player_color

    # Load model (or fall back to material heuristic)
    model = None
    if args.model:
        try:
            model = load_model(args.model)
            print(f"Loaded model from {args.model}")
        except FileNotFoundError:
            print(f"Model file '{args.model}' not found — using material heuristic.")

    engine = AlphaBetaEngine(model=model, depth=args.depth)
    board  = chess.Board()

    print("\n── Chess AI ─────────────────────────────")
    print(f"  You play as: {'White' if player_color == chess.WHITE else 'Black'}")
    print(f"  AI depth:    {args.depth}")
    print("  Commands: undo | board | quit")
    print("─────────────────────────────────────────\n")

    while True:
        result = get_game_result(board)
        if result:
            print_board(board, player_color)
            print(f"\n{result}")
            break

        print_board(board, player_color)

        # ── AI turn ──
        if board.turn == ai_color:
            print("AI is thinking…")
            t0 = time.time()
            move = engine.best_move(board)
            elapsed = time.time() - t0

            if move is None:
                print("AI has no moves — game over.")
                break

            san = board.san(move)
            board.push(move)
            print(f"AI plays: {san}  ({elapsed:.1f}s)\n")
            continue

        # ── Player turn ──
        try:
            raw = input("Your move: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if raw.lower() in ("quit", "exit"):
            print("Thanks for playing!")
            sys.exit(0)

        if raw.lower() == "board":
            continue  # loop will reprint

        if raw.lower() == "undo":
            # Undo both AI and player moves
            try:
                board.pop()   # undo AI
                board.pop()   # undo player
                print("Move undone.\n")
            except IndexError:
                print("Nothing to undo.\n")
            continue

        # Parse SAN or UCI
        try:
            move = board.parse_san(raw)
        except ValueError:
            try:
                move = chess.Move.from_uci(raw)
                if move not in board.legal_moves:
                    raise ValueError
            except ValueError:
                print(f"  Invalid move: '{raw}'. Try again (e.g. e4, Nf3, O-O).\n")
                continue

        board.push(move)


def parse_args():
    p = argparse.ArgumentParser(description="Play chess against the AI")
    p.add_argument("--model",  default=None, help="Path to trained .pt model")
    p.add_argument("--depth",  type=int, default=3, help="Search depth (plies)")
    p.add_argument("--color",  choices=["white", "black"], default="white")
    return p.parse_args()


if __name__ == "__main__":
    play(parse_args())
