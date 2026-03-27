# Chess AI — Neural Network + Alpha-Beta Search

A chess engine that combines a **convolutional neural network** position evaluator with **alpha-beta minimax search** — conceptually similar to how AlphaZero works, built from scratch in Python.

---

## How it works

```
Board state (8×8)
      │
      ▼
Board encoder          →  (12, 8, 8) tensor
      │                   12 planes, one per piece type per colour
      ▼
CNN Evaluator          →  score in [-1, 1]
  3 × residual blocks      +1 = White winning
  Global avg pool          -1 = Black winning
      │
      ▼
Alpha-Beta Search      →  best move
  Minimax + pruning
  Depth 3-5 plies
```

The neural network is trained via **supervised learning** on PGN game databases — it learns to predict game outcomes from positions, developing an implicit understanding of piece activity, king safety, and pawn structure.

---

## Project structure

```
chess-ai/
├── board_encoder.py   # Converts chess.Board → (12,8,8) tensor
├── model.py           # CNN architecture (ChessEvaluator)
├── train.py           # Training pipeline (PGN → model weights)
├── search.py          # Alpha-beta engine
├── play.py            # Terminal UI to play against the AI
├── requirements.txt
└── models/            # Saved model weights (created after training)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Play immediately (no training needed)

The engine works right away using a material-count heuristic:

```bash
python play.py --depth 3
```

### 3. Train the neural network

Download free PGN files from [Lichess](https://lichess.org/api) or [PGN Mentor](https://www.pgnmentor.com), then:

```bash
python train.py --pgn data/games.pgn --epochs 20 --max-games 5000
```

Training logs validation loss each epoch and saves the best checkpoint to `models/chess_eval.pt`.

### 4. Play against the trained model

```bash
python play.py --model models/chess_eval.pt --depth 3
```

Play as Black:

```bash
python play.py --model models/chess_eval.pt --depth 3 --color black
```

---

## Controls (in-game)

| Input | Action |
|-------|--------|
| `e4`, `Nf3`, `O-O` | Make a move (SAN notation) |
| `undo` | Take back your last move |
| `board` | Reprint the board |
| `quit` | Exit |

---

## Key concepts demonstrated

| Concept | Where |
|---------|-------|
| **Board representation** | `board_encoder.py` — 12-plane bitboard tensor |
| **Residual CNNs** | `model.py` — ResNet-style architecture |
| **Supervised learning** | `train.py` — result-based position labelling |
| **Alpha-beta pruning** | `search.py` — classic AI search algorithm |
| **CLI application design** | `play.py` — ANSI terminal chess board |

---

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--depth` | 3 | Search depth in plies. 3 = fast, 5 = strong |
| `--epochs` | 20 | Training epochs |
| `--max-games` | 5000 | PGN games to train on |
| `--res-blocks` | 3 | Residual blocks in model |
| `--batch-size` | 256 | Training batch size |

---

## Extending the project

- **Self-play training** — generate games between two copies of the engine and label with game result
- **Policy head** — add a second output head predicting move probabilities (full AlphaZero style)
- **Opening book** — load ECO openings and skip search in the first 10 moves
- **UCI protocol** — implement the Universal Chess Interface so the engine works in GUIs like Arena or Lichess's board editor
- **Web UI** — expose the engine via FastAPI and build a browser board with `chess.js`

---

## References

- [python-chess documentation](https://python-chess.readthedocs.io/)
- [AlphaZero paper (Silver et al., 2018)](https://arxiv.org/abs/1712.01815)
- [Chess Programming Wiki](https://www.chessprogramming.org/)
