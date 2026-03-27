"""
model.py
CNN-based position evaluator.

Architecture:
  Input  : (batch, 12, 8, 8) board tensor
  Body   : 3 residual conv blocks (64 filters, 3×3)
  Head   : Global average pool → 128-d FC → 1 scalar (tanh)
  Output : score in [-1, 1]  (+1 = White winning, -1 = Black winning)

The tanh output mirrors the convention used in self-play / MCTS systems
and is easy to interpret as a win-probability proxy.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """One residual convolutional block."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ChessEvaluator(nn.Module):
    """
    Predicts a position score in [-1, 1].
    Positive → good for White, Negative → good for Black.
    """

    def __init__(self, num_res_blocks: int = 3, channels: int = 64):
        super().__init__()

        # Stem: project 12 input planes → `channels`
        self.stem = nn.Sequential(
            nn.Conv2d(12, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual body
        self.body = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # (batch, channels, 1, 1)
            nn.Flatten(),                # (batch, channels)
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),                   # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 12, 8, 8) float tensor

        Returns:
            (batch,) score tensor
        """
        x = self.stem(x)
        x = self.body(x)
        return self.value_head(x).squeeze(-1)


def load_model(path: str, device: str = "cpu") -> ChessEvaluator:
    """Load a saved model checkpoint."""
    model = ChessEvaluator()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    model = ChessEvaluator()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")   # ~180k — fast and deployable

    dummy = torch.randn(4, 12, 8, 8)
    scores = model(dummy)
    print(f"Output shape: {scores.shape}")          # (4,)
    print(f"Sample scores: {scores.detach().numpy()}")
