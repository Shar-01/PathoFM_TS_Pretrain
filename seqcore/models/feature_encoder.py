"""Input adapter modules."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """Two-layer MLP that maps feature vectors into the working dimension."""

    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
