"""Shared sequence stack."""
from __future__ import annotations

import torch
import torch.nn as nn


class SequenceStack(nn.Module):
    """Pre-norm attention stack operating on ``[B, T, D]``."""

    def __init__(self, depth: int, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transformer(z)
