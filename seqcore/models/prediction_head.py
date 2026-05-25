"""Output-head construction utilities."""
from __future__ import annotations

import torch.nn as nn


def make_projection_head(embed_dim: int, output_dim: int) -> nn.Sequential:
    """Create a layer-normalized linear projection head."""
    return nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, output_dim))
