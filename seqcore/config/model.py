"""Transformer model configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Axial embedding and attention-stack configuration."""

    embed_dim: int = 128
    depth: int = 8
    heads: int = 8
