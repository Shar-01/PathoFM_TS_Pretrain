"""Optimizer, scheduler, and early-stopping configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer, scheduler, and early-stopping configuration."""

    epochs: int = 250
    patience: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    early_stop_min_delta: float = 1e-4
