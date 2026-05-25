"""Loss-weight and masking configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConfig:
    """Loss weights and masking configuration."""

    mask_ratio: float = 0.70
    weight_masked: float = 1.0
    weight_projected: float = 1.0
    weight_table: float = 1.0
