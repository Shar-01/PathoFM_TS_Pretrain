"""Reproducibility helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs using the original deterministic settings."""
    np.random.seed(seed)
    torch.manual_seed(seed)
