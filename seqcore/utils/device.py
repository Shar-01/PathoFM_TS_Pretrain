"""Device selection utilities."""
from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
