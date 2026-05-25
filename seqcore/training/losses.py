"""Generic criterion functions."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_error(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over selected entries."""
    return F.mse_loss(recon[mask], target[mask])


def projection_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over projected entries."""
    return F.mse_loss(pred, target)
