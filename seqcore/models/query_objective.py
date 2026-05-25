"""Model-side relation helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def query_group_error(rec: torch.Tensor, target: torch.Tensor, row_type: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE over masked query-group entries."""
    idx = row_type == 1
    rec_q = rec[idx]
    target_q = target[idx]
    mask_q = query_mask[idx]
    return F.mse_loss(rec_q[mask_q], target_q[mask_q])
