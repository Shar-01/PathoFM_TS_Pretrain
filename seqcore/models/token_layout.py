"""Token layout helpers."""
from __future__ import annotations

import torch


def add_time_offset(z: torch.Tensor, time_pos: torch.Tensor, length: int) -> torch.Tensor:
    """Add learned temporal offsets in place."""
    z += time_pos[:length]
    return z


def add_group_offset(z: torch.Tensor, row_type_emb: torch.Tensor, row_type: torch.Tensor) -> torch.Tensor:
    """Add group-type offsets over the temporal axis."""
    z += row_type_emb[row_type].unsqueeze(1)
    return z
