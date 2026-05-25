"""Utilities for constructing abstract query tables."""
from __future__ import annotations

import torch


def build_legacy_query_table(batch_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask one query group from the batch to form an abstract table."""
    R, T, F = batch_full.shape
    idx_q = torch.randint(0, R, (), device=batch_full.device).item()
    table = batch_full.clone()
    query_mask = torch.ones_like(table[idx_q], dtype=torch.bool)
    table[idx_q] = 0.0
    row_type = torch.zeros(R, dtype=torch.long, device=batch_full.device)
    row_type[idx_q] = 1
    return table, row_type, query_mask


def build_query_table(batch_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask one query group from the batch to form an abstract table."""
    R, T, F = batch_full.shape
    idx_q = torch.randint(0, R, (), device=batch_full.device).item()
    table = batch_full.clone()
    query_mask = torch.zeros_like(table, dtype=torch.bool)
    query_mask[idx_q] = True
    table[idx_q] = 0.0
    row_type = torch.zeros(R, dtype=torch.long, device=batch_full.device)
    row_type[idx_q] = 1
    return table, row_type, query_mask
