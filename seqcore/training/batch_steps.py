"""Batch-level pass orchestration."""
from __future__ import annotations

from typing import Optional

import torch

from seqcore.models import SequenceUnit
from seqcore.training.branches import (
    ChannelLosses,
    TableOutput,
    compute_eval_table_loss,
    compute_masked_loss,
    compute_projection_loss,
    compute_train_table_loss,
)
from seqcore.training.objective import apply_optimizer_step, weighted_total_loss

__all__ = [
    "ChannelLosses",
    "TableOutput",
    "compute_eval_table_loss",
    "compute_masked_loss",
    "compute_projection_loss",
    "compute_train_table_loss",
    "evaluate_step",
    "train_step",
    "weighted_total_loss",
]


def train_step(
    model: SequenceUnit,
    full: torch.Tensor,
    optim: torch.optim.Optimizer,
    weight_masked: float = 1.0,
    weight_projected: float = 1.0,
    weight_table: float = 1.0,
    mask_ratio: float = 0.7,
    grad_clip_norm: float = 1.0,
) -> ChannelLosses:
    """Train on one batch and return channel-wise losses."""
    loss_masked, _ = compute_masked_loss(model, full, mask_ratio, enabled=weight_masked > 0)
    loss_projected, _ = compute_projection_loss(model, full, enabled=weight_projected > 0)
    loss_table = compute_train_table_loss(model, full, enabled=weight_table > 0)
    losses = ChannelLosses(masked=loss_masked, projected=loss_projected, table=loss_table)

    loss = weighted_total_loss(losses, weight_masked, weight_projected, weight_table)
    apply_optimizer_step(model, optim, loss, grad_clip_norm)
    return losses


def evaluate_step(
    model: SequenceUnit,
    full: torch.Tensor,
    masked: bool = True,
    projected: bool = True,
    table: bool = True,
) -> tuple[ChannelLosses, Optional[torch.Tensor], Optional[torch.Tensor], Optional[TableOutput]]:
    """Evaluate one batch and return losses plus path predictions."""
    loss_masked, recon = compute_masked_loss(model, full, enabled=masked)
    loss_projected, pred_fut = compute_projection_loss(model, full, enabled=projected)
    loss_table, table_output = compute_eval_table_loss(model, full, enabled=table)
    return ChannelLosses(masked=loss_masked, projected=loss_projected, table=loss_table), recon, pred_fut, table_output
