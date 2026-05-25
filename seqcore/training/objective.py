"""Loss composition and optimizer helpers."""
from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

from seqcore.models import SequenceUnit
from seqcore.training.branches import ChannelLosses


def weighted_total_loss(
    losses: ChannelLosses,
    weight_masked: float,
    weight_projected: float,
    weight_table: float,
) -> torch.Tensor:
    """Combine channel losses using configured weights."""
    return weight_masked * losses.masked + weight_projected * losses.projected + weight_table * losses.table


def reset_gradients(optimizer: torch.optim.Optimizer) -> None:
    """Clear gradients before a backward pass."""
    optimizer.zero_grad(set_to_none=True)


def clip_model_gradients(model: SequenceUnit, grad_clip_norm: float) -> None:
    """Apply the configured gradient-norm bound."""
    clip_grad_norm_(model.parameters(), grad_clip_norm)


def apply_optimizer_step(
    model: SequenceUnit,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    grad_clip_norm: float,
) -> None:
    """Run the standard backward, clip, and step sequence."""
    reset_gradients(optimizer)
    if not loss.requires_grad:
        return
    loss.backward()
    clip_model_gradients(model, grad_clip_norm)
    optimizer.step()
