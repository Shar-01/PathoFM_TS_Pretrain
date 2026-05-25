"""Makers for unit and optimization objects."""
from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from seqcore.config import ExperimentConfig
from seqcore.models import SequenceUnit


def make_unit(config: ExperimentConfig, input_dim: int, device: torch.device) -> SequenceUnit:
    """Instantiate the composed unit."""
    model_cfg = config.model
    return SequenceUnit(
        input_dim=input_dim,
        embed_dim=model_cfg.embed_dim,
        depth=model_cfg.depth,
        heads=model_cfg.heads,
    ).to(device)


def make_optimizer(config: ExperimentConfig, model: SequenceUnit) -> AdamW:
    """Create the optimizer."""
    return AdamW(model.parameters(), lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay)


def build_scheduler(config: ExperimentConfig, optimizer: AdamW) -> CosineAnnealingLR:
    """Create the cosine annealing scheduler."""
    return CosineAnnealingLR(optimizer, T_max=config.optim.epochs)


build_model = make_unit
build_optimizer = make_optimizer
