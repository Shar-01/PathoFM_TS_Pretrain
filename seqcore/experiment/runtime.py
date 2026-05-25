"""Runtime assembly for experiments."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from seqcore.config import ExperimentConfig
from seqcore.data.dataset import SequenceCorpus
from seqcore.data.factories import build_datasets, build_loaders
from seqcore.experiment.builders import build_model, build_optimizer, build_scheduler
from seqcore.models import SequenceUnit
from seqcore.utils import get_default_device, set_seed


@dataclass(frozen=True)
class ExperimentRuntime:
    """Concrete objects used by one experiment run."""

    device: torch.device
    train_ds: SequenceCorpus
    val_ds: SequenceCorpus
    test_ds: SequenceCorpus
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    model: SequenceUnit
    optimizer: torch.optim.Optimizer
    scheduler: object


def infer_input_dim(train_ds) -> int:
    """Infer feature dimensionality using the original sample access pattern."""
    return train_ds[0][0].shape[-1]


def build_runtime(config: ExperimentConfig) -> ExperimentRuntime:
    """Create datasets, loaders, model, optimizer, and scheduler."""
    set_seed(config.seed)
    device = get_default_device()

    train_ds, val_ds, test_ds = build_datasets(config)
    train_loader, val_loader, test_loader = build_loaders(config, train_ds, val_ds, test_ds)

    model = build_model(config, input_dim=infer_input_dim(train_ds), device=device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    return ExperimentRuntime(
        device=device,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
