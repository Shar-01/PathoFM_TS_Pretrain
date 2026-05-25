"""Single-epoch training and evaluation routines."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from seqcore.models import SequenceUnit
from seqcore.training.metrics import compute_test_metrics
from seqcore.training.state import PredictionBuffers, RunningLossTotals
from seqcore.training.batch_steps import evaluate_step, train_step


def train_one_epoch(
    model: SequenceUnit,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    weight_masked: float = 1.0,
    weight_projected: float = 1.0,
    weight_table: float = 1.0,
    mask_ratio: float = 0.7,
    grad_clip_norm: float = 1.0,
) -> tuple[float, float, float]:
    """Train ``model`` for one epoch and return branch-wise mean losses."""
    model.train()
    totals = RunningLossTotals()

    for full in loader:
        full = full.to(device)
        losses = train_step(
            model,
            full,
            optim,
            weight_masked=weight_masked,
            weight_projected=weight_projected,
            weight_table=weight_table,
            mask_ratio=mask_ratio,
            grad_clip_norm=grad_clip_norm,
        )
        totals.update(losses.masked, losses.projected, losses.table)

    return totals.averages(
        masked_enabled=weight_masked > 0,
        projected_enabled=weight_projected > 0,
        table_enabled=weight_table > 0,
    )


@torch.no_grad()
def eval_epoch(
    model: SequenceUnit,
    loader: DataLoader,
    device: torch.device,
    mode: str = "train",
    masked: bool = True,
    projected: bool = True,
    table: bool = True,
):
    """Evaluate one epoch.

    Returns:
        ``train`` mode: masked, projected, table losses.
        ``test`` mode: masked, projected, table losses plus metrics.
    """
    assert mode in {"train", "test"}
    model.eval()

    totals = RunningLossTotals()
    buffers = PredictionBuffers() if mode == "test" else None

    for full in loader:
        full = full.to(device)
        losses, recon, pred_fut, table_output = evaluate_step(
            model,
            full,
            masked=masked,
            projected=projected,
            table=table,
        )
        totals.update(losses.masked, losses.projected, losses.table)

        if buffers is not None:
            if recon is not None:
                buffers.append_reconstruction(target=full, prediction=recon)
            if pred_fut is not None:
                buffers.append_forecasting(target=full[:, 50:], prediction=pred_fut)
            if table_output is not None:
                buffers.append_table(target=table_output.target, prediction=table_output.prediction, mask=table_output.mask)

    masked_loss, projected_loss, table_loss = totals.averages(
        masked_enabled=masked,
        projected_enabled=projected,
        table_enabled=table,
    )

    if mode == "train":
        return masked_loss, projected_loss, table_loss

    arrays = buffers.arrays()
    metrics = compute_test_metrics(**arrays)
    return masked_loss, projected_loss, table_loss, metrics
