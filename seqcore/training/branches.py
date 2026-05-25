"""Channel-specific batch computations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from seqcore.models import SequenceUnit
from seqcore.training.losses import masked_error, projection_error
from seqcore.training.query_tables import build_query_table


@dataclass(frozen=True)
class ChannelLosses:
    """Channel-wise losses for one batch."""

    masked: torch.Tensor
    projected: torch.Tensor
    table: torch.Tensor


@dataclass(frozen=True)
class TableOutput:
    """Table reconstruction artifacts needed for metrics."""

    prediction: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor


def inactive_loss(reference: torch.Tensor) -> torch.Tensor:
    """Return a disabled-branch loss on the same device as ``reference``."""
    return reference.new_zeros(())


def has_branch(model: SequenceUnit, name: str) -> bool:
    """Return whether a model exposes a callable branch."""
    return callable(getattr(model, name, None))


def has_table_eval_surface(model: SequenceUnit) -> bool:
    """Return whether eval-time table reconstruction can be computed."""
    return (
        has_branch(model, "forward_table")
        and has_branch(model, "_patchify_table")
        and callable(getattr(model, "encoder", None))
        and callable(getattr(model, "decoder", None))
    )


def mask_has_values(mask: torch.Tensor) -> bool:
    """Return whether a boolean mask selects at least one value."""
    return bool(mask.any().item())


def run_masked_reconstruction(
    model: SequenceUnit,
    full: torch.Tensor,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the reconstruction branch and return raw model outputs."""
    return model.forward_masked(full, mask_ratio)


def run_suffix_projection(model: SequenceUnit, full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the suffix projection channel and return raw model outputs."""
    return model.forward_projection(full)


def query_rows(
    reconstruction: torch.Tensor,
    table: torch.Tensor,
    row_type: torch.Tensor,
    query_mask: torch.Tensor,
) -> TableOutput:
    """Collect reconstructed, target, and mask tensors for query rows."""
    idx_query = row_type == 1
    return TableOutput(
        prediction=reconstruction[idx_query],
        target=table[idx_query],
        mask=query_mask[idx_query],
    )


def reconstruct_table(model: SequenceUnit, table: torch.Tensor, row_type: torch.Tensor) -> torch.Tensor:
    """Encode and decode an abstract query table."""
    encoded = model.encoder(model._patchify_table(table, row_type))
    return model.decoder(encoded)


def compute_masked_loss(
    model: SequenceUnit,
    full: torch.Tensor,
    mask_ratio: float = 0.7,
    enabled: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the masked channel and return loss plus reconstruction."""
    if not enabled or not has_branch(model, "forward_masked"):
        return inactive_loss(full), None

    recon_mae, mask_mae = run_masked_reconstruction(model, full, mask_ratio)
    if not mask_has_values(mask_mae):
        return inactive_loss(full), recon_mae
    return masked_error(recon_mae, full, mask_mae), recon_mae


def compute_projection_loss(
    model: SequenceUnit,
    full: torch.Tensor,
    enabled: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the projection channel and return loss plus prediction."""
    if not enabled or not has_branch(model, "forward_projection"):
        return inactive_loss(full), None

    pred_fut, _ = run_suffix_projection(model, full)
    target = full[:, 50:]
    if pred_fut.numel() == 0 or target.numel() == 0:
        return inactive_loss(full), pred_fut
    return projection_error(pred_fut, target), pred_fut


def compute_train_table_loss(model: SequenceUnit, full: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Run the train-time table objective when enabled."""
    if not enabled or not has_branch(model, "forward_table"):
        return inactive_loss(full)

    table, row_t, qmask = build_query_table(full)
    if not mask_has_values(qmask):
        return inactive_loss(full)
    return model.forward_table(table, row_t, qmask)


def compute_eval_table_loss(
    model: SequenceUnit,
    full: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, Optional[TableOutput]]:
    """Run the eval-time table objective and expose query-row predictions."""
    if not enabled or not has_table_eval_surface(model):
        return inactive_loss(full), None

    table, row_t, qmask = build_query_table(full)
    if not mask_has_values(qmask):
        return inactive_loss(full), None

    output = query_rows(reconstruct_table(model, table, row_t), table, row_t, qmask)
    loss_table = F.mse_loss(output.prediction[output.mask], output.target[output.mask])
    return loss_table, output
