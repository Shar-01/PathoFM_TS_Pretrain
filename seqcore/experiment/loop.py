"""Training-loop orchestration helpers."""
from __future__ import annotations

from seqcore.config import ExperimentConfig
from seqcore.experiment.checkpointing import save_model_state
from seqcore.experiment.early_stopping import EarlyStoppingState
from seqcore.experiment.scoring import validation_total
from seqcore.reporting import print_epoch_summary
from seqcore.training import eval_epoch, train_one_epoch


def build_stopper(config: ExperimentConfig) -> EarlyStoppingState:
    """Create the validation stopper from optimization settings."""
    return EarlyStoppingState(
        patience=config.optim.patience,
        min_delta=config.optim.early_stop_min_delta,
    )


def train_epoch(config: ExperimentConfig, model, train_loader, optimizer, device) -> tuple[float, float, float]:
    """Run one configured training epoch."""
    return train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        weight_masked=config.objective.weight_masked,
        weight_projected=config.objective.weight_projected,
        weight_table=config.objective.weight_table,
        mask_ratio=config.objective.mask_ratio,
        grad_clip_norm=config.optim.grad_clip_norm,
    )


def validate_epoch(config: ExperimentConfig, model, val_loader, device) -> tuple[float, float, float]:
    """Run one validation epoch."""
    return eval_epoch(
        model,
        val_loader,
        device,
        mode="train",
        masked=config.objective.weight_masked > 0,
        projected=config.objective.weight_projected > 0,
        table=config.objective.weight_table > 0,
    )


def checkpoint_if_improved(
    config: ExperimentConfig,
    model,
    stopper: EarlyStoppingState,
    val_losses: tuple[float, float, float],
) -> bool:
    """Update validation state and persist a checkpoint when it improves."""
    improved = stopper.update(validation_total(*val_losses))
    if improved:
        save_model_state(model, config.save_path)
    return improved


def emit_stop_message(epoch: int, patience: int) -> None:
    """Print the early-stop notice."""
    print(f"Early stop at epoch {epoch} (no validation improvement for {patience})")


def run_training_loop(config: ExperimentConfig, model, train_loader, val_loader, optimizer, scheduler, device) -> None:
    """Run training, validation, checkpointing, and early stopping."""
    stopper = build_stopper(config)

    for epoch in range(1, config.optim.epochs + 1):
        tr_masked, tr_projected, tr_table = train_epoch(config, model, train_loader, optimizer, device)
        va_masked, va_projected, val_table = validate_epoch(config, model, val_loader, device)
        scheduler.step()

        improved = checkpoint_if_improved(config, model, stopper, (va_masked, va_projected, val_table))
        print_epoch_summary(epoch, config.optim.epochs, tr_masked, va_masked, tr_projected, va_projected, tr_table, improved)

        if stopper.should_stop():
            emit_stop_message(epoch, config.optim.patience)
            break
