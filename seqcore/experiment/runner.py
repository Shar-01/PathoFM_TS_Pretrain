"""Top-level experiment runner."""
from __future__ import annotations

from seqcore.config import ExperimentConfig
from seqcore.experiment.final_eval import run_final_test
from seqcore.experiment.loop import run_training_loop
from seqcore.experiment.runtime import build_runtime, infer_input_dim


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> None:
    """Run training, early stopping, checkpointing, and final test evaluation."""
    runtime = build_runtime(config)
    run_training_loop(
        config,
        runtime.model,
        runtime.train_loader,
        runtime.val_loader,
        runtime.optimizer,
        runtime.scheduler,
        runtime.device,
    )
    run_final_test(config, runtime.model, runtime.test_loader, runtime.device)
