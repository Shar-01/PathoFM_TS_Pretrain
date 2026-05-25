"""CLI command implementations."""
from __future__ import annotations

from seqcore.config import ExperimentConfig
from seqcore.experiment import run_experiment


def run_train(config: ExperimentConfig) -> int:
    """Execute the train/evaluate experiment command."""
    run_experiment(config)
    return 0
