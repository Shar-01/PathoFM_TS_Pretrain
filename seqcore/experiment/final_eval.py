"""Final checkpoint evaluation helpers."""
from __future__ import annotations

from seqcore.config import ExperimentConfig
from seqcore.experiment.checkpointing import load_model_state
from seqcore.reporting import print_test_report
from seqcore.training import eval_epoch


def table_enabled(config: ExperimentConfig) -> bool:
    """Return whether the abstract table objective participates in testing."""
    return config.objective.weight_table > 0


def run_final_test(config: ExperimentConfig, model, test_loader, device) -> None:
    """Load the best checkpoint and print the final test report."""
    load_model_state(model, config.save_path, device)
    masked_path, projected_path, _table_path, extra = eval_epoch(
        model,
        test_loader,
        device,
        mode="test",
        masked=config.objective.weight_masked > 0,
        projected=config.objective.weight_projected > 0,
        table=table_enabled(config),
    )
    print_test_report(masked_path, projected_path, extra)
