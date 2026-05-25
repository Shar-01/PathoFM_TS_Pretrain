"""Top-level experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

from seqcore.config.data import DataConfig
from seqcore.config.model import ModelConfig
from seqcore.config.objective import ObjectiveConfig
from seqcore.config.optimization import OptimizationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete training/evaluation configuration."""

    seed: int = 42
    save_path: str = "best_state.pt"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
