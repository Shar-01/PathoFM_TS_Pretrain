"""Conversion from parsed CLI options to immutable experiment configs."""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any, Iterable, Mapping

from seqcore.config import DataConfig, ExperimentConfig, ModelConfig, ObjectiveConfig, OptimizationConfig

ROOT_FIELDS = ("seed", "save_path")
DATA_FIELDS = (
    "data_dir",
    "batch_size",
    "num_workers",
    "past_len",
    "future_len",
    "stride",
    "n_test_subjects",
    "n_val_subjects",
)
MODEL_FIELDS = ("embed_dim", "depth", "heads")
OPTIMIZATION_FIELDS = (
    "epochs",
    "patience",
    "learning_rate",
    "weight_decay",
    "grad_clip_norm",
    "early_stop_min_delta",
)
OBJECTIVE_FIELDS = ("mask_ratio", "weight_masked", "weight_projected", "weight_table")


def build_default_config() -> ExperimentConfig:
    """Return the legacy-equivalent default experiment configuration."""
    return ExperimentConfig()


def namespace_to_mapping(namespace: argparse.Namespace) -> Mapping[str, Any]:
    """Expose parsed CLI values as a mapping."""
    return vars(namespace)


def selected_overrides(values: Mapping[str, Any], field_names: Iterable[str]) -> dict[str, Any]:
    """Return only CLI fields that were explicitly supplied with non-None values."""
    return {name: values[name] for name in field_names if values.get(name) is not None}


def build_data_config(base: DataConfig, values: Mapping[str, Any]) -> DataConfig:
    """Build the data section from CLI overrides."""
    return replace(base, **selected_overrides(values, DATA_FIELDS))


def build_model_config(base: ModelConfig, values: Mapping[str, Any]) -> ModelConfig:
    """Build the model section from CLI overrides."""
    return replace(base, **selected_overrides(values, MODEL_FIELDS))


def build_optimization_config(base: OptimizationConfig, values: Mapping[str, Any]) -> OptimizationConfig:
    """Build the optimization section from CLI overrides."""
    return replace(base, **selected_overrides(values, OPTIMIZATION_FIELDS))


def build_objective_config(base: ObjectiveConfig, values: Mapping[str, Any]) -> ObjectiveConfig:
    """Build the objective section from CLI overrides."""
    return replace(base, **selected_overrides(values, OBJECTIVE_FIELDS))


def build_experiment_config(namespace: argparse.Namespace) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from parsed command-line arguments."""
    values = namespace_to_mapping(namespace)
    base = build_default_config()
    root_overrides = selected_overrides(values, ROOT_FIELDS)
    return replace(
        base,
        **root_overrides,
        data=build_data_config(base.data, values),
        model=build_model_config(base.model, values),
        optim=build_optimization_config(base.optim, values),
        objective=build_objective_config(base.objective, values),
    )
