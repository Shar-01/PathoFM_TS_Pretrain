"""Dataset, split, and loader configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Dataset and subject-level split configuration."""

    data_dir: str = r"D:\\Temporal_Project\\Data"
    batch_size: int = 128
    num_workers: int = 4
    past_len: int = 50
    future_len: int = 50
    stride: int = 20
    n_test_subjects: int = 10
    n_val_subjects: int = 10
