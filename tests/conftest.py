from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import torch

# Keep small synthetic Transformer tests fast and deterministic on CPU.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


FEATURE_COLUMNS = [
    "hip_angles",
    "gait_progression",
    "ankle_power",
    "knee_moment",
]


def make_subject_csv(data_dir: pathlib.Path, subject: str, *, length: int = 120, offset: float = 0.0) -> pathlib.Path:
    """Create one synthetic temporal CSV whose filename matches the subject convention."""
    t = np.arange(length, dtype=np.float32)
    data = {
        "unrelated_column": np.full(length, 999.0, dtype=np.float32),
        "hip_angles": t + offset,
        "gait_progression": 0.5 * t + 1.0 + offset,
        "ankle_power": np.sin(t / 10.0) + offset,
        "knee_moment": np.cos(t / 11.0) + offset,
    }
    path = data_dir / f"trial_walk_{subject}_left_cycle.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.fixture()
def synthetic_data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    for idx, subject in enumerate(["S001", "S002", "S003", "S004"]):
        make_subject_csv(tmp_path, subject, length=120, offset=float(idx) * 10.0)
    return tmp_path
