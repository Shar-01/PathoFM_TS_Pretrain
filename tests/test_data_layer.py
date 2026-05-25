from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from seqcore.config import DataConfig, ExperimentConfig
from seqcore.data.columns import DEFAULT_COLUMN_KEYWORDS, select_columns
from seqcore.data.dataset import SequenceCorpus
from seqcore.data.factories import build_datasets, build_loaders
from seqcore.data.files import glob_csv_files, list_csv_files
from seqcore.data.normalization import (
    normalize_per_snippet,
    normalize_snippet,
    normalize_with_global_bounds,
    sanitize_array,
)
from seqcore.data.selection import select_columns as select_columns_compat
from seqcore.data.statistics import collect_rows_for_subjects, compute_global_stats, read_selected_rows
from seqcore.data.subjects import collect_subjects, split_subjects, subject_from_filepath
from seqcore.data.windows import iter_sliding_windows, window_length


def test_select_columns_is_keyword_based_and_order_preserving() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "HIP_ANGLES": [3, 4],
            "something_progression": [5, 6],
            "left_power": [7, 8],
            "ankle_moment": [9, 10],
            "ignored": [11, 12],
        }
    )
    selected = select_columns(df)
    assert selected.tolist() == [[3, 5, 7, 9], [4, 6, 8, 10]]
    assert select_columns_compat(df).tolist() == selected.tolist()
    assert DEFAULT_COLUMN_KEYWORDS == ("angles", "progression", "power", "moment")


def test_file_listing_and_subject_split(synthetic_data_dir) -> None:
    globbed = glob_csv_files(str(synthetic_data_dir))
    listed = list_csv_files(str(synthetic_data_dir))
    assert len(globbed) == 4
    assert listed == sorted(globbed)
    assert subject_from_filepath(listed[0]) == "S001"
    assert collect_subjects(str(synthetic_data_dir)) == ["S001", "S002", "S003", "S004"]

    train_a, val_a, test_a = split_subjects(str(synthetic_data_dir), n_test=1, n_val=1, seed=7)
    train_b, val_b, test_b = split_subjects(str(synthetic_data_dir), n_test=1, n_val=1, seed=7)
    assert (train_a, val_a, test_a) == (train_b, val_b, test_b)
    assert len(train_a) == 2
    assert len(val_a) == 1
    assert len(test_a) == 1
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a).isdisjoint(test_a)
    assert set(val_a).isdisjoint(test_a)

    with pytest.raises(ValueError, match="Not enough unique subjects"):
        split_subjects(str(synthetic_data_dir), n_test=3, n_val=3, seed=7)


def test_window_helpers_preserve_legacy_boundaries() -> None:
    arr = np.arange(20).reshape(10, 2)
    windows = list(iter_sliding_windows(arr, past_len=3, future_len=2, stride=2))
    assert window_length(3, 2) == 5
    assert len(windows) == 3
    assert windows[0].tolist() == arr[0:5].tolist()
    assert windows[1].tolist() == arr[2:7].tolist()
    assert windows[2].tolist() == arr[4:9].tolist()


def test_sanitize_and_normalization_helpers() -> None:
    arr = np.array([[np.nan, np.inf, -np.inf], [1.0, 2.0, -3.0]], dtype=np.float32)
    sanitized = sanitize_array(arr)
    assert sanitized.tolist() == [[0.0, 0.0, 0.0], [1.0, 2.0, -3.0]]

    snippet = np.array([[0.0, 2.0], [2.0, 6.0]], dtype=np.float32)
    per = normalize_per_snippet(snippet)
    np.testing.assert_allclose(per, np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32), atol=1e-6)

    min_values = np.array([[0.0, 2.0]], dtype=np.float32)
    max_values = np.array([[4.0, 10.0]], dtype=np.float32)
    global_norm = normalize_with_global_bounds(snippet, min_values, max_values)
    np.testing.assert_allclose(global_norm, np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(normalize_snippet(snippet, min_values, max_values), global_norm, atol=1e-6)
    np.testing.assert_allclose(normalize_snippet(snippet), per, atol=1e-6)


def test_statistics_and_dataset_construction(synthetic_data_dir) -> None:
    csv_path = sorted(glob_csv_files(str(synthetic_data_dir)))[0]
    selected = read_selected_rows(csv_path)
    assert selected.shape == (120, 4)

    rows = collect_rows_for_subjects(str(synthetic_data_dir), ["S001", "S002"])
    assert len(rows) == 2
    col_min, col_max = compute_global_stats(str(synthetic_data_dir), ["S001", "S002"])
    assert col_min.shape == (1, 4)
    assert col_max.shape == (1, 4)
    assert np.all(col_max >= col_min)

    ds = SequenceCorpus(
        str(synthetic_data_dir),
        ["S001", "S002"],
        past_len=50,
        future_len=50,
        stride=20,
        min=col_min,
        max=col_max,
    )
    # Two files, each length 120, total window length 100, stride 20 => starts 0 and 20.
    assert len(ds) == 4
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (100, 4)
    assert sample.dtype == torch.float32
    assert torch.isfinite(sample).all()


def test_dataset_per_snippet_normalization_without_global_bounds(synthetic_data_dir) -> None:
    ds = SequenceCorpus(str(synthetic_data_dir), ["S001"], past_len=50, future_len=50, stride=100)
    assert len(ds) == 1
    sample = ds[0]
    assert sample.shape == (100, 4)
    assert float(sample.max()) <= 1.000001
    assert float(sample.min()) >= -1.000001


def test_dataset_and_loader_factories(synthetic_data_dir) -> None:
    config = ExperimentConfig(
        data=DataConfig(
            data_dir=str(synthetic_data_dir),
            batch_size=2,
            num_workers=0,
            past_len=50,
            future_len=50,
            stride=100,
            n_test_subjects=1,
            n_val_subjects=1,
        )
    )
    train_ds, val_ds, test_ds = build_datasets(config)
    assert len(train_ds) == 2
    assert len(val_ds) == 1
    assert len(test_ds) == 1

    train_loader, val_loader, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    train_step = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    assert train_step.shape == (2, 100, 4)
    assert val_batch.shape == (1, 100, 4)
    assert test_batch.shape == (1, 100, 4)
