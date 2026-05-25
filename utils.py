"""Compatibility exports for legacy imports."""
from seqcore.data import compute_global_stats, split_subjects
from seqcore.data.columns import select_columns as _select_columns
from seqcore.training import (
    build_legacy_query_table,
    build_query_table,
    masked_error,
    projection_error,
)
from seqcore.utils import set_seed

__all__ = [
    "_select_columns",
    "build_legacy_query_table",
    "build_query_table",
    "compute_global_stats",
    "masked_error",
    "projection_error",
    "set_seed",
    "split_subjects",
]
