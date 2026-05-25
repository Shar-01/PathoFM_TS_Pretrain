"""Compatibility exports for legacy imports."""
from seqcore.training import (
    build_legacy_query_table,
    build_query_table,
    eval_epoch,
    masked_error,
    projection_error,
    train_one_epoch,
)

__all__ = [
    "build_legacy_query_table",
    "build_query_table",
    "eval_epoch",
    "masked_error",
    "projection_error",
    "train_one_epoch",
]
