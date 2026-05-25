"""Validation scoring helpers."""
from __future__ import annotations


def validation_total(mae: float, mf: float, icl: float) -> float:
    """Return the validation score used for checkpoint selection."""
    return mae + mf + icl
