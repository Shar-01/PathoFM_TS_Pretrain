"""Declarative command-line option specifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class OptionSpec:
    """One argparse option described independently from parser construction."""

    flags: Tuple[str, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionGroupSpec:
    """A named argparse option group."""

    title: str
    description: str
    options: Tuple[OptionSpec, ...]
