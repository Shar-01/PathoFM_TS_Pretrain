"""Executable CLI entry point."""
from __future__ import annotations

from typing import Optional, Sequence

from seqcore.cli.commands import run_train
from seqcore.cli.configuration import build_experiment_config
from seqcore.cli.parser import TRAIN_ALIAS, build_parser


def dispatch(command: str, config) -> int:
    """Dispatch a parsed CLI command to its implementation."""
    if command == TRAIN_ALIAS:
        return run_train(config)
    raise ValueError(f"Unsupported command: {command}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments, build configuration, and execute a command."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    config = build_experiment_config(namespace)
    return dispatch(namespace.command, config)
