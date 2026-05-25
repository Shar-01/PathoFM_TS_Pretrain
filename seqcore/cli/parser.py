"""Argparse construction for the sequence pipeline command-line interface."""
from __future__ import annotations

import argparse
from typing import Optional

from seqcore.cli.options import OPTION_GROUPS
from seqcore.cli.specs import OptionGroupSpec

PROGRAM_DESCRIPTION = "Train and evaluate the modular sequence representation model."
TRAIN_ALIAS = "train"


def add_option_group(parser: argparse.ArgumentParser, spec: OptionGroupSpec) -> None:
    """Attach a declarative option group to an argparse parser."""
    group = parser.add_argument_group(title=spec.title, description=spec.description)
    for option in spec.options:
        group.add_argument(*option.flags, **dict(option.kwargs))


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    """Build the CLI parser used by both ``main.py`` and ``python -m seqcore.cli``."""
    parser = argparse.ArgumentParser(prog=prog, description=PROGRAM_DESCRIPTION)
    parser.add_argument(
        "command",
        nargs="?",
        default=TRAIN_ALIAS,
        choices=(TRAIN_ALIAS,),
        help="Command to run. Currently only 'train' is supported; omitted defaults to 'train'.",
    )
    for group_spec in OPTION_GROUPS:
        add_option_group(parser, group_spec)
    return parser
