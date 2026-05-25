# -*- coding: utf-8 -*-
"""Abstract script shim for the sequence pipeline command-line interface."""
from __future__ import annotations

from typing import Optional, Sequence

from seqcore.cli import build_default_config, main as cli_main

make_config = build_default_config


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI entry point."""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
