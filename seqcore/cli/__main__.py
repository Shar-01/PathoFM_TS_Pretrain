"""Module execution support for ``python -m seqcore.cli``."""
from __future__ import annotations

from seqcore.cli.entrypoint import main

if __name__ == "__main__":
    raise SystemExit(main())
