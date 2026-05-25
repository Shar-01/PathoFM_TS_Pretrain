"""State persistence helpers."""
from __future__ import annotations

import torch

from seqcore.models import SequenceUnit


def save_unit_state(model: SequenceUnit, path: str) -> None:
    """Save a unit state dictionary."""
    torch.save(model.state_dict(), path)


def load_unit_state(model: SequenceUnit, path: str, device: torch.device) -> None:
    """Load a unit state dictionary onto ``device``."""
    model.load_state_dict(torch.load(path, map_location=device))


save_model_state = save_unit_state
load_model_state = load_unit_state
