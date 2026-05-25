"""Early-stopping state management."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    """Track validation improvement and patience."""

    patience: int
    min_delta: float = 1e-4
    best_value: float = math.inf
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        """Update state and return whether the value improved."""
        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    def should_stop(self) -> bool:
        """Return whether patience has been exhausted."""
        return self.bad_epochs >= self.patience
