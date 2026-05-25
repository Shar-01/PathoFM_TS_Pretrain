"""Small state containers for epoch-level bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RunningLossTotals:
    """Accumulate branch-wise scalar losses over batches."""

    masked: float = 0.0
    projected: float = 0.0
    table: float = 0.0
    steps: int = 0

    def update(self, masked: torch.Tensor, projected: torch.Tensor, table: torch.Tensor) -> None:
        self.masked += masked.item()
        self.projected += projected.item()
        self.table += table.item()
        self.steps += 1

    def averages(
        self,
        masked_enabled: bool = True,
        projected_enabled: bool = True,
        table_enabled: bool = True,
    ) -> tuple[float, float, float]:
        return (
            self.masked / self.steps if masked_enabled else 0.0,
            self.projected / self.steps if projected_enabled else 0.0,
            self.table / self.steps if table_enabled else 0.0,
        )


@dataclass
class PredictionBuffers:
    """Hold flattened test predictions and targets before metric aggregation."""

    y_true_rec: list[np.ndarray] = field(default_factory=list)
    y_pred_rec: list[np.ndarray] = field(default_factory=list)
    y_true_fc: list[np.ndarray] = field(default_factory=list)
    y_pred_fc: list[np.ndarray] = field(default_factory=list)
    y_true_table: list[np.ndarray] = field(default_factory=list)
    y_pred_table: list[np.ndarray] = field(default_factory=list)

    def append_reconstruction(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        self.y_true_rec.append(target.cpu().numpy().reshape(-1))
        self.y_pred_rec.append(prediction.cpu().numpy().reshape(-1))

    def append_forecasting(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        self.y_true_fc.append(target.cpu().numpy().reshape(-1))
        self.y_pred_fc.append(prediction.cpu().numpy().reshape(-1))

    def append_table(self, target: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor) -> None:
        self.y_true_table.append(target[mask].cpu().numpy().reshape(-1))
        self.y_pred_table.append(prediction[mask].cpu().numpy().reshape(-1))

    def append_empty_table_placeholder(self) -> None:
        self.y_true_table.append(np.array([0.0]))
        self.y_pred_table.append(np.array([0.0]))

    @staticmethod
    def _concatenate_or_empty(values: list[np.ndarray]) -> np.ndarray:
        """Return concatenated values, or an empty array when a path was skipped."""
        if not values:
            return np.array([])
        return np.concatenate(values)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "y_true_rec": self._concatenate_or_empty(self.y_true_rec),
            "y_pred_rec": self._concatenate_or_empty(self.y_pred_rec),
            "y_true_fc": self._concatenate_or_empty(self.y_true_fc),
            "y_pred_fc": self._concatenate_or_empty(self.y_pred_fc),
            "y_true_table": self._concatenate_or_empty(self.y_true_table),
            "y_pred_table": self._concatenate_or_empty(self.y_pred_table),
        }
