"""Evaluation metrics for masked, projected, and table paths."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nrmse(rmse_value: float, target: np.ndarray) -> float:
    """Normalize RMSE by target range using the original epsilon."""
    return float(rmse_value / (target.max() - target.min() + 1e-8))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error with the original denominator convention."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def has_metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Return whether a metric pair contains values from an available path."""
    return y_true.size > 0 and y_pred.size > 0


def compute_test_metrics(
    y_true_rec: np.ndarray,
    y_pred_rec: np.ndarray,
    y_true_fc: np.ndarray,
    y_pred_fc: np.ndarray,
    y_true_table: np.ndarray,
    y_pred_table: np.ndarray,
) -> dict[str, float]:
    """Compute all scalar metrics emitted by the original test loop."""
    metrics: dict[str, float] = {}

    if has_metric_values(y_true_rec, y_pred_rec):
        rmse_rec = rmse(y_true_rec, y_pred_rec)
        metrics.update(
            {
                "rec_rmse": rmse_rec,
                "rec_nrmse": nrmse(rmse_rec, y_true_rec),
            }
        )

    if has_metric_values(y_true_fc, y_pred_fc):
        rmse_fc = rmse(y_true_fc, y_pred_fc)
        metrics.update(
            {
                "fc_rmse": rmse_fc,
                "fc_mape": mape(y_true_fc, y_pred_fc),
                "fc_smape": smape(y_true_fc, y_pred_fc),
                "fc_r2": float(r2_score(y_true_fc, y_pred_fc)),
                "fc_r": pearson_r(y_true_fc, y_pred_fc),
            }
        )

    if has_metric_values(y_true_table, y_pred_table):
        rmse_table = rmse(y_true_table, y_pred_table)
        metrics.update(
            {
                "table_error": float(np.mean(np.abs(y_true_table - y_pred_table))),
                "table_rmse": rmse_table,
                "table_nrmse": nrmse(rmse_table, y_true_table),
            }
        )

    return metrics
