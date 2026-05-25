from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from seqcore.models import SequenceUnit
from seqcore.training.epoch import eval_epoch, train_one_epoch
from seqcore.training.query_tables import build_query_table, build_legacy_query_table
from seqcore.training.losses import masked_error, projection_error
from seqcore.training.metrics import compute_test_metrics, mape, nrmse, pearson_r, rmse, smape
from seqcore.training.state import PredictionBuffers, RunningLossTotals
from seqcore.training.batch_steps import (
    ChannelLosses,
    compute_eval_table_loss,
    compute_masked_loss,
    compute_projection_loss,
    compute_train_table_loss,
    evaluate_step,
    train_step,
    weighted_total_loss,
)


def make_loader(batch_size: int = 2) -> DataLoader:
    torch.manual_seed(0)
    full = torch.randn(4, 100, 4)
    return DataLoader(full, batch_size=batch_size, shuffle=False)


def make_model() -> SequenceUnit:
    torch.manual_seed(1)
    return SequenceUnit(input_dim=4, embed_dim=8, depth=1, heads=2)


def test_losses_match_torch_mse_contract() -> None:
    recon = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 2.0], [5.0, 1.0]])
    mask = torch.tensor([[True, False], [True, True]])
    assert torch.equal(masked_error(recon, target, mask), torch.nn.functional.mse_loss(recon[mask], target[mask]))
    assert torch.equal(projection_error(recon, target), torch.nn.functional.mse_loss(recon, target))


def test_query_table_builders_shape_and_mask_one_query_row() -> None:
    torch.manual_seed(0)
    batch = torch.randn(5, 100, 4)
    table, row_type, query_mask = build_query_table(batch)
    assert table.shape == batch.shape
    assert row_type.shape == (5,)
    assert query_mask.shape == batch.shape
    assert int(row_type.sum().item()) == 1
    query_idx = int(torch.nonzero(row_type == 1, as_tuple=False)[0].item())
    assert query_mask[query_idx].all()
    assert not query_mask[row_type == 0].any()
    assert torch.equal(table[query_idx], torch.zeros_like(table[query_idx]))
    assert torch.equal(table[row_type == 0], batch[row_type == 0])

    old_table, old_row_type, old_query_mask = build_legacy_query_table(batch)
    assert old_table.shape == batch.shape
    assert old_row_type.shape == (5,)
    assert old_query_mask.shape == (100, 4)
    assert int(old_row_type.sum().item()) == 1


def test_running_totals_and_prediction_buffers() -> None:
    totals = RunningLossTotals()
    totals.update(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))
    totals.update(torch.tensor(3.0), torch.tensor(4.0), torch.tensor(5.0))
    assert totals.averages() == (2.0, 3.0, 4.0)
    assert totals.averages(table_enabled=False) == (2.0, 3.0, 0.0)

    buffers = PredictionBuffers()
    target = torch.tensor([[1.0, 2.0]])
    prediction = torch.tensor([[3.0, 4.0]])
    mask = torch.tensor([[True, False]])
    buffers.append_reconstruction(target, prediction)
    buffers.append_forecasting(target, prediction)
    buffers.append_table(target, prediction, mask)
    arrays = buffers.arrays()
    assert arrays["y_true_rec"].tolist() == [1.0, 2.0]
    assert arrays["y_pred_fc"].tolist() == [3.0, 4.0]
    assert arrays["y_true_table"].tolist() == [1.0]

    buffers_empty = PredictionBuffers()
    buffers_empty.append_reconstruction(target, prediction)
    buffers_empty.append_forecasting(target, prediction)
    buffers_empty.append_empty_table_placeholder()
    arrays_empty = buffers_empty.arrays()
    assert arrays_empty["y_true_table"].tolist() == [0.0]
    assert arrays_empty["y_pred_table"].tolist() == [0.0]


def test_metric_functions_and_test_metric_dict_are_finite() -> None:
    y_true = np.array([1.0, 2.0, 4.0, 8.0])
    y_pred = np.array([1.5, 1.5, 5.0, 7.0])
    assert rmse(y_true, y_pred) == pytest.approx(float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
    assert nrmse(2.0, y_true) == pytest.approx(2.0 / (8.0 - 1.0 + 1e-8))
    assert mape(y_true, y_pred) > 0
    assert smape(y_true, y_pred) > 0
    assert pearson_r(y_true, y_pred) == pytest.approx(float(np.corrcoef(y_true, y_pred)[0, 1]))

    metrics = compute_test_metrics(y_true, y_pred, y_true, y_pred, y_true, y_pred)
    expected_keys = {
        "rec_rmse",
        "rec_nrmse",
        "fc_rmse",
        "fc_mape",
        "fc_smape",
        "fc_r2",
        "fc_r",
        "table_error",
        "table_rmse",
        "table_nrmse",
    }
    assert set(metrics) == expected_keys
    assert all(math.isfinite(value) for value in metrics.values())


def test_batch_step_helpers_return_scalar_finite_losses() -> None:
    model = make_model()
    full = next(iter(make_loader(batch_size=2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_mae, recon = compute_masked_loss(model, full, mask_ratio=0.7)
    assert loss_mae.ndim == 0
    assert recon.shape == full.shape

    loss_mf, pred = compute_projection_loss(model, full)
    assert loss_mf.ndim == 0
    assert pred.shape == (2, 50, 4)

    loss_table_enabled = compute_train_table_loss(model, full, enabled=True)
    loss_table_disabled = compute_train_table_loss(model, full, enabled=False)
    assert torch.isfinite(loss_table_enabled)
    assert torch.equal(loss_table_disabled, torch.zeros_like(loss_table_disabled))

    eval_loss, table_output = compute_eval_table_loss(model, full, enabled=True)
    assert torch.isfinite(eval_loss)
    assert table_output is not None
    assert table_output.prediction.shape == table_output.target.shape
    assert table_output.mask.shape == table_output.target.shape

    disabled_eval_loss, disabled_output = compute_eval_table_loss(model, full, enabled=False)
    assert torch.equal(disabled_eval_loss, torch.zeros_like(disabled_eval_loss))
    assert disabled_output is None

    losses = ChannelLosses(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))
    assert torch.equal(weighted_total_loss(losses, 1.0, 2.0, 3.0), torch.tensor(14.0))

    params_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    batch_losses = train_step(model, full, optimizer, weight_table=0.0, mask_ratio=0.7)
    assert torch.isfinite(batch_losses.masked)
    assert torch.isfinite(batch_losses.projected)
    assert torch.equal(batch_losses.table, torch.zeros_like(batch_losses.table))
    params_after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(before, after) for before, after in zip(params_before, params_after))

    eval_losses, eval_recon, eval_pred, eval_table = evaluate_step(model, full, table=True)
    assert torch.isfinite(eval_losses.masked)
    assert eval_recon.shape == full.shape
    assert eval_pred.shape == (2, 50, 4)
    assert eval_table is not None


def test_missing_objective_paths_are_skipped_without_errors() -> None:
    model = torch.nn.Linear(1, 1)
    full = next(iter(make_loader(batch_size=2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = train_step(
        model,
        full,
        optimizer,
        weight_masked=1.0,
        weight_projected=1.0,
        weight_table=1.0,
    )
    assert torch.equal(losses.masked, torch.zeros_like(losses.masked))
    assert torch.equal(losses.projected, torch.zeros_like(losses.projected))
    assert torch.equal(losses.table, torch.zeros_like(losses.table))

    eval_losses, recon, pred_fut, table_output = evaluate_step(
        model,
        full,
        masked=True,
        projected=True,
        table=True,
    )
    assert torch.equal(eval_losses.masked, torch.zeros_like(eval_losses.masked))
    assert torch.equal(eval_losses.projected, torch.zeros_like(eval_losses.projected))
    assert torch.equal(eval_losses.table, torch.zeros_like(eval_losses.table))
    assert recon is None
    assert pred_fut is None
    assert table_output is None


def test_epoch_training_and_evaluation_train_and_test_modes() -> None:
    device = torch.device("cpu")
    model = make_model().to(device)
    loader = make_loader(batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tr_masked, tr_projected, tr_table = train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        weight_masked=1.0,
        weight_projected=1.0,
        weight_table=0.0,
        mask_ratio=0.7,
        grad_clip_norm=1.0,
    )
    assert math.isfinite(tr_masked)
    assert math.isfinite(tr_projected)
    assert tr_table == 0.0

    va_masked, va_projected, va_table = eval_epoch(model, loader, device, mode="train", table=False)
    assert math.isfinite(va_masked)
    assert math.isfinite(va_projected)
    assert va_table == 0.0

    te_masked, te_projected, te_table, metrics = eval_epoch(model, loader, device, mode="test", table=True)
    assert math.isfinite(te_masked)
    assert math.isfinite(te_projected)
    assert math.isfinite(te_table)
    assert "fc_smape" in metrics
    assert all(math.isfinite(value) for value in metrics.values())

    with pytest.raises(AssertionError):
        eval_epoch(model, loader, device, mode="invalid")
