from __future__ import annotations

import math

import pytest
import torch

import main
from seqcore.cli import build_default_config, build_experiment_config, build_parser
from seqcore.cli.entrypoint import dispatch, main as cli_main
from seqcore.config import DataConfig, ExperimentConfig, ModelConfig, ObjectiveConfig, OptimizationConfig
from seqcore.data.factories import build_datasets, build_loaders
from seqcore.experiment.builders import build_model, build_optimizer, build_scheduler
from seqcore.experiment.checkpointing import load_model_state, save_model_state
from seqcore.experiment.early_stopping import EarlyStoppingState
from seqcore.experiment.runner import infer_input_dim, run_experiment, run_final_test, run_training_loop
from seqcore.experiment.scoring import validation_total
from seqcore.reporting.console import print_epoch_summary, print_test_report
from seqcore.utils.device import get_default_device
from seqcore.utils.reproducibility import set_seed


def small_experiment_config(data_dir: str, save_path: str, *, epochs: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        seed=123,
        save_path=save_path,
        data=DataConfig(
            data_dir=data_dir,
            batch_size=2,
            num_workers=0,
            past_len=50,
            future_len=50,
            stride=100,
            n_test_subjects=1,
            n_val_subjects=1,
        ),
        model=ModelConfig(embed_dim=8, depth=1, heads=2),
        optim=OptimizationConfig(epochs=epochs, patience=2, learning_rate=1e-3, weight_decay=0.0),
        objective=ObjectiveConfig(mask_ratio=0.7, weight_masked=1.0, weight_projected=1.0, weight_table=0.0),
    )


def cli_args_for_small_run(data_dir: str, save_path: str, *, epochs: int = 1) -> list[str]:
    return [
        "train",
        "--data-dir",
        data_dir,
        "--save-path",
        save_path,
        "--seed",
        "123",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--past-len",
        "50",
        "--future-len",
        "50",
        "--stride",
        "100",
        "--n-test-subjects",
        "1",
        "--n-val-subjects",
        "1",
        "--model-dim",
        "8",
        "--depth",
        "1",
        "--heads",
        "2",
        "--epochs",
        str(epochs),
        "--patience",
        "2",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0",
        "--weight-table",
        "0.0",
    ]


def test_config_objects_main_alias_and_cli_config_builder() -> None:
    config = ExperimentConfig()
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.optim, OptimizationConfig)
    assert isinstance(config.objective, ObjectiveConfig)

    assert main.make_config() == build_default_config()

    parser = build_parser(prog="sequence-lab")
    namespace = parser.parse_args(
        [
            "train",
            "--data-dir",
            "/tmp/gait-data",
            "--save-path",
            "checkpoint.pt",
            "--seed",
            "77",
            "--batch-size",
            "16",
            "--num-workers",
            "0",
            "--past-len",
            "25",
            "--future-len",
            "75",
            "--stride",
            "10",
            "--n-test-subjects",
            "2",
            "--n-val-subjects",
            "3",
            "--embed-dim",
            "32",
            "--depth",
            "2",
            "--heads",
            "4",
            "--epochs",
            "5",
            "--patience",
            "1",
            "--learning-rate",
            "0.002",
            "--weight-decay",
            "0.03",
            "--grad-clip-norm",
            "0.5",
            "--early-stop-min-delta",
            "0.01",
            "--mask-ratio",
            "0.6",
            "--weight-mask",
            "1.5",
            "--weight-projection",
            "2.5",
            "--weight-table",
            "0.0",
        ]
    )
    cli_config = build_experiment_config(namespace)
    assert cli_config.seed == 77
    assert cli_config.save_path == "checkpoint.pt"
    assert cli_config.data.data_dir == "/tmp/gait-data"
    assert cli_config.data.batch_size == 16
    assert cli_config.data.num_workers == 0
    assert cli_config.data.past_len == 25
    assert cli_config.data.future_len == 75
    assert cli_config.data.stride == 10
    assert cli_config.data.n_test_subjects == 2
    assert cli_config.data.n_val_subjects == 3
    assert cli_config.model.embed_dim == 32
    assert cli_config.model.depth == 2
    assert cli_config.model.heads == 4
    assert cli_config.optim.epochs == 5
    assert cli_config.optim.patience == 1
    assert cli_config.optim.learning_rate == pytest.approx(0.002)
    assert cli_config.optim.weight_decay == pytest.approx(0.03)
    assert cli_config.optim.grad_clip_norm == pytest.approx(0.5)
    assert cli_config.optim.early_stop_min_delta == pytest.approx(0.01)
    assert cli_config.objective.mask_ratio == pytest.approx(0.6)
    assert cli_config.objective.weight_masked == pytest.approx(1.5)
    assert cli_config.objective.weight_projected == pytest.approx(2.5)
    assert cli_config.objective.weight_table == pytest.approx(0.0)


def test_parser_allows_omitted_train_command() -> None:
    parser = build_parser()
    namespace = parser.parse_args(["--epochs", "3"])
    assert namespace.command == "train"
    assert namespace.epochs == 3


def test_cli_dispatch_rejects_unknown_commands() -> None:
    with pytest.raises(ValueError, match="Unsupported command"):
        dispatch("unknown", build_default_config())


def test_reproducibility_and_device_helpers() -> None:
    set_seed(999)
    first = torch.rand(3)
    set_seed(999)
    second = torch.rand(3)
    assert torch.equal(first, second)
    assert get_default_device().type in {"cpu", "cuda"}


def test_makers_storage_and_infer_input_dim(tmp_path, synthetic_data_dir) -> None:
    config = small_experiment_config(str(synthetic_data_dir), str(tmp_path / "best.pt"))
    train_ds, val_ds, test_ds = build_datasets(config)
    assert infer_input_dim(train_ds) == 4
    train_loader, val_loader, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    assert next(iter(train_loader)).shape[-1] == 4
    assert next(iter(val_loader)).shape[-1] == 4
    assert next(iter(test_loader)).shape[-1] == 4

    device = torch.device("cpu")
    model = build_model(config, input_dim=4, device=device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert scheduler.T_max == config.optim.epochs

    save_model_state(model, config.save_path)
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    load_model_state(model, config.save_path, device)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original_state[key])


def test_guard_and_validation_scoring() -> None:
    stopper = EarlyStoppingState(patience=2, min_delta=0.1)
    assert stopper.update(10.0) is True
    assert stopper.best_value == 10.0
    assert stopper.update(9.95) is False
    assert stopper.bad_epochs == 1
    assert stopper.update(9.0) is True
    assert stopper.bad_epochs == 0
    assert stopper.update(9.05) is False
    assert stopper.update(9.06) is False
    assert stopper.should_stop() is True
    assert validation_total(1.0, 2.0, 3.0) == 6.0


def test_reporting_output_formats(capsys) -> None:
    print_epoch_summary(1, 2, 0.1, 0.2, 0.3, 0.4, 0.5, True)
    epoch_out = capsys.readouterr().out
    assert "Ep 001/2" in epoch_out
    assert "*" in epoch_out

    metrics = {
        "rec_rmse": 1.0,
        "rec_nrmse": 2.0,
        "fc_rmse": 3.0,
        "fc_smape": 4.0,
        "fc_r2": 0.5,
        "fc_r": 0.6,
        "table_error": 7.0,
        "table_rmse": 8.0,
        "table_nrmse": 9.0,
    }
    print_test_report(0.1, 0.2, metrics)
    report_out = capsys.readouterr().out
    assert "Masked path:" in report_out
    assert "Projection path:" in report_out
    assert "Table path:" in report_out


def test_training_loop_and_final_test_with_synthetic_data(tmp_path, synthetic_data_dir, capsys) -> None:
    config = small_experiment_config(str(synthetic_data_dir), str(tmp_path / "best_loop.pt"), epochs=1)
    set_seed(config.seed)
    device = torch.device("cpu")
    train_ds, val_ds, test_ds = build_datasets(config)
    train_loader, val_loader, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    model = build_model(config, input_dim=infer_input_dim(train_ds), device=device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    run_training_loop(config, model, train_loader, val_loader, optimizer, scheduler, device)
    assert (tmp_path / "best_loop.pt").exists()
    run_final_test(config, model, test_loader, device)
    output = capsys.readouterr().out
    assert "Ep 001/1" in output
    assert "Masked path:" in output


def test_complete_run_experiment_smoke_test(tmp_path, synthetic_data_dir, capsys) -> None:
    config = small_experiment_config(str(synthetic_data_dir), str(tmp_path / "best_full.pt"), epochs=1)
    run_experiment(config)
    assert (tmp_path / "best_full.pt").exists()
    output = capsys.readouterr().out
    assert "Ep 001/1" in output
    assert "Projection path:" in output


def test_complete_cli_smoke_test(tmp_path, synthetic_data_dir, capsys) -> None:
    save_path = tmp_path / "best_cli.pt"
    exit_code = cli_main(cli_args_for_small_run(str(synthetic_data_dir), str(save_path), epochs=1))
    assert exit_code == 0
    assert save_path.exists()
    output = capsys.readouterr().out
    assert "Ep 001/1" in output
    assert "Projection path:" in output


def test_training_loop_early_stop_branch(tmp_path, synthetic_data_dir, capsys) -> None:
    config = small_experiment_config(str(synthetic_data_dir), str(tmp_path / "best_early.pt"), epochs=3)
    config = ExperimentConfig(
        seed=config.seed,
        save_path=config.save_path,
        data=config.data,
        model=config.model,
        optim=OptimizationConfig(
            epochs=3,
            patience=0,
            learning_rate=config.optim.learning_rate,
            weight_decay=config.optim.weight_decay,
            grad_clip_norm=config.optim.grad_clip_norm,
            early_stop_min_delta=config.optim.early_stop_min_delta,
        ),
        objective=config.objective,
    )
    set_seed(config.seed)
    device = torch.device("cpu")
    train_ds, val_ds, _test_ds = build_datasets(config)
    train_loader, val_loader, _test_loader = build_loaders(config, train_ds, val_ds, _test_ds)
    model = build_model(config, input_dim=infer_input_dim(train_ds), device=device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    run_training_loop(config, model, train_loader, val_loader, optimizer, scheduler, device)
    output = capsys.readouterr().out
    assert "Early stop at epoch" in output
    assert (tmp_path / "best_early.pt").exists()


def test_main_entrypoint_delegates_to_cli(monkeypatch) -> None:
    seen = {}

    def fake_cli(argv=None):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(main, "cli_main", fake_cli)
    assert main.main(["train", "--epochs", "1"]) == 0
    assert seen["argv"] == ["train", "--epochs", "1"]


def test_main_module_guard_invokes_cli_when_executed_as_script(monkeypatch) -> None:
    import runpy

    monkeypatch.setattr("sys.argv", ["main.py", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("main", run_name="__main__")
    assert exc_info.value.code == 0


def test_package_cli_module_guard_invokes_help(monkeypatch) -> None:
    import runpy

    monkeypatch.setattr("sys.argv", ["python -m seqcore.cli", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("seqcore.cli.__main__", run_name="__main__")
    assert exc_info.value.code == 0
