"""Console reporting helpers."""
from __future__ import annotations


def print_epoch_summary(
    epoch: int,
    epochs: int,
    train_mae: float,
    val_mae: float,
    train_mf: float,
    val_mf: float,
    train_table: float,
    improved: bool,
) -> None:
    """Print a compact epoch summary."""
    print(
        f"Ep {epoch:03}/{epochs} | masked {train_mae:.4f}/{val_mae:.4f}  "
        f"projected {train_mf:.4f}/{val_mf:.4f}  table {train_table:.4f}  {'*' if improved else ''}"
    )


def print_test_report(masked_value: float, projected_value: float, extra: dict[str, float]) -> None:
    """Print the final test report."""
    print("=" * 60)
    if "rec_rmse" in extra:
        print("Masked path:")
        print(f"  error : {masked_value:.4f}")
        print(f"  RMSE  : {extra['rec_rmse']:.4f}")
        print(f"  NRMSE : {extra['rec_nrmse']:.4f}")
    else:
        print("Masked path: skipped")

    if "fc_rmse" in extra:
        print("\nProjection path:")
        print(f"  error : {projected_value:.4f}")
        print(f"  RMSE  : {extra['fc_rmse']:.4f}")
        print(f"  sMAPE : {extra['fc_smape']:.2f}%")
        print(f"  R2    : {extra['fc_r2']:.3f}")
        print(f"  r     : {extra['fc_r']:.3f}")
    else:
        print("\nProjection path: skipped")

    if "table_error" in extra:
        print("\nTable path:")
        print(f"  error : {extra['table_error']:.4f}")
        print(f"  RMSE  : {extra['table_rmse']:.4f}")
        print(f"  NRMSE : {extra['table_nrmse']:.4f}")
    else:
        print("\nTable path: skipped")
