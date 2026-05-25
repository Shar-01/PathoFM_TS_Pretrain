"""Command-line option groups for training experiments."""
from __future__ import annotations

from seqcore.cli.specs import OptionGroupSpec, OptionSpec

EXPERIMENT_OPTIONS = OptionGroupSpec(
    title="Experiment",
    description="Top-level run identity and persistence options.",
    options=(
        OptionSpec(("--seed",), {"type": int, "default": None, "help": "Random seed."}),
        OptionSpec(("--save-path",), {"type": str, "default": None, "help": "Path for the best saved state."}),
    ),
)

DATA_OPTIONS = OptionGroupSpec(
    title="Data",
    description="Dataset discovery, split, window, and loader options.",
    options=(
        OptionSpec(("--data-dir",), {"type": str, "default": None, "help": "Directory containing temporal CSV files."}),
        OptionSpec(("--batch-size",), {"type": int, "default": None, "help": "Mini-batch size."}),
        OptionSpec(("--num-workers",), {"type": int, "default": None, "help": "DataLoader worker count."}),
        OptionSpec(("--past-len",), {"type": int, "default": None, "help": "Number of past time steps."}),
        OptionSpec(("--future-len",), {"type": int, "default": None, "help": "Number of future time steps."}),
        OptionSpec(("--stride",), {"type": int, "default": None, "help": "Sliding-window stride."}),
        OptionSpec(("--n-test-subjects",), {"type": int, "default": None, "help": "Number of held-out test subjects."}),
        OptionSpec(("--n-val-subjects",), {"type": int, "default": None, "help": "Number of validation subjects."}),
    ),
)

MODEL_OPTIONS = OptionGroupSpec(
    title="Model",
    description="Shared-stack architecture options.",
    options=(
        OptionSpec(("--model-dim", "--embed-dim"), {"dest": "embed_dim", "type": int, "default": None, "help": "Token adapter dimension."}),
        OptionSpec(("--depth",), {"type": int, "default": None, "help": "Number of shared stack layers."}),
        OptionSpec(("--heads",), {"type": int, "default": None, "help": "Number of attention heads."}),
    ),
)

OPTIMIZATION_OPTIONS = OptionGroupSpec(
    title="Optimization",
    description="Optimizer, scheduler, clipping, and early-stopping options.",
    options=(
        OptionSpec(("--epochs",), {"type": int, "default": None, "help": "Maximum number of training epochs."}),
        OptionSpec(("--patience",), {"type": int, "default": None, "help": "Early-stopping patience."}),
        OptionSpec(("--learning-rate",), {"type": float, "default": None, "help": "AdamW learning rate."}),
        OptionSpec(("--weight-decay",), {"type": float, "default": None, "help": "AdamW weight decay."}),
        OptionSpec(("--grad-clip-norm",), {"type": float, "default": None, "help": "Gradient clipping norm."}),
        OptionSpec(("--early-stop-min-delta",), {"type": float, "default": None, "help": "Minimum validation improvement."}),
    ),
)

OBJECTIVE_OPTIONS = OptionGroupSpec(
    title="Objectives",
    description="Transform and objective-weight options.",
    options=(
        OptionSpec(("--mask-ratio",), {"type": float, "default": None, "help": "Local transform ratio."}),
        OptionSpec(("--weight-mask",), {"dest": "weight_masked", "type": float, "default": None, "help": "Masked path weight."}),
        OptionSpec(("--weight-projection",), {"dest": "weight_projected", "type": float, "default": None, "help": "Projection path weight."}),
        OptionSpec(("--weight-table",), {"dest": "weight_table", "type": float, "default": None, "help": "Table path weight; set 0 to disable."}),
    ),
)

OPTION_GROUPS = (
    EXPERIMENT_OPTIONS,
    DATA_OPTIONS,
    MODEL_OPTIONS,
    OPTIMIZATION_OPTIONS,
    OBJECTIVE_OPTIONS,
)
