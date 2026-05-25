# PATHOFM

> Codebase for **"On the Role of Inductive Bias in Time-Series Pretraining:
> A Case Study in Learning Generalizable Representations for Clinical Time Series"**

This repository contains the research code for the PATHOFM study. The paper studies how different
pretraining objectives shape representations for clinical time-series data,
with a case study in pathological gait analysis for spinal cord injury.

Paper link: `https://openreview.net/forum?id=axSCiaQE9l`

Code link: `https://github.com/Shar-01/PathoFM_TS_Pretrain`

---

## At A Glance

| Item | Summary |
| --- | --- |
| Paper focus | Inductive bias in clinical time-series pretraining |
| Case study | Pathological gait analysis for spinal cord injury |
| Evaluation style | Subject-aware transfer to classification and regression tasks |
---

## Paper In A Nutshell

Clinical time-series learning often has to work with limited cohorts, noisy
measurements, missing values, protocol drift, and strong subject-to-subject
variation. The paper asks a practical question:

**Which pretraining biases help a time-series model transfer well to
both classification and regression tasks under subject shift?**

The study compares objective families that encourage different representation
properties.

---

## Compact Flow

```text
CSV records
    |
    v
+------------+     +--------------+     +------------------+
| subjects   | --> | windows      | --> | normalized data  |
+------------+     +--------------+     +------------------+
                                                 |
                                                 v
                         +--------------------------------+
                         | sequence model                 |
                         |  - masked path                 |
                         |  - projection path             |
                         |  - table path                  |
                         +--------------------------------+
                                                 |
                                                 v
                         +--------------------------------+
                         | checkpointing + final metrics  |
                         +--------------------------------+
```
---

## File Surface

```text
seqcore/
  cli/          specs, options, parser, configuration, commands, entrypoint
  config/       small immutable config objects
  common/       shared aliases
  data/         constants, columns, files, subjects, windows, records,
                normalization, statistics, dataset, factories, selection
  models/       feature_encoder, position_encoding, token_layout, masking,
                sequence_encoder, prediction_head, query_objective,
                sequence_model, components
  training/     losses, query_tables, query_builders, branches, objective,
                batch_steps, epoch, metrics, state
  experiment/   builders, runtime, loop, checkpointing, early_stopping,
                scoring, final_eval, runner
  reporting/    compact console output
  utils/        device and reproducibility helpers
```

Compatibility wrappers remain at the repository root for older local scripts:

```text
main.py
dataset.py
model.py
training.py
utils.py
```

Prefer the package imports for new work:

```python
from seqcore.data import SequenceCorpus
from seqcore.models import SequenceUnit
from seqcore.training import masked_error, projection_error, build_query_table
```

---

## Install

Create and activate a Python environment, then install the package from the
repository root:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e .[dev]
```

The training code expects the scientific Python stack used by the paper,
including PyTorch, NumPy, pandas, and scikit-learn.

---

## Run

Default training entry point:

```bash
python main.py
```

Example training run:

```bash
python main.py train \
  --data-dir "D:\\Temporal_Project\\Data" \
  --save-path best_state.pt \
  --epochs 250 \
  --batch-size 128 \
  --patience 20 \
  --seed 42 \
  --mask-ratio 0.70 \
  --weight-mask 1.0 \
  --weight-projection 1.0 \
  --weight-table 1.0 \
  --model-dim 128 \
  --depth 8 \
  --heads 8
```

Package module:

```bash
python -m seqcore.cli train --data-dir /path/to/csvs --epochs 250
```

Installed console script:

```bash
sequence-lab train --data-dir /path/to/csvs
```

---

## Design Notes

| Principle | Implementation |
| --- | --- |
| Meaningful naming | Files use names like `subjects`, `windows`, `normalization`, `sequence_model`, `batch_steps`, and `checkpointing`. |
| Small call-throughs | Larger operations are split into one-purpose helpers |
| Reproducibility checks | Config objects, CLI wiring, checkpointing, and tests remain available. |

---

## Validation

Basic import and syntax checks:

```bash
python -m compileall -q .
```

Run the test suite after installing development dependencies:

```bash
pytest -q
```

If PyTorch is not installed in the active environment, the test suite will fail
at import time. Install the PyTorch build appropriate for your system before
running model or training tests.

---

## Citation

If you use any part of the paper or code, please cite us with the citations
below.

### Paper and Code

```bibtex
@article{dey2026pathofm,
  title   = {On the Role of Inductive Bias in Time-Series Pretraining: A Case Study Through a Foundation Model for Clinical Time Series},
  author  = {Dey, Sharmita and Paez-Granados, Diego},
  year    = {2026},
  note    = {Preprint},
  url     = {https://openreview.net/forum?id=axSCiaQE9l}
}
```

---

## Notes

- This repository is intended for research use. 
- If you use any part of the paper or code, please cite us with the citations
as mentioned above.
- Keep subject-level splits explicit when comparing results, because the paper's
  claims are about transfer under subject shift.
