# Train-only weighted ridge and MLP baselines

This attempt uses `datasets/train.parquet` exclusively. The script contains no
path to `valid.parquet`, `valid_mask.parquet`, or the generated validation
subsets.

## Experimental boundary

The training file is split by complete sequence using seed `20261102`:

- 128 sequences form an internal holdout.
- Every remaining sequence contributes 64 uniformly sampled rows from steps
  99–19,999 to fitting.
- All 19,901 eligible rows of every holdout sequence are used for evaluation.

The holdout is part of the supplied training dataset. Its score is useful for
comparing train-only attempts but is not directly comparable to the competition's
masked validation or hidden-test scores.

## Shared preprocessing

Feature means and standard deviations are fitted only on sampled fitting rows.
Both targets are clipped to `[-2, 2]`. Target `k` uses
`abs(clipped_target_k)` as its sample weight, matching the competition metric's
target-specific emphasis. Separate models are fitted for `t0` and `t1` because
their row weights differ.

## Models

The weighted ridge model is a standardized linear reference with L2 penalty
`alpha=1000`.

The small MLP uses two hidden layers of 32 and 16 ReLU units. It is trained for
12 epochs with Adam and batches of 4,096 rows. It deliberately sees the same raw
112 features as ridge: no temporal features, feature selection, or validation
information is included.

## Run

From the starter-pack root with the virtual environment active:

```bash
python -m pip install -r requirements-train.txt
python scripts/train_static_baselines.py
python scripts/verify_static_baselines.py
python scripts/build_static_submissions.py
```

Generated NumPy artifacts are written under `artifacts/train_only_static_v1/`
and remain ignored by Git. The tracked experiment manifest and metrics are
written under `experiments/models/train_only_static_v1/`.

`src/static_models.py` implements competition-style, row-at-a-time inference in
NumPy. These static models update no temporal state, but they validate ordering,
return `None` during warm-up, and reset their sequence bookkeeping correctly.

Standalone attempts are under `solutions/train_only_static_v1/`. The packaging
script combines each `solution.py` with its corresponding `model.npz` at the ZIP
root and checks archive integrity and the 20 MiB limit.

## Interpretation

Ridge asks how much target signal is captured by a weighted linear combination
of the current row. The MLP asks how much additional signal is available through
small nonlinear feature interactions. Neither model uses history. Their gap is
therefore evidence about nonlinearity in the current observation, not about the
value of recurrence.
