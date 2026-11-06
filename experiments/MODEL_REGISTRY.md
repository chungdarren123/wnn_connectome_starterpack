# Model registry

This document records every model family trained during development of the
first submission. The JSON files under `experiments/models/` are the canonical
machine-readable records; this page is the human-readable index.

## Metric and comparison rules

All reported scores are Global Weighted Pearson (WP), averaged across `t0` and
`t1`. Unless explicitly noted, development and confirmation scores use every
eligible row after the 99-step warm-up. The final supplied-validation score
uses `need_prediction AND is_scored`.

Scores should only be compared within the same evaluation partition. The
initial static experiment used a different 128-sequence holdout and is listed
separately.

| Partition | Sequences | Purpose |
|---|---:|---|
| Fit | 9,547 | Model development training |
| Development | 530 | Architecture, feature and epoch selection |
| Confirmation | 530 | One-time check after the design was frozen |
| All training | 10,607 | Final refit: fit + development + confirmation |
| Supplied validation | 1,873 | One final comparison after refit |

The final refit never trained on `valid.parquet`; it was evaluated once after
all architecture and training choices were frozen.

## Supplied reference baseline

The supplied vanilla GRU was not trained in this work. Its documented complete
validation score is **0.617052**. It is included only as a reference point.

## Initial static benchmark

This first experiment used 10,479 fit sequences, 64 sampled rows per sequence,
and a separate 128-sequence holdout. Its scores are not directly comparable to
the later 530-sequence development results.

| Model | Features | Holdout t0 | Holdout t1 | Holdout WP |
|---|---|---:|---:|---:|
| Weighted ridge | Raw 112 | 0.325371 | 0.325145 | 0.325258 |
| Small MLP (32, 16) | Raw 112 | 0.353913 | 0.362670 | 0.358292 |

Record: `experiments/models/train_only_static_v1/`.

## Causal feature ablations

Ridge and MLP models used the standardized 9,547/530/530 sequence split. Each
fit sequence contributed 64 sampled eligible rows; development used all
10,547,530 eligible rows.

| Feature family | Dimensions | Ridge WP | MLP WP | MLP delta vs raw |
|---|---:|---:|---:|---:|
| Raw measurements | 112 | 0.326441 | 0.354872 | — |
| Raw + causal first differences | 224 | 0.333510 | 0.356093 | +0.001222 |
| Raw + EMA half-lives 8 and 64 | 336 | 0.370890 | **0.380101** | +0.025229 |
| Raw + repeated-value indicators | 224 | 0.328946 | 0.353589 | -0.001283 |

Decision: EMA features were the strongest static representation, while first
differences and repetition flags added little. Explicit EMA inputs were still
screened later with a recurrent model rather than assumed to transfer.

Record: `experiments/models/train_only_causal_ablation_v1/`.

## Compact GRU pilot: output-head design

Both models used 2,048 fit sequences for one epoch, a 64-value recurrent state,
1,024-step truncated backpropagation, and the full 530-sequence development
partition.

| Model | Parameters | Development t0 | Development t1 | Development WP |
|---|---:|---:|---:|---:|
| Shared `64 -> 32 -> 2` head | 34,338 | 0.342904 | 0.348519 | 0.345712 |
| Two target-specific `64 -> 16 -> 1` heads | 34,306 | 0.347050 | 0.353697 | **0.350374** |

Decision: retain target-specific heads.

Record: `experiments/models/train_only_compact_gru_v1/`.

## GRU architecture screen

The screen used the same 2,048 fit sequences for one epoch.

| Model | Inputs | Hidden state | Parameters | Development WP |
|---|---:|---:|---:|---:|
| Raw H64 | 112 | 64 | 34,306 | 0.350079 |
| EMA H64 | 336 | 64 | 48,642 | 0.332075 |
| Raw H128 | 112 | 128 | 117,698 | **0.358408** |

Decision: explicit EMA features hurt the recurrent model and were rejected.
H128 won the short screen, so it advanced to a full-data comparison alongside
H64.

Record: `experiments/models/train_only_gru_architecture_screen_v1/`.

## Full split GRU training

These models trained on all 9,547 fit sequences. Development contains 530
complete sequences; confirmation remained unread during epoch selection.

| Architecture / seed | Epoch | Learning rate | Training loss | Development WP |
|---|---:|---:|---:|---:|
| H64 / 20261102 | 1 | 1e-3 | 0.798658 | 0.374926 |
| H64 / 20261102 | 2 | 3e-4 | 0.773539 | 0.389858 |
| H64 / 20261102 | 3 | 1e-4 | 0.764917 | **0.392336** |
| H128 / 20261104 | 1 | 1e-3 | 0.797389 | 0.373874 |
| H128 / 20261104 | 2 | 3e-4 | 0.771648 | **0.387339** |
| H64 / 20261105 | 1 | 1e-3 | 0.798675 | 0.376721 |
| H64 / 20261105 | 2 | 3e-4 | 0.773493 | 0.385231 |
| H64 / 20261105 | 3 | 1e-4 | 0.765058 | **0.391512** |

Decision: the short H128 advantage did not survive full training. H64 was both
more accurate and smaller. Epoch 3 and the three-rate schedule were retained.
The second H64 seed confirmed that the result was reproducible.

Records:

- `experiments/models/train_only_compact_gru_full_v1/`
- `experiments/models/train_only_gru_h128_seed20261104_v1/`
- `experiments/models/train_only_gru_h64_seed20261105_v1/`

## Frozen two-seed ensemble

The ensemble is a fixed arithmetic mean. Its weight was not tuned.

| Candidate | Development WP | Confirmation WP |
|---|---:|---:|
| H64 seed 20261102 | 0.392336 | 0.418306 |
| H64 seed 20261105 | 0.391512 | 0.416950 |
| 50/50 ensemble | **0.396738** | **0.422444** |

Decision: freeze the equal-weight two-seed ensemble. Confirmation was opened
once after this choice and was not used to alter the design.

Record: `experiments/models/train_only_gru_h64_two_seed_v1/`.

## Final refit and supplied validation

Both frozen H64 models were reinitialized with their original seeds and trained
for three epochs on all 10,607 training sequences. No development checkpoint
selection was performed during the final refit.

| Seed | Epoch 1 loss | Epoch 2 loss | Epoch 3 loss |
|---|---:|---:|---:|
| 20261102 | 0.797337 | 0.772797 | 0.765252 |
| 20261105 | 0.798069 | 0.772250 | 0.764818 |

One final evaluation was then run on all 1,873 supplied validation sequences
and 3,528,273 scored rows.

| Candidate | Validation t0 | Validation t1 | Validation WP |
|---|---:|---:|---:|
| H64 seed 20261102 | 0.657512 | 0.663947 | 0.660729 |
| H64 seed 20261105 | 0.656564 | 0.664605 | 0.660585 |
| 50/50 ensemble | **0.659805** | **0.667161** | **0.663483** |

The ensemble was selected for the first official submission package.

Record: `experiments/models/final_gru_h64_two_seed_all_train_v1/`.

## Final submission artifact

Local path (ignored by Git):

`artifacts/submissions/final_gru_h64_two_seed_all_train_v1/h64_gru_two_seed_ensemble_full_train.zip`

- Contents: `solution.py`, `model_1.onnx`, `model_2.onnx`
- ZIP bytes: 259,987
- ZIP SHA-256:
  `793d92ca53606cdb9ae70f8c3343dd44941808045a5cf30ea53848d65e89268c`
- ONNX parity, ZIP integrity, 99-row warm-up, state reset and prediction-shape
  checks passed.

The ZIP, checkpoints and ONNX files are generated artifacts and remain outside
Git. Their hashes and provenance are tracked in `submission_export.json` and
the corresponding selection and metrics records.

## Reproduction map

| Stage | Primary script |
|---|---|
| Static ridge/MLP baseline | `scripts/train_static_baselines.py` |
| Causal feature ablations | `scripts/train_causal_feature_ablations.py` |
| Shared vs target-specific pilot | `scripts/train_compact_gru.py` |
| EMA/H64/H128 screen | `scripts/screen_gru_ema_hidden.py` |
| Full H64 seed 20261102 | `scripts/train_compact_gru_full.py`, `scripts/refine_compact_gru.py` |
| Full H128 | `scripts/train_gru_h128_full.py`, `scripts/resume_gru_h128_epoch2.py` |
| Full H64 seed 20261105 | `scripts/train_gru_h64_second_seed.py` |
| Seed ensemble comparison | `scripts/evaluate_h64_two_seed_ensemble.py` |
| Final all-training refit | `scripts/train_final_h64_two_seed_all_train.py` |
| One-shot validation | `scripts/evaluate_final_h64_on_validation.py` |
| Final ONNX/ZIP export | `scripts/package_final_h64_submission.py` |

Use `requirements-train.txt` for the training environment. The canonical
manifests contain seeds, sequence counts, feature definitions, schedules and
artifact hashes.
