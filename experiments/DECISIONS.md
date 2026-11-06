# Experiment decision log

This log explains the decisions behind the first submission. It complements
the numerical tables in [MODEL_REGISTRY.md](MODEL_REGISTRY.md).

## 1. Establish sequence-safe evaluation

The 10,607 training sequences were partitioned at sequence level:

- 9,547 fit sequences
- 530 development sequences
- 530 confirmation sequences

Rows from one sequence never crossed partitions. The supplied validation set
was excluded from architecture development. This avoided both row-level
leakage and repeated selection against the final local benchmark.

Decision: use development for iteration, reserve confirmation for one mature
candidate, and reserve supplied validation for one final comparison.

## 2. Establish inexpensive static references

Weighted ridge and small target-specific MLP models provided fast sanity
checks. The raw MLP outperformed ridge, showing that modest nonlinearity was
useful, but neither could represent long sequential context.

Decision: retain static models as diagnostics, then move to stateful sequence
models.

## 3. Ablate causal feature families

First differences, causal EMAs and repeated-value indicators were added one
family at a time. EMAs gave the only substantial static improvement: the MLP
rose from 0.354872 to 0.380101 development WP. First differences produced a
small gain, while repetition flags slightly hurt the MLP.

Decision: investigate EMA information in the GRU screen, but do not carry the
other engineered families forward.

## 4. Choose target-specific output heads

On a 2,048-sequence pilot, separate nonlinear heads scored 0.350374 versus
0.345712 for one shared two-output head.

Decision: freeze two independent `hidden -> 16 -> SiLU -> 1` heads.

## 5. Screen explicit smoothing and memory size

The one-epoch screen favored raw H128 (0.358408) over raw H64 (0.350079), but
explicit EMA H64 fell to 0.332075. The GRU could learn useful temporal
summaries directly; supplying the static EMA representation was counterproductive.

Decision: reject explicit EMA inputs and advance raw H64 and H128 to fuller
training.

## 6. Prefer H64 after full training

On all 9,547 fit sequences, H128 reached 0.387339 after two epochs. H64 reached
0.392336 after the third low-learning-rate epoch while using less than one
third as many parameters (34,306 versus 117,698).

Decision: select raw H64 and retain the three-rate schedule
`[1e-3, 3e-4, 1e-4]`.

## 7. Confirm seed robustness and ensemble without weight tuning

A second H64 seed reached 0.391512, close to the first seed's 0.392336. Their
fixed 50/50 average improved development to 0.396738. No ensemble-weight grid
was searched.

Decision: freeze the equal-weight two-seed ensemble.

## 8. Open confirmation once

The frozen ensemble scored 0.422444 on the reserved confirmation sequences,
ahead of both individual seeds. Architecture, weights and schedule were not
changed afterward.

Decision: proceed to final refit without further confirmation tuning.

## 9. Refit the frozen recipe on every training sequence

Both seeds were retrained from initialization for three fixed epochs on all
10,607 sequences (fit + development + confirmation). The final stage performed
no epoch selection because the epoch count had already been frozen.

Decision: use the two epoch-3 checkpoints as the final candidate set.

## 10. Evaluate supplied validation once and package

The final single models scored 0.660729 and 0.660585 on the supplied validation
mask. Their fixed ensemble scored 0.663483 and was selected. Validation was
used for final comparison only, not gradient updates.

The final package contains two stateful ONNX models and a callback that resets
both hidden states at sequence boundaries and averages predictions. Numerical
parity, warm-up behavior, output validity, archive contents and integrity were
verified.

Decision: submit
`h64_gru_two_seed_ensemble_full_train.zip` as the first official candidate.

## Rejected or superseded branches

- Initial 128-sequence static holdout: superseded by the fixed 530-sequence
  development partition.
- Repeated-value indicators: no useful static gain.
- First differences: too small a gain to justify added runtime and state.
- Explicit EMA inputs to GRU: materially worse in the controlled screen.
- Shared output head: weaker than target-specific heads.
- H128: screen winner, but weaker and larger after full training.
- Single-seed submission: reproducible, but consistently weaker than the
  equal-weight ensemble.

Interrupted launches without completed checkpoints are retained only as local
operational history and were not treated as model results.
