# Fixed local validation subsets

The builder at `scripts/build_validation_subsets.py` selects complete sequences
uniformly at random, without inspecting scores or optimizing the selection. Seed:
20261101. The development and confirmation selections are disjoint.

| Subset | Sequences | Rows | Use |
|---|---:|---:|---|
| `datasets/subsets/valid_dev64.parquet` | 64 | 1,280,000 | Routine theory and feature comparisons |
| `datasets/subsets/valid_confirm256.parquet` | 256 | 5,120,000 | Occasional confirmation of promising candidates |
| Outside both subsets | 1,553 | 31,060,000 | Reserved for later evaluation |

These counts apply to the supplied 1,873-sequence validation file. The subsets
preserve every column and row within each selected sequence, including warm-up,
targets and the official scoring mask. Each output row group is one sequence.
The subsets do not modify the source datasets or create any training data.

## Build and verify

From the starter-pack root with the Python environment activated:

```bash
python scripts/build_validation_subsets.py
python scripts/build_validation_subsets.py --verify
```

The first command writes both datasets and
`experiments/splits/validation_v1.json`. It refuses to overwrite existing outputs.
The manifest records exact source row groups, sequence IDs, checksums, sizes,
scored-row counts, target-weight sums, and the remaining source row groups.
Verification checks checksums and compares every selected table with its source.
It reads the full source file for its checksum, but never loads it all into RAM.

Commit the script, this guide, and the JSON manifest. The Parquet files remain
ignored by the existing `.gitignore`. No extra dependencies are needed.

## Evaluate

```bash
python baseline/solution.py --validation datasets/subsets/valid_dev64.parquet
python baseline/solution.py --validation datasets/subsets/valid_confirm256.parquet
```

Use the corresponding path to your own `solution.py` when it supports the same
command-line interface. Otherwise, pass your `PredictionModel()` instance to
`utils.ScorerStepByStep(path).score(model)` in a local runner. The scored model
must start fresh for each evaluation.

Record the code commit, model checkpoint hash, subset name, per-target/global WP,
and runtime. Compare models on identical subsets using the official global
accumulator, never the mean of sequence correlations.

## Interpretation

A random subset is a screening tool, not a representative-score guarantee.
Use development frequently and confirmation sparingly. If you repeatedly tune
against confirmation results, it becomes development data too. Keep both out of
training while using them to judge models. Do not change the seed to find an
easier score.

The remaining 1,553 sequences are disjoint from both subsets. Full validation
includes all three groups, so it is not an independent holdout once development
has been used. For an independent late-stage check, evaluate only the remaining
row groups using the manifest before looking at their results.

Full local validation is:

```bash
python baseline/solution.py --validation datasets/valid.parquet
```

The competition website evaluates a submission on hidden test data; it does not
run the public `valid.parquet` benchmark. Subset results cannot certify hidden-test
performance or the full-test time limit. Test the exported, packaged solution as
well as the training-framework version before submitting.
