"""Train weighted ridge and small MLP baselines using train.parquet only.

The script creates a fixed sequence-level holdout from the training file. It
never opens valid.parquet or valid_mask.parquet. All remaining sequences each
contribute the same number of sampled post-warm-up rows to model fitting.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import pyarrow.parquet as pq
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

sys.dont_write_bytecode = True
SEED = 20261102
HOLDOUT_SEQUENCES = 128
SAMPLES_PER_FIT_SEQUENCE = 64
RIDGE_ALPHA = 1_000.0
MLP_HIDDEN_LAYERS = (32, 16)
MLP_EPOCHS = 12
MLP_BATCH_SIZE = 4096
FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{i}" for i in range(8)]
TARGETS = ("t0", "t1")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sequence(table, expected_columns: list[str]) -> int:
    if table.num_rows != 20_000 or table.column_names != expected_columns:
        raise ValueError("unexpected training row-group shape or columns")
    ids = table["seq_ix"].to_numpy()
    steps = table["step_in_seq"].to_numpy()
    need = table["need_prediction"].to_numpy().astype(bool)
    if not np.all(ids == ids[0]):
        raise ValueError("multiple sequence IDs in a row group")
    if not np.array_equal(steps, np.arange(20_000)):
        raise ValueError("incorrect step order")
    if not np.array_equal(need, steps >= 99):
        raise ValueError("incorrect warm-up flags")
    return int(ids[0])


def weighted_pearson_from_moments(weight, sum_y, sum_p, sum_yy, sum_pp, sum_yp) -> float:
    if weight < 1e-8:
        return 0.0
    mean_y, mean_p = sum_y / weight, sum_p / weight
    var_y = max(0.0, sum_yy / weight - mean_y * mean_y)
    var_p = max(0.0, sum_pp / weight - mean_p * mean_p)
    covariance = sum_yp / weight - mean_y * mean_p
    if var_y <= 1e-16 or var_p <= 1e-16:
        return 0.0
    return float(np.clip(covariance / np.sqrt(var_y * var_p), -1, 1))


class MetricAccumulator:
    def __init__(self) -> None:
        self.weight = np.zeros(2)
        self.sum_y = np.zeros(2)
        self.sum_p = np.zeros(2)
        self.sum_yy = np.zeros(2)
        self.sum_pp = np.zeros(2)
        self.sum_yp = np.zeros(2)
        self.rows = 0

    def add(self, target: np.ndarray, prediction: np.ndarray) -> None:
        y = np.clip(target, -2, 2).astype(np.float64)
        p = np.clip(prediction, -2, 2).astype(np.float64)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("nonfinite evaluation values")
        w = np.abs(y)
        self.weight += w.sum(axis=0)
        self.sum_y += (w * y).sum(axis=0)
        self.sum_p += (w * p).sum(axis=0)
        self.sum_yy += (w * y * y).sum(axis=0)
        self.sum_pp += (w * p * p).sum(axis=0)
        self.sum_yp += (w * y * p).sum(axis=0)
        self.rows += len(y)

    def result(self) -> dict:
        sides = [
            weighted_pearson_from_moments(
                self.weight[k], self.sum_y[k], self.sum_p[k], self.sum_yy[k],
                self.sum_pp[k], self.sum_yp[k],
            )
            for k in range(2)
        ]
        return {"t0": sides[0], "t1": sides[1], "weighted_pearson": float(np.mean(sides)), "rows": self.rows}


def export_mlp(path: Path, mean: np.ndarray, scale: np.ndarray, models: list[MLPRegressor]) -> None:
    arrays = {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}
    for target, model in enumerate(models):
        for layer, (weight, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
            arrays[f"t{target}_w{layer}"] = weight.astype(np.float32)
            arrays[f"t{target}_b{layer}"] = bias.astype(np.float32)
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-name", default="train_only_static_v1")
    args = parser.parse_args()
    root = args.root.resolve()
    train_path = root / "datasets" / "train.parquet"
    artifact_dir = root / "artifacts" / args.artifact_name
    experiment_dir = root / "experiments" / "models" / args.artifact_name
    if artifact_dir.exists() or experiment_dir.exists():
        raise FileExistsError("refusing to overwrite an existing model attempt")
    artifact_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)
    started = time.perf_counter()
    parquet = pq.ParquetFile(train_path)
    expected_schema = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != expected_schema:
        raise ValueError("train.parquet schema differs from the official training schema")

    groups = list(range(parquet.num_row_groups))
    shuffled = groups.copy()
    random.Random(SEED).shuffle(shuffled)
    holdout_groups = sorted(shuffled[:HOLDOUT_SEQUENCES])
    holdout_set = set(holdout_groups)
    fit_groups = [group for group in groups if group not in holdout_set]
    fit_rows = len(fit_groups) * SAMPLES_PER_FIT_SEQUENCE
    print(f"Train-only split: {len(fit_groups):,} fit sequences, {len(holdout_groups)} untouched holdout sequences", flush=True)
    print(f"Allocating {fit_rows:,} uniformly sampled fitting rows", flush=True)
    x_fit = np.empty((fit_rows, 112), np.float32)
    y_fit = np.empty((fit_rows, 2), np.float32)
    fit_sequence_ids = []
    holdout_sequence_ids = []
    rng = np.random.default_rng(SEED)
    columns = expected_schema
    cursor = 0
    for index, group in enumerate(fit_groups):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        sequence_id = validate_sequence(table, columns)
        fit_sequence_ids.append(sequence_id)
        chosen = np.sort(rng.choice(np.arange(99, 20_000), SAMPLES_PER_FIT_SEQUENCE, replace=False))
        x_fit[cursor:cursor + SAMPLES_PER_FIT_SEQUENCE] = np.column_stack([
            table[c].to_numpy()[chosen] for c in FEATURES
        ]).astype(np.float32)
        y_fit[cursor:cursor + SAMPLES_PER_FIT_SEQUENCE] = np.column_stack([
            table[c].to_numpy()[chosen] for c in TARGETS
        ]).astype(np.float32)
        cursor += SAMPLES_PER_FIT_SEQUENCE
        if (index + 1) % 1000 == 0 or index + 1 == len(fit_groups):
            print(f"Sampled {index + 1:,}/{len(fit_groups):,} fit sequences", flush=True)
    if not np.isfinite(x_fit).all() or not np.isfinite(y_fit).all():
        raise ValueError("nonfinite fitting data")

    mean = x_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x_fit.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x_fit -= mean
    x_fit /= scale
    y_fit = np.clip(y_fit, -2, 2)

    print("Fitting two target-specific weighted ridge regressions", flush=True)
    ridge_models = []
    ridge_weight = np.empty((2, 112), np.float32)
    ridge_bias = np.empty(2, np.float32)
    for target in range(2):
        model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=1e-5, max_iter=1000)
        model.fit(x_fit, y_fit[:, target], sample_weight=np.abs(y_fit[:, target]))
        ridge_models.append(model)
        ridge_weight[target] = model.coef_.astype(np.float32)
        ridge_bias[target] = np.float32(model.intercept_)
    ridge_path = artifact_dir / "weighted_ridge.npz"
    np.savez_compressed(ridge_path, mean=mean, scale=scale, weight=ridge_weight, bias=ridge_bias)

    print(f"Fitting two target-specific MLPs: {MLP_HIDDEN_LAYERS}, {MLP_EPOCHS} epochs", flush=True)
    mlp_models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for target in range(2):
            print(f"MLP target t{target}", flush=True)
            model = MLPRegressor(
                hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=MLP_BATCH_SIZE,
                learning_rate_init=1e-3,
                max_iter=MLP_EPOCHS,
                shuffle=True,
                random_state=SEED + target,
                early_stopping=False,
                tol=0,
                n_iter_no_change=MLP_EPOCHS + 1,
                verbose=True,
            )
            model.fit(x_fit, y_fit[:, target], sample_weight=np.abs(y_fit[:, target]))
            mlp_models.append(model)
    mlp_path = artifact_dir / "small_mlp.npz"
    export_mlp(mlp_path, mean, scale, mlp_models)
    del x_fit, y_fit

    sys.path.insert(0, str(root / "src"))
    from static_models import RidgePredictionModel, MLPPredictionModel
    ridge_runtime = RidgePredictionModel(ridge_path)
    mlp_runtime = MLPPredictionModel(mlp_path)
    ridge_metric, mlp_metric = MetricAccumulator(), MetricAccumulator()
    ridge_seconds = mlp_seconds = 0.0
    print("Evaluating both models on every eligible row of the train-only holdout", flush=True)
    for index, group in enumerate(holdout_groups):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        sequence_id = validate_sequence(table, columns)
        holdout_sequence_ids.append(sequence_id)
        x = np.column_stack([table[c].to_numpy() for c in FEATURES]).astype(np.float32)
        y = np.column_stack([table[c].to_numpy() for c in TARGETS]).astype(np.float32)
        start = time.perf_counter()
        ridge_prediction = ridge_runtime.batch_predict(x[99:])
        ridge_seconds += time.perf_counter() - start
        start = time.perf_counter()
        mlp_prediction = mlp_runtime.batch_predict(x[99:])
        mlp_seconds += time.perf_counter() - start
        ridge_metric.add(y[99:], ridge_prediction)
        mlp_metric.add(y[99:], mlp_prediction)
        if (index + 1) % 16 == 0 or index + 1 == len(holdout_groups):
            print(f"Evaluated {index + 1}/{len(holdout_groups)} holdout sequences", flush=True)

    ridge_result = ridge_metric.result()
    mlp_result = mlp_metric.result()
    ridge_result.update({"batch_inference_seconds": ridge_seconds, "artifact_bytes": ridge_path.stat().st_size})
    mlp_result.update({"batch_inference_seconds": mlp_seconds, "artifact_bytes": mlp_path.stat().st_size})
    manifest = {
        "version": 1,
        "policy": "train.parquet only; public validation files never opened",
        "seed": SEED,
        "source": {"path": "datasets/train.parquet", "rows": parquet.metadata.num_rows, "sequences": parquet.num_row_groups},
        "split": {
            "fit_sequences": len(fit_groups),
            "holdout_sequences": len(holdout_groups),
            "holdout_source_row_groups": holdout_groups,
            "holdout_seq_ix": holdout_sequence_ids,
            "fit_seq_ix": fit_sequence_ids,
        },
        "sampling": {
            "rows_per_fit_sequence": SAMPLES_PER_FIT_SEQUENCE,
            "fit_rows": fit_rows,
            "eligible_steps": [99, 19999],
            "method": "one seeded numpy RNG; uniform without replacement per fit sequence in source-row-group order",
        },
        "targets": "clipped to [-2, 2]; separate models weighted by abs(clipped target)",
        "ridge": {"alpha": RIDGE_ALPHA},
        "mlp": {"hidden_layers": list(MLP_HIDDEN_LAYERS), "epochs": MLP_EPOCHS, "batch_size": MLP_BATCH_SIZE, "activation": "relu"},
        "script_sha256": file_sha256(Path(__file__)),
    }
    metrics = {
        "evaluation": "all 19,901 eligible rows in each train-only holdout sequence; no scoring mask",
        "ridge": ridge_result,
        "mlp": mlp_result,
        "artifacts": {
            "ridge": {"path": str(ridge_path.relative_to(root)), "sha256": file_sha256(ridge_path)},
            "mlp": {"path": str(mlp_path.relative_to(root)), "sha256": file_sha256(mlp_path)},
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(experiment_dir / "manifest.json", manifest)
    write_json(experiment_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
