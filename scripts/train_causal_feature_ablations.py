"""Ablate causal feature families on a train-only sequence split.

Only datasets/train.parquet is opened. Models are fitted on complete fit
sequences (using uniform row samples), compared on complete development
sequences, and never evaluated on the reserved confirmation sequences here.
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
from scipy.signal import lfilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

sys.dont_write_bytecode = True
SEED = 20261102
DEVELOPMENT_SEQUENCES = 530
CONFIRMATION_SEQUENCES = 530
SAMPLES_PER_FIT_SEQUENCE = 64
RIDGE_ALPHA = 1_000.0
MLP_HIDDEN_LAYERS = (32, 16)
MLP_EPOCHS = 12
MLP_BATCH_SIZE = 4096
EMA_HALF_LIVES = (8, 64)
FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{i}" for i in range(8)]
TARGETS = ("t0", "t1")
CONFIGS = ("raw", "first_difference", "exponential_average", "repeated_value")


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


def ema(x: np.ndarray, half_life: int) -> np.ndarray:
    """Causal EMA initialized at x[0], with no future observations."""
    alpha = np.float32(1.0 - 2.0 ** (-1.0 / half_life))
    initial_state = ((1.0 - alpha) * x[0])[None, :]
    values, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0, zi=initial_state)
    return values.astype(np.float32, copy=False)


def sampled_family(x: np.ndarray, chosen: np.ndarray, config: str) -> np.ndarray:
    if config == "raw":
        return x[chosen]
    if config == "first_difference":
        delta = x[chosen] - x[chosen - 1]
        return np.concatenate((x[chosen], delta), axis=1)
    if config == "exponential_average":
        blocks = [x[chosen]]
        for half_life in EMA_HALF_LIVES:
            blocks.append(ema(x, half_life)[chosen])
        return np.concatenate(blocks, axis=1)
    if config == "repeated_value":
        repeated = (x[chosen] == x[chosen - 1]).astype(np.float32)
        return np.concatenate((x[chosen], repeated), axis=1)
    raise ValueError(f"unknown feature configuration: {config}")


def full_family(x: np.ndarray, config: str) -> np.ndarray:
    current = x[99:]
    if config == "raw":
        return current.copy()
    if config == "first_difference":
        return np.concatenate((current, x[99:] - x[98:-1]), axis=1)
    if config == "exponential_average":
        return np.concatenate([current, *[ema(x, h)[99:] for h in EMA_HALF_LIVES]], axis=1)
    if config == "repeated_value":
        repeated = (x[99:] == x[98:-1]).astype(np.float32)
        return np.concatenate((current, repeated), axis=1)
    raise ValueError(f"unknown feature configuration: {config}")


def weighted_pearson(weight, sum_y, sum_p, sum_yy, sum_pp, sum_yp) -> float:
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
            weighted_pearson(self.weight[k], self.sum_y[k], self.sum_p[k], self.sum_yy[k], self.sum_pp[k], self.sum_yp[k])
            for k in range(2)
        ]
        return {"t0": sides[0], "t1": sides[1], "weighted_pearson": float(np.mean(sides)), "rows": self.rows}


def export_ridge(path: Path, mean: np.ndarray, scale: np.ndarray, models: list[Ridge], config: str) -> None:
    np.savez_compressed(
        path,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        weight=np.stack([m.coef_ for m in models]).astype(np.float32),
        bias=np.asarray([m.intercept_ for m in models], dtype=np.float32),
        feature_config=np.asarray(config),
        ema_half_lives=np.asarray(EMA_HALF_LIVES, dtype=np.int32),
    )


def export_mlp(path: Path, mean: np.ndarray, scale: np.ndarray, models: list[MLPRegressor], config: str) -> None:
    arrays = {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "feature_config": np.asarray(config),
        "ema_half_lives": np.asarray(EMA_HALF_LIVES, dtype=np.int32),
    }
    for target, model in enumerate(models):
        for layer, (weight, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
            arrays[f"t{target}_w{layer}"] = weight.astype(np.float32)
            arrays[f"t{target}_b{layer}"] = bias.astype(np.float32)
    np.savez_compressed(path, **arrays)


def predict_mlp(models: list[MLPRegressor], x: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(x) for model in models]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-name", default="train_only_causal_ablation_v1")
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
        raise ValueError("train.parquet schema differs from expected training schema")

    groups = list(range(parquet.num_row_groups))
    shuffled = groups.copy()
    random.Random(SEED).shuffle(shuffled)
    development_groups = sorted(shuffled[:DEVELOPMENT_SEQUENCES])
    confirmation_groups = sorted(shuffled[DEVELOPMENT_SEQUENCES:DEVELOPMENT_SEQUENCES + CONFIRMATION_SEQUENCES])
    reserved = set(development_groups) | set(confirmation_groups)
    fit_groups = [g for g in groups if g not in reserved]
    print(
        f"Train-only split: {len(fit_groups):,} fit / {len(development_groups):,} development / "
        f"{len(confirmation_groups):,} reserved confirmation sequences",
        flush=True,
    )

    rows = len(fit_groups) * SAMPLES_PER_FIT_SEQUENCE
    dimensions = {"raw": 112, "first_difference": 224, "exponential_average": 336, "repeated_value": 224}
    sampled = {name: np.empty((rows, dimensions[name]), np.float32) for name in CONFIGS}
    y_fit = np.empty((rows, 2), np.float32)
    fit_sequence_ids = []
    rng = np.random.default_rng(SEED)
    columns = expected_schema
    cursor = 0
    print(f"Building {rows:,} shared fitting examples for all feature configurations", flush=True)
    for index, group in enumerate(fit_groups):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        fit_sequence_ids.append(validate_sequence(table, columns))
        x = np.column_stack([table[c].to_numpy() for c in FEATURES]).astype(np.float32)
        chosen = np.sort(rng.choice(np.arange(99, 20_000), SAMPLES_PER_FIT_SEQUENCE, replace=False))
        span = slice(cursor, cursor + SAMPLES_PER_FIT_SEQUENCE)
        for config in CONFIGS:
            sampled[config][span] = sampled_family(x, chosen, config)
        y_fit[span] = np.column_stack([table[c].to_numpy()[chosen] for c in TARGETS]).astype(np.float32)
        cursor += SAMPLES_PER_FIT_SEQUENCE
        if (index + 1) % 500 == 0 or index + 1 == len(fit_groups):
            print(f"Sampled {index + 1:,}/{len(fit_groups):,} fit sequences", flush=True)
    y_fit = np.clip(y_fit, -2, 2)

    trained = {}
    artifact_records = {}
    for config in CONFIGS:
        print(f"Training controlled ablation: {config} ({dimensions[config]} inputs)", flush=True)
        x_fit = sampled.pop(config)
        mean = x_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x_fit.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        x_fit -= mean
        x_fit /= scale

        ridge_models = []
        for target in range(2):
            model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=1e-5, max_iter=1000)
            model.fit(x_fit, y_fit[:, target], sample_weight=np.abs(y_fit[:, target]))
            ridge_models.append(model)

        mlp_models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for target in range(2):
                print(f"  small MLP target t{target}", flush=True)
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
                    verbose=False,
                )
                model.fit(x_fit, y_fit[:, target], sample_weight=np.abs(y_fit[:, target]))
                mlp_models.append(model)

        ridge_path = artifact_dir / f"ridge_{config}.npz"
        mlp_path = artifact_dir / f"mlp_{config}.npz"
        export_ridge(ridge_path, mean, scale, ridge_models, config)
        export_mlp(mlp_path, mean, scale, mlp_models, config)
        artifact_records[config] = {
            "ridge": {"path": str(ridge_path.relative_to(root)).replace("\\", "/"), "sha256": file_sha256(ridge_path)},
            "mlp": {"path": str(mlp_path.relative_to(root)).replace("\\", "/"), "sha256": file_sha256(mlp_path)},
        }
        trained[config] = {"mean": mean, "scale": scale, "ridge": ridge_models, "mlp": mlp_models}
        del x_fit

    metrics = {
        config: {"ridge": MetricAccumulator(), "mlp": MetricAccumulator(), "seconds": {"ridge": 0.0, "mlp": 0.0}}
        for config in CONFIGS
    }
    development_sequence_ids = []
    print("Evaluating every eligible development row; confirmation remains untouched", flush=True)
    for index, group in enumerate(development_groups):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        development_sequence_ids.append(validate_sequence(table, columns))
        x = np.column_stack([table[c].to_numpy() for c in FEATURES]).astype(np.float32)
        y = np.column_stack([table[c].to_numpy()[99:] for c in TARGETS]).astype(np.float32)
        for config in CONFIGS:
            design = full_family(x, config)
            state = trained[config]
            design -= state["mean"]
            design /= state["scale"]
            start = time.perf_counter()
            ridge_prediction = np.column_stack([m.predict(design) for m in state["ridge"]]).astype(np.float32)
            metrics[config]["seconds"]["ridge"] += time.perf_counter() - start
            start = time.perf_counter()
            mlp_prediction = predict_mlp(state["mlp"], design)
            metrics[config]["seconds"]["mlp"] += time.perf_counter() - start
            metrics[config]["ridge"].add(y, ridge_prediction)
            metrics[config]["mlp"].add(y, mlp_prediction)
        if (index + 1) % 25 == 0 or index + 1 == len(development_groups):
            print(f"Evaluated {index + 1:,}/{len(development_groups):,} development sequences", flush=True)

    result = {
        "evaluation": "all eligible rows of development sequences; confirmation not evaluated",
        "feature_definitions": {
            "raw": "current 112 features",
            "first_difference": "raw plus x[t] - x[t-1]",
            "exponential_average": f"raw plus causal EMAs with half-lives {EMA_HALF_LIVES}",
            "repeated_value": "raw plus float32 indicators x[t] == x[t-1]",
        },
        "models": {},
    }
    for config in CONFIGS:
        ridge_result = metrics[config]["ridge"].result()
        mlp_result = metrics[config]["mlp"].result()
        ridge_result["batch_inference_seconds"] = metrics[config]["seconds"]["ridge"]
        mlp_result["batch_inference_seconds"] = metrics[config]["seconds"]["mlp"]
        result["models"][config] = {"ridge": ridge_result, "mlp": mlp_result}
    baseline_ridge = result["models"]["raw"]["ridge"]["weighted_pearson"]
    baseline_mlp = result["models"]["raw"]["mlp"]["weighted_pearson"]
    for config in CONFIGS:
        result["models"][config]["ridge"]["delta_vs_raw"] = result["models"][config]["ridge"]["weighted_pearson"] - baseline_ridge
        result["models"][config]["mlp"]["delta_vs_raw"] = result["models"][config]["mlp"]["weighted_pearson"] - baseline_mlp

    elapsed = time.perf_counter() - started
    result["artifacts"] = artifact_records
    result["elapsed_seconds"] = elapsed
    manifest = {
        "version": 1,
        "policy": "only datasets/train.parquet; sequence-level fit/development/confirmation; confirmation not evaluated",
        "seed": SEED,
        "source": {"path": "datasets/train.parquet", "rows": parquet.metadata.num_rows, "sequences": parquet.num_row_groups},
        "split": {
            "fit_sequences": len(fit_groups),
            "development_sequences": len(development_groups),
            "confirmation_sequences": len(confirmation_groups),
            "fit_source_row_groups": fit_groups,
            "development_source_row_groups": development_groups,
            "confirmation_source_row_groups": confirmation_groups,
            "fit_sequence_ids": fit_sequence_ids,
            "development_sequence_ids": development_sequence_ids,
        },
        "sampling": {"rows_per_fit_sequence": SAMPLES_PER_FIT_SEQUENCE, "eligible_steps": [99, 19999]},
        "models": {
            "ridge": {"alpha": RIDGE_ALPHA, "sample_weight": "abs(clip(target, -2, 2))"},
            "mlp": {"hidden_layers": list(MLP_HIDDEN_LAYERS), "epochs": MLP_EPOCHS, "batch_size": MLP_BATCH_SIZE},
        },
        "feature_dimensions": dimensions,
        "ema_half_lives": list(EMA_HALF_LIVES),
    }
    write_json(experiment_dir / "manifest.json", manifest)
    write_json(experiment_dir / "metrics.json", result)
    print(json.dumps(result["models"], indent=2), flush=True)
    print(f"Finished in {elapsed / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
