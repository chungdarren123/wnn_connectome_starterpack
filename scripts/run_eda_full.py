"""Full-dataset, sequence-aware EDA for Alfa Connectome.

Every row in train.parquet and valid.parquet contributes to exact streaming
statistics. Bounded, deterministic samples are used only for multivariate
diagnostics whose memory/compute scales quadratically or requires model fitting.
The source files are never modified.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import base64
import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
SEED = 20261101
FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{i}" for i in range(8)]
TARGETS = ("t0", "t1")
LAGS = (1, 4, 16, 64, 256, 1024)
CORRELATION_LAGS = (0, 1, 4, 16, 64, 256)
SAMPLE_ROWS_PER_SEQUENCE = 64


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_lengths(flags: np.ndarray) -> list[int]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, flags, False].astype(np.int8)))
    return (boundaries[1::2] - boundaries[::2]).tolist()


def metadata(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    sizes = Counter(parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups))
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "columns": len(parquet.schema_arrow),
        "row_group_sizes": {str(k): v for k, v in sizes.items()},
        "schema": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in parquet.schema_arrow],
    }


class TargetAccumulator:
    def __init__(self) -> None:
        self.counts = [Counter(), Counter()]
        self.n = 0
        self.sum = np.zeros(2, np.float64)
        self.sumsq = np.zeros(2, np.float64)
        self.cross = 0.0
        self.minimum = np.full(2, np.inf)
        self.maximum = np.full(2, -np.inf)

    def add(self, y: np.ndarray) -> None:
        if not len(y):
            return
        y64 = y.astype(np.float64)
        self.n += len(y)
        self.sum += y64.sum(axis=0)
        self.sumsq += np.square(y64).sum(axis=0)
        self.cross += float(np.dot(y64[:, 0], y64[:, 1]))
        self.minimum = np.minimum(self.minimum, y64.min(axis=0))
        self.maximum = np.maximum(self.maximum, y64.max(axis=0))
        for k in range(2):
            values, counts = np.unique(y[:, k], return_counts=True)
            self.counts[k].update({float(v): int(c) for v, c in zip(values, counts)})

    @staticmethod
    def quantile(counter: Counter, q: float) -> float:
        total = sum(counter.values())
        target = q * (total - 1)
        lo = int(np.floor(target))
        hi = int(np.ceil(target))
        positions = (lo, hi)
        found = []
        cumulative = 0
        for value, count in sorted(counter.items()):
            next_cumulative = cumulative + count
            for position in positions[len(found):]:
                if position < next_cumulative:
                    found.append(value)
                else:
                    break
            cumulative = next_cumulative
            if len(found) == 2:
                break
        if len(found) == 1:
            found.append(found[0])
        return float(found[0] + (target - lo) * (found[1] - found[0]))

    @staticmethod
    def weight_concentration(counter: Counter) -> tuple[float, float, float]:
        weighted = sorted(((abs(np.clip(v, -2, 2)), c) for v, c in counter.items()), reverse=True)
        total_rows = sum(c for _, c in weighted)
        total_weight = sum(w * c for w, c in weighted)
        weight_sq = sum(w * w * c for w, c in weighted)
        wanted = max(1, int(np.ceil(total_rows * 0.10)))
        remaining = wanted
        top_weight = 0.0
        for weight, count in weighted:
            take = min(remaining, count)
            top_weight += weight * take
            remaining -= take
            if not remaining:
                break
        share = top_weight / total_weight if total_weight else 0.0
        ess = total_weight**2 / weight_sq if weight_sq else 0.0
        return float(total_weight), float(share), float(ess)

    def result(self) -> dict:
        output = {}
        for k, name in enumerate(TARGETS):
            counter = self.counts[k]
            mean = self.sum[k] / self.n
            variance = max(0.0, self.sumsq[k] / self.n - mean**2)
            total_weight, top_share, ess = self.weight_concentration(counter)
            output[name] = {
                "rows": self.n,
                "mean": float(mean),
                "std": float(np.sqrt(variance)),
                "min": float(self.minimum[k]),
                "max": float(self.maximum[k]),
                "quantiles": {
                    label: self.quantile(counter, q)
                    for label, q in (("p01", .01), ("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95), ("p99", .99))
                },
                "global_unique_values": len(counter),
                "zero_fraction": counter.get(0.0, 0) / self.n,
                "negative_fraction": sum(c for v, c in counter.items() if v < 0) / self.n,
                "outside_clip_fraction": sum(c for v, c in counter.items() if abs(v) > 2) / self.n,
                "clipped_abs_weight_sum": total_weight,
                "largest_10pct_rows_weight_share": top_share,
                "weight_effective_sample_size": ess,
            }
        cov = self.cross / self.n - (self.sum[0] / self.n) * (self.sum[1] / self.n)
        sd0, sd1 = output["t0"]["std"], output["t1"]["std"]
        output["target_correlation"] = float(cov / (sd0 * sd1)) if sd0 > 0 and sd1 > 0 else None
        return output


class CrossMomentAccumulator:
    def __init__(self) -> None:
        self.sx = np.zeros(112)
        self.sxx = np.zeros(112)
        self.sy = np.zeros(2)
        self.syy = np.zeros(2)
        self.sxy = np.zeros((112, 2))
        self.n = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        a, b = x[valid].astype(np.float64), y[valid].astype(np.float64)
        self.sx += a.sum(axis=0)
        self.sxx += np.square(a).sum(axis=0)
        self.sy += b.sum(axis=0)
        self.syy += np.square(b).sum(axis=0)
        self.sxy += a.T @ b
        self.n += len(a)

    def correlations(self) -> np.ndarray:
        numerator = self.sxy - np.outer(self.sx, self.sy) / self.n
        denominator = np.sqrt(
            np.maximum(0, self.sxx - self.sx * self.sx / self.n)[:, None]
            * np.maximum(0, self.syy - self.sy * self.sy / self.n)[None, :]
        )
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def analyze(path: Path, name: str, output: Path) -> dict:
    parquet = pq.ParquetFile(path)
    rng = np.random.default_rng(SEED + (0 if name == "train" else 1))
    finite_count = np.zeros(112, np.int64)
    nonfinite_count = np.zeros(112, np.int64)
    arrow_nulls = np.zeros(112, np.int64)
    sums = np.zeros(112)
    sumsq = np.zeros(112)
    minimum = np.full(112, np.inf)
    maximum = np.full(112, -np.inf)
    changes = np.zeros(112, np.int64)
    change_denominator = np.zeros(112, np.int64)
    below_minus4 = np.zeros(112, np.int64)
    exact_repeated_minus_5199337 = np.zeros(112, np.int64)
    eligible_targets = TargetAccumulator()
    scored_targets = TargetAccumulator() if name == "valid" else None
    target_acf = {lag: [[], []] for lag in LAGS}
    cross = {lag: CrossMomentAccumulator() for lag in CORRELATION_LAGS}
    sequence_rows = []
    mask_runs = []
    sample_x, sample_y, sample_groups, sample_steps, sample_scored = [], [], [], [], []
    seen_ids = set()
    first_example = None

    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, use_threads=False)
        if table.num_rows != 20_000:
            raise ValueError(f"{name} row group {group} is not a complete sequence")
        ids = table["seq_ix"].to_numpy()
        sequence_id = int(ids[0])
        steps = table["step_in_seq"].to_numpy()
        need = table["need_prediction"].to_numpy().astype(bool)
        if not np.all(ids == sequence_id) or sequence_id in seen_ids:
            raise ValueError(f"{name} sequence identity violation")
        if not np.array_equal(steps, np.arange(20_000)) or not np.array_equal(need, steps >= 99):
            raise ValueError(f"{name} ordering/warm-up violation")
        seen_ids.add(sequence_id)
        x = np.column_stack([table[c].to_numpy() for c in FEATURES]).astype(np.float32)
        y = np.column_stack([table[c].to_numpy() for c in TARGETS]).astype(np.float32)
        if not np.isfinite(y).all():
            raise ValueError(f"{name} contains nonfinite targets")
        ok = np.isfinite(x)
        clean = np.where(ok, x, 0).astype(np.float64)
        finite_count += ok.sum(axis=0)
        nonfinite_count += (~ok).sum(axis=0)
        arrow_nulls += np.array([table[c].null_count for c in FEATURES])
        sums += clean.sum(axis=0)
        sumsq += np.square(clean).sum(axis=0)
        minimum = np.minimum(minimum, np.where(ok, x, np.inf).min(axis=0))
        maximum = np.maximum(maximum, np.where(ok, x, -np.inf).max(axis=0))
        adjacent_ok = ok[1:] & ok[:-1]
        changes += ((x[1:] != x[:-1]) & adjacent_ok).sum(axis=0)
        change_denominator += adjacent_ok.sum(axis=0)
        below_minus4 += (x < -4).sum(axis=0)
        exact_repeated_minus_5199337 += np.isclose(x, -5.199337482452393, rtol=0, atol=1e-7).sum(axis=0)
        eligible_targets.add(y[need])
        scored = table["is_scored"].to_numpy().astype(bool) if name == "valid" else np.zeros(20_000, bool)
        if name == "valid":
            if scored[:99].any():
                raise ValueError("Validation mask selects warm-up rows")
            scored_targets.add(y[need & scored])
            mask_runs.extend(run_lengths(need & scored))

        chosen = np.sort(rng.choice(np.arange(1024, 20_000), SAMPLE_ROWS_PER_SEQUENCE, replace=False))
        sample_x.append(x[chosen])
        sample_y.append(y[chosen])
        sample_groups.append(np.full(len(chosen), sequence_id, np.int64))
        sample_steps.append(chosen)
        sample_scored.append(scored[chosen])
        for lag in CORRELATION_LAGS:
            cross[lag].add(x[chosen - lag], y[chosen])
        for lag in LAGS:
            for k in range(2):
                value = safe_corr(y[lag:, k], y[:-lag, k])
                target_acf[lag][k].append(np.nan if value is None else value)

        sequence_record = {
            "dataset": name,
            "source_row_group": group,
            "seq_ix": sequence_id,
            "scored_rows": int((need & scored).sum()) if name == "valid" else "",
            "feature_nonfinite": int((~ok).sum()),
            "constant_feature_columns": int(np.sum(np.std(clean, axis=0) <= 1e-12)),
        }
        for k, target in enumerate(TARGETS):
            values = y[need, k]
            sequence_record.update({
                f"{target}_mean": float(values.mean(dtype=np.float64)),
                f"{target}_std": float(values.std(dtype=np.float64)),
                f"{target}_zero_fraction": float(np.mean(values == 0)),
                f"{target}_unique_values": int(len(np.unique(values))),
                f"{target}_outside_clip_fraction": float(np.mean(np.abs(values) > 2)),
            })
        sequence_rows.append(sequence_record)
        if first_example is None:
            first_example = {
                "seq_ix": sequence_id,
                "step": steps[:2000].tolist(),
                "t0": y[:2000, 0].tolist(),
                "t1": y[:2000, 1].tolist(),
                "is_scored": scored[:2000].tolist(),
                "features": {c: x[:2000, FEATURES.index(c)].tolist() for c in ("i0_p0", "i1_p0", "a0")},
            }
        if (group + 1) % 250 == 0 or group + 1 == parquet.num_row_groups:
            print(f"{name}: {group + 1:,}/{parquet.num_row_groups:,} sequences", flush=True)

    sample_x_array = np.concatenate(sample_x)
    sample_y_array = np.concatenate(sample_y)
    sample_groups_array = np.concatenate(sample_groups)
    sample_steps_array = np.concatenate(sample_steps)
    sample_scored_array = np.concatenate(sample_scored)
    means = sums / np.maximum(finite_count, 1)
    stds = np.sqrt(np.maximum(0, sumsq / np.maximum(finite_count, 1) - means**2))
    sample_quantiles = np.nanquantile(
        np.where(np.isfinite(sample_x_array), sample_x_array, np.nan),
        [0, .01, .05, .25, .5, .75, .95, .99, 1], axis=0,
    )
    feature_rows = []
    for j, feature in enumerate(FEATURES):
        values, counts = np.unique(sample_x_array[np.isfinite(sample_x_array[:, j]), j], return_counts=True)
        mode_index = int(np.argmax(counts)) if len(counts) else None
        row = {
            "dataset": name,
            "feature": feature,
            "finite_count": int(finite_count[j]),
            "nonfinite_count": int(nonfinite_count[j]),
            "arrow_null_count": int(arrow_nulls[j]),
            "mean": float(means[j]),
            "std": float(stds[j]),
            "min": float(minimum[j]),
            "max": float(maximum[j]),
            "change_fraction": float(changes[j] / max(1, change_denominator[j])),
            "full_fraction_below_minus4": float(below_minus4[j] / max(1, finite_count[j])),
            "full_fraction_equal_minus5_199337": float(exact_repeated_minus_5199337[j] / max(1, finite_count[j])),
            "sample_unique_values": int(len(values)),
            "sample_mode": float(values[mode_index]) if mode_index is not None else None,
            "sample_mode_fraction": float(counts[mode_index] / len(sample_x_array)) if mode_index is not None else 0,
        }
        for index, label in enumerate(("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")):
            row[f"sample_{label}"] = float(sample_quantiles[index, j])
        feature_rows.append(row)
    lag_rows = []
    for lag, accumulator in cross.items():
        correlations = accumulator.correlations()
        for j, feature in enumerate(FEATURES):
            lag_rows.append({
                "dataset": name,
                "lag": lag,
                "feature": feature,
                "t0_correlation": float(correlations[j, 0]),
                "t1_correlation": float(correlations[j, 1]),
                "sample_rows": accumulator.n,
            })
    write_csv(output / f"{name}_features.csv", feature_rows)
    write_csv(output / f"{name}_sequences.csv", sequence_rows)
    write_csv(output / f"{name}_lag_correlations.csv", lag_rows)
    summary = {
        "sequences": parquet.num_row_groups,
        "rows": parquet.metadata.num_rows,
        "multivariate_sample_rows": len(sample_x_array),
        "sample_rows_per_sequence": SAMPLE_ROWS_PER_SEQUENCE,
        "sample_step_range": [1024, 19999],
        "feature_nonfinite_count": int(nonfinite_count.sum()),
        "feature_arrow_null_count": int(arrow_nulls.sum()),
        "target_statistics_post_warmup": eligible_targets.result(),
        "target_autocorrelation_median_by_sequence": {
            str(lag): [float(np.nanmedian(target_acf[lag][k])) for k in range(2)] for lag in LAGS
        },
    }
    if name == "valid":
        summary.update({
            "target_statistics_scored": scored_targets.result(),
            "scored_rows": scored_targets.n,
            "scored_fraction_post_warmup": scored_targets.n / (parquet.num_row_groups * 19_901),
            "scored_run_lengths": {
                "runs": len(mask_runs),
                "median": float(np.median(mask_runs)),
                "p90": float(np.quantile(mask_runs, .9)),
                "max": int(max(mask_runs)),
            },
        })
    return {
        "summary": summary,
        "features": feature_rows,
        "sequences": sequence_rows,
        "lags": lag_rows,
        "x": sample_x_array,
        "y": sample_y_array,
        "groups": sample_groups_array,
        "steps": sample_steps_array,
        "scored": sample_scored_array,
        "target_counts": [eligible_targets.counts[0], eligible_targets.counts[1]],
        "scored_target_counts": [scored_targets.counts[0], scored_targets.counts[1]] if scored_targets else None,
        "example": first_example,
    }


def classifier_diagnostic(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    splitter = GroupShuffleSplit(n_splits=1, test_size=.3, random_state=seed)
    train, test = next(splitter.split(x, y, groups))
    valid = np.isfinite(x).all(axis=1)
    train, test = train[valid[train]], test[valid[test]]
    result = {
        "training_sequences": int(len(np.unique(groups[train]))),
        "heldout_sequences": int(len(np.unique(groups[test]))),
        "training_rows": int(len(train)),
        "heldout_rows": int(len(test)),
        "heldout_positive_fraction": float(y[test].mean()),
    }
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)),
        "shallow_boosting": HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1, random_state=seed),
    }
    for name, model in models.items():
        model.fit(x[train], y[train])
        result[f"{name}_heldout_auc"] = float(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    return result


def target_plot_values(counter: Counter) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(sorted(counter), np.float64)
    counts = np.array([counter[v] for v in values], np.float64)
    return values, counts


def create_figures(train: dict, valid: dict, shifts: list[dict], pca: PCA, output: Path) -> list[Path]:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    images = []

    def save(name: str, figure) -> None:
        path = output / f"{name}.png"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        images.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for k, target in enumerate(TARGETS):
        for counters, label, color in (
            (train["target_counts"], "All training", "#3366aa"),
            (valid["target_counts"], "All validation eligible", "#e69528"),
            (valid["scored_target_counts"], "All validation scored", "#16856c"),
        ):
            values, counts = target_plot_values(counters[k])
            bins = np.linspace(-4, 4, 81)
            histogram, _ = np.histogram(values, bins=bins, weights=counts)
            axes[0, k].stairs(histogram / counts.sum(), edges=bins, label=label, color=color)
        axes[0, k].set(xlim=(-4, 4), title=f"{target}: exact full-dataset target mass", xlabel="Target (values outside display omitted)", ylabel="Probability mass per 0.1-wide bin")
        axes[0, k].axvline(-2, color="grey", ls="--")
        axes[0, k].axvline(2, color="grey", ls="--")
        axes[0, k].legend(fontsize=8)
        for counters, label, color in (
            (train["target_counts"], "All training", "#3366aa"),
            (valid["scored_target_counts"], "All validation scored", "#16856c"),
        ):
            weights = sorted(((abs(np.clip(v, -2, 2)), c) for v, c in counters[k].items()), reverse=True)
            row_fraction, weight_fraction = [0.0], [0.0]
            rows = sum(c for _, c in weights)
            total = sum(w * c for w, c in weights)
            row_accum = weight_accum = 0.0
            for weight, count in weights:
                row_accum += count
                weight_accum += weight * count
                row_fraction.append(row_accum / rows)
                weight_fraction.append(weight_accum / total)
            axes[1, k].plot(row_fraction, weight_fraction, label=label, color=color)
        axes[1, k].plot([0, 1], [0, 1], ":", color="grey")
        axes[1, k].set(title=f"{target}: exact metric-weight concentration", xlabel="Fraction of rows, highest weight first", ylabel="Fraction of target weight")
        axes[1, k].legend(fontsize=8)
    fig.tight_layout()
    save("01_targets_and_metric_weights", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    top = shifts[:20]
    axes[0].barh([r["feature"] for r in top][::-1], [r["ks_distance"] for r in top][::-1], color="#3366aa")
    axes[0].set(title="Largest train/validation marginal differences", xlabel="KS distance on bounded row samples")
    feature = top[0]["feature"]
    j = FEATURES.index(feature)
    combined = np.r_[train["x"][:, j], valid["x"][:, j]]
    bounds = np.quantile(combined[np.isfinite(combined)], [.005, .995])
    for dataset, label, color in ((train, "Train", "#3366aa"), (valid, "Validation", "#e69528")):
        axes[1].hist(dataset["x"][:, j], bins=np.linspace(*bounds, 61), density=True, histtype="step", label=label, color=color)
    axes[1].set(title=f"{feature}: strongest sampled marginal shift", xlabel="Feature value (central 99% display)", ylabel="Density")
    axes[1].legend()
    fig.tight_layout()
    save("02_distribution_shift", fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    example = valid["example"]
    for feature in ("i0_p0", "i1_p0", "a0"):
        axes[0, 0].plot(example["step"], example["features"][feature], label=feature, lw=.7)
    axes[0, 0].set(title=f"Validation sequence {example['seq_ix']}: first 2,000 steps", xlabel="Step", ylabel="Anonymized value")
    axes[0, 0].legend(fontsize=8)
    for target in TARGETS:
        axes[0, 1].plot(example["step"], example[target], label=target, lw=.8)
    mask = np.array(example["is_scored"])
    axes[0, 1].scatter(np.arange(2000)[mask], np.zeros(mask.sum()), s=5, color="black", alpha=.4, label="Scored")
    axes[0, 1].set(title="Targets and validation mask", xlabel="Step", ylabel="Target")
    axes[0, 1].legend(fontsize=8)
    for dataset, label, style in ((train, "Train", "-"), (valid, "Validation", "--")):
        acf = dataset["summary"]["target_autocorrelation_median_by_sequence"]
        for k, target in enumerate(TARGETS):
            axes[1, 0].plot(LAGS, [acf[str(lag)][k] for lag in LAGS], style, marker="o", label=f"{label} {target}")
    axes[1, 0].set(xscale="log", title="Median within-sequence target autocorrelation", xlabel="Lag in rows", ylabel="Pearson correlation")
    axes[1, 0].legend(fontsize=8)
    fractions = [int(r["scored_rows"]) / 19_901 for r in valid["sequences"]]
    axes[1, 1].hist(fractions, bins=35, color="#16856c")
    axes[1, 1].set(title="Scored-row fraction across all validation sequences", xlabel="Scored / required rows", ylabel="Sequences")
    fig.tight_layout()
    save("03_temporal_structure_and_mask", fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for k, target in enumerate(TARGETS):
        rows = [r for r in train["lags"] if r["lag"] == 0]
        rows.sort(key=lambda r: abs(r[f"{target}_correlation"]))
        selected = rows[-15:]
        axes[0, k].barh([r["feature"] for r in selected], [r[f"{target}_correlation"] for r in selected], color="#3366aa")
        axes[0, k].set(title=f"Train: strongest sampled correlations with {target}", xlabel="Unweighted Pearson")
        for dataset, label, color in ((train, "Train", "#3366aa"), (valid, "Validation", "#e69528")):
            best = []
            for lag in CORRELATION_LAGS:
                candidates = [abs(r[f"{target}_correlation"]) for r in dataset["lags"] if r["lag"] == lag]
                best.append(max(candidates))
            axes[1, k].plot(CORRELATION_LAGS, best, marker="o", label=label, color=color)
        axes[1, k].set(xscale="symlog", title=f"Strongest sampled lagged-input relation: {target}", xlabel="Past input lag", ylabel="Maximum absolute feature correlation")
        axes[1, k].legend()
    fig.tight_layout()
    save("04_feature_target_relationships", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    correlation = np.corrcoef(np.where(np.isfinite(train["x"]), train["x"], 0), rowvar=False)
    axes[0].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set(title="Training feature correlation matrix (sampled rows)", xlabel="Official feature index", ylabel="Official feature index")
    axes[1].plot(np.arange(1, 113), np.cumsum(pca.explained_variance_ratio_), color="#3366aa")
    axes[1].set(title="Training standardized PCA (sampled rows)", xlabel="Principal components", ylabel="Cumulative explained variance", ylim=(0, 1.01))
    fig.tight_layout()
    save("05_redundancy_and_pca", fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for dataset, label, color in ((train, "Train", "#3366aa"), (valid, "Validation", "#e69528")):
        axes[0].plot([r["change_fraction"] for r in dataset["features"]], label=label, color=color)
        axes[1].plot([r["full_fraction_equal_minus5_199337"] for r in dataset["features"]], label=label, color=color)
    axes[0].set(title="Exact fraction of adjacent finite rows that change", xlabel="Official feature index", ylabel="Change fraction")
    axes[1].set(title="Exact frequency of repeated value -5.199337", xlabel="Official feature index", ylabel="Row fraction")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    save("06_feature_activity_and_extremes", fig)
    return images


def html_table(rows: list[dict], keys: list[str]) -> str:
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in keys) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def create_report(output: Path, summary: dict, train: dict, valid: dict, shifts: list[dict], images: list[Path]) -> None:
    target_rows = []
    for dataset, label, field in (
        (train, "training eligible", "target_statistics_post_warmup"),
        (valid, "validation eligible", "target_statistics_post_warmup"),
        (valid, "validation scored", "target_statistics_scored"),
    ):
        for target in TARGETS:
            stats = dataset["summary"][field][target]
            target_rows.append({
                "population": label, "target": target, "rows": f"{stats['rows']:,}",
                "mean": f"{stats['mean']:.4f}", "std": f"{stats['std']:.4f}",
                "zero fraction": f"{stats['zero_fraction']:.3%}",
                "outside ±2": f"{stats['outside_clip_fraction']:.3%}",
                "unique values": stats["global_unique_values"],
            })
    domain = summary["domain_classification"]
    mask = summary["scoring_mask_classification"]
    findings = [
        f"All {train['summary']['rows']:,} training rows and all {valid['summary']['rows']:,} validation rows were streamed. Exact feature moments, activity rates, target counts, target moments, sequence summaries, and validation-mask summaries use the complete files.",
        f"No Arrow nulls or nonfinite feature values were found in either complete dataset. This is an exact full-file result.",
        f"Validation scores {valid['summary']['scored_rows']:,} rows: {valid['summary']['scored_fraction_post_warmup']:.3%} of required prediction rows. Scored runs have median length {valid['summary']['scored_run_lengths']['median']} and 90th percentile {valid['summary']['scored_run_lengths']['p90']} rows.",
        f"The full training target correlation is {train['summary']['target_statistics_post_warmup']['target_correlation']:.4f}; the correlation on scored validation rows is {valid['summary']['target_statistics_scored']['target_correlation']:.4f}.",
        f"A sequence-held-out feature-only domain diagnostic reached AUC {domain['logistic_heldout_auc']:.4f} (logistic) and {domain['shallow_boosting_heldout_auc']:.4f} (shallow boosting). This measures separability of sampled train and validation rows, not hidden-test similarity.",
        f"A sequence-held-out feature-only validation-mask diagnostic reached AUC {mask['logistic_heldout_auc']:.4f} and {mask['shallow_boosting_heldout_auc']:.4f}. The mask is unavailable to submissions; this is descriptive only.",
        f"The largest sampled marginal distribution shift is {shifts[0]['feature']} (KS distance {shifts[0]['ks_distance']:.4f}). KS values are effect sizes on bounded row samples; dependent rows invalidate ordinary IID p-value interpretations.",
    ]
    for target in TARGETS:
        stats = valid["summary"]["target_statistics_scored"][target]
        findings.append(
            f"For scored validation {target}, zero targets are {stats['zero_fraction']:.2%}, values outside ±2 are {stats['outside_clip_fraction']:.2%}, and the largest-weight 10% of rows supply {stats['largest_10pct_rows_weight_share']:.2%} of total metric weight."
        )
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Alfa Connectome full EDA</title><style>
body{max-width:1160px;margin:36px auto;padding:0 24px;font:16px/1.58 system-ui;color:#172b3a;background:#fafbfd}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:8px;border:1px solid #d5dce3;text-align:left}th{background:#e8eef5}img{max-width:100%;background:white;border:1px solid #d5dce3;margin:18px 0}.scope{background:#e8f1fa;padding:18px;border-left:4px solid #3366aa}code{background:#edf1f4;padding:2px 5px}li{margin-bottom:9px}</style></head><body>"""
    page += "<h1>Alfa Connectome exploratory data analysis — full datasets</h1>"
    page += "<p class='scope'><b>Scope:</b> complete training and validation datasets. Exact streaming statistics are explicitly separated from bounded-sample multivariate analyses. The validation scoring mask and targets are used for evaluation research only and are unavailable at test inference.</p>"
    page += "<h2>Measured findings</h2><ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in findings) + "</ul>"
    page += "<h2>Dataset metadata</h2>" + html_table([
        {"file": item["path"], "rows": f"{item['rows']:,}", "sequences": f"{item['row_groups']:,}", "columns": item["columns"], "GB": f"{item['bytes']/1e9:.3f}"}
        for item in summary["metadata"]
    ], ["file", "rows", "sequences", "columns", "GB"])
    page += "<h2>Exact target summaries</h2>" + html_table(target_rows, list(target_rows[0]))
    page += "<h2>Figures</h2>"
    for image in images:
        encoded = base64.b64encode(image.read_bytes()).decode()
        page += f"<img alt='{html.escape(image.stem)}' src='data:image/png;base64,{encoded}'>"
    page += "<h2>Largest sampled feature-distribution differences</h2>" + html_table([
        {k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()} for row in shifts[:25]
    ], list(shifts[0]))
    page += """<h2>Interpretation limits</h2><ul>
<li>Target and mask statistics are exact, but they do not reveal the hidden test distribution or target definition.</li>
<li>PCA, feature correlations, lagged feature-target correlations, KS distances, and classifiers use 64 deterministic rows per sequence. Nearby rows remain dependent; these results are exploratory and carry no IID significance claims.</li>
<li>Sequence IDs are grouping keys, not chronological features. Any model feature must remain causal within a sequence.</li>
<li>The repeated value −5.199337 is measured exactly, but its semantic meaning is undocumented. Treat a sentinel indicator as a hypothesis to test.</li>
<li>The public validation mask should remain an evaluation tool. Optimizing specifically for a predictable public mask risks overfitting a mechanism that may differ on hidden test data.</li>
</ul><h2>Recommended next experiments</h2><ol>
<li>Weighted ridge and small MLP references on raw features, evaluated with official global WP.</li>
<li>Ablate causal first differences, exponential averages, and repeated-value indicators one family at a time.</li>
<li>Train a compact GRU with sequence-level state handling; compare shared versus target-specific heads.</li>
<li>Compare MSE, target-specific weighted MSE, and a stable mixed correlation/regression objective.</li>
<li>Measure every candidate on identical sequences and benchmark exported row-at-a-time inference.</li>
</ol>"""
    page += "<h2>Reproduction</h2><p>Run <code>python scripts/run_eda_full.py</code>. The script refuses to overwrite its output directory. Machine-readable summaries and CSV tables accompany this report.</p>"
    page += f"<p>Seed: {SEED}. Script SHA-256 and source metadata are recorded in <code>summary.json</code>.</p></body></html>"
    (output / "report.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("experiments/eda/eda_full_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    paths = {name: root / "datasets" / f"{name}.parquet" for name in ("train", "valid", "valid_mask")}
    exact_metadata = [metadata(paths[name]) for name in ("train", "valid", "valid_mask")]
    print("Streaming every training sequence...", flush=True)
    train = analyze(paths["train"], "train", output)
    print("Streaming every validation sequence...", flush=True)
    valid = analyze(paths["valid"], "valid", output)
    if set(int(r["seq_ix"]) for r in train["sequences"]) & set(int(r["seq_ix"]) for r in valid["sequences"]):
        raise ValueError("Training and validation sequence IDs overlap")

    # Confirm the separate mask is byte-for-byte equivalent at the table level.
    print("Verifying the full standalone validation mask...", flush=True)
    mask_file = pq.ParquetFile(paths["valid_mask"])
    valid_file = pq.ParquetFile(paths["valid"])
    mask_matches = True
    for group in range(mask_file.num_row_groups):
        left = mask_file.read_row_group(group)
        right = valid_file.read_row_group(group, columns=left.column_names)
        if not left.equals(right):
            mask_matches = False
            break
        if (group + 1) % 500 == 0:
            print(f"mask: {group + 1:,}/{mask_file.num_row_groups:,} sequences", flush=True)
    if not mask_matches:
        raise ValueError("valid_mask.parquet differs from valid.parquet")

    shifts = []
    for j, feature in enumerate(FEATURES):
        a = train["x"][:, j]
        b = valid["x"][:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        train_std = max(float(np.std(a, dtype=np.float64)), 1e-8)
        shifts.append({
            "feature": feature,
            "ks_distance": float(ks_2samp(a, b, method="asymp").statistic),
            "mean_difference_in_train_std": float((np.mean(b, dtype=np.float64) - np.mean(a, dtype=np.float64)) / train_std),
            "train_sample_mean": float(np.mean(a, dtype=np.float64)),
            "validation_sample_mean": float(np.mean(b, dtype=np.float64)),
        })
    shifts.sort(key=lambda row: row["ks_distance"], reverse=True)
    write_csv(output / "feature_shift.csv", shifts)

    training_x = np.where(np.isfinite(train["x"]), train["x"], 0)
    scaler = StandardScaler().fit(training_x)
    pca = PCA(n_components=112, svd_solver="full").fit(scaler.transform(training_x))
    correlation = np.corrcoef(training_x, rowvar=False)
    write_csv(output / "train_feature_correlations.csv", [
        {"feature": feature, **{other: float(correlation[j, k]) for k, other in enumerate(FEATURES)}}
        for j, feature in enumerate(FEATURES)
    ])

    print("Fitting sequence-held-out diagnostic classifiers...", flush=True)
    # Equalize domains so AUC reflects separation rather than dataset size.
    sample_count = min(len(train["x"]), len(valid["x"]))
    rng = np.random.default_rng(SEED)
    train_pick = np.sort(rng.choice(len(train["x"]), sample_count, replace=False))
    domain_x = np.r_[train["x"][train_pick], valid["x"]]
    domain_y = np.r_[np.zeros(sample_count, np.int8), np.ones(len(valid["x"]), np.int8)]
    domain_groups = np.r_[np.char.add("train_", train["groups"][train_pick].astype(str)), np.char.add("valid_", valid["groups"].astype(str))]
    domain = classifier_diagnostic(domain_x, domain_y, domain_groups, SEED)
    mask = classifier_diagnostic(valid["x"], valid["scored"].astype(np.int8), valid["groups"], SEED)

    summary = {
        "version": 1,
        "scope": "complete train.parquet and valid.parquet",
        "seed": SEED,
        "metadata": exact_metadata,
        "training": train["summary"],
        "validation": valid["summary"],
        "standalone_validation_mask_matches_full_validation": mask_matches,
        "domain_classification": domain,
        "scoring_mask_classification": mask,
        "largest_feature_shifts": shifts[:20],
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sampling_note": f"Exact streaming statistics use every row. Multivariate diagnostics use {SAMPLE_ROWS_PER_SEQUENCE} deterministic rows per sequence from steps 1024..19999.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Save a checkpoint before plotting/report rendering so completed full-file
    # statistics survive any presentation-layer failure.
    summary["elapsed_seconds_before_rendering"] = time.perf_counter() - started
    write_json(output / "summary_checkpoint.json", summary)
    images = create_figures(train, valid, shifts, pca, output)
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "summary.json", summary)
    create_report(output, summary, train, valid, shifts, images)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pyarrow", "matplotlib", "scipy", "scikit-learn", "onnxruntime")}
    (output / "environment.txt").write_text("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "elapsed_seconds": summary["elapsed_seconds"],
        "training_rows": train["summary"]["rows"],
        "validation_rows": valid["summary"]["rows"],
        "domain_classifier": domain,
        "mask_classifier": mask,
        "top_shifts": shifts[:3],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
