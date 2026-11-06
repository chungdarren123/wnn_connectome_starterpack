"""Verify exported static models using train-only holdout sequences."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{i}" for i in range(8)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments/models/train_only_static_v1"
    manifest = json.loads((experiment / "manifest.json").read_text())
    metrics = json.loads((experiment / "metrics.json").read_text())
    if manifest["policy"] != "train.parquet only; public validation files never opened":
        raise ValueError("unexpected experiment policy")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from utils import DataPoint
    from static_models import RidgePredictionModel, MLPPredictionModel

    paths = {
        "ridge": root / metrics["artifacts"]["ridge"]["path"],
        "mlp": root / metrics["artifacts"]["mlp"]["path"],
    }
    classes = {"ridge": RidgePredictionModel, "mlp": MLPPredictionModel}
    for name, path in paths.items():
        if sha256(path) != metrics["artifacts"][name]["sha256"]:
            raise ValueError(f"{name} artifact checksum mismatch")

    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    groups = manifest["split"]["holdout_source_row_groups"][:5]
    results = {}
    for name in ("ridge", "mlp"):
        model = classes[name](paths[name])
        maximum_difference = 0.0
        required_predictions = 0
        rows = 0
        started = time.perf_counter()
        for group in groups:
            table = parquet.read_row_group(group, columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURES])
            seq_ix = int(table["seq_ix"][0].as_py())
            features = np.column_stack([table[c].to_numpy() for c in FEATURES]).astype(np.float32)
            batch = model.batch_predict(features)
            streamed = np.full((20_000, 2), np.nan, np.float32)
            for step in range(20_000):
                need = step >= 99
                value = model.predict(DataPoint(seq_ix, step, need, features[step]))
                if not need:
                    if value is not None:
                        raise ValueError(f"{name} returned a warm-up prediction")
                else:
                    value = np.asarray(value, np.float32)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        raise ValueError(f"{name} returned an invalid prediction")
                    streamed[step] = value
            maximum_difference = max(maximum_difference, float(np.max(np.abs(streamed[99:] - batch[99:]))))
            required_predictions += 19_901
            rows += 20_000
        elapsed = time.perf_counter() - started
        if maximum_difference > 2e-5:
            raise ValueError(f"{name} streamed and batch predictions differ")
        results[name] = {
            "sequences": len(groups),
            "rows": rows,
            "required_predictions": required_predictions,
            "row_at_a_time_seconds": elapsed,
            "microseconds_per_row": elapsed / rows * 1e6,
            "rough_39_4m_row_minutes_on_this_machine": elapsed / rows * 39_400_000 / 60,
            "maximum_stream_vs_batch_abs_difference": maximum_difference,
            "artifact_sha256_verified": True,
        }
    output = {
        "scope": "first five fixed train-only holdout sequences",
        "note": "Runtime extrapolation is local-machine guidance, not a grader guarantee.",
        "models": results,
    }
    (experiment / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
