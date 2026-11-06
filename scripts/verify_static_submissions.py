"""Load each packaged submission and exercise it on train-only rows."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import pyarrow.parquet as pq

FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{i}" for i in range(8)]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from utils import DataPoint
    manifest = json.loads((root / "experiments/models/train_only_static_v1/manifest.json").read_text())
    group = manifest["split"]["holdout_source_row_groups"][0]
    table = pq.ParquetFile(root / "datasets/train.parquet").read_row_group(
        group, columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURES]
    )
    seq_ix = int(table["seq_ix"][0].as_py())
    features = np.column_stack([table[c].to_numpy() for c in FEATURES]).astype(np.float32)
    archives = root / "artifacts/submissions/train_only_static_v1"
    results = {}
    for name in ("weighted_ridge", "small_mlp"):
        archive_path = archives / f"{name}.zip"
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(directory)
            solution_path = Path(directory) / "solution.py"
            spec = importlib.util.spec_from_file_location(f"submission_{name}", solution_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.PredictionModel()
            outputs = []
            for step in range(101):
                value = model.predict(DataPoint(seq_ix, step, step >= 99, features[step]))
                if step < 99:
                    if value is not None:
                        raise ValueError(f"{name} returned output during warm-up")
                else:
                    value = np.asarray(value, np.float32)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        raise ValueError(f"{name} returned invalid output")
                    outputs.append(value.tolist())
            results[name] = {"archive_loaded": True, "first_two_predictions": outputs}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
