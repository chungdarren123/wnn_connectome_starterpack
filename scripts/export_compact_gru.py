"""Export and verify the selected compact GRU as a stateful ONNX package."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint_path = root / "artifacts/train_only_compact_gru_v1/compact_gru_target_specific.pt"
    onnx_path = root / "artifacts/train_only_compact_gru_v1/compact_gru_target_specific.onnx"
    zip_dir = root / "artifacts/submissions/train_only_compact_gru_v1"
    zip_path = zip_dir / "compact_gru_target_specific.zip"
    solution_path = root / "solutions/train_only_compact_gru_v1/target_specific/solution.py"
    record_path = root / "experiments/models/train_only_compact_gru_v1/export.json"
    if onnx_path.exists() or zip_path.exists() or record_path.exists():
        raise FileExistsError("refusing to overwrite an existing compact GRU export")

    sys.path.insert(0, str(root / "src"))
    from compact_gru import CompactGRU
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    model = CompactGRU(state["feature_mean"], state["feature_scale"], "target_specific")
    model.load_state_dict(state)
    model.eval()
    features = torch.zeros((1, 16, 112), dtype=torch.float32)
    hidden = torch.zeros((1, 1, 64), dtype=torch.float32)
    torch.onnx.export(
        model, (features, hidden), onnx_path,
        input_names=["features", "hidden"],
        output_names=["prediction", "next_hidden"],
        dynamic_axes={"features": {0: "batch", 1: "steps"}, "hidden": {1: "batch"},
                      "prediction": {0: "batch", 1: "steps"}, "next_hidden": {1: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20261102)
    test_x = rng.normal(size=(2, 257, 112)).astype(np.float32)
    test_h = rng.normal(size=(1, 2, 64)).astype(np.float32)
    with torch.inference_mode():
        torch_p, torch_h = model(torch.from_numpy(test_x), torch.from_numpy(test_h))
    ort_p, ort_h = session.run(None, {"features": test_x, "hidden": test_h})
    max_prediction_error = float(np.max(np.abs(torch_p.numpy() - ort_p)))
    max_hidden_error = float(np.max(np.abs(torch_h.numpy() - ort_h)))
    if max_prediction_error > 1e-5 or max_hidden_error > 1e-5:
        raise ValueError("ONNX output differs from PyTorch")

    zip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(solution_path, "solution.py")
        archive.write(onnx_path, "model.onnx")
    with zipfile.ZipFile(zip_path) as archive:
        if sorted(archive.namelist()) != ["model.onnx", "solution.py"]:
            raise ValueError("unexpected submission contents")
        with tempfile.TemporaryDirectory() as temporary:
            archive.extractall(temporary)
            spec = importlib.util.spec_from_file_location("packaged_solution", Path(temporary) / "solution.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            predictor = module.PredictionModel()
            split = json.loads((root / "experiments/models/train_only_causal_ablation_v1/manifest.json").read_text())
            group = int(split["split"]["development_source_row_groups"][0])
            columns = ["seq_ix", "step_in_seq", "need_prediction"]
            parquet = pq.ParquetFile(root / "datasets/train.parquet")
            table = parquet.read_row_group(group)
            feature_names = table.column_names[3:-2]
            sequence_id = int(table["seq_ix"][0].as_py())
            outputs = []
            for step in range(101):
                point = type("Point", (), {
                    "seq_ix": sequence_id,
                    "step_in_seq": step,
                    "need_prediction": step >= 99,
                    "state": np.asarray([table[name][step].as_py() for name in feature_names], np.float32),
                })()
                outputs.append(predictor.predict(point))
            if any(value is not None for value in outputs[:99]):
                raise ValueError("warm-up contract failed")
            if any(value is None or np.asarray(value).shape != (2,) or not np.isfinite(value).all() for value in outputs[99:]):
                raise ValueError("prediction contract failed")

    record = {
        "checkpoint": {"path": str(checkpoint_path.relative_to(root)), "sha256": sha256(checkpoint_path)},
        "onnx": {"path": str(onnx_path.relative_to(root)), "sha256": sha256(onnx_path), "bytes": onnx_path.stat().st_size},
        "submission": {"path": str(zip_path.relative_to(root)), "sha256": sha256(zip_path), "bytes": zip_path.stat().st_size},
        "verification": {"max_prediction_error": max_prediction_error, "max_hidden_error": max_hidden_error,
                         "warmup_rows": 99, "verified_prediction_rows": 2},
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
