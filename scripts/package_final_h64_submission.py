"""Export and verify the validation-selected final H64 submission package."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def solution_source(model_names: list[str]) -> str:
    return f'''"""Stateful ONNX inference for the selected final H64 GRU candidate."""
from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np
import onnxruntime as ort

MODEL_NAMES = {model_names!r}

class PredictionModel:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        base = Path(__file__).resolve().parent
        self.sessions = [ort.InferenceSession(str(base / name), sess_options=options,
                         providers=["CPUExecutionProvider"]) for name in MODEL_NAMES]
        self.hidden = [np.zeros((1, 1, 64), np.float32) for _ in self.sessions]
        self.current_seq_ix = None
        self.previous_step = None

    def predict(self, data_point):
        seq_ix, step = int(data_point.seq_ix), int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix, self.previous_step = seq_ix, None
            for hidden in self.hidden:
                hidden.fill(0.0)
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step
        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")
        predictions = []
        for index, session in enumerate(self.sessions):
            prediction, self.hidden[index] = session.run(None, {{
                "features": state.reshape(1, 1, 112), "hidden": self.hidden[index]
            }})
            predictions.append(prediction[0, 0])
        if not data_point.need_prediction:
            return None
        return np.mean(np.stack(predictions), axis=0, dtype=np.float32)
'''


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment_dir = root / "experiments/models/final_gru_h64_two_seed_all_train_v1"
    validation = json.loads((experiment_dir / "validation_evaluation.json").read_text(encoding="utf-8"))
    selection = json.loads((experiment_dir / "selection.json").read_text(encoding="utf-8"))
    candidate = validation["selected_candidate"]
    artifacts = selection["selected_artifacts"]
    chosen = list(artifacts) if candidate == "ensemble_50_50" else [candidate]
    if any(name not in artifacts for name in chosen):
        raise ValueError("validation selected an unknown candidate")

    output_dir = root / "artifacts/submissions/final_gru_h64_two_seed_all_train_v1"
    zip_path = output_dir / "h64_gru_two_seed_ensemble_full_train.zip"
    record_path = experiment_dir / "submission_export.json"
    solution_path = output_dir / "solution.py"
    if output_dir.exists() or record_path.exists():
        raise FileExistsError("refusing to overwrite an existing final submission export")
    output_dir.mkdir(parents=True)

    sys.path.insert(0, str(root / "src"))
    from compact_gru import CompactGRU
    rng = np.random.default_rng(20261106)
    test_x = rng.normal(size=(1, 257, 112)).astype(np.float32)
    test_h = rng.normal(size=(1, 1, 64)).astype(np.float32)
    exports = []
    model_names = []
    for index, name in enumerate(chosen):
        checkpoint_path = root / artifacts[name]["path"]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
        model = CompactGRU(state["feature_mean"], state["feature_scale"], "target_specific",
                           input_size=112, hidden_size=64)
        model.load_state_dict(state)
        model.eval()
        model_name = f"model_{index + 1}.onnx"
        model_names.append(model_name)
        onnx_path = output_dir / model_name
        torch.onnx.export(
            model, (torch.from_numpy(test_x[:, :16]), torch.zeros((1, 1, 64))), onnx_path,
            input_names=["features", "hidden"], output_names=["prediction", "next_hidden"],
            dynamic_axes={"features": {0: "batch", 1: "steps"}, "hidden": {1: "batch"},
                          "prediction": {0: "batch", 1: "steps"}, "next_hidden": {1: "batch"}},
            opset_version=17, dynamo=False,
        )
        onnx.checker.check_model(onnx.load(onnx_path))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            torch_p, torch_h = model(torch.from_numpy(test_x), torch.from_numpy(test_h))
        ort_p, ort_h = session.run(None, {"features": test_x, "hidden": test_h})
        p_error = float(np.max(np.abs(torch_p.numpy() - ort_p)))
        h_error = float(np.max(np.abs(torch_h.numpy() - ort_h)))
        if p_error > 1e-5 or h_error > 1e-5:
            raise ValueError("ONNX parity failure")
        exports.append({"candidate": name, "checkpoint": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
                        "onnx": model_name, "onnx_sha256": sha256(onnx_path),
                        "max_prediction_error": p_error, "max_hidden_error": h_error})

    solution_path.write_text(solution_source(model_names), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(solution_path, "solution.py")
        for name in model_names:
            archive.write(output_dir / name, name)

    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temporary)
        spec = importlib.util.spec_from_file_location("final_packaged_solution", Path(temporary) / "solution.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        predictor = module.PredictionModel()
        outputs = []
        states = rng.normal(size=(101, 112)).astype(np.float32)
        for step, state in enumerate(states):
            point = type("Point", (), {"seq_ix": 123, "step_in_seq": step,
                                       "need_prediction": step >= 99, "state": state})()
            outputs.append(predictor.predict(point))
        if any(value is not None for value in outputs[:99]):
            raise ValueError("packaged warm-up contract failure")
        if any(value is None or np.asarray(value).shape != (2,) or not np.isfinite(value).all()
               for value in outputs[99:]):
            raise ValueError("packaged prediction contract failure")

    record = {
        "selected_candidate": candidate,
        "validation_score": validation["selected_score"],
        "models": exports,
        "solution_sha256": sha256(solution_path),
        "submission": {"path": str(zip_path.relative_to(root)).replace("\\", "/"),
                       "sha256": sha256(zip_path), "bytes": zip_path.stat().st_size},
        "verification": {"warmup_rows": 99, "verified_prediction_rows": 2},
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
