"""Evaluate the frozen final H64 candidates exactly once on provided validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch


FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{i}" for i in range(8)]
TARGETS = ("t0", "t1")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment_dir = root / "experiments/models/final_gru_h64_two_seed_all_train_v1"
    output_path = experiment_dir / "validation_evaluation.json"
    if output_path.exists():
        raise FileExistsError("refusing to repeat final validation evaluation")
    selection = json.loads((experiment_dir / "selection.json").read_text(encoding="utf-8"))
    paths = {
        name: root / record["path"]
        for name, record in selection["selected_artifacts"].items()
    }

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from compact_gru import CompactGRU
    from train_compact_gru import MetricAccumulator

    models = {}
    for name, path in paths.items():
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        model = CompactGRU(state["feature_mean"], state["feature_scale"], "target_specific",
                           input_size=112, hidden_size=64)
        model.load_state_dict(state)
        model.eval()
        models[name] = model

    parquet = pq.ParquetFile(root / "datasets/valid.parquet")
    expected = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != expected or parquet.num_row_groups != 1_873:
        raise ValueError("unexpected valid.parquet schema or sequence count")
    metrics = {name: MetricAccumulator() for name in (*models, "ensemble_50_50")}
    batch_size, chunk_length = 8, 1024
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_start in range(0, parquet.num_row_groups, batch_size):
            groups = list(range(batch_start, min(batch_start + batch_size, parquet.num_row_groups)))
            x = np.empty((len(groups), 20_000, 112), np.float32)
            y = np.empty((len(groups), 20_000, 2), np.float32)
            scored = np.empty((len(groups), 20_000), bool)
            for index, group in enumerate(groups):
                table = parquet.read_row_group(group, columns=expected, use_threads=False)
                steps = table["step_in_seq"].to_numpy()
                need = table["need_prediction"].to_numpy().astype(bool)
                mask = table["is_scored"].to_numpy().astype(bool)
                if table.num_rows != 20_000 or not np.array_equal(steps, np.arange(20_000)):
                    raise ValueError("invalid validation sequence")
                if not np.array_equal(need, steps >= 99) or np.any(mask[:99]):
                    raise ValueError("invalid validation masks")
                x[index] = np.column_stack([table[column].to_numpy() for column in FEATURES])
                y[index] = np.column_stack([table[column].to_numpy() for column in TARGETS])
                scored[index] = need & mask
            x_tensor = torch.from_numpy(x)
            hidden = {name: model.initial_hidden(len(groups)) for name, model in models.items()}
            pieces = {name: [] for name in models}
            for start in range(0, 20_000, chunk_length):
                stop = min(20_000, start + chunk_length)
                for name, model in models.items():
                    prediction, hidden[name] = model(x_tensor[:, start:stop], hidden[name])
                    pieces[name].append(prediction)
            predictions = {name: torch.cat(value, 1).numpy() for name, value in pieces.items()}
            target = y[scored]
            for name, prediction in predictions.items():
                metrics[name].add(target, prediction[scored])
            metrics["ensemble_50_50"].add(
                target, (0.5 * (predictions["seed_20261102"] + predictions["seed_20261105"]))[scored]
            )
            done = min(batch_start + batch_size, parquet.num_row_groups)
            if done % 64 == 0 or done == parquet.num_row_groups:
                print(f"validation: {done:,}/{parquet.num_row_groups:,}, {(time.perf_counter()-started)/60:.1f} min", flush=True)

    results = {name: metric.result() for name, metric in metrics.items()}
    selected = max(results, key=lambda name: results[name]["weighted_pearson"])
    record = {
        "policy": "single final evaluation on need_prediction AND is_scored rows; no tuning",
        "sequences": parquet.num_row_groups,
        "results": results,
        "selected_candidate": selected,
        "selected_score": results[selected]["weighted_pearson"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    selection["valid_parquet_evaluated"] = True
    selection["validation_selected_candidate"] = selected
    selection["validation_selected_score"] = record["selected_score"]
    (experiment_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
