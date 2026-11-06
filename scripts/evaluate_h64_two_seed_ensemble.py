"""Compare the two selected H64 GRU seeds and their fixed 50/50 ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
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
    parser.add_argument("--partition", choices=("development", "confirmation"), default="development")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "experiments/models/train_only_gru_h64_two_seed_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.partition}_comparison.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    split_path = root / "experiments/models/train_only_causal_ablation_v1/manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["split"]
    groups = [int(value) for value in split[f"{args.partition}_source_row_groups"]]
    seed_paths = {
        "seed_20261102": root / "artifacts/train_only_compact_gru_full_v1/compact_gru_target_specific_epoch_3.pt",
        "seed_20261105": root / "artifacts/train_only_gru_h64_seed20261105_v1/gru_h64_epoch_3.pt",
    }

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from compact_gru import CompactGRU
    from train_compact_gru import FEATURES, TARGETS, MetricAccumulator, batches, read_batch

    models = {}
    for name, path in seed_paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        model = CompactGRU(state["feature_mean"], state["feature_scale"], "target_specific",
                           input_size=112, hidden_size=64)
        model.load_state_dict(state)
        model.eval()
        models[name] = model

    metrics = {name: MetricAccumulator() for name in (*models, "ensemble_50_50")}
    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    started = time.perf_counter()
    sequence_ids = []
    with torch.inference_mode():
        for batch_index, group_batch in enumerate(batches(groups, args.batch_size), 1):
            x_np, y_np, ids = read_batch(parquet, group_batch, columns)
            sequence_ids.extend(ids)
            x = torch.from_numpy(x_np)
            hidden = {name: model.initial_hidden(len(group_batch)) for name, model in models.items()}
            pieces = {name: [] for name in models}
            for start in range(0, 20_000, args.chunk_length):
                stop = min(20_000, start + args.chunk_length)
                for name, model in models.items():
                    prediction, hidden[name] = model(x[:, start:stop], hidden[name])
                    pieces[name].append(prediction)
            target = y_np[:, 99:].reshape(-1, 2)
            predictions = {
                name: torch.cat(value, 1).numpy()[:, 99:].reshape(-1, 2)
                for name, value in pieces.items()
            }
            for name, prediction in predictions.items():
                metrics[name].add(target, prediction)
            metrics["ensemble_50_50"].add(
                target, 0.5 * (predictions["seed_20261102"] + predictions["seed_20261105"])
            )
            if batch_index % 8 == 0 or batch_index * args.batch_size >= len(groups):
                done = min(batch_index * args.batch_size, len(groups))
                print(f"{args.partition}: {done:,}/{len(groups):,}, {(time.perf_counter()-started)/60:.1f} min", flush=True)

    results = {name: accumulator.result() for name, accumulator in metrics.items()}
    best_name = max(results, key=lambda name: results[name]["weighted_pearson"])
    record = {
        "partition": args.partition,
        "policy": "fixed 50/50 ensemble; no ensemble-weight tuning",
        "sequences": len(groups),
        "sequence_ids": sequence_ids,
        "checkpoints": {
            name: {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256(path)}
            for name, path in seed_paths.items()
        },
        "results": results,
        "selected_candidate": best_name,
        "selected_score": results[best_name]["weighted_pearson"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results, "selected_candidate": best_name,
                      "selected_score": record["selected_score"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
