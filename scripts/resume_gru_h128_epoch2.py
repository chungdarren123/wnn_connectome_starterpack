"""Resume the full 128-memory GRU from its completed epoch-1 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.dont_write_bytecode = True
SEED = 20261104
LEARNING_RATE = 3e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_dir = root / "artifacts/train_only_gru_h128_seed20261104_v1"
    experiment_dir = root / "experiments/models/train_only_gru_h128_seed20261104_v1"
    source = artifact_dir / "gru_h128_epoch_1.pt"
    target = artifact_dir / "gru_h128_epoch_2.pt"
    if not source.exists():
        raise FileNotFoundError("epoch-1 checkpoint is missing")
    if target.exists():
        raise FileExistsError("refusing to overwrite epoch-2 checkpoint")

    split_path = root / "experiments/models/train_only_causal_ablation_v1/manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["split"]
    fit_groups = [int(value) for value in split["fit_source_row_groups"]]
    development_groups = [int(value) for value in split["development_source_row_groups"]]
    confirmation_groups = [int(value) for value in split["confirmation_source_row_groups"]]
    if set(fit_groups) & set(development_groups) or set(fit_groups) & set(confirmation_groups) or set(development_groups) & set(confirmation_groups):
        raise ValueError("overlapping partitions")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from compact_gru import CompactGRU, count_parameters
    from train_compact_gru import FEATURES, TARGETS, evaluate, train_epoch

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    model = CompactGRU(state["feature_mean"], state["feature_scale"], "target_specific",
                       input_size=112, hidden_size=128)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE
    models, optimizers = {"raw_h128": model}, {"raw_h128": optimizer}
    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != columns:
        raise ValueError("unexpected training schema")
    print(f"resumed h128 epoch 1: {count_parameters(model):,} parameters; seed={SEED}", flush=True)
    print(f"epoch 2 lr={LEARNING_RATE}; fit={len(fit_groups):,}; development={len(development_groups):,}; confirmation reserved", flush=True)

    started = time.perf_counter()
    model.train()
    losses, fit_ids = train_epoch(models, optimizers, parquet, fit_groups, columns,
                                  args.batch_size, args.chunk_length, epoch=1)
    training_seconds = time.perf_counter() - started
    model.eval()
    development, development_ids = evaluate(models, parquet, development_groups, columns,
                                            args.batch_size, args.chunk_length)
    torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": {"input_size": 112, "hidden_size": 128, "head_type": "target_specific"},
                "seed": SEED, "completed_epoch": 2, "learning_rates": [1e-3, LEARNING_RATE]}, target)
    record = {"epoch": 2, "learning_rate": LEARNING_RATE,
              "training_loss": losses["raw_h128"], "training_seconds": training_seconds,
              "development": development["raw_h128"], "fit_sequence_ids": fit_ids,
              "development_sequence_ids": development_ids,
              "artifact": {"path": str(target.relative_to(root)).replace(os.sep, "/"),
                           "sha256": sha256(target), "bytes": target.stat().st_size}}
    metrics_path = experiment_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(metrics["epochs"]) != 1 or metrics["epochs"][0]["epoch"] != 1:
        raise ValueError("expected one completed epoch")
    metrics["epochs"].append(record)
    metrics["elapsed_seconds_before_resume"] = metrics.pop("elapsed_seconds")
    metrics["resume_elapsed_seconds"] = time.perf_counter() - started
    metrics["total_elapsed_seconds"] = metrics["elapsed_seconds_before_resume"] + metrics["resume_elapsed_seconds"]
    write_json(metrics_path, metrics)
    best = max(metrics["epochs"], key=lambda value: value["development"]["weighted_pearson"])
    write_json(experiment_dir / "selection.json", {
        "criterion": "highest development weighted Pearson", "selected_epoch": best["epoch"],
        "selected_score": best["development"]["weighted_pearson"],
        "selected_artifact": best["artifact"], "confirmation_evaluated": False})
    write_json(experiment_dir / "resume_epoch_2.json", {
        "source_checkpoint": {"path": str(source.relative_to(root)).replace(os.sep, "/"), "sha256": sha256(source)},
        "result": record, "selected_epoch": best["epoch"], "confirmation_evaluated": False})
    print(json.dumps({"epoch": 2, "loss": record["training_loss"], "development": record["development"],
                      "artifact": record["artifact"], "selected_epoch": best["epoch"]}, indent=2), flush=True)
    print(f"Resume run finished in {(time.perf_counter()-started)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
