"""Resume the selected compact GRU for one all-fit refinement epoch.

Only datasets/train.parquet is opened. The model resumes from epoch 2, trains
on all fit sequences at 1e-4, and evaluates development. Confirmation remains
unread and provided validation files are never opened.
"""
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
SEED = 20261102
REFINEMENT_LEARNING_RATE = 1e-4


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
    artifact_dir = root / "artifacts/train_only_compact_gru_full_v1"
    experiment_dir = root / "experiments/models/train_only_compact_gru_full_v1"
    source_checkpoint = artifact_dir / "compact_gru_target_specific_epoch_2.pt"
    target_checkpoint = artifact_dir / "compact_gru_target_specific_epoch_3.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError("epoch-2 checkpoint is missing")
    if target_checkpoint.exists():
        raise FileExistsError("refusing to overwrite an existing epoch-3 checkpoint")

    split_path = root / "experiments/models/train_only_causal_ablation_v1/manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["split"]
    fit_groups = [int(value) for value in split["fit_source_row_groups"]]
    development_groups = [int(value) for value in split["development_source_row_groups"]]
    confirmation_groups = [int(value) for value in split["confirmation_source_row_groups"]]
    if set(fit_groups) & set(development_groups) or set(fit_groups) & set(confirmation_groups) or set(development_groups) & set(confirmation_groups):
        raise ValueError("overlapping sequence partitions")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from compact_gru import CompactGRU, count_parameters
    from train_compact_gru import FEATURES, TARGETS, evaluate, train_epoch

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    model = CompactGRU(state["feature_mean"], state["feature_scale"], "target_specific")
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=REFINEMENT_LEARNING_RATE, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for group in optimizer.param_groups:
        group["lr"] = REFINEMENT_LEARNING_RATE
    models = {"target_specific": model}
    optimizers = {"target_specific": optimizer}

    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != columns:
        raise ValueError("unexpected training schema")
    print(f"resumed epoch 2: {count_parameters(model):,} parameters", flush=True)
    print(f"refinement lr={REFINEMENT_LEARNING_RATE}; fit={len(fit_groups):,}; development={len(development_groups):,}; confirmation reserved", flush=True)

    started = time.perf_counter()
    model.train()
    losses, fitted_ids = train_epoch(models, optimizers, parquet, fit_groups, columns,
                                     args.batch_size, args.chunk_length, epoch=2)
    training_seconds = time.perf_counter() - started
    model.eval()
    development, development_ids = evaluate(models, parquet, development_groups, columns,
                                            args.batch_size, args.chunk_length)
    torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": {"input_size": 112, "hidden_size": 64, "head_type": "target_specific"},
                "seed": SEED, "completed_epoch": 3,
                "learning_rates": [1e-3, 3e-4, REFINEMENT_LEARNING_RATE]}, target_checkpoint)
    artifact = {"epoch": 3, "path": str(target_checkpoint.relative_to(root)).replace(os.sep, "/"),
                "sha256": sha256(target_checkpoint), "bytes": target_checkpoint.stat().st_size}
    record = {"epoch": 3, "learning_rate": REFINEMENT_LEARNING_RATE,
              "training_loss": losses["target_specific"], "training_seconds": training_seconds,
              "development": development["target_specific"], "fit_sequence_ids": fitted_ids,
              "development_sequence_ids": development_ids, "artifact": artifact}

    metrics_path = experiment_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(metrics["epochs"]) != 2:
        raise ValueError("expected exactly two completed epochs before refinement")
    metrics["epochs"].append(record)
    metrics["elapsed_seconds_before_refinement"] = metrics.pop("elapsed_seconds")
    metrics["refinement_elapsed_seconds"] = time.perf_counter() - started
    metrics["total_elapsed_seconds"] = metrics["elapsed_seconds_before_refinement"] + metrics["refinement_elapsed_seconds"]
    write_json(metrics_path, metrics)
    best = max(metrics["epochs"], key=lambda value: value["development"]["weighted_pearson"])
    write_json(experiment_dir / "selection.json", {
        "criterion": "highest development weighted Pearson",
        "selected_epoch": best["epoch"], "selected_score": best["development"]["weighted_pearson"],
        "selected_artifact": best["artifact"], "confirmation_evaluated": False,
    })
    write_json(experiment_dir / "refinement_epoch_3.json", {
        "source_checkpoint": {"path": str(source_checkpoint.relative_to(root)).replace(os.sep, "/"),
                              "sha256": sha256(source_checkpoint)},
        "result": record, "selected_epoch_after_refinement": best["epoch"],
        "confirmation_evaluated": False,
    })
    print(json.dumps({"epoch": 3, "loss": record["training_loss"],
                      "development": record["development"], "artifact": artifact,
                      "selected_epoch": best["epoch"]}, indent=2), flush=True)
    print(f"Refinement finished in {(time.perf_counter() - started) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
