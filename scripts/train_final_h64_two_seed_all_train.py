"""Retrain the frozen two-seed H64 GRU ensemble on every train.parquet sequence."""
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


SEEDS = {"seed_20261102": 20261102, "seed_20261105": 20261105}
LEARNING_RATES = (1e-3, 3e-4, 1e-4)


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
    parser.add_argument("--artifact-name", default="final_gru_h64_two_seed_all_train_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_dir = root / "artifacts" / args.artifact_name
    experiment_dir = root / "experiments/models" / args.artifact_name
    if artifact_dir.exists() or experiment_dir.exists():
        raise FileExistsError("refusing to overwrite an existing final refit")
    artifact_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)

    scaler_path = root / "artifacts/train_only_causal_ablation_v1/mlp_raw.npz"
    scaler = np.load(scaler_path, allow_pickle=False)
    mean = torch.from_numpy(scaler["mean"].astype(np.float32))
    scale = torch.from_numpy(scaler["scale"].astype(np.float32))

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    np.random.seed(20261106)
    random.seed(20261106)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from compact_gru import CompactGRU, count_parameters
    from train_compact_gru import FEATURES, TARGETS, train_epoch

    models = {}
    optimizers = {}
    for name, seed in SEEDS.items():
        torch.manual_seed(seed)
        model = CompactGRU(mean, scale, "target_specific", input_size=112, hidden_size=64)
        models[name] = model
        optimizers[name] = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATES[0], weight_decay=1e-4)

    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != columns:
        raise ValueError("unexpected train.parquet schema")
    groups = list(range(parquet.num_row_groups))
    if len(groups) != 10_607 or parquet.metadata.num_rows != 212_140_000:
        raise ValueError("unexpected train.parquet size")

    manifest = {
        "version": 1,
        "role": "final refit after architecture and equal-weight ensemble were frozen",
        "data_policy": "all 10,607 sequences in train.parquet; valid.parquet unread",
        "source_scaler": {"path": str(scaler_path.relative_to(root)).replace(os.sep, "/"),
                          "sha256": sha256(scaler_path)},
        "seeds": SEEDS,
        "architecture": {"input_size": 112, "hidden_size": 64, "gru_layers": 1,
                         "head": "two independent 64 -> 16 -> SiLU -> 1",
                         "parameters_per_model": count_parameters(next(iter(models.values()))),
                         "ensemble": "fixed arithmetic mean of the two model predictions"},
        "training": {"sequences_per_epoch": len(groups), "epochs": 3,
                     "learning_rates": list(LEARNING_RATES), "batch_size": args.batch_size,
                     "chunk_length": args.chunk_length, "optimizer": "AdamW",
                     "weight_decay": 1e-4,
                     "loss": "abs(clipped target)-weighted SmoothL1(beta=0.25)",
                     "gradient_clip_norm": 1.0,
                     "warmup_steps_used_for_state_not_loss": 99, "threads": args.threads},
    }
    write_json(experiment_dir / "manifest.json", manifest)
    print(f"models={list(models)}; {count_parameters(next(iter(models.values()))):,} parameters each", flush=True)
    print(f"training all {len(groups):,} train sequences; validation unread", flush=True)

    started = time.perf_counter()
    history = []
    for epoch, learning_rate in enumerate(LEARNING_RATES):
        for optimizer in optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        for model in models.values():
            model.train()
        epoch_started = time.perf_counter()
        losses, sequence_ids = train_epoch(models, optimizers, parquet, groups, columns,
                                           args.batch_size, args.chunk_length, epoch)
        records = {}
        for name, model in models.items():
            checkpoint_path = artifact_dir / f"{name}_epoch_{epoch + 1}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizers[name].state_dict(),
                "config": {"input_size": 112, "hidden_size": 64, "head_type": "target_specific"},
                "seed": SEEDS[name],
                "completed_epoch": epoch + 1,
                "learning_rates": list(LEARNING_RATES),
                "trained_sequences": len(groups),
            }, checkpoint_path)
            records[name] = {"training_loss": losses[name],
                             "artifact": {"path": str(checkpoint_path.relative_to(root)).replace(os.sep, "/"),
                                          "sha256": sha256(checkpoint_path),
                                          "bytes": checkpoint_path.stat().st_size}}
        history.append({"epoch": epoch + 1, "learning_rate": learning_rate,
                        "training_seconds": time.perf_counter() - epoch_started,
                        "sequence_ids": sequence_ids, "models": records})
        write_json(experiment_dir / "metrics.json", {
            "evaluation": "none during final refit; fixed three-epoch schedule",
            "epochs": history, "elapsed_seconds": time.perf_counter() - started,
        })
        print(json.dumps({"epoch": epoch + 1, "learning_rate": learning_rate,
                          "models": records}, indent=2), flush=True)

    final = history[-1]["models"]
    write_json(experiment_dir / "selection.json", {
        "criterion": "frozen three-epoch two-seed recipe",
        "ensemble_weights": {name: 0.5 for name in SEEDS},
        "selected_artifacts": {name: record["artifact"] for name, record in final.items()},
        "valid_parquet_evaluated": False,
    })
    print(f"Finished final refit in {(time.perf_counter()-started)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
