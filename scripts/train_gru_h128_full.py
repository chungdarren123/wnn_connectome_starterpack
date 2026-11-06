"""Fully train the selected raw-input 128-memory GRU.

Only train.parquet is opened. All fit sequences train the model, development
is evaluated after each epoch, and confirmation remains unread.
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
SEED = 20261104
LEARNING_RATES = (1e-3, 3e-4)


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
    parser.add_argument("--artifact-name", default="train_only_gru_h128_seed20261104_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_dir = root / "artifacts" / args.artifact_name
    experiment_dir = root / "experiments/models" / args.artifact_name
    if artifact_dir.exists() or experiment_dir.exists():
        raise FileExistsError("refusing to overwrite an existing h128 attempt")
    artifact_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)

    split_path = root / "experiments/models/train_only_causal_ablation_v1/manifest.json"
    scaler_path = root / "artifacts/train_only_causal_ablation_v1/mlp_raw.npz"
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

    scaler = np.load(scaler_path, allow_pickle=False)
    model = CompactGRU(torch.from_numpy(scaler["mean"].astype(np.float32)),
                       torch.from_numpy(scaler["scale"].astype(np.float32)),
                       "target_specific", input_size=112, hidden_size=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATES[0], weight_decay=1e-4)
    models, optimizers = {"raw_h128": model}, {"raw_h128": optimizer}
    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != columns:
        raise ValueError("unexpected training schema")

    print(f"raw_h128: {count_parameters(model):,} parameters; seed={SEED}", flush=True)
    print(f"fit={len(fit_groups):,}; development={len(development_groups):,}; confirmation={len(confirmation_groups):,} reserved", flush=True)
    started = time.perf_counter()
    history = []
    write_json(experiment_dir / "manifest.json", {
        "version": 1, "policy": "train.parquet only; confirmation unread", "seed": SEED,
        "source_split_manifest": str(split_path.relative_to(root)).replace(os.sep, "/"),
        "source_scaler": {"path": str(scaler_path.relative_to(root)).replace(os.sep, "/"), "sha256": sha256(scaler_path)},
        "partitions": {"fit_sequences": len(fit_groups), "development_sequences": len(development_groups),
                       "confirmation_sequences": len(confirmation_groups), "confirmation_evaluated": False},
        "architecture": {"input_size": 112, "hidden_size": 128, "gru_layers": 1,
                         "head": "two independent 128 -> 16 -> SiLU -> 1", "parameters": count_parameters(model)},
        "training": {"epochs": 2, "learning_rates": list(LEARNING_RATES), "batch_size": args.batch_size,
                     "chunk_length": args.chunk_length, "optimizer": "AdamW", "weight_decay": 1e-4,
                     "loss": "abs(clipped target)-weighted SmoothL1(beta=0.25)", "gradient_clip_norm": 1.0,
                     "warmup_steps_used_for_state_not_loss": 99, "threads": args.threads,
                     "torch_version": torch.__version__},
    })

    for epoch, learning_rate in enumerate(LEARNING_RATES):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        epoch_started = time.perf_counter()
        losses, fit_ids = train_epoch(models, optimizers, parquet, fit_groups, columns,
                                      args.batch_size, args.chunk_length, epoch)
        training_seconds = time.perf_counter() - epoch_started
        model.eval()
        development, development_ids = evaluate(models, parquet, development_groups, columns,
                                                args.batch_size, args.chunk_length)
        checkpoint_path = artifact_dir / f"gru_h128_epoch_{epoch + 1}.pt"
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "config": {"input_size": 112, "hidden_size": 128, "head_type": "target_specific"},
                    "seed": SEED, "completed_epoch": epoch + 1, "learning_rates": list(LEARNING_RATES)}, checkpoint_path)
        record = {"epoch": epoch + 1, "learning_rate": learning_rate,
                  "training_loss": losses["raw_h128"], "training_seconds": training_seconds,
                  "development": development["raw_h128"], "fit_sequence_ids": fit_ids,
                  "development_sequence_ids": development_ids,
                  "artifact": {"path": str(checkpoint_path.relative_to(root)).replace(os.sep, "/"),
                               "sha256": sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size}}
        history.append(record)
        write_json(experiment_dir / "metrics.json", {
            "evaluation": "all eligible development rows after each epoch; confirmation not evaluated",
            "epochs": history, "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps({"epoch": epoch + 1, "loss": record["training_loss"],
                          "development": record["development"], "artifact": record["artifact"]}, indent=2), flush=True)

    best = max(history, key=lambda value: value["development"]["weighted_pearson"])
    write_json(experiment_dir / "selection.json", {
        "criterion": "highest development weighted Pearson", "selected_epoch": best["epoch"],
        "selected_score": best["development"]["weighted_pearson"],
        "selected_artifact": best["artifact"], "confirmation_evaluated": False})
    print(f"Selected epoch {best['epoch']} at {best['development']['weighted_pearson']:.9f}", flush=True)
    print(f"Finished in {(time.perf_counter()-started)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
