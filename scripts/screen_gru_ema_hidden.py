"""Screen explicit EMA inputs and a 128-unit GRU on the train-only dev split.

Three target-specific-head models see the same 2,048 fit sequences for one
epoch: raw/64 control, raw+EMA(8,64)/64, and raw/128. Confirmation and provided
validation data remain unread.
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
from scipy.signal import lfilter
import torch

sys.dont_write_bytecode = True
SEED = 20261102
EMA_HALF_LIVES = (8, 64)
MODEL_NAMES = ("raw_h64", "ema_h64", "raw_h128")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ema(values: np.ndarray, half_life: int) -> np.ndarray:
    alpha = np.float32(1.0 - 2.0 ** (-1.0 / half_life))
    initial = ((1.0 - alpha) * values[0])[None, :]
    result, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], values, axis=0, zi=initial)
    return result.astype(np.float32, copy=False)


def add_ema(raw: np.ndarray) -> np.ndarray:
    result = np.empty((raw.shape[0], raw.shape[1], 336), np.float32)
    for index in range(raw.shape[0]):
        result[index] = np.concatenate((raw[index], ema(raw[index], 8), ema(raw[index], 64)), axis=1)
    return result


def batches(items: list[int], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def train(models, optimizers, parquet, groups, columns, read_batch, weighted_loss,
          batch_size, chunk_length):
    order = groups.copy()
    random.Random(SEED).shuffle(order)
    totals = {name: 0.0 for name in models}
    updates = {name: 0 for name in models}
    sequence_ids = []
    started = time.perf_counter()
    for batch_index, group_batch in enumerate(batches(order, batch_size), 1):
        raw_np, target_np, ids = read_batch(parquet, group_batch, columns)
        sequence_ids.extend(ids)
        ema_np = add_ema(raw_np)
        raw, ema_features, target = torch.from_numpy(raw_np), torch.from_numpy(ema_np), torch.from_numpy(target_np)
        hidden = {name: model.initial_hidden(len(group_batch)) for name, model in models.items()}
        for start in range(0, 20_000, chunk_length):
            stop = min(20_000, start + chunk_length)
            loss_start = max(start, 99)
            local_start = loss_start - start
            for name, model in models.items():
                optimizers[name].zero_grad(set_to_none=True)
                source = ema_features if name == "ema_h64" else raw
                prediction, next_hidden = model(source[:, start:stop], hidden[name])
                hidden[name] = next_hidden.detach()
                loss = weighted_loss(prediction[:, local_start:], target[:, loss_start:stop])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name] += float(loss.detach())
                updates[name] += 1
        if batch_index % 8 == 0 or batch_index * batch_size >= len(order):
            done = min(batch_index * batch_size, len(order))
            text = ", ".join(f"{name}={totals[name]/updates[name]:.5f}" for name in models)
            print(f"training: {done:,}/{len(order):,}, {text}, {(time.perf_counter()-started)/60:.1f} min", flush=True)
    return {name: totals[name] / updates[name] for name in models}, sequence_ids


@torch.inference_mode()
def evaluate(models, parquet, groups, columns, read_batch, accumulator_class,
             batch_size, chunk_length):
    metrics = {name: accumulator_class() for name in models}
    seconds = {name: 0.0 for name in models}
    sequence_ids = []
    started = time.perf_counter()
    for batch_index, group_batch in enumerate(batches(groups, batch_size), 1):
        raw_np, target_np, ids = read_batch(parquet, group_batch, columns)
        sequence_ids.extend(ids)
        ema_np = add_ema(raw_np)
        raw, ema_features = torch.from_numpy(raw_np), torch.from_numpy(ema_np)
        hidden = {name: model.initial_hidden(len(group_batch)) for name, model in models.items()}
        predictions = {name: [] for name in models}
        for start in range(0, 20_000, chunk_length):
            stop = min(20_000, start + chunk_length)
            for name, model in models.items():
                source = ema_features if name == "ema_h64" else raw
                tick = time.perf_counter()
                prediction, hidden[name] = model(source[:, start:stop], hidden[name])
                seconds[name] += time.perf_counter() - tick
                predictions[name].append(prediction)
        target = target_np[:, 99:].reshape(-1, 2)
        for name in models:
            prediction = torch.cat(predictions[name], 1).numpy()[:, 99:].reshape(-1, 2)
            metrics[name].add(target, prediction)
        if batch_index % 8 == 0 or batch_index * batch_size >= len(groups):
            done = min(batch_index * batch_size, len(groups))
            print(f"development: {done:,}/{len(groups):,}, {(time.perf_counter()-started)/60:.1f} min", flush=True)
    result = {}
    for name in models:
        result[name] = metrics[name].result()
        result[name]["batched_model_seconds"] = seconds[name]
    return result, sequence_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-name", default="train_only_gru_architecture_screen_v1")
    parser.add_argument("--max-fit-sequences", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_dir = root / "artifacts" / args.artifact_name
    experiment_dir = root / "experiments/models" / args.artifact_name
    if artifact_dir.exists() or experiment_dir.exists():
        raise FileExistsError("refusing to overwrite an existing architecture screen")
    artifact_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    sys.path[:0] = [str(root / "scripts"), str(root / "src")]
    from compact_gru import CompactGRU, count_parameters
    from train_compact_gru import FEATURES, TARGETS, MetricAccumulator, read_batch, weighted_smooth_l1

    split_path = root / "experiments/models/train_only_causal_ablation_v1/manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))["split"]
    fit_all = [int(value) for value in split["fit_source_row_groups"]]
    development_groups = [int(value) for value in split["development_source_row_groups"]]
    confirmation_groups = [int(value) for value in split["confirmation_source_row_groups"]]
    selected = fit_all.copy()
    random.Random(SEED).shuffle(selected)
    selected = sorted(selected[:min(args.max_fit_sequences, len(selected))])
    if set(selected) & set(development_groups) or set(selected) & set(confirmation_groups):
        raise ValueError("overlapping partitions")

    raw_scaler = np.load(root / "artifacts/train_only_causal_ablation_v1/mlp_raw.npz", allow_pickle=False)
    ema_scaler = np.load(root / "artifacts/train_only_causal_ablation_v1/mlp_exponential_average.npz", allow_pickle=False)
    model_specs = {
        "raw_h64": (raw_scaler, 112, 64),
        "ema_h64": (ema_scaler, 336, 64),
        "raw_h128": (raw_scaler, 112, 128),
    }
    models = {}
    for offset, (name, (scaler, input_size, hidden_size)) in enumerate(model_specs.items()):
        torch.manual_seed(SEED + offset)
        models[name] = CompactGRU(torch.from_numpy(scaler["mean"].astype(np.float32)),
                                  torch.from_numpy(scaler["scale"].astype(np.float32)),
                                  "target_specific", input_size=input_size, hidden_size=hidden_size)
        print(f"{name}: {count_parameters(models[name]):,} parameters", flush=True)
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                  for name, model in models.items()}
    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != columns:
        raise ValueError("unexpected training schema")
    print(f"screen fit={len(selected):,}; development={len(development_groups):,}; confirmation={len(confirmation_groups):,} reserved", flush=True)
    started = time.perf_counter()
    for model in models.values():
        model.train()
    losses, fit_ids = train(models, optimizers, parquet, selected, columns, read_batch,
                            weighted_smooth_l1, args.batch_size, args.chunk_length)
    for model in models.values():
        model.eval()
    results, development_ids = evaluate(models, parquet, development_groups, columns, read_batch,
                                        MetricAccumulator, args.batch_size, args.chunk_length)
    artifacts = {}
    for name, model in models.items():
        path = artifact_dir / f"{name}.pt"
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizers[name].state_dict(),
                    "config": {"input_size": model.input_size, "hidden_size": model.hidden_size,
                               "head_type": "target_specific", "feature_type": "ema" if name == "ema_h64" else "raw"},
                    "seed": SEED + list(model_specs).index(name)}, path)
        artifacts[name] = {"path": str(path.relative_to(root)).replace(os.sep, "/"), "sha256": sha256(path),
                           "bytes": path.stat().st_size, "parameters": count_parameters(model)}
    winner = max(results, key=lambda name: results[name]["weighted_pearson"])
    metrics = {"evaluation": "all eligible development rows; confirmation not evaluated",
               "training_losses": losses, "models": results, "winner": winner,
               "artifacts": artifacts, "elapsed_seconds": time.perf_counter() - started}
    manifest = {"version": 1, "policy": "train.parquet only; confirmation unread",
                "seed": SEED, "fit_sequences": len(selected), "fit_source_row_groups": selected,
                "fit_sequence_ids": fit_ids, "development_sequences": len(development_groups),
                "development_sequence_ids": development_ids, "confirmation_sequences": len(confirmation_groups),
                "confirmation_evaluated": False, "ema_half_lives": list(EMA_HALF_LIVES),
                "models": {name: {"input_size": spec[1], "hidden_size": spec[2],
                                   "head_type": "target_specific", "parameters": count_parameters(models[name])}
                           for name, spec in model_specs.items()},
                "training": {"epochs": 1, "learning_rate": 1e-3, "batch_size": args.batch_size,
                             "chunk_length": args.chunk_length, "loss": "weighted SmoothL1(beta=0.25)"}}
    write_json(experiment_dir / "metrics.json", metrics)
    write_json(experiment_dir / "manifest.json", manifest)
    print(json.dumps({"models": results, "winner": winner}, indent=2), flush=True)
    print(f"Finished in {(time.perf_counter()-started)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
