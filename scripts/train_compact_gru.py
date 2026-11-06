"""Train compact stateful GRUs using the fixed train-only split.

Only datasets/train.parquet is opened. Fit sequences train the models,
development sequences evaluate them, and confirmation sequences remain unread.
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
from torch.nn import functional as F

sys.dont_write_bytecode = True
SEED = 20261102
FEATURES = [
    f"i{i}_{group}{j}"
    for i in range(2)
    for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for j in range(count)
] + [f"a{i}" for i in range(8)]
TARGETS = ("t0", "t1")
HEAD_TYPES = ("shared", "target_specific")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_table(table, columns: list[str]) -> int:
    if table.num_rows != 20_000 or table.column_names != columns:
        raise ValueError("unexpected training row-group shape or columns")
    ids = table["seq_ix"].to_numpy()
    steps = table["step_in_seq"].to_numpy()
    need = table["need_prediction"].to_numpy().astype(bool)
    if not np.all(ids == ids[0]) or not np.array_equal(steps, np.arange(20_000)):
        raise ValueError("invalid sequence identity or step order")
    if not np.array_equal(need, steps >= 99):
        raise ValueError("unexpected warm-up flags")
    return int(ids[0])


def read_batch(parquet: pq.ParquetFile, groups: list[int], columns: list[str]):
    x = np.empty((len(groups), 20_000, 112), np.float32)
    y = np.empty((len(groups), 20_000, 2), np.float32)
    sequence_ids = []
    for index, group in enumerate(groups):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        sequence_ids.append(validate_table(table, columns))
        x[index] = np.column_stack([table[column].to_numpy() for column in FEATURES])
        y[index] = np.column_stack([table[column].to_numpy() for column in TARGETS])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite sequence values")
    return x, np.clip(y, -2, 2), sequence_ids


def weighted_smooth_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = target.abs()
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.25)
    return (weight * loss).sum() / weight.sum().clamp_min(1e-8)


def weighted_pearson(weight, sum_y, sum_p, sum_yy, sum_pp, sum_yp) -> float:
    mean_y, mean_p = sum_y / weight, sum_p / weight
    var_y = max(0.0, sum_yy / weight - mean_y * mean_y)
    var_p = max(0.0, sum_pp / weight - mean_p * mean_p)
    covariance = sum_yp / weight - mean_y * mean_p
    if var_y <= 1e-16 or var_p <= 1e-16:
        return 0.0
    return float(np.clip(covariance / np.sqrt(var_y * var_p), -1, 1))


class MetricAccumulator:
    def __init__(self) -> None:
        self.weight = np.zeros(2)
        self.sum_y = np.zeros(2)
        self.sum_p = np.zeros(2)
        self.sum_yy = np.zeros(2)
        self.sum_pp = np.zeros(2)
        self.sum_yp = np.zeros(2)
        self.rows = 0

    def add(self, target: np.ndarray, prediction: np.ndarray) -> None:
        y = np.clip(target, -2, 2).astype(np.float64)
        p = np.clip(prediction, -2, 2).astype(np.float64)
        w = np.abs(y)
        self.weight += w.sum(0)
        self.sum_y += (w * y).sum(0)
        self.sum_p += (w * p).sum(0)
        self.sum_yy += (w * y * y).sum(0)
        self.sum_pp += (w * p * p).sum(0)
        self.sum_yp += (w * y * p).sum(0)
        self.rows += len(y)

    def result(self) -> dict:
        scores = [weighted_pearson(self.weight[k], self.sum_y[k], self.sum_p[k],
                                   self.sum_yy[k], self.sum_pp[k], self.sum_yp[k]) for k in range(2)]
        return {"t0": scores[0], "t1": scores[1], "weighted_pearson": float(np.mean(scores)), "rows": self.rows}


def batches(items: list[int], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def train_epoch(models, optimizers, parquet, groups, columns, batch_size, chunk_length, epoch):
    order = groups.copy()
    random.Random(SEED + epoch).shuffle(order)
    totals = {name: 0.0 for name in models}
    updates = {name: 0 for name in models}
    sequence_ids = []
    started = time.perf_counter()
    for batch_index, group_batch in enumerate(batches(order, batch_size), 1):
        x_np, y_np, ids = read_batch(parquet, group_batch, columns)
        sequence_ids.extend(ids)
        x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)
        hidden = {name: model.initial_hidden(len(group_batch)) for name, model in models.items()}
        for start in range(0, 20_000, chunk_length):
            stop = min(20_000, start + chunk_length)
            loss_start = max(start, 99)
            local_start = loss_start - start
            for name, model in models.items():
                optimizers[name].zero_grad(set_to_none=True)
                prediction, next_hidden = model(x[:, start:stop], hidden[name])
                hidden[name] = next_hidden.detach()
                if loss_start >= stop:
                    continue
                loss = weighted_smooth_l1(prediction[:, local_start:], y[:, loss_start:stop])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name] += float(loss.detach())
                updates[name] += 1
        if batch_index % 8 == 0 or batch_index * batch_size >= len(order):
            done = min(batch_index * batch_size, len(order))
            text = ", ".join(f"{name}={totals[name] / max(updates[name], 1):.5f}" for name in models)
            print(f"epoch {epoch + 1}: {done:,}/{len(order):,}, loss {text}, {(time.perf_counter()-started)/60:.1f} min", flush=True)
    return {name: totals[name] / max(updates[name], 1) for name in models}, sequence_ids


@torch.inference_mode()
def evaluate(models, parquet, groups, columns, batch_size, chunk_length):
    metrics = {name: MetricAccumulator() for name in models}
    seconds = {name: 0.0 for name in models}
    sequence_ids = []
    started = time.perf_counter()
    for batch_index, group_batch in enumerate(batches(groups, batch_size), 1):
        x_np, y_np, ids = read_batch(parquet, group_batch, columns)
        sequence_ids.extend(ids)
        x = torch.from_numpy(x_np)
        hidden = {name: model.initial_hidden(len(group_batch)) for name, model in models.items()}
        predictions = {name: [] for name in models}
        for start in range(0, 20_000, chunk_length):
            stop = min(20_000, start + chunk_length)
            for name, model in models.items():
                tick = time.perf_counter()
                prediction, hidden[name] = model(x[:, start:stop], hidden[name])
                seconds[name] += time.perf_counter() - tick
                predictions[name].append(prediction)
        target = y_np[:, 99:].reshape(-1, 2)
        for name in models:
            prediction = torch.cat(predictions[name], 1).numpy()[:, 99:].reshape(-1, 2)
            metrics[name].add(target, prediction)
        if batch_index % 8 == 0 or batch_index * batch_size >= len(groups):
            done = min(batch_index * batch_size, len(groups))
            print(f"development: {done:,}/{len(groups):,}, {(time.perf_counter()-started)/60:.1f} min", flush=True)
    results = {}
    for name in models:
        results[name] = metrics[name].result()
        results[name]["batched_model_seconds"] = seconds[name]
    return results, sequence_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-name", default="train_only_compact_gru_v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-fit-sequences", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    split_path = root / "experiments/models/train_only_causal_ablation_v1/manifest.json"
    scaler_path = root / "artifacts/train_only_causal_ablation_v1/mlp_raw.npz"
    artifact_dir = root / "artifacts" / args.artifact_name
    experiment_dir = root / "experiments/models" / args.artifact_name
    if artifact_dir.exists() or experiment_dir.exists():
        raise FileExistsError("refusing to overwrite an existing attempt")
    artifact_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    split = json.loads(split_path.read_text(encoding="utf-8"))["split"]
    fit_all = [int(x) for x in split["fit_source_row_groups"]]
    development = [int(x) for x in split["development_source_row_groups"]]
    confirmation = [int(x) for x in split["confirmation_source_row_groups"]]
    if set(fit_all) & set(development) or set(fit_all) & set(confirmation) or set(development) & set(confirmation):
        raise ValueError("overlapping partitions")
    selected = fit_all.copy()
    random.Random(SEED).shuffle(selected)
    selected = sorted(selected[:min(args.max_fit_sequences, len(selected))])

    scaler = np.load(scaler_path, allow_pickle=False)
    mean = torch.from_numpy(scaler["mean"].astype(np.float32))
    scale = torch.from_numpy(scaler["scale"].astype(np.float32))
    sys.path.insert(0, str(root / "src"))
    from compact_gru import CompactGRU, count_parameters
    models = {name: CompactGRU(mean, scale, name) for name in HEAD_TYPES}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
                  for name, model in models.items()}
    for name, model in models.items():
        print(f"{name}: {count_parameters(model):,} trainable parameters", flush=True)

    parquet = pq.ParquetFile(root / "datasets/train.parquet")
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURES, *TARGETS]
    if parquet.schema_arrow.names != columns:
        raise ValueError("unexpected training schema")
    print(f"fit={len(selected):,}; development={len(development):,}; confirmation={len(confirmation):,} reserved", flush=True)
    started = time.perf_counter()
    epoch_losses, fitted_ids = [], []
    for epoch in range(args.epochs):
        for model in models.values():
            model.train()
        losses, ids = train_epoch(models, optimizers, parquet, selected, columns,
                                  args.batch_size, args.chunk_length, epoch)
        epoch_losses.append(losses)
        fitted_ids.append(ids)
    for model in models.values():
        model.eval()
    results, development_ids = evaluate(models, parquet, development, columns,
                                        args.batch_size, args.chunk_length)

    artifacts = {}
    for name, model in models.items():
        path = artifact_dir / f"compact_gru_{name}.pt"
        torch.save({"state_dict": model.state_dict(), "config": {"input_size": 112, "hidden_size": 64, "head_type": name}, "seed": SEED}, path)
        artifacts[name] = {"path": str(path.relative_to(root)).replace(os.sep, "/"), "sha256": file_sha256(path),
                           "bytes": path.stat().st_size, "trainable_parameters": count_parameters(model)}
    elapsed = time.perf_counter() - started
    write_json(experiment_dir / "metrics.json", {"evaluation": "all eligible development rows; confirmation not evaluated",
               "training_losses": epoch_losses, "models": results, "artifacts": artifacts, "elapsed_seconds": elapsed})
    write_json(experiment_dir / "manifest.json", {
        "version": 1, "policy": "datasets/train.parquet only; confirmation unread",
        "source_split_manifest": str(split_path.relative_to(root)).replace(os.sep, "/"),
        "source_scaler": {"path": str(scaler_path.relative_to(root)).replace(os.sep, "/"), "sha256": file_sha256(scaler_path)},
        "seed": SEED, "fit": {"available_sequences": len(fit_all), "selected_sequences": len(selected),
        "source_row_groups": selected, "sequence_ids_by_epoch": fitted_ids},
        "development": {"sequences": len(development), "sequence_ids": development_ids},
        "confirmation": {"sequences": len(confirmation), "evaluated": False},
        "architecture": {"input_projection": [112, 64], "gru_layers": 1, "gru_hidden_size": 64,
        "heads": {"shared": "64 -> 32 -> SiLU -> 2", "target_specific": "two 64 -> 16 -> SiLU -> 1"}},
        "training": {"epochs": args.epochs, "batch_size_sequences": args.batch_size,
        "chunk_length": args.chunk_length, "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
        "loss": "abs(clipped target)-weighted SmoothL1(beta=0.25)",
        "warmup_steps_used_for_state_not_loss": 99, "threads": args.threads, "torch_version": torch.__version__}})
    print(json.dumps(results, indent=2), flush=True)
    print(f"Finished in {elapsed / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
