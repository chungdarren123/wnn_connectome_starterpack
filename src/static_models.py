"""Portable NumPy inference for train-only static baselines.

These models use only the current 112-feature row. They still enforce the
competition sequence contract and return None during warm-up.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


class _StaticPredictionModel:
    def __init__(self) -> None:
        self.current_seq_ix = None
        self.previous_step = None

    def _validate(self, data_point) -> np.ndarray:
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step
        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")
        return state


class RidgePredictionModel(_StaticPredictionModel):
    def __init__(self, model_path: str | Path) -> None:
        super().__init__()
        artifact = np.load(model_path)
        self.mean = artifact["mean"].astype(np.float32)
        self.scale = artifact["scale"].astype(np.float32)
        self.weight = artifact["weight"].astype(np.float32)
        self.bias = artifact["bias"].astype(np.float32)
        if self.mean.shape != (112,) or self.scale.shape != (112,):
            raise ValueError("invalid ridge normalization arrays")
        if self.weight.shape != (2, 112) or self.bias.shape != (2,):
            raise ValueError("invalid ridge model arrays")

    def batch_predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        return ((features - self.mean) / self.scale) @ self.weight.T + self.bias

    def predict(self, data_point):
        state = self._validate(data_point)
        prediction = self.batch_predict(state[None, :])[0]
        if not data_point.need_prediction:
            return None
        return prediction.astype(np.float32)


class MLPPredictionModel(_StaticPredictionModel):
    def __init__(self, model_path: str | Path) -> None:
        super().__init__()
        artifact = np.load(model_path)
        self.mean = artifact["mean"].astype(np.float32)
        self.scale = artifact["scale"].astype(np.float32)
        self.layers = []
        for target in range(2):
            target_layers = []
            layer = 0
            while f"t{target}_w{layer}" in artifact:
                target_layers.append((
                    artifact[f"t{target}_w{layer}"].astype(np.float32),
                    artifact[f"t{target}_b{layer}"].astype(np.float32),
                ))
                layer += 1
            if not target_layers:
                raise ValueError(f"no layers found for target {target}")
            self.layers.append(target_layers)

    @staticmethod
    def _forward(features: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        value = features
        for index, (weight, bias) in enumerate(layers):
            value = value @ weight + bias
            if index + 1 < len(layers):
                value = np.maximum(value, 0)
        return value[:, 0]

    def batch_predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        normalized = (features - self.mean) / self.scale
        return np.column_stack([
            self._forward(normalized, self.layers[0]),
            self._forward(normalized, self.layers[1]),
        ]).astype(np.float32)

    def predict(self, data_point):
        state = self._validate(data_point)
        prediction = self.batch_predict(state[None, :])[0]
        if not data_point.need_prediction:
            return None
        return prediction
