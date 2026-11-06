"""Train-only small MLP submission attempt."""
from pathlib import Path
import numpy as np


class PredictionModel:
    def __init__(self):
        artifact = np.load(Path(__file__).resolve().parent / "model.npz")
        self.mean = artifact["mean"].astype(np.float32)
        self.scale = artifact["scale"].astype(np.float32)
        self.layers = []
        for target in range(2):
            layers = []
            layer = 0
            while f"t{target}_w{layer}" in artifact:
                layers.append((
                    artifact[f"t{target}_w{layer}"].astype(np.float32),
                    artifact[f"t{target}_b{layer}"].astype(np.float32),
                ))
                layer += 1
            self.layers.append(layers)
        self.current_seq_ix = None
        self.previous_step = None

    @staticmethod
    def _forward(value, layers):
        for index, (weight, bias) in enumerate(layers):
            value = value @ weight + bias
            if index + 1 < len(layers):
                value = np.maximum(value, 0)
        return np.float32(value[0])

    def predict(self, data_point):
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
        normalized = (state - self.mean) / self.scale
        prediction = np.array([
            self._forward(normalized, self.layers[0]),
            self._forward(normalized, self.layers[1]),
        ], dtype=np.float32)
        if not data_point.need_prediction:
            return None
        return prediction
