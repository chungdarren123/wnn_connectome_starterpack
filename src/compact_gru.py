"""Compact causal GRU models shared by training and ONNX export."""
from __future__ import annotations

import torch
from torch import nn


class SharedNonlinearHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.SiLU(), nn.Linear(32, 2)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class TargetSpecificHeads(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.t0 = nn.Sequential(nn.Linear(hidden_size, 16), nn.SiLU(), nn.Linear(16, 1))
        self.t1 = nn.Sequential(nn.Linear(hidden_size, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.t0(state), self.t1(state)), dim=-1)


class CompactGRU(nn.Module):
    """One-layer causal GRU with preprocessing embedded in the model."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, head_type: str,
                 input_size: int = 112, hidden_size: int = 64) -> None:
        super().__init__()
        if mean.shape != (input_size,) or scale.shape != (input_size,):
            raise ValueError("mean and scale must match the raw feature count")
        if head_type not in {"shared", "target_specific"}:
            raise ValueError(f"unknown head type: {head_type}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.head_type = head_type
        self.register_buffer("feature_mean", mean.detach().float().clone())
        self.register_buffer("feature_scale", scale.detach().float().clone())
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_activation = nn.SiLU()
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.head = SharedNonlinearHead(hidden_size) if head_type == "shared" else TargetSpecificHeads(hidden_size)

    def initial_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        device = self.feature_mean.device if device is None else device
        return torch.zeros((1, batch_size, self.hidden_size), dtype=torch.float32, device=device)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_scale
        projected = self.input_activation(self.input_projection(normalized))
        state, next_hidden = self.gru(projected, hidden)
        return self.head(state), next_hidden


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
