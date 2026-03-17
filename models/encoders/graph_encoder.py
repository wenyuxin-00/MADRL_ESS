"""Graph encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.utils import orthogonal_init


class GraphEncoder(nn.Module):
    """Simple adjacency-based message-passing encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, use_orthogonal_init: bool):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.layers = nn.ModuleList(
            [nn.Linear(int(hidden_dim), int(hidden_dim)) for _ in range(int(num_layers))]
        )
        if use_orthogonal_init:
            orthogonal_init(self.input_proj)
            for layer in self.layers:
                orthogonal_init(layer)

    def forward(self, payload: dict):
        x = self.input_proj(payload["node_features"])
        adjacency = payload["adjacency"]
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)

        for layer in self.layers:
            aggregated = torch.matmul(adjacency, x) / degree
            x = F.relu(layer(x + aggregated))

        if "target_index" in payload:
            return x[:, int(payload["target_index"]), :]
        return x.mean(dim=1)
