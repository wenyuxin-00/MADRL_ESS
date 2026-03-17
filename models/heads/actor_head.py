"""Actor heads."""

from __future__ import annotations

import torch
from torch import nn

from models.utils import orthogonal_init


class DeterministicContinuousActorHead(nn.Module):
    """Deterministic continuous actor head."""

    def __init__(self, hidden_dim: int, action_dim: int, max_action: float, use_orthogonal_init: bool):
        super().__init__()
        self.max_action = float(max_action)
        self.fc = nn.Linear(int(hidden_dim), int(action_dim))
        if use_orthogonal_init:
            orthogonal_init(self.fc)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.fc(embedding))
