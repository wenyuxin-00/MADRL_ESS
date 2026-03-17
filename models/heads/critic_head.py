"""Critic heads."""

from __future__ import annotations

from torch import nn

from models.utils import orthogonal_init


class SingleQHead(nn.Module):
    """Single-Q critic head."""

    def __init__(self, hidden_dim: int, use_orthogonal_init: bool):
        super().__init__()
        self.fc = nn.Linear(int(hidden_dim), 1)
        if use_orthogonal_init:
            orthogonal_init(self.fc)

    def forward(self, embedding):
        return self.fc(embedding)


class TwinQHead(nn.Module):
    """Twin-Q critic head for MATD3."""

    def __init__(self, hidden_dim: int, use_orthogonal_init: bool):
        super().__init__()
        self.q1 = nn.Linear(int(hidden_dim), 1)
        self.q2 = nn.Linear(int(hidden_dim), 1)
        if use_orthogonal_init:
            orthogonal_init(self.q1)
            orthogonal_init(self.q2)

    def forward(self, embedding):
        return self.q1(embedding), self.q2(embedding)

    def Q1(self, embedding):
        return self.q1(embedding)
