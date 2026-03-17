"""Transformer encoder."""

from __future__ import annotations

from torch import nn

from models.utils import orthogonal_init


class TransformerEncoder(nn.Module):
    """Mean-pooled Transformer encoder over token sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, use_orthogonal_init: bool):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        if use_orthogonal_init:
            orthogonal_init(self.input_proj)

    def forward(self, payload: dict):
        x = self.input_proj(payload["tokens"])
        x = self.encoder(x)
        return x.mean(dim=1)
