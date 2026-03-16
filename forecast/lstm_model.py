"""Shared LSTM price prediction model.

Keep the network definition in one public module so notebook training,
runtime forecasters, and tests all depend on the same source of truth.
"""

import torch
import torch.nn as nn


class LSTMPricePredictor(nn.Module):
    """LSTM direct multi-step price predictor."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.23,
        pred_len: int = 4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D.")

        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
