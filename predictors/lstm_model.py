from __future__ import annotations

import torch
from torch import nn


class LSTMForecastModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float, pred_len: int, input_size: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=int(input_size), hidden_size=int(hidden_size), num_layers=int(num_layers), dropout=float(dropout) if int(num_layers) > 1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(int(hidden_size), 128), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(128, int(pred_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
