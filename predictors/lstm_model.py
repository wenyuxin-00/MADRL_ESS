"""LSTM 预测网络定义。

定义 LSTM 回归网络的 PyTorch 实现。

主要类:
    LSTMModel -- LSTM 网络模型
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecastModel(nn.Module):
    """用于 price / load / pv 标量序列的统一 LSTM 主干。"""

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
        """输入形状支持 `(B, T)` 和 `(B, T, 1)`。"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() != 3:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D.")

        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMPricePredictor(LSTMForecastModel):
    """兼容旧名。

    历史上该类只服务于 `price`，现在保留旧名字只是为了兼容测试、旧脚本和旧 artifact。
    新代码应优先使用 `LSTMForecastModel`。
    """

