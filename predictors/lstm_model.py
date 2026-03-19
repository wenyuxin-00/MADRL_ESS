"""LSTM 预测网络定义。

定义 LSTM 回归网络的 PyTorch 实现。

主要类:
    LSTMModel -- LSTM 网络模型
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecastModel(nn.Module):
    """用于 price / load / pv 标量序列的统一 LSTM 回归主干网络。

    该模型采用多层 LSTM 编码输入时间序列，再通过全连接预测头输出多步预测值。
    输入为单变量序列（每个特征维度为 1），适用于电价、负荷、光伏等标量信号。

    属性:
        lstm: 多层 LSTM 编码器，input_size=1，batch_first=True。
        head: 全连接预测头，将 LSTM 最后时刻的隐状态映射为 pred_len 步预测。
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.23,
        pred_len: int = 4,
    ):
        """初始化 LSTM 预测模型。

        参数:
            hidden_size: LSTM 隐藏层维度，默认 128。
            num_layers: LSTM 层数，默认 2。
            dropout: Dropout 比率，默认 0.23。单层 LSTM 时自动禁用层间 dropout。
            pred_len: 预测步数（输出维度），默认 4。
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # 单层时无层间 dropout
        )
        # 预测头：隐状态 -> 128 -> ReLU -> Dropout -> pred_len
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输入形状支持 ``(B, T)`` 和 ``(B, T, 1)``。

        参数:
            x: 输入张量，形状为 ``(batch, seq_len)`` 或 ``(batch, seq_len, 1)``。

        返回:
            预测输出，形状为 ``(batch, pred_len)``。
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)
        elif x.dim() != 3:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D.")

        out, _ = self.lstm(x)  # out: (B, T, hidden_size)
        return self.head(out[:, -1, :])  # 取最后时刻的隐状态进行预测


class LSTMPricePredictor(LSTMForecastModel):
    """兼容旧名称的 LSTM 预测模型别名。

    历史上该类仅服务于 ``price`` 信号预测，现在保留旧名字是为了兼容
    已有的测试、旧脚本和旧 artifact 加载。新代码应优先使用 ``LSTMForecastModel``。
    """

