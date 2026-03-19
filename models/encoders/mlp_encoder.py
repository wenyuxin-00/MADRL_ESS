"""MLP 编码器模块。

提供两层全连接网络作为特征提取骨干，适用于扁平向量输入。
这是最简单、最快速的编码器，适合作为默认选择。

主要类：
    - MLPEncoder: 两层 MLP 编码器，使用 ReLU 激活
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from models.utils import orthogonal_init


class MLPEncoder(nn.Module):
    """两层 MLP 编码器。

    网络结构: input -> fc1 -> ReLU -> fc2 -> ReLU -> output
    输入为扁平的一维特征向量，输出为 hidden_dim 维的特征表示。

    属性:
        fc1: 第一层全连接层 (input_dim -> hidden_dim)
        fc2: 第二层全连接层 (hidden_dim -> hidden_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int, use_orthogonal_init: bool):
        """初始化 MLP 编码器。

        参数:
            input_dim: 输入特征维度（由适配器的 output_dim 决定）
            hidden_dim: 隐藏层维度，同时也是输出维度
            use_orthogonal_init: 是否使用正交初始化
        """
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        if use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, x):
        """前向传播：两层全连接 + ReLU 激活。

        参数:
            x: 输入特征张量，形状为 (batch, input_dim)

        返回:
            编码后的特征张量，形状为 (batch, hidden_dim)
        """
        x = F.relu(self.fc1(x))   # 第一层: 线性变换 + ReLU
        return F.relu(self.fc2(x)) # 第二层: 线性变换 + ReLU
