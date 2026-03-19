"""Critic 输出头模块。

将编码器输出的特征向量映射为 Q 值估计。提供两种变体：
    - SingleQHead: 单 Q 网络，用于 MADDPG 算法
    - TwinQHead: 双 Q 网络，用于 MATD3 算法（缓解 Q 值过估计）
"""

from __future__ import annotations

from torch import nn

from models.utils import orthogonal_init


class SingleQHead(nn.Module):
    """单 Q 值 Critic 输出头，用于 MADDPG 算法。

    将编码器输出的特征向量通过单个全连接层映射为标量 Q 值。

    属性:
        fc: 全连接层 (hidden_dim -> 1)
    """

    def __init__(self, hidden_dim: int, use_orthogonal_init: bool):
        """初始化单 Q 值输出头。

        参数:
            hidden_dim: 编码器输出的特征维度
            use_orthogonal_init: 是否使用正交初始化
        """
        super().__init__()
        self.fc = nn.Linear(int(hidden_dim), 1)
        if use_orthogonal_init:
            orthogonal_init(self.fc)

    def forward(self, embedding):
        """前向传播：计算 Q 值。

        参数:
            embedding: 编码器输出的特征向量，形状为 (batch, hidden_dim)

        返回:
            Q 值标量，形状为 (batch, 1)
        """
        return self.fc(embedding)


class TwinQHead(nn.Module):
    """双 Q 值 Critic 输出头，用于 MATD3 算法。

    使用两个独立的 Q 网络，取较小值作为目标 Q 值，
    以缓解 Q 值过高估计问题（Clipped Double Q-learning）。

    属性:
        q1: 第一个 Q 网络 (hidden_dim -> 1)
        q2: 第二个 Q 网络 (hidden_dim -> 1)
    """

    def __init__(self, hidden_dim: int, use_orthogonal_init: bool):
        """初始化双 Q 值输出头。

        参数:
            hidden_dim: 编码器输出的特征维度
            use_orthogonal_init: 是否使用正交初始化
        """
        super().__init__()
        self.q1 = nn.Linear(int(hidden_dim), 1)
        self.q2 = nn.Linear(int(hidden_dim), 1)
        if use_orthogonal_init:
            orthogonal_init(self.q1)
            orthogonal_init(self.q2)

    def forward(self, embedding):
        """前向传播：同时计算两个 Q 值。

        参数:
            embedding: 编码器输出的特征向量，形状为 (batch, hidden_dim)

        返回:
            (Q1, Q2) 元组，每个形状为 (batch, 1)
        """
        return self.q1(embedding), self.q2(embedding)

    def Q1(self, embedding):
        """仅计算第一个 Q 值（用于策略更新时的梯度计算）。

        参数:
            embedding: 编码器输出的特征向量，形状为 (batch, hidden_dim)

        返回:
            第一个 Q 网络的输出，形状为 (batch, 1)

        注意:
            MATD3 在更新策略时只需要一个 Q 值的梯度，
            因此提供此方法避免不必要的 Q2 计算。
        """
        return self.q1(embedding)
