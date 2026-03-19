"""Actor 输出头模块。

将编码器输出的特征向量映射为连续动作空间中的确定性动作。
使用 tanh 激活函数将输出限制在 [-max_action, max_action] 范围内。

主要类：
    - DeterministicContinuousActorHead: 确定性连续动作输出头
"""

from __future__ import annotations

import torch
from torch import nn

from models.utils import orthogonal_init


class DeterministicContinuousActorHead(nn.Module):
    """确定性连续动作输出头。

    将编码器输出的特征向量通过全连接层映射到动作空间，
    再使用 tanh 激活函数将输出限制在 [-max_action, max_action] 范围内。

    适用于 MADDPG / MATD3 等确定性策略梯度算法。

    属性:
        max_action: 动作的最大绝对值（用于缩放 tanh 输出）
        fc: 全连接层 (hidden_dim -> action_dim)
    """

    def __init__(self, hidden_dim: int, action_dim: int, max_action: float, use_orthogonal_init: bool):
        """初始化确定性连续动作输出头。

        参数:
            hidden_dim: 编码器输出的特征维度
            action_dim: 动作空间维度
            max_action: 动作的最大绝对值
            use_orthogonal_init: 是否使用正交初始化
        """
        super().__init__()
        self.max_action = float(max_action)
        self.fc = nn.Linear(int(hidden_dim), int(action_dim))
        if use_orthogonal_init:
            orthogonal_init(self.fc)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """前向传播：特征 -> 线性变换 -> tanh 缩放 -> 动作。

        参数:
            embedding: 编码器输出的特征向量，形状为 (batch, hidden_dim)

        返回:
            确定性动作，形状为 (batch, action_dim)，值域 [-max_action, max_action]
        """
        # tanh 将输出限制在 [-1, 1]，再乘以 max_action 缩放到目标范围
        return self.max_action * torch.tanh(self.fc(embedding))
