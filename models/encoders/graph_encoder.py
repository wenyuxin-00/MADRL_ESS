"""GNN 图编码器模块。

基于邻接矩阵的消息传递图神经网络，对智能体之间的拓扑关系进行建模。
每一层通过邻接矩阵聚合邻居节点的特征，并与自身特征相加后经过线性变换。

主要类：
    - GraphEncoder: 基于邻接矩阵的消息传递编码器
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.utils import orthogonal_init


class GraphEncoder(nn.Module):
    """基于邻接矩阵的消息传递图编码器。

    网络结构: node_features -> 投影 -> [消息传递层 x N] -> 读出
    每层消息传递: 聚合邻居特征（度归一化）-> 与自身相加 -> 线性变换 + ReLU

    读出策略:
        - 如果指定 target_index（Actor 路径）：取目标节点的特征
        - 否则（Critic 路径）：对所有节点均值池化

    属性:
        input_proj: 输入投影层 (input_dim -> hidden_dim)
        layers: 消息传递层列表，每层为一个线性变换
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, use_orthogonal_init: bool):
        """初始化图编码器。

        参数:
            input_dim: 每个节点的输入特征维度
            hidden_dim: 隐藏层维度，同时也是输出维度
            num_layers: 消息传递的层数（决定信息传播的跳数）
            use_orthogonal_init: 是否使用正交初始化
        """
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        # 每一层消息传递使用独立的线性变换
        self.layers = nn.ModuleList(
            [nn.Linear(int(hidden_dim), int(hidden_dim)) for _ in range(int(num_layers))]
        )
        if use_orthogonal_init:
            orthogonal_init(self.input_proj)
            for layer in self.layers:
                orthogonal_init(layer)

    def forward(self, payload: dict):
        """前向传播：投影 -> 多轮消息传递 -> 读出。

        参数:
            payload: 适配器输出的字典，包含：
                - "node_features": 节点特征 (batch, n_nodes, input_dim)
                - "adjacency": 邻接矩阵 (batch, n_nodes, n_nodes)
                - "target_index" (可选): 目标节点索引

        返回:
            若有 target_index: 目标节点特征 (batch, hidden_dim)
            否则: 所有节点均值池化后的特征 (batch, hidden_dim)
        """
        x = self.input_proj(payload["node_features"])  # 将节点特征投影到隐藏空间
        adjacency = payload["adjacency"]
        # 计算节点度数，用于归一化聚合（clamp 防止孤立节点除零）
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)

        for layer in self.layers:
            # 消息传递：邻接矩阵乘以节点特征 -> 度归一化聚合
            aggregated = torch.matmul(adjacency, x) / degree
            # 残差连接：自身特征 + 邻居聚合特征 -> 线性变换 + ReLU
            x = F.relu(layer(x + aggregated))

        # 读出：Actor 取目标节点，Critic 取均值池化
        if "target_index" in payload:
            return x[:, int(payload["target_index"]), :]
        return x.mean(dim=1)
