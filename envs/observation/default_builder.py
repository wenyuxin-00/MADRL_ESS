"""默认观测构造器。

默认输出包含：
  - `local`：当前时刻的局部特征拼接
  - `xxx_seq`：按 feature 注册表选择的序列特征
  - `adjacency`：图模型需要的邻接矩阵
"""

from __future__ import annotations

import numpy as np

from envs.observation.base import ObservationBuilder
from envs.observation.features import (
    build_adjacency_field,
    resolve_local_features,
    resolve_sequence_features,
)


class DefaultObservationBuilder(ObservationBuilder):
    """默认观测构造器，根据配置选择的 feature 列表构造结构化观测。

    输出字段:
        - local: (n_agents, local_dim) 各局部特征拼接而成
        - adjacency: (n_agents, n_agents) 邻接矩阵
        - xxx_seq: 各序列特征，shape 由 feature spec 决定

    属性:
        local_feature_names: 局部特征名称列表
        sequence_feature_names: 序列特征名称列表
        future_horizon: 未来预测窗口长度
        adjacency_type: 邻接矩阵类型（"identity" 或 "fully_connected_no_self"）
        sequence_length: 序列总长度 = future_horizon + 1（含当前步）
        local_specs: 解析后的局部特征规格列表
        sequence_specs: 解析后的序列特征规格列表
        local_dim: 局部特征拼接后的总维度
    """

    def __init__(
        self,
        local_features: list[str],
        sequence_features: list[str],
        future_horizon: int,
        adjacency_type: str = "identity",
    ):
        """初始化观测构造器，解析特征规格。

        参数:
            local_features: 局部特征名称列表，如 ["time", "price", "soc"]
            sequence_features: 序列特征名称列表，如 ["price", "load"]
            future_horizon: 未来预测窗口长度
            adjacency_type: 邻接矩阵类型
        """
        self.local_feature_names = list(local_features)
        self.sequence_feature_names = list(sequence_features)
        self.future_horizon = int(future_horizon)
        self.adjacency_type = adjacency_type
        # 序列总长度包含当前步 + 未来 horizon 步
        self.sequence_length = self.future_horizon + 1
        # 从全局注册表解析特征规格
        self.local_specs = resolve_local_features(self.local_feature_names)
        self.sequence_specs = resolve_sequence_features(self.sequence_feature_names)
        # 计算所有局部特征拼接后的总维度
        self.local_dim = sum(spec.dim for spec in self.local_specs)

    def get_schema(self, n_agents: int) -> dict[str, tuple[int, ...]]:
        """返回各观测字段的 shape 定义。

        参数:
            n_agents: 智能体数量

        返回:
            dict: 字段名 → shape 元组
        """
        schema = {
            "local": (int(n_agents), self.local_dim),
            "adjacency": (int(n_agents), int(n_agents)),
        }
        for spec in self.sequence_specs:
            schema[spec.field_name()] = spec.schema(int(n_agents), self.sequence_length)
        return schema

    def get_layout(self, n_agents: int) -> dict[str, dict]:
        """返回各观测字段的语义布局信息。

        参数:
            n_agents: 智能体数量

        返回:
            dict: 字段名 → 布局描述字典
        """
        layout = {
            "local": {
                "group": "local",
                "scope": "per_agent",
                "dim": int(self.local_dim),
                "fields": list(self.local_feature_names),
            },
            "adjacency": {
                "group": "graph",
                "scope": "shared",
                "dim": int(n_agents),
                "fields": ["adjacency"],
            },
        }
        # 追加各序列特征的布局信息
        for spec in self.sequence_specs:
            field_name = spec.field_name()
            layout[field_name] = {
                **spec.layout(),
                "shape": self.get_schema(n_agents)[field_name],
            }
        return layout

    def build(self, env) -> dict[str, np.ndarray]:
        """从环境当前状态构造一步结构化观测。

        参数:
            env: 环境实例，需提供 n、cur_step、episode_length、
                 get_signal_step/get_signal 等接口

        返回:
            dict[str, np.ndarray]: 包含 local、adjacency 及各序列字段
        """
        # 构造各局部特征并在 axis=1 上拼接
        local_parts = [spec.builder(env, self.sequence_length) for spec in self.local_specs]
        if local_parts:
            local = np.concatenate(local_parts, axis=1).astype(np.float32)
        else:
            # 无局部特征时返回零维占位符
            local = np.zeros((env.n, 0), dtype=np.float32)

        obs = {
            "local": local,
            "adjacency": build_adjacency_field(env.n, self.adjacency_type),
        }
        # 构造各序列特征
        for spec in self.sequence_specs:
            obs[spec.field_name()] = spec.builder(env, self.sequence_length)
        return obs
