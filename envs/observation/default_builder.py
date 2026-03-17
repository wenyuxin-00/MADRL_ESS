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
    """根据配置选择的 feature 列表构造观测。"""

    def __init__(
        self,
        local_features: list[str],
        sequence_features: list[str],
        future_horizon: int,
        adjacency_type: str = "identity",
    ):
        self.local_feature_names = list(local_features)
        self.sequence_feature_names = list(sequence_features)
        self.future_horizon = int(future_horizon)
        self.adjacency_type = adjacency_type
        self.sequence_length = self.future_horizon + 1
        self.local_specs = resolve_local_features(self.local_feature_names)
        self.sequence_specs = resolve_sequence_features(self.sequence_feature_names)
        self.local_dim = sum(spec.dim for spec in self.local_specs)

    def get_schema(self, n_agents: int) -> dict[str, tuple[int, ...]]:
        schema = {
            "local": (int(n_agents), self.local_dim),
            "adjacency": (int(n_agents), int(n_agents)),
        }
        for spec in self.sequence_specs:
            schema[spec.field_name()] = spec.schema(int(n_agents), self.sequence_length)
        return schema

    def get_layout(self, n_agents: int) -> dict[str, dict]:
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
        for spec in self.sequence_specs:
            field_name = spec.field_name()
            layout[field_name] = {
                **spec.layout(),
                "shape": self.get_schema(n_agents)[field_name],
            }
        return layout

    def build(self, env) -> dict[str, np.ndarray]:
        local_parts = [spec.builder(env, self.sequence_length) for spec in self.local_specs]
        if local_parts:
            local = np.concatenate(local_parts, axis=1).astype(np.float32)
        else:
            local = np.zeros((env.n, 0), dtype=np.float32)

        obs = {
            "local": local,
            "adjacency": build_adjacency_field(env.n, self.adjacency_type),
        }
        for spec in self.sequence_specs:
            obs[spec.field_name()] = spec.builder(env, self.sequence_length)
        return obs
