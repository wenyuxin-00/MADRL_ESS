"""观测构造器接口。

环境状态可以变化，但对外暴露给模型的结构化观测必须稳定。
这里统一约束 schema、layout 与 build 逻辑，方便 notebook 调试和模型装配。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ObservationBuilder(ABC):
    """为单个环境步构造结构化观测。"""

    @abstractmethod
    def get_schema(self, n_agents: int) -> dict[str, tuple[int, ...]]:
        """返回每个观测字段的 shape。"""

    @abstractmethod
    def get_layout(self, n_agents: int) -> dict[str, dict]:
        """返回每个观测字段的语义布局信息。"""

    @abstractmethod
    def build(self, env) -> dict[str, np.ndarray]:
        """构造一步结构化观测。"""

    def zeros(self, n_agents: int) -> dict[str, np.ndarray]:
        """按当前 schema 返回全零观测。"""
        return {
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in self.get_schema(n_agents).items()
        }
