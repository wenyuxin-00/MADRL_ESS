"""观测构造器接口。

环境状态可以变化，但对外暴露给模型的结构化观测必须稳定。
这里统一约束 schema、layout 与 build 逻辑，方便 notebook 调试和模型装配。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ObservationBuilder(ABC):
    """观测构造器抽象基类，为单个环境步构造结构化观测。

    所有观测构造器需实现以下三个抽象方法:
        - get_schema(): 定义观测字段的 shape
        - get_layout(): 定义观测字段的语义信息（用于模型装配和调试）
        - build(): 从环境状态构造实际观测数据
    """

    @abstractmethod
    def get_schema(self, n_agents: int) -> dict[str, tuple[int, ...]]:
        """返回每个观测字段的 shape 定义。

        参数:
            n_agents: 智能体数量

        返回:
            dict: 字段名 → shape 元组，如 {"local": (3, 5), "adjacency": (3, 3)}
        """

    @abstractmethod
    def get_layout(self, n_agents: int) -> dict[str, dict]:
        """返回每个观测字段的语义布局信息。

        布局信息包含 group、scope、dim、fields 等元数据，
        供模型装配和 notebook 调试时使用。

        参数:
            n_agents: 智能体数量

        返回:
            dict: 字段名 → 布局描述字典
        """

    @abstractmethod
    def build(self, env) -> dict[str, np.ndarray]:
        """从环境当前状态构造一步结构化观测。

        参数:
            env: 环境实例，需提供信号查询接口

        返回:
            dict[str, np.ndarray]: 字段名 → 观测数组
        """

    def zeros(self, n_agents: int) -> dict[str, np.ndarray]:
        """按当前 schema 返回全零观测，用于初始化缓冲区。

        参数:
            n_agents: 智能体数量

        返回:
            dict[str, np.ndarray]: 字段名 → 全零数组
        """
        return {
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in self.get_schema(n_agents).items()
        }
