"""控制器抽象基类。

定义所有控制器（MADRL、MPC、经典 DRL 等）的统一接口，
包含动作选择、模型保存加载等基本方法。

主要类:
    BaseController -- 控制器抽象基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseController(ABC):
    """所有评估控制器的最小统一接口（抽象基类）。

    所有控制器（MADRL、MPC、经典 DRL 等）必须继承此类，
    并实现 ``reset`` 与 ``act`` 两个抽象方法，以便在统一的
    评估流程中被调用。

    注意:
        子类在实现 ``act`` 时，返回的动作列表长度应与环境中
        智能体数量一致，每个元素为该智能体的动作向量。
    """

    @abstractmethod
    def reset(self) -> None:
        """在新 episode 开始前重置控制器内部状态。

        注意:
            无状态的控制器（如 ZeroController）可以保持空实现。
        """

    @abstractmethod
    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """根据结构化观测输出环境可直接执行的动作。

        参数:
            obs: 环境返回的结构化观测字典，包含键如
                ``"local"``（局部观测）、``"price_seq"``（电价序列）、
                ``"adjacency"``（邻接矩阵）等。
            deterministic: 若为 True，则抑制探索噪声，使用确定性策略。

        返回:
            list[np.ndarray]: 每个智能体一个动作数组，形状为
            ``(action_dim,)``，取值范围 ``[-1, 1]``。
        """
