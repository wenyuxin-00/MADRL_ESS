"""预测器抽象基类。

定义所有预测器（perfect、naive、LSTM 等）的统一接口。

主要类:
    BaseForecaster -- 预测器抽象基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """观测侧预测器的抽象基类。

    预测器仅替换观测中的前瞻窗口部分，不改变环境的真实奖励计算。

    接口约定同时支持:
    - 共享的一维信号（如 ``price``），形状为 ``(T,)``
    - 每智能体的二维信号（如 ``load`` / ``pv``），形状为 ``(T, N)``
    """

    @abstractmethod
    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        """对单个信号返回前瞻预测窗口。

        参数:
            history: 截至当前时刻的观测信号历史。
                共享信号形状为 ``(t + 1,)``，
                每智能体信号形状为 ``(t + 1, n_agents)``。
            horizon: 需要返回的预测值数量。第一个位置包含当前值，
                即 ``result[..., 0]`` 始终代表当前时刻。
            signal_name: 信号的标准名称，如 ``price``、``load`` 或 ``pv``。

        返回:
            共享信号返回形状 ``(horizon,)``，
            每智能体信号返回形状 ``(n_agents, horizon)``。
        """

    def reset(self) -> None:
        """重置可选的回合内部状态（每个新回合开始时调用）。"""
        return None
