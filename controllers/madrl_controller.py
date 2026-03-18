"""多智能体深度强化学习（MADRL）控制器。

封装 MADRL 算法（MADDPG / MATD3）的高层控制器接口，
负责多智能体联合动作选择与训练调度。

主要类:
    MADRLController -- MADRL 控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class MADRLController(BaseController):
    """包装训练好的 MADRL agents。"""

    def __init__(self, agent_n: list, noise_std: float = 0.0) -> None:
        self.agent_n = list(agent_n)
        self.noise_std = float(noise_std)

    def reset(self) -> None:
        """当前前馈策略无额外状态。"""

    @staticmethod
    def _format_action(action: object) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 0:
            return action_array.reshape(1)
        return action_array

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """运行所有 agent，并返回环境可执行的动作列表。"""
        noise_std = 0.0 if deterministic else self.noise_std
        return [
            self._format_action(agent.choose_action(obs, noise_std=noise_std))
            for agent in self.agent_n
        ]
