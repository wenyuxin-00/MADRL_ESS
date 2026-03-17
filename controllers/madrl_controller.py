"""MADRL controller。

职责只有一件事：把一组训练好的多智能体策略包装成统一的 compare 接口。
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class MADRLController(BaseController):
    """包装训练好的 MADRL agents。"""

    def __init__(self, agent_n, noise_std: float = 0.0):
        self.agent_n = list(agent_n)
        self.noise_std = float(noise_std)

    def reset(self) -> None:
        """当前前馈策略无额外状态。"""

    @staticmethod
    def _format_action(action) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim == 0:
            return action_array.reshape(1)
        return action_array

    def act(self, obs, deterministic: bool = True):
        """运行所有 agent，并返回环境可执行的动作列表。"""
        noise_std = 0.0 if deterministic else self.noise_std
        return [
            self._format_action(agent.choose_action(obs, noise_std=noise_std))
            for agent in self.agent_n
        ]
