"""零动作基线控制器。

始终输出零动作的简单基线控制器，用于与其他算法对比。

主要类:
    ZeroController -- 零动作基线控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class ZeroController(BaseController):
    """始终输出零动作，作为最简单的对照基线。"""

    def __init__(self, action_dim_n: list[int] | None = None) -> None:
        self.action_dim_n = None if action_dim_n is None else [int(dim) for dim in action_dim_n]

    def reset(self) -> None:
        """Zero controller 无状态，无需重置。"""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """返回与环境兼容的全零动作列表。"""
        if self.action_dim_n is None:
            num_agents = int(np.asarray(obs["local"]).shape[-2])
            action_dim_n = [1 for _ in range(num_agents)]
        else:
            action_dim_n = self.action_dim_n
        return [np.zeros((action_dim,), dtype=np.float32) for action_dim in action_dim_n]
