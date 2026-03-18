"""经典单智能体 DRL 控制器。

将多智能体问题退化为单智能体处理的基线控制器。

主要类:
    ClassicDRLController -- 单智能体 DRL 控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class ClassicDRLController(BaseController):
    """Placeholder for single-agent DRL baselines (PPO/DQN/SAC).
    单智能体 DRL baseline 占位控制器。
    """

    def reset(self) -> None:
        """No internal state yet. Override when implementing."""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        raise NotImplementedError(
            "ClassicDRLController is a placeholder. See the module docstring for implementation guidance."
        )
