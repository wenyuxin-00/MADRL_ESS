"""Unified controller interface for evaluation.
统一控制器评估接口。

All controllers (MADRL, Zero, MPC, classic DRL) are evaluated through this
minimal interface, enabling fair comparison in the compare notebook.

See ``controllers/zero_controller.py`` for the simplest implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseController(ABC):
    """Minimal interface for all evaluation controllers.
    所有评估控制器都要实现的最小接口。
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of a new episode.
        在新 episode 开始前重置内部状态。
        """

    @abstractmethod
    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """Return environment-executable actions given structured observations.
        根据结构化观测输出环境可直接执行的动作。

        Args:
            obs: Structured observation dict from the environment, with keys
                like ``"local"``, ``"price_seq"``, ``"adjacency"``, etc.
            deterministic: If True, suppress exploration noise.

        Returns:
            list[np.ndarray]: One action array per agent, each of shape
            ``(action_dim,)`` with values in ``[-1, 1]``.
        """
