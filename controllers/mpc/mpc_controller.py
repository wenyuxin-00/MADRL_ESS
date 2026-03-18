"""模型预测控制（MPC）控制器。

基于线性规划的 MPC 基线控制器，使用未来预测信息
求解最优充放电策略。

主要类:
    MPCController -- MPC 控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class MPCController(BaseController):
    """Placeholder for Model Predictive Control baseline.
    MPC baseline 占位控制器。
    """

    def reset(self) -> None:
        """No internal state yet. Override when implementing."""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        raise NotImplementedError(
            "MPCController is a placeholder. See the module docstring for implementation guidance."
        )
