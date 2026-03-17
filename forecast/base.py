"""Forecaster abstract base class.
预测器抽象基类。

All forecasters follow the same lightweight contract:
所有预测器都遵循同一份轻量契约：

    - Input: real price history up to current step ``[p0, ..., pt]``
      输入：到当前时刻为止的真实价格历史
    - Output: price window of length ``horizon``, where ``result[0] == pt``
      输出：长度为 horizon 的价格窗口，其中 result[0] == pt
    - Forecasters only affect the observation's price window, NOT the
      environment's true reward computation.
      预测器只影响观测中的价格窗口，不影响环境真实奖励计算

See ``forecast/oracle.py`` (simplest) and ``forecast/lstm_forecaster.py``
(most complex) for implementation examples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Price forecaster base class.
    价格预测器基类。
    """

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Return a look-ahead price window given price history.
        基于价格历史返回一个前瞻窗口。

        Args:
            history: 1-D array of observed prices ``[p0, ..., pt]``.
            horizon: Number of future steps to predict.

        Returns:
            np.ndarray of shape ``(horizon,)`` where ``result[0] == pt``
            (current price) and ``result[1:]`` are forecasted future prices.
        """

    def reset(self) -> None:
        """Reset internal state at episode boundaries (optional).
        在 episode 重置时清理内部状态（可选）。
        """
        return None
