"""预测器抽象基类。

所有预测器都遵循同一份轻量契约：
- 输入：到当前时刻为止的真实价格历史 `history = [p0, ..., pt]`
- 输出：长度为 `horizon` 的价格窗口，其中 `result[0] == pt`
- 预测器只影响观测中的价格窗口，不影响环境真实奖励计算
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """价格预测器基类。"""

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """基于价格历史返回一个前瞻窗口。"""

    def reset(self) -> None:
        """在 episode 重置时清理内部状态。"""
        return None
