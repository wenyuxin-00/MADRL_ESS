"""完美预测器。

直接返回当前 episode 的真实未来价格，用于上界对照实验。
"""

from __future__ import annotations

import numpy as np

from forecast.base import Forecaster


class PerfectForecaster(Forecaster):
    """Oracle 预测器。"""

    def __init__(self, full_price: np.ndarray | None = None):
        self._full_price = None if full_price is None else np.asarray(full_price, dtype=np.float32)

    def set_episode(self, full_price: np.ndarray) -> None:
        """在 `env.reset()` 时注入当前 episode 的完整价格序列。"""
        self._full_price = np.asarray(full_price, dtype=np.float32)

    def reset(self) -> None:
        """Oracle 本身无额外滚动状态。"""
        return None

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._full_price is None:
            raise RuntimeError("PerfectForecaster 尚未设置 episode 价格，请先调用 set_episode()。")
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        t = len(history) - 1
        chunk = self._full_price[t : t + horizon]
        if len(chunk) < horizon:
            chunk = np.concatenate([chunk, np.zeros(horizon - len(chunk), dtype=np.float32)])
        return chunk[:horizon].astype(np.float32)
