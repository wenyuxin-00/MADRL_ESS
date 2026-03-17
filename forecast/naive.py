"""朴素价格预测器。

用途：提供一个几乎零参数的基线预测器，便于衡量“有预测信息”相对“无预测信息”的收益。
"""

from __future__ import annotations

import numpy as np

from forecast.base import Forecaster


class NaiveForecaster(Forecaster):
    """用最近窗口的均值填充未来价格。"""

    def __init__(self, window: int = 96):
        self.window = int(window)

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32).reshape(-1)
        current_price = float(history[-1]) if history.size > 0 else 0.0
        if horizon == 1:
            return np.array([current_price], dtype=np.float32)

        tail = history[-self.window :] if history.size >= self.window else history
        mean_price = float(np.mean(tail)) if tail.size > 0 else current_price
        future = np.full((horizon - 1,), mean_price, dtype=np.float32)
        return np.concatenate([np.array([current_price], dtype=np.float32), future], axis=0)
