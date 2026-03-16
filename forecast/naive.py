import numpy as np
from forecast.base import Forecaster


class NaiveForecaster(Forecaster):
    """朴素预测器：用历史滚动均值填充未来价格。

    统一 forecaster 契约下：
      - 第一个输出元素始终是当前真实价格 history[-1]
      - 后续 horizon-1 个位置用历史滚动均值填充

    实现方案：取历史最后 `window` 步的平均值，用该均值重复填充未来部分。
    这是最简单的无参数基准，用于量化"有任何预测信息"相对于"无信息"的收益。

    用途：消融实验中作为最低基准（低于此基准说明预测模型有负面影响）。
    """

    def __init__(self, window: int = 96):
        """
        Args:
            window: 历史滚动窗口大小（默认96步 = 1天）
        """
        self.window = window

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32).reshape(-1)
        current_price = float(history[-1]) if history.size > 0 else 0.0
        if horizon == 1:
            return np.array([current_price], dtype=np.float32)

        tail = history[-self.window:] if history.size >= self.window else history
        mean_price = float(np.mean(tail)) if tail.size > 0 else current_price
        future = np.full((horizon - 1,), mean_price, dtype=np.float32)
        return np.concatenate(
            [np.array([current_price], dtype=np.float32), future],
            axis=0,
        )
