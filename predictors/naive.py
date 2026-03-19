"""朴素预测器。

基于历史窗口复制的简单基线预测器。

主要类:
    NaiveForecaster -- 朴素预测器
"""

from __future__ import annotations

import numpy as np

from predictors.base import Forecaster


class NaiveForecaster(Forecaster):
    """朴素预测器：将近期历史均值重复作为轻量级基线预测。

    属性:
        window: 用于计算均值的历史窗口大小（步数），默认 96（即一天的 15 分钟步长）。
    """

    def __init__(self, window: int = 96):
        self.window = int(window)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        """基于历史窗口均值生成前瞻预测。

        参数:
            history: 历史信号数组，一维 ``(T,)`` 或二维 ``(T, N)``。
            horizon: 预测步数（包含当前值）。
            signal_name: 信号名称（本预测器未使用，保留接口兼容性）。

        返回:
            共享信号返回 ``(horizon,)``，每智能体信号返回 ``(N, horizon)``。
        """
        del signal_name  # 朴素预测器不区分信号类型
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32)
        # 统一为二维处理，记录是否需要在输出时压缩回一维
        if history.ndim == 1:
            history = history.reshape(-1, 1)
            squeeze_output = True
        elif history.ndim == 2:
            squeeze_output = False
        else:
            raise ValueError(f"NaiveForecaster expects 1D or 2D history, got shape {history.shape}")

        # 提取当前时刻的值（历史序列的最后一行）
        current_value = history[-1] if history.shape[0] > 0 else np.zeros((history.shape[1],), dtype=np.float32)
        # 只需一步预测时直接返回当前值
        if horizon == 1:
            return current_value.astype(np.float32) if squeeze_output else current_value[:, None].astype(np.float32)

        # 取最近 window 步的历史计算均值作为未来预测
        tail = history[-self.window :] if history.shape[0] >= self.window else history
        mean_value = tail.mean(axis=0) if tail.size > 0 else current_value
        # 将均值重复 horizon-1 次构成未来预测部分
        future = np.repeat(mean_value.reshape(-1, 1), horizon - 1, axis=1).astype(np.float32)
        # 拼接当前值和未来预测
        window = np.concatenate([current_value.reshape(-1, 1), future], axis=1).astype(np.float32)
        return window[0] if squeeze_output else window
