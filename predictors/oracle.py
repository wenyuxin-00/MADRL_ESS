"""Oracle（完美预知）预测器。

直接返回真实未来值，用作预测上界基线。

主要类:
    OracleForecaster -- 完美预知预测器
"""

from __future__ import annotations

import numpy as np

from predictors.base import Forecaster


class PerfectForecaster(Forecaster):
    """完美预知预测器：直接返回当前回合的真实未来值。

    该预测器在回合开始时注入完整信号序列，预测时直接切片返回真实值，
    作为预测精度的理论上界基线。

    属性:
        _episode_signals: 当前回合的完整信号字典，键为信号名，值为 numpy 数组。
    """

    def __init__(self, episode_signals: dict[str, np.ndarray] | np.ndarray | None = None):
        self._episode_signals: dict[str, np.ndarray] = {}
        if episode_signals is not None:
            self.set_episode(episode_signals)

    def set_episode(self, episode_signals: dict[str, np.ndarray] | np.ndarray) -> None:
        """在 ``env.reset()`` 时注入当前回合的完整信号数据。

        参数:
            episode_signals: 信号字典 ``{name: array}``，或单个数组（默认视为 price）。
        """
        if isinstance(episode_signals, dict):
            self._episode_signals = {
                name: np.asarray(values, dtype=np.float32)
                for name, values in episode_signals.items()
            }
            return

        # 非字典输入默认作为 price 信号处理
        self._episode_signals = {"price": np.asarray(episode_signals, dtype=np.float32)}

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        """从预存的真实信号中切片返回未来值。

        参数:
            history: 截至当前时刻的历史信号，用于推断当前时间索引。
            horizon: 需要返回的预测步数。
            signal_name: 信号名称，如 ``price``、``load``、``pv``。

        返回:
            一维信号返回 ``(horizon,)``，二维信号返回 ``(n_agents, horizon)``。
        """
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)
        # 校验请求的信号是否已注入
        if signal_name not in self._episode_signals:
            available = sorted(self._episode_signals)
            raise RuntimeError(
                f"PerfectForecaster has no episode signal '{signal_name}'. Available signals: {available}"
            )

        full_signal = np.asarray(self._episode_signals[signal_name], dtype=np.float32)
        history = np.asarray(history, dtype=np.float32)
        # 根据历史长度推断当前所处的时间步
        time_index = max(0, history.shape[0] - 1)

        # 处理一维共享信号
        if full_signal.ndim == 1:
            chunk = full_signal[time_index : time_index + horizon]
            # 如果剩余信号不足 horizon 步，用零填充
            if chunk.size < horizon:
                chunk = np.concatenate(
                    [chunk, np.zeros((horizon - chunk.size,), dtype=np.float32)],
                    axis=0,
                )
            return chunk[:horizon].astype(np.float32)

        # 处理二维每智能体信号
        if full_signal.ndim != 2:
            raise ValueError(f"PerfectForecaster expects 1D or 2D episode signals, got {full_signal.shape}")

        chunk = full_signal[time_index : time_index + horizon, :]
        # 不足 horizon 步时用零填充
        if chunk.shape[0] < horizon:
            pad = np.zeros((horizon - chunk.shape[0], chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        # 转置为 (n_agents, horizon) 格式返回
        return chunk[:horizon, :].T.astype(np.float32)
