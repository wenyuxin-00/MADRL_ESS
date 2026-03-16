"""
forecast/oracle.py
职责：完美预测器（Oracle）——直接返回当前 episode 的真实未来价格。

PerfectForecaster 的行为与重构前 hems_env._get_obs_matrix 的价格窗口完全一致：
  - 末尾使用零填充（而非重复最后值），与 _pad_window_1d 对齐
  - 需要在 env.reset() 时通过 set_episode(ep_price) 注入当前 episode 价格

用途：
  - 默认配置（forecaster_type="perfect"）的回归测试基准
  - 性能上界对照实验
"""

import numpy as np
from forecast.base import Forecaster


class PerfectForecaster(Forecaster):
    """完美预测器（Oracle）：直接返回真实未来价格。

    输出契约与 Forecaster 基类一致：
      - result[0] 是当前真实价格
      - result[1:] 是后续真实价格，末尾不足时零填充

    末尾零填充，与 hems_env._pad_window_1d 完全一致，确保回归行为不变。
    """

    def __init__(self, full_price: np.ndarray = None):
        """
        Parameters
        ----------
        full_price : np.ndarray, optional
            当前 episode 完整价格序列 shape (T,)。
            可在 env.reset() 时通过 set_episode() 注入，无需在构造时提供。
        """
        self._full_price = full_price

    def set_episode(self, full_price: np.ndarray) -> None:
        """在 env.reset() 时调用，注入本 episode 的真实价格序列。"""
        self._full_price = np.asarray(full_price, dtype=np.float32)

    def reset(self) -> None:
        # set_episode 负责在 reset 时注入，此处无需额外操作
        pass

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """返回从当前步（t = len(history)-1）开始的 horizon 步真实价格窗口。

        第一个元素是当前真实价格，末尾零填充（与 _pad_window_1d 行为一致）。
        """
        if self._full_price is None:
            raise RuntimeError(
                "PerfectForecaster 尚未设置 episode 价格，"
                "请在 env.reset() 前调用 set_episode()。"
            )
        t = len(history) - 1
        chunk = self._full_price[t: t + horizon]
        if len(chunk) < horizon:
            # 零填充，与 hems_env._pad_window_1d 行为一致
            chunk = np.concatenate(
                [chunk, np.zeros(horizon - len(chunk), dtype=np.float32)]
            )
        return chunk[:horizon].astype(np.float32)
