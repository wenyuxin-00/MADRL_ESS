from abc import ABC, abstractmethod
import numpy as np


class Forecaster(ABC):
    """价格预测模型的抽象基类。

    所有实现需支持：给定到当前时步为止的历史价格序列，
    返回一个长度为 horizon 的价格窗口，且**第一个元素必须是当前时刻
    已知真实价格**。

    统一契约：
      - 输入 history = [p0, ..., pt]
      - 输出 predict(history, horizon) = [pt, ..., future...]
      - 当 horizon == 1 时，只返回 [pt]

    这影响的是环境**观测**中的价格窗口，不影响 PBRS 奖励塑形
    （PBRS 依然使用真实价格以避免奖励偏差）。
    """

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """
        Args:
            history: 到当前时步（含）的历史电价序列，shape (t+1,)
            horizon: 需要返回的窗口长度（K+1，含当前步）

        Returns:
            价格窗口，shape (horizon,), dtype float32，
            且 result[0] == history[-1]
        """
        ...

    def reset(self) -> None:
        """episode 重置时调用（有状态的预测模型需要实现此方法）。"""
        pass
