"""经典单智能体 DRL baseline 占位符。

这里预留 compare notebook 入口，后续可接 PPO / DQN / SAC 等基线。
"""

from __future__ import annotations

from controllers.base import BaseController


class ClassicDRLController(BaseController):
    """未来用于接入经典 DRL baseline 的统一占位 controller。"""

    def reset(self) -> None:
        """占位实现，无内部状态。"""

    def act(self, obs, deterministic: bool = True):
        raise NotImplementedError("ClassicDRLController 目前仍是占位符，尚未接入具体 baseline。")
