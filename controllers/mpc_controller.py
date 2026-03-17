"""MPC controller 占位符。

本轮重构先把 compare 入口和接口留好，具体 MPC 逻辑后续再接入。
"""

from __future__ import annotations

from controllers.base import BaseController


class MPCController(BaseController):
    """未来用于接入 MPC 的统一占位 controller。"""

    def reset(self) -> None:
        """占位实现，无内部状态。"""

    def act(self, obs, deterministic: bool = True):
        raise NotImplementedError("MPCController 目前仍是占位符，尚未实现优化求解逻辑。")
