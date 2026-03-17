"""统一 controller 接口。

compare notebook 里无论是 MADRL、Zero、MPC 还是经典 DRL，
都应该通过这一组最小方法进行评估。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseController(ABC):
    """所有评估控制器都要实现的最小接口。"""

    @abstractmethod
    def reset(self) -> None:
        """在新 episode 开始前重置内部状态。"""

    @abstractmethod
    def act(self, obs, deterministic: bool = True):
        """根据结构化观测输出环境可直接执行的动作。"""
