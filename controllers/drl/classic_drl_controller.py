"""经典单智能体 DRL 控制器。

将多智能体问题退化为单智能体处理的基线控制器。

主要类:
    ClassicDRLController -- 单智能体 DRL 控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class ClassicDRLController(BaseController):
    """经典单智能体深度强化学习基线控制器（占位实现）。

    将多智能体储能系统问题退化为单智能体处理的基线方案。
    可接入 PPO、DQN、SAC 等经典 DRL 算法，将所有储能单元
    视为一个统一的决策实体进行集中式控制。

    注意:
        当前为占位实现，调用 ``act`` 会抛出 NotImplementedError。
        实现时需要：
        1. 在 ``__init__`` 中初始化单智能体 DRL 策略网络
        2. 将多智能体观测拼接为单一观测向量
        3. 将单一动作输出拆分为各智能体的独立动作
    """

    def reset(self) -> None:
        """重置控制器内部状态。当前占位实现无状态。"""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """根据观测选择动作（尚未实现）。

        参数:
            obs: 环境观测字典。
            deterministic: 若为 True，使用确定性策略。

        异常:
            NotImplementedError: 当前为占位实现。
        """
        raise NotImplementedError(
            "ClassicDRLController is a placeholder. See the module docstring for implementation guidance."
        )
