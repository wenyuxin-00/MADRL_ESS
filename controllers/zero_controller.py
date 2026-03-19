"""零动作基线控制器。

始终输出零动作的简单基线控制器，用于与其他算法对比。

主要类:
    ZeroController -- 零动作基线控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class ZeroController(BaseController):
    """始终输出零动作的基线控制器。

    作为最简单的对照基线，所有智能体在每个时间步均输出全零动作，
    即储能系统既不充电也不放电。可用于评估其他算法相较于
    "什么都不做"的收益提升。

    属性:
        action_dim_n: 各智能体的动作维度列表；为 None 时自动推断。
    """

    def __init__(self, action_dim_n: list[int] | None = None) -> None:
        """初始化零动作控制器。

        参数:
            action_dim_n: 各智能体的动作维度列表。若为 None，则在
                ``act`` 时根据观测自动推断智能体数量，默认每个维度为 1。
        """
        # 将外部传入的维度列表转为整数列表，确保类型安全
        self.action_dim_n = None if action_dim_n is None else [int(dim) for dim in action_dim_n]

    def reset(self) -> None:
        """零动作控制器无内部状态，无需重置。"""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """返回与环境兼容的全零动作列表。

        参数:
            obs: 环境观测字典，至少包含 ``"local"`` 键。
            deterministic: 该参数对零控制器无影响，保留以兼容接口。

        返回:
            list[np.ndarray]: 每个智能体一个全零动作数组。
        """
        if self.action_dim_n is None:
            # 未指定动作维度时，从观测的倒数第二维推断智能体数量
            num_agents = int(np.asarray(obs["local"]).shape[-2])
            action_dim_n = [1 for _ in range(num_agents)]
        else:
            action_dim_n = self.action_dim_n
        return [np.zeros((action_dim,), dtype=np.float32) for action_dim in action_dim_n]
