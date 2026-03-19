"""模型预测控制（MPC）控制器。

基于线性规划的 MPC 基线控制器，使用未来预测信息
求解最优充放电策略。

主要类:
    MPCController -- MPC 控制器
"""

from __future__ import annotations

import numpy as np

from controllers.base import BaseController


class MPCController(BaseController):
    """模型预测控制（MPC）基线控制器（占位实现）。

    MPC 方法利用系统动力学模型和未来预测信息（如电价、负荷预测），
    通过求解有限时域的优化问题（通常为线性规划或二次规划），
    得到最优充放电策略。

    注意:
        当前为占位实现，调用 ``act`` 会抛出 NotImplementedError。
        实现时需要：
        1. 在 ``__init__`` 中接收预测模型和优化求解器配置
        2. 在 ``act`` 中构建并求解滚动时域优化问题
        3. 返回第一个时间步的最优动作
    """

    def reset(self) -> None:
        """重置控制器内部状态。当前占位实现无状态。"""

    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        """根据观测计算最优控制动作（尚未实现）。

        参数:
            obs: 环境观测字典。
            deterministic: MPC 本身为确定性方法，该参数保留以兼容接口。

        异常:
            NotImplementedError: 当前为占位实现。
        """
        raise NotImplementedError(
            "MPCController is a placeholder. See the module docstring for implementation guidance."
        )
