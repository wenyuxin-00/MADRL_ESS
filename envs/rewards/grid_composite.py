"""带电网约束的复合奖励函数。

在 CompositeReward 基础上增加电压越界和线路过载惩罚。

主要类:
    GridCompositeReward -- 电网约束复合奖励
"""

from __future__ import annotations

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn
from envs.rewards.composite import CompositeReward


class GridCompositeReward(RewardFn):
    """带电网约束的复合奖励函数。

    在 CompositeReward 的五个基础分量（r_inc, r_pen, r_pbrs, r_soc, r_bonus）
    上追加两个电网约束惩罚项:
        - r_v_pen: 电压越限惩罚（逐智能体，基于所在节点的电压偏差）
        - r_l_pen: 线路过载惩罚（全局标量，广播到所有智能体）

    属性:
        _base: 基础复合奖励实例（CompositeReward）
        w_v_pen: 电压越限惩罚权重
        w_l_pen: 线路过载惩罚权重
    """

    def __init__(self, cfg: object) -> None:
        """初始化电网复合奖励，创建基础奖励实例并读取电网惩罚权重。

        参数:
            cfg: 实验配置对象，需包含 cfg.grid.w_v_pen 和 cfg.grid.w_l_pen
        """
        self._base = CompositeReward(cfg)
        self.w_v_pen = float(cfg.grid.w_v_pen)    # 电压越限惩罚权重
        self.w_l_pen = float(cfg.grid.w_l_pen)    # 线路过载惩罚权重

    @property
    def component_meta(self) -> list[ComponentMeta]:
        """返回七个奖励分量的元数据（基础五项 + 电网约束两项）。"""
        return [
            *self._base.component_meta,
            ComponentMeta("r_v_pen", "- r_v_pen (voltage violation)", "red",   -1),
            ComponentMeta("r_l_pen", "- r_l_pen (line overload)",     "brown", -1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """计算带电网约束的复合奖励。

        总奖励 = 基础五项总和 - r_v_pen - r_l_pen

        参数:
            env_state: 环境状态字典，除基础键外还需包含:
                - v_violation (N,): 各智能体节点的电压越限量（可选，默认0）
                - l_violation float: 线路最大过载量（可选，默认0）

        返回:
            tuple: (total_reward, components) 包含全部七个分量
        """
        # 先计算基础的五个奖励分量
        base_total, components = self._base.compute(env_state)

        n_agents = base_total.shape[0]

        # ---- r_v_pen: 电压越限惩罚（逐智能体） ----
        # v_violation[i] = max(0, v_min - vm[i]) + max(0, vm[i] - v_max)
        v_violation = np.asarray(
            env_state.get("v_violation", np.zeros(n_agents, dtype=np.float32)),
            dtype=np.float32,
        )
        # 支持从 env_state 动态覆盖权重（如课程学习中逐步增大惩罚）
        w_v = float(env_state.get("w_v_pen", self.w_v_pen))
        r_v_pen = (w_v * v_violation).astype(np.float32)

        # ---- r_l_pen: 线路过载惩罚（全局标量，广播到所有智能体） ----
        # l_violation = max(0, max(loading%) - limit%) / 100
        l_violation = float(env_state.get("l_violation", 0.0))
        w_l = float(env_state.get("w_l_pen", self.w_l_pen))
        r_l_pen = np.full(n_agents, w_l * l_violation, dtype=np.float32)

        # 在基础奖励上减去电网约束惩罚
        total = (base_total - r_v_pen - r_l_pen).astype(np.float32)
        components["r_v_pen"] = r_v_pen
        components["r_l_pen"] = r_l_pen
        return total, components
