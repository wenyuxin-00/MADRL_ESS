"""稀疏奖励函数。

仅在 episode 结束时给出奖励的稀疏奖励方案。

主要类:
    SparseReward -- 稀疏奖励函数
"""

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn


class SparseArbitrageReward(RewardFn):
    """稀疏电价套利奖励，不包含任何塑形项。

    仅计算当前步的电价套利收益（baseline_cost - actual_cost），
    不使用 PBRS、SoC正则化等辅助奖励项。适用于评估智能体
    在无奖励塑形条件下的学习能力。

    属性:
        无需配置参数。
    """

    def __init__(self, args: object) -> None:
        """初始化稀疏奖励（无需额外参数）。

        参数:
            args: 实验配置对象（此处未使用，保留接口一致性）
        """
        _ = args

    @property
    def component_meta(self) -> list[ComponentMeta]:
        """返回单个奖励分量的元数据。"""
        return [
            ComponentMeta("r_arb", "+ r_arb (arbitrage profit)", "green", +1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """计算电价套利奖励。

        参数:
            env_state: 环境状态字典

        返回:
            tuple: (r_arb, {"r_arb": r_arb})，shape (N,)
        """
        e_bat = np.asarray(env_state["e_bat"], dtype=np.float32)
        price_t = float(env_state["price_t"])
        # 兼容无 PV 场景（回退到 load_t）
        net_load_t = np.asarray(env_state.get("net_load_t", env_state.get("load_t")), dtype=np.float32)
        dt = float(env_state.get("dt", 1.0))

        # 无储能时的用电成本（基线）
        baseline_cost = net_load_t * dt * price_t
        # 有储能时的用电成本
        actual_cost = (net_load_t + e_bat) * dt * price_t
        # 套利收益 = 节省的用电成本 = baseline_cost - actual_cost
        r_arb = (-(actual_cost) - (-(baseline_cost))).astype(np.float32)
        return r_arb, {"r_arb": r_arb}
