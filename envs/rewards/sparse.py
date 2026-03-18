"""稀疏奖励函数。

仅在 episode 结束时给出奖励的稀疏奖励方案。

主要类:
    SparseReward -- 稀疏奖励函数
"""

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn


class SparseArbitrageReward(RewardFn):
    """Sparse arbitrage reward without shaping terms."""

    def __init__(self, args: object) -> None:
        _ = args

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            ComponentMeta("r_arb", "+ r_arb (arbitrage profit)", "green", +1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        e_bat = np.asarray(env_state["e_bat"], dtype=np.float32)
        price_t = float(env_state["price_t"])
        net_load_t = np.asarray(env_state.get("net_load_t", env_state.get("load_t")), dtype=np.float32)
        dt = float(env_state.get("dt", 1.0))

        baseline_cost = net_load_t * dt * price_t
        actual_cost = (net_load_t + e_bat) * dt * price_t
        r_arb = (-(actual_cost) - (-(baseline_cost))).astype(np.float32)
        return r_arb, {"r_arb": r_arb}
