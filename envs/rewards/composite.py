"""复合奖励函数。

组合电价套利奖励、SoC 惩罚等多个分量的加权奖励函数。

主要类:
    CompositeReward -- 复合奖励函数
"""

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn


class CompositeReward(RewardFn):
    """Composite reward with cost, penalty, PBRS, SoC, and throughput terms."""

    def __init__(self, cfg: object) -> None:
        self.w_pen = float(cfg.reward.w_pen)
        self.w_soc = float(cfg.reward.w_soc)
        self.lambda_bonus = float(cfg.reward.lambda_bonus)
        self.p_max = float(cfg.env.max_charge_rate)
        self.soc_target = float(cfg.env.soc_target)

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            ComponentMeta("r_inc", "+ r_inc (incremental cost)", "green", +1),
            ComponentMeta("r_pen", "- r_pen (action penalty)", "orange", -1),
            ComponentMeta("r_pbrs", "+ r_pbrs (PBRS)", "blue", +1),
            ComponentMeta("r_soc", "- r_soc (SoC regularization)", "purple", -1),
            ComponentMeta("r_bonus", "+ r_bonus (throughput bonus)", "teal", +1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        e_bat_req = np.asarray(env_state["e_bat_req"], dtype=np.float32)
        e_bat = np.asarray(env_state["e_bat"], dtype=np.float32)
        soc_t = np.asarray(env_state["soc_t"], dtype=np.float32)
        e_t = np.asarray(env_state["e_t"], dtype=np.float32)
        e_next = np.asarray(env_state["e_next"], dtype=np.float32)
        price_t = float(env_state["price_t"])
        net_load_t = np.asarray(env_state.get("net_load_t", env_state.get("load_t")), dtype=np.float32)
        mu_t = float(env_state["mu_t"])
        mu_next = float(env_state["mu_next"])
        gamma = float(env_state["gamma"])
        dt = float(env_state.get("dt", 1.0))
        p_max = np.asarray(env_state.get("p_max", self.p_max), dtype=np.float32)

        grid_power = net_load_t + e_bat
        baseline_cost = net_load_t * dt * price_t
        actual_cost = grid_power * dt * price_t
        r_inc = (-(actual_cost) - (-(baseline_cost))).astype(np.float32)

        r_pen = (self.w_pen * np.abs(e_bat_req - e_bat) / (p_max + 1e-6)).astype(np.float32)

        phi_t = (mu_t * e_t).astype(np.float32)
        phi_next = (mu_next * e_next).astype(np.float32)
        r_pbrs = (gamma * phi_next - phi_t).astype(np.float32)

        r_soc = (self.w_soc * (soc_t - self.soc_target) ** 2).astype(np.float32)

        r_bonus = (self.lambda_bonus * np.abs(e_bat) * dt).astype(np.float32)

        total = (r_inc - r_pen + r_pbrs - r_soc + r_bonus).astype(np.float32)
        components = {
            "r_inc": r_inc,
            "r_pen": r_pen,
            "r_pbrs": r_pbrs,
            "r_soc": r_soc,
            "r_bonus": r_bonus,
        }
        return total, components
