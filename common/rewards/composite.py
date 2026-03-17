import numpy as np
from common.rewards.base import RewardFn, ComponentMeta


class CompositeReward(RewardFn):
    """5-component composite reward function (project default).
    5 分量复合奖励函数（项目默认方案）。

    reward = r_inc - r_pen + r_pbrs - r_soc + r_bonus
    奖励 = r_inc - r_pen + r_pbrs - r_soc + r_bonus

    Components / 各分量说明:
        - r_inc   : Incremental cost — "how much more did this action earn vs. idle?"
                    增量成本奖励，衡量"当前动作比闲置多赚了多少"
        - r_pen   : Action infeasibility penalty — soft constraint on battery limits
                    动作越限惩罚，软约束边界
        - r_pbrs  : Potential-Based Reward Shaping — guides buy-low-sell-high behavior
                    基于势能的奖励塑形，引导低买高卖的跨步行为
        - r_soc   : SoC regularization — encourages mid-range SoC for flexibility
                    SoC 正则化，鼓励保持中间 SoC 以维持灵活性
        - r_bonus : Throughput bonus — prevents degenerate always-idle policies
                    吞吐量奖励，防止智能体退化为永久闲置策略

    To add a new reward component / 新增分量时只需:
        1. Compute it in ``compute()``, add to return dict, update total formula
        2. Append a ``ComponentMeta`` entry to ``component_meta``
        -> No changes needed in hems_env.py or notebooks
    """

    def __init__(self, cfg):
        self.w_pen = float(cfg.reward.w_pen)
        self.w_soc = float(cfg.reward.w_soc)
        self.lambda_bonus = float(cfg.reward.lambda_bonus)
        self.p_max = float(cfg.env.max_charge_rate)
        self.soc_target = float(cfg.env.soc_target)

    @property
    def component_meta(self) -> list:
        return [
            ComponentMeta("r_inc",   "+ r_inc (incremental cost)",        "green",  +1),
            ComponentMeta("r_pen",   "- r_pen (action penalty)",          "orange", -1),
            ComponentMeta("r_pbrs",  "+ r_pbrs (PBRS)",                   "blue",   +1),
            ComponentMeta("r_soc",   "- r_soc (SoC regularization)",      "purple", -1),
            ComponentMeta("r_bonus", "+ r_bonus (throughput bonus)",       "teal",   +1),
        ]

    def compute(self, env_state: dict) -> tuple:
        e_bat_req = env_state["e_bat_req"]
        e_bat     = env_state["e_bat"]
        soc_t     = env_state["soc_t"]
        e_t       = env_state["e_t"]
        e_next    = env_state["e_next"]
        price_t   = env_state["price_t"]
        load_t    = env_state["load_t"]
        mu_t      = env_state["mu_t"]
        mu_next   = env_state["mu_next"]
        gamma     = env_state["gamma"]

        # (1) Incremental cost reward: negative of net energy cost minus baseline (idle) cost
        # 增量成本奖励：= -(net_load * price) - (-(load * price)) = -e_bat * price
        e_net  = load_t + e_bat
        r_inc  = (-(e_net * price_t) - (-(load_t * price_t))).astype(np.float32)

        # (2) Action infeasibility penalty: penalize gap between requested and executed power
        # 动作越限惩罚：请求功率与实际执行功率的差距（归一化）
        r_pen  = (self.w_pen * np.abs(e_bat_req - e_bat) / (self.p_max + 1e-6)).astype(np.float32)

        # (3) Potential-Based Reward Shaping (PBRS): phi(s) = mean_future_price * stored_energy
        # 基于势能的奖励塑形：phi(s) = 未来均价 × 当前储能
        phi_t    = (mu_t   * e_t).astype(np.float32)
        phi_next = (mu_next * e_next).astype(np.float32)
        r_pbrs   = (gamma * phi_next - phi_t).astype(np.float32)

        # (4) SoC regularization: quadratic penalty for deviation from target SoC
        # SoC 正则化：偏离目标 SoC 的二次惩罚
        r_soc  = (self.w_soc * (soc_t - self.soc_target) ** 2).astype(np.float32)

        # (5) Throughput bonus: reward proportional to |power|, discourages idle behavior
        # 吞吐量奖励：与|功率|成正比，避免智能体"什么都不做"
        r_bonus = (self.lambda_bonus * np.abs(e_bat)).astype(np.float32)

        total = (r_inc - r_pen + r_pbrs - r_soc + r_bonus).astype(np.float32)
        components = {
            "r_inc":   r_inc,
            "r_pen":   r_pen,
            "r_pbrs":  r_pbrs,
            "r_soc":   r_soc,
            "r_bonus": r_bonus,
        }
        return total, components
