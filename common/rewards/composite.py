import numpy as np
from common.rewards.base import RewardFn, ComponentMeta


class CompositeReward(RewardFn):
    """5 分量复合奖励函数（项目默认方案）。

    奖励 = r_inc - r_pen + r_pbrs - r_soc + r_bonus

    各分量说明：
    - r_inc  : 增量成本奖励，衡量"当前动作比闲置多赚了多少"
    - r_pen  : 动作越限惩罚，软约束边界
    - r_pbrs : 基于势能的奖励塑形，引导低买高卖的跨步行为
    - r_soc  : SoC 正则化，鼓励保持中间 SoC 以维持灵活性
    - r_bonus: 吞吐量奖励，防止智能体退化为永久闲置策略

    新增分量时只需：
      1. 在 compute() 中计算新分量，加入返回 dict，并更新 total 公式
      2. 在 component_meta 中追加 ComponentMeta 行
      → 无需修改 hems_env.py 或 train_madrl.ipynb
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

        # (1) 增量成本奖励
        e_net  = load_t + e_bat
        r_inc  = (-(e_net * price_t) - (-(load_t * price_t))).astype(np.float32)

        # (2) 动作越限惩罚（绝对值，符号由 sign=-1 管理）
        r_pen  = (self.w_pen * np.abs(e_bat_req - e_bat) / (self.p_max + 1e-6)).astype(np.float32)

        # (3) 基于势能的奖励塑形（PBRS）
        phi_t    = (mu_t   * e_t).astype(np.float32)
        phi_next = (mu_next * e_next).astype(np.float32)
        r_pbrs   = (gamma * phi_next - phi_t).astype(np.float32)

        # (4) SoC 正则化（绝对值，符号由 sign=-1 管理）
        r_soc  = (self.w_soc * (soc_t - self.soc_target) ** 2).astype(np.float32)

        # (5) 吞吐量奖励
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
