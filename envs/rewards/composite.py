"""复合奖励函数。

组合电价套利奖励、SoC 惩罚等多个分量的加权奖励函数。

主要类:
    CompositeReward -- 复合奖励函数
"""

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn


class CompositeReward(RewardFn):
    """复合奖励函数，组合电价套利、动作惩罚、势能塑形、SoC正则化和吞吐量奖励。

    五个分量:
        - r_inc:   增量成本奖励（电价套利收益）
        - r_pen:   动作越限惩罚（请求功率与实际功率的差异）
        - r_pbrs:  基于势能的奖励塑形（Potential-Based Reward Shaping）
        - r_soc:   SoC 正则化惩罚（偏离目标 SoC 的二次惩罚）
        - r_bonus: 吞吐量奖励（鼓励储能系统活跃充放电）

    属性:
        w_pen: 动作越限惩罚权重
        w_soc: SoC 正则化惩罚权重
        lambda_bonus: 吞吐量奖励系数
        p_max: 最大充放电功率（用于归一化惩罚项）
        soc_target: 目标 SoC（用于 SoC 正则化）
    """

    def __init__(self, cfg: object) -> None:
        """从实验配置中读取奖励函数的各项权重。

        参数:
            cfg: 实验配置对象，需包含 cfg.reward 和 cfg.env 子配置
        """
        self.w_pen = float(cfg.reward.w_pen)
        self.w_soc = float(cfg.reward.w_soc)
        self.lambda_bonus = float(cfg.reward.lambda_bonus)
        self.p_max = float(cfg.env.max_charge_rate)
        self.soc_target = float(cfg.env.soc_target)

    @property
    def component_meta(self) -> list[ComponentMeta]:
        """返回五个奖励分量的元数据，用于驱动可视化和历史记录。"""
        return [
            ComponentMeta("r_inc", "+ r_inc (incremental cost)", "green", +1),
            ComponentMeta("r_pen", "- r_pen (action penalty)", "orange", -1),
            ComponentMeta("r_pbrs", "+ r_pbrs (PBRS)", "blue", +1),
            ComponentMeta("r_soc", "- r_soc (SoC regularization)", "purple", -1),
            ComponentMeta("r_bonus", "+ r_bonus (throughput bonus)", "teal", +1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """计算复合奖励的各分量并汇总。

        总奖励公式: total = r_inc - r_pen + r_pbrs - r_soc + r_bonus

        参数:
            env_state: 环境状态字典，键说明见 RewardFn 基类

        返回:
            tuple: (total_reward, components)
                - total_reward: shape (N,)，各智能体的总奖励
                - components: 各分量的绝对值字典
        """
        # ---- 提取环境状态 ----
        e_bat_req = np.asarray(env_state["e_bat_req"], dtype=np.float32)   # 请求功率（可能越限）
        e_bat = np.asarray(env_state["e_bat"], dtype=np.float32)           # 实际执行功率（可行域投影后）
        soc_t = np.asarray(env_state["soc_t"], dtype=np.float32)           # 当前 SoC
        e_t = np.asarray(env_state["e_t"], dtype=np.float32)               # 当前储能量
        e_next = np.asarray(env_state["e_next"], dtype=np.float32)         # 下一步储能量
        price_t = float(env_state["price_t"])                              # 当前电价
        # net_load_t = 负荷 - 光伏，兼容无 PV 场景（回退到 load_t）
        net_load_t = np.asarray(env_state.get("net_load_t", env_state.get("load_t")), dtype=np.float32)
        mu_t = float(env_state["mu_t"])           # 当前步的未来 K 步平均电价
        mu_next = float(env_state["mu_next"])      # 下一步的未来 K 步平均电价
        gamma = float(env_state["gamma"])          # 折扣因子（PBRS 使用）
        dt = float(env_state.get("dt", 1.0))      # 时间步长（小时）
        p_max = np.asarray(env_state.get("p_max", self.p_max), dtype=np.float32)

        # ---- r_inc: 增量成本奖励（电价套利） ----
        # 有储能时的电网功率 = 净负荷 + 电池功率（充电为正，放电为负）
        grid_power = net_load_t + e_bat
        baseline_cost = net_load_t * dt * price_t    # 无储能时的用电成本
        actual_cost = grid_power * dt * price_t      # 有储能时的用电成本
        # r_inc = -actual_cost - (-baseline_cost) = baseline_cost - actual_cost
        r_inc = (-(actual_cost) - (-(baseline_cost))).astype(np.float32)

        # ---- r_pen: 动作越限惩罚 ----
        # 请求功率与实际功率差异越大，惩罚越重；除以 p_max 做归一化
        r_pen = (self.w_pen * np.abs(e_bat_req - e_bat) / (p_max + 1e-6)).astype(np.float32)

        # ---- r_pbrs: 基于势能的奖励塑形 ----
        # 势函数 phi = mu * e（平均电价 x 储能量），鼓励在低价时蓄能
        phi_t = (mu_t * e_t).astype(np.float32)
        phi_next = (mu_next * e_next).astype(np.float32)
        r_pbrs = (gamma * phi_next - phi_t).astype(np.float32)

        # ---- r_soc: SoC 正则化惩罚 ----
        # 二次惩罚，使 SoC 趋向目标值
        r_soc = (self.w_soc * (soc_t - self.soc_target) ** 2).astype(np.float32)

        # ---- r_bonus: 吞吐量奖励 ----
        # 鼓励储能系统积极参与充放电操作
        r_bonus = (self.lambda_bonus * np.abs(e_bat) * dt).astype(np.float32)

        # ---- 汇总 ----
        total = (r_inc - r_pen + r_pbrs - r_soc + r_bonus).astype(np.float32)
        components = {
            "r_inc": r_inc,
            "r_pen": r_pen,
            "r_pbrs": r_pbrs,
            "r_soc": r_soc,
            "r_bonus": r_bonus,
        }
        return total, components
