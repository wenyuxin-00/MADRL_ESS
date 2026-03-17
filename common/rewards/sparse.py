import numpy as np
from common.rewards.base import RewardFn, ComponentMeta


class SparseArbitrageReward(RewardFn):
    """简单套利奖励函数（消融实验对照组）。

    奖励 = -(e_net * price_t) 的增量，即仅衡量"电池动作与闲置相比
    在当前时步节省/赚取的金额"。没有 PBRS、SoC 正则化等辅助项。

    切换到此奖励函数后，画图自动变为 2 个子图（1 total + 1 套利），
    无需修改 train_madrl.ipynb。
    """

    def __init__(self, args):
        pass  # 无超参数

    @property
    def component_meta(self) -> list:
        return [
            ComponentMeta("r_arb", "+ r_arb (arbitrage profit)", "green", +1),
        ]

    def compute(self, env_state: dict) -> tuple:
        e_bat   = env_state["e_bat"]
        price_t = env_state["price_t"]
        load_t  = env_state["load_t"]

        r_arb = (-(load_t + e_bat) * price_t + load_t * price_t).astype(np.float32)
        return r_arb, {"r_arb": r_arb}
