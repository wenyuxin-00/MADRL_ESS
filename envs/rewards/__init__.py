"""奖励函数框架。

提供可插拔的奖励函数注册与实例化机制。

已实现奖励函数:
    CompositeReward       -- 5 分量加权奖励（成本、惩罚、PBRS、SoC、吞吐量）
    GridCompositeReward   -- 带电网约束惩罚的复合奖励
    SparseArbitrageReward -- 稀疏 episode 末套利奖励

如何添加新奖励函数:
    1. 创建 ``envs/rewards/your_reward.py``，实现 ``RewardFn``
    2. 在此处注册::

           register_reward("your_reward", YourReward)

    3. 在配置中使用: ``cfg.reward.type = "your_reward"``

主要函数:
    get_reward_fn    -- 根据名称实例化奖励函数
    register_reward  -- 注册新奖励函数类
"""

from envs.rewards.base import RewardFn
from envs.rewards.composite import CompositeReward
from envs.rewards.grid_composite import GridCompositeReward
from envs.rewards.sparse import SparseArbitrageReward

REWARD_REGISTRY: dict[str, type[RewardFn]] = {
    "composite":       CompositeReward,
    "sparse":          SparseArbitrageReward,
    "grid_composite":  GridCompositeReward,
}


def register_reward(name: str, reward_cls: type[RewardFn]) -> None:
    """注册一个新的奖励函数类。"""
    REWARD_REGISTRY[name] = reward_cls


def get_reward_fn(name: str, args) -> RewardFn:
    """根据名称实例化奖励函数。

    Args:
        name: 奖励函数名称，可选 'composite' | 'sparse' | 'grid_composite'
        args: 传递给奖励函数构造函数的配置对象

    Returns:
        RewardFn 实例
    """
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward type '{name}', available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name](args)
