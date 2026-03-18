"""Reward function framework.
奖励函数框架。

Available reward functions / 已实现奖励函数:
    - CompositeReward       -- 5-component reward (cost, penalty, PBRS, SoC, throughput)
    - SparseArbitrageReward -- sparse end-of-episode arbitrage reward

How to add a new reward function / 如何添加新奖励函数:
    1. Create ``common/rewards/your_reward.py``, implementing ``RewardFn`` from ``base.py``
    2. Register here::

           register_reward("your_reward", YourReward)

    3. Use in config: ``cfg.reward.type = "your_reward"``

See ``common/rewards/base.py`` for the interface contract and ``composite.py``
for a full implementation example.
"""

from common.rewards.base import RewardFn
from common.rewards.composite import CompositeReward
from common.rewards.grid_composite import GridCompositeReward
from common.rewards.sparse import SparseArbitrageReward

REWARD_REGISTRY: dict[str, type[RewardFn]] = {
    "composite":       CompositeReward,
    "sparse":          SparseArbitrageReward,
    "grid_composite":  GridCompositeReward,
}


def register_reward(name: str, reward_cls: type[RewardFn]) -> None:
    """Register a new reward function class.
    注册一个新的奖励函数类。
    """
    REWARD_REGISTRY[name] = reward_cls


def get_reward_fn(name: str, args) -> RewardFn:
    """Instantiate a reward function by name.
    根据名称实例化奖励函数。

    Args:
        name: Reward function name, one of 'composite' | 'sparse'
              奖励函数名称，可选 'composite' | 'sparse'
        args: Config object passed to the reward function constructor
              传递给奖励函数构造函数的配置对象

    Returns:
        RewardFn instance / RewardFn 实例
    """
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward type '{name}', available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name](args)
