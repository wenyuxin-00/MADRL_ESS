from common.rewards.base import RewardFn
from common.rewards.composite import CompositeReward
from common.rewards.sparse import SparseArbitrageReward

REWARD_REGISTRY: dict = {
    "composite": CompositeReward,
    "sparse":    SparseArbitrageReward,
}


def get_reward_fn(name: str, args) -> RewardFn:
    """根据名称实例化奖励函数。

    Args:
        name: 奖励函数名称，可选 'composite' | 'sparse'
        args: Config 对象，传递给奖励函数构造函数

    Returns:
        RewardFn 实例
    """
    if name not in REWARD_REGISTRY:
        raise ValueError(f"未知奖励函数 '{name}'，可选: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name](args)
