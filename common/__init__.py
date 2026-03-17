"""Common utilities shared across the MADRL-ESS platform.
通用工具模块，为整个平台提供共享基础设施。

Submodules / 子模块:
    common.rewards          -- Reward function framework (base, composite, sparse)
                               奖励函数框架（基类、组合奖励、稀疏奖励）
    common.replay_buffer    -- Experience replay buffer for off-policy MADRL
                               经验回放缓冲区
    common.vec_env          -- DummyVecEnv for serial environment rollouts
                               串行向量化环境
    common.subproc_vec_env  -- SubprocVecEnv for parallel environment rollouts
                               并行子进程向量化环境
    common.nested           -- Helpers for nested numpy/torch batch structures
                               嵌套字典/数组批处理工具
"""

from common.nested import add_batch_dim, index_nested, stack_nested, to_torch_nested
from common.replay_buffer import ReplayBuffer
from common.vec_env import DummyVecEnv

__all__ = [
    "DummyVecEnv",
    "ReplayBuffer",
    "add_batch_dim",
    "index_nested",
    "stack_nested",
    "to_torch_nested",
]
