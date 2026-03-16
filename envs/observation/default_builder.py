"""
envs/observation/default_builder.py
职责：由 args.obs_config 真正驱动的默认观测构造器。

DefaultObservationBuilder 将 obs_config 中的特征名映射到 Feature Block，
在 build(env) 时按顺序拼接各 block 的输出。

关键特性：
  - obs_config 中去掉任意项 → obs_dim 自动减少
  - obs_config = ["time","price","load","soc"] 且 future_horizon=24 → obs_dim=53（与旧版一致）
  - forecaster_type 切换 → PriceBlock 自动使用对应预测器
  - 无需修改此文件即可切换预测策略或消融特征

obs_dim 计算（future_horizon=24，K+1=25）：
  time  →  2
  price → 25
  load  → 25
  soc   →  1
  total = 53
"""

import numpy as np

from envs.observation.base import ObservationBuilder
from envs.observation.feature_blocks import DEFAULT_BLOCK_REGISTRY


class DefaultObservationBuilder(ObservationBuilder):
    """由 obs_config 列表驱动的默认观测构造器。

    Parameters
    ----------
    obs_config : list[str]
        特征名列表，e.g. ["time","price","load","soc"]。
        顺序决定观测向量的拼接顺序。
    future_horizon : int
        未来视界 K（PBRS 使用），决定 price/load block 的维度。
    """

    def __init__(self, obs_config: list, future_horizon: int):
        self.obs_config = list(obs_config)
        self.future_horizon = int(future_horizon)

        # 实例化各 block
        self._blocks = []
        for name in self.obs_config:
            if name not in DEFAULT_BLOCK_REGISTRY:
                raise ValueError(
                    f"未知 obs_config 项 '{name}'，可选: {list(DEFAULT_BLOCK_REGISTRY)}"
                )
            block = DEFAULT_BLOCK_REGISTRY[name](self.future_horizon)
            self._blocks.append((name, block))

        # 预计算 obs_dim（无需 env 实例）
        self._obs_dim = sum(b.dim() for _, b in self._blocks)

    def get_obs_dim(self) -> int:
        return self._obs_dim

    def build(self, env) -> np.ndarray:
        """构造当前时步的观测矩阵 (N, obs_dim)。"""
        N = env.n
        obs = np.zeros((N, self._obs_dim), dtype=np.float32)
        idx = 0
        for _, block in self._blocks:
            block_out = block.build(env)   # (N, block_dim)
            d = block.dim()
            obs[:, idx:idx + d] = block_out
            idx += d
        return obs
