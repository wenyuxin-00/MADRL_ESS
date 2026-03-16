"""
envs/observation/base.py
职责：定义观测构造器的统一抽象接口。

ObservationBuilder 将 env 的内部状态映射到 agent 的观测向量。
- get_obs_dim() 在构建时即可确定，无需 env 实例
- build(env) 在每次 reset/step 后调用，返回 (N, obs_dim) 矩阵

设计意图：
  通过依赖注入将观测逻辑与物理 env 解耦，使 obs_config 消融实验
  只需替换 obs_builder，无需修改 hems_env.py 物理代码。
"""

from abc import ABC, abstractmethod
import numpy as np


class ObservationBuilder(ABC):
    """观测构造器抽象基类。"""

    @abstractmethod
    def get_obs_dim(self) -> int:
        """返回单个 agent 的观测维度。在构造时即可确定，无需 env 实例。"""
        ...

    @abstractmethod
    def build(self, env) -> np.ndarray:
        """
        构造当前时步的完整观测矩阵。

        Parameters
        ----------
        env : EnergyStorageEnv
            已初始化并 reset 的环境实例。

        Returns
        -------
        np.ndarray, shape (N, obs_dim), float32
            N 个 agent 的观测向量。
        """
        ...
