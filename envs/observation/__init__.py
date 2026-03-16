"""
envs/observation/ — 观测构造模块
职责：将环境状态转换为 agent 观测向量，与 env 物理逻辑解耦。

主要导出：
  - ObservationBuilder  抽象接口
  - DefaultObservationBuilder  默认实现（由 args.obs_config 驱动）
  - build_obs_builder(args)  工厂函数
"""

from envs.observation.base import ObservationBuilder
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.registry import build_obs_builder

__all__ = ["ObservationBuilder", "DefaultObservationBuilder", "build_obs_builder"]
