"""强化学习环境模块。

封装 Gym 接口的多智能体储能调度环境，支持 HEMS（无电网约束）
和 GridEnv（含电网潮流约束）两种模式。

已实现环境:
    EnergyStorageEnv -- HEMS 多智能体电池储能环境
    GridEnv          -- 带电网潮流约束的储能环境

子模块:
    grid/        -- pandapower 电网模型（核心、拓扑、配置）
    observation/ -- 结构化观测空间构建框架
    rewards/     -- 可插拔奖励函数

如何添加新环境:
    1. 创建 ``envs/your_env.py``，继承 ``gym.Env``
    2. 实现 ``reset()``, ``step()``
    3. 在 ``envs/registry.py`` 中注册::

           register_env("your_env", YourEnv)

    4. 在配置中使用: ``cfg.env.env_type = "your_env"``
"""

from envs.hems_env import EnergyStorageEnv
from envs.registry import ENV_REGISTRY, get_env_cls, register_env

__all__ = [
    "ENV_REGISTRY",
    "EnergyStorageEnv",
    "get_env_cls",
    "register_env",
]
