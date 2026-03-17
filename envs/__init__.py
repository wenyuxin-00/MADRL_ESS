"""Reinforcement learning environments.
强化学习环境。

Current implementations / 已实现环境:
    - EnergyStorageEnv  -- multi-agent battery storage with price/load signals

How to add a new environment / 如何添加新环境:
    1. Create ``envs/your_env.py``, subclassing ``gym.Env``
    2. Implement ``reset()``, ``step()``, and provide observation via an ObservationBuilder
    3. Register in ``envs/registry.py``::

           register_env("your_env", YourEnv)

    4. Use in config: ``cfg.env.env_type = "your_env"``

See also:
    - ``envs/observation/``  -- structured observation building framework
"""

from envs.hems_env import EnergyStorageEnv
from envs.registry import ENV_REGISTRY, get_env_cls, register_env

__all__ = [
    "ENV_REGISTRY",
    "EnergyStorageEnv",
    "get_env_cls",
    "register_env",
]
