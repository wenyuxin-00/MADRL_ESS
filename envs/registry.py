"""Environment registry.
环境注册表。

How to add a new environment / 如何添加新环境:
    1. Create ``envs/your_env.py`` subclassing ``gym.Env``
       - Implement ``reset() -> obs_dict`` and ``step(actions) -> (obs, rewards, done, info)``
       - Accept dataset, reward_fn, forecaster, obs_builder via constructor
    2. Register here::

           register_env("your_env", YourEnv)

    3. Use in config: ``cfg.env.env_type = "your_env"``
    4. If needed, update ``core/builder.py:build_env()`` for custom construction logic
"""

from __future__ import annotations

from envs.grid_env import GridEnv
from envs.hems_env import EnergyStorageEnv

ENV_REGISTRY: dict[str, type] = {
    "energy_storage": EnergyStorageEnv,
    "grid_pf":        GridEnv,
}


def register_env(name: str, env_cls: type) -> None:
    """注册一个环境类。"""
    ENV_REGISTRY[name] = env_cls


def get_env_cls(name: str) -> type:
    """按名称读取环境类。"""
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env.env_type '{name}', available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]
