"""环境 registry。

当前只注册一个主环境，但保留最小扩展点，方便后续加入新的 RL 环境。
"""

from __future__ import annotations

from envs.hems_env import EnergyStorageEnv

ENV_REGISTRY: dict[str, type] = {
    "energy_storage": EnergyStorageEnv,
}


def register_env(name: str, env_cls: type) -> None:
    """注册一个环境类。"""
    ENV_REGISTRY[name] = env_cls


def get_env_cls(name: str) -> type:
    """按名称读取环境类。"""
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env.env_type '{name}', available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]
