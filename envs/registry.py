"""环境注册表。

根据环境类型名称（如 "energy_storage"、"grid_pf"）
返回对应的环境类。

主要函数:
    build_env -- 根据配置创建环境实例
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
