"""环境注册表。

根据环境类型名称（如 "energy_storage"、"grid_pf"）
返回对应的环境类。

主要函数:
    build_env -- 根据配置创建环境实例
"""

from __future__ import annotations

from envs.grid_env import GridEnv
from envs.hems_env import EnergyStorageEnv

# 全局环境类注册表：环境类型名称 → 环境类
ENV_REGISTRY: dict[str, type] = {
    "energy_storage": EnergyStorageEnv,
    "grid_pf":        GridEnv,
}


def register_env(name: str, env_cls: type) -> None:
    """注册一个环境类到全局注册表。

    参数:
        name: 环境类型名称（如 "energy_storage"）
        env_cls: 对应的环境类
    """
    ENV_REGISTRY[name] = env_cls


def get_env_cls(name: str) -> type:
    """按名称从注册表中读取环境类。

    参数:
        name: 环境类型名称

    返回:
        type: 对应的环境类

    异常:
        ValueError: 当名称不存在于注册表中时抛出
    """
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env.env_type '{name}', available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]
