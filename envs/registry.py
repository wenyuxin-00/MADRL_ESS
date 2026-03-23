"""Environment registry narrowed to the GridEnv mainline."""

from __future__ import annotations

from envs.grid_env import GridEnv

ENV_REGISTRY: dict[str, type] = {
    "grid_pf": GridEnv,
}


def register_env(name: str, env_cls: type) -> None:
    ENV_REGISTRY[name] = env_cls


def get_env_cls(name: str) -> type:
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env.env_type '{name}', available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]
