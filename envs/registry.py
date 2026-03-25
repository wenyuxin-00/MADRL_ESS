"""Compatibility helpers for the single GridEnv mainline."""

from __future__ import annotations

from envs.grid_env import GridEnv

DEFAULT_ENV_NAME = "grid"
SUPPORTED_ENV_NAMES = (DEFAULT_ENV_NAME,)


def get_env_cls(name: str | None = None) -> type[GridEnv]:
    normalized = DEFAULT_ENV_NAME if name is None else str(name).strip().lower()
    if normalized not in SUPPORTED_ENV_NAMES:
        raise ValueError(
            f"Unsupported environment '{name}'. Only the grid mainline is available."
        )
    return GridEnv
