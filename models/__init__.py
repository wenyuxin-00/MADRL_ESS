"""Formal model entrypoints.

The third refactor round moves actor/critic implementations out of
``common.networks`` into ``models`` so algorithms can scale to more network
families without changing training code.
"""

from models.registry import (
    ACTOR_REGISTRY,
    CRITIC_REGISTRY,
    build_actor,
    build_critic,
    register_actor,
    register_critic,
)

__all__ = [
    "ACTOR_REGISTRY",
    "CRITIC_REGISTRY",
    "build_actor",
    "build_critic",
    "register_actor",
    "register_critic",
]
