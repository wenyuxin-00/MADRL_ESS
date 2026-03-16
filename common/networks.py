"""Compatibility layer for legacy network imports.

Formal actor/critic implementations were moved to ``models/`` in the third
refactor round. Keep re-exporting the old symbols here so older imports do not
break, but do not place new implementations in this module anymore.
"""

from models.actors.mlp_actor import MLPActor as Actor
from models.critics.maddpg_mlp_critic import MADDPGMLPCritic as Critic_MADDPG
from models.critics.matd3_mlp_critic import MATD3MLPCritic as Critic_MATD3
from models.registry import (
    ACTOR_REGISTRY,
    CRITIC_REGISTRY,
    build_actor,
    build_critic,
    register_actor,
    register_critic,
)
from models.utils import orthogonal_init

__all__ = [
    "ACTOR_REGISTRY",
    "Actor",
    "CRITIC_REGISTRY",
    "Critic_MADDPG",
    "Critic_MATD3",
    "build_actor",
    "build_critic",
    "orthogonal_init",
    "register_actor",
    "register_critic",
]
