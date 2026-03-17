"""模型组件注册表。"""

from __future__ import annotations

from models.family_adapters import (
    GraphActorAdapter,
    GraphCriticAdapter,
    MLPActorAdapter,
    MLPCriticAdapter,
    TransformerActorAdapter,
    TransformerCriticAdapter,
)
from models.encoders.graph_encoder import GraphEncoder
from models.encoders.mlp_encoder import MLPEncoder
from models.encoders.transformer_encoder import TransformerEncoder
from models.heads.actor_head import DeterministicContinuousActorHead
from models.heads.critic_head import SingleQHead, TwinQHead

ADAPTER_REGISTRY = {
    "mlp": {"actor": MLPActorAdapter, "critic": MLPCriticAdapter},
    "transformer": {"actor": TransformerActorAdapter, "critic": TransformerCriticAdapter},
    "graph": {"actor": GraphActorAdapter, "critic": GraphCriticAdapter},
}
ENCODER_REGISTRY = {
    "mlp": MLPEncoder,
    "transformer": TransformerEncoder,
    "graph": GraphEncoder,
}
ACTOR_HEAD_REGISTRY = {
    "deterministic_continuous": DeterministicContinuousActorHead,
}
CRITIC_HEAD_REGISTRY = {
    "single_q": SingleQHead,
    "twin_q": TwinQHead,
}


def get_adapter_cls(family: str, role: str):
    """按模型 family 读取 adapter。"""
    if family not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model.family '{family}', available: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[family][role]


def get_encoder_cls(family: str):
    """按模型 family 读取 encoder。"""
    if family not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown model.family '{family}', available: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[family]


def get_actor_head_cls(head_type: str):
    """Return the actor head class for the given name."""
    if head_type not in ACTOR_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown model.actor_head_type '{head_type}', available: {list(ACTOR_HEAD_REGISTRY)}"
        )
    return ACTOR_HEAD_REGISTRY[head_type]


def get_critic_head_cls(head_type: str):
    """Return the critic head class for the given name."""
    if head_type not in CRITIC_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown model.critic_head_type '{head_type}', available: {list(CRITIC_HEAD_REGISTRY)}"
        )
    return CRITIC_HEAD_REGISTRY[head_type]
