"""Model component registry -- adapters, encoders, and heads.
模型组件注册表 -- 适配器、编码器、输出头。

Architecture: Adapter -> Encoder -> Head
    - Adapter: converts structured obs dict to model-specific input format
    - Encoder: feature extraction backbone (MLP / Transformer / Graph)
    - Head: output projection (actions for actor, Q-values for critic)

How to add a new model family / 如何添加新模型家族:
    1. Create encoder in ``models/encoders/your_encoder.py``
    2. Create adapters in ``models/family_adapters.py`` (actor + critic)
    3. Register all three::

           register_encoder("your_family", YourEncoder)
           register_adapter("your_family", "actor", YourActorAdapter)
           register_adapter("your_family", "critic", YourCriticAdapter)

    4. Use in config: ``cfg.model.family = "your_family"``
"""

from __future__ import annotations

from torch import nn

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

ADAPTER_REGISTRY: dict[str, dict[str, type[nn.Module]]] = {
    "mlp": {"actor": MLPActorAdapter, "critic": MLPCriticAdapter},
    "transformer": {"actor": TransformerActorAdapter, "critic": TransformerCriticAdapter},
    "graph": {"actor": GraphActorAdapter, "critic": GraphCriticAdapter},
}
ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLPEncoder,
    "transformer": TransformerEncoder,
    "graph": GraphEncoder,
}
ACTOR_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "deterministic_continuous": DeterministicContinuousActorHead,
}
CRITIC_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "single_q": SingleQHead,
    "twin_q": TwinQHead,
}


def register_adapter(family: str, role: str, adapter_cls: type) -> None:
    """Register an adapter class for a model family and role (actor/critic)."""
    if family not in ADAPTER_REGISTRY:
        ADAPTER_REGISTRY[family] = {}
    ADAPTER_REGISTRY[family][role] = adapter_cls


def register_encoder(family: str, encoder_cls: type) -> None:
    """Register an encoder class for a model family."""
    ENCODER_REGISTRY[family] = encoder_cls


def register_actor_head(head_type: str, head_cls: type) -> None:
    """Register an actor head class."""
    ACTOR_HEAD_REGISTRY[head_type] = head_cls


def register_critic_head(head_type: str, head_cls: type) -> None:
    """Register a critic head class."""
    CRITIC_HEAD_REGISTRY[head_type] = head_cls


def get_adapter_cls(family: str, role: str) -> type[nn.Module]:
    """Get adapter class by model family and role (actor/critic).
    按模型 family 读取 adapter。"""
    if family not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model.family '{family}', available: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[family][role]


def get_encoder_cls(family: str) -> type[nn.Module]:
    """Get encoder class by model family.
    按模型 family 读取 encoder。"""
    if family not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown model.family '{family}', available: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[family]


def get_actor_head_cls(head_type: str) -> type[nn.Module]:
    """Return the actor head class for the given name."""
    if head_type not in ACTOR_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown model.actor_head_type '{head_type}', available: {list(ACTOR_HEAD_REGISTRY)}"
        )
    return ACTOR_HEAD_REGISTRY[head_type]


def get_critic_head_cls(head_type: str) -> type[nn.Module]:
    """Return the critic head class for the given name."""
    if head_type not in CRITIC_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown model.critic_head_type '{head_type}', available: {list(CRITIC_HEAD_REGISTRY)}"
        )
    return CRITIC_HEAD_REGISTRY[head_type]
