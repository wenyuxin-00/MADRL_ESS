"""模型组件注册表 -- 适配器、编码器、输出头。

本模块维护四个全局注册表，通过字符串键查找对应的组件类：
    - ADAPTER_REGISTRY:  适配器注册表（按 family + role 索引）
    - ENCODER_REGISTRY:  编码器注册表（按 family 索引）
    - ACTOR_HEAD_REGISTRY:  Actor 输出头注册表
    - CRITIC_HEAD_REGISTRY: Critic 输出头注册表

架构: Adapter -> Encoder -> Head
    - Adapter: 将结构化观测字典转换为模型特定的输入格式
    - Encoder: 特征提取骨干网络 (MLP / Transformer / Graph)
    - Head: 输出投影（Actor 输出动作，Critic 输出 Q 值）

如何添加新模型家族:
    1. 在 ``models/encoders/your_encoder.py`` 中创建编码器
    2. 在 ``models/family_adapters.py`` 中创建适配器（actor + critic）
    3. 调用注册函数注册三个组件::

           register_encoder("your_family", YourEncoder)
           register_adapter("your_family", "actor", YourActorAdapter)
           register_adapter("your_family", "critic", YourCriticAdapter)

    4. 在配置中使用: ``cfg.model.family = "your_family"``
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

# 适配器注册表：按 family -> role(actor/critic) 二级索引
ADAPTER_REGISTRY: dict[str, dict[str, type[nn.Module]]] = {
    "mlp": {"actor": MLPActorAdapter, "critic": MLPCriticAdapter},
    "transformer": {"actor": TransformerActorAdapter, "critic": TransformerCriticAdapter},
    "graph": {"actor": GraphActorAdapter, "critic": GraphCriticAdapter},
}
# 编码器注册表：按 family 一级索引
ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLPEncoder,
    "transformer": TransformerEncoder,
    "graph": GraphEncoder,
}
# Actor 输出头注册表：按 head_type 索引
ACTOR_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "deterministic_continuous": DeterministicContinuousActorHead,
}
# Critic 输出头注册表：按 head_type 索引（single_q 对应 MADDPG，twin_q 对应 MATD3）
CRITIC_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "single_q": SingleQHead,
    "twin_q": TwinQHead,
}


def register_adapter(family: str, role: str, adapter_cls: type) -> None:
    """注册一个适配器类到指定的模型家族和角色。

    参数:
        family: 模型家族名称，如 "mlp"、"transformer"、"graph"
        role: 角色名称，"actor" 或 "critic"
        adapter_cls: 适配器类（nn.Module 子类）
    """
    if family not in ADAPTER_REGISTRY:
        ADAPTER_REGISTRY[family] = {}
    ADAPTER_REGISTRY[family][role] = adapter_cls


def register_encoder(family: str, encoder_cls: type) -> None:
    """注册一个编码器类到指定的模型家族。

    参数:
        family: 模型家族名称
        encoder_cls: 编码器类（nn.Module 子类）
    """
    ENCODER_REGISTRY[family] = encoder_cls


def register_actor_head(head_type: str, head_cls: type) -> None:
    """注册一个 Actor 输出头类。

    参数:
        head_type: 输出头类型名称，如 "deterministic_continuous"
        head_cls: 输出头类（nn.Module 子类）
    """
    ACTOR_HEAD_REGISTRY[head_type] = head_cls


def register_critic_head(head_type: str, head_cls: type) -> None:
    """注册一个 Critic 输出头类。

    参数:
        head_type: 输出头类型名称，如 "single_q"、"twin_q"
        head_cls: 输出头类（nn.Module 子类）
    """
    CRITIC_HEAD_REGISTRY[head_type] = head_cls


def get_adapter_cls(family: str, role: str) -> type[nn.Module]:
    """根据模型家族和角色查找适配器类。

    参数:
        family: 模型家族名称
        role: 角色名称，"actor" 或 "critic"

    返回:
        对应的适配器类

    注意:
        如果 family 未注册，抛出 ValueError 并提示可用选项。
    """
    if family not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model.family '{family}', available: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[family][role]


def get_encoder_cls(family: str) -> type[nn.Module]:
    """根据模型家族查找编码器类。

    参数:
        family: 模型家族名称

    返回:
        对应的编码器类

    注意:
        如果 family 未注册，抛出 ValueError 并提示可用选项。
    """
    if family not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown model.family '{family}', available: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[family]


def get_actor_head_cls(head_type: str) -> type[nn.Module]:
    """根据类型名称查找 Actor 输出头类。

    参数:
        head_type: 输出头类型名称

    返回:
        对应的 Actor 输出头类

    注意:
        如果 head_type 未注册，抛出 ValueError 并提示可用选项。
    """
    if head_type not in ACTOR_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown model.actor_head_type '{head_type}', available: {list(ACTOR_HEAD_REGISTRY)}"
        )
    return ACTOR_HEAD_REGISTRY[head_type]


def get_critic_head_cls(head_type: str) -> type[nn.Module]:
    """根据类型名称查找 Critic 输出头类。

    参数:
        head_type: 输出头类型名称

    返回:
        对应的 Critic 输出头类

    注意:
        如果 head_type 未注册，抛出 ValueError 并提示可用选项。
    """
    if head_type not in CRITIC_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown model.critic_head_type '{head_type}', available: {list(CRITIC_HEAD_REGISTRY)}"
        )
    return CRITIC_HEAD_REGISTRY[head_type]
