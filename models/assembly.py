"""Actor / Critic network assembly.
Actor / Critic 网络组装入口。

Assembly pipeline / 组装流水线::

    structured obs dict
         |
         v
    Adapter (family_adapters.py)
         |  -- converts structured obs to model-specific input format:
         |     MLP: flat vector,  Transformer: token sequence,  Graph: node features
         v
    Encoder (encoders/)
         |  -- feature extraction backbone
         v
    Head (heads/)
         |  -- output projection: actions (actor) or Q-values (critic)
         v
    output

The ``family`` config field selects which adapter + encoder to use:
  - ``mlp``         -- fast, simple, good default
  - ``transformer`` -- self-attention over tokens (local + sequence)
  - ``graph``       -- message-passing GNN over agent graph
"""

from __future__ import annotations

from torch import nn

from models.registry import (
    get_actor_head_cls,
    get_adapter_cls,
    get_critic_head_cls,
    get_encoder_cls,
)


class ActorNetwork(nn.Module):
    """组装后的 actor：adapter -> encoder -> actor head。"""

    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder
        self.head = head

    def forward(self, obs):
        features = self.adapter(obs)
        embedding = self.encoder(features)
        return self.head(embedding)


class CriticNetwork(nn.Module):
    """组装后的 critic：adapter -> encoder -> critic head。"""

    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder
        self.head = head

    def _encode(self, obs, action):
        features = self.adapter(obs, action)
        return self.encoder(features)

    def forward(self, obs, action):
        return self.head(self._encode(obs, action))

    def Q1(self, obs, action):
        if not hasattr(self.head, "Q1"):
            raise AttributeError("The configured critic head does not expose Q1.")
        return self.head.Q1(self._encode(obs, action))


def validate_and_finalize_model_config(cfg):
    """校验模型 family，并补齐与算法相关的 critic head 默认值。"""
    algorithm = cfg.algo.name
    if algorithm not in {"MADDPG", "MATD3"}:
        raise ValueError(f"Unknown algo.name '{algorithm}', available: ['MADDPG', 'MATD3']")

    if cfg.model.family not in {"mlp", "transformer", "graph"}:
        raise ValueError("model.family must be one of ['mlp', 'transformer', 'graph']")

    if cfg.model.critic_head_type is None:
        cfg.model.critic_head_type = "single_q" if algorithm == "MADDPG" else "twin_q"

    expected_critic_head = "single_q" if algorithm == "MADDPG" else "twin_q"
    if cfg.model.critic_head_type != expected_critic_head:
        raise ValueError(
            f"algo.name='{algorithm}' requires model.critic_head_type='{expected_critic_head}', "
            f"got '{cfg.model.critic_head_type}'."
        )
    return cfg


def _build_encoder(cfg, input_dim: int):
    family = cfg.model.family
    encoder_cls = get_encoder_cls(family)
    common_kwargs = {
        "input_dim": int(input_dim),
        "hidden_dim": int(cfg.model.hidden_dim),
        "use_orthogonal_init": bool(cfg.model.use_orthogonal_init),
    }
    if family == "mlp":
        return encoder_cls(**common_kwargs)
    if family == "transformer":
        return encoder_cls(
            **common_kwargs,
            num_heads=int(cfg.model.transformer_num_heads),
            num_layers=int(cfg.model.transformer_num_layers),
        )
    if family == "graph":
        return encoder_cls(
            **common_kwargs,
            num_layers=int(cfg.model.graph_num_layers),
        )
    raise ValueError(f"Unsupported model.family '{family}'.")


def build_actor_network(cfg, agent_id: int) -> ActorNetwork:
    """Assemble one actor network: adapter -> encoder -> head.
    按当前 family 组装一个 actor：适配器 -> 编码器 -> 输出头。
    """
    validate_and_finalize_model_config(cfg)
    adapter_cls = get_adapter_cls(cfg.model.family, "actor")
    actor_head_cls = get_actor_head_cls(cfg.model.actor_head_type)

    adapter = adapter_cls(cfg, agent_id)
    encoder = _build_encoder(cfg, adapter.output_dim)
    head = actor_head_cls(
        hidden_dim=int(cfg.model.hidden_dim),
        action_dim=int(cfg.runtime.action_dim),
        max_action=float(cfg.model.max_action),
        use_orthogonal_init=bool(cfg.model.use_orthogonal_init),
    )
    return ActorNetwork(adapter=adapter, encoder=encoder, head=head)


def build_critic_network(cfg) -> CriticNetwork:
    """Assemble one critic network: adapter -> encoder -> head.
    按当前 family 组装一个 critic：适配器 -> 编码器 -> 输出头。
    """
    validate_and_finalize_model_config(cfg)
    adapter_cls = get_adapter_cls(cfg.model.family, "critic")
    critic_head_cls = get_critic_head_cls(cfg.model.critic_head_type)

    adapter = adapter_cls(cfg)
    encoder = _build_encoder(cfg, adapter.output_dim)
    head = critic_head_cls(
        hidden_dim=int(cfg.model.hidden_dim),
        use_orthogonal_init=bool(cfg.model.use_orthogonal_init),
    )
    return CriticNetwork(adapter=adapter, encoder=encoder, head=head)
