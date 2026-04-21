from __future__ import annotations
from torch import nn
from models.encoders.mlp_encoder import MLPEncoder
from models.family_adapters import MLPActorAdapter, MLPCriticAdapter
from models.heads.actor_head import DeterministicContinuousActorHead
from models.heads.critic_head import SingleQHead, TwinQHead
_TWIN_Q_ALGORITHMS = {"MATD3", "MATD3_SAFE_POC"}
class ActorNetwork(nn.Module):

    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder
        self.head = head

    def forward(self, obs):
        return self.head(self.encoder(self.adapter(obs)))

class CriticNetwork(nn.Module):

    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder
        self.head = head

    def _encode(self, obs, action):
        return self.encoder(self.adapter(obs, action))

    def forward(self, obs, action):
        return self.head(self._encode(obs, action))

    def Q1(self, obs, action):
        if not hasattr(self.head, "Q1"):
            raise AttributeError("The configured critic head does not expose Q1.")
        return self.head.Q1(self._encode(obs, action))

def validate_and_finalize_model_config(cfg):
    algorithm = cfg.algo.name
    if algorithm not in {"MADDPG", *_TWIN_Q_ALGORITHMS}:
        raise ValueError(
            f"Unknown algo.name '{algorithm}', available: ['MADDPG', 'MATD3', 'MATD3_SAFE_POC']"
        )
    if cfg.model.family != "mlp":
        raise ValueError("model.family must be 'mlp' for the retained mainline.")
    if cfg.model.critic_head_type is None:
        cfg.model.critic_head_type = "single_q" if algorithm == "MADDPG" else "twin_q"
    expected_critic_head = "single_q" if algorithm == "MADDPG" else "twin_q"
    if cfg.model.critic_head_type != expected_critic_head:
        raise ValueError(
            f"algo.name='{algorithm}' requires model.critic_head_type='{expected_critic_head}', "
            f"got '{cfg.model.critic_head_type}'."
        )
    return cfg

def _build_encoder(cfg, input_dim: int) -> MLPEncoder:
    return MLPEncoder(
        input_dim=int(input_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        use_orthogonal_init=bool(cfg.model.use_orthogonal_init),
    )

def build_actor_network(cfg, agent_id: int) -> ActorNetwork:
    validate_and_finalize_model_config(cfg)
    adapter = MLPActorAdapter(cfg, agent_id)
    encoder = _build_encoder(cfg, adapter.output_dim)
    head = DeterministicContinuousActorHead(
        hidden_dim=int(cfg.model.hidden_dim),
        action_dim=int(cfg.runtime.action_dim),
        max_action=float(cfg.model.max_action),
        use_orthogonal_init=bool(cfg.model.use_orthogonal_init),
    )
    return ActorNetwork(adapter=adapter, encoder=encoder, head=head)

def build_critic_network(cfg) -> CriticNetwork:
    validate_and_finalize_model_config(cfg)
    adapter = MLPCriticAdapter(cfg)
    encoder = _build_encoder(cfg, adapter.output_dim)
    head_cls = SingleQHead if cfg.model.critic_head_type == "single_q" else TwinQHead
    head = head_cls(
        hidden_dim=int(cfg.model.hidden_dim),
        use_orthogonal_init=bool(cfg.model.use_orthogonal_init),
    )
    return CriticNetwork(adapter=adapter, encoder=encoder, head=head)
