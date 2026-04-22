from __future__ import annotations
from math import prod
import torch
import torch.nn.functional as F
from torch import nn
_TWIN_Q_ALGORITHMS = {"MATD3", "MATD3_SAFE_POC"}
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name: nn.init.constant_(param, 0)
        elif "weight" in name: nn.init.orthogonal_(param, gain=gain)
def _numel(shape) -> int:
    return int(prod(shape)) if shape else 0
def get_sequence_field_infos(cfg) -> list[dict]:
    layout = cfg.runtime.observation_layout or {}
    infos = []
    for name in cfg.obs.sequence_features:
        field_name = f"{name}_seq"
        if field_name not in layout: raise KeyError(f"observation_layout is missing '{field_name}'")
        infos.append({"field_name": field_name, **layout[field_name]})
    return infos
def _sequence_flat_dim(cfg, infos: list[dict], *, actor: bool) -> int:
    schema, n_agents, total = (cfg.runtime.observation_schema or {}, int(cfg.env.num_agents), 0)
    for info in infos:
        shape = schema[info["field_name"]]
        total += _numel(shape) if (not actor or info["scope"] == "shared") else _numel(shape[1:] if shape and shape[0] == n_agents else shape)
    return total
def _flatten_actor_sequences(obs: dict[str, torch.Tensor], infos: list[dict], agent_id: int) -> list[torch.Tensor]:
    return [value.reshape(value.shape[0], -1) if info["scope"] == "shared" else value[:, agent_id].reshape(value.shape[0], -1) for info in infos for value in [obs[info["field_name"]]]]
def _flatten_critic_sequences(obs: dict[str, torch.Tensor], infos: list[dict]) -> list[torch.Tensor]:
    return [obs[info["field_name"]].reshape(obs[info["field_name"]].shape[0], -1) for info in infos]
class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, use_orthogonal_init: bool):
        super().__init__()
        self.fc1, self.fc2 = (nn.Linear(int(input_dim), int(hidden_dim)), nn.Linear(int(hidden_dim), int(hidden_dim)))
        if use_orthogonal_init: orthogonal_init(self.fc1); orthogonal_init(self.fc2)
    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))
class MLPActorAdapter(nn.Module):
    def __init__(self, cfg, agent_id: int):
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.agent_id, self.local_dim, self.sequence_infos = (int(agent_id), int(schema["local"][1]), get_sequence_field_infos(cfg))
        self.output_dim = self.local_dim + _sequence_flat_dim(cfg, self.sequence_infos, actor=True)
    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs["local"][:, self.agent_id], *_flatten_actor_sequences(obs, self.sequence_infos, self.agent_id)], dim=-1)
class MLPCriticAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        schema, n_agents = (cfg.runtime.observation_schema or {}, int(cfg.env.num_agents))
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.output_dim = n_agents * int(schema["local"][1]) + _sequence_flat_dim(cfg, self.sequence_infos, actor=False) + n_agents * int(cfg.runtime.action_dim)
    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        batch_size = action.shape[0]
        return torch.cat([obs["local"].reshape(batch_size, -1), *_flatten_critic_sequences(obs, self.sequence_infos), action.reshape(batch_size, -1)], dim=-1)
class DeterministicContinuousActorHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, max_action: float, use_orthogonal_init: bool):
        super().__init__()
        self.max_action, self.fc = (float(max_action), nn.Linear(int(hidden_dim), int(action_dim)))
        if use_orthogonal_init: orthogonal_init(self.fc)
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.fc(embedding))
class SingleQHead(nn.Module):
    def __init__(self, hidden_dim: int, use_orthogonal_init: bool):
        super().__init__()
        self.fc = nn.Linear(int(hidden_dim), 1)
        if use_orthogonal_init: orthogonal_init(self.fc)
    def forward(self, embedding):
        return self.fc(embedding)
class TwinQHead(nn.Module):
    def __init__(self, hidden_dim: int, use_orthogonal_init: bool):
        super().__init__()
        self.q1, self.q2 = (nn.Linear(int(hidden_dim), 1), nn.Linear(int(hidden_dim), 1))
        if use_orthogonal_init: orthogonal_init(self.q1); orthogonal_init(self.q2)
    def forward(self, embedding):
        return self.q1(embedding), self.q2(embedding)
    def Q1(self, embedding):
        return self.q1(embedding)
class ActorNetwork(nn.Module):
    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter, self.encoder, self.head = (adapter, encoder, head)
    def forward(self, obs):
        return self.head(self.encoder(self.adapter(obs)))
class CriticNetwork(nn.Module):
    def __init__(self, adapter: nn.Module, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.adapter, self.encoder, self.head = (adapter, encoder, head)
    def _encode(self, obs, action):
        return self.encoder(self.adapter(obs, action))
    def forward(self, obs, action):
        return self.head(self._encode(obs, action))
    def Q1(self, obs, action):
        if not hasattr(self.head, "Q1"): raise AttributeError("The configured critic head does not expose Q1.")
        return self.head.Q1(self._encode(obs, action))
def validate_and_finalize_model_config(cfg):
    algorithm = cfg.algo.name
    if algorithm not in {"MADDPG", *_TWIN_Q_ALGORITHMS}: raise ValueError(f"Unknown algo.name '{algorithm}', available: ['MADDPG', 'MATD3', 'MATD3_SAFE_POC']")
    if cfg.model.family != "mlp": raise ValueError("model.family must be 'mlp' for the retained mainline.")
    expected = "single_q" if algorithm == "MADDPG" else "twin_q"
    if cfg.model.critic_head_type is None: cfg.model.critic_head_type = expected
    if cfg.model.critic_head_type != expected: raise ValueError(f"algo.name='{algorithm}' requires model.critic_head_type='{expected}', got '{cfg.model.critic_head_type}'.")
    return cfg
def _build_encoder(cfg, input_dim: int) -> MLPEncoder:
    return MLPEncoder(input_dim=int(input_dim), hidden_dim=int(cfg.model.hidden_dim), use_orthogonal_init=bool(cfg.model.use_orthogonal_init))
def build_actor_network(cfg, agent_id: int) -> ActorNetwork:
    validate_and_finalize_model_config(cfg)
    adapter = MLPActorAdapter(cfg, agent_id)
    return ActorNetwork(adapter=adapter, encoder=_build_encoder(cfg, adapter.output_dim), head=DeterministicContinuousActorHead(hidden_dim=int(cfg.model.hidden_dim), action_dim=int(cfg.runtime.action_dim), max_action=float(cfg.model.max_action), use_orthogonal_init=bool(cfg.model.use_orthogonal_init)))
def build_critic_network(cfg) -> CriticNetwork:
    validate_and_finalize_model_config(cfg)
    adapter = MLPCriticAdapter(cfg)
    head_cls = SingleQHead if cfg.model.critic_head_type == "single_q" else TwinQHead
    return CriticNetwork(adapter=adapter, encoder=_build_encoder(cfg, adapter.output_dim), head=head_cls(hidden_dim=int(cfg.model.hidden_dim), use_orthogonal_init=bool(cfg.model.use_orthogonal_init)))
