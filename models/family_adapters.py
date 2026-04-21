from __future__ import annotations
import torch
from torch import nn
from models.utils import (
    actor_sequence_flat_dim,
    critic_sequence_flat_dim,
    flatten_actor_sequence_tensors,
    flatten_critic_sequence_tensors,
    get_sequence_field_infos,
)
class MLPActorAdapter(nn.Module):

    def __init__(self, cfg, agent_id: int):
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.agent_id = int(agent_id)
        self.local_dim = int(schema["local"][1])
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.output_dim = self.local_dim + actor_sequence_flat_dim(cfg, self.sequence_infos)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        local = obs["local"][:, self.agent_id]
        parts = [local] + flatten_actor_sequence_tensors(obs, self.sequence_infos, self.agent_id)
        return torch.cat(parts, dim=-1)

class MLPCriticAdapter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        n_agents = int(cfg.env.num_agents)
        local_dim = int(schema["local"][1])
        action_dim = int(cfg.runtime.action_dim)
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.output_dim = n_agents * local_dim + critic_sequence_flat_dim(cfg, self.sequence_infos) + n_agents * action_dim

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        batch_size = action.shape[0]
        local = obs["local"].reshape(batch_size, -1)
        sequence_parts = flatten_critic_sequence_tensors(obs, self.sequence_infos)
        joint_action = action.reshape(batch_size, -1)
        return torch.cat([local, *sequence_parts, joint_action], dim=-1)
