"""不同模型 family 的观测适配器。

职责很单纯：
  - MLP: 直接把结构化观测压平
  - Transformer: 整理成 token 序列
  - Graph: 整理成节点特征 + 邻接矩阵
"""

from __future__ import annotations

import torch
from torch import nn

from models.utils import (
    actor_sequence_flat_dim,
    critic_sequence_channels,
    critic_sequence_flat_dim,
    flatten_actor_sequence_tensors,
    flatten_critic_sequence_tensors,
    get_sequence_field_infos,
    graph_agent_sequence_tensor,
    graph_shared_sequence_tensor,
    sequence_channel_dim,
)


class MLPActorAdapter(nn.Module):
    """把单个 agent 的结构化观测压平成 MLP 输入向量。"""

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
    """把 joint observation 与 joint action 压平成 MLP critic 输入向量。"""

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


class TransformerActorAdapter(nn.Module):
    """把单个 agent 的结构化观测整理成 token 序列。"""

    def __init__(self, cfg, agent_id: int):
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.agent_id = int(agent_id)
        self.local_dim = int(schema["local"][1])
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.sequence_dim = sequence_channel_dim(self.sequence_infos)
        self.output_dim = self.local_dim + self.sequence_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        local = obs["local"][:, self.agent_id]
        batch_size = local.shape[0]

        local_token = local
        if self.sequence_dim > 0:
            zeros = torch.zeros(batch_size, self.sequence_dim, device=local.device, dtype=local.dtype)
            local_token = torch.cat([local, zeros], dim=-1)
        local_token = local_token.unsqueeze(1)

        sequence_channels = []
        for info in self.sequence_infos:
            value = obs[info["field_name"]]
            if info["scope"] == "shared":
                channel = value
            else:
                channel = value[:, self.agent_id]
            sequence_channels.append(channel if channel.dim() == 3 else channel.unsqueeze(-1))

        if not sequence_channels:
            return {"tokens": local_token}

        sequence_tokens = torch.cat(sequence_channels, dim=-1)
        prefix = torch.zeros(
            batch_size,
            sequence_tokens.shape[1],
            self.local_dim,
            device=local.device,
            dtype=local.dtype,
        )
        return {"tokens": torch.cat([local_token, torch.cat([prefix, sequence_tokens], dim=-1)], dim=1)}


class TransformerCriticAdapter(nn.Module):
    """把 joint observation 与 action 整理成 critic token 序列。"""

    def __init__(self, cfg):
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.local_dim = int(schema["local"][1])
        self.action_dim = int(cfg.runtime.action_dim)
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.sequence_dim = sequence_channel_dim(self.sequence_infos)
        self.output_dim = self.local_dim + self.action_dim + self.sequence_dim

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, n_agents = action.shape[:2]
        local = obs["local"]
        agent_tokens = torch.cat([local, action], dim=-1)

        if self.sequence_dim > 0:
            zeros = torch.zeros(batch_size, n_agents, self.sequence_dim, device=local.device, dtype=local.dtype)
            agent_tokens = torch.cat([agent_tokens, zeros], dim=-1)

        sequence_tokens = critic_sequence_channels(obs, self.sequence_infos)
        if sequence_tokens is None:
            return {"tokens": agent_tokens}

        prefix = torch.zeros(
            batch_size,
            sequence_tokens.shape[1],
            self.local_dim + self.action_dim,
            device=local.device,
            dtype=local.dtype,
        )
        return {"tokens": torch.cat([agent_tokens, torch.cat([prefix, sequence_tokens], dim=-1)], dim=1)}


class GraphActorAdapter(nn.Module):
    """为 actor 构造每个节点的图特征。"""

    def __init__(self, cfg, agent_id: int):
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.agent_id = int(agent_id)
        self.local_dim = int(schema["local"][1])
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.output_dim = self.local_dim + actor_sequence_flat_dim(cfg, self.sequence_infos)

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        local = obs["local"]
        node_features = torch.cat(
            [
                local,
                graph_agent_sequence_tensor(obs, self.sequence_infos, local),
                graph_shared_sequence_tensor(obs, self.sequence_infos, local),
            ],
            dim=-1,
        )
        return {
            "node_features": node_features,
            "adjacency": obs["adjacency"],
            "target_index": self.agent_id,
        }


class GraphCriticAdapter(nn.Module):
    """为 critic 构造 joint 图特征。"""

    def __init__(self, cfg):
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.local_dim = int(schema["local"][1])
        self.action_dim = int(cfg.runtime.action_dim)
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.output_dim = self.local_dim + self.action_dim + actor_sequence_flat_dim(cfg, self.sequence_infos)

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        local = obs["local"]
        node_features = torch.cat(
            [
                local,
                action,
                graph_agent_sequence_tensor(obs, self.sequence_infos, local),
                graph_shared_sequence_tensor(obs, self.sequence_infos, local),
            ],
            dim=-1,
        )
        return {
            "node_features": node_features,
            "adjacency": obs["adjacency"],
        }
