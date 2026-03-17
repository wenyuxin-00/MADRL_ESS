"""模型层会复用的通用工具。"""

from __future__ import annotations

from math import prod

import torch
import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    """按项目现有习惯做正交初始化。"""
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)


def _numel(shape) -> int:
    return int(prod(shape)) if shape else 0


def get_sequence_field_infos(cfg) -> list[dict]:
    """从 runtime observation_layout 中读取序列字段信息。"""
    layout = cfg.runtime.observation_layout or {}
    infos = []
    for name in cfg.obs.sequence_features:
        field_name = f"{name}_seq"
        if field_name not in layout:
            raise KeyError(f"observation_layout 中缺少字段 '{field_name}'")
        infos.append({"field_name": field_name, **layout[field_name]})
    return infos


def actor_sequence_flat_dim(cfg, infos: list[dict]) -> int:
    """计算 actor 路径需要拼接的序列总维度。"""
    schema = cfg.runtime.observation_schema or {}
    n_agents = int(cfg.env.num_agents)
    total = 0
    for info in infos:
        shape = schema[info["field_name"]]
        if info["scope"] == "shared":
            total += _numel(shape)
        else:
            total += _numel(shape[1:] if shape and shape[0] == n_agents else shape)
    return total


def critic_sequence_flat_dim(cfg, infos: list[dict]) -> int:
    """计算 critic 路径需要拼接的序列总维度。"""
    schema = cfg.runtime.observation_schema or {}
    total = 0
    for info in infos:
        total += _numel(schema[info["field_name"]])
    return total


def sequence_channel_dim(infos: list[dict]) -> int:
    """计算 Transformer 每个时间步的总通道数。"""
    return int(sum(int(info["dim"]) for info in infos))


def flatten_actor_sequence_tensors(obs: dict[str, torch.Tensor], infos: list[dict], agent_id: int) -> list[torch.Tensor]:
    """把 actor 需要的 sequence 字段展开成扁平向量列表。"""
    parts = []
    for info in infos:
        value = obs[info["field_name"]]
        if info["scope"] == "shared":
            parts.append(value.reshape(value.shape[0], -1))
        else:
            parts.append(value[:, agent_id].reshape(value.shape[0], -1))
    return parts


def flatten_critic_sequence_tensors(obs: dict[str, torch.Tensor], infos: list[dict]) -> list[torch.Tensor]:
    """把 critic 需要的 joint sequence 字段展开成扁平向量列表。"""
    return [obs[info["field_name"]].reshape(obs[info["field_name"]].shape[0], -1) for info in infos]


def critic_sequence_channels(obs: dict[str, torch.Tensor], infos: list[dict]) -> torch.Tensor | None:
    """提取 critic Transformer 使用的逐时间步序列通道。"""
    channels = []
    for info in infos:
        value = obs[info["field_name"]]
        if info["scope"] == "per_agent":
            value = value.mean(dim=1)
        channels.append(value if value.dim() == 3 else value.unsqueeze(-1))
    if not channels:
        return None
    return torch.cat(channels, dim=-1)


def graph_shared_sequence_tensor(obs: dict[str, torch.Tensor], infos: list[dict], local: torch.Tensor) -> torch.Tensor:
    """提取 graph 路径中需要广播到所有节点的共享序列特征。"""
    shared_parts = []
    for info in infos:
        if info["scope"] != "shared":
            continue
        shared_parts.append(obs[info["field_name"]].reshape(local.shape[0], -1))
    if not shared_parts:
        return torch.zeros(local.shape[0], local.shape[1], 0, device=local.device, dtype=local.dtype)
    shared = torch.cat(shared_parts, dim=-1)
    return shared.unsqueeze(1).expand(-1, local.shape[1], -1)


def graph_agent_sequence_tensor(obs: dict[str, torch.Tensor], infos: list[dict], local: torch.Tensor) -> torch.Tensor:
    """提取 graph 路径中每个节点自己的序列特征。"""
    agent_parts = []
    for info in infos:
        if info["scope"] != "per_agent":
            continue
        value = obs[info["field_name"]]
        agent_parts.append(value.reshape(value.shape[0], value.shape[1], -1))
    if not agent_parts:
        return torch.zeros(local.shape[0], local.shape[1], 0, device=local.device, dtype=local.dtype)
    return torch.cat(agent_parts, dim=-1)
