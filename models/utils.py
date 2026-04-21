from __future__ import annotations
from math import prod
import torch
import torch.nn as nn
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)

def _numel(shape) -> int:
    return int(prod(shape)) if shape else 0

def get_sequence_field_infos(cfg) -> list[dict]:
    layout = cfg.runtime.observation_layout or {}
    infos = []
    for name in cfg.obs.sequence_features:
        field_name = f"{name}_seq"
        if field_name not in layout:
            raise KeyError(f"observation_layout 中缺少字段 '{field_name}'")
        infos.append({"field_name": field_name, **layout[field_name]})
    return infos

def actor_sequence_flat_dim(cfg, infos: list[dict]) -> int:
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
    schema = cfg.runtime.observation_schema or {}
    total = 0
    for info in infos:
        total += _numel(schema[info["field_name"]])
    return total

def flatten_actor_sequence_tensors(obs: dict[str, torch.Tensor], infos: list[dict], agent_id: int) -> list[torch.Tensor]:
    parts = []
    for info in infos:
        value = obs[info["field_name"]]
        if info["scope"] == "shared":
            parts.append(value.reshape(value.shape[0], -1))
        else:
            parts.append(value[:, agent_id].reshape(value.shape[0], -1))
    return parts

def flatten_critic_sequence_tensors(obs: dict[str, torch.Tensor], infos: list[dict]) -> list[torch.Tensor]:
    return [obs[info["field_name"]].reshape(obs[info["field_name"]].shape[0], -1) for info in infos]
