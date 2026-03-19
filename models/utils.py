"""模型层会复用的通用工具函数。

本模块提供适配器和编码器共用的辅助函数，包括：
    - orthogonal_init: 正交初始化
    - get_sequence_field_infos: 从配置中提取序列字段元信息
    - actor/critic 路径的序列维度计算和张量展开
    - graph 路径的节点特征提取

这些工具函数被 family_adapters.py 中的各适配器广泛调用，
用于处理结构化观测中的序列特征字段。
"""

from __future__ import annotations

from math import prod

import torch
import torch.nn as nn


def orthogonal_init(layer, gain=1.0):
    """对网络层执行正交初始化。

    遍历层的所有参数，对权重执行正交初始化，对偏置初始化为零。
    正交初始化有助于缓解深层网络的梯度消失/爆炸问题。

    参数:
        layer: 需要初始化的 nn.Module 层
        gain: 正交初始化的增益系数，默认 1.0
    """
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)        # 偏置初始化为零
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)  # 权重正交初始化


def _numel(shape) -> int:
    """计算给定形状的元素总数（内部辅助函数）。

    参数:
        shape: 张量形状元组或列表

    返回:
        元素总数；如果 shape 为空则返回 0
    """
    return int(prod(shape)) if shape else 0


def get_sequence_field_infos(cfg) -> list[dict]:
    """从运行时观测布局中提取序列字段的元信息。

    根据配置中声明的 sequence_features 列表，从 observation_layout
    中查找每个序列字段的 scope（共享/每智能体）、dim 等信息。

    参数:
        cfg: 全局配置对象，需包含 obs.sequence_features 和 runtime.observation_layout

    返回:
        序列字段信息的列表，每项为包含 field_name、scope、dim 等键的字典

    注意:
        如果 observation_layout 中缺少某个序列字段，会抛出 KeyError。
    """
    layout = cfg.runtime.observation_layout or {}
    infos = []
    for name in cfg.obs.sequence_features:
        field_name = f"{name}_seq"
        if field_name not in layout:
            raise KeyError(f"observation_layout 中缺少字段 '{field_name}'")
        infos.append({"field_name": field_name, **layout[field_name]})
    return infos


def actor_sequence_flat_dim(cfg, infos: list[dict]) -> int:
    """计算 Actor 路径中序列特征展平后的总维度。

    对于共享 (shared) 序列：使用完整形状的元素数。
    对于每智能体 (per_agent) 序列：去掉 agent 维度后计算元素数。

    参数:
        cfg: 全局配置对象
        infos: 由 get_sequence_field_infos 返回的序列字段信息列表

    返回:
        Actor 序列特征展平后的总维度
    """
    schema = cfg.runtime.observation_schema or {}
    n_agents = int(cfg.env.num_agents)
    total = 0
    for info in infos:
        shape = schema[info["field_name"]]
        if info["scope"] == "shared":
            total += _numel(shape)  # 共享序列使用完整形状
        else:
            # per_agent 序列去掉第一维（agent 维度）后计算
            total += _numel(shape[1:] if shape and shape[0] == n_agents else shape)
    return total


def critic_sequence_flat_dim(cfg, infos: list[dict]) -> int:
    """计算 Critic 路径中序列特征展平后的总维度。

    Critic 使用联合观测，因此直接使用完整形状（包含所有智能体）。

    参数:
        cfg: 全局配置对象
        infos: 由 get_sequence_field_infos 返回的序列字段信息列表

    返回:
        Critic 序列特征展平后的总维度
    """
    schema = cfg.runtime.observation_schema or {}
    total = 0
    for info in infos:
        total += _numel(schema[info["field_name"]])
    return total


def sequence_channel_dim(infos: list[dict]) -> int:
    """计算 Transformer 编码器每个时间步的总通道维度。

    将所有序列字段的通道维度求和，用于确定 token 中序列部分的宽度。

    参数:
        infos: 由 get_sequence_field_infos 返回的序列字段信息列表

    返回:
        所有序列字段通道维度之和
    """
    return int(sum(int(info["dim"]) for info in infos))


def flatten_actor_sequence_tensors(obs: dict[str, torch.Tensor], infos: list[dict], agent_id: int) -> list[torch.Tensor]:
    """将 Actor 需要的序列字段展平为一维张量列表。

    对于共享序列：直接展平整个张量。
    对于每智能体序列：先按 agent_id 索引，再展平。

    参数:
        obs: 结构化观测字典
        infos: 序列字段信息列表
        agent_id: 当前智能体编号

    返回:
        展平后的张量列表，每个元素形状为 (batch, flat_dim)
    """
    parts = []
    for info in infos:
        value = obs[info["field_name"]]
        if info["scope"] == "shared":
            parts.append(value.reshape(value.shape[0], -1))           # 共享序列直接展平
        else:
            parts.append(value[:, agent_id].reshape(value.shape[0], -1))  # 取当前智能体的部分
    return parts


def flatten_critic_sequence_tensors(obs: dict[str, torch.Tensor], infos: list[dict]) -> list[torch.Tensor]:
    """将 Critic 需要的联合序列字段展平为一维张量列表。

    Critic 使用全部智能体的信息，因此直接展平（不按 agent_id 索引）。

    参数:
        obs: 结构化观测字典
        infos: 序列字段信息列表

    返回:
        展平后的张量列表，每个元素形状为 (batch, flat_dim)
    """
    return [obs[info["field_name"]].reshape(obs[info["field_name"]].shape[0], -1) for info in infos]


def critic_sequence_channels(obs: dict[str, torch.Tensor], infos: list[dict]) -> torch.Tensor | None:
    """提取 Critic Transformer 使用的逐时间步序列通道。

    对于 per_agent 类型的序列，先在智能体维度上取均值聚合，
    然后将所有序列通道在最后一维拼接。

    参数:
        obs: 结构化观测字典
        infos: 序列字段信息列表

    返回:
        拼接后的序列通道张量，形状为 (batch, seq_len, total_channel_dim)；
        如果没有序列字段则返回 None
    """
    channels = []
    for info in infos:
        value = obs[info["field_name"]]
        if info["scope"] == "per_agent":
            value = value.mean(dim=1)  # 在智能体维度上均值聚合
        channels.append(value if value.dim() == 3 else value.unsqueeze(-1))
    if not channels:
        return None
    return torch.cat(channels, dim=-1)  # 在通道维度拼接


def graph_shared_sequence_tensor(obs: dict[str, torch.Tensor], infos: list[dict], local: torch.Tensor) -> torch.Tensor:
    """提取 Graph 路径中的共享序列特征，并广播到所有节点。

    共享序列特征对所有智能体相同，需要通过 unsqueeze + expand
    广播到 (batch, n_agents, shared_dim) 的形状。

    参数:
        obs: 结构化观测字典
        infos: 序列字段信息列表
        local: 局部观测张量，用于获取 batch_size 和 n_agents 维度

    返回:
        广播后的共享序列特征，形状为 (batch, n_agents, shared_flat_dim)；
        如果没有共享序列字段则返回零张量
    """
    shared_parts = []
    for info in infos:
        if info["scope"] != "shared":
            continue
        # 展平共享序列: (batch, *shape) -> (batch, flat_dim)
        shared_parts.append(obs[info["field_name"]].reshape(local.shape[0], -1))
    if not shared_parts:
        return torch.zeros(local.shape[0], local.shape[1], 0, device=local.device, dtype=local.dtype)
    shared = torch.cat(shared_parts, dim=-1)
    # 广播到所有节点: (batch, flat_dim) -> (batch, 1, flat_dim) -> (batch, n_agents, flat_dim)
    return shared.unsqueeze(1).expand(-1, local.shape[1], -1)


def graph_agent_sequence_tensor(obs: dict[str, torch.Tensor], infos: list[dict], local: torch.Tensor) -> torch.Tensor:
    """提取 Graph 路径中每个节点自己的序列特征。

    per_agent 类型的序列特征每个智能体独立拥有，
    按 (batch, n_agents, -1) 形状展平后拼接。

    参数:
        obs: 结构化观测字典
        infos: 序列字段信息列表
        local: 局部观测张量，用于获取维度信息

    返回:
        每智能体序列特征，形状为 (batch, n_agents, agent_seq_flat_dim)；
        如果没有 per_agent 序列字段则返回零张量
    """
    agent_parts = []
    for info in infos:
        if info["scope"] != "per_agent":
            continue
        value = obs[info["field_name"]]
        # 保留 batch 和 agent 维度，展平其余: (batch, n_agents, *rest) -> (batch, n_agents, flat)
        agent_parts.append(value.reshape(value.shape[0], value.shape[1], -1))
    if not agent_parts:
        return torch.zeros(local.shape[0], local.shape[1], 0, device=local.device, dtype=local.dtype)
    return torch.cat(agent_parts, dim=-1)
