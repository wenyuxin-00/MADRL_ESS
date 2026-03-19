"""不同模型 family 的观测适配器。

本模块为每种模型家族（MLP / Transformer / Graph）提供 Actor 和 Critic
两种适配器，将环境输出的结构化观测字典转换为对应编码器所需的输入格式。

各适配器的职责：
  - MLP 适配器: 将结构化观测（局部观测 + 序列特征）压平为一维向量
  - Transformer 适配器: 将观测整理为 token 序列（字典格式 {"tokens": Tensor}）
  - Graph 适配器: 将观测整理为节点特征 + 邻接矩阵（字典格式 {"node_features": ..., "adjacency": ...}）

主要类：
    - MLPActorAdapter / MLPCriticAdapter: MLP 家族适配器
    - TransformerActorAdapter / TransformerCriticAdapter: Transformer 家族适配器
    - GraphActorAdapter / GraphCriticAdapter: Graph 家族适配器
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
    """MLP Actor 适配器：将单个智能体的结构化观测压平为一维向量。

    输出格式: [局部观测 | 序列特征(展平)] -> 一维向量

    属性:
        agent_id: 当前智能体编号
        local_dim: 局部观测的特征维度
        sequence_infos: 序列字段的元信息列表
        output_dim: 适配器输出的总维度（供编码器作为 input_dim）
    """

    def __init__(self, cfg, agent_id: int):
        """初始化 MLP Actor 适配器。

        参数:
            cfg: 全局配置对象
            agent_id: 当前智能体编号
        """
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.agent_id = int(agent_id)
        self.local_dim = int(schema["local"][1])       # 局部观测的特征维度
        self.sequence_infos = get_sequence_field_infos(cfg)
        # 总输出维度 = 局部观测维度 + 所有序列字段展平后的维度之和
        self.output_dim = self.local_dim + actor_sequence_flat_dim(cfg, self.sequence_infos)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播：提取当前智能体的观测并压平。

        参数:
            obs: 结构化观测字典，obs["local"] 形状为 (batch, n_agents, local_dim)

        返回:
            压平后的一维特征向量，形状为 (batch, output_dim)
        """
        local = obs["local"][:, self.agent_id]  # 取出当前智能体的局部观测
        # 将局部观测和展平的序列特征拼接
        parts = [local] + flatten_actor_sequence_tensors(obs, self.sequence_infos, self.agent_id)
        return torch.cat(parts, dim=-1)


class MLPCriticAdapter(nn.Module):
    """MLP Critic 适配器：将联合观测与联合动作压平为一维向量。

    Critic 使用中心化训练，因此输入包含所有智能体的观测和动作。
    输出格式: [所有局部观测(展平) | 序列特征(展平) | 所有动作(展平)]

    属性:
        sequence_infos: 序列字段的元信息列表
        output_dim: 适配器输出的总维度
    """

    def __init__(self, cfg):
        """初始化 MLP Critic 适配器。

        参数:
            cfg: 全局配置对象
        """
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        n_agents = int(cfg.env.num_agents)
        local_dim = int(schema["local"][1])
        action_dim = int(cfg.runtime.action_dim)
        self.sequence_infos = get_sequence_field_infos(cfg)
        # 总维度 = 所有智能体局部观测 + 序列特征 + 所有智能体动作
        self.output_dim = n_agents * local_dim + critic_sequence_flat_dim(cfg, self.sequence_infos) + n_agents * action_dim

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """前向传播：将联合观测和动作压平为一维向量。

        参数:
            obs: 结构化观测字典，obs["local"] 形状为 (batch, n_agents, local_dim)
            action: 联合动作张量，形状为 (batch, n_agents, action_dim)

        返回:
            压平后的一维特征向量，形状为 (batch, output_dim)
        """
        batch_size = action.shape[0]
        local = obs["local"].reshape(batch_size, -1)           # 所有智能体局部观测展平
        sequence_parts = flatten_critic_sequence_tensors(obs, self.sequence_infos)
        joint_action = action.reshape(batch_size, -1)          # 所有智能体动作展平
        return torch.cat([local, *sequence_parts, joint_action], dim=-1)


class TransformerActorAdapter(nn.Module):
    """Transformer Actor 适配器：将单个智能体的观测整理为 token 序列。

    Token 序列结构:
        - 第 1 个 token: [局部观测 | 零填充(序列维度)]（局部信息 token）
        - 后续 tokens: [零填充(局部维度) | 序列通道特征]（时间步 token）

    所有 token 的维度统一为 local_dim + sequence_dim，以便 Transformer 编码器处理。

    属性:
        agent_id: 当前智能体编号
        local_dim: 局部观测的特征维度
        sequence_infos: 序列字段的元信息列表
        sequence_dim: 序列通道的总维度
        output_dim: 每个 token 的维度（= local_dim + sequence_dim）
    """

    def __init__(self, cfg, agent_id: int):
        """初始化 Transformer Actor 适配器。

        参数:
            cfg: 全局配置对象
            agent_id: 当前智能体编号
        """
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.agent_id = int(agent_id)
        self.local_dim = int(schema["local"][1])
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.sequence_dim = sequence_channel_dim(self.sequence_infos)
        # 每个 token 的维度 = 局部观测维度 + 序列通道维度
        self.output_dim = self.local_dim + self.sequence_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """前向传播：构建 token 序列。

        参数:
            obs: 结构化观测字典

        返回:
            字典 {"tokens": Tensor}，tokens 形状为 (batch, num_tokens, token_dim)
        """
        local = obs["local"][:, self.agent_id]  # (batch, local_dim)
        batch_size = local.shape[0]

        # 构造局部信息 token：[局部观测 | 零填充] 使其维度与序列 token 对齐
        local_token = local
        if self.sequence_dim > 0:
            zeros = torch.zeros(batch_size, self.sequence_dim, device=local.device, dtype=local.dtype)
            local_token = torch.cat([local, zeros], dim=-1)
        local_token = local_token.unsqueeze(1)  # (batch, 1, token_dim)

        # 收集所有序列字段的通道特征
        sequence_channels = []
        for info in self.sequence_infos:
            value = obs[info["field_name"]]
            if info["scope"] == "shared":
                channel = value                    # 共享序列，所有智能体共用
            else:
                channel = value[:, self.agent_id]  # 按智能体索引提取
            sequence_channels.append(channel if channel.dim() == 3 else channel.unsqueeze(-1))

        if not sequence_channels:
            return {"tokens": local_token}

        # 拼接序列通道并添加零前缀使维度对齐
        sequence_tokens = torch.cat(sequence_channels, dim=-1)  # (batch, seq_len, seq_dim)
        prefix = torch.zeros(
            batch_size,
            sequence_tokens.shape[1],
            self.local_dim,
            device=local.device,
            dtype=local.dtype,
        )
        # 序列 token: [零填充(局部维度) | 序列通道]，然后与局部 token 拼接
        return {"tokens": torch.cat([local_token, torch.cat([prefix, sequence_tokens], dim=-1)], dim=1)}


class TransformerCriticAdapter(nn.Module):
    """Transformer Critic 适配器：将联合观测与动作整理为 token 序列。

    Token 序列结构:
        - 前 n_agents 个 token: [局部观测 | 动作 | 零填充(序列维度)]（智能体 token）
        - 后续 tokens: [零填充(局部+动作维度) | 序列通道特征]（时间步 token）

    属性:
        local_dim: 局部观测的特征维度
        action_dim: 单个智能体的动作维度
        sequence_infos: 序列字段的元信息列表
        sequence_dim: 序列通道的总维度
        output_dim: 每个 token 的维度（= local_dim + action_dim + sequence_dim）
    """

    def __init__(self, cfg):
        """初始化 Transformer Critic 适配器。

        参数:
            cfg: 全局配置对象
        """
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.local_dim = int(schema["local"][1])
        self.action_dim = int(cfg.runtime.action_dim)
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.sequence_dim = sequence_channel_dim(self.sequence_infos)
        # 每个 token 的维度 = 局部观测 + 动作 + 序列通道
        self.output_dim = self.local_dim + self.action_dim + self.sequence_dim

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播：构建 Critic 的 token 序列。

        参数:
            obs: 结构化观测字典，obs["local"] 形状为 (batch, n_agents, local_dim)
            action: 联合动作张量，形状为 (batch, n_agents, action_dim)

        返回:
            字典 {"tokens": Tensor}，tokens 形状为 (batch, num_tokens, token_dim)
        """
        batch_size, n_agents = action.shape[:2]
        local = obs["local"]
        # 智能体 token: 每个智能体的 [局部观测 | 动作]
        agent_tokens = torch.cat([local, action], dim=-1)

        # 如果有序列特征，在智能体 token 后面补零以对齐维度
        if self.sequence_dim > 0:
            zeros = torch.zeros(batch_size, n_agents, self.sequence_dim, device=local.device, dtype=local.dtype)
            agent_tokens = torch.cat([agent_tokens, zeros], dim=-1)

        # 提取序列通道特征作为时间步 token
        sequence_tokens = critic_sequence_channels(obs, self.sequence_infos)
        if sequence_tokens is None:
            return {"tokens": agent_tokens}

        # 序列 token 前面补零（局部观测+动作部分）以对齐维度
        prefix = torch.zeros(
            batch_size,
            sequence_tokens.shape[1],
            self.local_dim + self.action_dim,
            device=local.device,
            dtype=local.dtype,
        )
        # 拼接智能体 token 和序列 token
        return {"tokens": torch.cat([agent_tokens, torch.cat([prefix, sequence_tokens], dim=-1)], dim=1)}


class GraphActorAdapter(nn.Module):
    """Graph Actor 适配器：为 GNN 编码器构造节点特征和邻接矩阵。

    将所有智能体的观测组织为图结构：
        - 节点特征: [局部观测 | 每智能体序列特征 | 共享序列特征(广播)]
        - 邻接矩阵: 直接从观测中获取
        - target_index: 指定当前智能体的节点索引，编码后只取该节点的特征

    属性:
        agent_id: 当前智能体编号（用于 GNN 编码后提取目标节点）
        local_dim: 局部观测的特征维度
        sequence_infos: 序列字段的元信息列表
        output_dim: 每个节点的特征维度
    """

    def __init__(self, cfg, agent_id: int):
        """初始化 Graph Actor 适配器。

        参数:
            cfg: 全局配置对象
            agent_id: 当前智能体编号
        """
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.agent_id = int(agent_id)
        self.local_dim = int(schema["local"][1])
        self.sequence_infos = get_sequence_field_infos(cfg)
        self.output_dim = self.local_dim + actor_sequence_flat_dim(cfg, self.sequence_infos)

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """前向传播：构造图输入。

        参数:
            obs: 结构化观测字典，需包含 "local" 和 "adjacency"

        返回:
            字典，包含 node_features、adjacency 和 target_index
        """
        local = obs["local"]
        # 拼接局部观测、每智能体序列特征、共享序列特征（广播到所有节点）
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
            "target_index": self.agent_id,  # GNN 编码后只提取该节点的特征
        }


class GraphCriticAdapter(nn.Module):
    """Graph Critic 适配器：为 GNN 编码器构造联合图特征。

    与 GraphActorAdapter 类似，但额外将动作信息拼接到节点特征中。
    Critic 不需要 target_index，因为编码后对所有节点均值池化。

    属性:
        local_dim: 局部观测的特征维度
        action_dim: 单个智能体的动作维度
        sequence_infos: 序列字段的元信息列表
        output_dim: 每个节点的特征维度
    """

    def __init__(self, cfg):
        """初始化 Graph Critic 适配器。

        参数:
            cfg: 全局配置对象
        """
        super().__init__()
        schema = cfg.runtime.observation_schema or {}
        self.local_dim = int(schema["local"][1])
        self.action_dim = int(cfg.runtime.action_dim)
        self.sequence_infos = get_sequence_field_infos(cfg)
        # 节点特征维度 = 局部观测 + 动作 + 序列特征
        self.output_dim = self.local_dim + self.action_dim + actor_sequence_flat_dim(cfg, self.sequence_infos)

    def forward(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播：构造 Critic 的图输入。

        参数:
            obs: 结构化观测字典，需包含 "local" 和 "adjacency"
            action: 联合动作张量，形状为 (batch, n_agents, action_dim)

        返回:
            字典，包含 node_features 和 adjacency（无 target_index，均值池化）
        """
        local = obs["local"]
        # 拼接局部观测、动作、每智能体序列特征、共享序列特征
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
