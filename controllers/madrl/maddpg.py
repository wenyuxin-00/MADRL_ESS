"""MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 实现。

基于集中式训练、分布式执行（CTDE）范式的多智能体连续动作算法。

主要类:
    MADDPG -- MADDPG 算法智能体
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from controllers.madrl.base_agent import BaseAgent
from scripts.utils.replay_buffer import to_torch_batch
from models import build_actor_network, build_critic_network


class MADDPG(BaseAgent):
    """MADDPG（多智能体深度确定性策略梯度）算法智能体。

    基于集中式训练、分布式执行（CTDE）范式：
    - Critic（集中式）：使用所有智能体的全局观测和联合动作来评估 Q 值
    - Actor（分布式）：仅使用本智能体的局部观测来输出动作

    核心更新逻辑：
    1. Critic 更新：最小化 TD 目标与当前 Q 值的 MSE 损失
    2. Actor 更新：最大化 Critic 对当前策略输出动作的 Q 值评估
    3. 目标网络软更新：Polyak 平均缓慢跟踪在线网络

    属性:
        cfg: 全局配置对象。
        device: 计算设备（CPU / CUDA）。
        num_agents: 多智能体系统中的智能体总数。
        agent_id: 本智能体在系统中的索引。
        max_action: 动作空间的最大绝对值。
        action_dim: 单个智能体的动作维度。
        gamma: 折扣因子。
        tau: 目标网络软更新系数。
        use_grad_clip: 是否启用梯度裁剪。
        grad_clip_norm: 梯度裁剪的最大范数。
    """

    def __init__(self, cfg: object, agent_id: int) -> None:
        """初始化 MADDPG 智能体。

        参数:
            cfg: 全局配置对象，包含 env、model、algo、train、runtime 等子配置。
            agent_id: 本智能体在多智能体系统中的索引（从 0 开始）。
        """
        self.cfg = cfg
        self.device = cfg.runtime.device
        self.num_agents = int(cfg.env.num_agents)
        self.agent_id = int(agent_id)
        self.max_action = float(cfg.model.max_action)
        self.action_dim = int(cfg.runtime.action_dim)
        self.gamma = float(cfg.algo.gamma)
        self.tau = float(cfg.algo.tau)
        # 梯度裁剪配置：可通过配置文件控制是否启用及裁剪范数上限
        self.use_grad_clip = bool(cfg.model.use_grad_clip)
        self.grad_clip_norm = float(cfg.model.grad_clip_norm)

        # 构建 Actor 网络（每个智能体独立的策略网络）
        self.actor = build_actor_network(cfg, self.agent_id).to(self.device)
        # 构建 Critic 网络（集中式，接收全局信息）
        self.critic = build_critic_network(cfg).to(self.device)
        # 深拷贝创建目标网络，用于计算稳定的 TD 目标
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Actor 和 Critic 使用独立的 Adam 优化器，可设置不同的学习率
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.train.actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.train.critic_lr))

    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        """基于 PyTorch 观测批次选择动作（分布式执行）。

        参数:
            obs_t: 已在目标设备上的 torch 观测字典。
            noise_std: 探索噪声标准差，为 0 时输出确定性动作。

        返回:
            torch.Tensor: 裁剪到 [-max_action, max_action] 范围内的动作张量。
        """
        # Actor 前向推理：仅使用本智能体的局部观测
        action = self.actor(obs_t)
        if noise_std > 0.0:
            # 添加高斯探索噪声以促进策略探索
            action = action + torch.randn_like(action) * float(noise_std)
        # 将动作裁剪到合法范围
        return action.clamp(-self.max_action, self.max_action)

    def train(self, replay_buffer: object, agent_n: list) -> None:
        """从经验回放缓冲区采样并执行一步训练更新。

        参数:
            replay_buffer: 共享经验回放缓冲区。
            agent_n: 所有智能体列表（CTDE 中 Critic 需要全局信息）。
        """
        # 采样一批经验并转换为 torch 张量
        batch = to_torch_batch(replay_buffer.sample(), self.device)
        self.train_on_batch(batch, agent_n)

    def train_on_batch(self, batch: dict, agent_n: list) -> None:
        """基于一个预采样批次执行 Critic 和 Actor 的参数更新。

        MADDPG 更新流程：
        1. 用所有智能体的目标 Actor 计算下一步联合动作
        2. 用目标 Critic 计算 TD 目标值
        3. 更新 Critic：最小化 TD 误差
        4. 更新 Actor：最大化 Critic 的 Q 值评估
        5. 软更新目标网络

        参数:
            batch: 包含 obs, action, reward, next_obs, done 的字典。
            agent_n: 所有智能体列表。
        """
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # ---- Critic 更新 ----
        with torch.no_grad():
            # 使用所有智能体的目标 Actor 计算下一步联合动作（CTDE 的集中式部分）
            next_action = torch.stack([agent.actor_target(next_obs) for agent in agent_n], dim=1)
            # 计算 TD 目标：r + gamma * (1 - done) * Q_target(s', a')
            target_q = reward[:, self.agent_id] + self.gamma * (1 - done[:, self.agent_id]) * self.critic_target(
                next_obs,
                next_action,
            )

        # 计算当前 Q 值估计
        current_q = self.critic(obs, action)
        # Critic 损失：当前 Q 值与 TD 目标的均方误差
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            # 梯度裁剪：防止梯度爆炸导致训练不稳定
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # ---- Actor 更新 ----
        # 克隆联合动作，仅替换本智能体的动作为当前策略输出
        new_action = action.clone()
        new_action[:, self.agent_id] = self.actor(obs)
        # Actor 损失：最大化 Q 值（取负号转为最小化问题）
        actor_loss = -self.critic(obs, new_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        # ---- 目标网络软更新 ----
        self._soft_update()
