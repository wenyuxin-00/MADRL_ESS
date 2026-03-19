"""MATD3 (Multi-Agent Twin Delayed DDPG) 实现。

在 MADDPG 基础上引入 twin critic 和延迟策略更新，提高训练稳定性。

主要类:
    MATD3 -- MATD3 算法智能体
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from controllers.madrl.base_agent import BaseAgent
from scripts.utils.replay_buffer import to_torch_batch
from models import build_actor_network, build_critic_network


class MATD3(BaseAgent):
    """MATD3（多智能体双延迟深度确定性策略梯度）算法智能体。

    在 MADDPG 基础上引入两项关键改进以提高训练稳定性：
    1. Twin Critic（双 Q 网络）：使用两个独立的 Critic 网络，
       取较小的 Q 值作为 TD 目标，缓解 Q 值过高估计问题。
    2. 延迟策略更新：Actor 每隔 ``policy_update_freq`` 步才更新一次，
       确保 Critic 充分收敛后再更新策略，减少策略振荡。
    3. 目标策略平滑：对目标 Actor 输出的动作添加裁剪噪声，
       平滑 Q 函数估计，防止 Critic 利用 Q 函数的尖峰。

    属性:
        cfg: 全局配置对象。
        device: 计算设备。
        num_agents: 智能体总数。
        agent_id: 本智能体索引。
        max_action: 动作空间最大绝对值。
        action_dim: 动作维度。
        gamma: 折扣因子。
        tau: 目标网络软更新系数。
        use_grad_clip: 是否启用梯度裁剪。
        grad_clip_norm: 梯度裁剪最大范数。
        policy_noise: 目标策略平滑噪声标准差。
        noise_clip: 目标策略噪声裁剪范围。
        policy_update_freq: Actor 更新频率（每多少步 Critic 更新后更新一次 Actor）。
        actor_pointer: 训练步计数器，用于控制延迟更新。
    """

    def __init__(self, cfg: object, agent_id: int) -> None:
        """初始化 MATD3 智能体。

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
        # MATD3 特有的超参数
        self.policy_noise = float(cfg.algo.policy_noise)      # 目标策略平滑噪声
        self.noise_clip = float(cfg.algo.noise_clip)           # 噪声裁剪范围
        self.policy_update_freq = int(cfg.algo.policy_update_freq)  # Actor 延迟更新频率
        self.actor_pointer = 0  # 训练步计数器，用于判断是否执行 Actor 更新

        # 构建 Actor 网络（每个智能体独立的策略网络）
        self.actor = build_actor_network(cfg, self.agent_id).to(self.device)
        # 构建 Twin Critic 网络（包含两个独立的 Q 网络）
        self.critic = build_critic_network(cfg).to(self.device)
        # 深拷贝创建目标网络
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Actor 和 Critic 使用独立的 Adam 优化器
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
            # 添加高斯探索噪声
            action = action + torch.randn_like(action) * float(noise_std)
        return action.clamp(-self.max_action, self.max_action)

    def train(self, replay_buffer: object, agent_n: list) -> None:
        """从经验回放缓冲区采样并执行一步训练更新。

        参数:
            replay_buffer: 共享经验回放缓冲区。
            agent_n: 所有智能体列表。
        """
        batch = to_torch_batch(replay_buffer.sample(), self.device)
        self.train_on_batch(batch, agent_n)

    def train_on_batch(self, batch: dict, agent_n: list) -> None:
        """基于一个预采样批次执行参数更新。

        MATD3 更新流程：
        1. 对每个智能体的目标 Actor 输出添加裁剪噪声（目标策略平滑）
        2. 用双 Critic 目标网络分别计算 Q 值，取较小值作为 TD 目标
        3. 更新双 Critic：最小化两个 Q 网络的 TD 误差之和
        4. 每隔 policy_update_freq 步才更新 Actor 和目标网络（延迟更新）

        参数:
            batch: 包含 obs, action, reward, next_obs, done 的字典。
            agent_n: 所有智能体列表。
        """
        # 递增训练步计数器（用于延迟策略更新判断）
        self.actor_pointer += 1
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # ---- Critic 更新（每步都执行）----
        with torch.no_grad():
            next_action_list = []
            for agent in agent_n:
                # 使用目标 Actor 计算下一步动作
                next_action = agent.actor_target(next_obs)
                # 目标策略平滑：添加裁剪的高斯噪声，防止 Critic 过拟合到目标策略的尖峰
                noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                    -self.noise_clip,
                    self.noise_clip,
                )
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                next_action_list.append(next_action)
            # 拼接所有智能体的下一步动作为联合动作
            next_action = torch.stack(next_action_list, dim=1)

            # Twin Critic：取两个 Q 值的较小值，缓解 Q 值过高估计
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            target_q = reward[:, self.agent_id] + self.gamma * (1 - done[:, self.agent_id]) * torch.min(
                q1_next,
                q2_next,
            )

        # 计算两个 Critic 的当前 Q 值估计
        current_q1, current_q2 = self.critic(obs, action)
        # 双 Critic 损失：两个 Q 网络的 MSE 之和
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # ---- Actor 延迟更新（每 policy_update_freq 步执行一次）----
        if self.actor_pointer % self.policy_update_freq == 0:
            # 克隆联合动作，仅替换本智能体的动作为当前策略输出
            new_action = action.clone()
            new_action[:, self.agent_id] = self.actor(obs)
            # 仅使用第一个 Q 网络（Q1）计算 Actor 损失，避免引入额外偏差
            actor_loss = -self.critic.Q1(obs, new_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            # 仅在 Actor 更新时才执行目标网络软更新
            self._soft_update()
