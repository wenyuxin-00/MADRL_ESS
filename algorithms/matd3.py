"""MATD3 implementation on top of canonical batches and assembled models."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from algorithms.base_agent import BaseAgent
from common.replay_buffer import to_torch_batch
from models import build_actor_network, build_critic_network


class MATD3(BaseAgent):
    """MATD3 agent with unchanged update math and new model assembly."""

    def __init__(self, cfg: object, agent_id: int) -> None:
        self.cfg = cfg
        self.device = cfg.runtime.device
        self.num_agents = int(cfg.env.num_agents)
        self.agent_id = int(agent_id)
        self.max_action = float(cfg.model.max_action)
        self.action_dim = int(cfg.runtime.action_dim)
        self.gamma = float(cfg.algo.gamma)
        self.tau = float(cfg.algo.tau)
        self.use_grad_clip = bool(cfg.model.use_grad_clip)
        self.policy_noise = float(cfg.algo.policy_noise)
        self.noise_clip = float(cfg.algo.noise_clip)
        self.policy_update_freq = int(cfg.algo.policy_update_freq)
        self.actor_pointer = 0

        self.actor = build_actor_network(cfg, self.agent_id).to(self.device)
        self.critic = build_critic_network(cfg).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.train.actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.train.critic_lr))

    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        action = self.actor(obs_t)
        if noise_std > 0.0:
            action = action + torch.randn_like(action) * float(noise_std)
        return action.clamp(-self.max_action, self.max_action)

    def train(self, replay_buffer: object, agent_n: list) -> None:
        batch = to_torch_batch(replay_buffer.sample(), self.device)
        self.train_on_batch(batch, agent_n)

    def train_on_batch(self, batch: dict, agent_n: list) -> None:
        self.actor_pointer += 1
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_action_list = []
            for agent in agent_n:
                next_action = agent.actor_target(next_obs)
                noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                    -self.noise_clip,
                    self.noise_clip,
                )
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                next_action_list.append(next_action)
            next_action = torch.stack(next_action_list, dim=1)

            q1_next, q2_next = self.critic_target(next_obs, next_action)
            target_q = reward[:, self.agent_id] + self.gamma * (1 - done[:, self.agent_id]) * torch.min(
                q1_next,
                q2_next,
            )

        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        if self.actor_pointer % self.policy_update_freq == 0:
            new_action = action.clone()
            new_action[:, self.agent_id] = self.actor(obs)
            actor_loss = -self.critic.Q1(obs, new_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            self._soft_update()
