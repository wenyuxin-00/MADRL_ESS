"""MADDPG algorithm implementation.

This is a platform-level copy of the existing MADDPG logic. The goal of this
refactor is ownership transfer, not math changes, so the training/update
formula is intentionally kept unchanged while network construction is routed
through ``models.registry``.
"""

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base_agent import BaseAgent
import models.registry as _model_registry


class MADDPG(BaseAgent):
    def __init__(self, args, agent_id):
        self.device = args.device
        self.N = args.N
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip

        # Use the formal model registry so algorithm code no longer depends on
        # legacy network locations.
        self.actor = _model_registry.build_actor(args, agent_id).to(self.device)
        self.critic = _model_registry.build_critic(args).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, obs_batch, noise_std):
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a_batch = self.actor(obs_tensor).cpu().data.numpy()

        noise = np.random.normal(0, noise_std, size=a_batch.shape)
        a_batch = np.clip(a_batch + noise, -self.max_action, self.max_action)
        return a_batch

    def train(self, replay_buffer, agent_n):
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        batch_obs_n = [obs.to(self.device, non_blocking=True) for obs in batch_obs_n]
        batch_a_n = [a.to(self.device, non_blocking=True) for a in batch_a_n]
        batch_r_n = [r.to(self.device, non_blocking=True) for r in batch_r_n]
        batch_obs_next_n = [obs_next.to(self.device, non_blocking=True) for obs_next in batch_obs_next_n]
        batch_done_n = [done.to(self.device, non_blocking=True) for done in batch_done_n]

        with torch.no_grad():
            batch_a_next_n = [
                agent.actor_target(batch_obs_next)
                for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)
            ]
            Q_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next

        current_Q = self.critic(batch_obs_n, batch_a_n)
        critic_loss = F.mse_loss(target_Q, current_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        batch_a_n_new = batch_a_n[:]
        batch_a_n_new[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
        actor_loss = -self.critic(batch_obs_n, batch_a_n_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, model_dir, episode):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, model_dir, episode):
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        print(f"Agent {self.agent_id}: Models loaded successfully from episode {episode}.")
