"""MADDPG implementation on top of canonical batches and assembled models."""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base_agent import BaseAgent
from common.nested import add_batch_dim, to_torch_nested
from common.replay_buffer import to_torch_batch
from models import build_actor_network, build_critic_network


class MADDPG(BaseAgent):
    """MADDPG agent with unchanged update math and new model assembly."""

    def __init__(self, cfg, agent_id: int):
        self.cfg = cfg
        self.device = cfg.runtime.device
        self.num_agents = int(cfg.env.num_agents)
        self.agent_id = int(agent_id)
        self.max_action = float(cfg.model.max_action)
        self.action_dim = int(cfg.runtime.action_dim)
        self.gamma = float(cfg.algo.gamma)
        self.tau = float(cfg.algo.tau)
        self.use_grad_clip = bool(cfg.model.use_grad_clip)

        self.actor = build_actor_network(cfg, self.agent_id).to(self.device)
        self.critic = build_critic_network(cfg).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.train.actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(cfg.train.critic_lr))

    def _prepare_obs(self, obs):
        has_batch_dim = obs["local"].ndim == 3
        if not has_batch_dim:
            obs = add_batch_dim(obs)
        obs_t = to_torch_nested(obs, self.device)
        return obs_t, has_batch_dim

    def choose_action(self, obs, noise_std: float):
        obs_t, has_batch_dim = self._prepare_obs(obs)
        with torch.no_grad():
            action = self.act_from_torch_obs(obs_t, noise_std=noise_std).cpu().numpy()

        if not has_batch_dim:
            action = action[0]

        return action.astype(np.float32)

    def act_from_torch_obs(self, obs_t, noise_std: float):
        action = self.actor(obs_t)
        if noise_std > 0.0:
            action = action + torch.randn_like(action) * float(noise_std)
        return action.clamp(-self.max_action, self.max_action)

    def train(self, replay_buffer, agent_n: list) -> None:
        batch = to_torch_batch(replay_buffer.sample(), self.device)
        self.train_on_batch(batch, agent_n)

    def train_on_batch(self, batch, agent_n: list) -> None:
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_action = torch.stack([agent.actor_target(next_obs) for agent in agent_n], dim=1)
            target_q = reward[:, self.agent_id] + self.gamma * (1 - done[:, self.agent_id]) * self.critic_target(
                next_obs,
                next_action,
            )

        current_q = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        new_action = action.clone()
        new_action[:, self.agent_id] = self.actor(obs)
        actor_loss = -self.critic(obs, new_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, model_dir: str, episode: int) -> None:
        """Save actor and critic state dicts."""
        os.makedirs(model_dir, exist_ok=True)
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, model_dir: str, episode: int) -> None:
        """Load actor and critic state dicts."""
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")

        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
