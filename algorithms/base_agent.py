"""Shared interface for all MADRL agents.
所有 MADRL 智能体的统一接口。

To implement a new agent, subclass ``BaseAgent`` and implement all abstract
methods. See ``algorithms/maddpg.py`` for a concrete example.
"""

from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from common.nested import add_batch_dim, to_torch_nested


class BaseAgent(ABC):
    """Abstract base class for all multi-agent RL agents.
    多智能体 RL 智能体的抽象基类。

    Each agent owns one actor network and one (or twin) critic network.
    The agent is responsible for action selection, parameter updates,
    and checkpoint save/load.

    Attributes set by subclasses:
        agent_id (int): Index of this agent within the multi-agent system.
        device (torch.device): Computation device (CPU or CUDA).
    """

    def _prepare_obs(self, obs: dict) -> tuple[dict, bool]:
        has_batch_dim = obs["local"].ndim == 3
        if not has_batch_dim:
            obs = add_batch_dim(obs)
        obs_t = to_torch_nested(obs, self.device)
        return obs_t, has_batch_dim

    def choose_action(self, obs: dict, noise_std: float) -> np.ndarray:
        """Select actions from structured observation(s).

        Args:
            obs: Structured observation dict. May be single-step
                ``{"local": (n_agents, local_dim), ...}`` or batched
                ``{"local": (batch, n_agents, local_dim), ...}``.
            noise_std: Standard deviation of exploration noise (0 = deterministic).

        Returns:
            np.ndarray: Action array of shape ``(action_dim,)`` for single obs,
            or ``(batch, action_dim)`` for batched obs. Values in ``[-1, 1]``.
        """
        obs_t, has_batch_dim = self._prepare_obs(obs)
        with torch.no_grad():
            action = self.act_from_torch_obs(obs_t, noise_std=noise_std).cpu().numpy()
        if not has_batch_dim:
            action = action[0]
        return action.astype(np.float32)

    def _soft_update(self) -> None:
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save_model(self, model_dir: str, episode: int) -> None:
        """Save actor and critic parameters to disk.

        Args:
            model_dir: Directory to save checkpoint files.
            episode: Episode number used as the checkpoint tag.
        """
        os.makedirs(model_dir, exist_ok=True)
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, model_dir: str, episode: int) -> None:
        """Load actor and critic parameters from disk.

        Args:
            model_dir: Directory containing checkpoint files.
            episode: Episode tag to load.
        """
        actor_path = os.path.join(model_dir, f"actor_agent_{self.agent_id}_ep_{episode}.pth")
        critic_path = os.path.join(model_dir, f"critic_agent_{self.agent_id}_ep_{episode}.pth")
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

    @abstractmethod
    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        """Select actions from a pre-converted torch observation batch.

        Args:
            obs_t: Torch observation dict, already on the correct device.
            noise_std: Exploration noise standard deviation.

        Returns:
            torch.Tensor: Actions of shape ``(batch, action_dim)``.
        """
        ...

    @abstractmethod
    def train(self, replay_buffer: Any, agent_n: list) -> None:
        """Sample a batch from the replay buffer and update parameters.

        Args:
            replay_buffer: The shared replay buffer to sample from.
            agent_n: List of all agents (for centralized critic training).
        """
        ...

    @abstractmethod
    def train_on_batch(self, batch: dict, agent_n: list) -> None:
        """Update parameters from one pre-sampled torch batch.

        Args:
            batch: Dict with keys ``"obs"``, ``"action"``, ``"reward"``,
                ``"next_obs"``, ``"done"`` — all torch tensors on device.
            agent_n: List of all agents.
        """
        ...
