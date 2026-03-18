"""Canonical replay buffer built around transition batches."""

from __future__ import annotations

import numpy as np

import torch

from common.nested import NestedArray, index_nested, stack_nested, to_torch_nested


class ReplayBuffer:
    """Simple ring-buffer for canonical transition dictionaries."""

    def __init__(self, cfg: object) -> None:
        self.buffer_size = int(cfg.train.buffer_size)
        self.batch_size = int(cfg.train.batch_size)
        self.storage: list[dict] = []
        self.position = 0
        self.current_size = 0

    def _store_transition(self, transition: dict) -> None:
        if self.current_size < self.buffer_size:
            self.storage.append(transition)
            self.current_size += 1
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size

    def store_transitions_batched(
        self,
        obs: NestedArray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: NestedArray,
        done: np.ndarray,
    ) -> None:
        """Store one batched env rollout step."""
        num_envs = int(np.asarray(action).shape[0])
        for env_idx in range(num_envs):
            transition = {
                "obs": index_nested(obs, env_idx),
                "action": np.asarray(action[env_idx], dtype=np.float32).copy(),
                "reward": np.asarray(reward[env_idx], dtype=np.float32).copy(),
                "next_obs": index_nested(next_obs, env_idx),
                "done": np.asarray(done[env_idx], dtype=np.float32).copy(),
            }
            self._store_transition(transition)

    def sample(self) -> dict[str, NestedArray]:
        """Sample a canonical transition batch."""
        indices = np.random.choice(
            self.current_size,
            size=self.batch_size,
            replace=self.current_size < self.batch_size,
        )
        transitions = [self.storage[idx] for idx in indices]
        return {
            "obs": stack_nested([transition["obs"] for transition in transitions]),
            "action": np.stack([transition["action"] for transition in transitions], axis=0),
            "reward": np.stack([transition["reward"] for transition in transitions], axis=0),
            "next_obs": stack_nested([transition["next_obs"] for transition in transitions]),
            "done": np.stack([transition["done"] for transition in transitions], axis=0),
        }


def to_torch_batch(batch: dict[str, NestedArray], device: torch.device | str) -> dict[str, NestedArray]:
    """Convert a sampled canonical batch to torch tensors."""
    return {
        "obs": to_torch_nested(batch["obs"], device),
        "action": to_torch_nested(batch["action"], device),
        "reward": to_torch_nested(batch["reward"], device),
        "next_obs": to_torch_nested(batch["next_obs"], device),
        "done": to_torch_nested(batch["done"], device),
    }
