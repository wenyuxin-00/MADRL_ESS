"""Shared interface for all MADRL agents."""

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all multi-agent RL agents."""

    @abstractmethod
    def choose_action(self, obs_batch: np.ndarray, noise_std: float) -> np.ndarray:
        """Choose actions from a batch of observations."""
        ...

    @abstractmethod
    def train(self, replay_buffer, agent_n: list) -> None:
        """Sample from replay buffer and update network parameters."""
        ...

    @abstractmethod
    def save_model(self, model_dir: str, episode: int) -> None:
        """Save actor and critic parameters."""
        ...

    @abstractmethod
    def load_model(self, model_dir: str, episode: int) -> None:
        """Load actor and critic parameters."""
        ...
