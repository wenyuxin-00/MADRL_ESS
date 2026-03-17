"""Shared interface for all MADRL agents."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all multi-agent RL agents."""

    @abstractmethod
    def choose_action(self, obs, noise_std: float):
        """Choose actions from one structured observation or a batched one."""
        ...

    @abstractmethod
    def act_from_torch_obs(self, obs_t, noise_std: float):
        """Choose actions from a shared torch observation batch."""
        ...

    @abstractmethod
    def train(self, replay_buffer, agent_n: list) -> None:
        """Sample a canonical transition batch and update parameters."""
        ...

    @abstractmethod
    def train_on_batch(self, batch, agent_n: list) -> None:
        """Update parameters from one shared torch batch."""
        ...

    @abstractmethod
    def save_model(self, model_dir: str, episode: int) -> None:
        """Save actor and critic parameters."""
        ...

    @abstractmethod
    def load_model(self, model_dir: str, episode: int) -> None:
        """Load actor and critic parameters."""
        ...
