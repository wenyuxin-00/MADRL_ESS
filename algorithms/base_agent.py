"""Shared interface for all MADRL agents.
所有 MADRL 智能体的统一接口。

To implement a new agent, subclass ``BaseAgent`` and implement all abstract
methods. See ``algorithms/maddpg.py`` for a concrete example.
"""

from abc import ABC, abstractmethod


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

    @abstractmethod
    def choose_action(self, obs, noise_std: float):
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
        ...

    @abstractmethod
    def act_from_torch_obs(self, obs_t, noise_std: float):
        """Select actions from a pre-converted torch observation batch.

        Args:
            obs_t: Torch observation dict, already on the correct device.
            noise_std: Exploration noise standard deviation.

        Returns:
            np.ndarray: Actions of shape ``(batch, action_dim)``.
        """
        ...

    @abstractmethod
    def train(self, replay_buffer, agent_n: list) -> None:
        """Sample a batch from the replay buffer and update parameters.

        Args:
            replay_buffer: The shared replay buffer to sample from.
            agent_n: List of all agents (for centralized critic training).
        """
        ...

    @abstractmethod
    def train_on_batch(self, batch, agent_n: list) -> None:
        """Update parameters from one pre-sampled torch batch.

        Args:
            batch: Dict with keys ``"obs"``, ``"action"``, ``"reward"``,
                ``"next_obs"``, ``"done"`` — all torch tensors on device.
            agent_n: List of all agents.
        """
        ...

    @abstractmethod
    def save_model(self, model_dir: str, episode: int) -> None:
        """Save actor and critic parameters to disk.

        Args:
            model_dir: Directory to save checkpoint files.
            episode: Episode number used as the checkpoint tag.
        """
        ...

    @abstractmethod
    def load_model(self, model_dir: str, episode: int) -> None:
        """Load actor and critic parameters from disk.

        Args:
            model_dir: Directory containing checkpoint files.
            episode: Episode tag to load.
        """
        ...
