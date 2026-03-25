"""Abstract dataset interface shared by episode-based loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEpisodeDataset(ABC):
    """Common interface for datasets that expose episode slices."""

    @abstractmethod
    def num_episodes(self) -> int:
        """Return the number of available episodes."""

    @abstractmethod
    def get_episode(self, episode_idx: int) -> dict:
        """Return the signals and metadata for one episode."""
