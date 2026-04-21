from __future__ import annotations
from abc import ABC, abstractmethod
class BaseEpisodeDataset(ABC):

    @abstractmethod
    def num_episodes(self) -> int:
        pass

    @abstractmethod
    def get_episode(self, episode_idx: int) -> dict:
        pass
