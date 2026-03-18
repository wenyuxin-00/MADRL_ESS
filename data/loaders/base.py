"""数据集抽象基类。

定义所有数据集加载器的统一 API，子类需实现 __len__ 和 __getitem__。

主要类:
    BaseDataset -- 数据集抽象基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEpisodeDataset(ABC):
    """Abstract interface for multi-episode signal datasets.

    See ``datasets/csv_price_load.py`` for a concrete implementation.
    """

    @abstractmethod
    def num_episodes(self) -> int:
        """Return the number of available episodes in the dataset."""

    @abstractmethod
    def get_episode(self, episode_idx: int) -> dict:
        """Return one episode's signal data.

        Args:
            episode_idx: Zero-based episode index.

        Returns:
            Dict with ``"signals"`` (containing ``"price"`` and ``"load"``
            arrays) and optional ``"meta"`` dict.
        """
