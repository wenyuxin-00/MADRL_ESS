"""Dataset contract for episode-level signal data.

The environment depends only on episode-level signal dicts, not on any
specific file format. Unified output format:

    {
        "signals": {
            "price": np.ndarray,   # shape (T,)   -- electricity price
            "load":  np.ndarray,   # shape (T, N) -- load per agent
            ...                    # extensible: "pv", "wind", etc.
        },
        "meta": {...}              # optional metadata
    }

To add new signal types (e.g., PV generation), just extend the ``signals``
dict -- no interface changes needed.
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
