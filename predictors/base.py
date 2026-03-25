"""Abstract forecasting interface used by the grid mainline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Base interface for perfect and learned forecasters.

    Forecasters only replace the forward-looking observation window. They do not
    change the environment dynamics or reward calculation.
    """

    @abstractmethod
    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        """Return the forecast window for one signal.

        Shared signals such as ``price`` use history shaped ``(t + 1,)`` and
        return ``(horizon,)``. Per-agent signals such as ``load`` and ``pv`` use
        history shaped ``(t + 1, n_agents)`` and return ``(n_agents, horizon)``.
        """

    def reset(self) -> None:
        """Reset any optional episode-level internal state."""
        return None
