"""Forecaster base contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Base class for observation-side forecasters.

    Forecasters only replace the future-looking observation windows. They do
    not change the environment's ground-truth reward computation.

    The contract supports both shared 1-D signals such as ``price`` and
    per-agent 2-D signals such as ``load`` / ``pv`` stored as ``(T, N)``.
    """

    @abstractmethod
    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        """Return a look-ahead window for one signal.

        Args:
            history: Observed signal history up to the current step.
                Shared signals use shape ``(t + 1,)``.
                Per-agent signals use shape ``(t + 1, n_agents)``.
            horizon: Number of values to return. The current value is included
                in the first position, so ``result[..., 0]`` always represents
                the current step.
            signal_name: Canonical signal key such as ``price``, ``load``, or
                ``pv``.

        Returns:
            ``(horizon,)`` for shared signals or ``(n_agents, horizon)`` for
            per-agent signals.
        """

    def reset(self) -> None:
        """Reset optional episode-local state."""
        return None
