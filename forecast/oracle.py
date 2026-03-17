"""Perfect forecaster based on episode ground truth."""

from __future__ import annotations

import numpy as np

from forecast.base import Forecaster


class PerfectForecaster(Forecaster):
    """Return the current episode's real future values."""

    def __init__(self, episode_signals: dict[str, np.ndarray] | np.ndarray | None = None):
        self._episode_signals: dict[str, np.ndarray] = {}
        if episode_signals is not None:
            self.set_episode(episode_signals)

    def set_episode(self, episode_signals: dict[str, np.ndarray] | np.ndarray) -> None:
        """Inject the current episode's full signal dict at ``env.reset()``."""
        if isinstance(episode_signals, dict):
            self._episode_signals = {
                name: np.asarray(values, dtype=np.float32)
                for name, values in episode_signals.items()
            }
            return

        self._episode_signals = {"price": np.asarray(episode_signals, dtype=np.float32)}

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)
        if signal_name not in self._episode_signals:
            available = sorted(self._episode_signals)
            raise RuntimeError(
                f"PerfectForecaster has no episode signal '{signal_name}'. Available signals: {available}"
            )

        full_signal = np.asarray(self._episode_signals[signal_name], dtype=np.float32)
        history = np.asarray(history, dtype=np.float32)
        time_index = max(0, history.shape[0] - 1)

        if full_signal.ndim == 1:
            chunk = full_signal[time_index : time_index + horizon]
            if chunk.size < horizon:
                chunk = np.concatenate(
                    [chunk, np.zeros((horizon - chunk.size,), dtype=np.float32)],
                    axis=0,
                )
            return chunk[:horizon].astype(np.float32)

        if full_signal.ndim != 2:
            raise ValueError(f"PerfectForecaster expects 1D or 2D episode signals, got {full_signal.shape}")

        chunk = full_signal[time_index : time_index + horizon, :]
        if chunk.shape[0] < horizon:
            pad = np.zeros((horizon - chunk.shape[0], chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk[:horizon, :].T.astype(np.float32)
