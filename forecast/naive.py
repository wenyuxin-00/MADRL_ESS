"""Simple baseline forecaster."""

from __future__ import annotations

import numpy as np

from forecast.base import Forecaster


class NaiveForecaster(Forecaster):
    """Repeat the recent mean as a light-weight baseline."""

    def __init__(self, window: int = 96):
        self.window = int(window)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        del signal_name
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 1:
            history = history.reshape(-1, 1)
            squeeze_output = True
        elif history.ndim == 2:
            squeeze_output = False
        else:
            raise ValueError(f"NaiveForecaster expects 1D or 2D history, got shape {history.shape}")

        current_value = history[-1] if history.shape[0] > 0 else np.zeros((history.shape[1],), dtype=np.float32)
        if horizon == 1:
            return current_value.astype(np.float32) if squeeze_output else current_value[:, None].astype(np.float32)

        tail = history[-self.window :] if history.shape[0] >= self.window else history
        mean_value = tail.mean(axis=0) if tail.size > 0 else current_value
        future = np.repeat(mean_value.reshape(-1, 1), horizon - 1, axis=1).astype(np.float32)
        window = np.concatenate([current_value.reshape(-1, 1), future], axis=1).astype(np.float32)
        return window[0] if squeeze_output else window
