from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
import pandas as pd

class Forecaster(ABC):

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int, *, signal_name: str='wholesale_price', history_timestamps: Sequence[str | pd.Timestamp] | None=None) -> np.ndarray:
        del history_timestamps

    def reset(self) -> None:
        return None

    def set_episode(self, episode_signals: dict[str, np.ndarray] | np.ndarray, episode_meta: dict[str, object] | None=None) -> None:
        del episode_signals, episode_meta
        return None
