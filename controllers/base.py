from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
class BaseController(ABC):

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def act(self, obs: dict, deterministic: bool = True) -> list[np.ndarray]:
        pass
