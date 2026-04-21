from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
class ObservationBuilder(ABC):

    @abstractmethod
    def get_schema(self, n_agents: int) -> dict[str, tuple[int, ...]]:
        pass

    @abstractmethod
    def get_layout(self, n_agents: int) -> dict[str, dict]:
        pass

    @abstractmethod
    def build(self, env) -> dict[str, np.ndarray]:
        pass

    def zeros(self, n_agents: int) -> dict[str, np.ndarray]:
        return {
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in self.get_schema(n_agents).items()
        }
