from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
@dataclass
class ComponentMeta:
    key: str       # info 瀛楀吀涓殑閿悕
    label: str     # 鍙鍖栧瓙鍥炬爣棰?
    color: str     # 鍙鍖栧瓙鍥鹃鑹?
    sign: int      # 鍦ㄦ€诲鍔变腑鐨勭鍙凤細+1 琛ㄧず姝ｈ础鐚紝-1 琛ㄧず璐熻础鐚紙鎯╃綒椤癸級

class RewardFn(ABC):

    @abstractmethod
    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        ...

    @property
    @abstractmethod
    def component_meta(self) -> list[ComponentMeta]:
        ...
