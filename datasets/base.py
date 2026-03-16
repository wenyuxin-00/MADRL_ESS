"""
datasets/base.py
职责：定义数据集的统一抽象接口。

所有数据集实现须继承 BaseEpisodeDataset，实现：
  - num_episodes() -> int
  - get_episode(episode_idx) -> dict

返回的 episode 字典统一格式：
  {
      "price": np.ndarray shape (T,)      电价时间序列
      "load":  np.ndarray shape (T, N)    负荷时间序列（N 个 agent）
      "meta":  dict                        元信息（episode 编号等）
  }

后续扩展方向：
  - SyntheticDataset（合成数据）
  - PandasDataset（DataFrame 输入）
  - WindEnergyDataset（风光储场景）
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEpisodeDataset(ABC):
    """多 episode 数据集抽象基类。"""

    @abstractmethod
    def num_episodes(self) -> int:
        """返回数据集中可用的 episode 总数。"""
        ...

    @abstractmethod
    def get_episode(self, episode_idx: int) -> dict:
        """
        返回指定 episode 的数据。

        Parameters
        ----------
        episode_idx : int
            Episode 编号，范围 [0, num_episodes()-1]。

        Returns
        -------
        dict with keys:
            "price" : np.ndarray, shape (T,), float32
            "load"  : np.ndarray, shape (T, N), float32
            "meta"  : dict
        """
        ...
