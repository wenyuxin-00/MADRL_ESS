"""数据集契约。

环境只依赖“episode 级 signals 数据”，不依赖具体文件格式。
统一输出格式：

{
    "signals": {
        "price": np.ndarray,
        "load": np.ndarray,
        ...
    },
    "meta": {...}
}

这样后续加入 `pv` 等新信号时，只需要扩展 signals 字典即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEpisodeDataset(ABC):
    """多 episode signal 数据集的抽象接口。"""

    @abstractmethod
    def num_episodes(self) -> int:
        """返回数据集中可用的 episode 数量。"""

    @abstractmethod
    def get_episode(self, episode_idx: int) -> dict:
        """返回一个 episode 的开放式 signals 数据。"""
