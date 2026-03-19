"""数据集抽象基类。

定义所有数据集加载器的统一 API，子类需实现 num_episodes 和 get_episode 方法。
所有具体数据集加载器（如 CsvPriceLoadDataset、CsvProsumerDataset）都继承自本模块
中的 BaseEpisodeDataset 类。

主要类:
    BaseEpisodeDataset -- 多回合信号数据集的抽象接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEpisodeDataset(ABC):
    """多回合信号数据集的抽象接口。

    定义了数据集加载器必须实现的两个核心方法：
    - num_episodes: 返回数据集中可用的回合数量
    - get_episode: 根据索引返回单个回合的信号数据

    具体实现参见 ``data/loaders/csv_price_load.py`` 和
    ``data/loaders/csv_prosumer.py``。
    """

    @abstractmethod
    def num_episodes(self) -> int:
        """返回数据集中可用的回合总数。

        Returns:
            int: 回合数量
        """

    @abstractmethod
    def get_episode(self, episode_idx: int) -> dict:
        """返回指定回合的信号数据。

        Args:
            episode_idx: 从零开始的回合索引。

        Returns:
            dict: 包含以下键的字典：
                - "signals": 信号数据字典，至少包含 "price"（电价数组）
                  和 "load"（负荷数组）
                - "meta": 可选的元数据字典，包含回合索引、数据路径等信息
        """
