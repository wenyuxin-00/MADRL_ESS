"""CSV 电价-负荷数据集加载器。

从 CSV 文件中加载电价（price）和负荷（load）时间序列数据，
并按照指定的回合长度（episode_length）将其切分为多个独立回合，
供强化学习训练使用。

支持两种 CSV 格式：
- 多智能体格式：包含 load1, load2, ..., loadN 列（每个智能体独立负荷）
- 共享负荷格式：包含单个 load 列（所有智能体共享同一负荷曲线）

主要类:
    CsvPriceLoadDataset -- CSV 电价负荷数据集，继承自 BaseEpisodeDataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from data.loaders.base import BaseEpisodeDataset


class CsvPriceLoadDataset(BaseEpisodeDataset):
    """从 CSV 读取电价/负荷信号，并切分成多个回合。

    属性:
        data_path (Path): CSV 数据文件路径
        episode_length (int): 每个回合包含的时间步数
        n_agents (int): 智能体数量（决定负荷列的读取方式）
    """

    def __init__(self, data_path: Union[str, Path], episode_length: int, n_agents: int):
        """初始化数据集并加载数据。

        Args:
            data_path: CSV 数据文件路径
            episode_length: 每个回合的时间步数
            n_agents: 智能体数量
        """
        self.data_path = Path(data_path)
        self.episode_length = int(episode_length)
        self.n_agents = int(n_agents)
        self._load()

    def _load(self) -> None:
        """从 CSV 文件加载并解析电价和负荷数据。

        解析逻辑：
        1. 读取 CSV 表头，确定列索引
        2. 提取 price 列作为电价序列
        3. 尝试读取 load1..loadN 列；若不存在则使用共享 load 列
        4. 根据总时间步数和回合长度计算可用回合数

        Raises:
            ValueError: CSV 缺少 price 列、缺少负荷列、或数据长度不足
        """
        # 读取 CSV 表头以确定各列名称
        with self.data_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")

        # 加载全部数值数据（跳过表头行）
        raw = np.loadtxt(self.data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)  # 单行数据时调整为二维

        # 提取电价列
        if "price" not in header:
            raise ValueError(f"CSV must contain a 'price' column, got header={header}")
        price_idx = header.index("price")
        self._all_price = raw[:, price_idx].astype(np.float32)

        # 提取负荷列：优先使用每个智能体独立的 load1..loadN 列
        load_cols = [header.index(f"load{i + 1}") for i in range(self.n_agents) if f"load{i + 1}" in header]
        if len(load_cols) == self.n_agents:
            # 每个智能体有独立的负荷列
            self._all_load = raw[:, load_cols].astype(np.float32)
        elif "load" in header:
            # 回退方案：使用共享负荷列，复制给所有智能体
            shared_load = raw[:, header.index("load")].astype(np.float32)[:, None]
            self._all_load = np.repeat(shared_load, self.n_agents, axis=1)
        else:
            raise ValueError(
                f"CSV must contain 'load1..loadN' or 'load', got header={header}"
            )

        # 计算可切分的回合数量
        total_steps = len(self._all_price)
        self._num_episodes = total_steps // self.episode_length
        if self._num_episodes <= 0:
            raise ValueError(
                f"Dataset is too short: len={total_steps}, episode_length={self.episode_length}"
            )

    def num_episodes(self) -> int:
        """返回数据集中可用的回合总数。

        Returns:
            int: 回合数量
        """
        return self._num_episodes

    def get_episode(self, episode_idx: int) -> dict:
        """返回指定索引的回合数据。

        Args:
            episode_idx: 从零开始的回合索引

        Returns:
            dict: 包含以下结构的字典：
                - "signals": {"price": 电价数组, "load": 负荷数组(n_steps, n_agents)}
                - "meta": 回合元数据（索引、数据路径、信号名称等）

        Raises:
            IndexError: 回合索引越界
        """
        if episode_idx < 0 or episode_idx >= self._num_episodes:
            raise IndexError(f"episode_idx={episode_idx} is out of range [0, {self._num_episodes - 1}]")

        # 根据回合索引计算起止位置
        start = episode_idx * self.episode_length
        end = start + self.episode_length
        return {
            "signals": {
                "price": self._all_price[start:end].copy(),
                "load": self._all_load[start:end, :].copy(),
            },
            "meta": {
                "episode_idx": int(episode_idx),
                "data_path": str(self.data_path),
                "signal_names": ["price", "load"],
                "note": "当前为最小 price/load 数据集实现，可扩展到 price/load/pv 等多信号版本。",
            },
        }
