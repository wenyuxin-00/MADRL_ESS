"""
datasets/csv_price_load.py
职责：从 CSV 文件加载电价 + 负荷数据，按 episode_length 切分为 episode 序列。

这是从 envs/hems_env.py 的 _load_data 方法中剥离出来的数据层，
使得 env 本身不再负责 CSV 解析。

CSV 格式要求：
  - 必须包含 "price" 列
  - 负荷列：优先识别 load1, load2, ..., loadN；若不存在则使用单列 "load" 广播到 N 个 agent
  - 行数必须 >= episode_length * 1

后续扩展：可以在此添加数据增强、归一化、缓存等功能。
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np

from datasets.base import BaseEpisodeDataset


class CsvPriceLoadDataset(BaseEpisodeDataset):
    """CSV 格式的电价+负荷数据集。

    Parameters
    ----------
    data_path : str or Path
        CSV 文件路径。
    episode_length : int
        每个 episode 的步数（时间步数）。
    n_agents : int
        智能体（电池）数量，决定负荷列的读取方式。
    """

    def __init__(self, data_path: Union[str, Path], episode_length: int, n_agents: int):
        self.data_path = Path(data_path)
        self.episode_length = episode_length
        self.n_agents = n_agents
        self._load()

    def _load(self) -> None:
        """读取并解析 CSV 文件。"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")

        raw = np.loadtxt(self.data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        # price 列
        if "price" not in header:
            raise ValueError(f"CSV 必须包含 'price' 列，当前 header={header}")
        price_idx = header.index("price")
        self._all_price = raw[:, price_idx].astype(np.float32)  # (T_all,)

        # load 列：优先 load1..loadN，否则广播单列 load
        load_cols = [header.index(f"load{i+1}") for i in range(self.n_agents)
                     if f"load{i+1}" in header]

        if len(load_cols) == self.n_agents:
            self._all_load = raw[:, load_cols].astype(np.float32)          # (T_all, N)
        elif "load" in header:
            one = raw[:, header.index("load")].astype(np.float32)[:, None]  # (T_all, 1)
            self._all_load = np.repeat(one, self.n_agents, axis=1)          # (T_all, N)
        else:
            raise ValueError(
                f"CSV 必须包含 load 列（load1..loadN 或 load）。当前 header={header}"
            )

        # 可用 episode 数
        total_steps = len(self._all_price)
        self._num_episodes = total_steps // self.episode_length
        if self._num_episodes <= 0:
            raise ValueError(
                f"数据长度不足：len={total_steps}, episode_length={self.episode_length}"
            )

    # ----------------------------------------------------------------
    # BaseEpisodeDataset API
    # ----------------------------------------------------------------

    def num_episodes(self) -> int:
        return self._num_episodes

    def get_episode(self, episode_idx: int) -> dict:
        if episode_idx < 0 or episode_idx >= self._num_episodes:
            raise IndexError(
                f"episode_idx={episode_idx} 超出范围 [0, {self._num_episodes - 1}]"
            )
        start = episode_idx * self.episode_length
        end   = start + self.episode_length
        return {
            "price": self._all_price[start:end].copy(),        # (T,)
            "load":  self._all_load[start:end, :].copy(),      # (T, N)
            "meta":  {"episode_idx": episode_idx,
                      "data_path": str(self.data_path)},
        }
