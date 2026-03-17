"""CSV 版最小数据集实现。

当前只实现 `price/load` 两类信号，作为整个实验框架的最小可运行版本。
后续如果加入 `pv`，建议新增新的 dataset 类，而不是把抽象层重新写死。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from datasets.base import BaseEpisodeDataset


class CsvPriceLoadDataset(BaseEpisodeDataset):
    """从 CSV 读取 `price/load`，并切分成多个 episode。"""

    def __init__(self, data_path: Union[str, Path], episode_length: int, n_agents: int):
        self.data_path = Path(data_path)
        self.episode_length = int(episode_length)
        self.n_agents = int(n_agents)
        self._load()

    def _load(self) -> None:
        with self.data_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")

        raw = np.loadtxt(self.data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        if "price" not in header:
            raise ValueError(f"CSV must contain a 'price' column, got header={header}")
        price_idx = header.index("price")
        self._all_price = raw[:, price_idx].astype(np.float32)

        load_cols = [header.index(f"load{i + 1}") for i in range(self.n_agents) if f"load{i + 1}" in header]
        if len(load_cols) == self.n_agents:
            self._all_load = raw[:, load_cols].astype(np.float32)
        elif "load" in header:
            shared_load = raw[:, header.index("load")].astype(np.float32)[:, None]
            self._all_load = np.repeat(shared_load, self.n_agents, axis=1)
        else:
            raise ValueError(
                f"CSV must contain 'load1..loadN' or 'load', got header={header}"
            )

        total_steps = len(self._all_price)
        self._num_episodes = total_steps // self.episode_length
        if self._num_episodes <= 0:
            raise ValueError(
                f"Dataset is too short: len={total_steps}, episode_length={self.episode_length}"
            )

    def num_episodes(self) -> int:
        return self._num_episodes

    def get_episode(self, episode_idx: int) -> dict:
        if episode_idx < 0 or episode_idx >= self._num_episodes:
            raise IndexError(f"episode_idx={episode_idx} is out of range [0, {self._num_episodes - 1}]")

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
