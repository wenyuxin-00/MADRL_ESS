"""CSV 产消者数据集加载器。

加载含负荷、光伏、电价等多信号的产消者时间序列数据，
支持多智能体场景。

主要类:
    CsvProsumerDataset -- 产消者数据集
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from data.loaders.base import BaseEpisodeDataset


class CsvProsumerDataset(BaseEpisodeDataset):
    """Load price/load/pv signals for a fixed set of prosumers from CSV."""

    def __init__(
        self,
        data_path: Union[str, Path],
        episode_length: int,
        n_agents: int,
        metadata_path: Union[str, Path, None] = None,
    ):
        self.data_path = Path(data_path)
        self.episode_length = int(episode_length)
        self.n_agents = int(n_agents)
        self.metadata_path = Path(metadata_path) if metadata_path is not None else self.data_path.with_name(
            "simbench_2016_metadata.json"
        )
        self._meta_template: dict[str, object] = {}
        self._segment_payloads: list[dict[str, object]] = []
        self._episode_slices: list[tuple[int, int, int]] = []
        self._load()

    def _load(self) -> None:
        frame = pd.read_csv(self.data_path)
        header = frame.columns.tolist()

        if "price" not in header:
            raise ValueError(f"CSV must contain a 'price' column, got header={header}")

        if "is_warmup" in frame.columns:
            frame = frame.loc[~frame["is_warmup"].astype(bool)].copy()
        if frame.empty:
            raise ValueError(f"CSV '{self.data_path}' does not contain any non-warmup rows.")

        load_cols = [f"load{i + 1}" for i in range(self.n_agents) if f"load{i + 1}" in header]
        pv_cols = [f"pv{i + 1}" for i in range(self.n_agents) if f"pv{i + 1}" in header]
        if len(load_cols) != self.n_agents:
            raise ValueError(f"CSV must contain load1..load{self.n_agents}, got header={header}")
        if len(pv_cols) != self.n_agents:
            raise ValueError(f"CSV must contain pv1..pv{self.n_agents}, got header={header}")

        if "segment_id" not in frame.columns:
            frame["segment_id"] = 0
        frame["segment_id"] = pd.to_numeric(frame["segment_id"], errors="coerce").fillna(-1).astype(int)

        self._segment_payloads = []
        self._episode_slices = []
        for segment_id, segment_frame in frame.groupby("segment_id", sort=False):
            if segment_frame.empty:
                continue
            payload = {
                "segment_id": int(segment_id),
                "price": segment_frame["price"].to_numpy(dtype=np.float32),
                "load": segment_frame.loc[:, load_cols].to_numpy(dtype=np.float32),
                "pv": segment_frame.loc[:, pv_cols].to_numpy(dtype=np.float32),
            }
            segment_index = len(self._segment_payloads)
            self._segment_payloads.append(payload)

            num_segment_episodes = len(payload["price"]) // self.episode_length
            for local_episode_idx in range(num_segment_episodes):
                start = local_episode_idx * self.episode_length
                end = start + self.episode_length
                self._episode_slices.append((segment_index, start, end))

        self._num_episodes = len(self._episode_slices)
        if self._num_episodes <= 0:
            raise ValueError(
                f"Dataset is too short: no segment in '{self.data_path}' can provide episode_length={self.episode_length}"
            )

        self._meta_template = self._resolve_metadata()

    def _resolve_metadata(self) -> dict[str, object]:
        pv_arrays = [payload["pv"] for payload in self._segment_payloads if len(payload["pv"]) > 0]
        if not pv_arrays:
            raise ValueError(f"CSV '{self.data_path}' does not contain any PV rows after filtering.")
        pv_peak_kw = np.max(np.concatenate(pv_arrays, axis=0), axis=0).astype(np.float32)
        if self.metadata_path.exists():
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            pv_peak_kw = np.asarray(payload.get("pv_peak_kw", pv_peak_kw), dtype=np.float32).reshape(self.n_agents)
            node_ids = list(payload.get("selected_buses", range(self.n_agents)))
        else:
            node_ids = list(range(self.n_agents))

        ess_power_kw = (pv_peak_kw * np.float32(0.5)).astype(np.float32)
        ess_capacity_kwh = (ess_power_kw * np.float32(2.5)).astype(np.float32)
        return {
            "node_ids": node_ids,
            "pv_peak_kw": pv_peak_kw.copy(),
            "ess_power_kw": ess_power_kw.copy(),
            "ess_capacity_kwh": ess_capacity_kwh.copy(),
            "price_unit": "EUR/kWh",
            "power_unit": "kW",
            "energy_unit": "kWh",
        }

    def num_episodes(self) -> int:
        return self._num_episodes

    def get_episode(self, episode_idx: int) -> dict:
        if episode_idx < 0 or episode_idx >= self._num_episodes:
            raise IndexError(f"episode_idx={episode_idx} is out of range [0, {self._num_episodes - 1}]")

        segment_index, start, end = self._episode_slices[episode_idx]
        payload = self._segment_payloads[segment_index]
        return {
            "signals": {
                "price": payload["price"][start:end].copy(),
                "load": payload["load"][start:end, :].copy(),
                "pv": payload["pv"][start:end, :].copy(),
            },
            "meta": {
                "episode_idx": int(episode_idx),
                "segment_id": int(payload["segment_id"]),
                "segment_episode_idx": int(start // self.episode_length),
                "data_path": str(self.data_path),
                "signal_names": ["price", "load", "pv"],
                **self._meta_template,
            },
        }
