"""CSV 产消者数据集加载器。

加载含负荷（load）、光伏（pv）、电价（price）等多信号的产消者时间序列数据，
支持多智能体场景和按段（segment）划分的数据组织方式。

该加载器可处理由 simbench_export 工具导出的训练/测试 CSV 文件，自动过滤
预热（warmup）行，并按 segment_id 分组后将每段数据切分为多个独立回合。

主要类:
    CsvProsumerDataset -- 产消者数据集，继承自 BaseEpisodeDataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from data.loaders.base import BaseEpisodeDataset


class CsvProsumerDataset(BaseEpisodeDataset):
    """从 CSV 加载产消者（prosumer）的电价/负荷/光伏信号数据。

    支持按段（segment_id）组织的数据格式，每段内独立切分回合。
    自动过滤 is_warmup 标记的预热行，并从元数据文件中读取储能参数。

    属性:
        data_path (Path): CSV 数据文件路径
        episode_length (int): 每个回合包含的时间步数
        n_agents (int): 智能体（产消者）数量
        metadata_path (Path): 元数据 JSON 文件路径（包含光伏峰值功率、节点 ID 等）
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        episode_length: int,
        n_agents: int,
        metadata_path: Union[str, Path, None] = None,
    ):
        """初始化产消者数据集并加载数据。

        Args:
            data_path: CSV 数据文件路径
            episode_length: 每个回合的时间步数
            n_agents: 智能体数量
            metadata_path: 元数据 JSON 文件路径；若为 None，则自动使用
                与 data_path 同目录下的 simbench_2016_metadata.json
        """
        self.data_path = Path(data_path)
        self.episode_length = int(episode_length)
        self.n_agents = int(n_agents)
        # 若未指定元数据路径，则默认使用同目录下的 simbench_2016_metadata.json
        self.metadata_path = Path(metadata_path) if metadata_path is not None else self.data_path.with_name(
            "simbench_2016_metadata.json"
        )
        self._meta_template: dict[str, object] = {}          # 元数据模板（含储能参数等）
        self._segment_payloads: list[dict[str, object]] = []  # 各段的信号数据
        self._episode_slices: list[tuple[int, int, int]] = [] # 回合索引 -> (段索引, 起始, 结束)
        self._load()

    def _load(self) -> None:
        """从 CSV 文件加载并解析产消者数据。

        解析逻辑：
        1. 读取 CSV 并过滤预热行（is_warmup=True）
        2. 校验 price、load1..loadN、pv1..pvN 列是否齐全
        3. 按 segment_id 分组，每段独立存储信号数组
        4. 在每段内按 episode_length 切分回合，记录切片索引
        5. 解析元数据（光伏峰值、储能参数等）

        Raises:
            ValueError: CSV 缺少必要列、无有效数据行、或数据长度不足
        """
        frame = pd.read_csv(self.data_path)
        header = frame.columns.tolist()

        if "price" not in header:
            raise ValueError(f"CSV must contain a 'price' column, got header={header}")

        # 过滤预热行（测试集中的 warmup 数据不参与回合切分）
        if "is_warmup" in frame.columns:
            frame = frame.loc[~frame["is_warmup"].astype(bool)].copy()
        if frame.empty:
            raise ValueError(f"CSV '{self.data_path}' does not contain any non-warmup rows.")

        # 校验每个智能体的负荷列和光伏列是否齐全
        load_cols = [f"load{i + 1}" for i in range(self.n_agents) if f"load{i + 1}" in header]
        pv_cols = [f"pv{i + 1}" for i in range(self.n_agents) if f"pv{i + 1}" in header]
        if len(load_cols) != self.n_agents:
            raise ValueError(f"CSV must contain load1..load{self.n_agents}, got header={header}")
        if len(pv_cols) != self.n_agents:
            raise ValueError(f"CSV must contain pv1..pv{self.n_agents}, got header={header}")

        # 若 CSV 中没有 segment_id 列，则视为单段数据
        if "segment_id" not in frame.columns:
            frame["segment_id"] = 0
        frame["segment_id"] = pd.to_numeric(frame["segment_id"], errors="coerce").fillna(-1).astype(int)

        # 按段分组，提取各段的信号数组并切分回合
        self._segment_payloads = []
        self._episode_slices = []
        for segment_id, segment_frame in frame.groupby("segment_id", sort=False):
            if segment_frame.empty:
                continue
            # 将当前段的信号提取为 numpy 数组
            payload = {
                "segment_id": int(segment_id),
                "price": segment_frame["price"].to_numpy(dtype=np.float32),
                "load": segment_frame.loc[:, load_cols].to_numpy(dtype=np.float32),
                "pv": segment_frame.loc[:, pv_cols].to_numpy(dtype=np.float32),
            }
            segment_index = len(self._segment_payloads)
            self._segment_payloads.append(payload)

            # 在当前段内按 episode_length 切分回合
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

        # 解析元数据模板（储能参数、节点 ID 等）
        self._meta_template = self._resolve_metadata()

    def _resolve_metadata(self) -> dict[str, object]:
        """解析元数据，计算各智能体的储能系统参数。

        从所有段的光伏数据中计算峰值功率，并据此推导储能功率和容量。
        若存在元数据 JSON 文件，则从中读取光伏峰值和节点 ID。

        储能参数计算规则：
        - 储能功率 = 光伏峰值功率 * 0.5
        - 储能容量 = 储能功率 * 2.5

        Returns:
            dict: 元数据字典，包含 node_ids、pv_peak_kw、ess_power_kw、
                ess_capacity_kwh 及单位信息

        Raises:
            ValueError: 所有段中均无光伏数据
        """
        # 收集所有段的光伏数据以计算峰值
        pv_arrays = [payload["pv"] for payload in self._segment_payloads if len(payload["pv"]) > 0]
        if not pv_arrays:
            raise ValueError(f"CSV '{self.data_path}' does not contain any PV rows after filtering.")
        # 计算每个智能体在所有段中的光伏峰值功率
        pv_peak_kw = np.max(np.concatenate(pv_arrays, axis=0), axis=0).astype(np.float32)
        if self.metadata_path.exists():
            # 从元数据 JSON 文件中读取精确参数
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            pv_peak_kw = np.asarray(payload.get("pv_peak_kw", pv_peak_kw), dtype=np.float32).reshape(self.n_agents)
            node_ids = list(payload.get("selected_buses", range(self.n_agents)))
        else:
            # 无元数据文件时使用默认的顺序编号
            node_ids = list(range(self.n_agents))

        # 根据光伏峰值推导储能系统参数
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
                - "signals": {"price": 电价数组, "load": 负荷数组, "pv": 光伏数组}
                - "meta": 回合元数据（索引、段 ID、储能参数、节点 ID 等）

        Raises:
            IndexError: 回合索引越界
        """
        if episode_idx < 0 or episode_idx >= self._num_episodes:
            raise IndexError(f"episode_idx={episode_idx} is out of range [0, {self._num_episodes - 1}]")

        # 根据回合索引查找对应的段和切片范围
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
