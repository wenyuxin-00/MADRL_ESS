from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

TZ_LOCAL = "Europe/Berlin"
DEFAULT_PROCESSED_SUBDIR = Path("processed/prosumer")


def _parse_optional_local_date(value: str | date | None) -> date | None:
    # 作用：把可选日期配置统一成本地自然日，方便后续按日期过滤时间序列。
    # 示例：输入 "2020-01-01 15:30" -> 输出 date(2020, 1, 1)；输入 "" -> 输出 None。
    return None if value in (None, "") else pd.Timestamp(value).date()


def _coerce_scale_vector(scale: Sequence[float] | float | None, *, name: str, n_agents: int) -> np.ndarray:
    # 作用：把标量、列表或空配置整理成长度为 n_agents 的 float32 缩放向量。
    # 示例：输入 scale=[1,1,1], n_agents=3 -> 输出 [1.0, 1.0, 1.0]；输入 [1.0, 0.5] -> 输出 [1.0, 0.5]。
    default = np.ones(n_agents, dtype=np.float32)
    if scale is None:
        return default

    values = np.asarray(scale, dtype=np.float32).reshape(-1)
    if values.size == 1:
        return np.full((n_agents,), float(values[0]), dtype=np.float32)

    if values.size != n_agents:
        raise ValueError(f"{name} length should equal n_agents={n_agents}, got {values.size}")

    return values.astype(np.float32, copy=False)


class ProsumerDataset:
    """把 processed/prosumer CSV 转成环境、归一化和 shared-data 共用的 episode 数据包。"""

    def __init__(
        self,
        data_dir: str | Path,
        episode_length: int,
        n_agents: int,
        agent_profiles: list[str],
        year: int,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        load_components: Sequence[str] = ("household", "heatpump"),
        pv_reference: str = "south",
        pv_capacity_kw: Sequence[float] | None = None,
        load_scale: Sequence[float] | float | None = None,
        pv_scale: Sequence[float] | float | None = None,
        node_ids: Sequence[int] | None = None,
        history_warmup_steps: int = 0,
        *,
        window_stride_steps: int | None = None,
        window_strategy: str = "cfg_window",
        window_days: int = 1,
        window_stride_days: int = 1,
        base_episode_length: int | None = None,
        split: str = "train",
    ) -> None:
        # 作用：创建 prosumer 数据集并完成参数校验、CSV 加载、信号对齐和 episode 切片。
        # 示例：CSV 中 household=1.0, heatpump=0.2, load_scale=10 -> 对象内部 load=(1.0+0.2)*10=12.0。
        # 这里固定输入契约：调用方传 data 根目录，loader 只读取规范的 processed/prosumer 子目录。
        self.data_dir = Path(data_dir) / DEFAULT_PROCESSED_SUBDIR
        self.episode_length = int(episode_length)
        self.window_stride_steps = int(window_stride_steps or episode_length)
        self.window_strategy = str(window_strategy)
        self.window_days = int(window_days)
        self.window_stride_days = int(window_stride_days)
        self.base_episode_length = int(base_episode_length or episode_length)
        self.split = str(split)
        self.history_warmup_steps = int(history_warmup_steps)
        self.n_agents = int(n_agents)
        self.agent_profiles = list(agent_profiles)
        self.year = int(year)
        self.start_date = _parse_optional_local_date(start_date)
        self.end_date = _parse_optional_local_date(end_date)
        self.load_components = tuple(str(component) for component in load_components)
        self.pv_reference = str(pv_reference).lower()
        self.pv_capacity_kw = None if pv_capacity_kw is None or len(pv_capacity_kw) == 0 else list(pv_capacity_kw)

        # scale 是用户配置，内部统一保存成逐智能体向量，避免后面每个信号分支都处理标量/列表两套形态。
        self.load_scale = _coerce_scale_vector(load_scale, name="load_scale", n_agents=self.n_agents)
        self.pv_scale = _coerce_scale_vector(pv_scale, name="pv_scale", n_agents=self.n_agents)
        self.node_ids = list(node_ids) if node_ids is not None else list(range(self.n_agents))

        # 这些缓存是加载完成后的主状态：信号矩阵、时间戳、episode 切片和归一化采样位置。
        self._signals: dict[str, np.ndarray] = {}
        self._timestamps: pd.Series | None = None
        self._meta_template: dict[str, object] = {}
        self._episode_slices: list[tuple[int, int, int, int | None]] = []
        self._normalization_positions = np.zeros(0, dtype=np.int64)
        self.dropped_tail_steps = 0
        self._num_episodes = 0

        self._validate_init_args()
        self._load()

    def _validate_init_args(self) -> None:
        # 作用：在入口处拒绝会导致矩阵错位或空 episode 的明显非法配置。
        # 示例：输入 episode_length=0 -> 输出 ValueError；输入 episode_length=96, window_stride_steps=24 -> 校验通过。
        for name in ("episode_length", "window_stride_steps"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.history_warmup_steps < 0:
            raise ValueError(f"history_warmup_steps must be non-negative, got {self.history_warmup_steps}")
        if len(self.agent_profiles) != self.n_agents:
            raise ValueError(
                f"agent_profiles length should equal n_agents={self.n_agents}, got {len(self.agent_profiles)}"
            )
        if not self.load_components:
            raise ValueError("At least one load component is required.")
        if self.pv_capacity_kw is not None and len(self.pv_capacity_kw) != self.n_agents:
            raise ValueError(
                f"pv_capacity_kw length should equal n_agents={self.n_agents}, got {len(self.pv_capacity_kw)}"
            )
        if len(self.node_ids) != self.n_agents:
            raise ValueError(f"node_ids length should equal n_agents={self.n_agents}, got {len(self.node_ids)}")

    def _read_csv(self, name: str) -> pd.DataFrame:
        # 作用：读取规范 CSV 并把 timestamp 解析为 Europe/Berlin 时区时间。
        # 示例：CSV timestamp="2019-12-31 23:00+00:00" -> 输出 timestamp="2020-01-01 00:00+01:00"。
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing processed prosumer file: {path}")
        frame = pd.read_csv(path)
        if "timestamp" not in frame.columns:
            raise ValueError(f"Processed prosumer file '{path}' must contain a 'timestamp' column.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(TZ_LOCAL)
        return frame

    def _build_split_masks(self, timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        # 作用：区分当前 split 的有效控制行 active 和允许补历史的可读取行 accessible。
        # 示例：日期 [1/1,1/2,1/3], start=end=1/2, history=1 -> active=[F,T,F], accessible=[T,T,T]。
        local_dates = timestamps.dt.date
        year_mask = (timestamps.dt.year == self.year).to_numpy(dtype=bool)

        active_mask = year_mask.copy()
        if self.start_date is not None:
            active_mask &= (local_dates >= self.start_date).to_numpy(dtype=bool)
        if self.end_date is not None:
            active_mask &= (local_dates <= self.end_date).to_numpy(dtype=bool)

        # LSTM 历史可以读 active 日期窗外的同年数据，但不能跨年份。
        accessible_mask = year_mask if self.history_warmup_steps > 0 else active_mask.copy()
        return active_mask, accessible_mask

    def _load_components(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
        # 作用：读取并相加 household、heatpump 等负荷分量，形成统一的总负荷时间轴。
        # 示例：同一时刻 household=1.0, heatpump=0.2, load_scale=10 -> 总负荷 (1.0+0.2)*10=12.0。
        component_frames: dict[str, pd.DataFrame] = {}
        base_timestamp: pd.Series | None = None
        active_on_accessible: np.ndarray | None = None
        source_positions: np.ndarray | None = None

        for component in self.load_components:
            raw_frame = self._read_csv(f"{component}.csv")
            active_mask, accessible_mask = self._build_split_masks(raw_frame["timestamp"])

            # 画像列在这里直接索引：缺列时让 pandas 报出真实列名，而不是吞掉错误或猜替代列。
            frame = raw_frame.loc[accessible_mask, ["timestamp", *self.agent_profiles]].copy().reset_index(drop=True)

            if base_timestamp is None:
                # active_mask 原本对应原 CSV；切到 accessible 时间轴后还要同步投影，episode 生成才不会错位。
                base_timestamp = frame["timestamp"].reset_index(drop=True)
                active_on_accessible = np.asarray(active_mask[accessible_mask], dtype=bool)
                source_positions = np.flatnonzero(accessible_mask).astype(np.int64)
            elif not base_timestamp.equals(frame["timestamp"].reset_index(drop=True)):
                # 分量之间不允许 reindex 静默补齐；错一行时间戳，总负荷逐点相加就没有物理意义。
                raise ValueError(f"Load component '{component}' timestamp axis does not match the base load axis.")

            component_frames[component] = frame

        if base_timestamp is None or active_on_accessible is None or source_positions is None:
            raise ValueError("No prosumer load components were resolved for dataset construction.")

        total_load = sum(
            component_frames[component].loc[:, self.agent_profiles].to_numpy(dtype=np.float32)
            for component in self.load_components
        )
        merged = pd.DataFrame({"timestamp": base_timestamp})
        merged.loc[:, self.agent_profiles] = total_load * self.load_scale[None, :]

        # source_positions 保留 accessible 行在原 CSV 中的位置，用来发现时间轴物理断点。
        return (
            merged,
            component_frames,
            active_on_accessible.astype(bool, copy=False),
            source_positions.astype(np.int64, copy=False),
        )

    def _aligned_numeric_column(self, name: str, column: str, base_timestamps: pd.Series, *, label: str) -> np.ndarray:
        # 作用：把 price 或 PV 这类单列数值按负荷时间轴精确 reindex 对齐。
        # 示例：base=[t0,t1], price 表为 t1=0.30,t0=0.20 -> 输出 [0.20,0.30]。
        frame = self._read_csv(name)
        _, accessible_mask = self._build_split_masks(frame["timestamp"])
        frame = frame.loc[accessible_mask].set_index("timestamp").sort_index().reindex(base_timestamps)
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(values).any():
            raise ValueError(f"{label} contains NaN values after timestamp alignment.")
        return values

    def _load_pv(self, base_timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        # 作用：把参考 PV 曲线转换成每个智能体的 PV 矩阵和峰值容量。
        # 示例：ref=[0,1.5,3.0], capacity=6, pv_scale=0.5 -> pv=[0,1.5,3.0], pv_peak_kw=3.0。
        reference = self._aligned_numeric_column(
            "pv_reference.csv",
            f"ref_{self.pv_reference}",
            base_timestamps,
            label="PV reference",
        )
        ref_peak_kw = float(np.max(reference)) if reference.size else 0.0

        if self.pv_capacity_kw is None:
            pv = np.repeat(reference[:, None], self.n_agents, axis=1).astype(np.float32)
            pv_peak_kw = np.full(self.n_agents, ref_peak_kw, dtype=np.float32)
        else:
            if ref_peak_kw <= 0.0:
                raise ValueError("PV reference peak is zero, cannot scale by pv_capacity_kw.")
            capacity = np.asarray(self.pv_capacity_kw, dtype=np.float32)
            pv = (reference[:, None] * (capacity / np.float32(ref_peak_kw))[None, :]).astype(np.float32)
            pv_peak_kw = capacity.copy()

        return (pv * self.pv_scale[None, :]).astype(np.float32), (pv_peak_kw * self.pv_scale).astype(np.float32)

    def _append_episode_windows(self, *, run_start_idx: int, run_end_idx: int, segment_start_idx: int) -> None:
        # 作用：把一段连续 active 索引切成固定长度 episode，并记录 history 与 bootstrap 边界。
        # 示例：run=[0,8), episode=3, stride=2, history=1 -> 切片 (0,1,4,4),(2,3,6,6),(4,5,8,None)。
        first_usable = max(run_start_idx, segment_start_idx + self.history_warmup_steps)
        usable_steps = run_end_idx - first_usable
        if usable_steps < self.episode_length:
            self.dropped_tail_steps += max(usable_steps, 0)
            return

        starts = list(range(first_usable, run_end_idx - self.episode_length + 1, self.window_stride_steps))
        for start in starts:
            end = start + self.episode_length

            # next_active 是可选的 bootstrap 步，给需要下一时刻目标值的训练/控制代码使用。
            next_active = end if end < run_end_idx else None
            self._episode_slices.append((start - self.history_warmup_steps, start, end, next_active))

        self.dropped_tail_steps += max(run_end_idx - (starts[-1] + self.episode_length), 0)

    def _load(self) -> None:
        # 作用：执行完整加载流程，构建信号缓存、元数据、归一化位置和 episode 切片。
        # 示例：负荷 1.2、电价 0.25、PV 3.0 对齐到同一 t0 -> self._signals 在 t0 保存 load=1.2, price=0.25, pv=3.0。
        # 加载顺序很重要：负荷先建立 canonical 时间轴，price 和 PV 再精确对齐到这条轴。
        load_frame, component_frames, active_mask, source_positions = self._load_components()
        base_timestamps = load_frame["timestamp"].reset_index(drop=True)
        wholesale_price = self._aligned_numeric_column(
            "price.csv",
            "price",
            base_timestamps,
            label="Processed prosumer wholesale price series",
        )
        pv, pv_peak_kw = self._load_pv(base_timestamps)
        load = load_frame.loc[:, self.agent_profiles].to_numpy(dtype=np.float32)
        if load.shape != pv.shape:
            raise ValueError(f"Load and PV shapes must match, got {load.shape} vs {pv.shape}")

        self._timestamps = base_timestamps
        self._signals = {"wholesale_price": wholesale_price, "load": load, "pv": pv}
        for component in self.load_components:
            # 分量级负荷和总负荷使用同一 load_scale，保证诊断信号与环境看到的口径一致。
            self._signals[f"load_{component}"] = (
                component_frames[component][self.agent_profiles].to_numpy(np.float32) * self.load_scale[None, :]
            )

        # meta_template 是每个 episode 共享的合同信息；下游会把它写进记录文件和 shared-data manifest。
        self._meta_template = {
            "node_ids": list(self.node_ids),
            "agent_profiles": list(self.agent_profiles),
            "year": int(self.year),
            "split": self.split,
            "date_range": {
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
            },
            "load_components": list(self.load_components),
            "pv_reference": self.pv_reference,
            "pv_peak_kw": pv_peak_kw.copy(),
            "load_scale": self.load_scale.copy(),
            "pv_scale": self.pv_scale.copy(),
            "wholesale_price_unit": "EUR/kWh",
            "power_unit": "kW",
            "energy_unit": "kWh",
            "data_dir": str(self.data_dir),
            "component_columns": {component: list(frame.columns) for component, frame in component_frames.items()},
            "window_strategy": self.window_strategy,
            "window_days": int(self.window_days),
            "window_stride_days": int(self.window_stride_days),
            "window_stride_steps": int(self.window_stride_steps),
            "base_episode_length": int(self.base_episode_length),
            "episode_length": int(self.episode_length),
            "history_warmup_steps": int(self.history_warmup_steps),
        }

        self._episode_slices = []
        self.dropped_tail_steps = 0
        active_positions = np.flatnonzero(active_mask)

        # 归一化只用 active 区间拟合，避免历史预热行影响训练窗口本身的统计分布。
        self._normalization_positions = active_positions.astype(np.int64, copy=False)
        if active_positions.size == 0:
            raise ValueError(
                f"No active prosumer rows for year={self.year}, start_date={self.start_date}, end_date={self.end_date}."
            )

        # source_positions 不连续说明原始时间轴中间有缺口；训练 split 禁止跨缺口拼 episode。
        split_points = np.flatnonzero(np.diff(source_positions) != 1) + 1
        if self.split == "train" and split_points.size:
            raise ValueError("ProsumerDataset train split contains a timestamp gap in the accessible source rows.")

        bounds = np.concatenate([[0], split_points, [len(source_positions)]])
        for segment_start, segment_end in zip(bounds[:-1], bounds[1:], strict=False):
            segment_active = active_positions[(active_positions >= segment_start) & (active_positions < segment_end)]
            if segment_active.size == 0:
                continue

            # active 本身也可能因为 start/end 被切成多段；训练 split 同样要求每段连续。
            run_split = np.flatnonzero(np.diff(segment_active) != 1) + 1
            if self.split == "train" and run_split.size:
                raise ValueError("ProsumerDataset train split contains an internal active timestamp gap.")

            for run_positions in np.split(segment_active, run_split):
                self._append_episode_windows(
                    run_start_idx=int(run_positions[0]),
                    run_end_idx=int(run_positions[-1]) + 1,
                    segment_start_idx=int(segment_start),
                )

        self._num_episodes = len(self._episode_slices)

    def num_episodes(self) -> int:
        # 作用：返回当前 split 可用的 episode 数量，供环境、归一化和 shared-data 枚举。
        # 示例：self._episode_slices=[(0,1,4,4),(2,3,6,6),(4,5,8,None)] -> 输出 3。
        return self._num_episodes

    def get_normalization_signal_values(self, signal_name: str) -> np.ndarray:
        # 作用：只取 active 区间内的信号样本，用于拟合观测归一化统计量。
        # 示例：load=[10,20,30], normalization_positions=[0,2] -> 输入 "load" 输出 [10,30]。
        if signal_name not in self._signals:
            raise KeyError(f"Unknown signal '{signal_name}' for normalization.")
        return np.asarray(self._signals[signal_name][self._normalization_positions], dtype=np.float32)

    def _signal_slice(self, start: int, end: int) -> dict[str, np.ndarray]:
        # 作用：从所有信号中截取同一段时间窗口并返回副本。
        # 示例：load=[10,20,30], pv=[1,2,3], start=1, end=3 -> 输出 load=[20,30], pv=[2,3]。
        return {name: values[start:end].copy() for name, values in self._signals.items()}

    def get_episode(self, episode_idx: int) -> dict:
        # 作用：导出一个完整 episode 数据包，包含 active 信号、历史信号、bootstrap 和 meta。
        # 示例：切片 (0,2,5,5) -> history 取 [0,1]，signals 取 [2,3,4]，bootstrap 取 [5]。
        if episode_idx < 0 or episode_idx >= self._num_episodes:
            raise IndexError(f"episode_idx={episode_idx} is out of range [0, {self._num_episodes - 1}]")

        history_start, start, end, next_active = self._episode_slices[episode_idx]
        timestamps = self._timestamps.iloc[start:end].reset_index(drop=True)
        history_timestamps = self._timestamps.iloc[history_start:start].reset_index(drop=True)
        bootstrap_timestamps = (
            []
            if next_active is None
            else self._timestamps.iloc[next_active : next_active + 1].astype(str).tolist()
        )

        return {
            "signals": self._signal_slice(start, end),
            "history_signals": self._signal_slice(history_start, start),
            "bootstrap_signals": {} if next_active is None else self._signal_slice(next_active, next_active + 1),
            "history_timestamps": history_timestamps.astype(str).tolist(),
            "history_length": int(start - history_start),
            "meta": {
                "episode_idx": int(episode_idx),
                "segment_id": int(self.year),
                "segment_episode_idx": int(episode_idx),
                "timestamps": timestamps.astype(str).tolist(),
                "bootstrap_timestamps": bootstrap_timestamps,
                "history_start_idx": int(history_start),
                "active_start_idx": int(start),
                "active_end_idx": int(end),
                "next_active_idx": None if next_active is None else int(next_active),
                "signal_names": ["wholesale_price", "load", "pv"],
                "dropped_tail_steps": int(self.dropped_tail_steps),
                **self._meta_template,
            },
        }
