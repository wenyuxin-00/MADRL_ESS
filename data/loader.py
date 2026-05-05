from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from configs.cfg import Cfg
from utils.paths import PROJECT_ROOT

TZ_LOCAL = "Europe/Berlin"


@dataclass(frozen=True)
class SeriesData:
    timestamps: list[str]
    price: np.ndarray
    load: np.ndarray
    load_components: dict[str, np.ndarray]
    pv: np.ndarray
    target_start: int
    target_length: int


def _scale(values: tuple[float, ...], n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.full((int(n),), float(arr[0]), dtype=np.float32)
    if arr.size != int(n):
        raise ValueError(f"{name} expected {n} values, got {arr.size}.")
    return arr


def _read_time_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(TZ_LOCAL)
    return frame


def _date_mask(timestamps: pd.Series, year: int, start: str, end: str) -> np.ndarray:
    dates = timestamps.dt.date
    mask = (timestamps.dt.year == int(year)).to_numpy(dtype=bool)
    mask &= (dates >= pd.Timestamp(start).date()).to_numpy(dtype=bool)
    mask &= (dates <= pd.Timestamp(end).date()).to_numpy(dtype=bool)
    return mask


def _shift_date(date_text: str, steps: int, dt_hours: float) -> str:
    shifted = pd.Timestamp(date_text) + pd.Timedelta(hours=float(steps) * float(dt_hours))
    return (shifted.ceil("D") if int(steps) > 0 else shifted).date().isoformat()


def load_prosumer_dataset(cfg: Cfg, split: str, *, pad_history_steps: int = 0, pad_future_steps: int = 0) -> SeriesData:
    n = int(cfg.env.num_agents)
    data_root = Path(cfg.data.data_root)
    root = (data_root if data_root.is_absolute() else PROJECT_ROOT / data_root) / "prosumer"
    year = cfg.data.share_data_train_test[0] if split == "train" else cfg.data.share_data_train_test[1]
    canonical_start = cfg.data.train_start_date if split == "train" else cfg.data.eval_start_date
    canonical_end = cfg.data.train_end_date if split == "train" else cfg.data.eval_end_date
    start = _shift_date(canonical_start, -int(pad_history_steps), cfg.env.dt_hours)
    end = _shift_date(canonical_end, int(pad_future_steps), cfg.env.dt_hours)
    load_total = None; base_ts = None; component_arrays = {}
    for component in cfg.data.load_components:
        frame = _read_time_frame(root / f"{component}.csv")
        mask = _date_mask(frame["timestamp"], year, start, end)
        part = frame.loc[mask, list(cfg.data.agent_profiles)].to_numpy(dtype=np.float32)
        component_arrays[component] = part * _scale(cfg.data.load_scale, n, "load_scale")[None, :]
        load_total = part if load_total is None else load_total + part
        ts = frame.loc[mask, "timestamp"].reset_index(drop=True)
        if base_ts is None: base_ts = ts
        else:
            pass
    price_frame = _read_time_frame(root / "price.csv")
    pv_frame = _read_time_frame(root / "pv_reference.csv")
    price = price_frame.loc[_date_mask(price_frame["timestamp"], year, start, end), ["timestamp", "price"]].reset_index(drop=True)
    pv_col = f"ref_{cfg.data.pv_reference}"
    pv = pv_frame.loc[_date_mask(pv_frame["timestamp"], year, start, end), ["timestamp", pv_col]].reset_index(drop=True)
    load = load_total * _scale(cfg.data.load_scale, n, "load_scale")[None, :]
    pv_ref = pv[pv_col].to_numpy(dtype=np.float32)
    peak = float(np.max(pv_ref))
    pv_scale = _scale(cfg.data.pv_scale, n, "pv_scale")
    capacity = np.asarray(cfg.data.pv_capacity_kw, dtype=np.float32).reshape(-1)
    if capacity.size:
        if peak <= 0.0:
            raise ValueError("pv_reference peak must be positive when pv_capacity_kw is provided.")
        if capacity.size != n:
            raise ValueError(f"pv_capacity_kw expected {n} values, got {capacity.size}.")
        pv_values = pv_ref[:, None] * (capacity / np.float32(peak))[None, :] * pv_scale[None, :]
    else:
        pv_values = np.repeat(pv_ref[:, None], n, axis=1) * pv_scale[None, :]
    timestamps = [ts.isoformat() for ts in base_ts]
    canonical_dates = base_ts.dt.date
    target_mask = canonical_dates.between(pd.Timestamp(canonical_start).date(), pd.Timestamp(canonical_end).date()).to_numpy(dtype=bool)
    target_indices = np.flatnonzero(target_mask)
    return SeriesData(timestamps=timestamps, price=price["price"].to_numpy(dtype=np.float32), load=load.astype(np.float32), load_components=component_arrays, pv=pv_values.astype(np.float32), target_start=int(target_indices[0]), target_length=int(target_indices.shape[0]))
