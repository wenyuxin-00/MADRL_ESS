from __future__ import annotations
import re
from typing import Sequence
import numpy as np
import pandas as pd
TIME_FEATURE_MODE_NONE = 'none'
TIME_FEATURE_MODE_HOUR_WEEK_YEAR = 'hour_week_year'
DEFAULT_FORECAST_STEP = pd.Timedelta(minutes=15)
DEFAULT_LOCAL_TIMEZONE = 'Europe/Berlin'
_TZ_SUFFIX_PATTERN = re.compile('(Z|[+-]\\d{2}:?\\d{2})$')

def normalize_time_feature_mode(mode: str | None) -> str:
    normalized = str(mode or TIME_FEATURE_MODE_NONE).strip().lower()
    if normalized not in {TIME_FEATURE_MODE_NONE, TIME_FEATURE_MODE_HOUR_WEEK_YEAR}:
        raise ValueError(f"Unsupported forecast time_feature_mode '{mode}'.")
    return normalized

def time_feature_dim(mode: str | None) -> int:
    normalized = normalize_time_feature_mode(mode)
    if normalized == TIME_FEATURE_MODE_NONE:
        return 0
    return 6

def coerce_timestamp_index(timestamps: Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    if timestamps is None:
        return pd.DatetimeIndex([])
    if isinstance(timestamps, pd.DatetimeIndex):
        return timestamps
    values = list(timestamps)
    if not values:
        return pd.DatetimeIndex([])
    requires_utc = False
    for value in values:
        if isinstance(value, str):
            if _TZ_SUFFIX_PATTERN.search(value.strip()):
                requires_utc = True
                break
            continue
        if getattr(value, 'tzinfo', None) is not None:
            requires_utc = True
            break
    parsed = pd.to_datetime(values, utc=requires_utc)
    return pd.DatetimeIndex(parsed)

def infer_timestamp_step(timestamps: Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None) -> pd.Timedelta:
    index = coerce_timestamp_index(timestamps)
    if len(index) < 2:
        return DEFAULT_FORECAST_STEP
    deltas_ns = np.diff(index.asi8)
    deltas_ns = deltas_ns[deltas_ns > 0]
    if deltas_ns.size == 0:
        return DEFAULT_FORECAST_STEP
    return pd.to_timedelta(int(np.median(deltas_ns)), unit='ns')

def pad_history_timestamps_left(timestamps: Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None, total_length: int, *, step: pd.Timedelta | None=None) -> pd.DatetimeIndex:
    total_length = int(total_length)
    if total_length <= 0:
        return pd.DatetimeIndex([])
    index = coerce_timestamp_index(timestamps)
    step = DEFAULT_FORECAST_STEP if step is None else pd.to_timedelta(step)
    if len(index) >= total_length:
        return index[-total_length:]
    if len(index) == 0:
        return pd.date_range(start=pd.Timestamp('2000-01-01 00:00:00+00:00'), periods=total_length, freq=step)
    missing = total_length - len(index)
    prefix = pd.date_range(end=index[0] - step, periods=missing, freq=step)
    return prefix.append(index)

def encode_forecast_time_features(timestamps: Sequence[str | pd.Timestamp] | pd.DatetimeIndex, mode: str | None) -> np.ndarray:
    normalized_mode = normalize_time_feature_mode(mode)
    index = coerce_timestamp_index(timestamps)
    if getattr(index, 'tz', None) is not None:
        index = index.tz_convert(DEFAULT_LOCAL_TIMEZONE)
    if normalized_mode == TIME_FEATURE_MODE_NONE:
        return np.zeros((len(index), 0), dtype=np.float32)
    if len(index) == 0:
        return np.zeros((0, time_feature_dim(normalized_mode)), dtype=np.float32)
    hour_of_day = index.hour.to_numpy(dtype=np.float32) + index.minute.to_numpy(dtype=np.float32) / np.float32(60.0) + index.second.to_numpy(dtype=np.float32) / np.float32(3600.0)
    day_of_week = index.dayofweek.to_numpy(dtype=np.float32) + hour_of_day / np.float32(24.0)
    day_of_year = index.dayofyear.to_numpy(dtype=np.float32) - np.float32(1.0) + hour_of_day / np.float32(24.0)
    hour_phase = np.float32(2.0 * np.pi) * hour_of_day / np.float32(24.0)
    week_phase = np.float32(2.0 * np.pi) * day_of_week / np.float32(7.0)
    year_phase = np.float32(2.0 * np.pi) * day_of_year / np.float32(365.25)
    out = np.empty((len(index), 6), dtype=np.float32)
    out[:, 0] = np.sin(hour_phase).astype(np.float32)
    out[:, 1] = np.cos(hour_phase).astype(np.float32)
    out[:, 2] = np.sin(week_phase).astype(np.float32)
    out[:, 3] = np.cos(week_phase).astype(np.float32)
    out[:, 4] = np.sin(year_phase).astype(np.float32)
    out[:, 5] = np.cos(year_phase).astype(np.float32)
    return out
