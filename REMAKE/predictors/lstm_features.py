from __future__ import annotations

from collections.abc import Sequence
import re
import numpy as np
import pandas as pd

TZ_LOCAL = "Europe/Berlin"
TIME_FEATURE_NONE = "none"
TIME_FEATURE_HOUR_WEEK_YEAR = "hour_week_year"
_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def time_feature_dim(mode: str) -> int:
    return 0 if str(mode) == TIME_FEATURE_NONE else 6


def coerce_timestamps(timestamps: Sequence[str | pd.Timestamp]) -> pd.DatetimeIndex:
    values = list(timestamps)
    has_tz = any(_TZ_SUFFIX.search(value.strip()) if isinstance(value, str) else getattr(value, "tzinfo", None) is not None for value in values)
    return pd.DatetimeIndex(pd.to_datetime(values, utc=has_tz))


def encode_time_features(timestamps: Sequence[str | pd.Timestamp], mode: str) -> np.ndarray:
    if str(mode) == TIME_FEATURE_NONE:
        return np.zeros((len(timestamps), 0), dtype=np.float32)
    index = coerce_timestamps(timestamps)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(TZ_LOCAL)
    hour = index.hour.to_numpy(dtype=np.float32) + index.minute.to_numpy(dtype=np.float32) / np.float32(60.0)
    week = index.dayofweek.to_numpy(dtype=np.float32) + hour / np.float32(24.0)
    year = index.dayofyear.to_numpy(dtype=np.float32) - np.float32(1.0) + hour / np.float32(24.0)
    return np.column_stack([
        np.sin(np.float32(2.0 * np.pi) * hour / np.float32(24.0)),
        np.cos(np.float32(2.0 * np.pi) * hour / np.float32(24.0)),
        np.sin(np.float32(2.0 * np.pi) * week / np.float32(7.0)),
        np.cos(np.float32(2.0 * np.pi) * week / np.float32(7.0)),
        np.sin(np.float32(2.0 * np.pi) * year / np.float32(365.25)),
        np.cos(np.float32(2.0 * np.pi) * year / np.float32(365.25)),
    ]).astype(np.float32)


def encode_time_feature_windows(timestamps: Sequence[Sequence[str | pd.Timestamp]], mode: str) -> np.ndarray:
    return np.stack([encode_time_features(items, mode) for items in timestamps]).astype(np.float32)
