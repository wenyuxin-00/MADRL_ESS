"""Reusable observation feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_time_features(cur_step: int, episode_length: int, n_agents: int) -> np.ndarray:
    """Encode episode progress with one sin/cos pair."""
    sin_value = float(np.sin(2.0 * np.pi * cur_step / episode_length))
    cos_value = float(np.cos(2.0 * np.pi * cur_step / episode_length))
    out = np.empty((n_agents, 2), dtype=np.float32)
    out[:, 0] = sin_value
    out[:, 1] = cos_value
    return out


def build_calendar_time_features(timestamp_value: str, n_agents: int) -> np.ndarray:
    """Encode absolute calendar time with hour-of-day and day-of-year cycles."""
    timestamp = pd.Timestamp(timestamp_value)
    hour_of_day = float(timestamp.hour) + float(timestamp.minute) / 60.0
    day_of_year = float(timestamp.dayofyear - 1) + hour_of_day / 24.0

    hour_phase = 2.0 * np.pi * hour_of_day / 24.0
    year_phase = 2.0 * np.pi * day_of_year / 365.25

    out = np.empty((n_agents, 4), dtype=np.float32)
    out[:, 0] = np.sin(hour_phase)
    out[:, 1] = np.cos(hour_phase)
    out[:, 2] = np.sin(year_phase)
    out[:, 3] = np.cos(year_phase)
    return out


def broadcast_scalar_feature(value: float, n_agents: int) -> np.ndarray:
    return np.full((n_agents, 1), float(value), dtype=np.float32)


def reshape_agent_scalar_feature(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def pad_sequence_1d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if start >= x.shape[0]:
        return np.zeros((length,), dtype=np.float32)

    chunk = x[start : start + length]
    if chunk.shape[0] < length:
        chunk = np.concatenate([chunk, np.zeros(length - chunk.shape[0], dtype=np.float32)], axis=0)
    return chunk.astype(np.float32)


def pad_sequence_2d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if start >= x.shape[0]:
        return np.zeros((length, x.shape[1]), dtype=np.float32)

    chunk = x[start : start + length]
    if chunk.shape[0] < length:
        pad = np.zeros((length - chunk.shape[0], x.shape[1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def build_adjacency(n_agents: int, adjacency_type: str) -> np.ndarray:
    if adjacency_type == "identity":
        return np.eye(n_agents, dtype=np.float32)
    if adjacency_type == "fully_connected_no_self":
        adjacency = np.ones((n_agents, n_agents), dtype=np.float32)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency
    raise ValueError(
        f"Unknown adjacency_type '{adjacency_type}', expected 'identity' or 'fully_connected_no_self'."
    )
