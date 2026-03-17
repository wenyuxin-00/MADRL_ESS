"""观测字段构造中会复用的底层小工具。"""

from __future__ import annotations

import numpy as np


def build_time_features(cur_step: int, episode_length: int, n_agents: int) -> np.ndarray:
    """构造共享的周期时间编码。"""
    sin_value = float(np.sin(2.0 * np.pi * cur_step / episode_length))
    cos_value = float(np.cos(2.0 * np.pi * cur_step / episode_length))
    out = np.empty((n_agents, 2), dtype=np.float32)
    out[:, 0] = sin_value
    out[:, 1] = cos_value
    return out


def broadcast_scalar_feature(value: float, n_agents: int) -> np.ndarray:
    """把共享标量广播成每个 agent 一列。"""
    return np.full((n_agents, 1), float(value), dtype=np.float32)


def reshape_agent_scalar_feature(values: np.ndarray) -> np.ndarray:
    """把每个 agent 的标量改成 `(n_agents, 1)`。"""
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def pad_sequence_1d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """截取 `x[start:start+length]`，尾部不足时补零。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if start >= x.shape[0]:
        return np.zeros((length,), dtype=np.float32)

    chunk = x[start : start + length]
    if chunk.shape[0] < length:
        chunk = np.concatenate([chunk, np.zeros(length - chunk.shape[0], dtype=np.float32)], axis=0)
    return chunk.astype(np.float32)


def pad_sequence_2d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """截取 `x[start:start+length]`，尾部不足时补零。"""
    x = np.asarray(x, dtype=np.float32)
    if start >= x.shape[0]:
        return np.zeros((length, x.shape[1]), dtype=np.float32)

    chunk = x[start : start + length]
    if chunk.shape[0] < length:
        pad = np.zeros((length - chunk.shape[0], x.shape[1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def build_adjacency(n_agents: int, adjacency_type: str) -> np.ndarray:
    """构造图模型使用的邻接矩阵。"""
    if adjacency_type == "identity":
        return np.eye(n_agents, dtype=np.float32)
    if adjacency_type == "fully_connected_no_self":
        adjacency = np.ones((n_agents, n_agents), dtype=np.float32)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency
    raise ValueError(
        f"Unknown adjacency_type '{adjacency_type}', expected 'identity' or 'fully_connected_no_self'."
    )
