"""观测字段构造底层工具函数。

提供观测构造过程中复用的基础构建块，包括时间编码、
标量广播、序列截取补零、邻接矩阵构造等。

主要函数:
    build_time_features          -- 构造周期时间编码 (sin/cos)
    broadcast_scalar_feature     -- 将共享标量广播为 (n_agents, 1)
    reshape_agent_scalar_feature -- 将逐智能体标量重塑为 (n_agents, 1)
    pad_sequence_1d              -- 一维序列截取与尾部补零
    pad_sequence_2d              -- 二维序列截取与尾部补零
    build_adjacency              -- 构造邻接矩阵
"""

from __future__ import annotations

import numpy as np


def build_time_features(cur_step: int, episode_length: int, n_agents: int) -> np.ndarray:
    """构造共享的周期时间编码（sin/cos 对）。

    将当前步数映射到 [0, 2*pi) 的周期位置，生成 sin 和 cos 值，
    广播到所有智能体。

    参数:
        cur_step: 当前步数
        episode_length: episode 总步数（决定周期长度）
        n_agents: 智能体数量

    返回:
        np.ndarray: shape (n_agents, 2)，第 0 列为 sin，第 1 列为 cos
    """
    # 计算当前步在周期中的 sin/cos 编码
    sin_value = float(np.sin(2.0 * np.pi * cur_step / episode_length))
    cos_value = float(np.cos(2.0 * np.pi * cur_step / episode_length))
    out = np.empty((n_agents, 2), dtype=np.float32)
    out[:, 0] = sin_value
    out[:, 1] = cos_value
    return out


def broadcast_scalar_feature(value: float, n_agents: int) -> np.ndarray:
    """将共享标量广播为所有智能体共用的列向量。

    参数:
        value: 标量值（如当前电价）
        n_agents: 智能体数量

    返回:
        np.ndarray: shape (n_agents, 1)，每行相同
    """
    return np.full((n_agents, 1), float(value), dtype=np.float32)


def reshape_agent_scalar_feature(values: np.ndarray) -> np.ndarray:
    """将逐智能体的标量数组重塑为 (n_agents, 1)。

    参数:
        values: 各智能体的标量值，shape (n_agents,) 或兼容形状

    返回:
        np.ndarray: shape (n_agents, 1)
    """
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def pad_sequence_1d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """从一维数组中截取指定窗口，尾部不足时补零。

    参数:
        x: 源一维数组
        start: 截取起始索引
        length: 目标窗口长度

    返回:
        np.ndarray: shape (length,)，尾部不足部分填零
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    # 起始位置超出数组范围时直接返回全零
    if start >= x.shape[0]:
        return np.zeros((length,), dtype=np.float32)

    chunk = x[start : start + length]
    # 尾部不足时补零
    if chunk.shape[0] < length:
        chunk = np.concatenate([chunk, np.zeros(length - chunk.shape[0], dtype=np.float32)], axis=0)
    return chunk.astype(np.float32)


def pad_sequence_2d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """从二维数组中截取指定窗口，尾部不足时补零。

    参数:
        x: 源二维数组，shape (T, n_features)
        start: 截取起始索引（沿 axis=0）
        length: 目标窗口长度

    返回:
        np.ndarray: shape (length, n_features)，尾部不足部分填零
    """
    x = np.asarray(x, dtype=np.float32)
    # 起始位置超出数组范围时直接返回全零
    if start >= x.shape[0]:
        return np.zeros((length, x.shape[1]), dtype=np.float32)

    chunk = x[start : start + length]
    # 尾部不足时补零
    if chunk.shape[0] < length:
        pad = np.zeros((length - chunk.shape[0], x.shape[1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def build_adjacency(n_agents: int, adjacency_type: str) -> np.ndarray:
    """构造图模型使用的邻接矩阵。

    参数:
        n_agents: 智能体数量
        adjacency_type: 邻接矩阵类型，支持:
            - "identity": 单位矩阵（无智能体间连接）
            - "fully_connected_no_self": 全连接无自环

    返回:
        np.ndarray: shape (n_agents, n_agents) 的邻接矩阵

    异常:
        ValueError: 当 adjacency_type 为未知类型时抛出
    """
    if adjacency_type == "identity":
        # 单位矩阵：每个智能体只与自身相连
        return np.eye(n_agents, dtype=np.float32)
    if adjacency_type == "fully_connected_no_self":
        # 全连接无自环：所有智能体两两相连，对角线为 0
        adjacency = np.ones((n_agents, n_agents), dtype=np.float32)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency
    raise ValueError(
        f"Unknown adjacency_type '{adjacency_type}', expected 'identity' or 'fully_connected_no_self'."
    )
