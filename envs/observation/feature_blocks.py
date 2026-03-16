"""
envs/observation/feature_blocks.py
职责：定义各特征块（Feature Block），每个块负责观测中的一段特征。

当前支持的 block：
  - TimeBlock   : 时间编码 (sin, cos)，维度 2
  - PriceBlock  : 未来价格窗口，通过 env.forecaster 获取，维度 K+1
  - LoadBlock   : 未来负荷窗口（使用真实值），每 agent 独立，维度 K+1
  - SoCBlock    : 当前 SoC（每 agent 独立），维度 1

Block 接口：
  - dim() -> int           该块对单个 agent 的贡献维度
  - build(env) -> ndarray  返回 shape (N, dim()) 的特征矩阵

扩展方法：新增特征只需添加新 Block 类，并在 DEFAULT_BLOCK_REGISTRY 中注册。
"""

import numpy as np


def _pad_2d(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """从 2D 数组 x (T, N) 取 [start, start+length)，末尾零填充。"""
    if start >= x.shape[0]:
        return np.zeros((length, x.shape[1]), dtype=np.float32)
    chunk = x[start:start + length, :]
    if chunk.shape[0] < length:
        pad = np.zeros((length - chunk.shape[0], x.shape[1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


# ------------------------------------------------------------------
# Feature Blocks
# ------------------------------------------------------------------

class TimeBlock:
    """时间编码：当前步在 episode 中的归一化位置（sin, cos）。所有 agent 共享。"""

    def dim(self) -> int:
        return 2

    def build(self, env) -> np.ndarray:
        t, T, N = env.cur_step, env.episode_length, env.n
        sin_val = float(np.sin(2.0 * np.pi * t / T))
        cos_val = float(np.cos(2.0 * np.pi * t / T))
        result = np.empty((N, 2), dtype=np.float32)
        result[:, 0] = sin_val
        result[:, 1] = cos_val
        return result


class PriceBlock:
    """当前步 + 未来 K 步价格窗口，通过 env.forecaster 获取预测值。所有 agent 共享。

    - forecaster_type="perfect" → 真实未来价格（行为与旧 env 完全一致）
    - forecaster_type="naive"   → 历史滚动均值填充
    - forecaster_type="lstm"    → LSTM 模型预测
    """

    def __init__(self, future_horizon: int):
        self._k1 = future_horizon + 1

    def dim(self) -> int:
        return self._k1

    def build(self, env) -> np.ndarray:
        t = env.cur_step
        # history = ep_price[0..t]（含当前步）
        history = env.ep_price[:t + 1]
        price_win = env.forecaster.predict(history, self._k1)  # (K+1,)
        return np.tile(price_win[None, :], (env.n, 1))          # (N, K+1)


class LoadBlock:
    """未来 K+1 步负荷窗口（使用真实 schedule，每 agent 独立）。"""

    def __init__(self, future_horizon: int):
        self._k1 = future_horizon + 1

    def dim(self) -> int:
        return self._k1

    def build(self, env) -> np.ndarray:
        # ep_load shape: (T, N)
        load_win = _pad_2d(env.ep_load, env.cur_step, self._k1)  # (K+1, N)
        return load_win.T.astype(np.float32)                       # (N, K+1)


class SoCBlock:
    """当前 SoC（每 agent 独立），维度 1。"""

    def dim(self) -> int:
        return 1

    def build(self, env) -> np.ndarray:
        return env.soc.reshape(-1, 1).astype(np.float32)  # (N, 1)


# ------------------------------------------------------------------
# 注册表：供 DefaultObservationBuilder 使用
# ------------------------------------------------------------------

def _make_time(_fh):    return TimeBlock()
def _make_price(fh):    return PriceBlock(fh)
def _make_load(fh):     return LoadBlock(fh)
def _make_soc(_fh):     return SoCBlock()

DEFAULT_BLOCK_REGISTRY: dict = {
    "time":  _make_time,
    "price": _make_price,
    "load":  _make_load,
    "soc":   _make_soc,
}
