from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


EPSILON = 1e-6
PV_CAPACITY_CLIP_MAX = 1.2


# 作用：从训练数据中拟合 robust_tanh 归一化所需的分位数统计量。
def _fit_robust_stats(values: np.ndarray, *, low_quantile: float, high_quantile: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim not in {1, 2}:
        raise ValueError(f"Expected 1D or 2D values for robust stats, got shape {array.shape}")

    axis = None if array.ndim == 1 else 0
    q_low, q_high, median, q25, q75 = np.quantile(array, [float(low_quantile), float(high_quantile), 0.5, 0.25, 0.75], axis=axis)
    iqr = np.maximum(q75 - q25, np.float32(EPSILON))
    return {"q_low": np.asarray(q_low, dtype=np.float32), "q_high": np.asarray(q_high, dtype=np.float32), "median": np.asarray(median, dtype=np.float32), "iqr": np.asarray(iqr, dtype=np.float32)}


# 作用：把每个 agent 一个值的归一化参数 reshape 到目标特征的广播形状。
def _reshape_per_agent_vector(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.shape[0] != vector.shape[0]:
        raise ValueError(f"Per-agent normalization vector should have {target.shape[0]} entries, got {vector.shape[0]}")

    return vector.reshape((target.shape[0],) + (1,) * max(0, target.ndim - 1))


# 作用：用训练集分位数裁剪并通过 tanh 得到稳定的归一化特征。
def _apply_robust_tanh(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    q_low, q_high, median, iqr = (np.asarray(spec[key], dtype=np.float32) for key in ("q_low", "q_high", "median", "iqr"))

    if str(spec.get("scope", "shared")) == "per_agent":
        q_low = _reshape_per_agent_vector(q_low, array)
        q_high = _reshape_per_agent_vector(q_high, array)
        median = _reshape_per_agent_vector(median, array)
        iqr = _reshape_per_agent_vector(iqr, array)

    clipped = np.clip(array, q_low, q_high)
    z_score = (clipped - median) / np.maximum(iqr, np.float32(EPSILON))
    tanh_scale = max(float(spec.get("tanh_scale", 1.0)), EPSILON)
    return np.tanh(z_score / tanh_scale).astype(np.float32)


# 作用：按每个 agent 的 PV 容量把 PV 功率缩放到容量比例。
def _apply_capacity_scale(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    denominator = np.asarray(spec["denominator"], dtype=np.float32)
    denominator = _reshape_per_agent_vector(np.maximum(denominator, np.float32(EPSILON)), array)
    return np.clip(array / denominator, 0.0, float(spec.get("clip_max", PV_CAPACITY_CLIP_MAX))).astype(np.float32)


# 作用：把有上下界的特征线性缩放到 -1 到 1。
def _apply_linear_pm1(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    lower = float(spec["lower"])
    upper = float(spec["upper"])
    scale = max(upper - lower, EPSILON)
    normalized = 2.0 * (array - lower) / scale - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


# 作用：从 dataset 中取出某个信号的完整训练样本，用于拟合归一化统计量。
def _extract_signal_values(dataset, signal_name: str) -> np.ndarray:
    getter = getattr(dataset, "get_normalization_signal_values", None)
    if not callable(getter):
        raise TypeError(f"Expected dataset with callable get_normalization_signal_values(signal_name), got {type(dataset).__name__}")

    values = np.asarray(getter(signal_name), dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"Dataset does not contain any normalization rows for signal '{signal_name}'.")
    return values.astype(np.float32, copy=False)


# 作用：确定每个 agent 的 PV 容量归一化分母。
def _extract_pv_denominator(dataset) -> np.ndarray:
    pv_peak_kw = np.asarray(getattr(dataset, "_meta_template", {}).get("pv_peak_kw", []), dtype=np.float32).reshape(-1)

    if pv_peak_kw.size == 0 or np.any(pv_peak_kw <= 0.0):
        pv_peak_kw = np.max(_extract_signal_values(dataset, "pv"), axis=0).astype(np.float32)

    return np.maximum(pv_peak_kw, np.float32(EPSILON)).astype(np.float32)


# 作用：生成归一化缓存签名，确保缓存只复用于相同数据和 observation 配置。
def _normalization_signature(cfg) -> dict[str, object]:
    data = cfg.data
    env = cfg.env
    obs = cfg.obs
    train_episode_limit = getattr(env, "resolved_train_episode_limit", lambda: int(env.episode_limit))()
    test_episode_limit = getattr(env, "resolved_test_episode_limit", lambda: int(env.episode_limit))()
    return {
        "agent_profiles": list(data.agent_profiles),
        "train_year": int(data.train_year),
        "train_start_date": data.train_start_date,
        "train_end_date": data.train_end_date,
        "test_year": int(data.test_year),
        "test_start_date": data.test_start_date,
        "test_end_date": data.test_end_date,
        "load_components": list(data.load_components),
        "pv_reference": str(data.pv_reference),
        "pv_capacity_kw": list(data.pv_capacity_kw),
        "load_scale": list(data.load_scale),
        "pv_scale": list(data.pv_scale),
        "num_agents": int(env.num_agents),
        "base_episode_limit": int(env.episode_limit),
        "train_window_days": int(getattr(env, "train_window_days", 1)),
        "window_stride_days": int(getattr(env, "window_stride_days", 1)),
        "train_episode_limit": int(train_episode_limit),
        "test_episode_limit": int(test_episode_limit),
        "local_features": list(obs.local_features),
        "sequence_features": list(obs.sequence_features),
        "wholesale_price_normalization": str(obs.wholesale_price_normalization),
        "load_normalization": str(obs.load_normalization),
        "pv_normalization": str(obs.pv_normalization),
        "soc_normalization": str(obs.soc_normalization),
        "clip_low": float(obs.normalization_clip_low_quantile),
        "clip_high": float(obs.normalization_clip_high_quantile),
        "wholesale_price_tanh_scale": float(obs.wholesale_price_tanh_scale),
        "wholesale_price_spread_scale_eur_per_kwh": float(getattr(obs, "wholesale_price_spread_scale_eur_per_kwh", 0.20)),
        "load_tanh_scale": float(obs.load_tanh_scale),
        "pv_tanh_scale": float(obs.pv_tanh_scale),
    }


# 作用：基于训练 dataset 拟合 observation 归一化状态。
def fit_observation_normalization_state(cfg, dataset) -> dict[str, object]:
    env = cfg.env
    obs = cfg.obs
    low_quantile = float(obs.normalization_clip_low_quantile)
    high_quantile = float(obs.normalization_clip_high_quantile)
    stats_kwargs = {"low_quantile": low_quantile, "high_quantile": high_quantile}
    wholesale_price_values = _extract_signal_values(dataset, "wholesale_price")
    load_values = _extract_signal_values(dataset, "load")
    pv_values = _extract_signal_values(dataset, "pv")
    pv_denominator = _extract_pv_denominator(dataset)

    return {
        "signature": _normalization_signature(cfg),
        "wholesale_price": {
            "method": str(obs.wholesale_price_normalization),
            "scope": "shared",
            "tanh_scale": float(obs.wholesale_price_tanh_scale),
            **_fit_robust_stats(wholesale_price_values, **stats_kwargs),
        },
        "load": {
            "method": str(obs.load_normalization),
            "scope": "per_agent",
            "tanh_scale": float(obs.load_tanh_scale),
            **_fit_robust_stats(load_values, **stats_kwargs),
        },
        "pv": {
            "method": str(obs.pv_normalization),
            "scope": "per_agent",
            "tanh_scale": float(obs.pv_tanh_scale),
            "denominator": pv_denominator.astype(np.float32),
            "clip_max": PV_CAPACITY_CLIP_MAX,
            **_fit_robust_stats(pv_values, **stats_kwargs),
        },
        "soc": {
            "method": str(obs.soc_normalization),
            "scope": "per_agent",
            "lower": float(env.soc_min),
            "upper": float(env.soc_max),
        },
    }


# 作用：持有已拟合的归一化状态，并按特征名转换 observation 数值。
class ObservationNormalizer:
    # 作用：复制归一化状态，避免外部继续修改缓存对象。
    def __init__(self, state: dict[str, object]):
        self.state = deepcopy(state)

    # 作用：根据单个特征的 method 选择对应归一化算法。
    def _transform(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        spec = self.state.get(str(feature_name), {})
        if spec is None:
            return np.asarray(values, dtype=np.float32)

        method = str(spec.get("method", "none")).strip().lower()
        if method in {"none", "identity"}:
            return np.asarray(values, dtype=np.float32)
        if method == "robust_tanh":
            return _apply_robust_tanh(values, spec)
        if method == "capacity":
            return _apply_capacity_scale(values, spec)
        if method == "linear_pm1":
            return _apply_linear_pm1(values, spec)

        raise ValueError(
            f"Unsupported observation normalization method '{method}' for "
            f"feature '{feature_name}'."
        )

    # 作用：本地特征和序列特征共用同一套按特征名归一化逻辑。
    transform_local = _transform
    transform_sequence = _transform


# 作用：从配置和 dataset 构造 observation normalizer，并复用匹配签名的缓存。
def build_observation_normalizer(cfg, dataset=None) -> ObservationNormalizer | None:
    if not bool(getattr(cfg.obs, "normalization_enabled", False)):
        return None

    signature = _normalization_signature(cfg)
    cached_state = getattr(cfg.runtime, "observation_normalization_state", None)
    if isinstance(cached_state, dict) and cached_state.get("signature") == signature:
        return ObservationNormalizer(cached_state)

    if dataset is None:
        from data.loaders.registry import build_dataset

        dataset = build_dataset(cfg, mode="train")

    state = fit_observation_normalization_state(cfg, dataset)
    cfg.runtime.observation_normalization_state = deepcopy(state)
    return ObservationNormalizer(state)
