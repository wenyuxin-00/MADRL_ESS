"""Observation normalization helpers for train-frozen statistics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

EPSILON = 1e-6
PV_CAPACITY_CLIP_MAX = 1.2


def _fit_robust_stats(values: np.ndarray, *, low_quantile: float, high_quantile: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        axis = None
    elif array.ndim == 2:
        axis = 0
    else:
        raise ValueError(f"Expected 1D or 2D values for robust stats, got shape {array.shape}")

    q_low = np.quantile(array, float(low_quantile), axis=axis)
    q_high = np.quantile(array, float(high_quantile), axis=axis)
    median = np.quantile(array, 0.5, axis=axis)
    q25 = np.quantile(array, 0.25, axis=axis)
    q75 = np.quantile(array, 0.75, axis=axis)
    iqr = np.maximum(q75 - q25, np.float32(EPSILON))
    return {
        "q_low": np.asarray(q_low, dtype=np.float32),
        "q_high": np.asarray(q_high, dtype=np.float32),
        "median": np.asarray(median, dtype=np.float32),
        "iqr": np.asarray(iqr, dtype=np.float32),
    }


def _reshape_per_agent_vector(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.shape[0] != vector.shape[0]:
        raise ValueError(
            f"Per-agent normalization vector should have {target.shape[0]} entries, got {vector.shape[0]}"
        )
    return vector.reshape((target.shape[0],) + (1,) * max(0, target.ndim - 1))


def _apply_robust_tanh(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    q_low = np.asarray(spec["q_low"], dtype=np.float32)
    q_high = np.asarray(spec["q_high"], dtype=np.float32)
    median = np.asarray(spec["median"], dtype=np.float32)
    iqr = np.asarray(spec["iqr"], dtype=np.float32)

    if str(spec.get("scope", "shared")) == "per_agent":
        q_low = _reshape_per_agent_vector(q_low, array)
        q_high = _reshape_per_agent_vector(q_high, array)
        median = _reshape_per_agent_vector(median, array)
        iqr = _reshape_per_agent_vector(iqr, array)

    clipped = np.clip(array, q_low, q_high)
    z_score = (clipped - median) / np.maximum(iqr, np.float32(EPSILON))
    tanh_scale = max(float(spec.get("tanh_scale", 1.0)), EPSILON)
    return np.tanh(z_score / tanh_scale).astype(np.float32)


def _apply_capacity_scale(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    denominator = np.asarray(spec["denominator"], dtype=np.float32)
    denominator = _reshape_per_agent_vector(np.maximum(denominator, np.float32(EPSILON)), array)
    return np.clip(array / denominator, 0.0, float(spec.get("clip_max", PV_CAPACITY_CLIP_MAX))).astype(np.float32)


def _apply_linear_pm1(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    lower = float(spec["lower"])
    upper = float(spec["upper"])
    scale = max(upper - lower, EPSILON)
    normalized = 2.0 * (array - lower) / scale - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _extract_signal_values(dataset, signal_name: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for episode_idx in range(dataset.num_episodes()):
        episode = dataset.get_episode(episode_idx)
        chunks.append(np.asarray(episode["signals"][signal_name], dtype=np.float32))
    if not chunks:
        raise ValueError(f"Dataset does not contain any episodes for signal '{signal_name}'.")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _extract_pv_denominator(dataset) -> np.ndarray:
    first_episode = dataset.get_episode(0)
    meta = dict(first_episode.get("meta", {}))
    pv_peak_kw = np.asarray(meta.get("pv_peak_kw", []), dtype=np.float32).reshape(-1)
    if pv_peak_kw.size == 0 or np.any(pv_peak_kw <= 0.0):
        pv_values = _extract_signal_values(dataset, "pv")
        pv_peak_kw = np.max(pv_values, axis=0).astype(np.float32)
    return np.maximum(pv_peak_kw, np.float32(EPSILON)).astype(np.float32)


def _normalization_signature(cfg) -> dict[str, object]:
    return {
        "agent_profiles": list(cfg.data.agent_profiles),
        "train_year": int(cfg.data.train_year),
        "train_start_date": cfg.data.train_start_date,
        "train_end_date": cfg.data.train_end_date,
        "test_year": int(cfg.data.test_year),
        "test_start_date": cfg.data.test_start_date,
        "test_end_date": cfg.data.test_end_date,
        "load_components": list(cfg.data.load_components),
        "pv_reference": str(cfg.data.pv_reference),
        "pv_capacity_kw": list(cfg.data.pv_capacity_kw),
        "load_scale": list(cfg.data.load_scale),
        "pv_scale": list(cfg.data.pv_scale),
        "num_agents": int(cfg.env.num_agents),
        "local_features": list(cfg.obs.local_features),
        "sequence_features": list(cfg.obs.sequence_features),
        "price_normalization": str(cfg.obs.price_normalization),
        "load_normalization": str(cfg.obs.load_normalization),
        "pv_normalization": str(cfg.obs.pv_normalization),
        "soc_normalization": str(cfg.obs.soc_normalization),
        "clip_low": float(cfg.obs.normalization_clip_low_quantile),
        "clip_high": float(cfg.obs.normalization_clip_high_quantile),
        "price_tanh_scale": float(cfg.obs.price_tanh_scale),
        "load_tanh_scale": float(cfg.obs.load_tanh_scale),
        "pv_tanh_scale": float(cfg.obs.pv_tanh_scale),
    }


def fit_observation_normalization_state(cfg, dataset) -> dict[str, object]:
    low_quantile = float(cfg.obs.normalization_clip_low_quantile)
    high_quantile = float(cfg.obs.normalization_clip_high_quantile)

    price_values = _extract_signal_values(dataset, "price")
    load_values = _extract_signal_values(dataset, "load")
    pv_values = _extract_signal_values(dataset, "pv")
    pv_denominator = _extract_pv_denominator(dataset)

    return {
        "signature": _normalization_signature(cfg),
        "price": {
            "method": str(cfg.obs.price_normalization),
            "scope": "shared",
            "tanh_scale": float(cfg.obs.price_tanh_scale),
            **_fit_robust_stats(price_values, low_quantile=low_quantile, high_quantile=high_quantile),
        },
        "load": {
            "method": str(cfg.obs.load_normalization),
            "scope": "per_agent",
            "tanh_scale": float(cfg.obs.load_tanh_scale),
            **_fit_robust_stats(load_values, low_quantile=low_quantile, high_quantile=high_quantile),
        },
        "pv": {
            "method": str(cfg.obs.pv_normalization),
            "scope": "per_agent",
            "tanh_scale": float(cfg.obs.pv_tanh_scale),
            "denominator": pv_denominator.astype(np.float32),
            "clip_max": PV_CAPACITY_CLIP_MAX,
            **_fit_robust_stats(pv_values, low_quantile=low_quantile, high_quantile=high_quantile),
        },
        "soc": {
            "method": str(cfg.obs.soc_normalization),
            "scope": "per_agent",
            "lower": float(cfg.env.soc_min),
            "upper": float(cfg.env.soc_max),
        },
    }


class ObservationNormalizer:
    """Transform observation fields with statistics fit on the training split."""

    def __init__(self, state: dict[str, object]):
        self.state = deepcopy(state)

    def transform_local(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        return self._transform(feature_name, values)

    def transform_sequence(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        return self._transform(feature_name, values)

    def describe(self, feature_name: str) -> dict[str, object] | None:
        spec = self.state.get(str(feature_name))
        if spec is None:
            return None
        return {
            key: value
            for key, value in dict(spec).items()
            if key in {"method", "scope", "tanh_scale", "clip_max", "lower", "upper"}
        }

    def _transform(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        spec = self.state.get(str(feature_name))
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
        raise ValueError(f"Unsupported observation normalization method '{method}' for feature '{feature_name}'.")


def build_observation_normalizer(cfg) -> ObservationNormalizer | None:
    if not bool(getattr(cfg.obs, "normalization_enabled", False)):
        return None

    signature = _normalization_signature(cfg)
    cached_state = getattr(cfg.runtime, "observation_normalization_state", None)
    if isinstance(cached_state, dict) and cached_state.get("signature") == signature:
        return ObservationNormalizer(cached_state)

    from data.loaders.registry import build_dataset

    train_dataset = build_dataset(cfg, mode="train")
    state = fit_observation_normalization_state(cfg, train_dataset)
    cfg.runtime.observation_normalization_state = deepcopy(state)
    return ObservationNormalizer(state)


__all__ = [
    "ObservationNormalizer",
    "build_observation_normalizer",
    "fit_observation_normalization_state",
]
