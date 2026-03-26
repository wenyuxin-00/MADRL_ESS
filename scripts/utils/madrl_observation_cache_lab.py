"""Build and load exact precomputed observation caches for MADRL fast-lab."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from data.loaders.registry import build_dataset
from predictors.time_features import DEFAULT_LOCAL_TIMEZONE, coerce_timestamp_index
from predictors.lstm_forecaster import load_lstm_forecaster_artifacts
from predictors.registry import build_forecaster
from scripts.utils.project_paths import project_root


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_artifact_bundle(bundle) -> list[tuple[str, str | None, str | None]]:
    if isinstance(bundle, tuple) and len(bundle) == 3 and not any(
        isinstance(item, (tuple, list)) for item in bundle
    ):
        return [bundle]
    if isinstance(bundle, list):
        return [tuple(item) for item in bundle]
    raise TypeError(f"Unsupported forecast artifact bundle: {bundle!r}")


def _artifact_signature(forecast_ready: dict[str, object] | None) -> dict[str, object]:
    if not forecast_ready or not forecast_ready.get("artifacts"):
        return {"mode": "no_artifacts"}

    signature: dict[str, object] = {}
    for signal_name, bundle in sorted(dict(forecast_ready["artifacts"]).items()):
        items: list[dict[str, object]] = []
        for model_path, meta_path, scaler_path in _normalize_artifact_bundle(bundle):
            model = Path(model_path)
            meta, _ = load_lstm_forecaster_artifacts(model_path, meta_path, scaler_path)
            items.append(
                {
                    "model_path": str(model.resolve()),
                    "model_mtime_ns": model.stat().st_mtime_ns,
                    "signal_name": meta.get("signal_name"),
                    "future_horizon": int(meta.get("future_horizon", meta["pred_len"])),
                    "seq_len": int(meta["seq_len"]),
                    "pred_len": int(meta["pred_len"]),
                    "hidden_size": int(meta["hidden_size"]),
                    "num_layers": int(meta["num_layers"]),
                    "dropout": float(meta["dropout"]),
                    "input_size": int(meta.get("input_size", 1)),
                    "time_feature_mode": str(meta.get("time_feature_mode", "none")),
                    "model_mode": str(meta.get("model_mode", "shared")),
                    "normalization_mode": str(meta.get("normalization_mode", "none")),
                    "postprocess_mode": str(meta.get("postprocess_mode", "none")),
                    "baseline_mode": str(meta.get("baseline_mode", "none")),
                    "blend_weight": meta.get("blend_weight"),
                    "component": meta.get("component"),
                    "agent_index": meta.get("agent_index"),
                    "source_signature": dict(meta.get("source_signature", {})),
                }
            )
        signature[str(signal_name)] = items
    return signature


def _cache_signature(cfg, *, split: str, forecast_ready: dict[str, object] | None) -> dict[str, object]:
    return {
        "version": 1,
        "split": str(split),
        "forecast_type": str(cfg.forecast.type),
        "num_agents": int(cfg.env.num_agents),
        "episode_limit": int(cfg.env.episode_limit),
        "future_horizon": int(cfg.env.future_horizon),
        "local_features": list(cfg.obs.local_features),
        "sequence_features": list(cfg.obs.sequence_features),
        "agent_profiles": list(cfg.data.agent_profiles),
        "train_year": int(cfg.data.train_year),
        "test_year": int(cfg.data.test_year),
        "train_start_date": cfg.data.train_start_date,
        "train_end_date": cfg.data.train_end_date,
        "test_start_date": cfg.data.test_start_date,
        "test_end_date": cfg.data.test_end_date,
        "load_components": list(cfg.data.load_components),
        "pv_reference": str(cfg.data.pv_reference),
        "pv_capacity_kw": list(cfg.data.pv_capacity_kw),
        "load_scale": list(cfg.data.load_scale),
        "pv_scale": list(cfg.data.pv_scale),
        "adjacency_type": str(cfg.obs.adjacency_type),
        "artifacts": _artifact_signature(forecast_ready),
    }


def _signature_hash(signature: dict[str, object]) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _compute_future_mean_price(price: np.ndarray, future_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(price, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    horizon = int(max(future_horizon, 1))
    idx = np.arange(values.size, dtype=np.int64)
    csum = np.concatenate([[0.0], np.cumsum(values.astype(np.float64), dtype=np.float64)], axis=0)

    start = idx + 1
    end = np.minimum(idx + 1 + horizon, values.size)
    valid = start < values.size
    sums = csum[end] - csum[start]
    lengths = np.maximum(end - start, 1)
    mu_t = np.where(valid, sums / lengths, values[np.minimum(idx, values.size - 1)]).astype(np.float32)

    next_idx = np.minimum(idx + 1, values.size - 1)
    start_next = next_idx + 1
    end_next = np.minimum(next_idx + 1 + horizon, values.size)
    valid_next = start_next < values.size
    sums_next = csum[end_next] - csum[start_next]
    lengths_next = np.maximum(end_next - start_next, 1)
    mu_next = np.where(valid_next, sums_next / lengths_next, values[next_idx]).astype(np.float32)
    return mu_t.astype(np.float32, copy=False), mu_next.astype(np.float32, copy=False)


def _build_calendar_time_matrix(
    timestamps: list[str | pd.Timestamp] | np.ndarray | tuple[str | pd.Timestamp, ...],
    n_agents: int,
) -> np.ndarray:
    index = coerce_timestamp_index(timestamps)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(DEFAULT_LOCAL_TIMEZONE)

    hour_of_day = (
        index.hour.to_numpy(dtype=np.float32)
        + index.minute.to_numpy(dtype=np.float32) / np.float32(60.0)
    )
    day_of_year = (index.dayofyear.to_numpy(dtype=np.float32) - np.float32(1.0)) + hour_of_day / np.float32(24.0)

    hour_phase = np.float32(2.0 * np.pi) * hour_of_day / np.float32(24.0)
    year_phase = np.float32(2.0 * np.pi) * day_of_year / np.float32(365.25)

    per_step = np.empty((len(index), 4), dtype=np.float32)
    per_step[:, 0] = np.sin(hour_phase).astype(np.float32)
    per_step[:, 1] = np.cos(hour_phase).astype(np.float32)
    per_step[:, 2] = np.sin(year_phase).astype(np.float32)
    per_step[:, 3] = np.cos(year_phase).astype(np.float32)
    return np.broadcast_to(per_step[:, None, :], (len(index), int(n_agents), 4)).astype(np.float32, copy=False)


def _default_cache_root(root: str | Path | None = None) -> Path:
    base = Path(root).resolve() if root is not None else project_root() / "artifacts" / "training" / "cache"
    return (base / "fastlab" / "observation").resolve()


def _required_array_names(cfg) -> list[str]:
    names: list[str] = []
    if "calendar_time" in cfg.obs.local_features:
        names.append("calendar_time")
    if "price" in cfg.obs.sequence_features:
        names.append("price_seq")
    if "load" in cfg.obs.sequence_features:
        names.append("load_seq")
    if "pv" in cfg.obs.sequence_features:
        names.append("pv_seq")
    names.extend(["mu_t", "mu_next"])
    return names


@dataclass(frozen=True)
class ObservationCacheResult:
    cache_dir: Path
    cache_hit: bool
    cache_build_time_s: float
    manifest: dict[str, object]


class ObservationCacheStore:
    """Read-only episode cache backed by memory-mapped .npy files."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        self._arrays = {
            name: np.load(self.cache_dir / file_name, mmap_mode="r")
            for name, file_name in dict(self.manifest.get("files", {})).items()
        }

    def episode(self, episode_idx: int) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(array[int(episode_idx)])
            for name, array in self._arrays.items()
        }


def build_or_load_observation_cache(
    cfg,
    *,
    split: str = "train",
    forecast_ready: dict[str, object] | None = None,
    refresh: bool = False,
    root: str | Path | None = None,
    batch_size: int | None = None,
) -> ObservationCacheResult:
    signature = _cache_signature(cfg, split=str(split), forecast_ready=forecast_ready)
    cache_root = _default_cache_root(root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / _signature_hash(signature)
    manifest_path = cache_dir / "manifest.json"
    required_names = _required_array_names(cfg)

    if not refresh and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            files = dict(manifest.get("files", {}))
            if all((cache_dir / files.get(name, "")).exists() for name in required_names):
                return ObservationCacheResult(
                    cache_dir=cache_dir,
                    cache_hit=True,
                    cache_build_time_s=0.0,
                    manifest=manifest,
                )

    started = time.perf_counter()
    dataset = build_dataset(cfg, mode=str(split))
    n_episodes = int(dataset.num_episodes())
    episode_length = int(cfg.env.episode_limit)
    n_agents = int(cfg.env.num_agents)
    sequence_length = int(cfg.env.future_horizon) + 1
    resolved_batch_size = max(
        int(
            batch_size
            if batch_size is not None
            else getattr(getattr(cfg, "runtime", None), "fastlab_observation_cache_batch_size", 8192)
        ),
        1,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    arrays: dict[str, np.memmap] = {}
    if "calendar_time" in required_names:
        files["calendar_time"] = "calendar_time.npy"
        arrays["calendar_time"] = open_memmap(
            cache_dir / files["calendar_time"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, n_agents, 4),
        )
    if "price_seq" in required_names:
        files["price_seq"] = "price_seq.npy"
        arrays["price_seq"] = open_memmap(
            cache_dir / files["price_seq"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, sequence_length),
        )
    if "load_seq" in required_names:
        files["load_seq"] = "load_seq.npy"
        arrays["load_seq"] = open_memmap(
            cache_dir / files["load_seq"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, n_agents, sequence_length),
        )
    if "pv_seq" in required_names:
        files["pv_seq"] = "pv_seq.npy"
        arrays["pv_seq"] = open_memmap(
            cache_dir / files["pv_seq"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, n_agents, sequence_length),
        )
    files["mu_t"] = "mu_t.npy"
    files["mu_next"] = "mu_next.npy"
    arrays["mu_t"] = open_memmap(
        cache_dir / files["mu_t"],
        mode="w+",
        dtype=np.float32,
        shape=(n_episodes, episode_length),
    )
    arrays["mu_next"] = open_memmap(
        cache_dir / files["mu_next"],
        mode="w+",
        dtype=np.float32,
        shape=(n_episodes, episode_length),
    )

    forecaster = build_forecaster(cfg)
    vectorized_forecaster = hasattr(forecaster, "predict_episode_matrix")

    for episode_idx in range(n_episodes):
        episode = dataset.get_episode(episode_idx)
        signals = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(episode.get("signals", {})).items()
        }
        meta = dict(episode.get("meta", {}))
        timestamps = list(meta.get("timestamps") or [])

        forecaster.reset()
        forecaster.set_episode(signals, meta)
        mu_t, mu_next = _compute_future_mean_price(signals["price"], int(cfg.env.future_horizon))
        arrays["mu_t"][episode_idx] = mu_t
        arrays["mu_next"][episode_idx] = mu_next

        if "calendar_time" in arrays:
            arrays["calendar_time"][episode_idx] = _build_calendar_time_matrix(timestamps, n_agents)

        if vectorized_forecaster:
            if "price_seq" in arrays:
                arrays["price_seq"][episode_idx] = forecaster.predict_episode_matrix(
                    signals["price"],
                    sequence_length,
                    signal_name="price",
                    history_timestamps=timestamps,
                    batch_size=resolved_batch_size,
                ).astype(np.float32)
            if "load_seq" in arrays:
                arrays["load_seq"][episode_idx] = forecaster.predict_episode_matrix(
                    signals["load"],
                    sequence_length,
                    signal_name="load",
                    history_timestamps=timestamps,
                    batch_size=resolved_batch_size,
                ).astype(np.float32)
            if "pv_seq" in arrays:
                arrays["pv_seq"][episode_idx] = forecaster.predict_episode_matrix(
                    signals["pv"],
                    sequence_length,
                    signal_name="pv",
                    history_timestamps=timestamps,
                    batch_size=resolved_batch_size,
                ).astype(np.float32)
            continue

        for step_idx in range(episode_length):
            history_timestamps = timestamps[: step_idx + 1]
            if "price_seq" in arrays:
                price_history = signals["price"][: step_idx + 1]
                arrays["price_seq"][episode_idx, step_idx] = forecaster.predict(
                    price_history,
                    sequence_length,
                    signal_name="price",
                    history_timestamps=history_timestamps,
                ).astype(np.float32)
            if "load_seq" in arrays:
                load_history = signals["load"][: step_idx + 1, :]
                arrays["load_seq"][episode_idx, step_idx] = forecaster.predict(
                    load_history,
                    sequence_length,
                    signal_name="load",
                    history_timestamps=history_timestamps,
                ).astype(np.float32)
            if "pv_seq" in arrays:
                pv_history = signals["pv"][: step_idx + 1, :]
                arrays["pv_seq"][episode_idx, step_idx] = forecaster.predict(
                    pv_history,
                    sequence_length,
                    signal_name="pv",
                    history_timestamps=history_timestamps,
                ).astype(np.float32)

    for array in arrays.values():
        array.flush()

    manifest = {
        "signature": signature,
        "split": str(split),
        "episode_length": episode_length,
        "num_agents": n_agents,
        "sequence_length": sequence_length,
        "num_episodes": n_episodes,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")

    return ObservationCacheResult(
        cache_dir=cache_dir,
        cache_hit=False,
        cache_build_time_s=max(time.perf_counter() - started, 0.0),
        manifest=manifest,
    )


__all__ = [
    "ObservationCacheResult",
    "ObservationCacheStore",
    "build_or_load_observation_cache",
]
