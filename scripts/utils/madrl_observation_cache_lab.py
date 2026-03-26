"""Build and load exact precomputed observation caches for MADRL fast-lab."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.loaders.registry import build_dataset
from envs.observation.feature_blocks import build_calendar_time_features
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
    price = np.asarray(price, dtype=np.float32).reshape(-1)
    horizon = int(max(future_horizon, 1))
    mu_t = np.zeros_like(price, dtype=np.float32)
    mu_next = np.zeros_like(price, dtype=np.float32)
    for t in range(price.shape[0]):
        start = t + 1
        end = min(t + 1 + horizon, price.shape[0])
        if start >= price.shape[0]:
            mu_t[t] = np.float32(price[min(t, price.shape[0] - 1)])
        else:
            chunk = price[start:end]
            mu_t[t] = np.float32(np.mean(chunk)) if chunk.size else np.float32(price[min(t, price.shape[0] - 1)])

        t_next = min(t + 1, price.shape[0] - 1)
        start_next = t_next + 1
        end_next = min(t_next + 1 + horizon, price.shape[0])
        if start_next >= price.shape[0]:
            mu_next[t] = np.float32(price[t_next])
        else:
            chunk_next = price[start_next:end_next]
            mu_next[t] = (
                np.float32(np.mean(chunk_next))
                if chunk_next.size
                else np.float32(price[t_next])
            )
    return mu_t.astype(np.float32), mu_next.astype(np.float32)


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

    cache_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    if "calendar_time" in required_names:
        arrays["calendar_time"] = np.zeros((n_episodes, episode_length, n_agents, 4), dtype=np.float32)
    if "price_seq" in required_names:
        arrays["price_seq"] = np.zeros((n_episodes, episode_length, sequence_length), dtype=np.float32)
    if "load_seq" in required_names:
        arrays["load_seq"] = np.zeros((n_episodes, episode_length, n_agents, sequence_length), dtype=np.float32)
    if "pv_seq" in required_names:
        arrays["pv_seq"] = np.zeros((n_episodes, episode_length, n_agents, sequence_length), dtype=np.float32)
    arrays["mu_t"] = np.zeros((n_episodes, episode_length), dtype=np.float32)
    arrays["mu_next"] = np.zeros((n_episodes, episode_length), dtype=np.float32)

    forecaster = build_forecaster(cfg)

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

        for step_idx in range(episode_length):
            if "calendar_time" in arrays:
                arrays["calendar_time"][episode_idx, step_idx] = build_calendar_time_features(
                    timestamps[step_idx],
                    n_agents,
                )
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

    files: dict[str, str] = {}
    for name, array in arrays.items():
        file_name = f"{name}.npy"
        np.save(cache_dir / file_name, array, allow_pickle=False)
        files[name] = file_name

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
