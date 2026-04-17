"""Build and reuse shared MADRL train/test data packages."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

from data.loaders.registry import _resolve_split_dates, _same_year_has_explicit_train_range, build_dataset
from predictors.lstm_forecaster import load_lstm_forecaster_artifacts
from predictors.registry import build_forecaster
from predictors.time_features import DEFAULT_LOCAL_TIMEZONE, coerce_timestamp_index
from predictors.training import ensure_lstm_artifacts
from scripts.utils.project_paths import get_shared_data_root

if TYPE_CHECKING:
    from configs.experiment_config import ExperimentConfig

_FLOAT_PRECISION = 6
_LOCK_TIMEOUT_S = 300.0
_LOCK_POLL_INTERVAL_S = 0.25
_SCHEMA_VERSION = 2


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


def _normalize_float(value: float) -> float:
    return float(round(float(value), _FLOAT_PRECISION))


def _normalize_for_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_signature(sub_value)
            for key, sub_value in sorted(dict(value).items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_signature(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return f"{_normalize_float(float(value)):.{_FLOAT_PRECISION}f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _signature_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(_normalize_for_signature(payload), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_artifact_bundle(bundle) -> list[tuple[str, str | None, str | None]]:
    if isinstance(bundle, tuple) and len(bundle) == 3 and not any(isinstance(item, (tuple, list)) for item in bundle):
        return [bundle]
    if isinstance(bundle, list):
        return [tuple(item) for item in bundle]
    raise TypeError(f"Unsupported forecast artifact bundle: {bundle!r}")


def _normalized_artifact_meta_subset(meta: dict[str, object]) -> dict[str, object]:
    return _normalize_for_signature(
        {
            "signal_name": meta.get("signal_name"),
            "future_horizon": int(meta.get("future_horizon", meta.get("pred_len", 0))),
            "seq_len": int(meta.get("seq_len", 0)),
            "hidden_size": int(meta.get("hidden_size", 0)),
            "num_layers": int(meta.get("num_layers", 0)),
            "dropout": float(meta.get("dropout", 0.0)),
            "input_size": int(meta.get("input_size", 1)),
            "time_feature_mode": str(meta.get("time_feature_mode", "none")),
            "model_mode": str(meta.get("model_mode", "shared")),
            "normalization_mode": str(meta.get("normalization_mode", "none")),
            "postprocess_mode": str(meta.get("postprocess_mode", "none")),
            "baseline_mode": str(meta.get("baseline_mode", "none")),
            "blend_weight": meta.get("blend_weight"),
            "component": meta.get("component"),
            "agent_index": meta.get("agent_index"),
            "agent_profile": meta.get("agent_profile"),
        }
    )


def _artifact_fingerprint(forecast_ready: dict[str, object]) -> dict[str, object]:
    artifacts = dict(forecast_ready.get("artifacts") or {})
    if not artifacts:
        raise FileNotFoundError("No managed LSTM artifacts are available for shared MADRL data generation.")

    fingerprint: dict[str, object] = {}
    for signal_name, bundle in sorted(artifacts.items()):
        entries: list[dict[str, object]] = []
        for model_path, meta_path, scaler_path in _normalize_artifact_bundle(bundle):
            meta, _ = load_lstm_forecaster_artifacts(model_path, meta_path, scaler_path)
            entry = {
                "model_sha256": _file_sha256(model_path),
                "meta_sha256": _file_sha256(meta_path),
                "scaler_sha256": _file_sha256(scaler_path),
                "meta": _normalized_artifact_meta_subset(meta),
            }
            entries.append(entry)
        fingerprint[str(signal_name)] = entries
    return fingerprint


def validate_lstm_artifacts_for_shared_data(cfg) -> dict[str, object]:
    if str(cfg.forecast.type).strip().lower() != "lstm":
        return {"artifacts": {}, "fingerprint": {"mode": "non_lstm"}}

    forecast_ready = ensure_lstm_artifacts(cfg, device=cfg.runtime.device)
    return {
        "forecast_ready": forecast_ready,
        "fingerprint": _artifact_fingerprint(forecast_ready),
    }


def _shared_data_signature_payload(cfg, *, artifact_fingerprint: dict[str, object]) -> dict[str, object]:
    include_test_window = _test_window_in_signature(cfg)
    return {
        "schema_version": _SCHEMA_VERSION,
        "num_agents": int(cfg.env.num_agents),
        "episode_limit": int(cfg.env.episode_limit),
        "future_horizon": int(cfg.env.future_horizon),
        "agent_profiles": list(cfg.data.agent_profiles),
        "agent_bus_ids": list(cfg.grid.agent_bus_ids),
        "train_year": int(cfg.data.train_year),
        "test_year": int(cfg.data.test_year),
        "train_start_date": cfg.data.train_start_date,
        "train_end_date": cfg.data.train_end_date,
        "same_year_has_explicit_train_range": bool(_same_year_has_explicit_train_range(cfg)),
        "test_window_in_signature": bool(include_test_window),
        "test_window_strategy": _test_window_strategy(cfg),
        "test_start_date": cfg.data.test_start_date if include_test_window else None,
        "test_end_date": cfg.data.test_end_date if include_test_window else None,
        "load_components": list(cfg.data.load_components),
        "pv_reference": str(cfg.data.pv_reference),
        "pv_capacity_kw": list(cfg.data.pv_capacity_kw),
        "load_scale": list(cfg.data.load_scale),
        "pv_scale": list(cfg.data.pv_scale),
        "artifacts": artifact_fingerprint,
    }


def build_shared_data_signature(cfg) -> dict[str, object]:
    artifact_info = validate_lstm_artifacts_for_shared_data(cfg)
    return _shared_data_signature_payload(cfg, artifact_fingerprint=dict(artifact_info["fingerprint"]))


def _test_window_in_signature(cfg) -> bool:
    return (
        int(cfg.data.train_year) == int(cfg.data.test_year)
        and not _same_year_has_explicit_train_range(cfg)
    )


def _test_window_strategy(cfg) -> str:
    return "cfg_window" if _test_window_in_signature(cfg) else "full_year_runtime_slice"


def _resolve_shared_split_controls(cfg, split: str) -> dict[str, object]:
    if str(split) == "test" and _test_window_strategy(cfg) == "full_year_runtime_slice":
        selected_year, start_date, end_date, exclude_start_date, exclude_end_date = _resolve_split_dates(
            cfg,
            split,
            override_start_date=None,
            override_end_date=None,
            override_exclude_start_date=None,
            override_exclude_end_date=None,
        )
    else:
        selected_year, start_date, end_date, exclude_start_date, exclude_end_date = _resolve_split_dates(cfg, split)
    return {
        "split": str(split),
        "year": int(selected_year),
        "start_date": start_date,
        "end_date": end_date,
        "exclude_start_date": exclude_start_date,
        "exclude_end_date": exclude_end_date,
        "window_strategy": (
            _test_window_strategy(cfg)
            if str(split) == "test"
            else "cfg_window"
        ),
    }


def _episode_manifest_entry(dataset, episode_idx: int, episode: dict[str, object]) -> dict[str, object]:
    episode_slices = list(getattr(dataset, "_episode_slices", []))
    history_start_idx, active_start_idx, active_end_idx = episode_slices[int(episode_idx)]
    timestamps = [str(value) for value in list(dict(episode.get("meta", {})).get("timestamps") or [])]
    first_timestamp = None if not timestamps else str(timestamps[0])
    last_timestamp = None if not timestamps else str(timestamps[-1])
    return {
        "episode_idx": int(episode_idx),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "first_local_date": None if first_timestamp is None else str(pd.Timestamp(first_timestamp).date()),
        "last_local_date": None if last_timestamp is None else str(pd.Timestamp(last_timestamp).date()),
        "history_start_idx": int(history_start_idx),
        "active_start_idx": int(active_start_idx),
        "active_end_idx": int(active_end_idx),
    }


def _full_episode_index_list(split_manifest: dict[str, object]) -> list[int]:
    episodes = list(split_manifest.get("episodes") or [])
    return [int(entry["episode_idx"]) for entry in episodes]


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


def _merge_history_with_episode_signals(
    signals: dict[str, np.ndarray],
    history_signals: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    merged: dict[str, np.ndarray] = {}
    prefix_length = 0
    for name, signal in signals.items():
        prefix = np.asarray(history_signals.get(name), dtype=np.float32)
        value = np.asarray(signal, dtype=np.float32)
        if prefix.size == 0:
            merged[name] = value.copy()
            continue
        prefix_length = max(prefix_length, int(prefix.shape[0]))
        merged[name] = np.concatenate([prefix, value], axis=0).astype(np.float32, copy=False)
    return merged, prefix_length


def _write_split_shared_data(cfg, *, split: str, split_dir: Path) -> dict[str, object]:
    split_controls = _resolve_shared_split_controls(cfg, str(split))
    dataset = build_dataset(
        cfg,
        mode=str(split),
        override_start_date=split_controls["start_date"],
        override_end_date=split_controls["end_date"],
        override_exclude_start_date=split_controls["exclude_start_date"],
        override_exclude_end_date=split_controls["exclude_end_date"],
    )
    n_episodes = int(dataset.num_episodes())
    episode_length = int(cfg.env.episode_limit)
    n_agents = int(cfg.env.num_agents)
    sequence_length = int(cfg.env.future_horizon) + 1

    split_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "calendar_time": "calendar_time.npy",
        "price_seq": "price_seq.npy",
        "load_seq": "load_seq.npy",
        "pv_seq": "pv_seq.npy",
        "mu_t": "mu_t.npy",
        "mu_next": "mu_next.npy",
    }

    arrays = {
        "calendar_time": open_memmap(
            split_dir / files["calendar_time"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, n_agents, 4),
        ),
        "price_seq": open_memmap(
            split_dir / files["price_seq"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, sequence_length),
        ),
        "load_seq": open_memmap(
            split_dir / files["load_seq"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, n_agents, sequence_length),
        ),
        "pv_seq": open_memmap(
            split_dir / files["pv_seq"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length, n_agents, sequence_length),
        ),
        "mu_t": open_memmap(
            split_dir / files["mu_t"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length),
        ),
        "mu_next": open_memmap(
            split_dir / files["mu_next"],
            mode="w+",
            dtype=np.float32,
            shape=(n_episodes, episode_length),
        ),
    }

    forecaster = build_forecaster(cfg)
    vectorized_forecaster = hasattr(forecaster, "predict_episode_matrix")
    episode_manifest: list[dict[str, object]] = []

    for episode_idx in tqdm(
        range(n_episodes),
        desc=f"shared_data[{split}] episodes",
        leave=False,
    ):
        episode = dataset.get_episode(episode_idx)
        signals = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(episode.get("signals", {})).items()
        }
        history_signals = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(episode.get("history_signals", {})).items()
        }
        meta = dict(episode.get("meta", {}))
        timestamps = list(meta.get("timestamps") or [])
        history_timestamps = [str(value) for value in list(episode.get("history_timestamps") or [])]
        combined_signals, prefix_length = _merge_history_with_episode_signals(signals, history_signals)
        combined_timestamps = [*history_timestamps, *timestamps]
        episode_manifest.append(_episode_manifest_entry(dataset, episode_idx, episode))

        forecaster.reset()
        forecaster.set_episode(combined_signals, meta)

        mu_t, mu_next = _compute_future_mean_price(signals["price"], int(cfg.env.future_horizon))
        arrays["mu_t"][episode_idx] = mu_t
        arrays["mu_next"][episode_idx] = mu_next
        arrays["calendar_time"][episode_idx] = _build_calendar_time_matrix(timestamps, n_agents)

        if vectorized_forecaster:
            arrays["price_seq"][episode_idx] = forecaster.predict_episode_matrix(
                combined_signals["price"],
                sequence_length,
                signal_name="price",
                history_timestamps=combined_timestamps,
            ).astype(np.float32)[prefix_length:]
            arrays["load_seq"][episode_idx] = forecaster.predict_episode_matrix(
                combined_signals["load"],
                sequence_length,
                signal_name="load",
                history_timestamps=combined_timestamps,
            ).astype(np.float32)[prefix_length:]
            arrays["pv_seq"][episode_idx] = forecaster.predict_episode_matrix(
                combined_signals["pv"],
                sequence_length,
                signal_name="pv",
                history_timestamps=combined_timestamps,
            ).astype(np.float32)[prefix_length:]
            continue

        for step_idx in range(episode_length):
            current_history_timestamps = combined_timestamps[: prefix_length + step_idx + 1]
            arrays["price_seq"][episode_idx, step_idx] = forecaster.predict(
                combined_signals["price"][: prefix_length + step_idx + 1],
                sequence_length,
                signal_name="price",
                history_timestamps=current_history_timestamps,
            ).astype(np.float32)
            arrays["load_seq"][episode_idx, step_idx] = forecaster.predict(
                combined_signals["load"][: prefix_length + step_idx + 1, :],
                sequence_length,
                signal_name="load",
                history_timestamps=current_history_timestamps,
            ).astype(np.float32)
            arrays["pv_seq"][episode_idx, step_idx] = forecaster.predict(
                combined_signals["pv"][: prefix_length + step_idx + 1, :],
                sequence_length,
                signal_name="pv",
                history_timestamps=current_history_timestamps,
            ).astype(np.float32)

    for array in arrays.values():
        array.flush()
    for array in arrays.values():
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    del arrays

    split_manifest = {
        "schema_version": _SCHEMA_VERSION,
        "split": str(split),
        "num_episodes": n_episodes,
        "episode_length": episode_length,
        "num_agents": n_agents,
        "sequence_length": sequence_length,
        "split_controls": split_controls,
        "episodes": episode_manifest,
        "files": files,
    }
    (split_dir / "manifest.json").write_text(json.dumps(split_manifest, indent=2, default=_json_default), encoding="utf-8")
    return split_manifest


def _shared_data_root(root: str | Path | None = None) -> Path:
    return get_shared_data_root(root) / "mainline"


def _root_manifest_is_complete(shared_dir: Path, signature_hash: str) -> bool:
    manifest_path = shared_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if str(manifest.get("signature_hash", "")) != str(signature_hash):
        return False
    for split in ("train", "test"):
        split_manifest = shared_dir / split / "manifest.json"
        if not split_manifest.exists():
            return False
    return True


def _try_acquire_lock(lock_dir: Path) -> bool:
    try:
        lock_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        return False
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "pid": int(os.getpid()),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def _release_lock(lock_dir: Path) -> None:
    if lock_dir.exists():
        shutil.rmtree(lock_dir, ignore_errors=True)


def _wait_for_existing_or_lock(shared_dir: Path, lock_dir: Path, signature_hash: str) -> bool:
    started = time.monotonic()
    while True:
        if _root_manifest_is_complete(shared_dir, signature_hash):
            return True
        if not lock_dir.exists():
            return False
        lock_age = max(time.time() - lock_dir.stat().st_mtime, 0.0)
        if lock_age >= _LOCK_TIMEOUT_S and not shared_dir.exists():
            shutil.rmtree(lock_dir, ignore_errors=True)
            return False
        if time.monotonic() - started >= _LOCK_TIMEOUT_S:
            raise TimeoutError(
                f"Timed out waiting for shared MADRL data '{shared_dir.name}' to finish generating."
            )
        time.sleep(_LOCK_POLL_INTERVAL_S)


@dataclass(frozen=True)
class SharedDataResult:
    shared_data_dir: Path
    signature_hash: str
    manifest: dict[str, object]
    reused: bool


def build_shared_data_status_summary(
    result: SharedDataResult,
    *,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
) -> dict[str, object]:
    if result.reused:
        status = "reused_existing_shared_data"
        message = "Reused existing shared MADRL data package."
    else:
        status = "generated_new_shared_data"
        message = "Generated a new shared MADRL data package from available forecast artifacts."

    return {
        "shared_data_status": status,
        "shared_data_message": message,
        "shared_data_dir": str(result.shared_data_dir),
        "shared_data_signature": str(result.signature_hash),
        "shared_data_reused": bool(result.reused),
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
    }


class PrecomputedObservationStore:
    """Read per-episode precomputed observations from a split directory."""

    def __init__(self, split_dir: str | Path):
        self.split_dir = Path(split_dir).resolve()
        self.manifest = json.loads((self.split_dir / "manifest.json").read_text(encoding="utf-8"))
        self._arrays = {
            name: np.load(self.split_dir / file_name, mmap_mode="r")
            for name, file_name in dict(self.manifest.get("files", {})).items()
        }

    def episode(self, episode_idx: int) -> dict[str, np.ndarray]:
        return {name: np.asarray(array[int(episode_idx)]) for name, array in self._arrays.items()}


def load_madrl_shared_data_manifest(path: str | Path) -> dict[str, object]:
    shared_dir = Path(path).resolve()
    return json.loads((shared_dir / "manifest.json").read_text(encoding="utf-8"))


def select_shared_data_episode_indices(
    manifest: dict[str, object],
    *,
    start_date: str | None,
    end_date: str | None,
) -> list[int]:
    episodes = list(manifest.get("episodes") or [])
    if not episodes:
        raise ValueError("Shared-data split manifest does not contain episode metadata.")

    if start_date in (None, "") and end_date in (None, ""):
        selected = [int(entry["episode_idx"]) for entry in episodes]
        if not selected:
            raise ValueError("Shared-data split manifest does not contain any selectable episodes.")
        return selected

    normalized_start = None if start_date in (None, "") else pd.Timestamp(str(start_date)).date()
    normalized_end = None if end_date in (None, "") else pd.Timestamp(str(end_date)).date()
    selected: list[int] = []
    for entry in episodes:
        first_local_date = None if entry.get("first_local_date") in (None, "") else pd.Timestamp(str(entry["first_local_date"])).date()
        last_local_date = None if entry.get("last_local_date") in (None, "") else pd.Timestamp(str(entry["last_local_date"])).date()
        if normalized_start is not None and (first_local_date is None or first_local_date < normalized_start):
            continue
        if normalized_end is not None and (last_local_date is None or last_local_date > normalized_end):
            continue
        selected.append(int(entry["episode_idx"]))
    if not selected:
        raise ValueError(
            "No shared-data episodes fall fully inside the requested date window: "
            f"start_date={start_date!r}, end_date={end_date!r}."
        )
    return selected


def ensure_madrl_shared_data(
    cfg,
    *,
    root: str | Path | None = None,
    force: bool = False,
) -> SharedDataResult:
    artifact_info = validate_lstm_artifacts_for_shared_data(cfg)
    forecast_ready = dict(artifact_info.get("forecast_ready") or {})
    signature_payload = _shared_data_signature_payload(
        cfg,
        artifact_fingerprint=dict(artifact_info["fingerprint"]),
    )
    signature_hash = _signature_hash(signature_payload)
    root_dir = _shared_data_root(root)
    root_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = (root_dir / signature_hash).resolve()
    lock_dir = (root_dir / f"{signature_hash}.lock").resolve()

    if not force and _root_manifest_is_complete(shared_dir, signature_hash):
        return SharedDataResult(
            shared_data_dir=shared_dir,
            signature_hash=signature_hash,
            manifest=load_madrl_shared_data_manifest(shared_dir),
            reused=True,
        )

    if not _try_acquire_lock(lock_dir):
        if _wait_for_existing_or_lock(shared_dir, lock_dir, signature_hash):
            return SharedDataResult(
                shared_data_dir=shared_dir,
                signature_hash=signature_hash,
                manifest=load_madrl_shared_data_manifest(shared_dir),
                reused=True,
            )
        if not _try_acquire_lock(lock_dir):
            return SharedDataResult(
                shared_data_dir=shared_dir,
                signature_hash=signature_hash,
                manifest=load_madrl_shared_data_manifest(shared_dir),
                reused=True,
            )

    temp_dir = (root_dir / f"{signature_hash}.tmp.{os.getpid()}.{uuid.uuid4().hex}").resolve()
    try:
        if force and shared_dir.exists():
            shutil.rmtree(shared_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=False)
        train_manifest = _write_split_shared_data(cfg, split="train", split_dir=temp_dir / "train")
        test_manifest = _write_split_shared_data(cfg, split="test", split_dir=temp_dir / "test")
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "signature_hash": signature_hash,
            "signature": _normalize_for_signature(signature_payload),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "shared_data_dir": str(shared_dir),
            "artifact_inventory": dict(artifact_info["fingerprint"]),
            "data_controls": {
                "agent_profiles": list(cfg.data.agent_profiles),
                "agent_bus_ids": list(cfg.grid.agent_bus_ids),
                "train_year": int(cfg.data.train_year),
                "test_year": int(cfg.data.test_year),
                "train_start_date": cfg.data.train_start_date,
                "train_end_date": cfg.data.train_end_date,
                "test_start_date": cfg.data.test_start_date,
                "test_end_date": cfg.data.test_end_date,
                "same_year_has_explicit_train_range": bool(_same_year_has_explicit_train_range(cfg)),
                "test_window_strategy": _test_window_strategy(cfg),
                "load_scale": _normalize_for_signature(list(cfg.data.load_scale)),
                "pv_scale": _normalize_for_signature(list(cfg.data.pv_scale)),
                "pv_capacity_kw": _normalize_for_signature(list(cfg.data.pv_capacity_kw)),
                "load_components": list(cfg.data.load_components),
                "pv_reference": str(cfg.data.pv_reference),
                "future_horizon": int(cfg.env.future_horizon),
                "episode_limit": int(cfg.env.episode_limit),
            },
            "splits": {
                "train": train_manifest,
                "test": test_manifest,
            },
        }
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        if shared_dir.exists():
            shutil.rmtree(shared_dir, ignore_errors=True)
        temp_dir.replace(shared_dir)
        return SharedDataResult(
            shared_data_dir=shared_dir,
            signature_hash=signature_hash,
            manifest=manifest,
            reused=False,
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        _release_lock(lock_dir)


def dump_cfg_json(cfg, path: str | Path) -> Path:
    payload = dataclasses.asdict(cfg)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return target


def _apply_dataclass_values(instance, payload: dict[str, object]) -> None:
    for field in dataclasses.fields(instance):
        if field.name not in payload:
            continue
        current_value = getattr(instance, field.name)
        next_value = payload[field.name]
        if dataclasses.is_dataclass(current_value):
            _apply_dataclass_values(current_value, dict(next_value))
            continue
        if isinstance(current_value, torch.device):
            setattr(instance, field.name, torch.device(str(next_value)))
            continue
        setattr(instance, field.name, next_value)


def load_cfg_json(path: str | Path) -> "ExperimentConfig":
    from configs.experiment_config import ExperimentConfig

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = ExperimentConfig()
    _apply_dataclass_values(cfg, dict(payload))
    return cfg


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare shared MADRL train/test data from a config snapshot.")
    parser.add_argument("--cfg-json", required=True)
    parser.add_argument("--root")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    cfg = load_cfg_json(args.cfg_json)
    result = ensure_madrl_shared_data(cfg, root=args.root, force=bool(args.force))
    print(
        json.dumps(
            {
                "shared_data_dir": str(result.shared_data_dir),
                "shared_data_signature": str(result.signature_hash),
                "reused": bool(result.reused),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
