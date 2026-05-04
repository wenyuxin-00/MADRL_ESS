from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np
from tqdm.auto import tqdm

from REMAKE.configs.cfg import Cfg
from REMAKE.data.loader import SeriesData, load_prosumer_dataset
from REMAKE.predictors.lstm_features import encode_time_features
from REMAKE.predictors.lstm_loader import LSTMForecastRuntime


@dataclass(frozen=True)
class ShareData:
    root: Path
    manifest: dict[str, Any]
    train: dict[str, np.ndarray]
    eval: dict[str, np.ndarray]


def _episode_starts(data: SeriesData, cfg: Cfg, split: str) -> list[int]:
    h, p, steps = int(cfg.forecast.history_window), int(cfg.obs.sequence_length), int(cfg.env.episode_steps)
    stride = int(cfg.env.window_stride_days) * steps if split == "train" else steps
    limit = int(data.target_start + data.target_length - steps + 1) if split == "eval" else int(len(data.price) - steps - p + 2)
    starts = list(range(max(h, int(data.target_start)), limit, stride))
    return starts


def _perfect_sequences(values: np.ndarray, starts: list[int], steps: int, horizon: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.stack([[arr[start + t:start + t + horizon] for t in range(steps)] for start in starts]).astype(np.float32)


def _lstm_sequences(cfg: Cfg, data: SeriesData, starts: list[int], runtime: LSTMForecastRuntime) -> dict[str, np.ndarray]:
    e, steps, horizon, n = len(starts), int(cfg.env.episode_steps), int(cfg.obs.sequence_length), int(cfg.env.num_agents)
    h = int(cfg.forecast.history_window)
    idxs = np.asarray([start + t for start in starts for t in range(steps)], dtype=np.int64)
    time_features = encode_time_features(data.timestamps, cfg.forecast.load_lstm_time_features)
    history_timestamps = np.stack([time_features[idx - h:idx + horizon] for idx in idxs]).astype(np.float32)
    price_histories = np.stack([data.price[idx - h:idx] for idx in idxs]).astype(np.float32)
    price = runtime.predict_batch(price_histories, "wholesale_price", timestamps=history_timestamps).reshape(e, steps, horizon)
    load = np.zeros((e, steps, horizon, n), dtype=np.float32)
    for agent in tqdm(range(n), desc="lstm share load agents", unit="agent", leave=False, ascii=True):
        histories = {component: np.stack([data.load_components[component][idx - h:idx, agent] for idx in idxs]).astype(np.float32) for component in cfg.data.load_components}
        load[:, :, :, agent] = runtime.predict_load_batch(histories, agent, timestamps=history_timestamps).reshape(e, steps, horizon)
    histories = np.stack([data.pv[idx - h:idx, 0] for idx in idxs]).astype(np.float32)
    pv = runtime.predict_batch(histories, "pv", timestamps=history_timestamps).reshape(e, steps, horizon)
    return {"lstm_price_seq": price, "lstm_load_seq": load, "lstm_pv_seq": pv}


def _build_split_share_data(cfg: Cfg, forecast_artifact_dir: Path, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    h, p = int(cfg.forecast.history_window), int(cfg.obs.sequence_length)
    data = load_prosumer_dataset(cfg, split, pad_history_steps=h if split == "eval" else 0, pad_future_steps=p if split == "eval" else 0)
    starts = _episode_starts(data, cfg, split)
    steps, horizon = int(cfg.env.episode_steps), int(cfg.obs.sequence_length)
    arrays = {
        "episode_starts": np.asarray(starts, dtype=np.int64),
        "timestamps": np.asarray([[data.timestamps[s + t] for t in range(steps)] for s in starts]),
        "price": np.stack([data.price[s:s + steps] for s in starts]).astype(np.float32),
        "load": np.stack([data.load[s:s + steps] for s in starts]).astype(np.float32),
        "pv": np.stack([data.pv[s:s + steps, 0] for s in starts]).astype(np.float32),
        "perfect_price_seq": _perfect_sequences(data.price, starts, steps, horizon),
        "perfect_load_seq": _perfect_sequences(data.load, starts, steps, horizon),
        "perfect_pv_seq": _perfect_sequences(data.pv[:, 0], starts, steps, horizon),
    }
    arrays.update(_lstm_sequences(cfg, data, starts, LSTMForecastRuntime(cfg, forecast_artifact_dir)))
    meta = {"split": split, "n_episodes": len(starts), "episode_indices": list(range(len(starts))), "first_timestamp": data.timestamps[starts[0]], "last_timestamp": data.timestamps[starts[-1] + steps - 1]}
    return arrays, meta


def build_share_data(cfg: Cfg, run_dir: Path, forecast_artifact_dir: Path, overwrite: bool = False) -> Path:
    out_dir = Path(run_dir) / "share_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_meta = {}
    for split in tqdm(("train", "eval"), desc="share_data splits", unit="split", ascii=True):
        arrays, meta = _build_split_share_data(cfg, Path(forecast_artifact_dir), split)
        np.savez_compressed(out_dir / f"{split}.npz", **arrays); split_meta[split] = meta
    manifest = {"cfg_hash": cfg.hash8(), "schema_version": 2, "pv_shared": True, "forecast_artifact_dir": str(Path(forecast_artifact_dir)), "num_agents": int(cfg.env.num_agents), "episode_steps": int(cfg.env.episode_steps), "sequence_length": int(cfg.obs.sequence_length), "splits": split_meta}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def load_share_data(path: str | Path, cfg: Cfg) -> ShareData:
    root = Path(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {}
    for split in ("train", "eval"):
        npz = root / f"{split}.npz"
        arrays[split] = dict(np.load(npz))
    return ShareData(root=root, manifest=manifest, train=arrays["train"], eval=arrays["eval"])
