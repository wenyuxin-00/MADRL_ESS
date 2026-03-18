"""Training and artifact management utilities for forecast models."""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from common.torch_runtime import TorchRuntimeState, configure_torch_runtime, resolve_device
from forecast.artifacts import (
    DEFAULT_SUPPORTED_FORECAST_SIGNALS,
    get_default_lstm_artifact_dir,
    get_default_lstm_artifact_paths,
    get_weekly_forecast_plot_path,
)
from forecast.lstm_forecaster import LSTMForecaster, save_lstm_forecaster_artifacts
from forecast.lstm_model import LSTMForecastModel

DEFAULT_WEEK_STEPS = 96 * 7
SIGNAL_TRAINING_OVERRIDE_FIELDS = {
    "history_window": "history_window",
    "hidden_size": "lstm_hidden_size",
    "num_layers": "lstm_num_layers",
    "dropout": "lstm_dropout",
    "batch_size": "lstm_batch_size",
    "epochs": "lstm_epochs",
    "lr": "lstm_lr",
    "train_ratio": "lstm_train_ratio",
    "val_ratio": "lstm_val_ratio",
}


@dataclass(frozen=True)
class SignalCsvSource:
    """Resolved train/test source files for one signal."""

    signal_name: str
    train_path: Path
    test_path: Path
    value_columns: tuple[str, ...]


@dataclass(frozen=True)
class SignalForecastEvaluation:
    """单个 signal 的周评估结果。"""

    signal_name: str
    evaluation_mode: str
    timestamps: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    metrics: dict[str, float | None]
    source_columns: tuple[str, ...]


def _normalize_signal_name(signal_name: str) -> str:
    return str(signal_name).strip().lower()


def configured_forecast_signals(cfg) -> list[str]:
    """返回 forecast 主线中声明的目标信号。"""
    unique = []
    for signal_name in cfg.forecast.target_signals:
        normalized = _normalize_signal_name(signal_name)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique or list(DEFAULT_SUPPORTED_FORECAST_SIGNALS)


def required_forecast_signals(cfg) -> list[str]:
    """返回当前观测链路真正会消费的 forecast 信号。"""
    active = []
    configured = set(configured_forecast_signals(cfg))
    for signal_name in cfg.obs.sequence_features:
        normalized = _normalize_signal_name(signal_name)
        if normalized in configured and normalized not in active:
            active.append(normalized)
    return active or configured_forecast_signals(cfg)


def clone_config_with_signal_overrides(cfg, overrides: dict[str, object] | None = None):
    """为单个 signal 训练生成局部配置副本。"""
    cloned = copy.deepcopy(cfg)
    if not overrides:
        return cloned

    unknown_keys = sorted(
        key for key in overrides if key not in SIGNAL_TRAINING_OVERRIDE_FIELDS and key not in {"future_horizon", "device"}
    )
    if unknown_keys:
        raise KeyError(
            "Unknown signal training override keys: "
            f"{unknown_keys}. Allowed keys: {sorted(SIGNAL_TRAINING_OVERRIDE_FIELDS)} + ['future_horizon', 'device']."
        )

    for key, value in overrides.items():
        if key == "future_horizon":
            cloned.env.future_horizon = int(value)
            continue
        if key == "device":
            cloned.runtime.device = resolve_device(value)
            continue
        setattr(cloned.forecast, SIGNAL_TRAINING_OVERRIDE_FIELDS[key], value)

    return cloned


def resolve_signal_training_settings(
    cfg,
    signal_name: str,
    overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    """解析单个 signal 的有效训练参数。"""
    signal_name = _normalize_signal_name(signal_name)
    local_cfg = clone_config_with_signal_overrides(cfg, overrides=overrides)
    settings = {
        "signal_name": signal_name,
        "history_window": int(local_cfg.forecast.history_window),
        "future_horizon": int(local_cfg.env.future_horizon),
        "hidden_size": int(local_cfg.forecast.lstm_hidden_size),
        "num_layers": int(local_cfg.forecast.lstm_num_layers),
        "dropout": float(local_cfg.forecast.lstm_dropout),
        "batch_size": int(local_cfg.forecast.lstm_batch_size),
        "epochs": int(local_cfg.forecast.lstm_epochs),
        "lr": float(local_cfg.forecast.lstm_lr),
        "train_ratio": float(local_cfg.forecast.lstm_train_ratio),
        "val_ratio": float(local_cfg.forecast.lstm_val_ratio),
        "device": str(local_cfg.runtime.device),
    }
    return local_cfg, settings


def forecast_artifact_root(cfg) -> Path:
    """Return the configured forecast artifact root."""
    root = cfg.forecast.lstm_artifact_root
    if root is None:
        return get_default_lstm_artifact_dir()
    return Path(root)


def expected_lstm_artifact_meta(
    cfg,
    signal_name: str,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """根据当前配置生成期望的 artifact meta。"""
    _, settings = resolve_signal_training_settings(cfg, signal_name, overrides=overrides)
    future_horizon = int(settings["future_horizon"])
    return {
        "artifact_format": "lstm_forecaster_v2",
        "signal_name": str(settings["signal_name"]),
        "future_horizon": future_horizon,
        "pred_len": future_horizon,
        "seq_len": int(settings["history_window"]),
        "hidden_size": int(settings["hidden_size"]),
        "num_layers": int(settings["num_layers"]),
        "dropout": float(settings["dropout"]),
    }


def compare_lstm_artifact_meta(
    actual_meta: dict[str, object] | None,
    expected_meta: dict[str, object],
) -> dict[str, object]:
    """比较 artifact meta 与当前配置是否一致，并返回可读差异。"""
    comparable_fields = (
        "artifact_format",
        "signal_name",
        "future_horizon",
        "pred_len",
        "seq_len",
        "hidden_size",
        "num_layers",
        "dropout",
    )
    actual_meta = dict(actual_meta or {})
    actual = {field: actual_meta.get(field) for field in comparable_fields}
    expected = {field: expected_meta.get(field) for field in comparable_fields}
    mismatches: dict[str, dict[str, object]] = {}

    for field in comparable_fields:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if field == "dropout":
            matches = actual_value is not None and bool(np.isclose(float(actual_value), float(expected_value)))
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches[field] = {
                "expected": expected_value,
                "actual": actual_value,
            }

    return {
        "compatible": not mismatches,
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }


def validate_lstm_artifact(
    cfg,
    signal_name: str,
    paths: dict[str, str | Path],
    *,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """校验单个 artifact 是否存在且与当前配置一致。"""
    normalized_signal = _normalize_signal_name(signal_name)
    expected = expected_lstm_artifact_meta(cfg, normalized_signal, overrides=overrides)
    resolved_paths = {name: Path(path) for name, path in paths.items()}
    required_files = {
        key: resolved_paths[key]
        for key in ("model_path", "meta_path", "scaler_path")
        if key in resolved_paths
    }
    existing_files = {key: path.exists() for key, path in required_files.items()}
    missing_files = [key for key, exists in existing_files.items() if not exists]
    any_existing = any(existing_files.values())
    result: dict[str, object] = {
        "signal_name": normalized_signal,
        "artifact_path": str(resolved_paths.get("model_path", "")),
        "paths": {key: str(path) for key, path in required_files.items()},
        "expected": expected,
        "actual": {},
        "mismatches": {},
        "missing_files": missing_files,
        "compatible": False,
        "issue_type": None,
    }

    if missing_files:
        result["issue_type"] = "incomplete" if any_existing else "missing"
        return result

    try:
        actual_meta = json.loads(required_files["meta_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["issue_type"] = "invalid_meta"
        result["error"] = f"无法读取 meta: {exc}"
        return result

    comparison = compare_lstm_artifact_meta(actual_meta, expected)
    result.update(comparison)
    result["issue_type"] = None if comparison["compatible"] else "mismatch"
    return result


def _signal_columns_from_header(columns: list[str], signal_name: str) -> list[str]:
    signal_name = _normalize_signal_name(signal_name)
    if signal_name == "price":
        return ["price"] if "price" in columns else []

    return [
        column
        for column in columns
        if column == signal_name or column.startswith(f"{signal_name}_") or column.startswith(signal_name)
    ]


def _candidate_signal_file_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    return [
        (data_dir / "simbench_2016_train.csv", data_dir / "simbench_2016_test.csv"),
        (data_dir / "simbench_train.csv", data_dir / "simbench_test.csv"),
        (data_dir / "train_prices.csv", data_dir / "test_prices.csv"),
    ]


def resolve_signal_csv_source(data_dir: str | Path, signal_name: str) -> SignalCsvSource | None:
    """Resolve the most suitable train/test CSV pair for one signal."""
    data_dir = Path(data_dir)
    signal_name = _normalize_signal_name(signal_name)

    for train_path, test_path in _candidate_signal_file_pairs(data_dir):
        if not train_path.exists() or not test_path.exists():
            continue

        train_columns = pd.read_csv(train_path, nrows=0).columns.tolist()
        test_columns = pd.read_csv(test_path, nrows=0).columns.tolist()
        value_columns = _signal_columns_from_header(train_columns, signal_name)
        if not value_columns:
            continue
        if any(column not in test_columns for column in value_columns):
            continue

        return SignalCsvSource(
            signal_name=signal_name,
            train_path=train_path,
            test_path=test_path,
            value_columns=tuple(value_columns),
        )

    return None


def load_signal_frame(csv_path: str | Path, signal_name: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load one CSV and select the columns belonging to one signal."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    value_columns = tuple(_signal_columns_from_header(df.columns.tolist(), signal_name))
    if not value_columns:
        raise ValueError(f"CSV '{csv_path}' does not contain signal columns for '{signal_name}'.")
    return df, value_columns


def load_signal_matrix(csv_path: str | Path, signal_name: str) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Load one signal matrix from CSV."""
    df, value_columns = load_signal_frame(csv_path, signal_name)
    values = df.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.reshape(-1)
    return df, values, value_columns


def _reshape_signal_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


def load_signal_segments(
    csv_path: str | Path,
    signal_name: str,
    *,
    drop_warmup: bool = False,
) -> tuple[pd.DataFrame, list[np.ndarray], tuple[str, ...]]:
    """Load one signal CSV and split it into contiguous, segment-safe arrays."""
    frame, value_columns = load_signal_frame(csv_path, signal_name)
    if drop_warmup and "is_warmup" in frame.columns:
        frame = frame.loc[~frame["is_warmup"].astype(bool)].copy()

    if frame.empty:
        raise ValueError(f"CSV '{csv_path}' does not contain any usable rows for signal '{signal_name}'.")

    working = frame.copy()
    if "segment_id" not in working.columns:
        working["segment_id"] = 0
    working["segment_id"] = pd.to_numeric(working["segment_id"], errors="coerce").fillna(-1).astype(int)

    segments: list[np.ndarray] = []
    for segment_id in sorted(working["segment_id"].unique()):
        segment_frame = working.loc[working["segment_id"] == segment_id]
        if segment_frame.empty:
            continue
        segment_values = segment_frame.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
        if segment_values.ndim == 2 and segment_values.shape[1] == 1:
            segment_values = segment_values.reshape(-1)
        segments.append(segment_values)

    if not segments:
        raise ValueError(f"CSV '{csv_path}' does not contain any segments for signal '{signal_name}'.")
    return working.reset_index(drop=True), segments, value_columns


def fit_signal_scaler(values: np.ndarray | list[np.ndarray]) -> StandardScaler:
    """Fit a scalar StandardScaler on all values of one signal family."""
    if isinstance(values, list):
        flattened = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in values], axis=0)
    else:
        flattened = np.asarray(values, dtype=np.float32).reshape(-1)

    scaler = StandardScaler()
    scaler.fit(flattened.reshape(-1, 1))
    return scaler


def summarize_signal_values(values: np.ndarray | list[np.ndarray]) -> dict[str, float]:
    """Return a compact numeric summary for debug logging."""
    if isinstance(values, list):
        flattened = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in values], axis=0)
    else:
        flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
    }


def temporal_split_matrix(
    values: np.ndarray,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, np.ndarray]:
    """Temporal train/val/test split for ``(T, D)`` matrices."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    total_steps = values.shape[0]
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= total_steps:
        raise ValueError("Invalid temporal split ratios for the given signal matrix.")

    return {
        "train": values[:train_end],
        "val": values[train_end:val_end],
        "test": values[val_end:],
        "train_end": train_end,
        "val_end": val_end,
    }


def build_supervised_windows_from_matrix(
    values: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    scaler: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a multi-column signal matrix into pooled univariate windows."""
    values = _reshape_signal_values(values)

    total_window = int(seq_len) + int(pred_len)
    if values.shape[0] < total_window:
        raise ValueError(
            f"Signal matrix is too short for seq_len={seq_len} and pred_len={pred_len}: shape={values.shape}"
        )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for column_idx in range(values.shape[1]):
        series = values[:, column_idx].astype(np.float32)
        if scaler is not None:
            series = scaler.transform(series.reshape(-1, 1)).reshape(-1).astype(np.float32)

        windows = np.lib.stride_tricks.sliding_window_view(series, total_window)
        x_parts.append(windows[:, :seq_len].astype(np.float32))
        y_parts.append(windows[:, seq_len:].astype(np.float32))

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def build_supervised_windows_from_segments(
    segments: list[np.ndarray],
    *,
    seq_len: int,
    pred_len: int,
    scaler: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pooled windows from a list of contiguous signal segments."""
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    skipped_segments = 0

    for segment in segments:
        segment_matrix = _reshape_signal_values(segment)
        if segment_matrix.shape[0] < int(seq_len) + int(pred_len):
            skipped_segments += 1
            continue
        x_chunk, y_chunk = build_supervised_windows_from_matrix(
            segment_matrix,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
        )
        x_parts.append(x_chunk)
        y_parts.append(y_chunk)

    if not x_parts:
        raise ValueError(
            "No valid supervised windows could be built from the provided segments. "
            f"All {len(segments)} segments were shorter than seq_len + pred_len."
        )
    if skipped_segments:
        print(f"[forecast] skipped {skipped_segments} short segment(s) while building supervised windows.")

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def temporal_split_segments(
    segments: list[np.ndarray],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seq_len: int,
    pred_len: int,
) -> dict[str, object]:
    """Temporal split performed independently inside each contiguous segment."""
    min_window = int(seq_len) + int(pred_len)
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Expected 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1.")

    train_segments: list[np.ndarray] = []
    val_segments: list[np.ndarray] = []
    segment_summaries: list[dict[str, int]] = []

    for segment_idx, raw_segment in enumerate(segments):
        segment = _reshape_signal_values(raw_segment)
        total_steps = int(segment.shape[0])
        train_end = int(total_steps * train_ratio)
        val_end = min(total_steps, int(total_steps * (train_ratio + val_ratio)))
        val_context_start = max(0, train_end - int(seq_len))

        if train_end >= min_window:
            train_segments.append(segment[:train_end].copy())
        if val_end - train_end > 0 and (val_end - val_context_start) >= min_window:
            val_segments.append(segment[val_context_start:val_end].copy())

        segment_summaries.append(
            {
                "segment_idx": int(segment_idx),
                "total_steps": total_steps,
                "train_end": int(train_end),
                "val_end": int(val_end),
            }
        )

    if not train_segments:
        raise ValueError("No train segments are long enough for the requested history/prediction windows.")
    if not val_segments:
        fallback = max(train_segments, key=lambda item: item.shape[0])
        fallback_tail = fallback[max(0, fallback.shape[0] - (2 * int(seq_len) + int(pred_len))) :]
        if fallback_tail.shape[0] < min_window:
            raise ValueError("No validation segments are long enough for the requested history/prediction windows.")
        val_segments = [fallback_tail.copy()]

    return {
        "train_segments": train_segments,
        "val_segments": val_segments,
        "segment_summaries": segment_summaries,
    }


def make_matrix_loader(
    values: np.ndarray | list[np.ndarray],
    *,
    seq_len: int,
    pred_len: int,
    scaler: object | None,
    batch_size: int,
    shuffle: bool,
    device: str | torch.device | TorchRuntimeState = "cpu",
    pin_memory: bool | None = None,
) -> DataLoader:
    """Create a DataLoader from a signal matrix."""
    if isinstance(values, list):
        x, y = build_supervised_windows_from_segments(
            values,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
        )
    else:
        x, y = build_supervised_windows_from_matrix(
            values,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
        )
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    resolved_device = resolve_device(device)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        pin_memory=resolved_device.type == "cuda" if pin_memory is None else bool(pin_memory),
    )


def train_lstm_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str | torch.device | TorchRuntimeState,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> dict[str, object]:
    """Train one LSTM and return the best state dict plus loss curves."""
    runtime_state = configure_torch_runtime(device)
    resolved_device = runtime_state.device
    model = model.to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    use_amp = resolved_device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    non_blocking = runtime_state.non_blocking_transfers and use_amp

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": []}

    epoch_iterator = range(int(epochs))
    progress = tqdm(
        epoch_iterator,
        desc=progress_label or "forecast epochs",
        leave=False,
        disable=not show_progress,
    )

    try:
        for _ in progress:
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(resolved_device, non_blocking=non_blocking)
                batch_y = batch_y.to(resolved_device, non_blocking=non_blocking)
                optimizer.zero_grad(set_to_none=True)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_amp
                    else nullcontext()
                )
                with autocast_context:
                    prediction = model(batch_x)
                    loss = criterion(prediction, batch_y)

                if use_amp:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(resolved_device, non_blocking=non_blocking)
                    batch_y = batch_y.to(resolved_device, non_blocking=non_blocking)
                    autocast_context = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_amp
                        else nullcontext()
                    )
                    with autocast_context:
                        prediction = model(batch_x)
                        val_loss = criterion(prediction, batch_y)
                    val_losses.append(float(val_loss.cpu().item()))

            mean_train = float(np.mean(train_losses)) if train_losses else 0.0
            mean_val = float(np.mean(val_losses)) if val_losses else mean_train
            history["train_loss"].append(mean_train)
            history["val_loss"].append(mean_val)

            if show_progress:
                progress.set_postfix(
                    {
                        "train": f"{mean_train:.4f}",
                        "val": f"{mean_val:.4f}",
                    }
                )

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                best_state_dict = copy.deepcopy(model.state_dict())
    finally:
        progress.close()

    model.load_state_dict(best_state_dict)
    return {
        "model": model,
        "history": history,
        "best_val_loss": float(best_val_loss),
        "best_state_dict": best_state_dict,
        "runtime": runtime_state,
    }


def build_runtime_forecaster_from_model(
    model,
    *,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> LSTMForecaster:
    """Wrap a trained model as a runtime forecaster for one signal."""
    forecaster = LSTMForecaster(
        model_path=None,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pred_len=pred_len,
        seq_len=seq_len,
        device=device,
        scaler=scaler,
    ).rename_default_signal(signal_name)
    forecaster.signal_runtimes[signal_name].model.load_state_dict(copy.deepcopy(model.state_dict()))
    forecaster.signal_runtimes[signal_name].model.eval()
    forecaster._set_legacy_attributes()
    return forecaster


def _select_week_evaluation_slice(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    signal_name: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one representative week for plotting."""
    if "week_id" in frame.columns and "is_warmup" in frame.columns:
        non_warmup = frame.loc[~frame["is_warmup"].astype(bool)]
        if not non_warmup.empty:
            first_target_row = non_warmup.iloc[0]
            target_week_id = int(first_target_row["week_id"])
            segment_mask = np.ones(len(frame), dtype=bool)
            if "segment_id" in frame.columns:
                segment_mask = frame["segment_id"].astype(int).to_numpy() == int(first_target_row["segment_id"])

            warmup_mask = segment_mask & frame["is_warmup"].astype(bool).to_numpy()
            target_mask = (
                segment_mask
                & ~frame["is_warmup"].astype(bool).to_numpy()
                & (frame["week_id"].astype(int).to_numpy() == target_week_id)
            )

            history_values = values[warmup_mask]
            target_values = values[target_mask]
            target_timestamps = frame.loc[target_mask, "timestamp"].to_numpy()
            if history_values.size == 0:
                segment_values = values[segment_mask]
                history_values = segment_values[: min(seq_len, len(segment_values))]
            return history_values, target_values, target_timestamps

    history = values[:seq_len]
    target = values[seq_len : seq_len + DEFAULT_WEEK_STEPS]
    if target.size == 0:
        target = values[seq_len:]
    if "timestamp" in frame.columns:
        timestamps = frame.loc[seq_len : seq_len + len(target) - 1, "timestamp"].to_numpy()
    else:
        timestamps = np.arange(len(target))
    return history, target, timestamps


def compute_forecast_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """计算周评估指标，默认基于展平后的全量值。"""
    target_array = np.asarray(target, dtype=np.float32)
    prediction_array = np.asarray(prediction, dtype=np.float32)
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            "Forecast metric computation requires target/prediction to share the same shape, "
            f"got {target_array.shape} vs {prediction_array.shape}."
        )

    flat_target = target_array.reshape(-1)
    flat_prediction = prediction_array.reshape(-1)
    error = flat_prediction - flat_target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))

    non_zero_mask = np.abs(flat_target) > 1e-8
    mape = None
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(error[non_zero_mask] / flat_target[non_zero_mask])) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def _to_forecaster_history_input(values: np.ndarray) -> np.ndarray:
    """把评估历史整理成 forecaster.predict 可消费的形状。"""
    history = np.asarray(values, dtype=np.float32)
    if history.ndim == 2 and history.shape[1] == 1:
        return history.reshape(-1)
    return history


def _reshape_evaluation_target(values: np.ndarray) -> np.ndarray:
    """统一评估 target 形状：单变量为 `(T,)`，多变量为 `(D, T)`。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values.reshape(-1)
    if values.ndim == 2 and values.shape[1] == 1:
        return values.reshape(-1)
    return values.T.astype(np.float32)


def evaluate_signal_open_loop_one_week(
    source: SignalCsvSource,
    model,
    *,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> SignalForecastEvaluation:
    """开环递推一周，用于长期 stress test。"""
    test_frame, test_values, _ = load_signal_matrix(source.test_path, signal_name)
    history_seed, target_values, timestamps = _select_week_evaluation_slice(
        test_frame,
        test_values,
        signal_name=signal_name,
        seq_len=seq_len,
    )

    forecaster = build_runtime_forecaster_from_model(
        model,
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    prediction = forecaster.predict(
        history_seed,
        horizon=target_values.shape[0] + 1,
        signal_name=signal_name,
    )
    if target_values.ndim == 1 or target_values.shape[1] == 1:
        prediction = np.asarray(prediction, dtype=np.float32)[1 : target_values.shape[0] + 1]
        target = target_values.reshape(-1).astype(np.float32)
    else:
        prediction = np.asarray(prediction, dtype=np.float32)[:, 1 : target_values.shape[0] + 1]
        target = target_values.T.astype(np.float32)

    return SignalForecastEvaluation(
        signal_name=signal_name,
        evaluation_mode="open_loop",
        timestamps=np.asarray(timestamps),
        target=target,
        prediction=np.asarray(prediction, dtype=np.float32),
        metrics=compute_forecast_metrics(target, prediction),
        source_columns=source.value_columns,
    )


def evaluate_signal_online_one_week(
    source: SignalCsvSource,
    model,
    *,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> SignalForecastEvaluation:
    """在线滚动评估一周，每个时刻都重新基于真实历史调用预测器。"""
    test_frame, test_values, _ = load_signal_matrix(source.test_path, signal_name)
    history_seed, target_values, timestamps = _select_week_evaluation_slice(
        test_frame,
        test_values,
        signal_name=signal_name,
        seq_len=seq_len,
    )

    forecaster = build_runtime_forecaster_from_model(
        model,
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    target_matrix = _reshape_signal_values(target_values)
    history_matrix = _reshape_signal_values(history_seed)
    one_step_predictions: list[np.ndarray | np.float32] = []

    for step_idx in range(target_matrix.shape[0]):
        if step_idx == 0:
            current_history = history_matrix
        else:
            current_history = np.concatenate([history_matrix, target_matrix[:step_idx]], axis=0)
        rollout = forecaster.predict(
            _to_forecaster_history_input(current_history),
            horizon=max(int(pred_len) + 1, 2),
            signal_name=signal_name,
        )
        rollout_array = np.asarray(rollout, dtype=np.float32)
        if target_matrix.shape[1] == 1:
            one_step_predictions.append(np.float32(rollout_array[1]))
        else:
            one_step_predictions.append(rollout_array[:, 1].astype(np.float32))

    target = _reshape_evaluation_target(target_matrix)
    if target_matrix.shape[1] == 1:
        prediction = np.asarray(one_step_predictions, dtype=np.float32)
    else:
        prediction = np.stack(one_step_predictions, axis=0).T.astype(np.float32)

    return SignalForecastEvaluation(
        signal_name=signal_name,
        evaluation_mode="online_aligned",
        timestamps=np.asarray(timestamps),
        target=target,
        prediction=prediction,
        metrics=compute_forecast_metrics(target, prediction),
        source_columns=source.value_columns,
    )


def evaluate_signal_one_week(
    source: SignalCsvSource,
    model,
    *,
    mode: str,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> SignalForecastEvaluation:
    """兼容旧入口，但要求显式指定评估协议，避免语义继续模糊。"""
    if mode == "online_aligned":
        return evaluate_signal_online_one_week(
            source,
            model,
            signal_name=signal_name,
            scaler=scaler,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
        )
    if mode == "open_loop":
        return evaluate_signal_open_loop_one_week(
            source,
            model,
            signal_name=signal_name,
            scaler=scaler,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
        )
    raise ValueError("Unknown evaluation mode, expected 'online_aligned' or 'open_loop'.")


def _managed_lstm_artifact_paths(cfg, signal_name: str) -> dict[str, Path]:
    """返回当前配置下单个 signal 的标准 artifact 路径。"""
    return get_default_lstm_artifact_paths(
        root=forecast_artifact_root(cfg),
        signal_name=_normalize_signal_name(signal_name),
        future_horizon=int(cfg.env.future_horizon),
    )


def _artifact_tuple_from_paths(paths: dict[str, Path]) -> tuple[str, str, str]:
    """把路径字典转成现有调用方使用的三元组。"""
    return (
        str(paths["model_path"]),
        str(paths["meta_path"]),
        str(paths["scaler_path"]),
    )


def _format_lstm_artifact_mismatches(mismatches: dict[str, dict[str, object]]) -> str:
    """把 mismatch 明细压缩成可读日志字符串。"""
    return ", ".join(
        f"{field}(expected={detail['expected']}, actual={detail['actual']})"
        for field, detail in sorted(mismatches.items())
    )


def _format_lstm_artifact_issue(validation: dict[str, object]) -> str:
    """生成面向用户的 artifact 问题说明。"""
    signal_name = validation["signal_name"]
    artifact_path = validation["artifact_path"]
    issue_type = validation.get("issue_type")
    if issue_type == "missing":
        missing_files = ", ".join(validation.get("missing_files", []))
        return (
            f"signal='{signal_name}', artifact='{artifact_path}', 缺少文件: {missing_files}. "
            "建议删除旧 artifact 残留，或开启 cfg.forecast.auto_train_missing=True 自动重训。"
        )
    if issue_type == "incomplete":
        missing_files = ", ".join(validation.get("missing_files", []))
        return (
            f"signal='{signal_name}', artifact='{artifact_path}', artifact 不完整，缺少: {missing_files}. "
            "建议删除旧 artifact 后重试，或开启 cfg.forecast.auto_train_missing=True 自动重训。"
        )
    if issue_type == "invalid_meta":
        return (
            f"signal='{signal_name}', artifact='{artifact_path}', {validation.get('error', 'meta 无法解析')}. "
            "建议删除旧 artifact，或开启 cfg.forecast.auto_train_missing=True 自动重训。"
        )
    mismatch_text = _format_lstm_artifact_mismatches(validation.get("mismatches", {}))
    return (
        f"signal='{signal_name}', artifact='{artifact_path}', 配置不一致: {mismatch_text}. "
        "建议删除旧 artifact，或开启 cfg.forecast.auto_train_missing=True 自动重训。"
    )


def _collect_lstm_artifact_inventory(cfg) -> dict[str, object]:
    """扫描所有 signal 的 artifact，并区分可用、缺失与不兼容。"""
    artifacts: dict[str, tuple[str, str, str]] = {}
    missing_signals: list[str] = []
    invalid_artifacts: dict[str, dict[str, object]] = {}

    for signal_name in configured_forecast_signals(cfg):
        paths = _managed_lstm_artifact_paths(cfg, signal_name)
        validation = validate_lstm_artifact(cfg, signal_name, paths)
        if validation["compatible"]:
            artifacts[signal_name] = _artifact_tuple_from_paths(paths)
            continue
        if validation.get("issue_type") == "missing":
            missing_signals.append(signal_name)
            continue
        invalid_artifacts[signal_name] = validation

    return {
        "artifacts": artifacts,
        "missing_signals": missing_signals,
        "invalid_artifacts": invalid_artifacts,
        "mismatched_signals": sorted(invalid_artifacts),
    }


def _raise_lstm_artifact_requirements_error(cfg, inventory: dict[str, object]) -> None:
    """在禁止自动重训时抛出带明细的 artifact 错误。"""
    problem_lines: list[str] = []
    for validation in inventory.get("invalid_artifacts", {}).values():
        problem_lines.append(f"- {_format_lstm_artifact_issue(validation)}")

    for signal_name in inventory.get("missing_signals", []):
        paths = _managed_lstm_artifact_paths(cfg, signal_name)
        missing_validation = validate_lstm_artifact(cfg, signal_name, paths)
        problem_lines.append(f"- {_format_lstm_artifact_issue(missing_validation)}")

    if not problem_lines:
        return

    mismatch_count = len(inventory.get("invalid_artifacts", {}))
    missing_count = len(inventory.get("missing_signals", []))
    if mismatch_count and missing_count:
        exc_type = RuntimeError
    elif mismatch_count:
        exc_type = ValueError
    else:
        exc_type = FileNotFoundError

    raise exc_type(
        "Managed LSTM forecast artifacts are missing or incompatible:\n"
        + "\n".join(problem_lines)
    )


def _print_signal_evaluation_summary(
    signal_name: str,
    online_evaluation: SignalForecastEvaluation,
    open_loop_evaluation: SignalForecastEvaluation,
) -> None:
    """打印训练后最关心的在线/开环指标。"""
    print(
        f"[forecast] {signal_name} metrics: "
        f"online_rmse={online_evaluation.metrics['rmse']:.6f}, "
        f"online_mae={online_evaluation.metrics['mae']:.6f}, "
        f"open_loop_rmse={open_loop_evaluation.metrics['rmse']:.6f}, "
        f"open_loop_mae={open_loop_evaluation.metrics['mae']:.6f}"
    )


def train_signal_lstm(
    cfg,
    signal_name: str,
    *,
    device: str | torch.device | TorchRuntimeState | None = None,
    overrides: dict[str, object] | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    """训练单个 signal 的 LSTM，并确保写回的 artifact 与当前配置一致。"""
    local_cfg, settings = resolve_signal_training_settings(cfg, signal_name, overrides=overrides)
    data_dir = Path(local_cfg.data.data_dir or (Path(__file__).resolve().parent.parent / "data"))
    runtime_state = configure_torch_runtime(
        local_cfg,
        device=local_cfg.runtime.device if device is None else device,
        seed=local_cfg.runtime.seed,
    )
    signal_name = _normalize_signal_name(signal_name)
    source = resolve_signal_csv_source(data_dir, signal_name)
    if source is None:
        raise FileNotFoundError(
            f"No train/test CSV pair found for forecast signal '{signal_name}' under '{data_dir}'."
        )

    _, train_segments, value_columns = load_signal_segments(
        source.train_path,
        signal_name,
        drop_warmup=True,
    )

    seq_len = int(local_cfg.forecast.history_window)
    pred_len = int(local_cfg.env.future_horizon)
    split = temporal_split_segments(
        train_segments,
        train_ratio=float(local_cfg.forecast.lstm_train_ratio),
        val_ratio=float(local_cfg.forecast.lstm_val_ratio),
        seq_len=seq_len,
        pred_len=pred_len,
    )
    train_values_only = split["train_segments"]
    val_context = split["val_segments"]
    train_stats = summarize_signal_values(train_values_only)

    print(
        f"[forecast] {signal_name}: train_segments={len(train_values_only)}, "
        f"val_segments={len(val_context)}, columns={list(value_columns)}, "
        f"train_stats={train_stats}, settings={settings}"
    )

    scaler = fit_signal_scaler(train_values_only)
    train_loader = make_matrix_loader(
        train_values_only,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        batch_size=int(local_cfg.forecast.lstm_batch_size),
        shuffle=True,
        device=runtime_state.device,
        pin_memory=runtime_state.pin_memory,
    )
    val_loader = make_matrix_loader(
        val_context,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        batch_size=int(local_cfg.forecast.lstm_batch_size),
        shuffle=False,
        device=runtime_state.device,
        pin_memory=runtime_state.pin_memory,
    )

    model = LSTMForecastModel(
        hidden_size=int(local_cfg.forecast.lstm_hidden_size),
        num_layers=int(local_cfg.forecast.lstm_num_layers),
        dropout=float(local_cfg.forecast.lstm_dropout),
        pred_len=pred_len,
    )
    result = train_lstm_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(local_cfg.forecast.lstm_epochs),
        lr=float(local_cfg.forecast.lstm_lr),
        device=runtime_state,
        show_progress=show_progress,
        progress_label=f"{signal_name} epochs",
    )

    artifact_paths = get_default_lstm_artifact_paths(
        root=forecast_artifact_root(local_cfg),
        signal_name=signal_name,
        future_horizon=pred_len,
    )
    artifact_paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    saved_paths = save_lstm_forecaster_artifacts(
        model_path=artifact_paths["model_path"],
        state_dict=result["best_state_dict"],
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=int(local_cfg.forecast.lstm_hidden_size),
        num_layers=int(local_cfg.forecast.lstm_num_layers),
        dropout=float(local_cfg.forecast.lstm_dropout),
        signal_name=signal_name,
        future_horizon=pred_len,
    )

    refreshed_validation = validate_lstm_artifact(local_cfg, signal_name, artifact_paths, overrides=overrides)
    if not refreshed_validation["compatible"]:
        raise RuntimeError(
            "Saved LSTM artifact failed validation after training: "
            f"{_format_lstm_artifact_issue(refreshed_validation)}"
        )
    print(f"[forecast] artifact refreshed at {saved_paths['model_path']}")

    online_evaluation = evaluate_signal_online_one_week(
        source,
        result["model"],
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=int(local_cfg.forecast.lstm_hidden_size),
        num_layers=int(local_cfg.forecast.lstm_num_layers),
        dropout=float(local_cfg.forecast.lstm_dropout),
        device=runtime_state.device,
    )
    open_loop_evaluation = evaluate_signal_open_loop_one_week(
        source,
        result["model"],
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=int(local_cfg.forecast.lstm_hidden_size),
        num_layers=int(local_cfg.forecast.lstm_num_layers),
        dropout=float(local_cfg.forecast.lstm_dropout),
        device=runtime_state.device,
    )
    _print_signal_evaluation_summary(signal_name, online_evaluation, open_loop_evaluation)

    return {
        "signal_name": signal_name,
        "source": source,
        "columns": value_columns,
        "segment_summary": split["segment_summaries"],
        "train_stats": train_stats,
        "artifact_paths": saved_paths,
        "training": result,
        "evaluation": online_evaluation,
        "online_evaluation": online_evaluation,
        "open_loop_evaluation": open_loop_evaluation,
        "settings": settings,
        "runtime": runtime_state,
    }


def collect_available_lstm_artifacts(cfg) -> dict[str, tuple[str, str, str]]:
    """只返回与当前配置兼容的 artifact。"""
    return dict(_collect_lstm_artifact_inventory(cfg)["artifacts"])


def ensure_lstm_artifacts(cfg, *, device: str | torch.device | None = None) -> dict[str, object]:
    """确保当前配置需要的 LSTM artifact 已存在，不兼容时按策略报错或重训。"""
    if cfg.forecast.lstm_model_path is not None:
        return {
            "mode": "legacy_single_model",
            "artifacts": {
                "price": (
                    str(cfg.forecast.lstm_model_path),
                    None,
                    None,
                )
            },
            "trained_signals": [],
            "retrained_signals": [],
            "mismatched_signals": [],
            "plot_path": None,
        }

    artifact_root = forecast_artifact_root(cfg)
    artifact_root.mkdir(parents=True, exist_ok=True)

    inventory_before = _collect_lstm_artifact_inventory(cfg)
    if not bool(cfg.forecast.auto_train_missing) and (
        inventory_before["missing_signals"] or inventory_before["invalid_artifacts"]
    ):
        _raise_lstm_artifact_requirements_error(cfg, inventory_before)

    trained_results: list[dict[str, object]] = []
    trained_signals: list[str] = []
    retrained_signals: list[str] = []
    data_dir = Path(cfg.data.data_dir or (Path(__file__).resolve().parent.parent / "data"))

    for signal_name in configured_forecast_signals(cfg):
        if signal_name in inventory_before["artifacts"]:
            continue

        invalid_validation = inventory_before["invalid_artifacts"].get(signal_name)
        if invalid_validation is not None:
            print(f"[forecast] artifact mismatch detected: {_format_lstm_artifact_issue(invalid_validation)}")

        source = resolve_signal_csv_source(data_dir, signal_name)
        if source is None:
            print(f"[forecast] skip '{signal_name}': no matching train/test CSV source found.")
            continue

        if not bool(cfg.forecast.auto_train_missing):
            continue

        print(f"[forecast] retraining signal={signal_name}")
        trained_result = train_signal_lstm(cfg, signal_name, device=device)
        trained_results.append(trained_result)
        if invalid_validation is not None:
            retrained_signals.append(signal_name)
        else:
            trained_signals.append(signal_name)

    inventory_after = _collect_lstm_artifact_inventory(cfg)
    if not bool(cfg.forecast.auto_train_missing) and (
        inventory_after["missing_signals"] or inventory_after["invalid_artifacts"]
    ):
        _raise_lstm_artifact_requirements_error(cfg, inventory_after)

    evaluations = [result["evaluation"] for result in trained_results]
    plot_path = get_weekly_forecast_plot_path(
        future_horizon=int(cfg.env.future_horizon),
        root=artifact_root,
    )
    if evaluations:
        plot_path = plot_weekly_forecasts(
            evaluations,
            save_path=plot_path,
        )
    elif not plot_path.exists():
        plot_path = None

    return {
        "mode": "managed_multi_signal",
        "artifacts": dict(inventory_after["artifacts"]),
        "trained_signals": trained_signals,
        "retrained_signals": retrained_signals,
        "mismatched_signals": list(inventory_before["mismatched_signals"]),
        "invalid_artifacts": dict(inventory_before["invalid_artifacts"]),
        "missing_signals": list(inventory_after["missing_signals"]),
        "plot_path": str(plot_path) if plot_path is not None else None,
    }


def _plot_evaluation_curve(axis, evaluation: SignalForecastEvaluation) -> None:
    """Draw one evaluation panel using English labels to avoid font warnings."""
    if evaluation.target.ndim == 1:
        target_curve = evaluation.target
        prediction_curve = evaluation.prediction
        ylabel = evaluation.signal_name
    else:
        target_curve = evaluation.target.sum(axis=0)
        prediction_curve = evaluation.prediction.sum(axis=0)
        ylabel = f"{evaluation.signal_name} (sum)"

    mode_title = {
        "online_aligned": "Online aligned forecast",
        "open_loop": "Open-loop stress test",
    }.get(evaluation.evaluation_mode, str(evaluation.evaluation_mode))
    prediction_label = "Prediction" if evaluation.evaluation_mode == "online_aligned" else "Open-loop prediction"

    axis.plot(evaluation.timestamps, target_curve, label="Ground truth", linewidth=1.5)
    axis.plot(evaluation.timestamps, prediction_curve, label=prediction_label, linewidth=1.5)
    axis.set_title(f"{evaluation.signal_name} - {mode_title}")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")


def plot_weekly_forecasts(
    evaluations: list[SignalForecastEvaluation],
    *,
    save_path: str | Path,
) -> Path:
    """Save one weekly comparison figure for several signals."""
    if not evaluations:
        raise ValueError("plot_weekly_forecasts requires at least one evaluation result.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(evaluations),
        1,
        figsize=(14, 4.2 * len(evaluations)),
        sharex=False,
    )
    axes = list(np.atleast_1d(axes))

    for axis, evaluation in zip(axes, evaluations):
        _plot_evaluation_curve(axis, evaluation)

    axes[-1].set_xlabel("timestamp")
    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    return save_path


def plot_signal_training_report(
    result: dict[str, object],
    *,
    figsize: tuple[float, float] = (18.0, 4.8),
):
    """Draw loss, online-aligned forecast, and optional open-loop stress test."""
    history = result["training"]["history"]
    evaluation = result["evaluation"]
    open_loop_evaluation = result.get("open_loop_evaluation")
    signal_name = str(result["signal_name"])

    subplot_count = 3 if open_loop_evaluation is not None else 2
    figure, axes = plt.subplots(1, subplot_count, figsize=figsize)
    axes = list(np.atleast_1d(axes))

    axes[0].plot(history["train_loss"], label="Train", linewidth=1.6)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=1.6)
    axes[0].set_title(f"{signal_name} - Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    _plot_evaluation_curve(axes[1], evaluation)
    axes[1].set_xlabel("timestamp")

    if open_loop_evaluation is not None:
        _plot_evaluation_curve(axes[2], open_loop_evaluation)
        axes[2].set_xlabel("timestamp")

    figure.tight_layout()
    return figure
