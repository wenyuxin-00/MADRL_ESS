"""Reusable diagnostics and offline experiments for SFH14 load forecasts."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from predictors.lstm_forecaster import load_lstm_forecaster_artifacts
from predictors.lstm_model import LSTMForecastModel
from predictors.time_features import (
    encode_forecast_time_features,
    normalize_time_feature_mode,
    time_feature_dim,
)
from predictors.training import (
    _load_signal_segment_frames_from_source,
    _restore_supervised_window_values,
    _select_load_blend_weight_from_validation,
    build_supervised_windows_from_time_feature_frames,
    collect_available_lstm_artifacts,
    compute_forecast_metrics,
    fit_signal_scaler,
    load_signal_matrix_from_source,
    make_tensor_loader,
    resolve_signal_csv_source,
    summarize_signal_values,
    temporal_split_segment_frames,
    train_lstm_model,
    train_signal_lstm,
)
from scripts.utils.torch_runtime import TorchRuntimeState, configure_torch_runtime

DEFAULT_ANALYSIS_PROFILE = "SFH14"
DEFAULT_BLOCKED_MONTH_GROUPS = (
    ("Jan-Mar", (1, 2, 3)),
    ("Apr-Jun", (4, 5, 6)),
    ("Jul-Sep", (7, 8, 9)),
    ("Oct-Dec", (10, 11, 12)),
)
DEFAULT_MONTHLY_BIAS_CAP_KW = 0.77
DEFAULT_BIAS_THRESHOLD_KW = 0.15
DEFAULT_CONTROL_DEGRADATION_PCT = 2.0
DEFAULT_CURRENT_BLEND_UPPER_BOUND = 0.557490
DEFAULT_BASELINE_TARGET = 0.474082


def _cuda_runtime_active(runtime_state: TorchRuntimeState) -> bool:
    return runtime_state.device.type == "cuda" and torch.cuda.is_available()


def _clear_runtime_memory(runtime_state: TorchRuntimeState | None = None) -> None:
    gc.collect()
    if runtime_state is None or not _cuda_runtime_active(runtime_state):
        return
    try:
        torch.cuda.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@dataclass(frozen=True)
class Step1WindowPayload:
    """Step-1 prediction payload for one profile on one split."""

    profile: str
    timestamps: pd.DatetimeIndex
    target: np.ndarray
    baseline: np.ndarray
    raw: np.ndarray
    blended: np.ndarray
    blend_weight: float


def apply_heatpump_jump_relief(
    payload: Step1WindowPayload,
    *,
    enabled: bool,
    threshold_kw: float = 0.8,
    min_weight: float = 0.3,
) -> dict[str, object]:
    """Raise the effective heatpump blend weight only on large positive raw-vs-baseline jumps."""

    threshold_kw = float(threshold_kw)
    min_weight = float(min_weight)
    if threshold_kw < 0.0:
        raise ValueError(f"threshold_kw must be non-negative, got {threshold_kw}.")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(f"min_weight must be between 0 and 1, got {min_weight}.")

    base_weight = float(payload.blend_weight)
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError(f"payload.blend_weight must be between 0 and 1, got {base_weight}.")

    baseline = np.asarray(payload.baseline, dtype=np.float32)
    raw = np.asarray(payload.raw, dtype=np.float32)
    raw_gap = raw - baseline
    effective_weights = np.full(raw.shape, np.float32(base_weight), dtype=np.float32)

    trigger_mask = np.zeros(raw.shape, dtype=bool)
    if bool(enabled):
        trigger_mask = (raw_gap > np.float32(threshold_kw)) & (raw > baseline)
        effective_weights[trigger_mask] = np.float32(max(base_weight, min_weight))

    adjusted_blended = (baseline + effective_weights * raw_gap).astype(np.float32)
    adjusted_payload = Step1WindowPayload(
        profile=str(payload.profile),
        timestamps=pd.DatetimeIndex(payload.timestamps),
        target=np.asarray(payload.target, dtype=np.float32).copy(),
        baseline=baseline.copy(),
        raw=raw.copy(),
        blended=adjusted_blended,
        blend_weight=base_weight,
    )
    return {
        "payload": adjusted_payload,
        "base_weight": base_weight,
        "effective_weights": effective_weights,
        "trigger_mask": trigger_mask,
        "threshold_kw": threshold_kw,
        "min_weight": min_weight,
        "triggered_count": int(np.sum(trigger_mask)),
        "triggered_timestamps": pd.DatetimeIndex(payload.timestamps[trigger_mask]),
    }


@dataclass(frozen=True)
class RolloutWindowPayload:
    """Recursive multi-step rollout payload for one profile on one split."""

    profile: str
    target_timestamps: np.ndarray
    target: np.ndarray
    baseline: np.ndarray
    raw: np.ndarray
    blended: np.ndarray
    blend_weight: float


@dataclass(frozen=True)
class InMemoryModelArtifacts:
    """In-memory model payload used by offline experiments."""

    profile: str
    model: LSTMForecastModel
    scaler: object | None
    seq_len: int
    pred_len: int
    input_size: int
    time_feature_mode: str
    blend_weight: float
    val_payload: Step1WindowPayload
    train_stats: dict[str, float]


def _resolve_runtime_state(runtime_state: TorchRuntimeState | str | torch.device | None) -> TorchRuntimeState:
    return configure_torch_runtime(runtime_state or "cpu")


def _deepcopy_cfg(cfg):
    return deepcopy(cfg)


def _resolve_load_source(cfg):
    data_dir = Path(cfg.data.data_dir)
    source = resolve_signal_csv_source(data_dir, "load", cfg=cfg)
    if source is None:
        raise ValueError("Could not resolve a load signal source from the current config.")
    return source


def _resolve_load_artifact_bundle(
    cfg,
    *,
    load_result: dict[str, object] | None = None,
    load_overrides: dict[str, object] | None = None,
) -> list[tuple[str, str, str]]:
    if load_result is not None:
        artifact_paths = load_result.get("artifact_paths")
        if isinstance(artifact_paths, list):
            return [
                (
                    str(item["model_path"]),
                    str(item["meta_path"]),
                    str(item["scaler_path"]),
                )
                for item in artifact_paths
            ]
        if isinstance(artifact_paths, tuple) and len(artifact_paths) == 3:
            return [tuple(str(value) for value in artifact_paths)]

    overrides_by_signal = {"load": dict(load_overrides or {})} if load_overrides else None
    artifact_map = collect_available_lstm_artifacts(cfg, overrides_by_signal=overrides_by_signal)
    bundle = artifact_map.get("load")
    if bundle is None:
        raise FileNotFoundError("Managed load artifacts are not available for diagnostics.")
    if isinstance(bundle, list):
        return [tuple(str(value) for value in item) for item in bundle]
    return [tuple(str(value) for value in bundle)]


def _load_component_frame(
    data_dir: str | Path,
    *,
    component: str,
    profiles: Sequence[str],
    year: int,
) -> pd.DataFrame:
    csv_path = Path(data_dir) / "processed" / "prosumer" / f"{component}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing processed component CSV: '{csv_path}'.")
    usecols = ["timestamp", *profiles]
    frame = pd.read_csv(csv_path, usecols=usecols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    filtered = frame.loc[frame["timestamp"].dt.year == int(year)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"Component '{component}' has no rows for year={year}.")
    return filtered


def build_component_drift_report(
    cfg,
    *,
    profiles: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare household, heatpump, and total-load distributions across train/test years."""

    data_dir = Path(cfg.data.data_dir)
    profiles = list(profiles or cfg.data.agent_profiles)
    household_train = _load_component_frame(data_dir, component="household", profiles=profiles, year=cfg.data.train_year)
    household_test = _load_component_frame(data_dir, component="household", profiles=profiles, year=cfg.data.test_year)
    heatpump_train = _load_component_frame(data_dir, component="heatpump", profiles=profiles, year=cfg.data.train_year)
    heatpump_test = _load_component_frame(data_dir, component="heatpump", profiles=profiles, year=cfg.data.test_year)

    rows: list[dict[str, object]] = []
    for split_name, household_frame, heatpump_frame in (
        ("train", household_train, heatpump_train),
        ("test", household_test, heatpump_test),
    ):
        total_frame = household_frame.copy()
        for profile in profiles:
            total_frame[profile] = household_frame[profile].astype(np.float32) + heatpump_frame[profile].astype(np.float32)

        for component_name, component_frame in (
            ("household", household_frame),
            ("heatpump", heatpump_frame),
            ("total", total_frame),
        ):
            for profile in profiles:
                series = component_frame[profile].astype(np.float32)
                rows.append(
                    {
                        "profile": str(profile),
                        "component": component_name,
                        "split": split_name,
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "p95": float(series.quantile(0.95)),
                        "max": float(series.max()),
                        "zero_frac": float((np.abs(series.to_numpy()) <= 1e-8).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _build_window_timestamps(
    segment_frames: Sequence[pd.DataFrame],
    *,
    seq_len: int,
    pred_len: int,
) -> pd.DatetimeIndex:
    timestamps: list[pd.DatetimeIndex] = []
    total_window = int(seq_len) + int(pred_len)
    for segment_frame in segment_frames:
        frame = segment_frame.reset_index(drop=True)
        n_windows = len(frame) - total_window + 1
        if n_windows <= 0:
            continue
        window_timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                frame["timestamp"].iloc[int(seq_len) : int(seq_len) + int(n_windows)].tolist(),
                utc=True,
            )
        )
        timestamps.append(window_timestamps)
    if not timestamps:
        return pd.DatetimeIndex([])
    combined = timestamps[0]
    for chunk in timestamps[1:]:
        combined = combined.append(chunk)
    return combined


def _build_feature_window_view(feature_matrix: np.ndarray, window_size: int) -> np.ndarray:
    matrix = np.asarray(feature_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}.")
    if matrix.shape[0] < int(window_size):
        return np.zeros((0, int(window_size), matrix.shape[1]), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(matrix, int(window_size), axis=0)
    return np.transpose(windows, (0, 2, 1)).astype(np.float32, copy=False)


def _predict_next_step_batch(
    model: LSTMForecastModel,
    histories: np.ndarray,
    *,
    scaler: object | None,
    time_windows: np.ndarray | None,
    input_size: int,
    time_feature_mode: str,
    runtime_state: TorchRuntimeState,
    batch_size: int = 4096,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    normalized_mode = normalize_time_feature_mode(time_feature_mode if int(input_size) > 1 else "none")
    use_amp = _cuda_runtime_active(runtime_state)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(histories), max(int(batch_size), 1)):
            end = start + max(int(batch_size), 1)
            history_batch = np.asarray(histories[start:end], dtype=np.float32)
            if scaler is not None:
                history_batch = scaler.transform(history_batch.reshape(-1, 1)).reshape(history_batch.shape).astype(np.float32)
            if int(input_size) <= 1 or normalized_mode == "none":
                features = history_batch
            else:
                if time_windows is None:
                    raise ValueError("time_windows are required when input_size > 1.")
                features = np.concatenate(
                    [history_batch[:, :, None], np.asarray(time_windows[start:end], dtype=np.float32)],
                    axis=2,
                ).astype(np.float32)
            tensor = torch.tensor(features, dtype=torch.float32, device=runtime_state.device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                batch_prediction = model(tensor)
            batch_prediction = batch_prediction.detach().float().cpu().numpy().astype(np.float32)[:, 0]
            del tensor
            if scaler is not None:
                batch_prediction = scaler.inverse_transform(batch_prediction.reshape(-1, 1)).reshape(-1).astype(np.float32)
            predictions.append(batch_prediction.astype(np.float32))
    if not predictions:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(predictions, axis=0).astype(np.float32)


def _compute_recursive_mode_rollout_matrix(
    history_windows: np.ndarray,
    *,
    model: LSTMForecastModel,
    scaler: object | None,
    time_window_view: np.ndarray | None,
    seq_len: int,
    pred_len: int,
    time_feature_mode: str,
    input_size: int,
    blend_weight: float,
    mode: str,
    runtime_state: TorchRuntimeState,
    batch_size: int = 4096,
) -> np.ndarray:
    if mode not in {"baseline", "raw", "blended"}:
        raise ValueError(f"Unsupported rollout mode '{mode}'.")
    windows = np.asarray(history_windows, dtype=np.float32)
    n_windows = int(windows.shape[0])
    if n_windows == 0:
        return np.zeros((0, int(pred_len)), dtype=np.float32)
    if mode == "baseline":
        return np.repeat(windows[:, -1:], int(pred_len), axis=1).astype(np.float32)

    rolling_history = windows.copy()
    predictions = np.empty((n_windows, int(pred_len)), dtype=np.float32)
    weight = np.float32(float(blend_weight))
    normalized_mode = normalize_time_feature_mode(time_feature_mode if int(input_size) > 1 else "none")
    for step_idx in range(int(pred_len)):
        current_time_windows = None
        if int(input_size) > 1 and normalized_mode != "none":
            if time_window_view is None:
                raise ValueError("time_window_view is required when input_size > 1.")
            current_time_windows = time_window_view[step_idx : step_idx + n_windows]
        raw_next = _predict_next_step_batch(
            model,
            rolling_history,
            scaler=scaler,
            time_windows=current_time_windows,
            input_size=input_size,
            time_feature_mode=normalized_mode,
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        if mode == "raw":
            next_values = raw_next
        else:
            baseline_next = rolling_history[:, -1]
            next_values = (baseline_next + weight * (raw_next - baseline_next)).astype(np.float32)
        predictions[:, step_idx] = next_values.astype(np.float32)
        if step_idx + 1 < int(pred_len):
            rolling_history = np.concatenate([rolling_history[:, 1:], next_values[:, None]], axis=1).astype(np.float32)
    return predictions.astype(np.float32)


def _compute_rollout_payload(
    *,
    profile: str,
    segment_frames: Sequence[pd.DataFrame],
    value_column: str,
    model: LSTMForecastModel,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    time_feature_mode: str,
    input_size: int,
    blend_weight: float,
    runtime_state: TorchRuntimeState,
    batch_size: int = 4096,
) -> RolloutWindowPayload:
    target_matrices: list[np.ndarray] = []
    timestamp_matrices: list[np.ndarray] = []
    baseline_matrices: list[np.ndarray] = []
    raw_matrices: list[np.ndarray] = []
    blended_matrices: list[np.ndarray] = []
    normalized_mode = normalize_time_feature_mode(time_feature_mode if int(input_size) > 1 else "none")

    for segment_frame in segment_frames:
        frame = segment_frame.loc[:, ["timestamp", value_column]].copy().reset_index(drop=True)
        values = frame[value_column].to_numpy(dtype=np.float32)
        if len(values) < int(seq_len) + int(pred_len):
            continue
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        n_windows = len(values) - int(seq_len) - int(pred_len) + 1
        history_windows = np.lib.stride_tricks.sliding_window_view(values, int(seq_len))[:n_windows].astype(np.float32, copy=True)
        target_windows = np.lib.stride_tricks.sliding_window_view(values[int(seq_len) :], int(pred_len))[:n_windows].astype(np.float32, copy=True)
        target_timestamps = np.lib.stride_tricks.sliding_window_view(
            timestamps.to_numpy(dtype="datetime64[ns]")[int(seq_len) :],
            int(pred_len),
        )[:n_windows].copy()
        time_window_view = None
        if int(input_size) > 1 and normalized_mode != "none":
            encoded_features = encode_forecast_time_features(timestamps, normalized_mode)
            time_window_view = _build_feature_window_view(encoded_features, int(seq_len))

        baseline_windows = _compute_recursive_mode_rollout_matrix(
            history_windows,
            model=model,
            scaler=scaler,
            time_window_view=time_window_view,
            seq_len=seq_len,
            pred_len=pred_len,
            time_feature_mode=normalized_mode,
            input_size=input_size,
            blend_weight=blend_weight,
            mode="baseline",
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        raw_windows = _compute_recursive_mode_rollout_matrix(
            history_windows,
            model=model,
            scaler=scaler,
            time_window_view=time_window_view,
            seq_len=seq_len,
            pred_len=pred_len,
            time_feature_mode=normalized_mode,
            input_size=input_size,
            blend_weight=blend_weight,
            mode="raw",
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        blended_windows = _compute_recursive_mode_rollout_matrix(
            history_windows,
            model=model,
            scaler=scaler,
            time_window_view=time_window_view,
            seq_len=seq_len,
            pred_len=pred_len,
            time_feature_mode=normalized_mode,
            input_size=input_size,
            blend_weight=blend_weight,
            mode="blended",
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        target_matrices.append(target_windows.astype(np.float32))
        timestamp_matrices.append(target_timestamps)
        baseline_matrices.append(baseline_windows.astype(np.float32))
        raw_matrices.append(raw_windows.astype(np.float32))
        blended_matrices.append(blended_windows.astype(np.float32))

    if not target_matrices:
        raise ValueError(f"No rollout windows were available for profile='{profile}' and value_column='{value_column}'.")

    return RolloutWindowPayload(
        profile=str(profile),
        target_timestamps=np.concatenate(timestamp_matrices, axis=0),
        target=np.concatenate(target_matrices, axis=0).astype(np.float32),
        baseline=np.concatenate(baseline_matrices, axis=0).astype(np.float32),
        raw=np.concatenate(raw_matrices, axis=0).astype(np.float32),
        blended=np.concatenate(blended_matrices, axis=0).astype(np.float32),
        blend_weight=float(blend_weight),
    )


def _build_model_from_meta(meta: dict[str, object]) -> LSTMForecastModel:
    return LSTMForecastModel(
        hidden_size=int(meta["hidden_size"]),
        num_layers=int(meta["num_layers"]),
        dropout=float(meta["dropout"]),
        pred_len=int(meta["pred_len"]),
        input_size=int(meta.get("input_size", 1)),
    )


def _predict_supervised_windows(
    model: LSTMForecastModel,
    windows: np.ndarray,
    *,
    runtime_state: TorchRuntimeState,
    batch_size: int = 4096,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    use_amp = _cuda_runtime_active(runtime_state)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), max(int(batch_size), 1)):
            batch = torch.tensor(
                windows[start : start + max(int(batch_size), 1)],
                dtype=torch.float32,
                device=runtime_state.device,
            )
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                batch_prediction = model(batch)
            predictions.append(batch_prediction.detach().float().cpu().numpy().astype(np.float32))
            del batch
    if not predictions:
        raise ValueError("No supervised windows were available for prediction.")
    return np.concatenate(predictions, axis=0).astype(np.float32)


def _profile_from_column(column_name: str, fallback: str | None = None) -> str:
    if fallback:
        return str(fallback)
    if "_" in str(column_name):
        return str(column_name).split("_", maxsplit=1)[1]
    return str(column_name)


def _restore_values(
    values: np.ndarray,
    *,
    scaler: object | None,
) -> np.ndarray:
    return _restore_supervised_window_values(values, scaler=scaler, physical_scale=None)


def _compute_step1_payload(
    *,
    profile: str,
    segment_frames: Sequence[pd.DataFrame],
    value_column: str,
    model: LSTMForecastModel,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    time_feature_mode: str,
    input_size: int,
    blend_weight: float,
    runtime_state: TorchRuntimeState,
    batch_size: int = 4096,
) -> Step1WindowPayload:
    frames = [frame.loc[:, ["timestamp", value_column]].copy() for frame in segment_frames]
    x, y = build_supervised_windows_from_time_feature_frames(
        frames,
        value_column=value_column,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        physical_scale_by_column=None,
        time_feature_mode=time_feature_mode if int(input_size) > 1 else "none",
    )
    original_device = _module_device(model)
    restore_model = original_device != runtime_state.device
    if restore_model:
        model = model.to(runtime_state.device)
    try:
        raw_predictions = _predict_supervised_windows(
            model,
            x,
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        raw_step1 = _restore_values(raw_predictions[:, 0], scaler=scaler)
        baseline_step1 = _restore_values(x[:, -1, 0], scaler=scaler)
        target_step1 = _restore_values(y[:, 0], scaler=scaler)
        blended_step1 = baseline_step1 + np.float32(blend_weight) * (raw_step1 - baseline_step1)
        timestamps = _build_window_timestamps(frames, seq_len=seq_len, pred_len=pred_len)
        if len(timestamps) != len(target_step1):
            raise ValueError(
                "Step-1 timestamps did not align with target windows, "
                f"got len(timestamps)={len(timestamps)} vs len(target)={len(target_step1)}."
            )
        return Step1WindowPayload(
            profile=str(profile),
            timestamps=timestamps,
            target=target_step1.astype(np.float32),
            baseline=baseline_step1.astype(np.float32),
            raw=raw_step1.astype(np.float32),
            blended=blended_step1.astype(np.float32),
            blend_weight=float(blend_weight),
        )
    finally:
        if restore_model:
            model.to(original_device)
            _clear_runtime_memory(runtime_state)


def _compute_metric_row(
    *,
    split_name: str,
    profile: str,
    mode: str,
    prediction: np.ndarray,
    target: np.ndarray,
    blend_weight: float,
) -> dict[str, object]:
    metrics = compute_forecast_metrics(target, prediction)
    error = np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    return {
        "split": split_name,
        "profile": profile,
        "mode": mode,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mape": None if metrics["mape"] is None else float(metrics["mape"]),
        "bias": float(np.mean(error)),
        "blend_weight": float(blend_weight),
        "n_windows": int(len(target)),
    }


def _payload_metric_rows(split_name: str, payload: Step1WindowPayload) -> list[dict[str, object]]:
    return [
        _compute_metric_row(
            split_name=split_name,
            profile=payload.profile,
            mode="baseline",
            prediction=payload.baseline,
            target=payload.target,
            blend_weight=payload.blend_weight,
        ),
        _compute_metric_row(
            split_name=split_name,
            profile=payload.profile,
            mode="raw",
            prediction=payload.raw,
            target=payload.target,
            blend_weight=payload.blend_weight,
        ),
        _compute_metric_row(
            split_name=split_name,
            profile=payload.profile,
            mode="blended",
            prediction=payload.blended,
            target=payload.target,
            blend_weight=payload.blend_weight,
        ),
    ]


def _load_artifact_bundle_entries(
    artifact_bundle: Sequence[tuple[str, str, str]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for model_path, meta_path, scaler_path in artifact_bundle:
        meta, scaler = load_lstm_forecaster_artifacts(
            model_path=model_path,
            meta_path=meta_path,
            scaler_path=scaler_path,
        )
        entries.append(
            {
                "model_path": str(model_path),
                "meta_path": str(meta_path),
                "scaler_path": str(scaler_path),
                "meta": meta,
                "scaler": scaler,
                "profile": str(meta.get("agent_profile") or meta.get("value_column") or ""),
                "component": meta.get("component"),
            }
        )
    return entries


def _load_model_from_artifact_entry(
    entry: dict[str, object],
    *,
    runtime_state: TorchRuntimeState,
) -> LSTMForecastModel:
    meta = dict(entry["meta"])
    model = _build_model_from_meta(meta)
    state_dict = torch.load(str(entry["model_path"]), map_location=runtime_state.device)
    model.load_state_dict(state_dict)
    return model.to(runtime_state.device)


def _infer_segment_year(segment_frames: Sequence[pd.DataFrame]) -> int:
    for frame in segment_frames:
        if "timestamp" in frame.columns and not frame.empty:
            return int(pd.Timestamp(frame["timestamp"].iloc[0]).year)
    raise ValueError("Could not infer a calendar year from the provided segment frames.")


def _load_component_segment_frames(
    cfg,
    *,
    segment_frames: Sequence[pd.DataFrame],
    component: str,
    profiles: Sequence[str],
) -> list[pd.DataFrame]:
    csv_path = Path(cfg.data.data_dir) / "processed" / "prosumer" / f"{component}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing processed component CSV: '{csv_path}'.")
    value_columns = [str(profile) for profile in profiles]
    component_frame = pd.read_csv(csv_path, usecols=["timestamp", *value_columns])
    component_frame["timestamp"] = pd.to_datetime(component_frame["timestamp"], utc=True)
    requested_timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            [
                timestamp
                for segment_frame in segment_frames
                for timestamp in segment_frame["timestamp"].tolist()
            ],
            utc=True,
        )
    )
    component_frame = component_frame.loc[component_frame["timestamp"].isin(requested_timestamps)].reset_index(drop=True)
    aligned_frames: list[pd.DataFrame] = []
    for segment_frame in segment_frames:
        aligned = segment_frame.loc[:, ["timestamp"]].merge(
            component_frame.loc[:, ["timestamp", *value_columns]],
            on="timestamp",
            how="left",
        )
        if aligned[value_columns].isna().any().any():
            raise ValueError(f"Component frame '{component}' did not align with the requested segment timestamps.")
        aligned_frames.append(aligned.reset_index(drop=True))
    return aligned_frames


def _combine_component_payloads(
    *,
    profile: str,
    component_payloads: dict[str, Step1WindowPayload],
) -> Step1WindowPayload:
    if not component_payloads:
        raise ValueError(f"Component payloads are required to recombine profile='{profile}'.")
    ordered_payloads = [component_payloads[key] for key in sorted(component_payloads)]
    reference = ordered_payloads[0]
    for payload in ordered_payloads[1:]:
        if payload.target.shape != reference.target.shape:
            raise ValueError(f"Component payload shapes do not align for profile='{profile}'.")
        if not np.array_equal(payload.timestamps, reference.timestamps):
            raise ValueError(f"Component timestamps do not align for profile='{profile}'.")
    preferred_payload = component_payloads.get("heatpump", reference)
    return Step1WindowPayload(
        profile=str(profile),
        timestamps=reference.timestamps,
        target=np.sum([payload.target for payload in ordered_payloads], axis=0, dtype=np.float32).astype(np.float32),
        baseline=np.sum([payload.baseline for payload in ordered_payloads], axis=0, dtype=np.float32).astype(np.float32),
        raw=np.sum([payload.raw for payload in ordered_payloads], axis=0, dtype=np.float32).astype(np.float32),
        blended=np.sum([payload.blended for payload in ordered_payloads], axis=0, dtype=np.float32).astype(np.float32),
        blend_weight=float(preferred_payload.blend_weight),
    )


def _evaluate_saved_bundle_on_segments(
    *,
    runtime_state: TorchRuntimeState,
    segment_frames: Sequence[pd.DataFrame],
    value_columns: Sequence[str],
    artifact_bundle: Sequence[tuple[str, str, str]],
    split_name: str,
    batch_size: int = 4096,
    cfg=None,
) -> tuple[pd.DataFrame, dict[str, Step1WindowPayload]]:
    rows: list[dict[str, object]] = []
    payloads: dict[str, Step1WindowPayload] = {}
    artifact_entries = _load_artifact_bundle_entries(artifact_bundle)
    has_component_split = any(entry.get("component") is not None for entry in artifact_entries)

    if has_component_split:
        if cfg is None:
            raise ValueError("cfg is required to evaluate component-split artifact bundles.")
        profiles = [str(profile) for profile in cfg.data.agent_profiles]
        component_segment_frames = {
            str(component): _load_component_segment_frames(
                cfg,
                segment_frames=segment_frames,
                component=str(component),
                profiles=profiles,
            )
            for component in sorted({str(entry["component"]) for entry in artifact_entries if entry.get("component")})
        }
        component_payloads_by_profile: dict[str, dict[str, Step1WindowPayload]] = {}
        for entry in artifact_entries:
            component = str(entry.get("component") or "")
            profile = str(entry.get("profile") or entry["meta"].get("agent_profile") or "")
            if component not in component_segment_frames:
                raise ValueError(f"Unsupported component '{component}' in saved artifact bundle.")
            if not profile:
                raise ValueError("Component-split artifact metadata is missing agent_profile.")
            payload = _compute_step1_payload(
                profile=profile,
                segment_frames=component_segment_frames[component],
                value_column=profile,
                model=_load_model_from_artifact_entry(entry, runtime_state=runtime_state),
                scaler=entry["scaler"],
                seq_len=int(entry["meta"]["seq_len"]),
                pred_len=int(entry["meta"]["pred_len"]),
                time_feature_mode=str(entry["meta"].get("time_feature_mode", "none")),
                input_size=int(entry["meta"].get("input_size", 1)),
                blend_weight=float(
                    entry["meta"].get("blend_weight", 1.0) if entry["meta"].get("blend_weight") is not None else 1.0
                ),
                runtime_state=runtime_state,
                batch_size=batch_size,
            )
            component_payloads_by_profile.setdefault(profile, {})[component] = payload
            rows.extend(dict(row, component=component) for row in _payload_metric_rows(split_name, payload))

        for profile, component_payloads in component_payloads_by_profile.items():
            combined_payload = _combine_component_payloads(profile=profile, component_payloads=component_payloads)
            payloads[profile] = combined_payload
            rows.extend(dict(row, component="total") for row in _payload_metric_rows(split_name, combined_payload))
        return pd.DataFrame(rows), payloads

    if len(value_columns) != len(artifact_bundle):
        raise ValueError(
            f"Expected artifact bundle size={len(value_columns)} for the current value columns, got {len(artifact_bundle)}."
        )

    for value_column, entry in zip(value_columns, artifact_entries):
        model = _load_model_from_artifact_entry(entry, runtime_state=runtime_state)
        meta = dict(entry["meta"])
        scaler = entry["scaler"]
        profile = _profile_from_column(value_column, fallback=meta.get("agent_profile"))
        payload = _compute_step1_payload(
            profile=profile,
            segment_frames=segment_frames,
            value_column=value_column,
            model=model,
            scaler=scaler,
            seq_len=int(meta["seq_len"]),
            pred_len=int(meta["pred_len"]),
            time_feature_mode=str(meta.get("time_feature_mode", "none")),
            input_size=int(meta.get("input_size", 1)),
            blend_weight=float(meta.get("blend_weight", 1.0) if meta.get("blend_weight") is not None else 1.0),
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        payloads[payload.profile] = payload
        rows.extend(_payload_metric_rows(split_name, payload))
    return pd.DataFrame(rows), payloads


def _select_profile_value_column(value_columns: Sequence[str], *, profile: str) -> str:
    for value_column in value_columns:
        if _profile_from_column(value_column) == str(profile):
            return str(value_column)
    raise KeyError(f"Could not find a value column for profile='{profile}'.")


def _filter_segment_frames_to_value_column(
    segment_frames: Sequence[pd.DataFrame],
    *,
    value_column: str,
) -> list[pd.DataFrame]:
    return [frame.loc[:, ["timestamp", value_column]].copy().reset_index(drop=True) for frame in segment_frames]


def _load_saved_profile_model(
    *,
    runtime_state: TorchRuntimeState,
    artifact_bundle: Sequence[tuple[str, str, str]],
    value_columns: Sequence[str],
    profile: str,
) -> tuple[str, LSTMForecastModel, object | None, dict[str, object]]:
    if len(value_columns) != len(artifact_bundle):
        raise ValueError(
            f"Expected artifact bundle size={len(value_columns)} for the current value columns, got {len(artifact_bundle)}."
        )
    for value_column, bundle in zip(value_columns, artifact_bundle):
        model_path, meta_path, scaler_path = bundle
        meta, scaler = load_lstm_forecaster_artifacts(
            model_path=model_path,
            meta_path=meta_path,
            scaler_path=scaler_path,
        )
        resolved_profile = _profile_from_column(value_column, fallback=meta.get("agent_profile"))
        if resolved_profile != str(profile):
            continue
        model = _build_model_from_meta(meta)
        state_dict = torch.load(model_path, map_location=runtime_state.device)
        model.load_state_dict(state_dict)
        model = model.to(runtime_state.device)
        return str(value_column), model, scaler, meta
    raise KeyError(f"Could not locate saved artifacts for profile='{profile}'.")


def _load_validation_segments(cfg):
    source = _resolve_load_source(cfg)
    _, train_segment_frames, value_columns = _load_signal_segment_frames_from_source(
        source,
        "load",
        split="train",
        drop_warmup=True,
    )
    split = temporal_split_segment_frames(
        train_segment_frames,
        train_ratio=float(cfg.forecast.lstm_train_ratio),
        val_ratio=float(cfg.forecast.lstm_val_ratio),
        seq_len=int(cfg.forecast.history_window),
        pred_len=int(cfg.env.future_horizon),
    )
    return split["val_segments"], value_columns


def _load_train_segments(cfg):
    source = _resolve_load_source(cfg)
    _, train_segment_frames, value_columns = _load_signal_segment_frames_from_source(
        source,
        "load",
        split="train",
        drop_warmup=True,
    )
    return train_segment_frames, value_columns


def _load_test_segments(cfg):
    source = _resolve_load_source(cfg)
    test_frame, _, value_columns = load_signal_matrix_from_source(source, "load", split="test")
    return [test_frame], value_columns


def collect_load_step1_diagnostics(
    cfg,
    *,
    runtime_state: TorchRuntimeState | str | torch.device | None = None,
    load_result: dict[str, object] | None = None,
    load_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Collect per-agent val/test diagnostics for the current load artifacts."""

    runtime = _resolve_runtime_state(runtime_state)
    artifact_bundle = _resolve_load_artifact_bundle(cfg, load_result=load_result, load_overrides=load_overrides)
    val_segments, value_columns = _load_validation_segments(cfg)
    test_segments, test_value_columns = _load_test_segments(cfg)
    if tuple(value_columns) != tuple(test_value_columns):
        raise ValueError("Validation/test value columns do not align for load diagnostics.")

    val_rows, val_payloads = _evaluate_saved_bundle_on_segments(
        runtime_state=runtime,
        segment_frames=val_segments,
        value_columns=value_columns,
        artifact_bundle=artifact_bundle,
        split_name="val2019",
        cfg=cfg,
    )
    test_rows, test_payloads = _evaluate_saved_bundle_on_segments(
        runtime_state=runtime,
        segment_frames=test_segments,
        value_columns=value_columns,
        artifact_bundle=artifact_bundle,
        split_name="test2020",
        cfg=cfg,
    )

    return {
        "artifact_bundle": artifact_bundle,
        "step1_metrics": pd.concat([val_rows, test_rows], ignore_index=True),
        "val_payloads": val_payloads,
        "test_payloads": test_payloads,
    }


def build_profile_monthly_error_report(
    payload: Step1WindowPayload,
    *,
    profile: str | None = None,
) -> pd.DataFrame:
    """Build a monthly step-1 MAE/bias report for one profile."""

    rows: list[dict[str, object]] = []
    target_frame = pd.DataFrame(
        {
            "timestamp": payload.timestamps,
            "target": payload.target,
            "baseline": payload.baseline,
            "raw": payload.raw,
            "blended": payload.blended,
        }
    )
    target_frame["month"] = target_frame["timestamp"].dt.month
    for month, month_frame in target_frame.groupby("month", sort=True):
        for mode in ("baseline", "raw", "blended"):
            prediction = month_frame[mode].to_numpy(dtype=np.float32)
            target = month_frame["target"].to_numpy(dtype=np.float32)
            metrics = compute_forecast_metrics(target, prediction)
            rows.append(
                {
                    "profile": str(profile or payload.profile),
                    "month": int(month),
                    "mode": mode,
                    "mae": float(metrics["mae"]),
                    "bias": float(np.mean(prediction - target)),
                    "target_mean": float(month_frame["target"].mean()),
                    "target_p95": float(month_frame["target"].quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def _resolve_rollout_focus_horizons(
    pred_len: int,
    *,
    requested_horizons: Sequence[int] = (1, 6, 12, 24),
) -> tuple[int, ...]:
    if int(pred_len) <= 0:
        return tuple()
    resolved: list[int] = []
    for horizon in requested_horizons:
        clipped = max(1, min(int(horizon), int(pred_len)))
        if clipped not in resolved:
            resolved.append(clipped)
    if not resolved:
        resolved.append(int(pred_len))
    return tuple(resolved)


def build_rollout_horizon_metrics(
    payload: RolloutWindowPayload,
    *,
    split_name: str,
    experiment: str,
    mode_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    labels = dict(mode_labels or {"baseline": "baseline", "raw": "raw", "blended": "blended"})
    rows: list[dict[str, object]] = []
    for mode_name, label in labels.items():
        predictions = np.asarray(getattr(payload, mode_name), dtype=np.float32)
        for horizon_idx in range(predictions.shape[1]):
            target = np.asarray(payload.target[:, horizon_idx], dtype=np.float32)
            prediction = np.asarray(predictions[:, horizon_idx], dtype=np.float32)
            rows.append(
                {
                    **_compute_metric_row(
                        split_name=split_name,
                        profile=payload.profile,
                        mode=str(label),
                        prediction=prediction,
                        target=target,
                        blend_weight=payload.blend_weight,
                    ),
                    "experiment": str(experiment),
                    "horizon": int(horizon_idx + 1),
                }
            )
    return pd.DataFrame(rows)


def build_profile_month_horizon_report(
    payload: RolloutWindowPayload,
    *,
    split_name: str,
    experiment: str,
    profile: str | None = None,
    mode_labels: dict[str, str] | None = None,
    requested_horizons: Sequence[int] = (1, 6, 12, 24),
) -> pd.DataFrame:
    labels = dict(mode_labels or {"baseline": "baseline", "raw": "raw", "blended": "blended"})
    focus_horizons = _resolve_rollout_focus_horizons(payload.target.shape[1], requested_horizons=requested_horizons)
    rows: list[dict[str, object]] = []
    for horizon in focus_horizons:
        horizon_idx = int(horizon - 1)
        month_index = pd.DatetimeIndex(pd.to_datetime(payload.target_timestamps[:, horizon_idx], utc=True))
        target = np.asarray(payload.target[:, horizon_idx], dtype=np.float32)
        for mode_name, label in labels.items():
            prediction = np.asarray(getattr(payload, mode_name)[:, horizon_idx], dtype=np.float32)
            for month in sorted(month_index.month.unique()):
                month_mask = month_index.month == int(month)
                target_month = target[month_mask]
                prediction_month = prediction[month_mask]
                if len(target_month) == 0:
                    continue
                metrics = compute_forecast_metrics(target_month, prediction_month)
                rows.append(
                    {
                        "experiment": str(experiment),
                        "split": split_name,
                        "profile": str(profile or payload.profile),
                        "mode": str(label),
                        "horizon": int(horizon),
                        "month": int(month),
                        "mae": float(metrics["mae"]),
                        "rmse": float(metrics["rmse"]),
                        "mape": None if metrics["mape"] is None else float(metrics["mape"]),
                        "bias": float(np.mean(prediction_month - target_month)),
                        "target_mean": float(np.mean(target_month)),
                        "target_p95": float(np.quantile(target_month, 0.95)),
                        "n_windows": int(len(target_month)),
                    }
                )
    return pd.DataFrame(rows)


def _select_current_best_rows(step1_metrics: pd.DataFrame) -> pd.DataFrame:
    test_rows = step1_metrics.loc[step1_metrics["split"] == "test2020"].copy()
    if test_rows.empty:
        return test_rows
    ranking = test_rows.sort_values(["profile", "mae", "rmse", "mode"]).reset_index(drop=True)
    best = ranking.groupby("profile", as_index=False).first()
    best = best.rename(
        columns={
            "mode": "current_best_mode",
            "mae": "current_best_mae",
            "bias": "current_best_bias",
        }
    )
    return best.loc[:, ["profile", "current_best_mode", "current_best_mae", "current_best_bias"]]


def select_blocked_blend_weight(
    payload: Step1WindowPayload,
    *,
    candidate_weights: Sequence[float],
    bias_threshold_kw: float = DEFAULT_BIAS_THRESHOLD_KW,
) -> dict[str, object]:
    """Select a blocked-validation blend weight over fixed quarter blocks."""

    base_frame = pd.DataFrame(
        {
            "timestamp": payload.timestamps,
            "target": payload.target,
            "baseline": payload.baseline,
            "raw": payload.raw,
        }
    )
    base_frame["month"] = base_frame["timestamp"].dt.month

    block_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for candidate_weight in candidate_weights:
        block_maes: list[float] = []
        block_abs_biases: list[float] = []
        passes_bias_guard = True
        for block_name, months in DEFAULT_BLOCKED_MONTH_GROUPS:
            block = base_frame.loc[base_frame["month"].isin(months)]
            if block.empty:
                continue
            blended = block["baseline"].to_numpy(dtype=np.float32) + np.float32(candidate_weight) * (
                block["raw"].to_numpy(dtype=np.float32) - block["baseline"].to_numpy(dtype=np.float32)
            )
            target = block["target"].to_numpy(dtype=np.float32)
            metrics = compute_forecast_metrics(target, blended)
            bias = float(np.mean(blended - target))
            abs_bias = abs(bias)
            passes_bias_guard = passes_bias_guard and abs_bias <= float(bias_threshold_kw)
            block_maes.append(float(metrics["mae"]))
            block_abs_biases.append(abs_bias)
            block_rows.append(
                {
                    "profile": payload.profile,
                    "candidate_weight": float(candidate_weight),
                    "block": block_name,
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "bias": bias,
                    "abs_bias": abs_bias,
                    "passes_bias_guard": abs_bias <= float(bias_threshold_kw),
                    "n_windows": int(len(block)),
                }
            )
        if not block_maes:
            continue
        candidate_rows.append(
            {
                "profile": payload.profile,
                "candidate_weight": float(candidate_weight),
                "avg_block_mae": float(np.mean(block_maes)),
                "avg_abs_block_bias": float(np.mean(block_abs_biases)),
                "max_abs_block_bias": float(np.max(block_abs_biases)),
                "passes_bias_guard": bool(passes_bias_guard),
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows)
    if candidate_frame.empty:
        raise ValueError(f"No blocked-validation candidates were produced for profile='{payload.profile}'.")

    viable = candidate_frame.loc[candidate_frame["passes_bias_guard"]].copy()
    if viable.empty:
        ranked = candidate_frame.sort_values(
            ["max_abs_block_bias", "avg_abs_block_bias", "avg_block_mae", "candidate_weight"]
        ).reset_index(drop=True)
        selected = ranked.iloc[0]
        satisfied = False
    else:
        ranked = viable.sort_values(["avg_block_mae", "avg_abs_block_bias", "candidate_weight"]).reset_index(drop=True)
        selected = ranked.iloc[0]
        satisfied = True

    selected_weight = float(selected["candidate_weight"])
    blocked_prediction = payload.baseline + np.float32(selected_weight) * (payload.raw - payload.baseline)
    return {
        "profile": payload.profile,
        "selected_weight": selected_weight,
        "bias_guard_satisfied": bool(satisfied),
        "candidate_metrics": candidate_frame,
        "block_metrics": pd.DataFrame(block_rows),
        "test_payload": Step1WindowPayload(
            profile=payload.profile,
            timestamps=payload.timestamps,
            target=payload.target,
            baseline=payload.baseline,
            raw=payload.raw,
            blended=blocked_prediction.astype(np.float32),
            blend_weight=selected_weight,
        ),
    }


def run_blocked_blend_experiment(
    cfg,
    *,
    runtime_state: TorchRuntimeState | str | torch.device | None = None,
    load_result: dict[str, object] | None = None,
    load_overrides: dict[str, object] | None = None,
    candidate_weights: Sequence[float] | None = None,
    bias_threshold_kw: float = DEFAULT_BIAS_THRESHOLD_KW,
) -> dict[str, object]:
    """Evaluate quarter-block blend selection against current load artifacts."""

    runtime = _resolve_runtime_state(runtime_state)
    artifact_bundle = _resolve_load_artifact_bundle(cfg, load_result=load_result, load_overrides=load_overrides)
    train_segments, value_columns = _load_train_segments(cfg)
    test_segments, _ = _load_test_segments(cfg)
    _, train_payloads = _evaluate_saved_bundle_on_segments(
        runtime_state=runtime,
        segment_frames=train_segments,
        value_columns=value_columns,
        artifact_bundle=artifact_bundle,
        split_name="train2019",
    )
    test_rows, test_payloads = _evaluate_saved_bundle_on_segments(
        runtime_state=runtime,
        segment_frames=test_segments,
        value_columns=value_columns,
        artifact_bundle=artifact_bundle,
        split_name="test2020",
    )

    current_best = _select_current_best_rows(test_rows)
    selection_rows: list[dict[str, object]] = []
    blocked_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    block_frames: list[pd.DataFrame] = []
    selected_payloads: dict[str, Step1WindowPayload] = {}

    for profile, train_payload in train_payloads.items():
        selection = select_blocked_blend_weight(
            train_payload,
            candidate_weights=tuple(candidate_weights or cfg.forecast.load_blend_candidates),
            bias_threshold_kw=bias_threshold_kw,
        )
        selected_weight = float(selection["selected_weight"])
        candidate_frames.append(selection["candidate_metrics"])
        block_frames.append(selection["block_metrics"])

        test_payload = test_payloads[profile]
        blocked_payload = Step1WindowPayload(
            profile=profile,
            timestamps=test_payload.timestamps,
            target=test_payload.target,
            baseline=test_payload.baseline,
            raw=test_payload.raw,
            blended=(
                test_payload.baseline + np.float32(selected_weight) * (test_payload.raw - test_payload.baseline)
            ).astype(np.float32),
            blend_weight=selected_weight,
        )
        selected_payloads[profile] = blocked_payload
        blocked_row = _compute_metric_row(
            split_name="test2020",
            profile=profile,
            mode="blocked_blended",
            prediction=blocked_payload.blended,
            target=blocked_payload.target,
            blend_weight=selected_weight,
        )
        blocked_rows.append(blocked_row)

        current_profile = current_best.loc[current_best["profile"] == profile].iloc[0]
        degradation_pct = 100.0 * (
            float(blocked_row["mae"]) - float(current_profile["current_best_mae"])
        ) / max(float(current_profile["current_best_mae"]), 1e-8)
        selection_rows.append(
            {
                "profile": profile,
                "current_best_mode": str(current_profile["current_best_mode"]),
                "current_best_mae": float(current_profile["current_best_mae"]),
                "selected_weight": selected_weight,
                "blocked_blended_mae": float(blocked_row["mae"]),
                "blocked_blended_bias": float(blocked_row["bias"]),
                "bias_guard_satisfied": bool(selection["bias_guard_satisfied"]),
                "control_degradation_pct": float(degradation_pct),
            }
        )

    summary = pd.DataFrame(selection_rows).sort_values("profile").reset_index(drop=True)
    test_metrics = pd.concat([test_rows, pd.DataFrame(blocked_rows)], ignore_index=True)
    sfh14_monthly = build_profile_monthly_error_report(selected_payloads[DEFAULT_ANALYSIS_PROFILE])
    return {
        "summary": summary,
        "test_metrics": test_metrics,
        "candidate_metrics": pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(),
        "block_metrics": pd.concat(block_frames, ignore_index=True) if block_frames else pd.DataFrame(),
        "selected_test_payloads": selected_payloads,
        "sfh14_monthly_metrics": sfh14_monthly,
    }


def _single_profile_cfg(
    cfg,
    *,
    profile: str,
    signal_time_feature_mode: str,
    artifact_root: str | Path,
) -> Any:
    local_cfg = _deepcopy_cfg(cfg)
    local_cfg.data.agent_profiles = [str(profile)]
    local_cfg.env.num_agents = 1
    local_cfg.forecast.target_signals = ["load"]
    local_cfg.obs.sequence_features = ["load"]
    local_cfg.forecast.load_time_feature_mode = str(signal_time_feature_mode)
    local_cfg.forecast.lstm_artifact_root = Path(artifact_root)
    if hasattr(local_cfg.grid, "agent_bus_ids"):
        local_cfg.grid.agent_bus_ids = list(local_cfg.grid.agent_bus_ids[:1])
    return local_cfg


def run_time_feature_none_experiment(
    cfg,
    *,
    runtime_state: TorchRuntimeState | str | torch.device | None = None,
    load_overrides: dict[str, object] | None = None,
    artifact_root: str | Path | None = None,
    focus_profile: str = DEFAULT_ANALYSIS_PROFILE,
    show_progress: bool = False,
) -> dict[str, object]:
    """Retrain a single-profile load model with time features disabled."""

    runtime = _resolve_runtime_state(runtime_state)
    base_root = Path(cfg.forecast.lstm_artifact_root).resolve()
    experiment_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else base_root.parent / "analysis" / "sfh14_load" / "e1_time_feature_none"
    )
    local_cfg = _single_profile_cfg(
        cfg,
        profile=focus_profile,
        signal_time_feature_mode="none",
        artifact_root=experiment_root,
    )
    source = _resolve_load_source(local_cfg)
    train_frame, _, value_columns = load_signal_matrix_from_source(source, "load", split="train")
    test_frame, _, _ = load_signal_matrix_from_source(source, "load", split="test")
    value_column = str(value_columns[0])
    settings = {
        "hidden_size": int((load_overrides or {}).get("hidden_size", local_cfg.forecast.lstm_hidden_size)),
        "num_layers": int((load_overrides or {}).get("num_layers", local_cfg.forecast.lstm_num_layers)),
        "dropout": float((load_overrides or {}).get("dropout", local_cfg.forecast.lstm_dropout)),
        "batch_size": int((load_overrides or {}).get("batch_size", local_cfg.forecast.lstm_batch_size)),
        "epochs": int((load_overrides or {}).get("epochs", local_cfg.forecast.lstm_epochs)),
        "lr": float((load_overrides or {}).get("lr", local_cfg.forecast.lstm_lr)),
    }
    model_artifacts = _train_component_model(
        train_frame.loc[:, ["timestamp", value_column]].copy(),
        value_column=value_column,
        runtime_state=runtime,
        seq_len=int(local_cfg.forecast.history_window),
        pred_len=int(local_cfg.env.future_horizon),
        hidden_size=settings["hidden_size"],
        num_layers=settings["num_layers"],
        dropout=settings["dropout"],
        batch_size=settings["batch_size"],
        epochs=settings["epochs"],
        lr=settings["lr"],
        time_feature_mode="none",
        candidate_weights=tuple(local_cfg.forecast.load_blend_candidates),
        show_progress=show_progress,
    )
    test_payload = _compute_step1_payload(
        profile=focus_profile,
        segment_frames=[test_frame.loc[:, ["timestamp", value_column]].copy()],
        value_column=value_column,
        model=model_artifacts.model,
        scaler=model_artifacts.scaler,
        seq_len=model_artifacts.seq_len,
        pred_len=model_artifacts.pred_len,
        time_feature_mode=model_artifacts.time_feature_mode,
        input_size=model_artifacts.input_size,
        blend_weight=model_artifacts.blend_weight,
        runtime_state=runtime,
    )
    step1_metrics = pd.DataFrame(
        _payload_metric_rows("val2019", model_artifacts.val_payload)
        + _payload_metric_rows("test2020", test_payload)
    )
    monthly = build_profile_monthly_error_report(test_payload, profile=focus_profile)
    return {
        "model_artifacts": model_artifacts,
        "step1_metrics": step1_metrics,
        "sfh14_monthly_metrics": monthly,
    }


def _series_to_segment_frame(series: pd.Series, *, name: str, timestamp_index: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamp_index, utc=True), name: series.astype(np.float32)}).reset_index(drop=True)


def _train_component_model(
    train_frame: pd.DataFrame,
    *,
    value_column: str,
    runtime_state: TorchRuntimeState,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    time_feature_mode: str,
    candidate_weights: Sequence[float],
    show_progress: bool = False,
) -> InMemoryModelArtifacts:
    split = temporal_split_segment_frames(
        [train_frame],
        train_ratio=0.7,
        val_ratio=0.15,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    train_values_only = [segment[value_column].to_numpy(dtype=np.float32) for segment in split["train_segments"]]
    train_stats = summarize_signal_values(train_values_only)
    scaler = fit_signal_scaler(train_values_only, physical_scale_by_column=None)
    input_size = 1 + int(time_feature_dim(time_feature_mode))
    x_train, y_train = build_supervised_windows_from_time_feature_frames(
        split["train_segments"],
        value_column=value_column,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        physical_scale_by_column=None,
        time_feature_mode=time_feature_mode,
    )
    x_val, y_val = build_supervised_windows_from_time_feature_frames(
        split["val_segments"],
        value_column=value_column,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        physical_scale_by_column=None,
        time_feature_mode=time_feature_mode,
    )
    train_loader = make_tensor_loader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        device=runtime_state,
        pin_memory=runtime_state.pin_memory,
    )
    val_loader = make_tensor_loader(
        x_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        device=runtime_state,
        pin_memory=runtime_state.pin_memory,
    )
    model = LSTMForecastModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pred_len=pred_len,
        input_size=input_size,
    )
    trained = train_lstm_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=runtime_state,
        show_progress=show_progress,
        progress_label=f"{value_column} epochs",
    )
    hybrid = _select_load_blend_weight_from_validation(
        model=trained["model"],
        x_val=x_val,
        y_val=y_val,
        scaler=scaler,
        physical_scale=None,
        runtime_state=runtime_state,
        batch_size=batch_size,
        candidate_weights=candidate_weights,
    )
    val_payload = _compute_step1_payload(
        profile=value_column,
        segment_frames=split["val_segments"],
        value_column=value_column,
        model=trained["model"],
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        time_feature_mode=time_feature_mode,
        input_size=input_size,
        blend_weight=float(hybrid["blend_weight"]),
        runtime_state=runtime_state,
    )
    return InMemoryModelArtifacts(
        profile=value_column,
        model=trained["model"],
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        input_size=input_size,
        time_feature_mode=time_feature_mode,
        blend_weight=float(hybrid["blend_weight"]),
        val_payload=val_payload,
        train_stats=train_stats,
    )


def _combine_component_predictions(
    *,
    household_payload: Step1WindowPayload,
    heatpump_payload: Step1WindowPayload,
    heatpump_mode: str,
) -> Step1WindowPayload:
    if len(household_payload.timestamps) != len(heatpump_payload.timestamps) or not household_payload.timestamps.equals(
        heatpump_payload.timestamps
    ):
        raise ValueError("Household and heatpump payload timestamps must align before recombination.")
    if heatpump_mode not in {"baseline", "raw", "blended"}:
        raise ValueError(f"Unsupported heatpump mode '{heatpump_mode}'.")
    heatpump_prediction = getattr(heatpump_payload, heatpump_mode)
    return Step1WindowPayload(
        profile=DEFAULT_ANALYSIS_PROFILE,
        timestamps=household_payload.timestamps,
        target=(household_payload.target + heatpump_payload.target).astype(np.float32),
        baseline=(household_payload.baseline + heatpump_payload.baseline).astype(np.float32),
        raw=(household_payload.raw + heatpump_payload.raw).astype(np.float32),
        blended=(household_payload.raw + heatpump_prediction).astype(np.float32),
        blend_weight=heatpump_payload.blend_weight,
    )


def run_component_split_experiment(
    cfg,
    *,
    runtime_state: TorchRuntimeState | str | torch.device | None = None,
    load_overrides: dict[str, object] | None = None,
    focus_profile: str = DEFAULT_ANALYSIS_PROFILE,
    show_progress: bool = False,
) -> dict[str, object]:
    """Train household/heatpump models separately and recombine them offline."""

    runtime = _resolve_runtime_state(runtime_state)
    settings = {
        "hidden_size": int((load_overrides or {}).get("hidden_size", cfg.forecast.lstm_hidden_size)),
        "num_layers": int((load_overrides or {}).get("num_layers", cfg.forecast.lstm_num_layers)),
        "dropout": float((load_overrides or {}).get("dropout", cfg.forecast.lstm_dropout)),
        "batch_size": int((load_overrides or {}).get("batch_size", cfg.forecast.lstm_batch_size)),
        "epochs": int((load_overrides or {}).get("epochs", cfg.forecast.lstm_epochs)),
        "lr": float((load_overrides or {}).get("lr", cfg.forecast.lstm_lr)),
        "time_feature_mode": str(cfg.forecast.load_time_feature_mode),
    }

    data_dir = Path(cfg.data.data_dir)
    household_train = _load_component_frame(data_dir, component="household", profiles=[focus_profile], year=cfg.data.train_year)
    household_test = _load_component_frame(data_dir, component="household", profiles=[focus_profile], year=cfg.data.test_year)
    heatpump_train = _load_component_frame(data_dir, component="heatpump", profiles=[focus_profile], year=cfg.data.train_year)
    heatpump_test = _load_component_frame(data_dir, component="heatpump", profiles=[focus_profile], year=cfg.data.test_year)

    household_train_frame = _series_to_segment_frame(household_train[focus_profile], name="household", timestamp_index=household_train["timestamp"])
    household_test_frame = _series_to_segment_frame(household_test[focus_profile], name="household", timestamp_index=household_test["timestamp"])
    heatpump_train_frame = _series_to_segment_frame(heatpump_train[focus_profile], name="heatpump", timestamp_index=heatpump_train["timestamp"])
    heatpump_test_frame = _series_to_segment_frame(heatpump_test[focus_profile], name="heatpump", timestamp_index=heatpump_test["timestamp"])

    candidate_weights = tuple(cfg.forecast.load_blend_candidates)
    household_model = _train_component_model(
        household_train_frame,
        value_column="household",
        runtime_state=runtime,
        seq_len=int(cfg.forecast.history_window),
        pred_len=int(cfg.env.future_horizon),
        hidden_size=settings["hidden_size"],
        num_layers=settings["num_layers"],
        dropout=settings["dropout"],
        batch_size=settings["batch_size"],
        epochs=settings["epochs"],
        lr=settings["lr"],
        time_feature_mode=settings["time_feature_mode"],
        candidate_weights=candidate_weights,
        show_progress=show_progress,
    )
    heatpump_model = _train_component_model(
        heatpump_train_frame,
        value_column="heatpump",
        runtime_state=runtime,
        seq_len=int(cfg.forecast.history_window),
        pred_len=int(cfg.env.future_horizon),
        hidden_size=settings["hidden_size"],
        num_layers=settings["num_layers"],
        dropout=settings["dropout"],
        batch_size=settings["batch_size"],
        epochs=settings["epochs"],
        lr=settings["lr"],
        time_feature_mode=settings["time_feature_mode"],
        candidate_weights=candidate_weights,
        show_progress=show_progress,
    )

    household_test_payload = _compute_step1_payload(
        profile="household",
        segment_frames=[household_test_frame],
        value_column="household",
        model=household_model.model,
        scaler=household_model.scaler,
        seq_len=household_model.seq_len,
        pred_len=household_model.pred_len,
        time_feature_mode=household_model.time_feature_mode,
        input_size=household_model.input_size,
        blend_weight=household_model.blend_weight,
        runtime_state=runtime,
    )
    heatpump_test_payload = _compute_step1_payload(
        profile="heatpump",
        segment_frames=[heatpump_test_frame],
        value_column="heatpump",
        model=heatpump_model.model,
        scaler=heatpump_model.scaler,
        seq_len=heatpump_model.seq_len,
        pred_len=heatpump_model.pred_len,
        time_feature_mode=heatpump_model.time_feature_mode,
        input_size=heatpump_model.input_size,
        blend_weight=heatpump_model.blend_weight,
        runtime_state=runtime,
    )

    component_rows: list[dict[str, object]] = []
    for split_name, household_payload, heatpump_payload in (
        ("val2019", household_model.val_payload, heatpump_model.val_payload),
        ("test2020", household_test_payload, heatpump_test_payload),
    ):
        component_rows.extend(_payload_metric_rows(split_name, household_payload))
        component_rows.extend(_payload_metric_rows(split_name, heatpump_payload))

    total_rows: list[dict[str, object]] = []
    total_test_payloads: dict[str, Step1WindowPayload] = {}
    for split_name, household_payload, heatpump_payload in (
        ("val2019", household_model.val_payload, heatpump_model.val_payload),
        ("test2020", household_test_payload, heatpump_test_payload),
    ):
        for heatpump_mode in ("baseline", "raw", "blended"):
            combined_payload = _combine_component_predictions(
                household_payload=household_payload,
                heatpump_payload=heatpump_payload,
                heatpump_mode=heatpump_mode,
            )
            total_rows.append(
                _compute_metric_row(
                    split_name=split_name,
                    profile=DEFAULT_ANALYSIS_PROFILE,
                    mode=f"household_raw_plus_heatpump_{heatpump_mode}",
                    prediction=combined_payload.blended,
                    target=combined_payload.target,
                    blend_weight=heatpump_payload.blend_weight,
                )
            )
            if split_name == "test2020":
                total_test_payloads[heatpump_mode] = combined_payload

    total_metrics = pd.DataFrame(total_rows)
    best_total = (
        total_metrics.loc[total_metrics["split"] == "test2020"]
        .sort_values(["mae", "rmse", "mode"])
        .iloc[0]
        .to_dict()
    )
    best_heatpump_mode = str(best_total["mode"]).replace("household_raw_plus_heatpump_", "", 1)
    monthly = build_profile_monthly_error_report(total_test_payloads[best_heatpump_mode], profile=DEFAULT_ANALYSIS_PROFILE)
    return {
        "component_metrics": pd.DataFrame(component_rows),
        "total_metrics": total_metrics,
        "best_total": best_total,
        "sfh14_monthly_metrics": monthly,
        "component_models": {"household": household_model, "heatpump": heatpump_model},
    }


def _lookup_month_bias(monthly_frame: pd.DataFrame, *, month: int, mode: str | None = None) -> float | None:
    filtered = monthly_frame.loc[monthly_frame["month"] == int(month)]
    if mode is not None and "mode" in filtered.columns:
        filtered = filtered.loc[filtered["mode"] == mode]
    if filtered.empty:
        return None
    return float(filtered.iloc[0]["bias"])


def _experiment_passes_acceptance(
    *,
    test_mae: float,
    test_bias: float,
    january_bias: float | None,
    february_bias: float | None,
    control_guardrails_ok: bool,
) -> bool:
    january_ok = january_bias is not None and abs(float(january_bias)) < DEFAULT_MONTHLY_BIAS_CAP_KW
    february_ok = february_bias is not None and abs(float(february_bias)) < DEFAULT_MONTHLY_BIAS_CAP_KW
    return (
        float(test_mae) < DEFAULT_CURRENT_BLEND_UPPER_BOUND
        and float(test_mae) <= DEFAULT_BASELINE_TARGET
        and abs(float(test_bias)) <= DEFAULT_BIAS_THRESHOLD_KW
        and january_ok
        and february_ok
        and bool(control_guardrails_ok)
    )


def build_experiment_summary(
    *,
    current_step1_metrics: pd.DataFrame,
    e1_result: dict[str, object],
    e2_result: dict[str, object],
    e3_result: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    current_best = _select_current_best_rows(current_step1_metrics)
    control_best = current_best.set_index("profile")

    e1_test = (
        e1_result["step1_metrics"]
        .loc[e1_result["step1_metrics"]["split"] == "test2020"]
        .sort_values(["mae", "rmse", "mode"])
        .iloc[0]
    )
    e1_monthly = e1_result["sfh14_monthly_metrics"]
    e1_row = {
        "experiment": "E1_time_feature_none",
        "selected_variant": str(e1_test["mode"]),
        "sfh14_test_mae": float(e1_test["mae"]),
        "sfh14_test_bias": float(e1_test["bias"]),
        "jan_bias": _lookup_month_bias(e1_monthly, month=1, mode=str(e1_test["mode"])),
        "feb_bias": _lookup_month_bias(e1_monthly, month=2, mode=str(e1_test["mode"])),
        "sfh12_guardrail_ok": True,
        "sfh16_guardrail_ok": True,
    }

    e2_summary = e2_result["summary"].set_index("profile")
    e2_monthly = e2_result["sfh14_monthly_metrics"]
    available_control_profiles = [
        profile
        for profile in ("SFH12", "SFH16")
        if profile in control_best.index and profile in e2_summary.index
    ]
    e2_control_ok = True
    for control_profile in available_control_profiles:
        current_mae = float(control_best.loc[control_profile, "current_best_mae"])
        blocked_mae = float(e2_summary.loc[control_profile, "blocked_blended_mae"])
        degradation_pct = 100.0 * (blocked_mae - current_mae) / max(current_mae, 1e-8)
        e2_control_ok = e2_control_ok and degradation_pct <= DEFAULT_CONTROL_DEGRADATION_PCT
    e2_row = {
        "experiment": "E2_blocked_blend",
        "selected_variant": f"blocked_blended_w={float(e2_summary.loc[DEFAULT_ANALYSIS_PROFILE, 'selected_weight']):.1f}",
        "sfh14_test_mae": float(e2_summary.loc[DEFAULT_ANALYSIS_PROFILE, "blocked_blended_mae"]),
        "sfh14_test_bias": float(e2_summary.loc[DEFAULT_ANALYSIS_PROFILE, "blocked_blended_bias"]),
        "jan_bias": _lookup_month_bias(e2_monthly, month=1, mode="blended"),
        "feb_bias": _lookup_month_bias(e2_monthly, month=2, mode="blended"),
        "sfh12_guardrail_ok": bool(
            "SFH12" not in available_control_profiles
            or float(e2_summary.loc["SFH12", "control_degradation_pct"]) <= DEFAULT_CONTROL_DEGRADATION_PCT
        ),
        "sfh16_guardrail_ok": bool(
            "SFH16" not in available_control_profiles
            or float(e2_summary.loc["SFH16", "control_degradation_pct"]) <= DEFAULT_CONTROL_DEGRADATION_PCT
        ),
    }

    e3_best = dict(e3_result["best_total"])
    e3_monthly = e3_result["sfh14_monthly_metrics"]
    e3_row = {
        "experiment": "E3_component_split",
        "selected_variant": str(e3_best["mode"]),
        "sfh14_test_mae": float(e3_best["mae"]),
        "sfh14_test_bias": float(e3_best["bias"]),
        "jan_bias": _lookup_month_bias(e3_monthly, month=1, mode="blended"),
        "feb_bias": _lookup_month_bias(e3_monthly, month=2, mode="blended"),
        "sfh12_guardrail_ok": True,
        "sfh16_guardrail_ok": True,
    }

    summary = pd.DataFrame([e1_row, e2_row, e3_row])
    summary["control_guardrails_ok"] = summary["sfh12_guardrail_ok"] & summary["sfh16_guardrail_ok"]
    summary["passes_acceptance"] = summary.apply(
        lambda row: _experiment_passes_acceptance(
            test_mae=float(row["sfh14_test_mae"]),
            test_bias=float(row["sfh14_test_bias"]),
            january_bias=row["jan_bias"],
            february_bias=row["feb_bias"],
            control_guardrails_ok=bool(row["control_guardrails_ok"]),
        ),
        axis=1,
    )
    ranked = summary.sort_values(
        ["passes_acceptance", "sfh14_test_mae", "sfh14_test_bias"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    preferred_row = ranked.iloc[0].to_dict()
    recommendation = {
        "preferred_experiment": preferred_row["experiment"] if bool(preferred_row["passes_acceptance"]) else None,
        "preferred_variant": preferred_row["selected_variant"] if bool(preferred_row["passes_acceptance"]) else None,
        "needs_weather_features_next": not bool(preferred_row["passes_acceptance"]),
        "current_reference_mae": DEFAULT_CURRENT_BLEND_UPPER_BOUND,
        "baseline_target_mae": DEFAULT_BASELINE_TARGET,
    }
    return ranked, recommendation


def _build_saved_profile_rollout_payload(
    cfg,
    *,
    runtime_state: TorchRuntimeState,
    segment_frames: Sequence[pd.DataFrame],
    artifact_bundle: Sequence[tuple[str, str, str]],
    value_columns: Sequence[str],
    profile: str,
    blend_weight_override: float | None = None,
    batch_size: int = 4096,
) -> RolloutWindowPayload:
    value_column, model, scaler, meta = _load_saved_profile_model(
        runtime_state=runtime_state,
        artifact_bundle=artifact_bundle,
        value_columns=value_columns,
        profile=profile,
    )
    return _compute_rollout_payload(
        profile=profile,
        segment_frames=_filter_segment_frames_to_value_column(segment_frames, value_column=value_column),
        value_column=value_column,
        model=model,
        scaler=scaler,
        seq_len=int(meta["seq_len"]),
        pred_len=int(meta["pred_len"]),
        time_feature_mode=str(meta.get("time_feature_mode", "none")),
        input_size=int(meta.get("input_size", 1)),
        blend_weight=float(
            meta.get("blend_weight", 1.0) if blend_weight_override is None else blend_weight_override
        ),
        runtime_state=runtime_state,
        batch_size=batch_size,
    )


def _combine_component_rollout_predictions(
    *,
    household_payload: RolloutWindowPayload,
    heatpump_payload: RolloutWindowPayload,
    heatpump_mode: str,
) -> RolloutWindowPayload:
    if household_payload.target.shape != heatpump_payload.target.shape:
        raise ValueError("Household and heatpump rollout payloads must align before recombination.")
    if household_payload.target_timestamps.shape != heatpump_payload.target_timestamps.shape:
        raise ValueError("Household and heatpump rollout timestamps must align before recombination.")
    if not np.array_equal(household_payload.target_timestamps, heatpump_payload.target_timestamps):
        raise ValueError("Household and heatpump rollout timestamps must be identical before recombination.")
    if heatpump_mode not in {"baseline", "raw", "blended"}:
        raise ValueError(f"Unsupported heatpump rollout mode '{heatpump_mode}'.")
    heatpump_prediction = np.asarray(getattr(heatpump_payload, heatpump_mode), dtype=np.float32)
    return RolloutWindowPayload(
        profile=DEFAULT_ANALYSIS_PROFILE,
        target_timestamps=np.asarray(household_payload.target_timestamps).copy(),
        target=(household_payload.target + heatpump_payload.target).astype(np.float32),
        baseline=(household_payload.raw + heatpump_payload.baseline).astype(np.float32),
        raw=(household_payload.raw + heatpump_payload.raw).astype(np.float32),
        blended=(household_payload.raw + heatpump_prediction).astype(np.float32),
        blend_weight=float(heatpump_payload.blend_weight),
    )


def _build_component_split_rollout_payloads(
    cfg,
    *,
    runtime_state: TorchRuntimeState,
    component_models: dict[str, InMemoryModelArtifacts],
    focus_profile: str,
    batch_size: int = 4096,
) -> dict[str, RolloutWindowPayload]:
    data_dir = Path(cfg.data.data_dir)
    household_train = _load_component_frame(data_dir, component="household", profiles=[focus_profile], year=cfg.data.train_year)
    household_test = _load_component_frame(data_dir, component="household", profiles=[focus_profile], year=cfg.data.test_year)
    heatpump_train = _load_component_frame(data_dir, component="heatpump", profiles=[focus_profile], year=cfg.data.train_year)
    heatpump_test = _load_component_frame(data_dir, component="heatpump", profiles=[focus_profile], year=cfg.data.test_year)

    household_train_frame = _series_to_segment_frame(
        household_train[focus_profile],
        name="household",
        timestamp_index=household_train["timestamp"],
    )
    household_test_frame = _series_to_segment_frame(
        household_test[focus_profile],
        name="household",
        timestamp_index=household_test["timestamp"],
    )
    heatpump_train_frame = _series_to_segment_frame(
        heatpump_train[focus_profile],
        name="heatpump",
        timestamp_index=heatpump_train["timestamp"],
    )
    heatpump_test_frame = _series_to_segment_frame(
        heatpump_test[focus_profile],
        name="heatpump",
        timestamp_index=heatpump_test["timestamp"],
    )

    household_split = temporal_split_segment_frames(
        [household_train_frame],
        train_ratio=0.7,
        val_ratio=0.15,
        seq_len=int(cfg.forecast.history_window),
        pred_len=int(cfg.env.future_horizon),
    )
    heatpump_split = temporal_split_segment_frames(
        [heatpump_train_frame],
        train_ratio=0.7,
        val_ratio=0.15,
        seq_len=int(cfg.forecast.history_window),
        pred_len=int(cfg.env.future_horizon),
    )
    household_model = component_models["household"]
    heatpump_model = component_models["heatpump"]

    payloads: dict[str, RolloutWindowPayload] = {}
    for split_name, household_segments, heatpump_segments in (
        ("val2019", household_split["val_segments"], heatpump_split["val_segments"]),
        ("test2020", [household_test_frame], [heatpump_test_frame]),
    ):
        household_payload = _compute_rollout_payload(
            profile="household",
            segment_frames=household_segments,
            value_column="household",
            model=household_model.model,
            scaler=household_model.scaler,
            seq_len=household_model.seq_len,
            pred_len=household_model.pred_len,
            time_feature_mode=household_model.time_feature_mode,
            input_size=household_model.input_size,
            blend_weight=household_model.blend_weight,
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        heatpump_payload = _compute_rollout_payload(
            profile="heatpump",
            segment_frames=heatpump_segments,
            value_column="heatpump",
            model=heatpump_model.model,
            scaler=heatpump_model.scaler,
            seq_len=heatpump_model.seq_len,
            pred_len=heatpump_model.pred_len,
            time_feature_mode=heatpump_model.time_feature_mode,
            input_size=heatpump_model.input_size,
            blend_weight=heatpump_model.blend_weight,
            runtime_state=runtime_state,
            batch_size=batch_size,
        )
        payloads[split_name] = _combine_component_rollout_predictions(
            household_payload=household_payload,
            heatpump_payload=heatpump_payload,
            heatpump_mode="blended",
        )
    return payloads


def _summarize_rollout_mode(
    horizon_metrics: pd.DataFrame,
    month_horizon_metrics: pd.DataFrame,
    *,
    experiment: str,
    mode: str,
) -> dict[str, object]:
    test_metrics = horizon_metrics.loc[
        (horizon_metrics["experiment"] == str(experiment))
        & (horizon_metrics["split"] == "test2020")
        & (horizon_metrics["mode"] == str(mode))
    ].copy()
    if test_metrics.empty:
        raise ValueError(f"No rollout metrics were available for experiment='{experiment}', mode='{mode}'.")
    max_horizon = int(test_metrics["horizon"].max())
    horizon_24 = min(24, max_horizon)
    winter_metrics = month_horizon_metrics.loc[
        (month_horizon_metrics["experiment"] == str(experiment))
        & (month_horizon_metrics["split"] == "test2020")
        & (month_horizon_metrics["mode"] == str(mode))
        & (month_horizon_metrics["month"].isin([1, 2]))
    ].copy()
    return {
        "experiment": str(experiment),
        "selected_variant": str(mode),
        "avg_mae_h1_24": float(test_metrics.sort_values("horizon")["mae"].mean()),
        "mae_h1": float(test_metrics.loc[test_metrics["horizon"] == 1, "mae"].iloc[0]),
        "mae_h24": float(test_metrics.loc[test_metrics["horizon"] == horizon_24, "mae"].iloc[0]),
        "effective_h24": int(horizon_24),
        "avg_abs_jan_feb_bias": (
            float(winter_metrics["bias"].abs().mean()) if not winter_metrics.empty else float("inf")
        ),
        "max_abs_jan_feb_bias": (
            float(winter_metrics["bias"].abs().max()) if not winter_metrics.empty else float("inf")
        ),
    }


def _select_best_rollout_mode(
    horizon_metrics: pd.DataFrame,
    month_horizon_metrics: pd.DataFrame,
    *,
    experiment: str,
    candidate_modes: Sequence[str],
) -> dict[str, object]:
    candidates = [
        _summarize_rollout_mode(
            horizon_metrics,
            month_horizon_metrics,
            experiment=experiment,
            mode=mode,
        )
        for mode in candidate_modes
    ]
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["avg_mae_h1_24"]),
            float(row["mae_h24"]),
            float(row["avg_abs_jan_feb_bias"]),
            str(row["selected_variant"]),
        ),
    )
    return dict(ranked[0])


def build_rollout_experiment_summary(
    *,
    horizon_metrics: pd.DataFrame,
    month_horizon_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    reference_blended = _summarize_rollout_mode(
        horizon_metrics,
        month_horizon_metrics,
        experiment="current_default",
        mode="blended",
    )
    baseline_reference = _summarize_rollout_mode(
        horizon_metrics,
        month_horizon_metrics,
        experiment="current_default",
        mode="baseline",
    )
    e1_summary = _select_best_rollout_mode(
        horizon_metrics,
        month_horizon_metrics,
        experiment="E1_time_feature_none",
        candidate_modes=("baseline", "raw", "blended"),
    )
    e2_summary = _summarize_rollout_mode(
        horizon_metrics,
        month_horizon_metrics,
        experiment="E2_blocked_blend",
        mode="blended",
    )
    e3_summary = _select_best_rollout_mode(
        horizon_metrics,
        month_horizon_metrics,
        experiment="E3_component_split",
        candidate_modes=(
            "household_raw_plus_heatpump_baseline",
            "household_raw_plus_heatpump_raw",
            "household_raw_plus_heatpump_blended",
        ),
    )

    rows = [reference_blended, e1_summary, e2_summary, e3_summary]
    summary = pd.DataFrame(rows)
    summary["beats_baseline_avg"] = summary["avg_mae_h1_24"] < float(baseline_reference["avg_mae_h1_24"])
    summary["not_worse_h1"] = summary["mae_h1"] <= float(baseline_reference["mae_h1"])
    summary["not_worse_h24"] = summary["mae_h24"] <= float(baseline_reference["mae_h24"])
    summary["lowers_jan_feb_bias"] = summary["avg_abs_jan_feb_bias"] < float(reference_blended["avg_abs_jan_feb_bias"])
    summary["passes_acceptance"] = (
        summary["experiment"] != "current_default"
    ) & summary["beats_baseline_avg"] & summary["not_worse_h1"] & summary["not_worse_h24"] & summary["lowers_jan_feb_bias"]

    candidate_summary = summary.loc[summary["experiment"] != "current_default"].copy()
    ranked_candidates = candidate_summary.sort_values(
        ["passes_acceptance", "avg_mae_h1_24", "mae_h24", "avg_abs_jan_feb_bias"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ordered = pd.concat([summary.loc[summary["experiment"] == "current_default"], ranked_candidates], ignore_index=True)

    preferred_row = ranked_candidates.iloc[0].to_dict()
    recommendation = {
        "preferred_experiment": preferred_row["experiment"] if bool(preferred_row["passes_acceptance"]) else None,
        "preferred_variant": preferred_row["selected_variant"] if bool(preferred_row["passes_acceptance"]) else None,
        "recommended_next_route": (
            "R1"
            if preferred_row["experiment"] == "E1_time_feature_none" and bool(preferred_row["passes_acceptance"])
            else "R2"
            if preferred_row["experiment"] == "E2_blocked_blend" and bool(preferred_row["passes_acceptance"])
            else "R3"
            if preferred_row["experiment"] == "E3_component_split" and bool(preferred_row["passes_acceptance"])
            else "R4"
        ),
        "needs_weather_features_next": not bool(preferred_row["passes_acceptance"]),
        "baseline_avg_h1_24_mae": float(baseline_reference["avg_mae_h1_24"]),
        "baseline_mae_h1": float(baseline_reference["mae_h1"]),
        "baseline_mae_h24": float(baseline_reference["mae_h24"]),
        "current_default_avg_abs_jan_feb_bias": float(reference_blended["avg_abs_jan_feb_bias"]),
    }
    return ordered, recommendation


__all__ = [
    "DEFAULT_ANALYSIS_PROFILE",
    "DEFAULT_BIAS_THRESHOLD_KW",
    "DEFAULT_BLOCKED_MONTH_GROUPS",
    "DEFAULT_CONTROL_DEGRADATION_PCT",
    "apply_heatpump_jump_relief",
    "build_component_drift_report",
    "build_profile_monthly_error_report",
    "collect_load_step1_diagnostics",
    "run_blocked_blend_experiment",
    "run_component_split_experiment",
    "run_time_feature_none_experiment",
    "select_blocked_blend_weight",
]
