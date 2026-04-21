"""Training utilities for the LSTM forecaster used by the grid mainline."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from data.loaders.prosumer import ProsumerDataset
from scripts.utils.torch_runtime import TorchRuntimeState, configure_torch_runtime, resolve_device
from predictors.artifacts import (
    DEFAULT_SUPPORTED_FORECAST_SIGNALS,
)
from predictors.lstm_forecaster import (
    BASELINE_MODE_LAST_VALUE,
    BASELINE_MODE_NONE,
    LSTMForecaster,
    LSTM_ARTIFACT_FORMAT,
    LSTM_LOAD_HYBRID_ARTIFACT_FORMAT,
    PHYSICAL_NORMALIZATION_LOAD_SCALE,
    PHYSICAL_NORMALIZATION_NONE,
    PHYSICAL_NORMALIZATION_PV_PEAK,
    POSTPROCESS_MODE_BASELINE_BLEND,
    POSTPROCESS_MODE_NONE,
    POSTPROCESS_MODE_PHYSICAL_CLIP,
    save_lstm_forecaster_artifacts,
)
from predictors.lstm_model import LSTMForecastModel
from predictors.training_artifacts import (
    _artifact_tuple_from_paths,
    _collect_lstm_artifact_inventory,
    _format_lstm_artifact_issue,
    _managed_lstm_artifact_paths,
    _raise_lstm_artifact_requirements_error,
    _signal_artifact_specs,
    collect_available_lstm_artifacts as _collect_available_lstm_artifacts_impl,
    compare_lstm_artifact_meta,
    ensure_lstm_artifacts as _ensure_lstm_artifacts_impl,
    expected_lstm_artifact_meta,
    forecast_artifact_root,
    validate_lstm_artifact,
)
from predictors.training_evaluation import (
    _plot_evaluation_curve,
    _reshape_evaluation_target,
    _select_week_evaluation_slice,
    _to_forecaster_history_input,
    compute_forecast_metrics,
    evaluate_signal_one_week,
    evaluate_signal_online_one_week,
    evaluate_signal_open_loop_one_week,
    plot_signal_training_report,
    plot_weekly_forecasts,
)
from predictors.time_features import (
    TIME_FEATURE_MODE_HOUR_WEEK_YEAR,
    TIME_FEATURE_MODE_NONE,
    coerce_timestamp_index,
    encode_forecast_time_features,
    infer_timestamp_step,
    normalize_time_feature_mode,
    time_feature_dim,
)
from scripts.utils.price_protocol import WHOLESALE_PRICE_SIGNAL, normalize_internal_signal_name

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
PHYSICAL_SCALE_EPS = np.float32(1e-6)
HEATPUMP_BLOCKED_MONTH_GROUPS = (
    ("Jan-Mar", (1, 2, 3)),
    ("Apr-Jun", (4, 5, 6)),
    ("Jul-Sep", (7, 8, 9)),
    ("Oct-Dec", (10, 11, 12)),
)
HEATPUMP_BLOCKED_BIAS_GUARD_KW = 0.15


def _normalize_load_model_mode(mode: str | None) -> str:
    normalized = str(mode or "per_agent").strip().lower()
    if normalized != "per_agent":
        raise ValueError(
            f"Unsupported forecast.load_model_mode '{mode}'. Only 'per_agent' is implemented."
        )
    return normalized


def _normalize_load_hybrid_mode(mode: str | None) -> str:
    normalized = str(mode or POSTPROCESS_MODE_BASELINE_BLEND).strip().lower()
    if normalized != POSTPROCESS_MODE_BASELINE_BLEND:
        raise ValueError(
            f"Unsupported forecast.load_hybrid_mode '{mode}'. Only '{POSTPROCESS_MODE_BASELINE_BLEND}' is implemented."
        )
    return normalized


def _normalize_pv_postprocess_mode(mode: str | None) -> str:
    normalized = str(mode or POSTPROCESS_MODE_PHYSICAL_CLIP).strip().lower()
    if normalized not in {POSTPROCESS_MODE_NONE, POSTPROCESS_MODE_PHYSICAL_CLIP}:
        raise ValueError(
            f"Unsupported forecast.pv_postprocess_mode '{mode}'. "
            f"Only '{POSTPROCESS_MODE_NONE}' and '{POSTPROCESS_MODE_PHYSICAL_CLIP}' are implemented."
        )
    return normalized


def _normalize_load_baseline_mode(mode: str | None) -> str:
    normalized = str(mode or BASELINE_MODE_LAST_VALUE).strip().lower()
    if normalized != BASELINE_MODE_LAST_VALUE:
        raise ValueError(
            f"Unsupported forecast.load_baseline_mode '{mode}'. Only '{BASELINE_MODE_LAST_VALUE}' is implemented."
        )
    return normalized


def _normalize_load_blend_candidates(candidates: Sequence[float] | None) -> tuple[float, ...]:
    if candidates is None:
        return tuple(float(index) / 10.0 for index in range(11))
    normalized = tuple(float(value) for value in candidates)
    if not normalized:
        raise ValueError("forecast.load_blend_candidates must contain at least one candidate weight.")
    if any(value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError("forecast.load_blend_candidates values must stay within [0.0, 1.0].")
    return normalized


def _normalize_date_range(start_date, end_date) -> dict[str, str | None]:
    return {
        "start_date": None if start_date in (None, "") else str(start_date),
        "end_date": None if end_date in (None, "") else str(end_date),
    }


def resolve_signal_physical_normalization_mode(signal_name: str) -> str:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return PHYSICAL_NORMALIZATION_LOAD_SCALE
    if normalized == "pv":
        return PHYSICAL_NORMALIZATION_PV_PEAK
    return PHYSICAL_NORMALIZATION_NONE


def resolve_signal_model_mode(cfg, signal_name: str) -> str:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return _normalize_load_model_mode(getattr(cfg.forecast, "load_model_mode", "per_agent"))
    return "shared"


def resolve_signal_time_feature_mode(cfg, signal_name: str) -> str:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return normalize_time_feature_mode(getattr(cfg.forecast, "load_time_feature_mode", TIME_FEATURE_MODE_NONE))
    if normalized == "pv":
        return normalize_time_feature_mode(
            getattr(cfg.forecast, "pv_time_feature_mode", TIME_FEATURE_MODE_HOUR_WEEK_YEAR)
        )
    return TIME_FEATURE_MODE_NONE


def resolve_signal_postprocess_mode(cfg, signal_name: str) -> str:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return _normalize_load_hybrid_mode(getattr(cfg.forecast, "load_hybrid_mode", POSTPROCESS_MODE_BASELINE_BLEND))
    if normalized == "pv":
        return _normalize_pv_postprocess_mode(
            getattr(cfg.forecast, "pv_postprocess_mode", POSTPROCESS_MODE_PHYSICAL_CLIP)
        )
    return POSTPROCESS_MODE_NONE


def resolve_signal_baseline_mode(cfg, signal_name: str) -> str:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return _normalize_load_baseline_mode(getattr(cfg.forecast, "load_baseline_mode", BASELINE_MODE_LAST_VALUE))
    return BASELINE_MODE_NONE


def resolve_signal_blend_candidates(cfg, signal_name: str) -> tuple[float, ...]:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return _normalize_load_blend_candidates(getattr(cfg.forecast, "load_blend_candidates", None))
    return (1.0,)


def resolve_signal_scaler_type(cfg, signal_name: str) -> str:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return str(getattr(cfg.forecast, "load_scaler_type", "standard")).strip().lower()
    return "standard"


def resolve_signal_component_split(cfg, signal_name: str) -> bool:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load":
        return bool(getattr(cfg.forecast, "load_component_split", False))
    return False


def resolve_signal_artifact_format(signal_name: str, settings: dict[str, object]) -> str:
    normalized = _normalize_signal_name(signal_name)
    if normalized == "load" and str(settings.get("postprocess_mode", POSTPROCESS_MODE_NONE)) == POSTPROCESS_MODE_BASELINE_BLEND:
        return LSTM_LOAD_HYBRID_ARTIFACT_FORMAT
    return LSTM_ARTIFACT_FORMAT


def resolve_signal_input_size(cfg, signal_name: str) -> int:
    return 1 + int(time_feature_dim(resolve_signal_time_feature_mode(cfg, signal_name)))


def build_lstm_source_signature(cfg, signal_name: str) -> dict[str, object]:
    normalized_signal = _normalize_signal_name(signal_name)
    train_exclusion = {"start_date": None, "end_date": None}
    if (
        int(cfg.data.train_year) == int(cfg.data.test_year)
        and not cfg.data.train_start_date
        and not cfg.data.train_end_date
        and (cfg.data.test_start_date or cfg.data.test_end_date)
    ):
        train_exclusion = _normalize_date_range(cfg.data.test_start_date, cfg.data.test_end_date)
    return {
        "signal_name": normalized_signal,
        "agent_profiles": [str(profile) for profile in cfg.data.agent_profiles],
        "train_year": int(cfg.data.train_year),
        "test_year": int(cfg.data.test_year),
        "train_date_range": _normalize_date_range(cfg.data.train_start_date, cfg.data.train_end_date),
        # The managed LSTM artifact identity should track the source data used to train the
        # forecaster, not the downstream evaluation slice chosen by a notebook or experiment.
        # When train/test years differ, narrowing cfg.data.test_start_date/end_date only changes
        # which episode window we evaluate on later; it does not change the trained artifact.
        # Same-year train/test exclusion is already encoded in train_excluded_date_range below.
        "test_date_range": {"start_date": None, "end_date": None},
        "train_excluded_date_range": train_exclusion,
        "load_components": [str(component) for component in cfg.data.load_components],
        "pv_reference": str(cfg.data.pv_reference),
        "pv_capacity_kw": [float(value) for value in (cfg.data.pv_capacity_kw or [])],
        "num_agents": int(cfg.env.num_agents),
        "future_horizon": int(cfg.env.future_horizon),
        "history_window": int(cfg.forecast.history_window),
    }


def _coerce_physical_scale_by_column(
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None,
    *,
    expected_size: int,
) -> np.ndarray | None:
    if physical_scale_by_column is None:
        return None
    scale = np.asarray(physical_scale_by_column, dtype=np.float32).reshape(-1)
    if scale.size == 0:
        return None
    if scale.size == 1 and expected_size > 1:
        scale = np.repeat(scale, expected_size)
    elif scale.size != expected_size:
        raise ValueError(
            f"physical_scale_by_column size mismatch: expected {expected_size}, got {scale.size}"
        )
    return scale.astype(np.float32, copy=True)


def _apply_physical_normalization(
    values: np.ndarray,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None,
) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float32)
    if normalized.ndim == 1:
        expected_size = 1
    elif normalized.ndim == 2:
        expected_size = normalized.shape[1]
    else:
        raise ValueError(f"Expected 1D or 2D values for physical normalization, got shape {normalized.shape}")

    scale = _coerce_physical_scale_by_column(physical_scale_by_column, expected_size=expected_size)
    if scale is None:
        return normalized.astype(np.float32, copy=True)

    divisor = np.maximum(scale, np.float32(PHYSICAL_SCALE_EPS)).astype(np.float32)
    if normalized.ndim == 1:
        return (normalized / divisor[0]).astype(np.float32)
    return (normalized / divisor[None, :]).astype(np.float32)


def _slice_scale_by_source_columns(
    values: Sequence[float] | np.ndarray | float | None,
    source: "SignalCsvSource",
) -> np.ndarray | None:
    if values is None:
        return None
    scale = np.asarray(values, dtype=np.float32).reshape(-1)
    if scale.size == 0:
        return None
    indices = tuple(source.column_indices or tuple(range(len(source.value_columns))))
    if not indices:
        return None
    if scale.size == len(indices) and indices == tuple(range(len(indices))):
        return scale.astype(np.float32, copy=True)
    if any(index < 0 or index >= scale.size for index in indices):
        raise IndexError(
            f"Signal source column_indices={indices} are out of bounds for scale size={scale.size}."
        )
    return scale[list(indices)].astype(np.float32, copy=True)


def resolve_signal_physical_scale_from_source(
    source: SignalCsvSource,
    signal_name: str,
    *,
    split: str,
) -> np.ndarray | None:
    mode = resolve_signal_physical_normalization_mode(signal_name)
    if mode == PHYSICAL_NORMALIZATION_NONE:
        return None
    dataset_kwargs = dict((source.dataset_kwargs or {}).get(split) or {})
    if not dataset_kwargs:
        return None
    if mode == PHYSICAL_NORMALIZATION_LOAD_SCALE:
        return _coerce_physical_scale_by_column(
            _slice_scale_by_source_columns(dataset_kwargs.get("load_scale"), source),
            expected_size=len(source.value_columns),
        )
    if mode == PHYSICAL_NORMALIZATION_PV_PEAK:
        dataset = ProsumerDataset(**dataset_kwargs)
        pv_peak_kw = np.asarray(dataset._meta_template.get("pv_peak_kw", []), dtype=np.float32).reshape(-1)
        if pv_peak_kw.size == 0:
            return None
        if np.any(pv_peak_kw <= 0.0):
            raise ValueError("ProsumerDataset produced non-positive pv_peak_kw for PV forecast normalization.")
        return _coerce_physical_scale_by_column(
            _slice_scale_by_source_columns(pv_peak_kw, source),
            expected_size=len(source.value_columns),
        )
    return None


@dataclass(frozen=True)
class SignalCsvSource:
    """Dataset-backed source information for one forecast signal."""

    signal_name: str
    train_path: Path | None
    test_path: Path | None
    value_columns: tuple[str, ...]
    column_indices: tuple[int, ...] | None = None
    source_kind: str = "prosumer"
    dataset_kwargs: dict[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class SignalForecastEvaluation:
    """Evaluation result for one forecast signal."""

    signal_name: str
    evaluation_mode: str
    timestamps: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    metrics: dict[str, float | None]
    source_columns: tuple[str, ...]


def _normalize_signal_name(signal_name: str) -> str:
    return normalize_internal_signal_name(signal_name)


def configured_forecast_signals(cfg) -> list[str]:
    """Return the forecast signals declared in the current config."""
    unique = []
    for signal_name in cfg.forecast.target_signals:
        normalized = _normalize_signal_name(signal_name)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique or list(DEFAULT_SUPPORTED_FORECAST_SIGNALS)


def required_forecast_signals(cfg) -> list[str]:
    """Return the forecast signals actually consumed by the observation stack."""
    active = []
    configured = set(configured_forecast_signals(cfg))
    for signal_name in cfg.obs.sequence_features:
        normalized = _normalize_signal_name(signal_name)
        if normalized in configured and normalized not in active:
            active.append(normalized)
    return active or configured_forecast_signals(cfg)


def clone_config_with_signal_overrides(cfg, overrides: dict[str, object] | None = None):
    """Create a config copy for one signal-specific training run."""
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
    """Resolve the effective training settings for one signal."""
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
        "model_mode": resolve_signal_model_mode(local_cfg, signal_name),
        "time_feature_mode": resolve_signal_time_feature_mode(local_cfg, signal_name),
        "input_size": resolve_signal_input_size(local_cfg, signal_name),
        "postprocess_mode": resolve_signal_postprocess_mode(local_cfg, signal_name),
        "baseline_mode": resolve_signal_baseline_mode(local_cfg, signal_name),
        "blend_candidates": resolve_signal_blend_candidates(local_cfg, signal_name),
        "scaler_type": resolve_signal_scaler_type(local_cfg, signal_name),
        "component_split": resolve_signal_component_split(local_cfg, signal_name),
    }
    return local_cfg, settings


def _signal_columns_from_header(columns: list[str], signal_name: str) -> list[str]:
    signal_name = _normalize_signal_name(signal_name)
    if signal_name == WHOLESALE_PRICE_SIGNAL:
        return [WHOLESALE_PRICE_SIGNAL] if WHOLESALE_PRICE_SIGNAL in columns else []

    return [
        column
        for column in columns
        if column == signal_name or column.startswith(f"{signal_name}_") or column.startswith(signal_name)
    ]


def _resolve_prosumer_dataset_kwargs(cfg, data_dir: Path, split: str) -> dict[str, object]:
    year = int(cfg.data.train_year if split == "train" else cfg.data.test_year)
    start_date = cfg.data.train_start_date if split == "train" else cfg.data.test_start_date
    end_date = cfg.data.train_end_date if split == "train" else cfg.data.test_end_date
    n_agents = int(cfg.env.num_agents)
    agent_profiles = [str(profile) for profile in cfg.data.agent_profiles]
    if len(agent_profiles) != n_agents:
        raise ValueError(
            "Forecast prosumer config mismatch: "
            f"cfg.env.num_agents={n_agents} but cfg.data.agent_profiles has {len(agent_profiles)} entry(ies). "
            "Keep forecast notebook/config values in sync, for example: "
            "cfg.env.num_agents = len(cfg.data.agent_profiles)."
        )
    exclude_start_date = None
    exclude_end_date = None
    if (
        split == "train"
        and int(cfg.data.train_year) == int(cfg.data.test_year)
        and not cfg.data.train_start_date
        and not cfg.data.train_end_date
        and (cfg.data.test_start_date or cfg.data.test_end_date)
    ):
        exclude_start_date = cfg.data.test_start_date
        exclude_end_date = cfg.data.test_end_date

    return {
        "data_dir": data_dir,
        "episode_length": 1,
        "n_agents": n_agents,
        "agent_profiles": agent_profiles,
        "year": year,
        "start_date": start_date,
        "end_date": end_date,
        "exclude_start_date": exclude_start_date,
        "exclude_end_date": exclude_end_date,
        "load_components": list(cfg.data.load_components),
        "pv_reference": str(cfg.data.pv_reference),
        "pv_capacity_kw": list(cfg.data.pv_capacity_kw),
        "load_scale": list(cfg.data.load_scale),
        "pv_scale": list(cfg.data.pv_scale),
        "node_ids": list(range(n_agents)),
    }


def _prosumer_value_columns(agent_profiles: Sequence[str], signal_name: str) -> tuple[str, ...]:
    signal_name = _normalize_signal_name(signal_name)
    if signal_name == WHOLESALE_PRICE_SIGNAL:
        return (WHOLESALE_PRICE_SIGNAL,)
    return tuple(f"{signal_name}_{profile}" for profile in agent_profiles)


def _resolve_prosumer_signal_source(cfg, data_dir: Path, signal_name: str) -> SignalCsvSource | None:
    known_components = {f"load_{comp}" for comp in cfg.data.load_components}
    if signal_name not in {WHOLESALE_PRICE_SIGNAL, "load", "pv"} and signal_name not in known_components:
        return None
    agent_profiles = [str(profile) for profile in cfg.data.agent_profiles]
    base_signal = "load" if signal_name in known_components else signal_name
    value_columns = _prosumer_value_columns(agent_profiles, base_signal)
    return SignalCsvSource(
        signal_name=signal_name,
        train_path=None,
        test_path=None,
        value_columns=value_columns,
        column_indices=tuple(range(len(value_columns))),
        source_kind="prosumer",
        dataset_kwargs={
            "train": _resolve_prosumer_dataset_kwargs(cfg, data_dir, "train"),
            "test": _resolve_prosumer_dataset_kwargs(cfg, data_dir, "test"),
        },
    )


def _load_prosumer_signal_frame_from_source(
    source: SignalCsvSource,
    signal_name: str,
    *,
    split: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    dataset_kwargs = dict((source.dataset_kwargs or {}).get(split) or {})
    if not dataset_kwargs:
        raise ValueError(f"Prosumer signal source is missing dataset kwargs for split='{split}'.")

    dataset = ProsumerDataset(**dataset_kwargs)
    timestamps = pd.Series(dataset._timestamps).reset_index(drop=True)
    dataset_signal_key = source.signal_name if source.signal_name in dataset._signals else signal_name
    signal_values = np.asarray(dataset._signals[dataset_signal_key], dtype=np.float32)

    if signal_values.ndim == 1:
        value_columns = (WHOLESALE_PRICE_SIGNAL,)
        frame = pd.DataFrame({"timestamp": timestamps, WHOLESALE_PRICE_SIGNAL: signal_values})
    else:
        column_indices = tuple(source.column_indices or tuple(range(signal_values.shape[1])))
        signal_values = signal_values[:, list(column_indices)]
        value_columns = tuple(source.value_columns)
        frame = pd.DataFrame(signal_values, columns=list(value_columns))
        frame.insert(0, "timestamp", timestamps)
    frame["segment_id"] = 0
    return frame, value_columns


def _load_signal_frame_from_source(
    source: SignalCsvSource,
    signal_name: str,
    *,
    split: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if source.source_kind != "prosumer":
        raise ValueError(f"Unsupported signal source kind '{source.source_kind}'.")
    return _load_prosumer_signal_frame_from_source(source, signal_name, split=split)


def _load_signal_matrix_from_source(
    source: SignalCsvSource,
    signal_name: str,
    *,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    frame, value_columns = _load_signal_frame_from_source(source, signal_name, split=split)
    values = frame.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.reshape(-1)
    return frame, values, value_columns


def load_signal_matrix_from_source(
    source: SignalCsvSource,
    signal_name: str,
    *,
    split: str,
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Public wrapper for reading one forecast signal from a dataset-backed source."""
    return _load_signal_matrix_from_source(source, signal_name, split=split)


def select_signal_source_columns(
    source: SignalCsvSource,
    column_indices: Sequence[int],
) -> SignalCsvSource:
    selected_indices = tuple(int(index) for index in column_indices)
    if not selected_indices:
        raise ValueError("select_signal_source_columns requires at least one column index.")
    if any(index < 0 or index >= len(source.value_columns) for index in selected_indices):
        raise IndexError(
            f"Requested column_indices={selected_indices} for source columns={source.value_columns}."
        )
    return SignalCsvSource(
        signal_name=source.signal_name,
        train_path=source.train_path,
        test_path=source.test_path,
        value_columns=tuple(source.value_columns[index] for index in selected_indices),
        column_indices=selected_indices,
        source_kind=source.source_kind,
        dataset_kwargs=copy.deepcopy(source.dataset_kwargs),
    )


def _load_signal_segments_from_source(
    source: SignalCsvSource,
    signal_name: str,
    *,
    split: str,
    drop_warmup: bool = False,
) -> tuple[pd.DataFrame, list[np.ndarray], tuple[str, ...]]:
    frame, value_columns = _load_signal_frame_from_source(source, signal_name, split=split)
    if drop_warmup and "is_warmup" in frame.columns:
        frame = frame.loc[~frame["is_warmup"].astype(bool)].copy()

    if frame.empty:
        raise ValueError(
            f"Signal source split='{split}' does not contain any usable rows for signal '{signal_name}'."
        )

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
        raise ValueError(
            f"Signal source split='{split}' does not contain any segments for signal '{signal_name}'."
        )
    return working.reset_index(drop=True), segments, value_columns


def _load_signal_segment_frames_from_source(
    source: SignalCsvSource,
    signal_name: str,
    *,
    split: str,
    drop_warmup: bool = False,
) -> tuple[pd.DataFrame, list[pd.DataFrame], tuple[str, ...]]:
    frame, value_columns = _load_signal_frame_from_source(source, signal_name, split=split)
    if drop_warmup and "is_warmup" in frame.columns:
        frame = frame.loc[~frame["is_warmup"].astype(bool)].copy()

    if frame.empty:
        raise ValueError(
            f"Signal source split='{split}' does not contain any usable rows for signal '{signal_name}'."
        )

    working = frame.copy()
    if "segment_id" not in working.columns:
        working["segment_id"] = 0
    working["segment_id"] = pd.to_numeric(working["segment_id"], errors="coerce").fillna(-1).astype(int)

    segment_frames: list[pd.DataFrame] = []
    for segment_id in sorted(working["segment_id"].unique()):
        segment_frame = working.loc[working["segment_id"] == segment_id].reset_index(drop=True)
        if segment_frame.empty:
            continue
        segment_frames.append(segment_frame)

    if not segment_frames:
        raise ValueError(
            f"Signal source split='{split}' does not contain any segment frames for signal '{signal_name}'."
        )
    return working.reset_index(drop=True), segment_frames, value_columns


def temporal_split_segment_frames(
    segment_frames: list[pd.DataFrame],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seq_len: int,
    pred_len: int,
) -> dict[str, object]:
    min_window = int(seq_len) + int(pred_len)
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Expected 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1.")

    train_segments: list[pd.DataFrame] = []
    val_segments: list[pd.DataFrame] = []
    segment_summaries: list[dict[str, int]] = []

    for segment_idx, segment_frame in enumerate(segment_frames):
        total_steps = int(len(segment_frame))
        train_end = int(total_steps * train_ratio)
        val_end = min(total_steps, int(total_steps * (train_ratio + val_ratio)))
        val_context_start = max(0, train_end - int(seq_len))

        if train_end >= min_window:
            train_segments.append(segment_frame.iloc[:train_end].copy())
        if val_end - train_end > 0 and (val_end - val_context_start) >= min_window:
            val_segments.append(segment_frame.iloc[val_context_start:val_end].copy())

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
        fallback = max(train_segments, key=len)
        fallback_tail = fallback.iloc[max(0, len(fallback) - (2 * int(seq_len) + int(pred_len))) :].copy()
        if len(fallback_tail) < min_window:
            raise ValueError("No validation segments are long enough for the requested history/prediction windows.")
        val_segments = [fallback_tail]

    return {
        "train_segments": train_segments,
        "val_segments": val_segments,
        "segment_summaries": segment_summaries,
    }


def build_supervised_windows_from_time_feature_frames(
    segment_frames: list[pd.DataFrame],
    *,
    value_column: str,
    seq_len: int,
    pred_len: int,
    scaler: object | None,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
) -> tuple[np.ndarray, np.ndarray]:
    total_window = int(seq_len) + int(pred_len)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for segment_frame in segment_frames:
        series = segment_frame.loc[:, value_column].to_numpy(dtype=np.float32).reshape(-1, 1)
        if series.shape[0] < total_window:
            continue

        normalized = _apply_physical_normalization(series, physical_scale_by_column).reshape(-1)
        if scaler is not None:
            normalized = scaler.transform(normalized.reshape(-1, 1)).reshape(-1).astype(np.float32)

        load_windows = np.lib.stride_tricks.sliding_window_view(normalized, total_window).astype(np.float32)
        timestamps = coerce_timestamp_index(segment_frame["timestamp"].tolist())
        time_features = encode_forecast_time_features(timestamps, time_feature_mode).astype(np.float32)
        time_windows = np.lib.stride_tricks.sliding_window_view(
            time_features,
            window_shape=total_window,
            axis=0,
        )
        time_windows = np.moveaxis(time_windows, -1, 1).astype(np.float32)

        load_history = load_windows[:, :seq_len].reshape(-1, seq_len, 1).astype(np.float32)
        if time_windows.shape[2] > 0:
            x_parts.append(np.concatenate([load_history, time_windows[:, :seq_len, :]], axis=2))
        else:
            x_parts.append(load_history)
        y_parts.append(load_windows[:, seq_len:].astype(np.float32))

    if not x_parts:
        raise ValueError("No valid supervised windows could be built from the provided time-feature segments.")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def build_supervised_windows_from_time_feature_segments(
    segment_frames: list[pd.DataFrame],
    *,
    value_columns: Sequence[str],
    seq_len: int,
    pred_len: int,
    scaler: object | None,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build shared-model windows for one or more value columns with aligned time features."""

    total_window = int(seq_len) + int(pred_len)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for segment_frame in segment_frames:
        values = _reshape_signal_values(
            segment_frame.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
        )
        if values.shape[0] < total_window:
            continue

        normalized_values = _apply_physical_normalization(values, physical_scale_by_column)
        timestamps = coerce_timestamp_index(segment_frame["timestamp"].tolist())
        time_features = encode_forecast_time_features(timestamps, time_feature_mode).astype(np.float32)
        time_windows = np.lib.stride_tricks.sliding_window_view(
            time_features,
            window_shape=total_window,
            axis=0,
        )
        time_windows = np.moveaxis(time_windows, -1, 1).astype(np.float32)

        for column_idx in range(normalized_values.shape[1]):
            series = normalized_values[:, column_idx].astype(np.float32)
            if scaler is not None:
                series = scaler.transform(series.reshape(-1, 1)).reshape(-1).astype(np.float32)

            value_windows = np.lib.stride_tricks.sliding_window_view(series, total_window).astype(np.float32)
            value_history = value_windows[:, :seq_len].reshape(-1, seq_len, 1).astype(np.float32)
            if time_windows.shape[2] > 0:
                x_parts.append(np.concatenate([value_history, time_windows[:, :seq_len, :]], axis=2))
            else:
                x_parts.append(value_history)
            y_parts.append(value_windows[:, seq_len:].astype(np.float32))

    if not x_parts:
        raise ValueError("No valid supervised windows could be built from the provided multi-column time-feature segments.")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def resolve_signal_csv_source(data_dir: str | Path, signal_name: str, cfg=None) -> SignalCsvSource | None:
    """Resolve the dataset-backed source for one forecast signal."""
    if cfg is None:
        return None
    data_dir = Path(data_dir)
    signal_name = _normalize_signal_name(signal_name)
    return _resolve_prosumer_signal_source(cfg, data_dir, signal_name)


def load_signal_frame(csv_path: str | Path, signal_name: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """读取一个 CSV，并挑出属于指定信号的列。"""
    csv_path = Path(csv_path)
    # 先整表读入，因为我们需要根据列名判断这个 CSV 是否真的包含该信号。
    df = pd.read_csv(csv_path)
    value_columns = tuple(_signal_columns_from_header(df.columns.tolist(), signal_name))
    if not value_columns:
        raise ValueError(f"CSV '{csv_path}' does not contain signal columns for '{signal_name}'.")
    return df, value_columns


def load_signal_matrix(csv_path: str | Path, signal_name: str) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """从 CSV 中读取指定信号的数值矩阵。"""
    df, value_columns = load_signal_frame(csv_path, signal_name)
    values = df.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
    # 只有一列时压成一维，方便后续统一按“单变量序列”处理。
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
    """读取信号 CSV，并按连续 segment 切成安全的数组列表。"""
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


def fit_signal_scaler(
    values: np.ndarray | list[np.ndarray],
    *,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    scaler_type: str = "standard",
) -> StandardScaler | RobustScaler:
    """在一类信号的所有数值上拟合 scaler（StandardScaler 或 RobustScaler）。"""
    if isinstance(values, list):
        flattened = np.concatenate(
            [
                _apply_physical_normalization(np.asarray(chunk, dtype=np.float32), physical_scale_by_column).reshape(-1)
                for chunk in values
            ],
            axis=0,
        )
    else:
        flattened = _apply_physical_normalization(
            np.asarray(values, dtype=np.float32),
            physical_scale_by_column,
        ).reshape(-1)

    if str(scaler_type).lower() == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(flattened.reshape(-1, 1))
    return scaler


def summarize_signal_values(values: np.ndarray | list[np.ndarray]) -> dict[str, float]:
    """返回便于调试日志查看的紧凑统计量。"""
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
    """对 ``(T, D)`` 形状的时序矩阵做按时间顺序的 train/val/test 切分。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    # 这里不随机打乱样本，而是严格按时间顺序切分，
    # 这对时序预测很重要，否则会产生信息泄漏。
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
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把多列信号矩阵转成可用于监督学习的滑动窗口样本。"""
    values = _apply_physical_normalization(_reshape_signal_values(values), physical_scale_by_column)

    total_window = int(seq_len) + int(pred_len)
    if values.shape[0] < total_window:
        raise ValueError(
            f"Signal matrix is too short for seq_len={seq_len} and pred_len={pred_len}: shape={values.shape}"
        )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    # 对每一列单独生成窗口，最后再汇总，
    # 这样可以把多变量矩阵视为多条可训练的序列来利用。
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
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从连续信号 segment 列表中构造滑动窗口样本。"""
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    skipped_segments = 0

    # 每个 segment 都表示一段连续时间序列，
    # 不能跨 segment 拼窗口，否则会把断点两边的数据错当成连续规律。
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
            physical_scale_by_column=physical_scale_by_column,
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
    """在每个连续 segment 内部独立执行时间切分。"""
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
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    batch_size: int,
    shuffle: bool,
    device: str | torch.device | TorchRuntimeState = "cpu",
    pin_memory: bool | None = None,
) -> DataLoader:
    """根据信号数据构建可迭代的 DataLoader。"""
    if isinstance(values, list):
        x, y = build_supervised_windows_from_segments(
            values,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=physical_scale_by_column,
        )
    else:
        x, y = build_supervised_windows_from_matrix(
            values,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=physical_scale_by_column,
        )
    # DataLoader 运行前，要先把 numpy 样本包装成 PyTorch Dataset。
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    resolved_device = resolve_device(device)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        pin_memory=resolved_device.type == "cuda" if pin_memory is None else bool(pin_memory),
    )


def make_tensor_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    device: str | torch.device | TorchRuntimeState = "cpu",
    pin_memory: bool | None = None,
) -> DataLoader:
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
    """训练一个 LSTM，并返回最佳参数以及 loss 曲线。"""
    # 这里统一解析 CPU/CUDA 运行态，避免训练主循环里到处写设备分支。
    runtime_state = configure_torch_runtime(device)
    resolved_device = runtime_state.device
    model = model.to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    use_amp = resolved_device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    non_blocking = runtime_state.non_blocking_transfers and use_amp

    # 训练过程中始终保留验证集表现最好的那份参数，
    # 这样可以减少后期过拟合对最终模型的影响。
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


def _restore_supervised_window_values(
    values: np.ndarray,
    *,
    scaler: object | None,
    physical_scale: float | None,
) -> np.ndarray:
    restored = np.asarray(values, dtype=np.float32)
    original_shape = restored.shape
    if scaler is not None:
        restored = scaler.inverse_transform(restored.reshape(-1, 1)).reshape(original_shape).astype(np.float32)
    if physical_scale is not None:
        restored = (restored * np.float32(physical_scale)).astype(np.float32)
    return restored.astype(np.float32)


def _predict_supervised_windows(
    model,
    windows: np.ndarray,
    *,
    runtime_state: TorchRuntimeState,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), max(int(batch_size), 1)):
            batch = torch.tensor(
                windows[start : start + max(int(batch_size), 1)],
                dtype=torch.float32,
                device=runtime_state.device,
            )
            predictions.append(model(batch).detach().cpu().numpy().astype(np.float32))
    if not predictions:
        raise ValueError("Validation windows are required to search load blend weights.")
    return np.concatenate(predictions, axis=0).astype(np.float32)


def _build_validation_window_timestamps(
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


def _select_load_blend_weight_from_validation(
    *,
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    scaler: object | None,
    physical_scale: float | None,
    runtime_state: TorchRuntimeState,
    batch_size: int,
    candidate_weights: Sequence[float],
    component: str | None = None,
    agent_profile: str | None = None,
    window_timestamps: pd.DatetimeIndex | None = None,
) -> dict[str, object]:
    raw_predictions = _predict_supervised_windows(
        model,
        x_val,
        runtime_state=runtime_state,
        batch_size=batch_size,
    )
    raw_step1 = _restore_supervised_window_values(
        raw_predictions[:, 0],
        scaler=scaler,
        physical_scale=physical_scale,
    )
    baseline_step1 = _restore_supervised_window_values(
        x_val[:, -1, 0],
        scaler=scaler,
        physical_scale=physical_scale,
    )
    target_step1 = _restore_supervised_window_values(
        y_val[:, 0],
        scaler=scaler,
        physical_scale=physical_scale,
    )

    baseline_mae = float(np.mean(np.abs(baseline_step1 - target_step1)))
    raw_mae = float(np.mean(np.abs(raw_step1 - target_step1)))
    best_weight = float(candidate_weights[0])
    best_mae = float("inf")
    optimized_metric = "mae_step1"
    bias_guard_satisfied: bool | None = None

    if (
        str(component or "").strip().lower() == "heatpump"
        and window_timestamps is not None
        and len(window_timestamps) == len(target_step1)
    ):
        selection = _select_heatpump_blocked_blend_weight(
            target_step1=target_step1,
            baseline_step1=baseline_step1,
            raw_step1=raw_step1,
            window_timestamps=pd.DatetimeIndex(window_timestamps),
            candidate_weights=tuple(float(weight) for weight in candidate_weights),
        )
        best_weight = float(selection["selected_weight"])
        optimized_metric = "blocked_bias_guard_step1"
        bias_guard_satisfied = bool(selection["bias_guard_satisfied"])
    else:
        for candidate_weight in candidate_weights:
            weight = np.float32(candidate_weight)
            blended_step1 = baseline_step1 + weight * (raw_step1 - baseline_step1)
            mae = float(np.mean(np.abs(blended_step1 - target_step1)))
            if mae < best_mae:
                best_mae = mae
                best_weight = float(candidate_weight)

    blended_step1 = baseline_step1 + np.float32(best_weight) * (raw_step1 - baseline_step1)
    best_mae = float(np.mean(np.abs(blended_step1 - target_step1)))

    return {
        "postprocess_mode": POSTPROCESS_MODE_BASELINE_BLEND,
        "baseline_mode": BASELINE_MODE_LAST_VALUE,
        "blend_weight": best_weight,
        "optimized_metric": optimized_metric,
        "baseline_mae_step1": baseline_mae,
        "raw_mae_step1": raw_mae,
        "blended_mae_step1": best_mae,
        "bias_guard_satisfied": bias_guard_satisfied,
    }


def _select_heatpump_blocked_blend_weight(
    *,
    target_step1: np.ndarray,
    baseline_step1: np.ndarray,
    raw_step1: np.ndarray,
    window_timestamps: pd.DatetimeIndex,
    candidate_weights: Sequence[float],
    bias_threshold_kw: float = HEATPUMP_BLOCKED_BIAS_GUARD_KW,
) -> dict[str, object]:
    """Rank heatpump blend weights conservatively over blocked validation."""

    if len(window_timestamps) != len(target_step1):
        raise ValueError(
            "window_timestamps must align with target_step1 for heatpump blocked blend selection, "
            f"got len(window_timestamps)={len(window_timestamps)} vs len(target_step1)={len(target_step1)}."
        )

    base_frame = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(window_timestamps),
            "target": np.asarray(target_step1, dtype=np.float32),
            "baseline": np.asarray(baseline_step1, dtype=np.float32),
            "raw": np.asarray(raw_step1, dtype=np.float32),
        }
    )
    base_frame["month"] = base_frame["timestamp"].dt.month

    candidate_rows: list[dict[str, object]] = []
    for candidate_weight in candidate_weights:
        block_rmses: list[float] = []
        block_maes: list[float] = []
        block_abs_biases: list[float] = []
        max_abs_biases: list[float] = []
        passes_bias_guard = True
        for _, months in HEATPUMP_BLOCKED_MONTH_GROUPS:
            block = base_frame.loc[base_frame["month"].isin(months)]
            if block.empty:
                continue
            blended = block["baseline"].to_numpy(dtype=np.float32) + np.float32(candidate_weight) * (
                block["raw"].to_numpy(dtype=np.float32) - block["baseline"].to_numpy(dtype=np.float32)
            )
            target = block["target"].to_numpy(dtype=np.float32)
            metrics = compute_forecast_metrics(target, blended)
            abs_bias = abs(float(np.mean(blended - target)))
            passes_bias_guard = passes_bias_guard and abs_bias <= float(bias_threshold_kw)
            block_rmses.append(float(metrics["rmse"]))
            block_maes.append(float(metrics["mae"]))
            block_abs_biases.append(abs_bias)
            max_abs_biases.append(abs_bias)
        if not block_rmses:
            continue
        candidate_rows.append(
            {
                "candidate_weight": float(candidate_weight),
                "avg_block_rmse": float(np.mean(block_rmses)),
                "avg_block_mae": float(np.mean(block_maes)),
                "avg_abs_block_bias": float(np.mean(block_abs_biases)),
                "max_abs_block_bias": float(np.max(max_abs_biases)),
                "passes_bias_guard": bool(passes_bias_guard),
            }
        )

    candidate_frame = pd.DataFrame(candidate_rows)
    if candidate_frame.empty:
        raise ValueError("No blocked-validation candidates were produced for heatpump blend selection.")

    viable = candidate_frame.loc[candidate_frame["passes_bias_guard"]].copy()
    if viable.empty:
        ranked = candidate_frame.sort_values(
            ["max_abs_block_bias", "avg_abs_block_bias", "avg_block_mae", "candidate_weight"]
        ).reset_index(drop=True)
        satisfied = False
    else:
        ranked = viable.sort_values(
            ["avg_block_mae", "avg_abs_block_bias", "candidate_weight"]
        ).reset_index(drop=True)
        satisfied = True

    selected = ranked.iloc[0]
    return {
        "selected_weight": float(selected["candidate_weight"]),
        "bias_guard_satisfied": bool(satisfied),
        "candidate_metrics": candidate_frame,
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
    input_size: int = 1,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
    model_mode: str = "shared",
    physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    postprocess_mode: str = POSTPROCESS_MODE_NONE,
    baseline_mode: str = BASELINE_MODE_NONE,
    blend_weight: float | None = None,
    optimized_metric: str | None = None,
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
        input_size=input_size,
        time_feature_mode=time_feature_mode,
        model_mode=model_mode,
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=physical_scale_by_column,
        postprocess_mode=postprocess_mode,
        baseline_mode=baseline_mode,
        blend_weight=blend_weight,
        optimized_metric=optimized_metric,
    ).rename_default_signal(signal_name)
    runtime = forecaster.signal_runtimes[signal_name][0]
    runtime.model.load_state_dict(copy.deepcopy(model.state_dict()))
    runtime.model.eval()
    runtime.agent_index = None if agent_index is None else int(agent_index)
    runtime.agent_profile = None if agent_profile is None else str(agent_profile)
    forecaster._set_legacy_attributes()
    return forecaster




def _extract_segment_value_arrays(
    segment_frames: list[pd.DataFrame],
    value_columns: Sequence[str],
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for segment_frame in segment_frames:
        values = segment_frame.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        arrays.append(values)
    return arrays


def _aggregate_per_agent_evaluations(
    signal_name: str,
    evaluation_mode: str,
    evaluations: Sequence[SignalForecastEvaluation],
) -> SignalForecastEvaluation:
    if not evaluations:
        raise ValueError("Per-agent evaluation aggregation requires at least one evaluation.")

    base_timestamps = np.asarray(evaluations[0].timestamps)
    target_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    source_columns: list[str] = []

    for evaluation in evaluations:
        evaluation_timestamps = np.asarray(evaluation.timestamps)
        if evaluation_timestamps.shape != base_timestamps.shape or not np.array_equal(
            evaluation_timestamps,
            base_timestamps,
        ):
            raise ValueError("Per-agent evaluation timestamps must align before aggregation.")
        target_array = np.asarray(evaluation.target, dtype=np.float32)
        prediction_array = np.asarray(evaluation.prediction, dtype=np.float32)
        if target_array.ndim == 1:
            target_rows.append(target_array)
            prediction_rows.append(prediction_array)
        elif target_array.ndim == 2 and target_array.shape[0] == 1:
            target_rows.append(target_array.reshape(-1))
            prediction_rows.append(prediction_array.reshape(-1))
        else:
            raise ValueError(
                "Per-agent aggregation expects 1D evaluations or 2D evaluations with one source column."
            )
        source_columns.extend(str(column) for column in evaluation.source_columns)

    target = np.stack(target_rows, axis=0).astype(np.float32)
    prediction = np.stack(prediction_rows, axis=0).astype(np.float32)
    return SignalForecastEvaluation(
        signal_name=signal_name,
        evaluation_mode=evaluation_mode,
        timestamps=base_timestamps,
        target=target,
        prediction=prediction,
        metrics=compute_forecast_metrics(target, prediction),
        source_columns=tuple(source_columns),
    )


def _aggregate_training_history(agent_results: Sequence[dict[str, object]]) -> dict[str, list[float]]:
    if not agent_results:
        return {"train_loss": [], "val_loss": []}
    train_curves = [
        np.asarray(result["training"]["history"]["train_loss"], dtype=np.float32)
        for result in agent_results
    ]
    val_curves = [
        np.asarray(result["training"]["history"]["val_loss"], dtype=np.float32)
        for result in agent_results
    ]
    return {
        "train_loss": np.mean(np.stack(train_curves, axis=0), axis=0).astype(np.float32).tolist(),
        "val_loss": np.mean(np.stack(val_curves, axis=0), axis=0).astype(np.float32).tolist(),
    }


def _summarize_per_agent_train_stats(agent_results: Sequence[dict[str, object]]) -> dict[str, object]:
    stats_by_agent = {
        str(result.get("agent_profile", result["columns"][0])): dict(result["train_stats"])
        for result in agent_results
    }
    return {
        "agent_stats": stats_by_agent,
        "min": float(min(stat["min"] for stat in stats_by_agent.values())),
        "max": float(max(stat["max"] for stat in stats_by_agent.values())),
        "mean": float(np.mean([stat["mean"] for stat in stats_by_agent.values()])),
        "std": float(np.mean([stat["std"] for stat in stats_by_agent.values()])),
    }


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


def _train_single_signal_lstm(
    local_cfg,
    signal_name: str,
    *,
    source: SignalCsvSource,
    settings: dict[str, object],
    runtime_state: TorchRuntimeState,
    overrides: dict[str, object] | None = None,
    show_progress: bool = False,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    component: str | None = None,
) -> dict[str, object]:
    seq_len = int(local_cfg.forecast.history_window)
    pred_len = int(local_cfg.env.future_horizon)
    physical_normalization_mode = resolve_signal_physical_normalization_mode(signal_name)
    train_physical_scale = resolve_signal_physical_scale_from_source(source, signal_name, split="train")
    test_physical_scale = resolve_signal_physical_scale_from_source(source, signal_name, split="test")
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    val_window_timestamps: pd.DatetimeIndex | None = None

    if int(settings["input_size"]) > 1:
        _, train_segment_frames, value_columns = _load_signal_segment_frames_from_source(
            source,
            signal_name,
            split="train",
            drop_warmup=True,
        )
        split = temporal_split_segment_frames(
            train_segment_frames,
            train_ratio=float(local_cfg.forecast.lstm_train_ratio),
            val_ratio=float(local_cfg.forecast.lstm_val_ratio),
            seq_len=seq_len,
            pred_len=pred_len,
        )
        train_values_only = _extract_segment_value_arrays(split["train_segments"], source.value_columns)
        train_stats = summarize_signal_values(train_values_only)
        scaler = fit_signal_scaler(
            train_values_only,
            physical_scale_by_column=train_physical_scale,
            scaler_type=str(settings.get("scaler_type", "standard")),
        )
        x_train, y_train = build_supervised_windows_from_time_feature_segments(
            split["train_segments"],
            value_columns=value_columns,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            time_feature_mode=str(settings["time_feature_mode"]),
        )
        x_val, y_val = build_supervised_windows_from_time_feature_segments(
            split["val_segments"],
            value_columns=value_columns,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            time_feature_mode=str(settings["time_feature_mode"]),
        )
        val_window_timestamps = _build_validation_window_timestamps(
            split["val_segments"],
            seq_len=seq_len,
            pred_len=pred_len,
        )
        train_loader = make_tensor_loader(
            x_train,
            y_train,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=True,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        val_loader = make_tensor_loader(
            x_val,
            y_val,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=False,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        train_segment_count = len(split["train_segments"])
        val_segment_count = len(split["val_segments"])
    else:
        _, train_segments, value_columns = _load_signal_segments_from_source(
            source,
            signal_name,
            split="train",
            drop_warmup=True,
        )
        split = temporal_split_segments(
            train_segments,
            train_ratio=float(local_cfg.forecast.lstm_train_ratio),
            val_ratio=float(local_cfg.forecast.lstm_val_ratio),
            seq_len=seq_len,
            pred_len=pred_len,
        )
        train_values_only = split["train_segments"]
        train_stats = summarize_signal_values(train_values_only)
        scaler = fit_signal_scaler(
            train_values_only,
            physical_scale_by_column=train_physical_scale,
            scaler_type=str(settings.get("scaler_type", "standard")),
        )
        train_loader = make_matrix_loader(
            train_values_only,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=True,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        val_loader = make_matrix_loader(
            split["val_segments"],
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=False,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        train_segment_count = len(train_values_only)
        val_segment_count = len(split["val_segments"])

    progress_name = signal_name if agent_profile is None else f"{signal_name}[{agent_profile}]"
    if component is not None:
        progress_name = f"{progress_name}/{component}"
    print(
        f"[forecast] {progress_name}: train_segments={train_segment_count}, "
        f"val_segments={val_segment_count}, columns={list(value_columns)}, "
        f"train_stats={train_stats}, settings={settings}"
    )

    model = LSTMForecastModel(
        hidden_size=int(local_cfg.forecast.lstm_hidden_size),
        num_layers=int(local_cfg.forecast.lstm_num_layers),
        dropout=float(local_cfg.forecast.lstm_dropout),
        pred_len=pred_len,
        input_size=int(settings["input_size"]),
    )
    result = train_lstm_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(local_cfg.forecast.lstm_epochs),
        lr=float(local_cfg.forecast.lstm_lr),
        device=runtime_state,
        show_progress=show_progress,
        progress_label=f"{progress_name} epochs",
    )

    hybrid_config = {
        "postprocess_mode": str(settings["postprocess_mode"]),
        "baseline_mode": str(settings["baseline_mode"]),
        "blend_weight": None,
        "optimized_metric": None,
    }
    if (
        _normalize_signal_name(signal_name) == "load"
        and str(settings["postprocess_mode"]) == POSTPROCESS_MODE_BASELINE_BLEND
    ):
        if x_val is None or y_val is None:
            raise ValueError("Load hybrid validation requires supervised validation windows.")
        physical_scale = None
        if train_physical_scale is not None:
            physical_scale = float(np.asarray(train_physical_scale, dtype=np.float32).reshape(-1)[0])
        hybrid_config = _select_load_blend_weight_from_validation(
            model=result["model"],
            x_val=x_val,
            y_val=y_val,
            scaler=scaler,
            physical_scale=physical_scale,
            runtime_state=runtime_state,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            candidate_weights=settings["blend_candidates"],
            component=component,
            agent_profile=agent_profile,
            window_timestamps=val_window_timestamps,
        )

    artifact_paths = _managed_lstm_artifact_paths(
        local_cfg,
        signal_name,
        agent_index=agent_index if str(settings["model_mode"]) == "per_agent" else None,
        component=component,
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
        artifact_format=resolve_signal_artifact_format(signal_name, settings),
        normalization_mode=physical_normalization_mode,
        source_signature=build_lstm_source_signature(local_cfg, signal_name),
        input_size=int(settings["input_size"]),
        time_feature_mode=str(settings["time_feature_mode"]),
        model_mode=str(settings["model_mode"]),
        agent_index=agent_index,
        agent_profile=agent_profile,
        postprocess_mode=str(hybrid_config["postprocess_mode"]),
        baseline_mode=str(hybrid_config["baseline_mode"]),
        blend_weight=hybrid_config["blend_weight"],
        optimized_metric=hybrid_config["optimized_metric"],
        component=component,
        scaler_type=str(settings.get("scaler_type", "standard")),
    )

    refreshed_validation = validate_lstm_artifact(
        local_cfg,
        signal_name,
        artifact_paths,
        overrides=overrides,
        agent_index=agent_index,
        agent_profile=agent_profile,
        component=component,
    )
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
        input_size=int(settings["input_size"]),
        time_feature_mode=str(settings["time_feature_mode"]),
        model_mode=str(settings["model_mode"]),
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=test_physical_scale,
        agent_index=agent_index,
        agent_profile=agent_profile,
        postprocess_mode=str(hybrid_config["postprocess_mode"]),
        baseline_mode=str(hybrid_config["baseline_mode"]),
        blend_weight=hybrid_config["blend_weight"],
        optimized_metric=hybrid_config["optimized_metric"],
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
        input_size=int(settings["input_size"]),
        time_feature_mode=str(settings["time_feature_mode"]),
        model_mode=str(settings["model_mode"]),
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=test_physical_scale,
        agent_index=agent_index,
        agent_profile=agent_profile,
        postprocess_mode=str(hybrid_config["postprocess_mode"]),
        baseline_mode=str(hybrid_config["baseline_mode"]),
        blend_weight=hybrid_config["blend_weight"],
        optimized_metric=hybrid_config["optimized_metric"],
    )
    _print_signal_evaluation_summary(progress_name, online_evaluation, open_loop_evaluation)

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
        "agent_index": agent_index,
        "agent_profile": agent_profile,
        "component": component,
        "hybrid": hybrid_config,
    }


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
    source = resolve_signal_csv_source(data_dir, signal_name, cfg=local_cfg)
    if source is None:
        raise FileNotFoundError(
            f"No train/test source found for forecast signal '{signal_name}' under '{data_dir}'."
        )
    if int(local_cfg.env.future_horizon) <= 0:
        raise ValueError("LSTM forecast training requires cfg.env.future_horizon > 0.")
    if signal_name == "load" and str(settings["model_mode"]) == "per_agent":
        profiles = [str(profile) for profile in local_cfg.data.agent_profiles][: len(source.value_columns)]
        agent_results: list[dict[str, object]] = []
        use_component_split = bool(settings.get("component_split", False))
        components = list(local_cfg.data.load_components) if use_component_split else [None]

        for agent_index, agent_profile in enumerate(profiles):
            for component in components:
                if component is not None:
                    comp_signal = f"load_{component}"
                    comp_source = resolve_signal_csv_source(data_dir, comp_signal, cfg=local_cfg)
                    if comp_source is None:
                        raise FileNotFoundError(
                            f"No source for component signal '{comp_signal}'. "
                            f"Ensure ProsumerDataset exposes load_{component}."
                        )
                    comp_source = select_signal_source_columns(comp_source, [agent_index])
                    agent_results.append(
                        _train_single_signal_lstm(
                            local_cfg,
                            signal_name,
                            source=comp_source,
                            settings=settings,
                            runtime_state=runtime_state,
                            overrides=overrides,
                            show_progress=show_progress,
                            agent_index=agent_index,
                            agent_profile=agent_profile,
                            component=component,
                        )
                    )
                else:
                    agent_source = select_signal_source_columns(source, [agent_index])
                    agent_results.append(
                        _train_single_signal_lstm(
                            local_cfg,
                            signal_name,
                            source=agent_source,
                            settings=settings,
                            runtime_state=runtime_state,
                            overrides=overrides,
                            show_progress=show_progress,
                            agent_index=agent_index,
                            agent_profile=agent_profile,
                        )
                    )

        online_evaluation = _aggregate_per_agent_evaluations(
            signal_name,
            "online_aligned",
            [result["online_evaluation"] for result in agent_results],
        )
        open_loop_evaluation = _aggregate_per_agent_evaluations(
            signal_name,
            "open_loop",
            [result["open_loop_evaluation"] for result in agent_results],
        )
        return {
            "signal_name": signal_name,
            "source": source,
            "columns": tuple(source.value_columns),
            "segment_summary": {str(result["agent_profile"]): result["segment_summary"] for result in agent_results},
            "train_stats": _summarize_per_agent_train_stats(agent_results),
            "artifact_paths": [result["artifact_paths"] for result in agent_results],
            "training": {
                "history": _aggregate_training_history(agent_results),
                "best_val_loss": float(
                    np.mean([result["training"]["best_val_loss"] for result in agent_results], dtype=np.float32)
                ),
            },
            "evaluation": online_evaluation,
            "online_evaluation": online_evaluation,
            "open_loop_evaluation": open_loop_evaluation,
            "settings": settings,
            "runtime": runtime_state,
            "agent_results": agent_results,
            "hybrid": {
                str(result["agent_profile"]): dict(result["hybrid"])
                for result in agent_results
            },
        }

    physical_normalization_mode = resolve_signal_physical_normalization_mode(signal_name)
    train_physical_scale = resolve_signal_physical_scale_from_source(source, signal_name, split="train")
    test_physical_scale = resolve_signal_physical_scale_from_source(source, signal_name, split="test")

    seq_len = int(local_cfg.forecast.history_window)
    pred_len = int(local_cfg.env.future_horizon)
    if int(settings["input_size"]) > 1:
        _, train_segment_frames, value_columns = _load_signal_segment_frames_from_source(
            source,
            signal_name,
            split="train",
            drop_warmup=True,
        )
        split = temporal_split_segment_frames(
            train_segment_frames,
            train_ratio=float(local_cfg.forecast.lstm_train_ratio),
            val_ratio=float(local_cfg.forecast.lstm_val_ratio),
            seq_len=seq_len,
            pred_len=pred_len,
        )
        train_values_only = _extract_segment_value_arrays(split["train_segments"], value_columns)
        train_stats = summarize_signal_values(train_values_only)
        scaler = fit_signal_scaler(
            train_values_only,
            physical_scale_by_column=train_physical_scale,
            scaler_type=str(settings.get("scaler_type", "standard")),
        )
        x_train, y_train = build_supervised_windows_from_time_feature_segments(
            split["train_segments"],
            value_columns=value_columns,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            time_feature_mode=str(settings["time_feature_mode"]),
        )
        x_val, y_val = build_supervised_windows_from_time_feature_segments(
            split["val_segments"],
            value_columns=value_columns,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            time_feature_mode=str(settings["time_feature_mode"]),
        )
        train_loader = make_tensor_loader(
            x_train,
            y_train,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=True,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        val_loader = make_tensor_loader(
            x_val,
            y_val,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=False,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        train_segment_count = len(split["train_segments"])
        val_segment_count = len(split["val_segments"])
    else:
        _, train_segments, value_columns = _load_signal_segments_from_source(
            source,
            signal_name,
            split="train",
            drop_warmup=True,
        )
        split = temporal_split_segments(
            train_segments,
            train_ratio=float(local_cfg.forecast.lstm_train_ratio),
            val_ratio=float(local_cfg.forecast.lstm_val_ratio),
            seq_len=seq_len,
            pred_len=pred_len,
        )
        train_values_only = split["train_segments"]
        train_stats = summarize_signal_values(train_values_only)
        scaler = fit_signal_scaler(
            train_values_only,
            physical_scale_by_column=train_physical_scale,
            scaler_type=str(settings.get("scaler_type", "standard")),
        )
        train_loader = make_matrix_loader(
            train_values_only,
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=True,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        val_loader = make_matrix_loader(
            split["val_segments"],
            seq_len=seq_len,
            pred_len=pred_len,
            scaler=scaler,
            physical_scale_by_column=train_physical_scale,
            batch_size=int(local_cfg.forecast.lstm_batch_size),
            shuffle=False,
            device=runtime_state.device,
            pin_memory=runtime_state.pin_memory,
        )
        train_segment_count = len(train_values_only)
        val_segment_count = len(split["val_segments"])

    print(
        f"[forecast] {signal_name}: train_segments={train_segment_count}, "
        f"val_segments={val_segment_count}, columns={list(value_columns)}, "
        f"train_stats={train_stats}, settings={settings}"
    )

    model = LSTMForecastModel(
        hidden_size=int(local_cfg.forecast.lstm_hidden_size),
        num_layers=int(local_cfg.forecast.lstm_num_layers),
        dropout=float(local_cfg.forecast.lstm_dropout),
        pred_len=pred_len,
        input_size=int(settings["input_size"]),
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

    artifact_paths = _managed_lstm_artifact_paths(local_cfg, signal_name)
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
        artifact_format=resolve_signal_artifact_format(signal_name, settings),
        normalization_mode=physical_normalization_mode,
        source_signature=build_lstm_source_signature(local_cfg, signal_name),
        input_size=int(settings["input_size"]),
        time_feature_mode=str(settings["time_feature_mode"]),
        model_mode=str(settings["model_mode"]),
        postprocess_mode=str(settings["postprocess_mode"]),
        baseline_mode=str(settings["baseline_mode"]),
        blend_weight=None,
        optimized_metric=None,
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
        input_size=int(settings["input_size"]),
        time_feature_mode=str(settings["time_feature_mode"]),
        model_mode=str(settings["model_mode"]),
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=test_physical_scale,
        postprocess_mode=str(settings["postprocess_mode"]),
        baseline_mode=str(settings["baseline_mode"]),
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
        input_size=int(settings["input_size"]),
        time_feature_mode=str(settings["time_feature_mode"]),
        model_mode=str(settings["model_mode"]),
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=test_physical_scale,
        postprocess_mode=str(settings["postprocess_mode"]),
        baseline_mode=str(settings["baseline_mode"]),
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
        "hybrid": {
            "postprocess_mode": str(settings["postprocess_mode"]),
            "baseline_mode": str(settings["baseline_mode"]),
            "blend_weight": None,
            "optimized_metric": None,
        },
    }


def collect_available_lstm_artifacts(
    cfg,
    *,
    overrides_by_signal: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """只返回与当前配置兼容的 artifact。"""
    return _collect_available_lstm_artifacts_impl(
        cfg,
        overrides_by_signal=overrides_by_signal,
    )


def ensure_lstm_artifacts(
    cfg,
    *,
    device: str | torch.device | None = None,
    overrides_by_signal: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Ensure the managed LSTM artifacts required by the current config exist."""
    return _ensure_lstm_artifacts_impl(
        cfg,
        device=device,
        overrides_by_signal=overrides_by_signal,
    )
