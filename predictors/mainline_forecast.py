from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Mapping
from configs.experiment_config import ExperimentConfig
from predictors.artifacts import get_default_lstm_artifact_dir
from predictors.training import ensure_lstm_artifacts
def _normalize_signal_training_overrides(overrides: Mapping[str, Mapping[str, object]] | None) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for signal_name, signal_overrides in dict(overrides or {}).items():
        if not isinstance(signal_overrides, Mapping):
            raise ValueError(f"signal_training_overrides['{signal_name}'] must be a mapping, got {signal_overrides!r}.")
        normalized[str(signal_name).strip().lower()] = {str(field_name): value for field_name, value in dict(signal_overrides).items()}
    return normalized
def build_mainline_forecast_controls(cfg) -> dict[str, object]:
    artifact_root = getattr(cfg.forecast, "lstm_artifact_root", None)
    return {
        "artifact_root": None if artifact_root in (None, "") else str(Path(artifact_root).resolve()),
        "target_signals": [str(value) for value in cfg.forecast.target_signals],
        "future_horizon": int(cfg.env.future_horizon),
        "history_window": int(cfg.forecast.history_window),
        "load_model_mode": str(cfg.forecast.load_model_mode),
        "load_component_split": bool(cfg.forecast.load_component_split),
        "load_scaler_type": str(cfg.forecast.load_scaler_type),
        "load_time_feature_mode": str(cfg.forecast.load_time_feature_mode),
        "pv_time_feature_mode": str(cfg.forecast.pv_time_feature_mode),
        "load_hybrid_mode": str(cfg.forecast.load_hybrid_mode),
        "load_baseline_mode": str(cfg.forecast.load_baseline_mode),
        "pv_postprocess_mode": str(cfg.forecast.pv_postprocess_mode),
        "auto_train_missing": bool(cfg.forecast.auto_train_missing),
        "signal_training_overrides": _normalize_signal_training_overrides(getattr(cfg.forecast, "signal_training_overrides", {})),
    }
def merge_managed_forecast_controls(canonical: dict[str, object], override: dict[str, object] | None) -> dict[str, object]:
    resolved = deepcopy(dict(canonical))
    incoming = dict(override or {})
    incoming_signal_overrides = deepcopy(dict(incoming.pop("signal_training_overrides", {}) or {}))
    resolved.update(incoming)
    canonical_signal_overrides = deepcopy(dict(canonical.get("signal_training_overrides", {}) or {}))
    merged_signal_overrides: dict[str, dict[str, object]] = {}
    signal_names = set(canonical_signal_overrides) | set(incoming_signal_overrides)
    for signal_name in signal_names:
        base = deepcopy(dict(canonical_signal_overrides.get(signal_name, {}) or {}))
        base.update(dict(incoming_signal_overrides.get(signal_name, {}) or {}))
        merged_signal_overrides[str(signal_name)] = base
    resolved["signal_training_overrides"] = merged_signal_overrides
    return resolved
def resolve_mainline_forecast_controls(cfg, forecast_controls: Mapping[str, object] | None = None) -> dict[str, object]:
    controls = dict(forecast_controls or {})
    configured_future_horizon = controls.get("future_horizon")
    if configured_future_horizon is not None and int(configured_future_horizon) != int(cfg.env.future_horizon):
        raise ValueError(f"forecast_controls.future_horizon must match cfg.env.future_horizon, got {configured_future_horizon} vs {cfg.env.future_horizon}.")
    merged = merge_managed_forecast_controls(build_mainline_forecast_controls(cfg), controls)
    artifact_root = merged.get("artifact_root", cfg.forecast.lstm_artifact_root)
    return {
        "target_signals": [str(value) for value in merged.get("target_signals", cfg.forecast.target_signals)],
        "history_window": int(merged.get("history_window", cfg.forecast.history_window)),
        "load_model_mode": str(merged.get("load_model_mode", cfg.forecast.load_model_mode)),
        "load_component_split": bool(merged.get("load_component_split", cfg.forecast.load_component_split)),
        "load_scaler_type": str(merged.get("load_scaler_type", cfg.forecast.load_scaler_type)),
        "load_time_feature_mode": str(merged.get("load_time_feature_mode", cfg.forecast.load_time_feature_mode)),
        "pv_time_feature_mode": str(merged.get("pv_time_feature_mode", cfg.forecast.pv_time_feature_mode)),
        "load_hybrid_mode": str(merged.get("load_hybrid_mode", cfg.forecast.load_hybrid_mode)),
        "load_baseline_mode": str(merged.get("load_baseline_mode", cfg.forecast.load_baseline_mode)),
        "pv_postprocess_mode": str(merged.get("pv_postprocess_mode", cfg.forecast.pv_postprocess_mode)),
        "auto_train_missing": bool(merged.get("auto_train_missing", cfg.forecast.auto_train_missing)),
        "artifact_root": None if artifact_root in (None, "") else str(Path(artifact_root).resolve()),
        "signal_training_overrides": _normalize_signal_training_overrides(
            merged.get("signal_training_overrides", getattr(cfg.forecast, "signal_training_overrides", {}))
        ),
    }
def get_mainline_forecast_controls(*, artifact_root: str | Path | None = None, auto_train_missing: bool = False) -> dict[str, object]:
    controls = build_mainline_forecast_controls(ExperimentConfig())
    root = Path(artifact_root).resolve() if artifact_root is not None else get_default_lstm_artifact_dir()
    controls["artifact_root"] = str(root)
    controls["auto_train_missing"] = bool(auto_train_missing)
    return controls
def ensure_mainline_forecast_ready(cfg) -> dict[str, object] | None:
    return None if cfg.forecast.type != "lstm" else ensure_lstm_artifacts(cfg, device=cfg.runtime.device)

get_managed_lstm_forecast_controls = get_mainline_forecast_controls
