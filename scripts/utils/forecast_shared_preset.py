"""Shared forecast notebook presets for managed LSTM artifacts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from configs.experiment_config import ExperimentConfig
from predictors.artifacts import get_default_lstm_artifact_dir


def _default_cfg() -> ExperimentConfig:
    return ExperimentConfig()


def get_managed_lstm_signal_training_overrides() -> dict[str, dict[str, object]]:
    """Return the canonical per-signal LSTM overrides shared by notebooks."""
    cfg = _default_cfg()
    return deepcopy(dict(cfg.forecast.signal_training_overrides))


def merge_managed_forecast_controls(
    canonical: dict[str, object],
    override: dict[str, object] | None,
) -> dict[str, object]:
    """Merge metadata or notebook overrides onto canonical forecast defaults."""
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


def get_managed_lstm_forecast_controls(
    *,
    artifact_root: str | Path | None = None,
    auto_train_missing: bool = False,
) -> dict[str, object]:
    """Return the canonical managed-LSTM controls shared by notebooks."""
    cfg = _default_cfg()
    root = Path(artifact_root).resolve() if artifact_root is not None else get_default_lstm_artifact_dir()
    return {
        "artifact_root": str(root),
        "target_signals": list(cfg.forecast.target_signals),
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
        "auto_train_missing": bool(auto_train_missing),
        "signal_training_overrides": get_managed_lstm_signal_training_overrides(),
    }


__all__ = [
    "get_managed_lstm_forecast_controls",
    "get_managed_lstm_signal_training_overrides",
    "merge_managed_forecast_controls",
]
