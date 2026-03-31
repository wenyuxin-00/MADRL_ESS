"""Shared forecast notebook presets for managed LSTM artifacts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from predictors.artifacts import get_default_lstm_artifact_dir

_DEFAULT_SIGNAL_TRAINING_OVERRIDES: dict[str, dict[str, object]] = {
    "price": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.10,
        "batch_size": 1024,
        "epochs": 20,
        "lr": 1e-3,
    },
    "load": {
        "hidden_size": 96,
        "num_layers": 2,
        "dropout": 0.10,
        "batch_size": 1024,
        "epochs": 20,
        "lr": 1e-3,
    },
    "pv": {
        "hidden_size": 96,
        "num_layers": 1,
        "dropout": 0.00,
        "batch_size": 1024,
        "epochs": 20,
        "lr": 8e-4,
    },
}


def get_managed_lstm_signal_training_overrides() -> dict[str, dict[str, object]]:
    """Return the canonical per-signal LSTM overrides shared by notebooks."""
    return deepcopy(_DEFAULT_SIGNAL_TRAINING_OVERRIDES)


def get_managed_lstm_forecast_controls(
    *,
    artifact_root: str | Path | None = None,
    auto_train_missing: bool = False,
) -> dict[str, object]:
    """Return the canonical managed-LSTM controls shared by notebooks."""
    root = Path(artifact_root).resolve() if artifact_root is not None else get_default_lstm_artifact_dir()
    return {
        "artifact_root": str(root),
        "target_signals": ["price", "load", "pv"],
        "future_horizon": 24,
        "history_window": 96 * 3,
        "load_model_mode": "per_agent",
        "load_component_split": True,
        "load_scaler_type": "robust",
        "load_time_feature_mode": "hour_week_year",
        "pv_time_feature_mode": "hour_week_year",
        "load_hybrid_mode": "baseline_blend",
        "load_baseline_mode": "last_value",
        "pv_postprocess_mode": "physical_clip",
        "auto_train_missing": bool(auto_train_missing),
        "signal_training_overrides": get_managed_lstm_signal_training_overrides(),
    }


__all__ = [
    "get_managed_lstm_forecast_controls",
    "get_managed_lstm_signal_training_overrides",
]
