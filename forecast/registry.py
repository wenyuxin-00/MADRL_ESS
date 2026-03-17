"""Forecaster registry and factory."""

from __future__ import annotations

from pathlib import Path

from forecast.artifacts import get_default_lstm_artifact_dir
from forecast.lstm_forecaster import LSTMForecaster, resolve_lstm_artifact_paths
from forecast.naive import NaiveForecaster
from forecast.oracle import PerfectForecaster
from forecast.training import collect_available_lstm_artifacts, ensure_lstm_artifacts

FORECASTER_REGISTRY: dict[str, type] = {
    "perfect": PerfectForecaster,
    "naive": NaiveForecaster,
    "lstm": LSTMForecaster,
}


def register_forecaster(name: str, forecaster_cls: type) -> None:
    """Register a new forecaster class."""
    FORECASTER_REGISTRY[name] = forecaster_cls


def build_forecaster(cfg):
    """Build the observation-side forecaster from config."""
    forecast_cfg = cfg.forecast
    forecaster_type = forecast_cfg.type

    if forecaster_type == "perfect":
        return PerfectForecaster()

    if forecaster_type == "naive":
        return NaiveForecaster(window=int(forecast_cfg.naive_window))

    if forecaster_type != "lstm":
        raise ValueError(
            f"Unknown forecaster_type '{forecaster_type}', available: {list(FORECASTER_REGISTRY)}"
        )

    if forecast_cfg.lstm_model_path is not None:
        model_path, meta_path, scaler_path = resolve_lstm_artifact_paths(forecast_cfg.lstm_model_path)
        return LSTMForecaster.from_artifacts(
            model_path=str(model_path),
            meta_path=str(meta_path),
            scaler_path=str(scaler_path),
            device=cfg.runtime.device,
            signal_name="price",
        )

    ensure_result = ensure_lstm_artifacts(cfg, device=cfg.runtime.device)
    artifact_map = collect_available_lstm_artifacts(cfg)
    if not artifact_map:
        artifact_root = Path(forecast_cfg.lstm_artifact_root or get_default_lstm_artifact_dir())
        raise FileNotFoundError(
            "No managed LSTM forecast artifacts are available. "
            f"Checked root: {artifact_root} for future_horizon={cfg.env.future_horizon}."
        )

    required_signals = [signal_name for signal_name in cfg.obs.sequence_features if signal_name in forecast_cfg.target_signals]
    missing_required = [signal_name for signal_name in required_signals if signal_name not in artifact_map]
    if missing_required:
        raise FileNotFoundError(
            "Missing required LSTM forecast artifacts for observation signals: "
            f"{missing_required}. Available managed signals: {sorted(artifact_map)}."
        )

    if ensure_result.get("trained_signals"):
        trained = ", ".join(ensure_result["trained_signals"])
        print(f"[forecast] trained new artifacts for signals: {trained}")
        if ensure_result.get("plot_path"):
            print(f"[forecast] weekly comparison plot saved to {ensure_result['plot_path']}")

    return LSTMForecaster.from_signal_artifacts(
        artifact_map,
        device=cfg.runtime.device,
    )
