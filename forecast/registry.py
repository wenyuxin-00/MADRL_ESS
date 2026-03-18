"""预测器注册表与统一工厂。"""

from __future__ import annotations

import warnings
from pathlib import Path

from forecast.artifacts import get_default_lstm_artifact_dir
from forecast.lstm_forecaster import LSTMForecaster, resolve_lstm_artifact_paths
from forecast.naive import NaiveForecaster
from forecast.oracle import PerfectForecaster
from forecast.training import (
    collect_available_lstm_artifacts,
    ensure_lstm_artifacts,
    required_forecast_signals,
)

FORECASTER_REGISTRY: dict[str, type] = {
    "perfect": PerfectForecaster,
    "naive": NaiveForecaster,
    "lstm": LSTMForecaster,
}


def register_forecaster(name: str, forecaster_cls: type) -> None:
    """注册新的预测器实现。"""
    FORECASTER_REGISTRY[name] = forecaster_cls


def _build_legacy_single_signal_forecaster(cfg) -> LSTMForecaster:
    """兼容历史的 `lstm_model_path` 单模型入口。"""
    warnings.warn(
        "cfg.forecast.lstm_model_path 已进入兼容层，新的多信号主线请改用 "
        "cfg.forecast.lstm_artifact_root + cfg.forecast.target_signals。",
        DeprecationWarning,
        stacklevel=2,
    )
    model_path, meta_path, scaler_path = resolve_lstm_artifact_paths(cfg.forecast.lstm_model_path)
    return LSTMForecaster.from_artifacts(
        model_path=str(model_path),
        meta_path=str(meta_path),
        scaler_path=str(scaler_path),
        device=cfg.runtime.device,
        signal_name="price",
    )


def build_forecaster(cfg):
    """按统一配置构建观测侧预测器。"""
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
        return _build_legacy_single_signal_forecaster(cfg)

    ensure_result = ensure_lstm_artifacts(cfg, device=cfg.runtime.device)
    artifact_map = dict(ensure_result.get("artifacts") or collect_available_lstm_artifacts(cfg))
    if not artifact_map:
        artifact_root = Path(forecast_cfg.lstm_artifact_root or get_default_lstm_artifact_dir())
        raise FileNotFoundError(
            "No managed LSTM forecast artifacts are available. "
            f"Checked root: {artifact_root} for future_horizon={cfg.env.future_horizon}."
        )

    active_signals = required_forecast_signals(cfg)
    missing_required = [signal_name for signal_name in active_signals if signal_name not in artifact_map]
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
    if ensure_result.get("retrained_signals"):
        retrained = ", ".join(ensure_result["retrained_signals"])
        print(f"[forecast] refreshed incompatible artifacts for signals: {retrained}")
        if ensure_result.get("plot_path"):
            print(f"[forecast] weekly comparison plot saved to {ensure_result['plot_path']}")

    selected_artifacts = {signal_name: artifact_map[signal_name] for signal_name in active_signals}
    return LSTMForecaster.from_signal_artifacts(
        selected_artifacts,
        device=cfg.runtime.device,
    )
