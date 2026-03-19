"""预测器注册表。

根据预测器类型名称（如 "perfect"、"naive"、"lstm"）
返回对应的预测器类。

主要函数:
    build_forecaster -- 根据配置构建预测器实例
"""

from __future__ import annotations

import warnings
from pathlib import Path

from predictors.artifacts import get_default_lstm_artifact_dir
from predictors.lstm_forecaster import LSTMForecaster, resolve_lstm_artifact_paths
from predictors.naive import NaiveForecaster
from predictors.oracle import PerfectForecaster
from predictors.training import (
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
    # 这是一个轻量的插件入口：注册后就可以按名字创建预测器。
    FORECASTER_REGISTRY[name] = forecaster_cls


def _build_legacy_single_signal_forecaster(cfg) -> LSTMForecaster:
    """兼容历史的 `lstm_model_path` 单模型入口。"""
    # 老版本只支持一个 price 模型；新版本改成“一个 signal 一套 artifact”的多信号方案。
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

    # 先处理不依赖训练产物的简单预测器。
    if forecaster_type == "perfect":
        return PerfectForecaster()

    if forecaster_type == "naive":
        return NaiveForecaster(window=int(forecast_cfg.naive_window))

    if forecaster_type != "lstm":
        raise ValueError(
            f"Unknown forecaster_type '{forecaster_type}', available: {list(FORECASTER_REGISTRY)}"
        )

    if forecast_cfg.lstm_model_path is not None:
        # 仍在使用旧字段时，统一走兼容层，避免和新主线逻辑混在一起。
        return _build_legacy_single_signal_forecaster(cfg)

    # 新主线：先确保所需 artifact 已准备好，不存在时按配置自动补齐。
    ensure_result = ensure_lstm_artifacts(cfg, device=cfg.runtime.device)
    artifact_map = dict(ensure_result.get("artifacts") or collect_available_lstm_artifacts(cfg))
    if not artifact_map:
        artifact_root = Path(forecast_cfg.lstm_artifact_root or get_default_lstm_artifact_dir())
        raise FileNotFoundError(
            "No managed LSTM forecast artifacts are available. "
            f"Checked root: {artifact_root} for future_horizon={cfg.env.future_horizon}."
        )

    active_signals = required_forecast_signals(cfg)
    # 只挑当前观测真正会消费的那部分信号，不把所有 artifact 都强行加载进来。
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

    # 最后把“当前实验真正需要的 artifact 子集”交给 forecaster。
    selected_artifacts = {signal_name: artifact_map[signal_name] for signal_name in active_signals}
    return LSTMForecaster.from_signal_artifacts(
        selected_artifacts,
        device=cfg.runtime.device,
    )
