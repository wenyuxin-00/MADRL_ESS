"""预测器构建入口。

注意：
- 预测器只影响观测中的价格窗口。
- 环境真实奖励仍然使用真实 price/load 计算。
- LSTM 预测器默认从统一 artifact 目录加载，保证 forecast notebook 与 runtime 闭环。
"""

from __future__ import annotations

from pathlib import Path

from forecast.artifacts import (
    DEFAULT_LSTM_META_NAME,
    DEFAULT_LSTM_MODEL_NAME,
    DEFAULT_LSTM_SCALER_NAME,
    get_default_lstm_artifact_paths,
)
from forecast.lstm_forecaster import LSTMForecaster, resolve_lstm_artifact_paths
from forecast.naive import NaiveForecaster
from forecast.oracle import PerfectForecaster

FORECASTER_REGISTRY: dict[str, type] = {
    "perfect": PerfectForecaster,
    "naive": NaiveForecaster,
    "lstm": LSTMForecaster,
}


def resolve_lstm_runtime_paths(model_path: str | Path | None) -> tuple[Path, Path, Path]:
    """解析 runtime 应加载的 LSTM artifact 路径。"""
    if model_path is None:
        default_paths = get_default_lstm_artifact_paths()
        return (
            default_paths["model_path"],
            default_paths["meta_path"],
            default_paths["scaler_path"],
        )

    candidate = Path(model_path)
    if candidate.suffix == "":
        return (
            candidate / DEFAULT_LSTM_MODEL_NAME,
            candidate / DEFAULT_LSTM_META_NAME,
            candidate / DEFAULT_LSTM_SCALER_NAME,
        )

    return resolve_lstm_artifact_paths(candidate)


def build_forecaster(cfg):
    """按配置创建预测器。"""
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

    model_path, meta_path, scaler_path = resolve_lstm_runtime_paths(forecast_cfg.lstm_model_path)
    missing = [path for path in (model_path, meta_path, scaler_path) if not path.exists()]
    if missing:
        default_paths = get_default_lstm_artifact_paths()
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "LSTM 预测器缺少 artifact 文件："
            f"{missing_text}。请先运行 forecast/forecast.ipynb 生成统一产物，"
            f"默认目录为 {default_paths['artifact_dir']}。"
        )

    return LSTMForecaster.from_artifacts(
        model_path=str(model_path),
        meta_path=str(meta_path),
        scaler_path=str(scaler_path),
        device=cfg.runtime.device,
    )
