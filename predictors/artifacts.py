"""预测器模型产物（artifact）路径管理。

管理 LSTM 等预测模型的保存/加载路径。

主要函数:
    get_default_lstm_artifact_dir -- 获取默认 LSTM 产物目录
"""

from __future__ import annotations

from pathlib import Path

from scripts.utils.project_paths import get_forecast_artifact_root, project_root as resolve_project_root

DEFAULT_LSTM_ARTIFACT_DIR = Path("lstm")
DEFAULT_PLOT_DIR_NAME = "plots"
DEFAULT_SUPPORTED_FORECAST_SIGNALS = ("price", "load", "pv")


def project_root() -> Path:
    """Return the repository root."""
    return resolve_project_root()


def get_default_lstm_artifact_dir(root: str | Path | None = None) -> Path:
    """Return the canonical root directory for LSTM forecast artifacts.

    这里的 `root` 指的是 *forecast/lstm artifact 根目录本身*。
    如果要从仓库根目录推导标准位置，请直接传 `None`，
    或先调用 `common.project_paths.get_forecast_artifact_root(...)`。
    """
    if root is not None:
        return Path(root)
    return get_forecast_artifact_root(root) / DEFAULT_LSTM_ARTIFACT_DIR


def get_lstm_horizon_artifact_dir(
    future_horizon: int,
    root: str | Path | None = None,
) -> Path:
    """Return the artifact directory for one forecast horizon."""
    return get_default_lstm_artifact_dir(root) / f"h{int(future_horizon)}"


def get_default_lstm_artifact_paths(
    root: str | Path | None = None,
    *,
    signal_name: str = "price",
    future_horizon: int = 24,
) -> dict[str, Path]:
    """Return the standard model/meta/scaler paths for one signal."""
    signal_name = str(signal_name)
    future_horizon = int(future_horizon)
    artifact_dir = get_lstm_horizon_artifact_dir(future_horizon, root) / signal_name
    stem = f"{signal_name}_lstm_h{future_horizon}"
    return {
        "artifact_dir": artifact_dir,
        "model_path": artifact_dir / f"{stem}.pt",
        "meta_path": artifact_dir / f"{stem}_meta.json",
        "scaler_path": artifact_dir / f"{stem}_scaler.pkl",
    }


def get_weekly_forecast_plot_path(
    future_horizon: int,
    root: str | Path | None = None,
) -> Path:
    """Return the default combined weekly comparison plot path."""
    plot_dir = get_lstm_horizon_artifact_dir(future_horizon, root) / DEFAULT_PLOT_DIR_NAME
    return plot_dir / f"weekly_forecast_h{int(future_horizon)}.png"
