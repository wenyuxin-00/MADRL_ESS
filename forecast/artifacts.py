"""LSTM 预测 artifact 路径约定。

项目统一使用固定目录保存 LSTM 预测器产物：
- artifacts/forecast/lstm/best_lstm.pt
- artifacts/forecast/lstm/best_lstm_meta.json
- artifacts/forecast/lstm/best_lstm_scaler.pkl

训练 notebook、预测 notebook、runtime 与测试都应复用这里的路径函数，
避免各自拼路径导致闭环断裂。
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_LSTM_ARTIFACT_DIR = Path("artifacts") / "forecast" / "lstm"
DEFAULT_LSTM_MODEL_NAME = "best_lstm.pt"
DEFAULT_LSTM_META_NAME = "best_lstm_meta.json"
DEFAULT_LSTM_SCALER_NAME = "best_lstm_scaler.pkl"


def project_root() -> Path:
    """返回仓库根目录。"""
    return Path(__file__).resolve().parents[1]


def get_default_lstm_artifact_dir(root: str | Path | None = None) -> Path:
    """返回统一的 LSTM artifact 目录。"""
    base = Path(root) if root is not None else project_root()
    return base / DEFAULT_LSTM_ARTIFACT_DIR


def get_default_lstm_artifact_paths(root: str | Path | None = None) -> dict[str, Path]:
    """返回统一的 LSTM artifact 三件套路径。"""
    artifact_dir = get_default_lstm_artifact_dir(root)
    return {
        "artifact_dir": artifact_dir,
        "model_path": artifact_dir / DEFAULT_LSTM_MODEL_NAME,
        "meta_path": artifact_dir / DEFAULT_LSTM_META_NAME,
        "scaler_path": artifact_dir / DEFAULT_LSTM_SCALER_NAME,
    }
