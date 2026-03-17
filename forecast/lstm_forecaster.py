"""基于已训练 LSTM 的运行时价格预测器。

职责：
- 统一加载 LSTM 模型、meta 和 scaler 三件套 artifact。
- 在环境 rollout 时按统一 forecaster 契约生成价格窗口。
- 与 forecast/forecast.ipynb 复用同一份网络定义与 artifact 格式。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from forecast.base import Forecaster
from forecast.lstm_model import LSTMPricePredictor

LSTM_META_SUFFIX = "_meta.json"
LSTM_SCALER_SUFFIX = "_scaler.pkl"
LSTM_REQUIRED_META_FIELDS = (
    "seq_len",
    "pred_len",
    "hidden_size",
    "num_layers",
    "dropout",
)


def resolve_lstm_artifact_paths(model_path, meta_path=None, scaler_path=None) -> tuple[Path, Path, Path]:
    """根据模型路径解析标准 sidecar artifact 路径。"""
    model_path = Path(model_path)
    stem = model_path.stem
    meta_path = Path(meta_path) if meta_path is not None else model_path.with_name(f"{stem}{LSTM_META_SUFFIX}")
    scaler_path = (
        Path(scaler_path)
        if scaler_path is not None
        else model_path.with_name(f"{stem}{LSTM_SCALER_SUFFIX}")
    )
    return model_path, meta_path, scaler_path


def save_lstm_forecaster_artifacts(
    model_path,
    state_dict,
    scaler,
    *,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> dict[str, str]:
    """保存运行时 LSTM 预测器所需的标准三件套。"""
    if scaler is None:
        raise ValueError("LSTM forecaster artifacts require a fitted scaler object.")

    model_path, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, model_path)

    meta = {
        "artifact_format": "lstm_forecaster_v1",
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "scaler_path": str(scaler_path),
    }


def load_lstm_forecaster_artifacts(model_path, meta_path=None, scaler_path=None) -> tuple[dict, object]:
    """加载运行时 LSTM 预测器所需的 meta 与 scaler。"""
    _, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path, meta_path, scaler_path)

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM meta artifact: '{meta_path}'. Expected the standard '<stem>_meta.json' file next to the model."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM scaler artifact: '{scaler_path}'. Expected the standard '<stem>_scaler.pkl' file next to the model."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing_fields = [field for field in LSTM_REQUIRED_META_FIELDS if field not in meta]
    if missing_fields:
        raise ValueError(
            f"LSTM meta artifact '{meta_path}' is missing required fields: {missing_fields}"
        )

    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    return meta, scaler


class LSTMForecaster(Forecaster):
    """运行时 LSTM 价格预测器。"""

    def __init__(
        self,
        model_path: str | None = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.23,
        pred_len: int = 4,
        seq_len: int = 1344,
        device: str | torch.device = "cpu",
        scaler=None,
    ):
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.device = torch.device(device)
        self.scaler = scaler
        self._cache = np.array([], dtype=np.float32)

        self.model = LSTMPricePredictor(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pred_len=pred_len,
        ).to(self.device)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

    @classmethod
    def from_artifacts(
        cls,
        model_path: str,
        meta_path: str | None = None,
        scaler_path: str | None = None,
        device: str | torch.device = "cpu",
    ):
        """从标准 artifact 三件套直接构建 forecaster。"""
        meta, scaler = load_lstm_forecaster_artifacts(
            model_path=model_path,
            meta_path=meta_path,
            scaler_path=scaler_path,
        )
        return cls(
            model_path=model_path,
            hidden_size=int(meta["hidden_size"]),
            num_layers=int(meta["num_layers"]),
            dropout=float(meta["dropout"]),
            pred_len=int(meta["pred_len"]),
            seq_len=int(meta["seq_len"]),
            device=device,
            scaler=scaler,
        )

    def reset(self) -> None:
        """在 episode 重置时清空滚动缓存。"""
        self._cache = np.array([], dtype=np.float32)

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """基于历史价格滚动预测 horizon 步。"""
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32).reshape(-1)
        if history.size == 0:
            return np.zeros((horizon,), dtype=np.float32)

        current_price = np.array([history[-1]], dtype=np.float32)
        if horizon == 1:
            return current_price.copy()

        self._cache = history.copy()
        result = []
        remaining = horizon - 1

        while remaining > 0:
            if len(self._cache) < self.seq_len:
                pad = np.zeros(self.seq_len - len(self._cache), dtype=np.float32)
                inp = np.concatenate([pad, self._cache])
            else:
                inp = self._cache[-self.seq_len :]

            if self.scaler is not None:
                inp = self.scaler.transform(inp.reshape(-1, 1)).flatten().astype(np.float32)

            inp_t = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(inp_t).cpu().numpy().flatten()

            if self.scaler is not None:
                out = self.scaler.inverse_transform(out.reshape(-1, 1)).flatten()
            out = np.asarray(out, dtype=np.float32)

            take = min(self.pred_len, remaining)
            pred_chunk = out[:take].astype(np.float32)
            result.append(pred_chunk)
            self._cache = np.concatenate([self._cache, pred_chunk], axis=0)
            remaining -= take

        future = np.concatenate(result).astype(np.float32)
        return np.concatenate([current_price, future], axis=0)[:horizon].astype(np.float32)
