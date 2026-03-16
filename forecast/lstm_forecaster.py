"""
forecast/lstm_forecaster.py
职责：基于已训练 LSTM 的价格预测器。

修复说明（第二步重构）：
  - 原版错误地使用 torch.load(model_path) 当完整模型对象，
    但 forecast.ipynb 保存的是 state_dict（torch.save(model.state_dict())）。
  - 本版本改为依赖 forecast/lstm_model.py 中的公开模型定义，
    notebook / runtime / tests 共用同一份网络结构真源。

输出契约（与 Forecaster 基类一致）：
  - 输入 history = [p0, ..., pt]
  - 输出窗口必须形如 [pt, p_{t+1}^, p_{t+2}^, ...]
  - 也就是第一个元素保留当前真实价格，后续才由模型滚动预测

使用方式：
  forecaster = LSTMForecaster(model_path="outputs/best_lstm.pt")
  # 在 env.reset() 后由 env 自动调用 forecaster.reset()
  price_win = forecaster.predict(ep_price[:t+1], K+1)
"""

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
    """Resolve the standard sidecar artifact paths for an LSTM forecaster."""
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
) -> dict:
    """Save the standard artifact trio used by runtime LSTM forecasters.

    Standard format:
      - `<stem>.pt`               : model state_dict
      - `<stem>_meta.json`        : architecture/runtime metadata
      - `<stem>_scaler.pkl`       : fitted scaler used during training
    """
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

    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "scaler_path": str(scaler_path),
    }


def load_lstm_forecaster_artifacts(model_path, meta_path=None, scaler_path=None) -> tuple[dict, object]:
    """Load metadata and scaler for the standard LSTM forecaster artifact format."""
    _, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path, meta_path, scaler_path)

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM meta artifact: '{meta_path}'. "
            "Expected the standard '<stem>_meta.json' file next to the model."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM scaler artifact: '{scaler_path}'. "
            "Expected the standard '<stem>_scaler.pkl' file next to the model."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing_fields = [field for field in LSTM_REQUIRED_META_FIELDS if field not in meta]
    if missing_fields:
        raise ValueError(
            f"LSTM meta artifact '{meta_path}' is missing required fields: {missing_fields}"
        )

    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    return meta, scaler

class LSTMForecaster(Forecaster):
    """基于已训练 LSTM 的滚动价格预测器。

    Parameters
    ----------
    model_path : str
        已训练模型的 .pth 文件路径（保存的是 state_dict，
        由 torch.save(model.state_dict(), path) 生成）。
    hidden_size : int
        LSTM 隐层维度，须与训练时一致，默认 128。
    num_layers : int
        LSTM 层数，须与训练时一致，默认 2。
    dropout : float
        Dropout 率，须与训练时一致，默认 0.23。
    pred_len : int
        单次预测步数，须与训练时一致，默认 4。
    seq_len : int
        输入历史窗口长度，须与训练时一致，默认 1344（14天×96步）。
    device : str
        推理设备，默认 "cpu"。
    scaler : sklearn.preprocessing.MinMaxScaler, optional
        与训练时相同的归一化器，用于输入归一化和输出逆归一化。
        None 表示不做归一化（当训练时未归一化时使用）。
    """

    def __init__(self, model_path: str = None, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.23, pred_len: int = 4,
                 seq_len: int = 1344, device: str = "cpu", scaler=None):
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.device    = torch.device(device)
        self.scaler    = scaler
        self._cache: np.ndarray = np.array([], dtype=np.float32)

        # 构建模型结构，然后加载 state_dict
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
    def from_artifacts(cls, model_path: str, meta_path: str = None,
                       scaler_path: str = None, device: str = "cpu"):
        """Build a forecaster directly from standard runtime artifacts."""
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
        """episode 重置时清空历史缓存。"""
        self._cache = np.array([], dtype=np.float32)

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """基于历史价格滚动预测 horizon 步。

        Parameters
        ----------
        history : np.ndarray, shape (t+1,)
            当前步（含）之前的真实价格序列。
        horizon : int
            需要预测的步数（K+1）。

        Returns
        -------
        np.ndarray, shape (horizon,), float32
            其中 result[0] == history[-1]，其余元素为模型滚动预测值。
        """
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
            # 填充不足 seq_len 的历史（左侧零填充）
            if len(self._cache) < self.seq_len:
                pad = np.zeros(self.seq_len - len(self._cache), dtype=np.float32)
                inp = np.concatenate([pad, self._cache])
            else:
                inp = self._cache[-self.seq_len:]

            # 可选归一化
            if self.scaler is not None:
                inp = self.scaler.transform(inp.reshape(-1, 1)).flatten().astype(np.float32)

            inp_t = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(inp_t).cpu().numpy().flatten()  # (pred_len,)

            # 可选逆归一化
            if self.scaler is not None:
                out = self.scaler.inverse_transform(out.reshape(-1, 1)).flatten()
            out = np.asarray(out, dtype=np.float32)

            take = min(self.pred_len, remaining)
            pred_chunk = out[:take].astype(np.float32)
            result.append(pred_chunk)
            # 关键修复：多步预测时将当前预测块滚动写回缓存，避免重复同一段输出。
            self._cache = np.concatenate([self._cache, pred_chunk], axis=0)
            remaining -= take

        future = np.concatenate(result).astype(np.float32)
        return np.concatenate([current_price, future], axis=0)[:horizon].astype(np.float32)
