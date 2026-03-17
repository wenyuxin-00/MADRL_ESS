"""Runtime LSTM forecaster with multi-signal support."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
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


def resolve_lstm_artifact_paths(
    model_path,
    meta_path=None,
    scaler_path=None,
) -> tuple[Path, Path, Path]:
    """Resolve the standard sidecar paths for one saved LSTM model."""
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
    signal_name: str = "price",
    future_horizon: int | None = None,
) -> dict[str, str]:
    """Save the standard model/meta/scaler triplet for one signal."""
    if scaler is None:
        raise ValueError("LSTM forecaster artifacts require a fitted scaler object.")

    model_path, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, model_path)

    meta = {
        "artifact_format": "lstm_forecaster_v2",
        "signal_name": str(signal_name),
        "future_horizon": int(pred_len if future_horizon is None else future_horizon),
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


def load_lstm_forecaster_artifacts(
    model_path,
    meta_path=None,
    scaler_path=None,
) -> tuple[dict, object]:
    """Load the standard meta/scaler sidecars for one signal model."""
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


@dataclass
class _SignalForecasterRuntime:
    signal_name: str
    seq_len: int
    pred_len: int
    hidden_size: int
    num_layers: int
    dropout: float
    model: torch.nn.Module
    scaler: object | None


class LSTMForecaster(Forecaster):
    """Runtime LSTM forecaster for shared and per-agent signals."""

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
        signal_runtimes: dict[str, _SignalForecasterRuntime] | None = None,
    ):
        self.device = torch.device(device)

        if signal_runtimes is None:
            runtime = self._build_runtime(
                signal_name="price",
                model_path=model_path,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                pred_len=pred_len,
                seq_len=seq_len,
                scaler=scaler,
                device=self.device,
            )
            signal_runtimes = {"price": runtime}

        self.signal_runtimes = dict(signal_runtimes)
        self._set_legacy_attributes()

    def _set_legacy_attributes(self) -> None:
        """Expose historical single-signal attributes for compatibility."""
        preferred_signal = "price" if "price" in self.signal_runtimes else next(iter(self.signal_runtimes))
        runtime = self.signal_runtimes[preferred_signal]
        self.seq_len = int(runtime.seq_len)
        self.pred_len = int(runtime.pred_len)
        self.scaler = runtime.scaler
        self.model = runtime.model

    def _sync_legacy_price_runtime(self) -> None:
        """Keep historical direct attribute mutation compatible with tests/notebooks."""
        if "price" not in self.signal_runtimes:
            return
        runtime = self.signal_runtimes["price"]
        runtime.seq_len = int(getattr(self, "seq_len", runtime.seq_len))
        runtime.pred_len = int(getattr(self, "pred_len", runtime.pred_len))
        runtime.scaler = getattr(self, "scaler", runtime.scaler)
        runtime.model = getattr(self, "model", runtime.model)

    @staticmethod
    def _build_runtime(
        *,
        signal_name: str,
        model_path: str | None,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pred_len: int,
        seq_len: int,
        scaler,
        device: torch.device,
    ) -> _SignalForecasterRuntime:
        model = LSTMPricePredictor(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pred_len=pred_len,
        ).to(device)

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

        model.eval()
        return _SignalForecasterRuntime(
            signal_name=str(signal_name),
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            model=model,
            scaler=scaler,
        )

    @classmethod
    def from_artifacts(
        cls,
        model_path: str,
        meta_path: str | None = None,
        scaler_path: str | None = None,
        device: str | torch.device = "cpu",
        signal_name: str = "price",
    ):
        """Build a single-signal forecaster from one artifact triplet."""
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
            signal_runtimes=None,
        ).rename_default_signal(meta.get("signal_name", signal_name))

    @classmethod
    def from_signal_artifacts(
        cls,
        signal_artifacts: dict[str, tuple[str, str | None, str | None]],
        *,
        device: str | torch.device = "cpu",
    ):
        """Build a multi-signal runtime forecaster from several artifact triplets."""
        device = torch.device(device)
        signal_runtimes: dict[str, _SignalForecasterRuntime] = {}
        for signal_name, (model_path, meta_path, scaler_path) in signal_artifacts.items():
            meta, scaler = load_lstm_forecaster_artifacts(
                model_path=model_path,
                meta_path=meta_path,
                scaler_path=scaler_path,
            )
            runtime = cls._build_runtime(
                signal_name=meta.get("signal_name", signal_name),
                model_path=model_path,
                hidden_size=int(meta["hidden_size"]),
                num_layers=int(meta["num_layers"]),
                dropout=float(meta["dropout"]),
                pred_len=int(meta["pred_len"]),
                seq_len=int(meta["seq_len"]),
                scaler=scaler,
                device=device,
            )
            signal_runtimes[str(signal_name)] = runtime

        return cls(device=device, signal_runtimes=signal_runtimes)

    def rename_default_signal(self, signal_name: str):
        """Rename the compatibility default signal after loading legacy artifacts."""
        if "price" in self.signal_runtimes and signal_name != "price":
            self.signal_runtimes[str(signal_name)] = self.signal_runtimes.pop("price")
            self.signal_runtimes[str(signal_name)].signal_name = str(signal_name)
        self._set_legacy_attributes()
        return self

    def available_signals(self) -> list[str]:
        """Return all signal names backed by loaded artifacts."""
        return sorted(self.signal_runtimes)

    def reset(self) -> None:
        """The runtime forecaster is stateless across episodes."""
        return None

    def _predict_univariate(
        self,
        runtime: _SignalForecasterRuntime,
        history: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        if horizon <= 0:
            return np.zeros((0,), dtype=np.float32)

        history = np.asarray(history, dtype=np.float32).reshape(-1)
        if history.size == 0:
            return np.zeros((horizon,), dtype=np.float32)

        current_value = np.array([history[-1]], dtype=np.float32)
        if horizon == 1:
            return current_value.copy()

        rolling_history = history.copy()
        future_chunks = []
        remaining = horizon - 1

        while remaining > 0:
            if rolling_history.size < runtime.seq_len:
                pad = np.zeros((runtime.seq_len - rolling_history.size,), dtype=np.float32)
                model_input = np.concatenate([pad, rolling_history], axis=0)
            else:
                model_input = rolling_history[-runtime.seq_len :]

            if runtime.scaler is not None:
                model_input = runtime.scaler.transform(model_input.reshape(-1, 1)).reshape(-1).astype(np.float32)

            model_tensor = torch.tensor(
                model_input,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                prediction = runtime.model(model_tensor).detach().cpu().numpy().reshape(-1)

            if runtime.scaler is not None:
                prediction = runtime.scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)

            prediction = np.asarray(prediction, dtype=np.float32)
            take = min(runtime.pred_len, remaining)
            prediction = prediction[:take].astype(np.float32)
            future_chunks.append(prediction)
            rolling_history = np.concatenate([rolling_history, prediction], axis=0)
            remaining -= take

        future = np.concatenate(future_chunks, axis=0).astype(np.float32)
        return np.concatenate([current_value, future], axis=0)[:horizon].astype(np.float32)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
    ) -> np.ndarray:
        if signal_name not in self.signal_runtimes:
            if len(self.signal_runtimes) == 1:
                signal_name = next(iter(self.signal_runtimes))
            else:
                available = self.available_signals()
                raise KeyError(f"LSTMForecaster has no runtime model for '{signal_name}'. Available: {available}")

        self._sync_legacy_price_runtime()
        runtime = self.signal_runtimes[signal_name]
        history = np.asarray(history, dtype=np.float32)

        if history.ndim == 1:
            return self._predict_univariate(runtime, history, horizon)

        if history.ndim != 2:
            raise ValueError(f"LSTMForecaster expects 1D or 2D history, got shape {history.shape}")

        predictions = [
            self._predict_univariate(runtime, history[:, column_idx], horizon)
            for column_idx in range(history.shape[1])
        ]
        return np.stack(predictions, axis=0).astype(np.float32)
