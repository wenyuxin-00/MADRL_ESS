"""Runtime LSTM forecaster with managed artifact metadata."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from predictors.base import Forecaster
from predictors.lstm_model import LSTMForecastModel
from predictors.time_features import (
    TIME_FEATURE_MODE_NONE,
    coerce_timestamp_index,
    encode_forecast_time_features,
    infer_timestamp_step,
    normalize_time_feature_mode,
    pad_history_timestamps_left,
)

LSTM_ARTIFACT_FORMAT = "lstm_forecaster_v4"
LSTM_LOAD_HYBRID_ARTIFACT_FORMAT = "lstm_forecaster_v5"
PHYSICAL_NORMALIZATION_NONE = "none"
PHYSICAL_NORMALIZATION_LOAD_SCALE = "load_scale"
PHYSICAL_NORMALIZATION_PV_PEAK = "pv_peak_kw"
POSTPROCESS_MODE_NONE = "none"
POSTPROCESS_MODE_BASELINE_BLEND = "baseline_blend"
BASELINE_MODE_NONE = "none"
BASELINE_MODE_LAST_VALUE = "last_value"
PHYSICAL_SCALE_EPS = np.float32(1e-6)

LSTM_META_SUFFIX = "_meta.json"
LSTM_SCALER_SUFFIX = "_scaler.pkl"
LSTM_REQUIRED_META_FIELDS = (
    "seq_len",
    "pred_len",
    "hidden_size",
    "num_layers",
    "dropout",
    "input_size",
    "time_feature_mode",
    "model_mode",
)


def _normalize_physical_mode(mode: str | None) -> str:
    normalized = str(mode or PHYSICAL_NORMALIZATION_NONE).strip().lower()
    if normalized not in {
        PHYSICAL_NORMALIZATION_NONE,
        PHYSICAL_NORMALIZATION_LOAD_SCALE,
        PHYSICAL_NORMALIZATION_PV_PEAK,
    }:
        raise ValueError(f"Unsupported physical_normalization_mode '{mode}'.")
    return normalized


def _coerce_physical_scale_array(
    scale_by_column,
    *,
    expected_size: int | None = None,
) -> np.ndarray | None:
    if scale_by_column is None:
        return None
    scale = np.asarray(scale_by_column, dtype=np.float32).reshape(-1)
    if scale.size == 0:
        return None
    if expected_size is not None:
        if scale.size == 1 and expected_size > 1:
            scale = np.repeat(scale, expected_size)
        elif scale.size != expected_size:
            raise ValueError(
                f"physical_scale_by_column size mismatch: expected {expected_size}, got {scale.size}"
            )
    return scale.astype(np.float32, copy=True)


def resolve_lstm_artifact_paths(
    model_path,
    meta_path=None,
    scaler_path=None,
) -> tuple[Path, Path, Path]:
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
    normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
    source_signature: dict[str, object] | None = None,
    input_size: int = 1,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
    model_mode: str = "shared",
    agent_index: int | None = None,
    agent_profile: str | None = None,
    artifact_format: str | None = None,
    postprocess_mode: str = POSTPROCESS_MODE_NONE,
    baseline_mode: str = BASELINE_MODE_NONE,
    blend_weight: float | None = None,
    optimized_metric: str | None = None,
) -> dict[str, str]:
    if scaler is None:
        raise ValueError("LSTM forecaster artifacts require a fitted scaler object.")

    model_path, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, model_path)

    meta = {
        "artifact_format": str(artifact_format or LSTM_ARTIFACT_FORMAT),
        "signal_name": str(signal_name),
        "future_horizon": int(pred_len if future_horizon is None else future_horizon),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "input_size": int(input_size),
        "time_feature_mode": normalize_time_feature_mode(time_feature_mode),
        "model_mode": str(model_mode),
        "normalization_mode": _normalize_physical_mode(normalization_mode),
        "source_signature": dict(source_signature or {}),
        "agent_index": None if agent_index is None else int(agent_index),
        "agent_profile": None if agent_profile is None else str(agent_profile),
    }
    if meta["artifact_format"] == LSTM_LOAD_HYBRID_ARTIFACT_FORMAT or str(postprocess_mode) != POSTPROCESS_MODE_NONE:
        meta.update(
            {
                "postprocess_mode": str(postprocess_mode),
                "baseline_mode": str(baseline_mode),
                "blend_weight": None if blend_weight is None else float(blend_weight),
                "optimized_metric": None if optimized_metric is None else str(optimized_metric),
            }
        )
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
    _, meta_path, scaler_path = resolve_lstm_artifact_paths(model_path, meta_path, scaler_path)

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM meta artifact: '{meta_path}'. Expected '<stem>_meta.json' next to the model."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM scaler artifact: '{scaler_path}'. Expected '<stem>_scaler.pkl' next to the model."
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
    input_size: int = 1
    time_feature_mode: str = TIME_FEATURE_MODE_NONE
    model_mode: str = "shared"
    physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE
    physical_scale_by_column: np.ndarray | None = None
    agent_index: int | None = None
    agent_profile: str | None = None
    postprocess_mode: str = POSTPROCESS_MODE_NONE
    baseline_mode: str = BASELINE_MODE_NONE
    blend_weight: float | None = None
    optimized_metric: str | None = None


class LSTMForecaster(Forecaster):
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
        input_size: int = 1,
        time_feature_mode: str = TIME_FEATURE_MODE_NONE,
        model_mode: str = "shared",
        physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
        physical_scale_by_column=None,
        postprocess_mode: str = POSTPROCESS_MODE_NONE,
        baseline_mode: str = BASELINE_MODE_NONE,
        blend_weight: float | None = None,
        optimized_metric: str | None = None,
        signal_runtimes: dict[str, list[_SignalForecasterRuntime]] | None = None,
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
                input_size=input_size,
                time_feature_mode=time_feature_mode,
                model_mode=model_mode,
                physical_normalization_mode=physical_normalization_mode,
                physical_scale_by_column=physical_scale_by_column,
                postprocess_mode=postprocess_mode,
                baseline_mode=baseline_mode,
                blend_weight=blend_weight,
                optimized_metric=optimized_metric,
                device=self.device,
            )
            signal_runtimes = {"price": [runtime]}

        self.signal_runtimes = {
            str(signal_name): list(runtimes)
            for signal_name, runtimes in signal_runtimes.items()
        }
        self._set_legacy_attributes()

    def _set_legacy_attributes(self) -> None:
        preferred_signal = "price" if "price" in self.signal_runtimes else next(iter(self.signal_runtimes))
        runtime = self.signal_runtimes[preferred_signal][0]
        self.seq_len = int(runtime.seq_len)
        self.pred_len = int(runtime.pred_len)
        self.scaler = runtime.scaler
        self.model = runtime.model

    def _sync_legacy_price_runtime(self) -> None:
        if "price" not in self.signal_runtimes or not self.signal_runtimes["price"]:
            return
        runtime = self.signal_runtimes["price"][0]
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
        input_size: int = 1,
        time_feature_mode: str = TIME_FEATURE_MODE_NONE,
        model_mode: str = "shared",
        physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
        physical_scale_by_column=None,
        agent_index: int | None = None,
        agent_profile: str | None = None,
        postprocess_mode: str = POSTPROCESS_MODE_NONE,
        baseline_mode: str = BASELINE_MODE_NONE,
        blend_weight: float | None = None,
        optimized_metric: str | None = None,
    ) -> _SignalForecasterRuntime:
        model = LSTMForecastModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pred_len=pred_len,
            input_size=input_size,
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
            input_size=int(input_size),
            time_feature_mode=normalize_time_feature_mode(time_feature_mode),
            model_mode=str(model_mode),
            physical_normalization_mode=_normalize_physical_mode(physical_normalization_mode),
            physical_scale_by_column=_coerce_physical_scale_array(physical_scale_by_column),
            agent_index=None if agent_index is None else int(agent_index),
            agent_profile=None if agent_profile is None else str(agent_profile),
            postprocess_mode=str(postprocess_mode or POSTPROCESS_MODE_NONE),
            baseline_mode=str(baseline_mode or BASELINE_MODE_NONE),
            blend_weight=None if blend_weight is None else float(blend_weight),
            optimized_metric=None if optimized_metric is None else str(optimized_metric),
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
            input_size=int(meta.get("input_size", 1)),
            time_feature_mode=meta.get("time_feature_mode", TIME_FEATURE_MODE_NONE),
            model_mode=meta.get("model_mode", "shared"),
            physical_normalization_mode=meta.get("normalization_mode", PHYSICAL_NORMALIZATION_NONE),
            postprocess_mode=meta.get("postprocess_mode", POSTPROCESS_MODE_NONE),
            baseline_mode=meta.get("baseline_mode", BASELINE_MODE_NONE),
            blend_weight=meta.get("blend_weight"),
            optimized_metric=meta.get("optimized_metric"),
            signal_runtimes=None,
        ).rename_default_signal(meta.get("signal_name", signal_name))

    @staticmethod
    def _normalize_artifact_bundle(bundle) -> list[tuple[str, str | None, str | None]]:
        if isinstance(bundle, tuple) and len(bundle) == 3 and not any(
            isinstance(item, (list, tuple)) for item in bundle
        ):
            return [bundle]
        if isinstance(bundle, list):
            if not bundle:
                return []
            return [tuple(item) for item in bundle]
        if isinstance(bundle, tuple) and bundle and all(isinstance(item, (list, tuple)) for item in bundle):
            return [tuple(item) for item in bundle]
        raise TypeError(f"Unsupported signal artifact bundle: {bundle!r}")

    @classmethod
    def from_signal_artifacts(
        cls,
        signal_artifacts: dict[str, object],
        *,
        device: str | torch.device = "cpu",
    ):
        device = torch.device(device)
        signal_runtimes: dict[str, list[_SignalForecasterRuntime]] = {}
        for signal_name, bundle in signal_artifacts.items():
            runtimes: list[_SignalForecasterRuntime] = []
            for model_path, meta_path, scaler_path in cls._normalize_artifact_bundle(bundle):
                meta, scaler = load_lstm_forecaster_artifacts(
                    model_path=model_path,
                    meta_path=meta_path,
                    scaler_path=scaler_path,
                )
                runtimes.append(
                    cls._build_runtime(
                        signal_name=meta.get("signal_name", signal_name),
                        model_path=model_path,
                        hidden_size=int(meta["hidden_size"]),
                        num_layers=int(meta["num_layers"]),
                        dropout=float(meta["dropout"]),
                        pred_len=int(meta["pred_len"]),
                        seq_len=int(meta["seq_len"]),
                        scaler=scaler,
                        input_size=int(meta.get("input_size", 1)),
                        time_feature_mode=meta.get("time_feature_mode", TIME_FEATURE_MODE_NONE),
                        model_mode=meta.get("model_mode", "shared"),
                        physical_normalization_mode=meta.get("normalization_mode", PHYSICAL_NORMALIZATION_NONE),
                        agent_index=meta.get("agent_index"),
                        agent_profile=meta.get("agent_profile"),
                        postprocess_mode=meta.get("postprocess_mode", POSTPROCESS_MODE_NONE),
                        baseline_mode=meta.get("baseline_mode", BASELINE_MODE_NONE),
                        blend_weight=meta.get("blend_weight"),
                        optimized_metric=meta.get("optimized_metric"),
                        device=device,
                    )
                )
            signal_runtimes[str(signal_name)] = runtimes
        return cls(device=device, signal_runtimes=signal_runtimes)

    def rename_default_signal(self, signal_name: str):
        if "price" in self.signal_runtimes and signal_name != "price":
            self.signal_runtimes[str(signal_name)] = self.signal_runtimes.pop("price")
            for runtime in self.signal_runtimes[str(signal_name)]:
                runtime.signal_name = str(signal_name)
        self._set_legacy_attributes()
        return self

    def available_signals(self) -> list[str]:
        return sorted(self.signal_runtimes)

    def reset(self) -> None:
        for runtimes in self.signal_runtimes.values():
            for runtime in runtimes:
                if runtime.physical_normalization_mode != PHYSICAL_NORMALIZATION_NONE:
                    runtime.physical_scale_by_column = None
        return None

    @staticmethod
    def _select_runtime_scale(
        scale_by_column: np.ndarray | None,
        runtime: _SignalForecasterRuntime,
    ) -> np.ndarray | None:
        if scale_by_column is None:
            return None
        if runtime.model_mode == "per_agent" and runtime.agent_index is not None:
            if runtime.agent_index >= scale_by_column.size:
                raise IndexError(
                    f"Runtime agent_index={runtime.agent_index} is out of range for scale size={scale_by_column.size}."
                )
            return np.asarray([scale_by_column[runtime.agent_index]], dtype=np.float32)
        return scale_by_column.astype(np.float32, copy=True)

    def set_episode(
        self,
        episode_signals: dict[str, np.ndarray] | np.ndarray,
        episode_meta: dict[str, object] | None = None,
    ) -> None:
        del episode_signals
        meta = dict(episode_meta or {})
        load_scale = _coerce_physical_scale_array(meta.get("load_scale"))
        pv_peak_kw = _coerce_physical_scale_array(meta.get("pv_peak_kw"))
        if pv_peak_kw is not None and np.any(pv_peak_kw <= 0.0):
            raise ValueError("episode_meta['pv_peak_kw'] must stay positive for PV forecast normalization.")

        for signal_name, runtimes in self.signal_runtimes.items():
            for runtime in runtimes:
                if runtime.physical_normalization_mode == PHYSICAL_NORMALIZATION_LOAD_SCALE:
                    runtime.physical_scale_by_column = self._select_runtime_scale(load_scale, runtime)
                elif runtime.physical_normalization_mode == PHYSICAL_NORMALIZATION_PV_PEAK:
                    runtime.physical_scale_by_column = self._select_runtime_scale(pv_peak_kw, runtime)

    @staticmethod
    def _resolve_column_scale(runtime: _SignalForecasterRuntime, column_idx: int | None) -> np.float32:
        scale_by_column = runtime.physical_scale_by_column
        if scale_by_column is None:
            return np.float32(1.0)
        scale = np.asarray(scale_by_column, dtype=np.float32).reshape(-1)
        if scale.size == 0:
            return np.float32(1.0)
        if column_idx is None:
            column_idx = 0
        if column_idx >= scale.size:
            if scale.size == 1:
                return np.float32(scale[0])
            raise IndexError(
                f"physical_scale_by_column is too short for column {column_idx}: size={scale.size}"
            )
        return np.float32(scale[column_idx])

    @staticmethod
    def _normalize_model_input(
        values: np.ndarray,
        runtime: _SignalForecasterRuntime,
        *,
        column_idx: int | None,
    ) -> tuple[np.ndarray, np.float32]:
        scale = LSTMForecaster._resolve_column_scale(runtime, column_idx)
        normalized = np.asarray(values, dtype=np.float32)
        if runtime.physical_normalization_mode != PHYSICAL_NORMALIZATION_NONE:
            divisor = np.float32(max(float(scale), float(PHYSICAL_SCALE_EPS)))
            normalized = (normalized / divisor).astype(np.float32)
        return normalized, scale

    @staticmethod
    def _restore_prediction_scale(
        values: np.ndarray,
        runtime: _SignalForecasterRuntime,
        *,
        scale: np.float32,
    ) -> np.ndarray:
        restored = np.asarray(values, dtype=np.float32)
        if runtime.physical_normalization_mode != PHYSICAL_NORMALIZATION_NONE:
            restored = (restored * np.float32(scale)).astype(np.float32)
        return restored

    @staticmethod
    def _default_timestamp_start() -> pd.Timestamp:
        return pd.Timestamp("2000-01-01 00:00:00+00:00")

    def _build_model_input_tensor(
        self,
        runtime: _SignalForecasterRuntime,
        model_history: np.ndarray,
        *,
        history_timestamps: pd.DatetimeIndex,
        column_idx: int | None,
    ) -> tuple[torch.Tensor, np.float32]:
        normalized_history, scale = self._normalize_model_input(
            model_history,
            runtime,
            column_idx=column_idx,
        )
        if runtime.scaler is not None:
            load_channel = runtime.scaler.transform(normalized_history.reshape(-1, 1)).reshape(-1).astype(np.float32)
        else:
            load_channel = normalized_history.astype(np.float32)

        if int(runtime.input_size) <= 1 or normalize_time_feature_mode(runtime.time_feature_mode) == TIME_FEATURE_MODE_NONE:
            features = load_channel.reshape(1, -1)
        else:
            time_features = encode_forecast_time_features(history_timestamps, runtime.time_feature_mode)
            features = np.concatenate([load_channel[:, None], time_features], axis=1)[None, :, :]

        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return tensor, scale

    @staticmethod
    def _apply_postprocess(
        runtime: _SignalForecasterRuntime,
        raw_prediction: np.ndarray,
        *,
        baseline_value: np.float32,
    ) -> np.ndarray:
        prediction = np.asarray(raw_prediction, dtype=np.float32)
        if str(runtime.postprocess_mode) != POSTPROCESS_MODE_BASELINE_BLEND:
            return prediction
        if str(runtime.baseline_mode) != BASELINE_MODE_LAST_VALUE:
            raise ValueError(f"Unsupported baseline_mode '{runtime.baseline_mode}'.")
        weight = np.float32(1.0 if runtime.blend_weight is None else float(runtime.blend_weight))
        baseline = np.full(prediction.shape, np.float32(baseline_value), dtype=np.float32)
        return (baseline + weight * (prediction - baseline)).astype(np.float32)

    def _predict_model_chunk(
        self,
        runtime: _SignalForecasterRuntime,
        model_history: np.ndarray,
        *,
        history_timestamps: pd.DatetimeIndex,
        column_idx: int | None,
    ) -> np.ndarray:
        model_tensor, scale = self._build_model_input_tensor(
            runtime,
            model_history,
            history_timestamps=history_timestamps,
            column_idx=column_idx,
        )

        with torch.no_grad():
            prediction = runtime.model(model_tensor).detach().cpu().numpy().reshape(-1)

        if runtime.scaler is not None:
            prediction = runtime.scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)
        return self._restore_prediction_scale(prediction, runtime, scale=scale)

    def _predict_univariate(
        self,
        runtime: _SignalForecasterRuntime,
        history: np.ndarray,
        horizon: int,
        *,
        column_idx: int | None = None,
        history_timestamps: Sequence[str | pd.Timestamp] | None = None,
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
        timestamp_index = coerce_timestamp_index(history_timestamps)
        step_delta = infer_timestamp_step(timestamp_index)
        rolling_timestamps = pad_history_timestamps_left(timestamp_index, rolling_history.size, step=step_delta)
        future_chunks: list[np.ndarray] = []
        remaining = horizon - 1

        while remaining > 0:
            if rolling_history.size < runtime.seq_len:
                model_history = np.concatenate(
                    [np.zeros((runtime.seq_len - rolling_history.size,), dtype=np.float32), rolling_history],
                    axis=0,
                )
            else:
                model_history = rolling_history[-runtime.seq_len :]
            model_timestamps = pad_history_timestamps_left(
                rolling_timestamps,
                runtime.seq_len,
                step=step_delta,
            )

            raw_prediction = self._predict_model_chunk(
                runtime,
                model_history,
                history_timestamps=model_timestamps,
                column_idx=column_idx,
            )

            if str(runtime.postprocess_mode) == POSTPROCESS_MODE_BASELINE_BLEND:
                next_value = self._apply_postprocess(
                    runtime,
                    np.asarray(raw_prediction[:1], dtype=np.float32),
                    baseline_value=np.float32(rolling_history[-1]),
                )
                prediction_chunk = np.asarray(next_value[:1], dtype=np.float32)
                take = 1
            else:
                take = min(runtime.pred_len, remaining)
                prediction_chunk = np.asarray(raw_prediction[:take], dtype=np.float32)
            future_chunks.append(prediction_chunk)

            rolling_history = np.concatenate([rolling_history, prediction_chunk], axis=0)
            if take > 0:
                last_timestamp = (
                    rolling_timestamps[-1]
                    if len(rolling_timestamps)
                    else self._default_timestamp_start()
                )
                extension = pd.date_range(start=last_timestamp + step_delta, periods=take, freq=step_delta)
                rolling_timestamps = rolling_timestamps.append(extension)
            remaining -= take

        future = np.concatenate(future_chunks, axis=0).astype(np.float32)
        return np.concatenate([current_value, future], axis=0)[:horizon].astype(np.float32)

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
        history_timestamps: Sequence[str | pd.Timestamp] | None = None,
    ) -> np.ndarray:
        if signal_name not in self.signal_runtimes:
            if len(self.signal_runtimes) == 1:
                signal_name = next(iter(self.signal_runtimes))
            else:
                raise KeyError(
                    f"LSTMForecaster has no runtime model for '{signal_name}'. Available: {self.available_signals()}"
                )

        self._sync_legacy_price_runtime()
        runtimes = self.signal_runtimes[signal_name]
        history = np.asarray(history, dtype=np.float32)

        if history.ndim == 1:
            runtime = runtimes[0]
            return self._predict_univariate(
                runtime,
                history,
                horizon,
                column_idx=0,
                history_timestamps=history_timestamps,
            )

        if history.ndim != 2:
            raise ValueError(f"LSTMForecaster expects 1D or 2D history, got shape {history.shape}")

        predictions: list[np.ndarray] = []
        for column_idx in range(history.shape[1]):
            runtime = runtimes[column_idx] if len(runtimes) == history.shape[1] else runtimes[0]
            runtime_column_idx = 0 if len(runtimes) == history.shape[1] else column_idx
            predictions.append(
                self._predict_univariate(
                    runtime,
                    history[:, column_idx],
                    horizon,
                    column_idx=runtime_column_idx,
                    history_timestamps=history_timestamps,
                )
            )
        return np.stack(predictions, axis=0).astype(np.float32)
