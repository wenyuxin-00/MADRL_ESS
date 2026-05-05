from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from configs.cfg import Cfg
from data.loader import load_prosumer_dataset
from predictors.lstm_artifacts import artifact_path, load_lstm_artifact
from predictors.lstm_features import TIME_FEATURE_NONE, encode_time_feature_windows
from utils.torch_runtime import resolve_device


class LSTMForecastRuntime:
    def __init__(self, cfg: Cfg, artifact_dir: str | Path) -> None:
        self.cfg, self.artifact_dir = cfg, Path(artifact_dir)
        self.history_window, self.pred_len = int(cfg.forecast.history_window), int(cfg.obs.sequence_length)
        self.batch_size = int(cfg.forecast.lstm_batch_size)
        self.device = resolve_device(cfg.runtime.device)
        self._cache: dict[tuple[str, int | None, str | None], object] = {}
        self._pv_scale: np.float32 | None = None

    def _model(self, signal_name: str, agent_index: int | None, component: str | None):
        key = (signal_name, agent_index, component)
        if key not in self._cache:
            path = artifact_path(self.artifact_dir, signal_name, agent_index if signal_name == "load" else None, component if signal_name == "load" else None)
            model, scaler, meta = load_lstm_artifact(path, signal_name=signal_name, history_window=self.history_window, pred_len=self.pred_len, agent_index=agent_index if signal_name == "load" else None, map_location=self.device)
            self._cache[key] = (model.to(self.device), scaler, meta)
        return self._cache[key]

    def _scale(self, signal_name: str, agent_index: int | None) -> np.float32:
        if signal_name == "load":
            return np.float32(np.asarray(self.cfg.data.load_scale, dtype=np.float32).reshape(-1)[int(agent_index)])
        if signal_name == "pv":
            capacity = np.asarray(self.cfg.data.pv_capacity_kw, dtype=np.float32).reshape(-1)
            if capacity.size:
                return np.float32(capacity[0])
            if self._pv_scale is None:
                self._pv_scale = np.float32(np.max(load_prosumer_dataset(self.cfg, "train").pv[:, 0]))
                if float(self._pv_scale) <= 0.0:
                    raise ValueError(f"PV physical scale must be positive, got {float(self._pv_scale)}.")
            return self._pv_scale
        return np.float32(1.0)

    def _transform(self, scaler, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape).astype(np.float32)

    def _inverse(self, scaler, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape).astype(np.float32)

    def _features(self, values: np.ndarray, timestamps, scaler, meta: dict, offset: int = 0) -> np.ndarray:
        mode = str(meta["time_feature_mode"])
        value_channel = self._transform(scaler, values)[:, :, None]
        if mode == TIME_FEATURE_NONE:
            return value_channel
        width = int(values.shape[1])
        time_channels = np.asarray(timestamps, dtype=np.float32)[:, int(offset):int(offset) + width, :] if isinstance(timestamps, np.ndarray) else encode_time_feature_windows([row[int(offset):int(offset) + width] for row in timestamps], mode)
        return np.concatenate([value_channel, time_channels], axis=2).astype(np.float32) if time_channels.shape[2] else value_channel

    def predict(self, history: np.ndarray, signal_name: str, agent_index: int | None = None, component: str | None = None, timestamps=None) -> np.ndarray:
        return self.predict_batch(np.asarray(history, dtype=np.float32)[None, :], signal_name, agent_index=agent_index, component=component, timestamps=[timestamps])[0]

    def _raw_prediction(self, values: np.ndarray, timestamps, signal_name: str, agent_index: int | None, component: str | None, offset: int = 0) -> np.ndarray:
        model, scaler, meta = self._model(signal_name, agent_index, component)
        physical_scale = self._scale(signal_name, agent_index)
        normalized = values / physical_scale
        x = torch.from_numpy(self._features(normalized, timestamps, scaler, meta, offset=offset)).to(self.device)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy()
        pred = self._inverse(scaler, pred) * physical_scale
        return pred.astype(np.float32)

    def _predict_batch_direct(self, values: np.ndarray, timestamps, signal_name: str, agent_index: int | None, component: str | None) -> np.ndarray:
        if signal_name == "load":
            _, _, meta = self._model(signal_name, agent_index, component)
            if meta["postprocess_mode"] == "baseline_blend":
                rolling = np.asarray(values, dtype=np.float32).copy()
                preds, weight = [], np.float32(meta["blend_weight"])
                for offset in range(self.pred_len):
                    raw_step = self._raw_prediction(rolling, timestamps, signal_name, agent_index, component, offset=offset)[:, 0]
                    next_step = (rolling[:, -1] + weight * (raw_step - rolling[:, -1])).astype(np.float32)
                    preds.append(next_step)
                    rolling = np.concatenate([rolling[:, 1:], next_step[:, None]], axis=1).astype(np.float32)
                return np.maximum(np.stack(preds, axis=1), 0.0).astype(np.float32)
        pred = self._raw_prediction(values, timestamps, signal_name, agent_index, component)
        return np.maximum(pred, 0.0).astype(np.float32) if signal_name in {"load", "pv"} else pred.astype(np.float32)

    def predict_load(self, component_history: dict[str, np.ndarray], agent_index: int, timestamps=None) -> np.ndarray:
        return sum((self.predict(history, "load", agent_index=agent_index, component=component, timestamps=timestamps) for component, history in component_history.items()), start=np.zeros((self.pred_len,), dtype=np.float32)).astype(np.float32)

    def predict_batch(self, histories: np.ndarray, signal_name: str, agent_index: int | None = None, component: str | None = None, timestamps=None) -> np.ndarray:
        values = np.asarray(histories, dtype=np.float32)
        timestamp_rows = timestamps if timestamps is not None else [None] * int(values.shape[0])
        chunks = []
        for start in range(0, values.shape[0], self.batch_size):
            chunks.append(self._predict_batch_direct(values[start:start + self.batch_size], timestamp_rows[start:start + self.batch_size], signal_name, agent_index, component))
        pred = np.concatenate(chunks, axis=0)
        return np.maximum(pred, 0.0).astype(np.float32) if signal_name in {"load", "pv"} else pred.astype(np.float32)

    def predict_load_batch(self, component_histories: dict[str, np.ndarray], agent_index: int, timestamps=None) -> np.ndarray:
        return sum((self.predict_batch(histories, "load", agent_index=agent_index, component=component, timestamps=timestamps) for component, histories in component_histories.items()), start=np.zeros((next(iter(component_histories.values())).shape[0], self.pred_len), dtype=np.float32)).astype(np.float32)
