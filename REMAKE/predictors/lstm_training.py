from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from REMAKE.configs.cfg import Cfg
from REMAKE.data.loader import load_prosumer_dataset
from REMAKE.predictors.lstm_artifacts import artifact_path, save_lstm_artifact
from REMAKE.predictors.lstm_features import TIME_FEATURE_NONE, encode_time_feature_windows, time_feature_dim
from REMAKE.predictors.lstm_model import LSTMForecastModel
from REMAKE.utils.torch_runtime import resolve_device


@dataclass(frozen=True)
class LSTMTrainSpec:
    signal: str
    agent_index: int | None
    component: str | None
    series: np.ndarray
    timestamps: list[str]


def _windows(series: np.ndarray, timestamps: list[str], history: int, pred_len: int) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    count = values.shape[0] - history - pred_len + 1
    x = np.stack([values[i:i + history] for i in range(count)]).astype(np.float32)
    y = np.stack([values[i + history:i + history + pred_len] for i in range(count)]).astype(np.float32)
    ts = [timestamps[i:i + history] for i in range(count)]
    return x, y, ts


def _temporal_split(series: np.ndarray, timestamps: list[str], cfg: Cfg) -> dict[str, object]:
    total = len(series)
    train_end = int(total * float(cfg.forecast.lstm_train_ratio))
    val_end = int(total * (float(cfg.forecast.lstm_train_ratio) + float(cfg.forecast.lstm_val_ratio)))
    val_start = max(0, train_end - int(cfg.forecast.history_window))
    return {"train_series": np.asarray(series[:train_end], dtype=np.float32), "train_timestamps": timestamps[:train_end], "val_series": np.asarray(series[val_start:val_end], dtype=np.float32), "val_timestamps": timestamps[val_start:val_end], "train_end": train_end, "val_end": val_end}


def _signal_params(cfg: Cfg, signal_name: str) -> dict[str, object]:
    prefix = "price" if signal_name == "wholesale_price" else signal_name
    return {
        "epochs": int(getattr(cfg.forecast, f"{prefix}_lstm_epochs")),
        "hidden_size": int(getattr(cfg.forecast, f"{prefix}_lstm_hidden_size")),
        "num_layers": int(getattr(cfg.forecast, f"{prefix}_lstm_num_layers")),
        "dropout": float(getattr(cfg.forecast, f"{prefix}_lstm_dropout")),
        "lr": float(getattr(cfg.forecast, f"{prefix}_lstm_lr")),
    }


def _time_feature_mode(cfg: Cfg, signal_name: str) -> str:
    if signal_name == "load":
        return str(cfg.forecast.load_lstm_time_features)
    if signal_name == "pv":
        return str(cfg.forecast.pv_lstm_time_features)
    return TIME_FEATURE_NONE


def _scaler_kind(cfg: Cfg, signal_name: str) -> str:
    return str(cfg.forecast.load_lstm_scaler) if signal_name == "load" else "standard"


def _physical_scale(cfg: Cfg, signal_name: str, agent_index: int | None, series: np.ndarray | None = None) -> float:
    if signal_name == "load":
        return float(np.asarray(cfg.data.load_scale, dtype=np.float32).reshape(-1)[int(agent_index)])
    if signal_name == "pv":
        capacity = np.asarray(cfg.data.pv_capacity_kw, dtype=np.float32).reshape(-1)
        if capacity.size:
            return float(capacity[0])
        if series is None:
            raise ValueError("PV physical scale requires the PV training series when pv_capacity_kw is empty.")
        scale = float(np.max(np.asarray(series, dtype=np.float32)))
        if scale <= 0.0:
            raise ValueError(f"PV physical scale must be positive, got {scale}.")
        return scale
    return 1.0


def _fit_scaler(values: np.ndarray, kind: str):
    scaler = RobustScaler() if str(kind) == "robust" else StandardScaler()
    return scaler.fit(np.asarray(values, dtype=np.float32).reshape(-1, 1))


def _transform(scaler, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape).astype(np.float32)


def _inverse(scaler, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape).astype(np.float32)


def _window_bundle(series: np.ndarray, timestamps: list[str], history: int, pred_len: int, scale: np.float32, scaler, time_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(series, dtype=np.float32)
    x, y, ts = _windows(raw / scale, timestamps, history, pred_len)
    value_channel = _transform(scaler, x)[:, :, None]
    time_channels = encode_time_feature_windows(ts, time_mode)
    features = np.concatenate([value_channel, time_channels], axis=2).astype(np.float32) if time_channels.shape[2] else value_channel
    return features, _transform(scaler, y), raw[history - 1:history - 1 + len(x)], raw[history:history + len(x)]


def _predict_step1_scaled(model: LSTMForecastModel, features: np.ndarray, cfg: Cfg) -> np.ndarray:
    device = resolve_device(cfg.runtime.device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(cfg.forecast.lstm_batch_size)):
            xb = torch.from_numpy(features[start:start + int(cfg.forecast.lstm_batch_size)]).to(device)
            outputs.append(model(xb)[:, 0].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _select_blend_weight(cfg: Cfg, model: LSTMForecastModel, scaler, val_features: np.ndarray, val_last: np.ndarray, val_target: np.ndarray, scale: np.float32, component: str | None) -> tuple[float, float, str]:
    raw = _inverse(scaler, _predict_step1_scaled(model, val_features, cfg)) * scale
    if component == "heatpump":
        return 0.0, float(np.mean(np.abs(val_last - val_target))), "blocked_bias_guard_step1"
    candidates = np.asarray(cfg.forecast.load_blend_candidates, dtype=np.float32)
    maes = np.asarray([np.mean(np.abs((val_last + weight * (raw - val_last)) - val_target)) for weight in candidates], dtype=np.float32)
    idx = int(np.argmin(maes))
    return float(candidates[idx]), float(maes[idx]), "mae_step1"


def _fit_model(cfg: Cfg, spec: LSTMTrainSpec, desc: str, params: dict[str, object]) -> tuple[LSTMForecastModel, object, dict[str, float]]:
    history, pred_len = int(cfg.forecast.history_window), int(cfg.obs.sequence_length)
    scale = np.float32(_physical_scale(cfg, spec.signal, spec.agent_index, spec.series))
    split = _temporal_split(spec.series, spec.timestamps, cfg)
    scaler = _fit_scaler(np.asarray(split["train_series"], dtype=np.float32) / scale, _scaler_kind(cfg, spec.signal))
    time_mode = _time_feature_mode(cfg, spec.signal)
    x_train, y_train, _, _ = _window_bundle(split["train_series"], split["train_timestamps"], history, pred_len, scale, scaler, time_mode)
    x_val, y_val, val_last, val_target = _window_bundle(split["val_series"], split["val_timestamps"], history, pred_len, scale, scaler, time_mode)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=int(cfg.forecast.lstm_batch_size), shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=int(cfg.forecast.lstm_batch_size), shuffle=False)
    device = resolve_device(cfg.runtime.device)
    model = LSTMForecastModel(params["hidden_size"], params["num_layers"], params["dropout"], pred_len, input_size=1 + time_feature_dim(time_mode)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    loss_fn = torch.nn.MSELoss()
    best_epoch, best_val, best_train = 0, float("inf"), float("inf")
    best_state = copy.deepcopy({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    for epoch in tqdm(range(1, int(params["epochs"]) + 1), desc=desc, unit="epoch", leave=False, ascii=True):
        model.train(); train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optim.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optim.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        if val_loss < best_val:
            best_epoch, best_val, best_train = epoch, val_loss, train_loss
            best_state = copy.deepcopy({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    model.load_state_dict(best_state); model.to(device)
    stats = {"train_loss": best_train, "val_loss": best_val, "best_epoch": float(best_epoch), "n_windows": int(x_train.shape[0]), "val_windows": int(x_val.shape[0])}
    if spec.signal == "load":
        blend_weight, blend_mae, metric = _select_blend_weight(cfg, model, scaler, x_val, val_last, val_target, scale, spec.component)
        stats.update({"blend_weight": blend_weight, "val_step1_mae": blend_mae, "optimized_metric": metric})
    return model.cpu(), scaler, stats


def _signal_series(cfg: Cfg, signal_name: str) -> list[LSTMTrainSpec]:
    data = load_prosumer_dataset(cfg, "train")
    if signal_name == "load":
        return [LSTMTrainSpec(signal_name, idx, component, data.load_components[component][:, idx], data.timestamps) for component in cfg.data.load_components for idx in range(int(cfg.env.num_agents))]
    return {"wholesale_price": [LSTMTrainSpec(signal_name, None, None, data.price, data.timestamps)], "pv": [LSTMTrainSpec(signal_name, None, None, data.pv[:, 0], data.timestamps)]}[signal_name]


def train_signal_lstm(cfg: Cfg, signal_name: str, artifact_dir: Path, overwrite: bool = False) -> dict[str, object]:
    rows, paths = [], {}
    params = _signal_params(cfg, signal_name)
    for spec in tqdm(_signal_series(cfg, signal_name), desc=f"{signal_name} models", unit="model", leave=False, ascii=True):
        out = artifact_path(artifact_dir, signal_name, spec.agent_index if signal_name == "load" else None, spec.component if signal_name == "load" else None)
        label = f"LSTM {signal_name}" if spec.agent_index is None else f"LSTM {signal_name}/{spec.component}/{spec.agent_index}"
        model, scaler, stats = _fit_model(cfg, spec, desc=label, params=params)
        time_mode = _time_feature_mode(cfg, signal_name)
        meta = {"signal_name": signal_name, "agent_index": spec.agent_index if signal_name == "load" else None, "component": spec.component, "history_window": int(cfg.forecast.history_window), "pred_len": int(cfg.obs.sequence_length), "hidden_size": int(params["hidden_size"]), "num_layers": int(params["num_layers"]), "dropout": float(params["dropout"]), "input_size": 1 + time_feature_dim(time_mode), "time_feature_mode": time_mode, "scaler_kind": _scaler_kind(cfg, signal_name), "physical_scale": "load_scale" if signal_name == "load" else "pv_reference_peak" if signal_name == "pv" else "none", "best_epoch": int(stats["best_epoch"]), "best_val_loss": float(stats["val_loss"])}
        if signal_name == "load":
            meta.update({"postprocess_mode": "baseline_blend", "baseline_mode": "last_value", "blend_weight": float(stats["blend_weight"]), "optimized_metric": str(stats["optimized_metric"])})
        save_lstm_artifact(out, model, scaler, meta)
        paths["shared" if spec.agent_index is None else f"{spec.component}/agent_{spec.agent_index}"] = out
        rows.append({"signal": signal_name, "agent_index": spec.agent_index, "component": spec.component, "model_path": str(out), **stats, **params})
    return {"artifact_paths": paths, "summary_rows": rows}


def train_forecasters(cfg: Cfg, run_dir: Path, signals: tuple[str, ...] | None = None, overwrite: bool = False) -> dict[str, object]:
    artifact_dir = Path(cfg.forecast.lstm_artifact_dir or Path(run_dir) / "forecast" / "artifacts")
    rows, artifacts = [], {}
    for signal_name in tqdm(tuple(signals or cfg.forecast.target_signals), desc="LSTM signals", unit="signal", ascii=True):
        result = train_signal_lstm(cfg, signal_name, artifact_dir=artifact_dir, overwrite=overwrite)
        artifacts[signal_name] = result["artifact_paths"]; rows.extend(result["summary_rows"])
    train_df = pd.DataFrame(rows)
    tables = Path(run_dir) / "forecast" / "tables"; tables.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(tables / "forecast_train_summary.csv", index=False)
    return {"artifact_dir": artifact_dir, "artifacts": artifacts, "summary": train_df}
