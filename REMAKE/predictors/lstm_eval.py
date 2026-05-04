from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from REMAKE.configs.cfg import Cfg
from REMAKE.data.loader import load_prosumer_dataset
from REMAKE.data.share_data import ShareData, load_share_data
from REMAKE.predictors.lstm_features import encode_time_features
from REMAKE.predictors.lstm_loader import LSTMForecastRuntime


def _metrics(frame: pd.DataFrame) -> pd.Series:
    ape = frame["ape"].dropna()
    return pd.Series({"row_count": int(len(frame)), "rmse": float(np.sqrt(frame["sq_error"].mean())), "mae": float(frame["abs_error"].mean()), "mape_pct": float(ape.mean() * 100.0)})


def _heatpump_step1(cfg: Cfg, share_data: ShareData) -> tuple[np.ndarray, np.ndarray]:
    data = load_prosumer_dataset(cfg, "eval", pad_history_steps=cfg.forecast.history_window, pad_future_steps=cfg.obs.sequence_length)
    starts = share_data.eval["episode_starts"].astype(np.int64)
    steps, h, horizon, n = int(cfg.env.episode_steps), int(cfg.forecast.history_window), int(cfg.obs.sequence_length), int(cfg.env.num_agents)
    idxs = np.asarray([int(start) + step for start in starts for step in range(steps)], dtype=np.int64)
    time_features = encode_time_features(data.timestamps, cfg.forecast.load_lstm_time_features)
    timestamps = np.stack([time_features[idx - h:idx + horizon] for idx in idxs]).astype(np.float32)
    runtime = LSTMForecastRuntime(cfg, share_data.manifest["forecast_artifact_dir"])
    target, prediction = np.zeros((len(starts), steps, n), dtype=np.float32), np.zeros((len(starts), steps, n), dtype=np.float32)
    for agent in range(n):
        histories = np.stack([data.load_components["heatpump"][idx - h:idx, agent] for idx in idxs]).astype(np.float32)
        prediction[:, :, agent] = runtime.predict_batch(histories, "load", agent_index=agent, component="heatpump", timestamps=timestamps)[:, 0].reshape(len(starts), steps)
        target[:, :, agent] = data.load_components["heatpump"][idxs, agent].reshape(len(starts), steps)
    return target, prediction


def build_test_predictions_from_share_data(cfg: Cfg, share_data_dir: str | Path) -> pd.DataFrame:
    share_data = load_share_data(share_data_dir, cfg)
    data = share_data.eval
    heatpump_target, heatpump_prediction = _heatpump_step1(cfg, share_data)
    rows = []
    episode_count, steps = data["price"].shape
    for episode in range(int(episode_count)):
        for step in range(int(steps)):
            timestamp = str(data["timestamps"][episode, step])
            rows.append({"episode": episode, "step": step, "timestamp": timestamp, "signal_name": "wholesale_price", "series_name": "wholesale_price", "target": float(data["price"][episode, step]), "prediction": float(data["lstm_price_seq"][episode, step, 0])})
            rows.append({"episode": episode, "step": step, "timestamp": timestamp, "signal_name": "pv", "series_name": "shared_pv", "target": float(data["pv"][episode, step]), "prediction": float(data["lstm_pv_seq"][episode, step, 0])})
            for agent, profile in enumerate(cfg.data.agent_profiles):
                rows.append({"episode": episode, "step": step, "timestamp": timestamp, "signal_name": "load", "series_name": str(profile), "target": float(data["load"][episode, step, agent]), "prediction": float(data["lstm_load_seq"][episode, step, 0, agent])})
                rows.append({"episode": episode, "step": step, "timestamp": timestamp, "signal_name": "heatpump", "series_name": str(profile), "target": float(heatpump_target[episode, step, agent]), "prediction": float(heatpump_prediction[episode, step, agent])})
    df = pd.DataFrame(rows)
    error = df["prediction"].to_numpy(dtype=np.float64) - df["target"].to_numpy(dtype=np.float64)
    df["abs_error"] = np.abs(error)
    df["sq_error"] = np.square(error)
    df["ape"] = df["abs_error"] / np.where(np.abs(df["target"].to_numpy(dtype=np.float64)) > 1e-6, np.abs(df["target"].to_numpy(dtype=np.float64)), np.nan)
    return df


def summarize_test_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(["signal_name", "series_name"], as_index=False).apply(_metrics, include_groups=False).reset_index(drop=True)


def write_test_outputs_from_share_data(cfg: Cfg, share_data_dir: str | Path, run_dir: str | Path) -> dict[str, pd.DataFrame]:
    tables = Path(run_dir) / "forecast" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    predictions = build_test_predictions_from_share_data(cfg, share_data_dir)
    metrics = summarize_test_predictions(predictions)
    predictions.to_csv(tables / "test_predictions.csv", index=False)
    metrics.to_csv(tables / "test_metrics.csv", index=False)
    metrics.to_csv(tables / "forecast_compare_summary.csv", index=False)
    return {"predictions": predictions, "metrics": metrics}
