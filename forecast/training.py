"""Training and artifact management utilities for forecast models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from forecast.artifacts import (
    DEFAULT_SUPPORTED_FORECAST_SIGNALS,
    get_default_lstm_artifact_dir,
    get_default_lstm_artifact_paths,
    get_weekly_forecast_plot_path,
)
from forecast.lstm_forecaster import LSTMForecaster, save_lstm_forecaster_artifacts
from forecast.lstm_model import LSTMPricePredictor

DEFAULT_WEEK_STEPS = 96 * 7


@dataclass(frozen=True)
class SignalCsvSource:
    """Resolved train/test source files for one signal."""

    signal_name: str
    train_path: Path
    test_path: Path
    value_columns: tuple[str, ...]


@dataclass(frozen=True)
class SignalForecastEvaluation:
    """One-week forecast data used for plotting."""

    signal_name: str
    timestamps: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    source_columns: tuple[str, ...]


def _normalize_signal_name(signal_name: str) -> str:
    return str(signal_name).strip().lower()


def configured_forecast_signals(cfg) -> list[str]:
    """Return normalized target signals from config."""
    unique = []
    for signal_name in cfg.forecast.target_signals:
        normalized = _normalize_signal_name(signal_name)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique or list(DEFAULT_SUPPORTED_FORECAST_SIGNALS)


def forecast_artifact_root(cfg) -> Path:
    """Return the configured forecast artifact root."""
    root = cfg.forecast.lstm_artifact_root
    if root is None:
        return get_default_lstm_artifact_dir()
    return Path(root)


def _signal_columns_from_header(columns: list[str], signal_name: str) -> list[str]:
    signal_name = _normalize_signal_name(signal_name)
    if signal_name == "price":
        return ["price"] if "price" in columns else []

    return [
        column
        for column in columns
        if column == signal_name or column.startswith(f"{signal_name}_") or column.startswith(signal_name)
    ]


def _candidate_signal_file_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    return [
        (data_dir / "simbench_train.csv", data_dir / "simbench_test.csv"),
        (data_dir / "train_prices.csv", data_dir / "test_prices.csv"),
    ]


def resolve_signal_csv_source(data_dir: str | Path, signal_name: str) -> SignalCsvSource | None:
    """Resolve the most suitable train/test CSV pair for one signal."""
    data_dir = Path(data_dir)
    signal_name = _normalize_signal_name(signal_name)

    for train_path, test_path in _candidate_signal_file_pairs(data_dir):
        if not train_path.exists() or not test_path.exists():
            continue

        train_columns = pd.read_csv(train_path, nrows=0).columns.tolist()
        test_columns = pd.read_csv(test_path, nrows=0).columns.tolist()
        value_columns = _signal_columns_from_header(train_columns, signal_name)
        if not value_columns:
            continue
        if any(column not in test_columns for column in value_columns):
            continue

        return SignalCsvSource(
            signal_name=signal_name,
            train_path=train_path,
            test_path=test_path,
            value_columns=tuple(value_columns),
        )

    return None


def load_signal_frame(csv_path: str | Path, signal_name: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load one CSV and select the columns belonging to one signal."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    value_columns = tuple(_signal_columns_from_header(df.columns.tolist(), signal_name))
    if not value_columns:
        raise ValueError(f"CSV '{csv_path}' does not contain signal columns for '{signal_name}'.")
    return df, value_columns


def load_signal_matrix(csv_path: str | Path, signal_name: str) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Load one signal matrix from CSV."""
    df, value_columns = load_signal_frame(csv_path, signal_name)
    values = df.loc[:, list(value_columns)].to_numpy(dtype=np.float32)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.reshape(-1)
    return df, values, value_columns


def fit_signal_scaler(values: np.ndarray) -> MinMaxScaler:
    """Fit a scalar MinMax scaler on all values of one signal family."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.asarray(values, dtype=np.float32).reshape(-1, 1))
    return scaler


def temporal_split_matrix(
    values: np.ndarray,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, np.ndarray]:
    """Temporal train/val/test split for ``(T, D)`` matrices."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    total_steps = values.shape[0]
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= total_steps:
        raise ValueError("Invalid temporal split ratios for the given signal matrix.")

    return {
        "train": values[:train_end],
        "val": values[train_end:val_end],
        "test": values[val_end:],
        "train_end": train_end,
        "val_end": val_end,
    }


def build_supervised_windows_from_matrix(
    values: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    scaler: MinMaxScaler | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a multi-column signal matrix into pooled univariate windows."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    total_window = int(seq_len) + int(pred_len)
    if values.shape[0] < total_window:
        raise ValueError(
            f"Signal matrix is too short for seq_len={seq_len} and pred_len={pred_len}: shape={values.shape}"
        )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for column_idx in range(values.shape[1]):
        series = values[:, column_idx].astype(np.float32)
        if scaler is not None:
            series = scaler.transform(series.reshape(-1, 1)).reshape(-1).astype(np.float32)

        windows = np.lib.stride_tricks.sliding_window_view(series, total_window)
        x_parts.append(windows[:, :seq_len].astype(np.float32))
        y_parts.append(windows[:, seq_len:].astype(np.float32))

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def make_matrix_loader(
    values: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    scaler: MinMaxScaler | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader from a signal matrix."""
    x, y = build_supervised_windows_from_matrix(
        values,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
    )
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle)


def train_lstm_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str | torch.device,
) -> dict[str, object]:
    """Train one LSTM and return the best state dict plus loss curves."""
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": []}

    for _ in range(int(epochs)):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                prediction = model(batch_x)
                val_losses.append(float(criterion(prediction, batch_y).cpu().item()))

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else mean_train
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    return {
        "model": model,
        "history": history,
        "best_val_loss": float(best_val_loss),
        "best_state_dict": best_state_dict,
    }


def build_runtime_forecaster_from_model(
    model,
    *,
    signal_name: str,
    scaler: MinMaxScaler | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> LSTMForecaster:
    """Wrap a trained model as a runtime forecaster for one signal."""
    forecaster = LSTMForecaster(
        model_path=None,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pred_len=pred_len,
        seq_len=seq_len,
        device=device,
        scaler=scaler,
    ).rename_default_signal(signal_name)
    forecaster.signal_runtimes[signal_name].model.load_state_dict(copy.deepcopy(model.state_dict()))
    forecaster.signal_runtimes[signal_name].model.eval()
    forecaster._set_legacy_attributes()
    return forecaster


def _select_week_evaluation_slice(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    signal_name: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one representative week for plotting."""
    if "week_id" in frame.columns and "is_warmup" in frame.columns:
        non_warmup = frame.loc[~frame["is_warmup"].astype(bool)]
        if not non_warmup.empty:
            week_id = int(non_warmup.iloc[0]["week_id"])
            week_mask = frame["week_id"].astype(int) == week_id
            week_frame = frame.loc[week_mask].copy()
            week_values = values[week_mask.to_numpy()]
            warmup_mask = week_frame["is_warmup"].astype(bool).to_numpy()
            history_values = week_values[warmup_mask]
            target_values = week_values[~warmup_mask]
            target_timestamps = week_frame.loc[
                ~week_frame["is_warmup"].astype(bool),
                "timestamp",
            ].to_numpy()
            if history_values.size == 0:
                history_values = week_values[: min(seq_len, len(week_values))]
            return history_values, target_values, target_timestamps

    history = values[:seq_len]
    target = values[seq_len : seq_len + DEFAULT_WEEK_STEPS]
    if target.size == 0:
        target = values[seq_len:]
    if "timestamp" in frame.columns:
        timestamps = frame.loc[seq_len : seq_len + len(target) - 1, "timestamp"].to_numpy()
    else:
        timestamps = np.arange(len(target))
    return history, target, timestamps


def evaluate_signal_one_week(
    source: SignalCsvSource,
    model,
    *,
    signal_name: str,
    scaler: MinMaxScaler | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> SignalForecastEvaluation:
    """Produce one-week predictions for plotting."""
    test_frame, test_values, _ = load_signal_matrix(source.test_path, signal_name)
    history_seed, target_values, timestamps = _select_week_evaluation_slice(
        test_frame,
        test_values,
        signal_name=signal_name,
        seq_len=seq_len,
    )

    forecaster = build_runtime_forecaster_from_model(
        model,
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    prediction = forecaster.predict(
        history_seed,
        horizon=target_values.shape[0] + 1,
        signal_name=signal_name,
    )
    if target_values.ndim == 1 or target_values.shape[1] == 1:
        prediction = np.asarray(prediction, dtype=np.float32)[1 : target_values.shape[0] + 1]
        target = target_values.reshape(-1).astype(np.float32)
    else:
        prediction = np.asarray(prediction, dtype=np.float32)[:, 1 : target_values.shape[0] + 1]
        target = target_values.T.astype(np.float32)

    return SignalForecastEvaluation(
        signal_name=signal_name,
        timestamps=np.asarray(timestamps),
        target=target,
        prediction=np.asarray(prediction, dtype=np.float32),
        source_columns=source.value_columns,
    )


def plot_weekly_forecasts(
    evaluations: list[SignalForecastEvaluation],
    *,
    save_path: str | Path,
) -> Path:
    """Save a combined weekly comparison plot."""
    if not evaluations:
        raise ValueError("plot_weekly_forecasts requires at least one evaluation result.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(len(evaluations), 1, figsize=(14, 4 * len(evaluations)), sharex=False)
    if len(evaluations) == 1:
        axes = [axes]

    for axis, evaluation in zip(axes, evaluations):
        timestamps = evaluation.timestamps
        if evaluation.target.ndim == 1:
            target_curve = evaluation.target
            prediction_curve = evaluation.prediction
            ylabel = evaluation.signal_name
        else:
            target_curve = evaluation.target.sum(axis=0)
            prediction_curve = evaluation.prediction.sum(axis=0)
            ylabel = f"{evaluation.signal_name} (sum)"

        axis.plot(timestamps, target_curve, label="ground truth", linewidth=1.5)
        axis.plot(timestamps, prediction_curve, label="forecast", linewidth=1.5)
        axis.set_title(f"{evaluation.signal_name} weekly forecast")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("timestamp")
    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    return save_path


def train_signal_lstm(cfg, signal_name: str, *, device: str | torch.device | None = None) -> dict[str, object]:
    """Train one signal-specific LSTM and persist its artifacts."""
    data_dir = Path(cfg.data.data_dir or (Path(__file__).resolve().parent.parent / "data"))
    device = torch.device(cfg.runtime.device if device is None else device)
    signal_name = _normalize_signal_name(signal_name)
    source = resolve_signal_csv_source(data_dir, signal_name)
    if source is None:
        raise FileNotFoundError(
            f"No train/test CSV pair found for forecast signal '{signal_name}' under '{data_dir}'."
        )

    _, train_values, value_columns = load_signal_matrix(source.train_path, signal_name)
    split = temporal_split_matrix(
        train_values,
        train_ratio=float(cfg.forecast.lstm_train_ratio),
        val_ratio=float(cfg.forecast.lstm_val_ratio),
    )

    seq_len = int(cfg.forecast.history_window)
    pred_len = int(cfg.env.future_horizon)
    train_values_only = split["train"]
    val_context_start = max(0, int(split["train_end"]) - seq_len)
    val_context = train_values[val_context_start : int(split["val_end"])]

    print(
        f"[forecast] {signal_name}: train_matrix={np.asarray(train_values_only).shape}, "
        f"val_context={np.asarray(val_context).shape}, columns={list(value_columns)}"
    )

    scaler = fit_signal_scaler(train_values_only)
    train_loader = make_matrix_loader(
        train_values_only,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        batch_size=int(cfg.forecast.lstm_batch_size),
        shuffle=True,
    )
    val_loader = make_matrix_loader(
        val_context,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        batch_size=int(cfg.forecast.lstm_batch_size),
        shuffle=False,
    )

    model = LSTMPricePredictor(
        hidden_size=int(cfg.forecast.lstm_hidden_size),
        num_layers=int(cfg.forecast.lstm_num_layers),
        dropout=float(cfg.forecast.lstm_dropout),
        pred_len=pred_len,
    )
    result = train_lstm_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg.forecast.lstm_epochs),
        lr=float(cfg.forecast.lstm_lr),
        device=device,
    )

    artifact_paths = get_default_lstm_artifact_paths(
        root=forecast_artifact_root(cfg),
        signal_name=signal_name,
        future_horizon=pred_len,
    )
    artifact_paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    saved_paths = save_lstm_forecaster_artifacts(
        model_path=artifact_paths["model_path"],
        state_dict=result["best_state_dict"],
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=int(cfg.forecast.lstm_hidden_size),
        num_layers=int(cfg.forecast.lstm_num_layers),
        dropout=float(cfg.forecast.lstm_dropout),
        signal_name=signal_name,
        future_horizon=pred_len,
    )

    evaluation = evaluate_signal_one_week(
        source,
        result["model"],
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=int(cfg.forecast.lstm_hidden_size),
        num_layers=int(cfg.forecast.lstm_num_layers),
        dropout=float(cfg.forecast.lstm_dropout),
        device=device,
    )

    return {
        "signal_name": signal_name,
        "source": source,
        "columns": value_columns,
        "artifact_paths": saved_paths,
        "training": result,
        "evaluation": evaluation,
    }


def collect_available_lstm_artifacts(cfg) -> dict[str, tuple[str, str, str]]:
    """Return all signal artifacts already available for the configured horizon."""
    future_horizon = int(cfg.env.future_horizon)
    artifact_map: dict[str, tuple[str, str, str]] = {}
    for signal_name in configured_forecast_signals(cfg):
        paths = get_default_lstm_artifact_paths(
            root=forecast_artifact_root(cfg),
            signal_name=signal_name,
            future_horizon=future_horizon,
        )
        if all(path.exists() for key, path in paths.items() if key != "artifact_dir"):
            artifact_map[signal_name] = (
                str(paths["model_path"]),
                str(paths["meta_path"]),
                str(paths["scaler_path"]),
            )
    return artifact_map


def ensure_lstm_artifacts(cfg, *, device: str | torch.device | None = None) -> dict[str, object]:
    """Ensure the configured horizon-specific LSTM artifacts exist."""
    if cfg.forecast.lstm_model_path is not None:
        return {
            "mode": "legacy_single_model",
            "artifacts": {
                "price": (
                    str(cfg.forecast.lstm_model_path),
                    None,
                    None,
                )
            },
            "trained_signals": [],
            "plot_path": None,
        }

    artifact_root = forecast_artifact_root(cfg)
    artifact_root.mkdir(parents=True, exist_ok=True)
    trained_results = []
    available_before = collect_available_lstm_artifacts(cfg)

    for signal_name in configured_forecast_signals(cfg):
        if signal_name in available_before:
            continue

        source = resolve_signal_csv_source(cfg.data.data_dir or (Path(__file__).resolve().parent.parent / "data"), signal_name)
        if source is None:
            print(f"[forecast] skip '{signal_name}': no matching train/test CSV source found.")
            continue

        if not bool(cfg.forecast.auto_train_missing):
            continue

        print(
            f"[forecast] training '{signal_name}' LSTM for future_horizon={cfg.env.future_horizon} "
            f"from {source.train_path.name} / {source.test_path.name}"
        )
        trained_results.append(train_signal_lstm(cfg, signal_name, device=device))

    available_after = collect_available_lstm_artifacts(cfg)

    evaluations = [result["evaluation"] for result in trained_results]
    plot_path = get_weekly_forecast_plot_path(
        future_horizon=int(cfg.env.future_horizon),
        root=artifact_root,
    )
    if evaluations:
        plot_path = plot_weekly_forecasts(
            evaluations,
            save_path=plot_path,
        )
    elif not plot_path.exists():
        plot_path = None

    return {
        "mode": "managed_multi_signal",
        "artifacts": available_after,
        "trained_signals": [result["signal_name"] for result in trained_results],
        "plot_path": str(plot_path) if plot_path is not None else None,
    }
