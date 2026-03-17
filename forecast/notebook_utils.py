"""forecast.ipynb 使用的轻量工具。

这里不做复杂框架，只收纳预测 notebook 直接需要的公共逻辑：
- 读取价格序列
- 时序切分
- 构造监督学习样本
- 训练 LSTM
- 执行 one-step rolling forecast
- 执行更贴近 runtime 的 block forecast 评估
- 计算评估指标
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from forecast.lstm_forecaster import LSTMForecaster


def load_price_series(csv_path: str | Path) -> np.ndarray:
    """从 CSV 中读取 price 列。"""
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")

    if "price" not in header:
        raise ValueError(f"CSV must contain a 'price' column, got header={header}")

    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw[:, header.index("price")].astype(np.float32)


def temporal_split(
    series: np.ndarray,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, np.ndarray]:
    """按时间顺序切分 train / val / test。"""
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    total = len(series)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= total:
        raise ValueError("Invalid temporal split ratios for the given series length.")
    return {
        "train": series[:train_end],
        "val": series[train_end:val_end],
        "test": series[val_end:],
    }


def fit_minmax_scaler(series: np.ndarray) -> MinMaxScaler:
    """只用训练集拟合归一化器。"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.asarray(series, dtype=np.float32).reshape(-1, 1))
    return scaler


def build_supervised_windows(
    series: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    scaler: MinMaxScaler | None,
) -> tuple[np.ndarray, np.ndarray]:
    """把单变量价格序列切成监督学习窗口。"""
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    min_length = seq_len + pred_len
    if len(series) < min_length:
        raise ValueError(
            f"Series is too short for seq_len={seq_len} and pred_len={pred_len}: len={len(series)}"
        )

    values = series
    if scaler is not None:
        values = scaler.transform(series.reshape(-1, 1)).reshape(-1).astype(np.float32)

    x_list, y_list = [], []
    for start in range(len(values) - seq_len - pred_len + 1):
        x_list.append(values[start : start + seq_len])
        y_list.append(values[start + seq_len : start + seq_len + pred_len])

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def make_sequence_loader(
    series: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    scaler: MinMaxScaler | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """基于单段时间序列创建 DataLoader。"""
    x, y = build_supervised_windows(series, seq_len=seq_len, pred_len=pred_len, scaler=scaler)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle)


def prepare_lstm_data_bundle(
    price_series: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
) -> dict[str, object]:
    """准备 forecast notebook 所需的数据切分、scaler 与 DataLoader。"""
    price_series = np.asarray(price_series, dtype=np.float32).reshape(-1)
    splits = temporal_split(price_series, train_ratio=train_ratio, val_ratio=val_ratio)
    train_end = len(splits["train"])
    val_end = train_end + len(splits["val"])
    val_context_start = max(0, train_end - seq_len)
    val_context = price_series[val_context_start:val_end]

    scaler = fit_minmax_scaler(splits["train"])
    train_loader = make_sequence_loader(
        splits["train"],
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = make_sequence_loader(
        val_context,
        seq_len=seq_len,
        pred_len=pred_len,
        scaler=scaler,
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "price_series": price_series,
        "splits": splits,
        "train_end": train_end,
        "val_end": val_end,
        "val_context": val_context,
        "test_history_seed": price_series[:val_end],
        "scaler": scaler,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def train_lstm_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str | torch.device,
) -> dict[str, object]:
    """训练 LSTM 并返回最优权重与损失历史。"""
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
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_losses.append(float(criterion(pred, batch_y).cpu().item()))

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


def _build_runtime_forecaster(
    model,
    *,
    scaler: MinMaxScaler | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> LSTMForecaster:
    """把训练好的模型包装成 runtime 同款 forecaster。"""
    forecaster = LSTMForecaster(
        model_path=None,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pred_len=pred_len,
        seq_len=seq_len,
        device=device,
        scaler=scaler,
    )
    forecaster.model.load_state_dict(copy.deepcopy(model.state_dict()))
    forecaster.model.eval()
    return forecaster


def rolling_forecast(
    model,
    *,
    history_seed: np.ndarray,
    target_series: np.ndarray,
    scaler: MinMaxScaler | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> np.ndarray:
    """执行 one-step rolling 预测。

    每一步都只使用当时已经观测到的真实价格，预测下一个时刻价格。
    """
    forecaster = _build_runtime_forecaster(
        model,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    history = np.asarray(history_seed, dtype=np.float32).reshape(-1).copy()
    target_series = np.asarray(target_series, dtype=np.float32).reshape(-1)
    predictions = []

    for true_value in target_series:
        pred_window = forecaster.predict(history, horizon=2)
        predictions.append(float(pred_window[-1]))
        history = np.concatenate([history, np.array([true_value], dtype=np.float32)])

    return np.asarray(predictions, dtype=np.float32)


def evaluate_one_step_forecast(
    model,
    *,
    history_seed: np.ndarray,
    target_series: np.ndarray,
    scaler: MinMaxScaler | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
) -> dict[str, object]:
    """执行 one-step rolling 评估并返回预测结果与指标。"""
    predictions = rolling_forecast(
        model,
        history_seed=history_seed,
        target_series=target_series,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )
    metrics = compute_forecast_metrics(target_series, predictions)
    return {
        "predictions": predictions,
        "metrics": metrics,
    }


def collect_block_forecasts(
    model,
    *,
    history_seed: np.ndarray,
    target_series: np.ndarray,
    scaler: MinMaxScaler | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
    forecast_horizon: int,
    block_stride: int | None = None,
) -> list[dict[str, object]]:
    """按块收集更贴近 runtime 用法的多步预测窗口。"""
    forecast_horizon = int(forecast_horizon)
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    block_stride = forecast_horizon if block_stride is None else int(block_stride)
    if block_stride <= 0:
        raise ValueError("block_stride must be positive.")

    forecaster = _build_runtime_forecaster(
        model,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    history = np.asarray(history_seed, dtype=np.float32).reshape(-1).copy()
    target_series = np.asarray(target_series, dtype=np.float32).reshape(-1)
    blocks: list[dict[str, object]] = []

    for start_step in range(0, len(target_series), block_stride):
        actual_window = target_series[start_step : start_step + forecast_horizon]
        if actual_window.size == 0:
            break

        pred_window = forecaster.predict(history, horizon=actual_window.size + 1)[1:]
        metrics = compute_forecast_metrics(actual_window, pred_window)
        blocks.append(
            {
                "start_step": int(start_step),
                "prediction": pred_window.astype(np.float32),
                "target": actual_window.astype(np.float32),
                "metrics": metrics,
            }
        )

        consumed_truth = target_series[start_step : start_step + min(block_stride, len(target_series) - start_step)]
        history = np.concatenate([history, consumed_truth.astype(np.float32)])

    return blocks


def summarize_block_forecasts(blocks: list[dict[str, object]]) -> dict[str, float]:
    """汇总 block forecast 的窗口级指标。"""
    if not blocks:
        return {"mean_block_mae": 0.0, "mean_block_rmse": 0.0, "mean_block_mape": 0.0, "n_blocks": 0}

    maes = [float(block["metrics"]["mae"]) for block in blocks]
    rmses = [float(block["metrics"]["rmse"]) for block in blocks]
    mapes = [float(block["metrics"]["mape"]) for block in blocks]
    return {
        "mean_block_mae": float(np.mean(maes)),
        "mean_block_rmse": float(np.mean(rmses)),
        "mean_block_mape": float(np.mean(mapes)),
        "n_blocks": int(len(blocks)),
    }


def evaluate_block_forecast(
    model,
    *,
    history_seed: np.ndarray,
    target_series: np.ndarray,
    scaler: MinMaxScaler | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
    forecast_horizon: int,
    block_stride: int | None = None,
) -> dict[str, object]:
    """执行 block forecast 评估，并返回窗口列表与汇总指标。"""
    blocks = collect_block_forecasts(
        model,
        history_seed=history_seed,
        target_series=target_series,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
        forecast_horizon=forecast_horizon,
        block_stride=block_stride,
    )
    return {
        "blocks": blocks,
        "metrics": summarize_block_forecasts(blocks),
    }


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算常用预测指标。"""
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)))
    return {"mae": mae, "rmse": rmse, "mape": mape}
