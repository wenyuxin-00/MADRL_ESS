"""Forecast evaluation and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from predictors.lstm_forecaster import (
    BASELINE_MODE_NONE,
    PHYSICAL_NORMALIZATION_NONE,
    POSTPROCESS_MODE_NONE,
)
from predictors.time_features import TIME_FEATURE_MODE_NONE

if TYPE_CHECKING:
    from predictors.training import SignalCsvSource


def _training_module():
    from predictors import training as training_module

    return training_module


def _signal_forecast_evaluation_cls():
    return _training_module().SignalForecastEvaluation


def _default_week_steps() -> int:
    return int(_training_module().DEFAULT_WEEK_STEPS)


def _load_signal_matrix_from_source(source, signal_name: str, *, split: str):
    return _training_module()._load_signal_matrix_from_source(source, signal_name, split=split)


def _build_runtime_forecaster_from_model(
    model,
    *,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
    input_size: int = 1,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
    model_mode: str = "shared",
    physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    postprocess_mode: str = POSTPROCESS_MODE_NONE,
    baseline_mode: str = BASELINE_MODE_NONE,
    blend_weight: float | None = None,
    optimized_metric: str | None = None,
):
    return _training_module().build_runtime_forecaster_from_model(
        model,
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
        input_size=input_size,
        time_feature_mode=time_feature_mode,
        model_mode=model_mode,
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=physical_scale_by_column,
        agent_index=agent_index,
        agent_profile=agent_profile,
        postprocess_mode=postprocess_mode,
        baseline_mode=baseline_mode,
        blend_weight=blend_weight,
        optimized_metric=optimized_metric,
    )


def _reshape_signal_values(values: np.ndarray) -> np.ndarray:
    return _training_module()._reshape_signal_values(values)


def _select_week_evaluation_slice(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    signal_name: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select one representative evaluation week for plotting."""
    _ = signal_name
    if "week_id" in frame.columns and "is_warmup" in frame.columns:
        non_warmup = frame.loc[~frame["is_warmup"].astype(bool)]
        if not non_warmup.empty:
            first_target_row = non_warmup.iloc[0]
            target_week_id = int(first_target_row["week_id"])
            segment_mask = np.ones(len(frame), dtype=bool)
            if "segment_id" in frame.columns:
                segment_mask = frame["segment_id"].astype(int).to_numpy() == int(first_target_row["segment_id"])

            warmup_mask = segment_mask & frame["is_warmup"].astype(bool).to_numpy()
            target_mask = (
                segment_mask
                & ~frame["is_warmup"].astype(bool).to_numpy()
                & (frame["week_id"].astype(int).to_numpy() == target_week_id)
            )

            history_values = values[warmup_mask]
            target_values = values[target_mask]
            target_timestamps = frame.loc[target_mask, "timestamp"].to_numpy()
            if history_values.size == 0:
                segment_values = values[segment_mask]
                history_values = segment_values[: min(seq_len, len(segment_values))]
                history_timestamps = frame.loc[segment_mask].iloc[: len(history_values)]["timestamp"].to_numpy()
            else:
                history_timestamps = frame.loc[warmup_mask, "timestamp"].to_numpy()
            return history_values, target_values, history_timestamps, target_timestamps

    history = values[:seq_len]
    target = values[seq_len : seq_len + _default_week_steps()]
    if target.size == 0:
        target = values[seq_len:]
    history_timestamps = (
        frame.loc[: len(history) - 1, "timestamp"].to_numpy()
        if "timestamp" in frame.columns
        else np.arange(len(history))
    )
    if "timestamp" in frame.columns:
        timestamps = frame.loc[seq_len : seq_len + len(target) - 1, "timestamp"].to_numpy()
    else:
        timestamps = np.arange(len(target))
    return history, target, np.asarray(history_timestamps), timestamps


def compute_forecast_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Compute RMSE, MAE, and MAPE over a one-week evaluation slice."""
    target_array = np.asarray(target, dtype=np.float32)
    prediction_array = np.asarray(prediction, dtype=np.float32)
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            "Forecast metric computation requires target/prediction to share the same shape, "
            f"got {target_array.shape} vs {prediction_array.shape}."
        )

    flat_target = target_array.reshape(-1)
    flat_prediction = prediction_array.reshape(-1)
    error = flat_prediction - flat_target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))

    non_zero_mask = np.abs(flat_target) > 1e-8
    mape = None
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(error[non_zero_mask] / flat_target[non_zero_mask])) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def _to_forecaster_history_input(values: np.ndarray) -> np.ndarray:
    """Shape evaluation history for ``forecaster.predict``."""
    history = np.asarray(values, dtype=np.float32)
    if history.ndim == 2 and history.shape[1] == 1:
        return history.reshape(-1)
    return history


def _reshape_evaluation_target(values: np.ndarray) -> np.ndarray:
    """Normalize evaluation targets to ``(T,)`` or ``(D, T)``."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values.reshape(-1)
    if values.ndim == 2 and values.shape[1] == 1:
        return values.reshape(-1)
    return values.T.astype(np.float32)


def evaluate_signal_open_loop_one_week(
    source: SignalCsvSource,
    model,
    *,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
    input_size: int = 1,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
    model_mode: str = "shared",
    physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    postprocess_mode: str = POSTPROCESS_MODE_NONE,
    baseline_mode: str = BASELINE_MODE_NONE,
    blend_weight: float | None = None,
    optimized_metric: str | None = None,
):
    """Run an open-loop one-week stress test."""
    test_frame, test_values, _ = _load_signal_matrix_from_source(source, signal_name, split="test")
    history_seed, target_values, history_timestamps, timestamps = _select_week_evaluation_slice(
        test_frame,
        test_values,
        signal_name=signal_name,
        seq_len=seq_len,
    )

    forecaster = _build_runtime_forecaster_from_model(
        model,
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        input_size=input_size,
        time_feature_mode=time_feature_mode,
        model_mode=model_mode,
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=physical_scale_by_column,
        agent_index=agent_index,
        agent_profile=agent_profile,
        postprocess_mode=postprocess_mode,
        baseline_mode=baseline_mode,
        blend_weight=blend_weight,
        optimized_metric=optimized_metric,
        device=device,
    )

    prediction = forecaster.predict(
        history_seed,
        horizon=target_values.shape[0] + 1,
        signal_name=signal_name,
        history_timestamps=history_timestamps,
    )
    if target_values.ndim == 1 or target_values.shape[1] == 1:
        prediction = np.asarray(prediction, dtype=np.float32)[1 : target_values.shape[0] + 1]
        target = target_values.reshape(-1).astype(np.float32)
    else:
        prediction = np.asarray(prediction, dtype=np.float32)[:, 1 : target_values.shape[0] + 1]
        target = target_values.T.astype(np.float32)

    return _signal_forecast_evaluation_cls()(
        signal_name=signal_name,
        evaluation_mode="open_loop",
        timestamps=np.asarray(timestamps),
        target=target,
        prediction=np.asarray(prediction, dtype=np.float32),
        metrics=compute_forecast_metrics(target, prediction),
        source_columns=source.value_columns,
    )


def evaluate_signal_online_one_week(
    source: SignalCsvSource,
    model,
    *,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
    input_size: int = 1,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
    model_mode: str = "shared",
    physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    postprocess_mode: str = POSTPROCESS_MODE_NONE,
    baseline_mode: str = BASELINE_MODE_NONE,
    blend_weight: float | None = None,
    optimized_metric: str | None = None,
):
    """Run an online-aligned one-week evaluation."""
    test_frame, test_values, _ = _load_signal_matrix_from_source(source, signal_name, split="test")
    history_seed, target_values, history_timestamps, timestamps = _select_week_evaluation_slice(
        test_frame,
        test_values,
        signal_name=signal_name,
        seq_len=seq_len,
    )

    forecaster = _build_runtime_forecaster_from_model(
        model,
        signal_name=signal_name,
        scaler=scaler,
        seq_len=seq_len,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        input_size=input_size,
        time_feature_mode=time_feature_mode,
        model_mode=model_mode,
        physical_normalization_mode=physical_normalization_mode,
        physical_scale_by_column=physical_scale_by_column,
        agent_index=agent_index,
        agent_profile=agent_profile,
        postprocess_mode=postprocess_mode,
        baseline_mode=baseline_mode,
        blend_weight=blend_weight,
        optimized_metric=optimized_metric,
        device=device,
    )

    target_matrix = _reshape_signal_values(target_values)
    history_matrix = _reshape_signal_values(history_seed)
    one_step_predictions: list[np.ndarray | np.float32] = []

    for step_idx in range(target_matrix.shape[0]):
        if step_idx == 0:
            current_history = history_matrix
        else:
            current_history = np.concatenate([history_matrix, target_matrix[:step_idx]], axis=0)
        current_timestamps = np.asarray(history_timestamps)
        if step_idx > 0:
            current_timestamps = np.concatenate(
                [current_timestamps, np.asarray(timestamps[:step_idx])],
                axis=0,
            )
        rollout = forecaster.predict(
            _to_forecaster_history_input(current_history),
            horizon=max(int(pred_len) + 1, 2),
            signal_name=signal_name,
            history_timestamps=current_timestamps,
        )
        rollout_array = np.asarray(rollout, dtype=np.float32)
        if target_matrix.shape[1] == 1:
            one_step_predictions.append(np.float32(rollout_array[1]))
        else:
            one_step_predictions.append(rollout_array[:, 1].astype(np.float32))

    target = _reshape_evaluation_target(target_matrix)
    if target_matrix.shape[1] == 1:
        prediction = np.asarray(one_step_predictions, dtype=np.float32)
    else:
        prediction = np.stack(one_step_predictions, axis=0).T.astype(np.float32)

    return _signal_forecast_evaluation_cls()(
        signal_name=signal_name,
        evaluation_mode="online_aligned",
        timestamps=np.asarray(timestamps),
        target=target,
        prediction=prediction,
        metrics=compute_forecast_metrics(target, prediction),
        source_columns=source.value_columns,
    )


def evaluate_signal_one_week(
    source: SignalCsvSource,
    model,
    *,
    mode: str,
    signal_name: str,
    scaler: object | None,
    seq_len: int,
    pred_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: str | torch.device,
    input_size: int = 1,
    time_feature_mode: str = TIME_FEATURE_MODE_NONE,
    model_mode: str = "shared",
    physical_normalization_mode: str = PHYSICAL_NORMALIZATION_NONE,
    physical_scale_by_column: Sequence[float] | np.ndarray | float | None = None,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    postprocess_mode: str = POSTPROCESS_MODE_NONE,
    baseline_mode: str = BASELINE_MODE_NONE,
    blend_weight: float | None = None,
    optimized_metric: str | None = None,
):
    """Preserve the legacy one-week evaluation entrypoint."""
    if mode == "online_aligned":
        return evaluate_signal_online_one_week(
            source,
            model,
            signal_name=signal_name,
            scaler=scaler,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            input_size=input_size,
            time_feature_mode=time_feature_mode,
            model_mode=model_mode,
            physical_normalization_mode=physical_normalization_mode,
            physical_scale_by_column=physical_scale_by_column,
            agent_index=agent_index,
            agent_profile=agent_profile,
            postprocess_mode=postprocess_mode,
            baseline_mode=baseline_mode,
            blend_weight=blend_weight,
            optimized_metric=optimized_metric,
        )
    if mode == "open_loop":
        return evaluate_signal_open_loop_one_week(
            source,
            model,
            signal_name=signal_name,
            scaler=scaler,
            seq_len=seq_len,
            pred_len=pred_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            input_size=input_size,
            time_feature_mode=time_feature_mode,
            model_mode=model_mode,
            physical_normalization_mode=physical_normalization_mode,
            physical_scale_by_column=physical_scale_by_column,
            agent_index=agent_index,
            agent_profile=agent_profile,
            postprocess_mode=postprocess_mode,
            baseline_mode=baseline_mode,
            blend_weight=blend_weight,
            optimized_metric=optimized_metric,
        )
    raise ValueError("Unknown evaluation mode, expected 'online_aligned' or 'open_loop'.")


def _plot_evaluation_curve(axis, evaluation) -> None:
    """Draw one evaluation subplot."""
    if evaluation.target.ndim == 1:
        target_curve = evaluation.target
        prediction_curve = evaluation.prediction
        ylabel = evaluation.signal_name
    else:
        target_curve = evaluation.target.sum(axis=0)
        prediction_curve = evaluation.prediction.sum(axis=0)
        ylabel = f"{evaluation.signal_name} (sum)"

    mode_title = {
        "online_aligned": "Online aligned forecast",
        "open_loop": "Open-loop stress test",
    }.get(evaluation.evaluation_mode, str(evaluation.evaluation_mode))
    prediction_label = "Prediction" if evaluation.evaluation_mode == "online_aligned" else "Open-loop prediction"

    axis.plot(evaluation.timestamps, target_curve, label="Ground truth", linewidth=1.5)
    axis.plot(evaluation.timestamps, prediction_curve, label=prediction_label, linewidth=1.5)
    axis.set_title(f"{evaluation.signal_name} - {mode_title}")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")


def plot_weekly_forecasts(
    evaluations: list,
    *,
    save_path: str | Path,
) -> Path:
    """Save a single weekly comparison figure for multiple signals."""
    if not evaluations:
        raise ValueError("plot_weekly_forecasts requires at least one evaluation result.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(evaluations),
        1,
        figsize=(14, 4.2 * len(evaluations)),
        sharex=False,
    )
    axes = list(np.atleast_1d(axes))

    for axis, evaluation in zip(axes, evaluations):
        _plot_evaluation_curve(axis, evaluation)

    axes[-1].set_xlabel("timestamp")
    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    return save_path


def plot_signal_training_report(
    result: dict[str, object],
    *,
    figsize: tuple[float, float] = (18.0, 4.8),
):
    """Draw training losses plus online/open-loop evaluation curves."""
    rows = list(result.get("agent_results") or [result])
    subplot_count = 3 if any(item.get("open_loop_evaluation") is not None for item in rows) else 2
    figure, axes = plt.subplots(
        len(rows),
        subplot_count,
        figsize=(figsize[0], figsize[1] * len(rows)),
        squeeze=False,
    )

    for row_idx, item in enumerate(rows):
        history = item["training"]["history"]
        evaluation = item["evaluation"]
        open_loop_evaluation = item.get("open_loop_evaluation")
        label = str(item["signal_name"])
        if item.get("agent_profile") is not None:
            label = f"{label}[{item['agent_profile']}]"

        axes[row_idx, 0].plot(history["train_loss"], label="Train", linewidth=1.6)
        axes[row_idx, 0].plot(history["val_loss"], label="Validation", linewidth=1.6)
        axes[row_idx, 0].set_title(f"{label} - Loss")
        axes[row_idx, 0].set_xlabel("epoch")
        axes[row_idx, 0].set_ylabel("MSE")
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].legend(loc="upper right")

        _plot_evaluation_curve(axes[row_idx, 1], evaluation)
        axes[row_idx, 1].set_xlabel("timestamp")

        if open_loop_evaluation is not None:
            _plot_evaluation_curve(axes[row_idx, 2], open_loop_evaluation)
            axes[row_idx, 2].set_xlabel("timestamp")

    figure.tight_layout()
    return figure


__all__ = [
    "_plot_evaluation_curve",
    "_reshape_evaluation_target",
    "_select_week_evaluation_slice",
    "_to_forecaster_history_input",
    "compute_forecast_metrics",
    "evaluate_signal_one_week",
    "evaluate_signal_online_one_week",
    "evaluate_signal_open_loop_one_week",
    "plot_signal_training_report",
    "plot_weekly_forecasts",
]
