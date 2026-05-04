from __future__ import annotations

from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_TITLE, _LABEL, _TICK, _LEGEND = 15, 12, 10, 10


def _style(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=_TITLE)
    axis.set_ylabel(ylabel, fontsize=_LABEL)
    axis.tick_params(axis="both", labelsize=_TICK)
    axis.grid(True, alpha=0.25)


def plot_test_truth_vs_prediction(predictions_df: pd.DataFrame):
    df = predictions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    profiles = list(df.loc[df["signal_name"] == "load", "series_name"].drop_duplicates())
    keys = [("load", profile) for profile in profiles] + [("heatpump", profile) for profile in profiles] + [("pv", "shared_pv"), ("wholesale_price", "wholesale_price")]
    titles = [f"{profile} total load" for profile in profiles] + [f"{profile} heatpump" for profile in profiles] + ["shared PV", "wholesale price"]
    fig, axes = plt.subplots(8, 1, figsize=(16.0, 24.0), sharex=True)
    for axis, title, (signal, series) in zip(np.atleast_1d(axes), titles, keys, strict=False):
        part = df.loc[(df["signal_name"] == signal) & (df["series_name"] == series)].sort_values(["episode", "step"])
        axis.plot(part["timestamp"], part["target"], color="#111827", linewidth=1.4, label="target")
        axis.plot(part["timestamp"], part["prediction"], color="#ea580c", linewidth=1.2, label="prediction")
        _style(axis, title, "value")
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    for axis in np.atleast_1d(axes):
        axis.xaxis.set_major_locator(locator); axis.xaxis.set_major_formatter(formatter)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=_LEGEND)
    np.atleast_1d(axes)[-1].set_xlabel("test timestamp", fontsize=_LABEL)
    fig.suptitle("Test Window: Ground Truth vs LSTM Prediction", fontsize=14, y=1.0)
    fig.tight_layout()
    return fig


def plot_predict_results(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    predictions_df = pd.read_csv(run_dir / "forecast" / "tables" / "test_predictions.csv")
    figures = {"test_truth_vs_prediction": plot_test_truth_vs_prediction(predictions_df)}
    out_dir = run_dir / "forecast" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(out_dir / f"{name}.png", dpi=150)
    return figures
