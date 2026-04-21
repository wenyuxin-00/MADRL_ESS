from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
_AGGREGATE_STYLES = {
    "grid_safety_penalty": ("#111827", "Grid penalty aggregate (signed contribution)"),
}
def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(data).rolling(window=window, min_periods=1).mean().to_numpy()

def _load_reward_summary(reward_summary) -> dict[str, Any]:
    if isinstance(reward_summary, (str, Path)):
        return json.loads(Path(reward_summary).read_text(encoding="utf-8"))
    if isinstance(reward_summary, dict):
        return reward_summary
    raise TypeError("reward_summary must be a mapping or a path to a JSON file.")

def plot_reward_decomposition(
    *,
    title: str = "Training Reward Decomposition",
    window: int = 20,
    reward_summary,
):
    resolved_summary = _load_reward_summary(reward_summary)
    if not resolved_summary:
        return None

    ep_total = np.asarray(resolved_summary.get("episode_total_reward", []), dtype=np.float32)
    components = dict(resolved_summary.get("components", {}))
    aggregates = dict(resolved_summary.get("aggregates", {}))
    if ep_total.size == 0 and not components:
        print("No history to plot.")
        return None

    episodes = np.asarray(resolved_summary.get("episodes", []), dtype=np.int32)
    if episodes.size != ep_total.size:
        episodes = np.arange(1, ep_total.size + 1, dtype=np.int32)

    n_plots = 1 + len(aggregates) + len(components)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10.5, 2.35 * n_plots), sharex=True)
    axs = np.atleast_1d(axs)
    fig.suptitle(title, fontsize=15)
    def _plot(ax, x_values, data, color, name):
        y_values = np.asarray(data, dtype=np.float32)
        ma = _moving_average(y_values, window=window)
        ax.plot(x_values, y_values, color=color, linewidth=1.1, alpha=0.28, label="Episode")
        ax.plot(x_values, ma, color=color, linewidth=2.0, label=f"MA({window})")
        ax.set_title(name)
        ax.grid(True, linestyle=":")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="best")

    plot_index = 0
    _plot(axs[plot_index], episodes, ep_total, "red", "1) Episode Total Reward (sum over agents)")
    plot_index += 1
    for aggregate_offset, (aggregate_key, values) in enumerate(aggregates.items(), start=plot_index + 1):
        color, label = _AGGREGATE_STYLES.get(
            aggregate_key,
            ("#111827", aggregate_key.replace("_", " ").title()),
        )
        _plot(axs[aggregate_offset - 1], episodes, values, color, f"{aggregate_offset}) {label}")
    plot_index += len(aggregates)
    for component_offset, (component_key, payload) in enumerate(components.items(), start=plot_index + 1):
        label = str(payload.get("label", component_key))
        color = str(payload.get("color", "#2563eb"))
        values = payload.get("values", [])
        _plot(axs[component_offset - 1], episodes, values, color, f"{component_offset}) {label}")

    axs[-1].set_xlabel("Episode")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
