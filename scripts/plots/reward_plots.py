"""Reward decomposition plotting helpers."""

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


def _build_reward_summary_from_history(history, episode_rewards, reward_fn) -> dict[str, Any] | None:
    if history is None or len(history) == 0:
        print("No history to plot.")
        return None
    if episode_rewards is None or reward_fn is None:
        raise ValueError("episode_rewards and reward_fn are required when reward_summary is not provided.")

    metas = list(reward_fn.component_meta)
    components: dict[str, dict[str, Any]] = {}
    for meta in metas:
        components[str(meta.key)] = {
            "label": str(getattr(meta, "label", meta.key)),
            "color": str(getattr(meta, "color", "#111827")),
            "sign": int(getattr(meta, "sign", 0)),
            "values": [
                float(np.sum(np.asarray(ep[f"{meta.key}_sum"], dtype=np.float32)))
                for ep in history
            ],
        }

    aggregates: dict[str, list[float]] = {}
    grid_safety_keys = [
        str(meta.key)
        for meta in metas
        if str(meta.key).startswith("r_safe_") and int(getattr(meta, "sign", 0)) == -1
    ]
    if grid_safety_keys and all(key in components for key in grid_safety_keys):
        num_episodes = len(episode_rewards)
        aggregates["grid_safety_penalty"] = [
            float(sum(components[key]["values"][episode_idx] for key in grid_safety_keys))
            for episode_idx in range(num_episodes)
        ]

    return {
        "episodes": list(range(1, len(episode_rewards) + 1)),
        "episode_total_reward": [float(value) for value in episode_rewards],
        "components": components,
        "aggregates": aggregates,
    }


def _load_reward_summary(reward_summary) -> dict[str, Any]:
    if isinstance(reward_summary, (str, Path)):
        return json.loads(Path(reward_summary).read_text(encoding="utf-8"))
    if isinstance(reward_summary, dict):
        return reward_summary
    raise TypeError("reward_summary must be a mapping or a path to a JSON file.")


def plot_reward_decomposition(
    history=None,
    episode_rewards=None,
    reward_fn=None,
    title: str = "Training Reward Decomposition",
    window: int = 20,
    reward_summary=None,
):
    """Plot total reward and reward components over episodes."""
    resolved_summary = (
        _load_reward_summary(reward_summary)
        if reward_summary is not None
        else _build_reward_summary_from_history(history, episode_rewards, reward_fn)
    )
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
