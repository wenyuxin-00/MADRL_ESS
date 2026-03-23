"""Reward decomposition plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(data).rolling(window=window, min_periods=1).mean().to_numpy()


def plot_reward_decomposition(history, episode_rewards, reward_fn, title, window=20):
    """Plot total reward, each component, and aggregate grid penalties when available."""
    if len(history) == 0:
        print("No history to plot.")
        return

    metas = list(reward_fn.component_meta)
    ep_total = np.array(episode_rewards, dtype=np.float32)
    ep_comps = {
        meta.key: np.array([np.sum(ep[f"{meta.key}_sum"]) for ep in history], dtype=np.float32)
        for meta in metas
    }

    # 自动检测全局安全惩罚分量：key 以 "r_safe_" 开头且 sign == -1
    grid_safety_keys = [
        meta.key for meta in metas
        if meta.key.startswith("r_safe_") and getattr(meta, "sign", 0) == -1
    ]
    aggregate_key = None
    if grid_safety_keys and all(k in ep_comps for k in grid_safety_keys):
        aggregate_key = "grid_safety_penalty"
        ep_comps[aggregate_key] = sum(ep_comps[k] for k in grid_safety_keys)

    n_plots = 1 + len(metas) + int(aggregate_key is not None)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10.5, 2.15 * n_plots), sharex=True)
    if n_plots == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=15)

    def _plot(ax, data, color, name):
        ma = _moving_average(np.asarray(data, dtype=np.float32), window=window)
        ax.plot(ma, color=color, linewidth=2, label=f"MA({window})")
        ax.set_title(name)
        ax.grid(True, linestyle=":")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="best")

    plot_index = 0
    _plot(axs[plot_index], ep_total, "red", "1) Episode Total Reward (sum over agents)")
    plot_index += 1

    if aggregate_key is not None:
        _plot(
            axs[plot_index],
            ep_comps[aggregate_key],
            "#111827",
            "2) Grid penalty aggregate (signed contribution)",
        )
        plot_index += 1

    for meta_idx, meta in enumerate(metas, start=plot_index + 1):
        _plot(axs[meta_idx - 1], ep_comps[meta.key], meta.color, f"{meta_idx}) {meta.label}")

    axs[-1].set_xlabel("Episode")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
