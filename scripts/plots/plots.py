"""Generic plotting helpers for training and evaluation notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_last_k_episodes_price_action_soc(history, k=2, n_agents=3, title_prefix="Train"):
    """Plot price, battery power, and SoC for the last ``k`` episodes."""
    if len(history) == 0:
        print("No history to plot.")
        return

    k = min(k, len(history))
    start = len(history) - k

    for ep_idx in range(start, len(history)):
        ep = history[ep_idx]
        price = np.asarray(ep["price"], dtype=np.float32)
        horizon = len(price)
        time_steps = np.arange(horizon)

        fig, axes = plt.subplots(1 + n_agents, 1, figsize=(11, 2.4 * (1 + n_agents)), sharex=True)
        fig.suptitle(
            f"{title_prefix} Episode {ep_idx + 1}: Price + Battery Power & SOC",
            fontsize=14,
            y=0.995,
        )

        price_axis = axes[0]
        price_axis.plot(time_steps, price, color="m", linewidth=2, label="Price")
        price_axis.set_title("Price")
        price_axis.set_ylabel("Price")
        price_axis.grid(True, linestyle=":")
        price_axis.legend(loc="best")

        for agent_idx in range(n_agents):
            axis = axes[agent_idx + 1]
            soc = np.asarray(ep["soc"][agent_idx], dtype=np.float32)
            requested = np.asarray(ep["e_bat_req"][agent_idx], dtype=np.float32)
            executed = np.asarray(ep["e_bat_exec"][agent_idx], dtype=np.float32)

            axis.bar(time_steps, np.maximum(0, executed), color="red", width=0.85, label="Exec Charge (+)")
            axis.bar(time_steps, np.minimum(0, executed), color="green", width=0.85, label="Exec Discharge (-)")
            axis.bar(
                time_steps,
                np.maximum(0, requested),
                color="red",
                width=0.85,
                alpha=0.25,
                label="Req Charge (+)",
            )
            axis.bar(
                time_steps,
                np.minimum(0, requested),
                color="green",
                width=0.85,
                alpha=0.25,
                label="Req Discharge (-)",
            )
            axis.set_title(f"Agent {agent_idx + 1}: e_bat (bar) & SOC (line)")
            axis.set_ylabel("e_bat")
            axis.grid(True, axis="y", linestyle=":")
            axis.legend(loc="upper left")

            soc_axis = axis.twinx()
            soc_plot = soc[1:] if soc.shape[0] == horizon + 1 else soc[:horizon]
            soc_axis.plot(time_steps, soc_plot, color="blue", marker=".", markersize=3, linewidth=1.5, label="SOC")
            soc_axis.set_ylabel("SOC", color="blue")
            soc_axis.tick_params(axis="y", labelcolor="blue")
            soc_axis.set_ylim(0.0, 1.05)
            soc_axis.legend(loc="upper right")

        axes[-1].set_xlabel("Time Step")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
