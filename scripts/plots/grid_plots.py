"""电网潮流结果可视化。

绘制电压分布、线路负载、功率注入等电网相关图表。

主要函数:
    plot_voltage_profile -- 绘制电压分布
    plot_line_loading    -- 绘制线路负载
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Voltage plot
# ---------------------------------------------------------------------------


def plot_node_voltages(
    grid_history: dict,
    v_min: float = 0.95,
    v_max: float = 1.05,
    title: str = "Node Voltage (pu)",
    agent_labels: list[str] | None = None,
) -> None:
    """Plot per-agent bus voltages over one episode.

    Parameters
    ----------
    grid_history:
        Dict produced by :func:`evaluation.grid_recorder.init_grid_record`.
    v_min, v_max:
        Voltage limits used to draw the constraint band.
    title:
        Chart title.
    agent_labels:
        Optional list of legend labels, one per agent.  Defaults to
        ``["Agent 0", "Agent 1", ...]``.
    """
    import matplotlib.pyplot as plt

    agent_vm_series: list[list[float]] = grid_history.get("agent_vm_pu", [])
    n_agents = len(agent_vm_series)
    if n_agents == 0:
        print("grid_history['agent_vm_pu'] is empty — nothing to plot.")
        return

    T = len(agent_vm_series[0])
    if T == 0:
        print("No timesteps recorded in grid_history.")
        return

    ts = np.arange(T) * 0.25  # 15-min intervals → hours

    if agent_labels is None:
        agent_labels = [f"Agent {i}" for i in range(n_agents)]

    fig, ax = plt.subplots(figsize=(12, 3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (series, label) in enumerate(zip(agent_vm_series, agent_labels)):
        vm = np.asarray(series)
        ax.plot(ts, vm, label=label, color=colors[i % len(colors)], linewidth=1.2)

    ax.axhline(v_min, color="red", linestyle="--", linewidth=0.9, label=f"v_min={v_min}")
    ax.axhline(v_max, color="red", linestyle="--", linewidth=0.9, label=f"v_max={v_max}")
    ax.fill_between(ts, v_min, v_max, alpha=0.06, color="green", label="Safe band")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncol=min(n_agents + 1, 4))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Combined battery + grid chart
# ---------------------------------------------------------------------------


def plot_battery_and_grid(
    history: dict,
    grid_history: dict,
    n_agents: int = 3,
    k: int = 1,
    title_prefix: str = "",
    v_min: float = 0.95,
    v_max: float = 1.05,
    agent_labels: list[str] | None = None,
) -> None:
    """Combined chart: voltage (top), then per-agent battery power + SoC.

    Parameters
    ----------
    history:
        Standard episode history from :func:`evaluation.episode_recorder.init_episode_record`.
        Expected keys: ``"soc"``, ``"e_bat_exec"``, ``"price"``.
    grid_history:
        Grid episode history from :func:`evaluation.grid_recorder.init_grid_record`.
    n_agents:
        Number of agents (= number of battery rows).
    k:
        Which of the last *k* episodes to draw.  Always 1 for this function
        since ``grid_history`` holds one episode.
    title_prefix:
        String prepended to each subplot title.
    v_min, v_max:
        Voltage constraint limits.
    agent_labels:
        Optional list of legend labels for agents.
    """
    import matplotlib.pyplot as plt

    if agent_labels is None:
        agent_labels = [f"Agent {i}" for i in range(n_agents)]

    agent_vm: list[list[float]] = grid_history.get("agent_vm_pu", [])
    has_voltages = len(agent_vm) > 0 and len(agent_vm[0]) > 0
    T = len(agent_vm[0]) if has_voltages else (
        len(history.get("price", [])) if history else 0
    )
    if T == 0:
        print("No timesteps to plot.")
        return

    ts = np.arange(T) * 0.25  # hours

    n_rows = 1 + n_agents   # voltage row + one row per agent
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.5 * n_rows), sharex=True)

    # ------------------------------------------------------------------
    # Row 0: voltage
    # ------------------------------------------------------------------
    ax_v = axes[0]
    if has_voltages:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for i, (series, label) in enumerate(zip(agent_vm[:n_agents], agent_labels)):
            ax_v.plot(ts, series, label=label, color=colors[i % len(colors)], linewidth=1.2)
        ax_v.axhline(v_min, color="red", linestyle="--", linewidth=0.8)
        ax_v.axhline(v_max, color="red", linestyle="--", linewidth=0.8)
        ax_v.fill_between(ts, v_min, v_max, alpha=0.06, color="green")
        ax_v.legend(fontsize=7, ncol=min(n_agents, 4))
    ax_v.set_ylabel("Voltage (pu)")
    ax_v.set_title(f"{title_prefix}Node Voltage" if title_prefix else "Node Voltage")
    ax_v.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Rows 1..n_agents: battery power (bar) + SoC (twin y)
    # ------------------------------------------------------------------
    e_bat_exec: list[list[float]] = history.get("e_bat_exec", [[] for _ in range(n_agents)])
    soc_series: list[list[float]] = history.get("soc", [[] for _ in range(n_agents)])
    price_series: list[float] = history.get("price", [])

    bar_colors_pos = "tab:red"
    bar_colors_neg = "tab:blue"

    for i in range(n_agents):
        ax = axes[1 + i]
        ax2 = ax.twinx()

        e_bat = np.asarray(e_bat_exec[i] if i < len(e_bat_exec) else [], dtype=np.float32)
        soc = np.asarray(soc_series[i][1:] if i < len(soc_series) else [], dtype=np.float32)

        if e_bat.size > 0:
            bar_c = np.where(e_bat >= 0, bar_colors_pos, bar_colors_neg)
            ax.bar(ts[: e_bat.size], e_bat, color=bar_c, width=0.22, alpha=0.8, label="P_batt (kW)")

        if soc.size > 0:
            ax2.plot(ts[: soc.size], soc, color="tab:green", linewidth=1.5, label="SoC")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("SoC")
            ax2.tick_params(axis="y", labelsize=8)

        ax.set_ylabel("P_batt (kW)")
        ax.set_title(f"{title_prefix}{agent_labels[i]}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

        # Combine legends.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Constraint summary bar chart
# ---------------------------------------------------------------------------


def plot_grid_constraint_summary(
    grid_histories: list[dict],
    title: str = "Constraint Violation Summary per Episode",
) -> None:
    """Bar chart showing voltage and line violation counts per episode.

    Parameters
    ----------
    grid_histories:
        List of grid_history dicts, one per evaluated episode.
    title:
        Chart title.
    """
    import matplotlib.pyplot as plt

    n_eps = len(grid_histories)
    v_counts = [sum(gh.get("n_v_violations", [])) for gh in grid_histories]
    l_counts = [sum(gh.get("n_l_violations", [])) for gh in grid_histories]

    x = np.arange(n_eps)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n_eps * 0.8), 3))
    ax.bar(x - width / 2, v_counts, width, label="Voltage Violation Steps", color="tab:red", alpha=0.8)
    ax.bar(x + width / 2, l_counts, width, label="Line Overload Steps", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ep {i}" for i in range(n_eps)])
    ax.set_ylabel("Violation Steps")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    plt.show()
