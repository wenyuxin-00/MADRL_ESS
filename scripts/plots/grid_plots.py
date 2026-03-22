"""Plotting utilities for grid experiments."""

from __future__ import annotations

import numpy as np


DEFAULT_AGENT_COLORS = ["#0f766e", "#b45309", "#7c3aed", "#be123c", "#1d4ed8"]
BASELINE_COLOR = "#475569"
SAFE_BAND_COLOR = "#d1fae5"
CHARGE_COLOR = "#dc2626"
DISCHARGE_COLOR = "#2563eb"
SOC_COLOR = "#111827"


def _time_axis(time_steps: int, dt_hours: float = 0.25) -> np.ndarray:
    return np.arange(int(time_steps), dtype=np.float32) * float(dt_hours)


def plot_node_voltages(
    grid_history: dict,
    v_min: float = 0.95,
    v_max: float = 1.05,
    title: str = "Node Voltage (pu)",
    agent_labels: list[str] | None = None,
) -> None:
    """Plot per-agent bus voltages over one episode."""
    import matplotlib.pyplot as plt

    agent_vm_series = grid_history.get("agent_vm_pu", [])
    n_agents = len(agent_vm_series)
    if n_agents == 0:
        print("grid_history['agent_vm_pu'] is empty; nothing to plot.")
        return

    time_steps = len(agent_vm_series[0])
    if time_steps == 0:
        print("No timesteps recorded in grid_history.")
        return

    ts = _time_axis(time_steps)
    if agent_labels is None:
        agent_labels = [f"Agent {i}" for i in range(n_agents)]

    fig, ax = plt.subplots(figsize=(12, 3.4))
    for i, (series, label) in enumerate(zip(agent_vm_series, agent_labels)):
        ax.plot(ts, np.asarray(series), label=label, color=DEFAULT_AGENT_COLORS[i % len(DEFAULT_AGENT_COLORS)], linewidth=1.4)

    ax.axhline(v_min, color="#b91c1c", linestyle="--", linewidth=0.9, label=f"Lower limit = {v_min:.2f} pu")
    ax.axhline(v_max, color="#b91c1c", linestyle="--", linewidth=0.9, label=f"Upper limit = {v_max:.2f} pu")
    ax.fill_between(ts, v_min, v_max, alpha=0.24, color=SAFE_BAND_COLOR, label="Voltage band")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncol=min(n_agents + 2, 4))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()


def plot_node_voltage_comparison(
    baseline_grid_history: dict,
    controlled_grid_history: dict,
    v_min: float = 0.95,
    v_max: float = 1.05,
    title: str = "Node Voltage Comparison",
    agent_labels: list[str] | None = None,
) -> None:
    """Overlay baseline and controlled node voltages for each tracked bus."""
    import matplotlib.pyplot as plt

    baseline_series = baseline_grid_history.get("agent_vm_pu", [])
    controlled_series = controlled_grid_history.get("agent_vm_pu", [])
    n_agents = min(len(baseline_series), len(controlled_series))
    if n_agents == 0:
        print("No voltage traces available for comparison.")
        return

    time_steps = min(len(baseline_series[0]), len(controlled_series[0]))
    if time_steps == 0:
        print("No timesteps recorded in voltage histories.")
        return

    ts = _time_axis(time_steps)
    if agent_labels is None:
        agent_labels = [f"Agent {i}" for i in range(n_agents)]

    fig, axes = plt.subplots(n_agents, 1, figsize=(13, 2.6 * n_agents), sharex=True)
    if n_agents == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        base = np.asarray(baseline_series[i][:time_steps], dtype=np.float32)
        ctrl = np.asarray(controlled_series[i][:time_steps], dtype=np.float32)
        color = DEFAULT_AGENT_COLORS[i % len(DEFAULT_AGENT_COLORS)]
        ax.fill_between(ts, v_min, v_max, alpha=0.22, color=SAFE_BAND_COLOR, zorder=0)
        ax.plot(ts, base, color=BASELINE_COLOR, linestyle=(0, (4, 2)), linewidth=1.3, label="Baseline")
        ax.plot(ts, ctrl, color=color, linewidth=1.8, label="Controlled")
        ax.axhline(v_min, color="#b91c1c", linestyle="--", linewidth=0.8)
        ax.axhline(v_max, color="#b91c1c", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{agent_labels[i]}\nVoltage")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout()
    plt.show()


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
    """Combined chart of voltages, battery power, and SoC."""
    import matplotlib.pyplot as plt

    del k
    if agent_labels is None:
        agent_labels = [f"Agent {i}" for i in range(n_agents)]

    agent_vm = grid_history.get("agent_vm_pu", [])
    has_voltages = len(agent_vm) > 0 and len(agent_vm[0]) > 0
    time_steps = len(agent_vm[0]) if has_voltages else len(history.get("price", []))
    if time_steps == 0:
        print("No timesteps to plot.")
        return

    ts = _time_axis(time_steps)
    n_rows = 1 + n_agents
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.6 * n_rows), sharex=True)

    ax_v = axes[0]
    if has_voltages:
        ax_v.fill_between(ts, v_min, v_max, alpha=0.22, color=SAFE_BAND_COLOR, zorder=0)
        for i, (series, label) in enumerate(zip(agent_vm[:n_agents], agent_labels)):
            ax_v.plot(ts, series, label=label, color=DEFAULT_AGENT_COLORS[i % len(DEFAULT_AGENT_COLORS)], linewidth=1.4)
        ax_v.axhline(v_min, color="#b91c1c", linestyle="--", linewidth=0.8)
        ax_v.axhline(v_max, color="#b91c1c", linestyle="--", linewidth=0.8)
        ax_v.legend(fontsize=7, ncol=min(n_agents, 4), loc="lower right")
    ax_v.set_ylabel("Voltage (pu)")
    ax_v.set_title(f"{title_prefix}Node Voltage" if title_prefix else "Node Voltage")
    ax_v.grid(True, alpha=0.22)

    e_bat_exec = history.get("e_bat_exec", [[] for _ in range(n_agents)])
    soc_series = history.get("soc", [[] for _ in range(n_agents)])

    for i in range(n_agents):
        ax = axes[1 + i]
        ax2 = ax.twinx()

        e_bat = np.asarray(e_bat_exec[i] if i < len(e_bat_exec) else [], dtype=np.float32)
        soc = np.asarray(soc_series[i][1:] if i < len(soc_series) else [], dtype=np.float32)

        if e_bat.size > 0:
            bar_colors = np.where(e_bat >= 0, CHARGE_COLOR, DISCHARGE_COLOR)
            ax.bar(ts[: e_bat.size], e_bat, color=bar_colors, width=0.18, alpha=0.82, label="Battery power (kW)")

        if soc.size > 0:
            ax2.plot(ts[: soc.size], soc, color=SOC_COLOR, linewidth=1.5, label="SoC")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("SoC")
            ax2.tick_params(axis="y", labelsize=8)

        ax.set_ylabel("Battery\npower (kW)")
        ax.set_title(f"{title_prefix}{agent_labels[i]}")
        ax.grid(True, alpha=0.22)
        ax.axhline(0, color="#0f172a", linewidth=0.7)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout()
    plt.show()


def plot_publication_day_summary(
    history: dict,
    baseline_grid_history: dict,
    controlled_grid_history: dict,
    *,
    title: str = "Typical-Day Grid Response",
    day_caption: str | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
    agent_labels: list[str] | None = None,
) -> None:
    """Create a compact report-style figure for one day.

    The top panel overlays baseline and controlled bus voltages. Each lower panel
    shows battery charging/discharging bars together with SoC for one agent.
    """
    import matplotlib.pyplot as plt

    controlled_series = controlled_grid_history.get("agent_vm_pu", [])
    baseline_series = baseline_grid_history.get("agent_vm_pu", [])
    n_agents = min(
        len(controlled_series),
        len(baseline_series),
        len(history.get("e_bat_exec", [])),
        len(history.get("soc", [])),
    )
    if n_agents == 0:
        print("No typical-day traces available for the publication summary plot.")
        return

    time_steps = min(len(controlled_series[0]), len(baseline_series[0]))
    if time_steps == 0:
        print("Typical-day traces are empty.")
        return

    if agent_labels is None:
        agent_labels = [f"Agent {i}" for i in range(n_agents)]

    ts = _time_axis(time_steps)
    fig, axes = plt.subplots(
        1 + n_agents,
        1,
        figsize=(13.5, 2.65 * (1 + n_agents)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35] + [1.0] * n_agents},
    )

    # Voltage panel.
    ax_v = axes[0]
    ax_v.fill_between(ts, v_min, v_max, color=SAFE_BAND_COLOR, alpha=0.28, zorder=0)
    for idx in range(n_agents):
        color = DEFAULT_AGENT_COLORS[idx % len(DEFAULT_AGENT_COLORS)]
        base = np.asarray(baseline_series[idx][:time_steps], dtype=np.float32)
        ctrl = np.asarray(controlled_series[idx][:time_steps], dtype=np.float32)
        ax_v.plot(ts, base, color=BASELINE_COLOR, linestyle=(0, (4, 2)), linewidth=1.2)
        ax_v.plot(ts, ctrl, color=color, linewidth=1.9, label=agent_labels[idx])
    ax_v.axhline(v_min, color="#b91c1c", linestyle="--", linewidth=0.9)
    ax_v.axhline(v_max, color="#b91c1c", linestyle="--", linewidth=0.9)
    ax_v.set_ylabel("Voltage (pu)")
    ax_v.set_title(title)
    if day_caption:
        ax_v.text(
            0.01,
            1.02,
            day_caption,
            transform=ax_v.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#334155",
        )
    voltage_handles = [
        plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle=(0, (4, 2)), linewidth=1.4, label="Baseline"),
        plt.Line2D([0], [0], color="#0f172a", linewidth=1.6, label="Controlled"),
    ]
    agent_handles = [
        plt.Line2D([0], [0], color=DEFAULT_AGENT_COLORS[i % len(DEFAULT_AGENT_COLORS)], linewidth=1.8, label=agent_labels[i])
        for i in range(n_agents)
    ]
    ax_v.legend(handles=voltage_handles + agent_handles, loc="lower right", fontsize=8, ncol=min(2 + n_agents, 5))
    ax_v.grid(True, alpha=0.22)

    # Battery and SoC panels.
    e_bat_exec = history.get("e_bat_exec", [])
    soc_series = history.get("soc", [])
    for idx in range(n_agents):
        ax = axes[1 + idx]
        ax_soc = ax.twinx()

        power = np.asarray(e_bat_exec[idx][:time_steps], dtype=np.float32)
        soc = np.asarray(soc_series[idx][1 : time_steps + 1], dtype=np.float32)
        charge = np.where(power > 0.0, power, 0.0)
        discharge = np.where(power < 0.0, power, 0.0)

        ax.bar(ts, charge, width=0.18, color=CHARGE_COLOR, alpha=0.78, label="Charge")
        ax.bar(ts, discharge, width=0.18, color=DISCHARGE_COLOR, alpha=0.78, label="Discharge")
        ax.axhline(0.0, color="#0f172a", linewidth=0.7)
        ax.set_ylabel(f"{agent_labels[idx]}\nP (kW)")
        ax.grid(True, alpha=0.18)

        ax_soc.plot(ts[: soc.size], soc, color=SOC_COLOR, linewidth=1.5, label="SoC")
        ax_soc.set_ylim(0.0, 1.02)
        ax_soc.set_ylabel("SoC")
        ax_soc.tick_params(axis="y", labelsize=8)

        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = ax_soc.get_legend_handles_labels()
        ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout()
    plt.show()


def plot_grid_constraint_summary(
    grid_histories: list[dict],
    title: str = "Constraint Violation Summary per Episode",
) -> None:
    """Bar chart showing voltage, line, and transformer violation counts."""
    import matplotlib.pyplot as plt

    n_eps = len(grid_histories)
    v_counts = [sum(gh.get("n_v_violations", [])) for gh in grid_histories]
    l_counts = [sum(gh.get("n_line_violations", gh.get("n_l_violations", []))) for gh in grid_histories]
    t_counts = [sum(gh.get("n_t_violations", [])) for gh in grid_histories]

    x = np.arange(n_eps)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n_eps * 0.9), 3.6))
    ax.bar(x - width, v_counts, width, label="Voltage", color="#b91c1c", alpha=0.82)
    ax.bar(x, l_counts, width, label="Line", color="#d97706", alpha=0.82)
    ax.bar(x + width, t_counts, width, label="Transformer", color="#7c2d12", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ep {i}" for i in range(n_eps)])
    ax.set_ylabel("Violation steps")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    plt.show()
