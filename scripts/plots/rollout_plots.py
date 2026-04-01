"""Plot helpers for rollout diagnostics used by the MADRL notebooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_voltage_and_net_load_dashboard(
    rollout: Any,
    *,
    title: str | None = None,
):
    """Plot all-bus voltage traces together with total net-load dynamics."""
    grid_df = rollout.grid_df.copy()
    step_df = rollout.step_df.copy()
    meta = dict(getattr(rollout, "meta", {}))

    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
    voltage_ax, net_load_ax = axes
    figure.suptitle(title or f"{meta.get('controller', 'Rollout')} Voltage and Net Load", fontsize=14)

    if not grid_df.empty:
        grid_sort_columns = [column for column in ("episode_idx", "step", "bus_id") if column in grid_df.columns]
        if grid_sort_columns:
            grid_df = grid_df.sort_values(grid_sort_columns).reset_index(drop=True)
        bus_groups = list(grid_df.groupby("bus_id", sort=True))
        agent_bus_ids = {int(bus_id) for bus_id in meta.get("agent_bus_ids", [])}
        background_color = "#94a3b8"
        highlight_palette = ["#0f766e", "#2563eb", "#dc2626", "#ea580c", "#7c3aed", "#ca8a04"]
        highlight_index = 0

        for bus_id, bus_df in bus_groups:
            bus_id = int(bus_id)
            timestamps = pd.to_datetime(bus_df["timestamp"])
            values = np.asarray(bus_df["vm_pu"], dtype=np.float32)
            if bus_id in agent_bus_ids:
                color = highlight_palette[highlight_index % len(highlight_palette)]
                highlight_index += 1
                voltage_ax.plot(
                    timestamps,
                    values,
                    color=color,
                    linewidth=1.7,
                    alpha=0.95,
                    label=f"Agent bus {bus_id}",
                )
            else:
                voltage_ax.plot(
                    timestamps,
                    values,
                    color=background_color,
                    linewidth=1.0,
                    alpha=0.45,
                )

    v_min_pu = meta.get("v_min_pu")
    v_max_pu = meta.get("v_max_pu")
    if v_min_pu is not None:
        voltage_ax.axhline(float(v_min_pu), color="#dc2626", linestyle="--", linewidth=1.1, label="v_min")
    if v_max_pu is not None:
        voltage_ax.axhline(float(v_max_pu), color="#ea580c", linestyle="--", linewidth=1.1, label="v_max")
    voltage_ax.set_ylabel("Voltage [p.u.]")
    voltage_ax.set_title("Bus Voltage Time Series")
    voltage_ax.grid(True, linestyle=":", alpha=0.45)
    voltage_ax.legend(loc="best")

    if not step_df.empty:
        step_sort_columns = [column for column in ("episode_idx", "step") if column in step_df.columns]
        if step_sort_columns:
            step_df = step_df.sort_values(step_sort_columns).reset_index(drop=True)
        timestamps = pd.to_datetime(step_df["timestamp"])
        net_load_ax.plot(
            timestamps,
            np.asarray(step_df.get("base_net_load_total", np.zeros(len(step_df))), dtype=np.float32),
            color="#111827",
            linewidth=1.5,
            label="Raw net load",
        )
        net_load_ax.plot(
            timestamps,
            np.asarray(step_df.get("base_net_load_effective_total", np.zeros(len(step_df))), dtype=np.float32),
            color="#0f766e",
            linewidth=1.5,
            label="Post-curtail net load",
        )
        net_load_ax.plot(
            timestamps,
            np.asarray(step_df.get("net_load_total", np.zeros(len(step_df))), dtype=np.float32),
            color="#2563eb",
            linewidth=1.7,
            label="Post-action net load",
        )

    trafo_limit_kw = meta.get("trafo_limit_kw")
    if trafo_limit_kw is not None and np.isfinite(float(trafo_limit_kw)):
        limit = float(trafo_limit_kw)
        net_load_ax.axhline(limit, color="#b91c1c", linestyle="--", linewidth=1.1, label="Trafo upper limit")
        net_load_ax.axhline(-limit, color="#b91c1c", linestyle="--", linewidth=1.1, label="Trafo lower limit")

    net_load_ax.set_ylabel("Power [kW]")
    net_load_ax.set_xlabel("Timestamp")
    net_load_ax.set_title("Total Net Load Time Series")
    net_load_ax.grid(True, linestyle=":", alpha=0.45)
    net_load_ax.legend(loc="best")

    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure._voltage_net_load_axes = axes
    return figure
