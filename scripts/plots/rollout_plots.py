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
    """Plot price, voltage, net load, and per-agent storage state for a rollout."""
    grid_df = rollout.grid_df.copy()
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    meta = dict(getattr(rollout, "meta", {}))
    if step_df.empty:
        raise ValueError("Rollout is empty; nothing to plot.")

    step_sort_columns = [column for column in ("episode_idx", "step") if column in step_df.columns]
    if step_sort_columns:
        step_df = step_df.sort_values(step_sort_columns).reset_index(drop=True)

    agent_profiles: list[str] = []
    if not agent_df.empty:
        agent_sort_columns = [column for column in ("episode_idx", "step", "agent_id") if column in agent_df.columns]
        if agent_sort_columns:
            agent_df = agent_df.sort_values(agent_sort_columns).reset_index(drop=True)
        agent_profiles = list(meta.get("agent_profiles") or [])
        if not agent_profiles:
            agent_profiles = [
                str(profile)
                for profile in agent_df.get("agent_profile", pd.Series(dtype=object)).dropna().drop_duplicates().tolist()
            ]

    axis_count = 3 + len(agent_profiles)
    figure, axes = plt.subplots(
        axis_count,
        1,
        figsize=(14.0, max(8.5, 2.8 * axis_count)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    price_ax, voltage_ax, net_load_ax = axes[:3]
    figure.suptitle(
        title or f"{meta.get('controller', 'Rollout')} Price, Voltage, Net Load, and Storage State",
        fontsize=14,
    )

    timestamps = pd.to_datetime(step_df["timestamp"])
    price_ax.plot(
        timestamps,
        np.asarray(step_df.get("price", np.zeros(len(step_df))), dtype=np.float32),
        color="#111827",
        linewidth=1.6,
        label="Actual",
    )
    if "price_pred" in step_df.columns:
        price_ax.plot(
            timestamps,
            np.asarray(step_df["price_pred"], dtype=np.float32),
            color="#dc2626",
            linewidth=1.4,
            linestyle="--",
            label="Forecast",
        )
    price_ax.set_ylabel("Price")
    price_ax.set_title("Electricity Price")
    price_ax.grid(True, linestyle=":", alpha=0.45)
    price_ax.legend(loc="best")

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

    for axis, profile in zip(axes[3:], agent_profiles, strict=False):
        agent_frame = agent_df.loc[agent_df["agent_profile"] == profile]
        if agent_frame.empty:
            axis.set_visible(False)
            continue
        agent_timestamps = pd.to_datetime(agent_frame["timestamp"])
        battery_power = agent_frame["e_bat"].to_numpy(dtype=np.float32)
        has_bat_req = "e_bat_req" in agent_frame.columns
        battery_power_req = agent_frame["e_bat_req"].to_numpy(dtype=np.float32) if has_bat_req else battery_power

        if has_bat_req:
            req_charge = np.clip(battery_power_req, 0.0, None)
            req_discharge = np.clip(battery_power_req, None, 0.0)
            axis.bar(agent_timestamps, req_charge, width=0.008, color="#dc2626", alpha=0.25, label="Charge (req)")
            axis.bar(agent_timestamps, req_discharge, width=0.008, color="#2563eb", alpha=0.25, label="Discharge (req)")

        charge = np.clip(battery_power, 0.0, None)
        discharge = np.clip(battery_power, None, 0.0)
        axis.bar(agent_timestamps, charge, width=0.008, color="#dc2626", alpha=0.7, label="Charge (exec)")
        axis.bar(agent_timestamps, discharge, width=0.008, color="#2563eb", alpha=0.7, label="Discharge (exec)")
        axis.set_ylabel(f"{profile}\nP_bat")
        axis.grid(True, linestyle=":", alpha=0.45)

        soc_axis = axis.twinx()
        soc_axis.plot(
            agent_timestamps,
            agent_frame["soc"].to_numpy(dtype=np.float32),
            color="#111827",
            linewidth=1.2,
            label="SoC",
        )
        soc_axis.set_ylabel("SoC")
        soc_axis.set_ylim(0.0, 1.0)

        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = soc_axis.get_legend_handles_labels()
        axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    axes[-1].set_xlabel("Timestamp")
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure._voltage_net_load_axes = list(axes)
    return figure
