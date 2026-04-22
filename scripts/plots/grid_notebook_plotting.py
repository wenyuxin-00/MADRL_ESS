from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN, IMPORT_PRICE_PRED_COLUMN

if TYPE_CHECKING:
    from scripts.utils.grid_notebook_workflow import RolloutResult

NORMAL_PREDICTION_MODE = "normal"
_TITLE, _LABEL, _TICK, _LEGEND = (20, 18, 16, 16)
_AGENT_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]
_MUTED = "#cbd5e1"
_WIDTH = 0.008
_POS_SPECS = [("load_total", "Load", "#111827"), ("battery_charge_total", "Charge", "#dc2626"), ("grid_export_total", "Grid export", "#f59e0b"), ("pv_curtail_total", "Curtailment loss", "#fca5a5")]
_NEG_SPECS = [("pv_raw_total", "PV raw", "#16a34a"), ("grid_import_total", "Grid import", "#2563eb"), ("battery_discharge_total", "Discharge", "#7c3aed")]


def _controller(rollout: RolloutResult) -> str:
    return str(rollout.meta.get("controller", "unknown"))


def _compare_style(axis, *, title: str, ylabel: str, xlabel: str | None = None) -> None:
    axis.set_title(title, fontsize=_TITLE)
    axis.set_ylabel(ylabel, fontsize=_LABEL)
    axis.tick_params(axis="both", labelsize=_TICK)
    axis.grid(True, alpha=0.25)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=_LABEL)


def _panel_axes(count: int, *, figsize, base_height: float, sharex: bool = True, sharey: bool = True):
    fig, axes = plt.subplots(count, 1, figsize=figsize or (20.0, max(base_height * count, base_height + 1.6)), sharex=sharex, sharey=sharey)
    return fig, np.atleast_1d(axes)


def _require(df: pd.DataFrame, columns: set[str], label: str, message: str) -> None:
    if df.empty or not columns.issubset(df.columns):
        raise ValueError(message.format(label=label, missing=sorted(columns.difference(df.columns))))


def _with_pv_raw_total(step_df: pd.DataFrame) -> pd.DataFrame:
    if "pv_raw_total" not in step_df.columns and {"pv_effective_total", "pv_curtail_total"}.issubset(step_df.columns):
        step_df["pv_raw_total"] = step_df["pv_effective_total"].to_numpy(dtype=np.float32) + step_df["pv_curtail_total"].to_numpy(dtype=np.float32)
    return step_df


def _plot_agent_forecasts(axes, step_df: pd.DataFrame, agent_df: pd.DataFrame, profiles, *, title: str) -> None:
    axes[0].plot(step_df["timestamp"], step_df[IMPORT_PRICE_COLUMN], color="#111827", linewidth=1.6, label="Actual")
    axes[0].plot(step_df["timestamp"], step_df[IMPORT_PRICE_PRED_COLUMN], color="#dc2626", linewidth=1.4, linestyle="--", label="Forecast")
    axes[0].set_title(title)
    axes[0].set_ylabel("Import Price")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    by_profile = {profile: agent_df.loc[agent_df["agent_profile"] == profile] for profile in profiles}
    for axis, signal in zip(axes[1:], ("pv", "load"), strict=False):
        for agent_idx, profile in enumerate(profiles):
            frame = by_profile[profile]
            color = _AGENT_COLORS[agent_idx % len(_AGENT_COLORS)]
            axis.plot(frame["timestamp"], frame[signal], color=color, linewidth=1.4, label=f"{profile} actual")
            axis.plot(frame["timestamp"], frame[f"{signal}_pred"], color=color, linewidth=1.2, linestyle="--", label=f"{profile} forecast")
        axis.set_ylabel(signal.upper())
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=2)


def _plot_battery_rows(axes, agent_df: pd.DataFrame, profiles) -> None:
    for axis, profile in zip(axes, profiles, strict=False):
        frame = agent_df.loc[agent_df["agent_profile"] == profile]
        power = frame["e_bat"].to_numpy(dtype=np.float32)
        axis.bar(frame["timestamp"], np.maximum(power, 0.0), width=_WIDTH, color="#dc2626", alpha=0.7, label="Charge")
        axis.bar(frame["timestamp"], np.minimum(power, 0.0), width=_WIDTH, color="#2563eb", alpha=0.7, label="Discharge")
        axis.set_ylabel(f"{profile}\nP_bat")
        axis.grid(True, alpha=0.25)
        soc_axis = axis.twinx()
        soc_axis.plot(frame["timestamp"], frame["soc"], color="#111827", linewidth=1.2, label="SoC")
        soc_axis.set_ylabel("SoC")
        soc_axis.set_ylim(0.0, 1.0)
        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = soc_axis.get_legend_handles_labels()
        axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")


def _plot_voltage(axis, rollout: RolloutResult, *, legend: bool) -> None:
    grid_df = rollout.grid_df
    if grid_df.empty:
        raise ValueError(f"Rollout '{_controller(rollout)}' has no grid_df.")
    for _, frame in grid_df.loc[~grid_df["is_agent_bus"]].groupby("bus_id"):
        axis.plot(frame["timestamp"], frame["vm_pu"], color=_MUTED, linewidth=0.9, alpha=0.35, zorder=1)
    for idx, bus_id in enumerate(rollout.meta["agent_bus_ids"]):
        frame = grid_df.loc[grid_df["bus_id"] == int(bus_id)]
        if not frame.empty:
            axis.plot(frame["timestamp"], frame["vm_pu"], color=_AGENT_COLORS[idx % len(_AGENT_COLORS)], linewidth=2.1, alpha=0.95, label=f"Agent bus {bus_id}" if legend else None, zorder=3)
    for key, color, label in [("v_min_pu", "#dc2626", "V min"), ("v_max_pu", "#ea580c", "V max")]:
        axis.axhline(float(rollout.meta[key]), color=color, linestyle="--", linewidth=1.1, label=label if legend else None)


def plot_rollout_dashboard(rollout: RolloutResult, *, figsize: tuple[float, float] | None = None):
    if rollout.step_df.empty or rollout.agent_df.empty:
        raise ValueError("Rollout is empty; nothing to plot.")
    profiles = list(rollout.meta["agent_profiles"])
    fig, axes = plt.subplots(6 + len(profiles), 1, figsize=figsize or (18.0, 2.8 * (6 + len(profiles))), sharex=True)
    axes = np.atleast_1d(axes)
    step_df = _with_pv_raw_total(rollout.step_df.copy())
    _plot_agent_forecasts(axes[:3], step_df, rollout.agent_df, profiles, title=f"Test Rollout Dashboard - {_controller(rollout)}")
    _plot_voltage(axes[3], rollout, legend=True)
    axes[3].set_ylabel("Voltage [p.u.]")
    axes[3].legend(loc="upper right", ncol=2)
    net_specs = [("base_net_load_total", "#111827", "-", "Raw net load"), ("base_net_load_effective_total", "#16a34a", "-.", "Post-curtail net load"), ("net_load_total", "#2563eb", "--", "Post-action net load")]
    for column, color, linestyle, label in net_specs:
        if column in step_df.columns:
            axes[4].plot(step_df["timestamp"], step_df[column], color=color, linewidth=1.6 if column == "base_net_load_total" else 1.5, linestyle=linestyle, label=label)
    limit = rollout.meta.get("trafo_limit_kw")
    if limit is not None and np.isfinite(float(limit)) and float(limit) > 0.0:
        for sign in (1.0, -1.0):
            axes[4].axhline(sign * float(limit), color="#dc2626", linestyle=":", linewidth=1.2, label=f"Approx trafo {('+' if sign > 0 else '-')}limit ({float(limit):.1f} kW)")
    axes[4].set_ylabel("Net load")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right")
    if "pv_effective_total" in step_df.columns:
        axes[5].plot(step_df["timestamp"], step_df["pv_raw_total"], color="#ea580c", linewidth=1.5, label="Raw PV")
        axes[5].plot(step_df["timestamp"], step_df["pv_effective_total"], color="#16a34a", linewidth=1.5, linestyle="--", label="Effective PV")
        axes[5].bar(step_df["timestamp"], step_df["pv_curtail_total"], width=_WIDTH, color="#dc2626", alpha=0.35, label="Curtailment")
        axes[5].legend(loc="upper right")
    axes[5].set_ylabel("PV")
    axes[5].grid(True, alpha=0.25)
    _plot_battery_rows(axes[6:], rollout.agent_df, profiles)
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig._dashboard_main_axes = list(axes)
    return fig


def plot_voltage_profile_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None = None):
    if not rollouts:
        raise ValueError("At least one rollout is required.")
    fig, axes = _panel_axes(len(rollouts), figsize=figsize, base_height=4.0)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        _plot_voltage(axis, rollout, legend=idx == 0)
        _compare_style(axis, title=_controller(rollout), ylabel="V [p.u.]")
        if idx == 0:
            axis.legend(loc="upper right", ncol=2, fontsize=_LEGEND)
    axes[-1].set_xlabel("Timestamp", fontsize=_LABEL)
    fig.tight_layout()
    return fig


def plot_price_prediction_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] = (20.0, 5.0)):
    if not rollouts:
        raise ValueError("At least one rollout is required.")
    ref = next((rollout for rollout in rollouts if str(rollout.meta.get("prediction_mode", "")) == NORMAL_PREDICTION_MODE), rollouts[0])
    _require(ref.step_df, {"timestamp", IMPORT_PRICE_COLUMN}, _controller(ref), "Rollout '{label}' is missing actual price columns: {missing}")
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    axis.plot(ref.step_df["timestamp"], ref.step_df[IMPORT_PRICE_COLUMN], color="#111827", linewidth=1.8, label="Actual import price")
    predicted_rollouts = [item for item in rollouts if str(item.meta.get("prediction_mode", "")) in ("", NORMAL_PREDICTION_MODE)] or [ref]
    for idx, rollout in enumerate(predicted_rollouts):
        step_df = rollout.step_df
        _require(step_df, {"timestamp", IMPORT_PRICE_PRED_COLUMN}, _controller(rollout), "Rollout '{label}' is missing predicted price columns: {missing}")
        label = "Predicted import price" if len(predicted_rollouts) == 1 else f"Predicted import price ({_controller(rollout)})"
        axis.plot(pd.to_datetime(step_df["timestamp"]).to_numpy(), step_df[IMPORT_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64), color=_AGENT_COLORS[(idx + 1) % len(_AGENT_COLORS)], linewidth=1.8, linestyle="--", label=label)
    _compare_style(axis, title="Import Price And Derived Forecast", ylabel="EUR/kWh", xlabel="Timestamp")
    axis.legend(loc="upper right", ncol=2, fontsize=_LEGEND)
    fig.tight_layout()
    return fig


def plot_net_load_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None = None):
    if not rollouts:
        raise ValueError("At least one rollout is required.")
    fig, axes = _panel_axes(len(rollouts), figsize=figsize, base_height=4.2)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df = rollout.step_df.copy()
        _require(step_df, {"timestamp", "feeder_raw_net_load_kw", "feeder_effective_net_load_kw", "feeder_post_action_net_load_kw"}, _controller(rollout), "Rollout '{label}' is missing feeder net-load columns required for compare plotting: {missing}")
        for column, color, linestyle, label in [("feeder_raw_net_load_kw", "#111827", "-", "Feeder raw net load"), ("feeder_effective_net_load_kw", "#16a34a", "-.", "Feeder post-curtail net load"), ("feeder_post_action_net_load_kw", "#2563eb", "--", "Feeder post-action net load")]:
            axis.plot(step_df["timestamp"], step_df[column], color=color, linewidth=1.6 if column == "feeder_raw_net_load_kw" else 1.5, linestyle=linestyle, label=label)
        root_col = "root_net_exchange_kw" if "root_net_exchange_kw" in step_df.columns else "pp_root_p_kw" if "pp_root_p_kw" in step_df.columns else None
        if root_col:
            axis.plot(step_df["timestamp"], step_df[root_col], color="#7c3aed", linewidth=1.7, label="Root net exchange" if root_col == "root_net_exchange_kw" else "Pandapower root exchange")
        limit = rollout.meta.get("trafo_limit_kw")
        if limit is not None and np.isfinite(float(limit)) and float(limit) > 0.0:
            axis.axhline(float(limit), color="#dc2626", linestyle=":", linewidth=1.2, label=f"Transformer S-limit ref (+P view) ({float(limit):.1f} kW)" if idx == 0 else None)
            axis.axhline(-float(limit), color="#dc2626", linestyle=":", linewidth=1.2, label=f"Transformer S-limit ref (-P view) ({float(limit):.1f} kW)" if idx == 0 else None)
        _compare_style(axis, title=f"{_controller(rollout)} - feeder total", ylabel="kW")
        if idx == 0:
            axis.legend(loc="upper right", fontsize=_LEGEND)
    axes[-1].set_xlabel("Timestamp", fontsize=_LABEL)
    fig.tight_layout()
    return fig


def plot_power_balance_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None = None):
    if not rollouts:
        raise ValueError("At least one rollout is required.")
    fig, axes = _panel_axes(len(rollouts), figsize=figsize, base_height=4.0)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df = _with_pv_raw_total(rollout.step_df.copy())
        _require(step_df, {key for key, *_ in [*_POS_SPECS, *_NEG_SPECS]}, _controller(rollout), "Rollout '{label}' is missing required agent power-balance columns: {missing}")
        pos = sum((step_df[col].to_numpy(dtype=np.float32) for col, *_ in _POS_SPECS), start=np.zeros(len(step_df), dtype=np.float32))
        neg = sum((step_df[col].to_numpy(dtype=np.float32) for col, *_ in _NEG_SPECS), start=np.zeros(len(step_df), dtype=np.float32))
        residual = float(np.max(np.abs(pos - neg))) if len(step_df) else 0.0
        if residual > 1.0:
            raise ValueError(f"Rollout '{_controller(rollout)}' violates agent-only power balance by {residual:.3f} kW (tolerance=1.000 kW).")
        for specs, sign in ((_POS_SPECS, 1.0), (_NEG_SPECS, -1.0)):
            bottom = np.zeros(len(step_df), dtype=np.float32)
            for col, label, color in specs:
                values = sign * step_df[col].to_numpy(dtype=np.float32)
                extra = {"hatch": "//", "edgecolor": "#dc2626", "linewidth": 1.0} if col == "pv_curtail_total" else {}
                axis.bar(step_df["timestamp"], values, width=_WIDTH, bottom=bottom, color=color, alpha=0.78, label=label if idx == 0 else None, **extra)
                bottom += values
        axis.axhline(0.0, color="#111827", linewidth=1.0)
        _compare_style(axis, title=f"{_controller(rollout)} - agent-only balance (residual<={residual:.3f} kW)", ylabel="kW")
        axis.grid(True, axis="y", alpha=0.25)
        if idx == 0:
            axis.legend(loc="upper right", ncol=4, fontsize=_LEGEND)
    axes[-1].set_xlabel("Timestamp", fontsize=_LABEL)
    fig.tight_layout()
    return fig


def plot_battery_power_and_soc_comparison(*rollouts: RolloutResult, figsize: tuple[float, float] | None = None):
    if not rollouts:
        raise ValueError("At least one rollout is required.")
    fig, axes = _panel_axes(len(rollouts), figsize=figsize, base_height=4.2)
    for idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df, agent_df = rollout.step_df, rollout.agent_df
        if step_df.empty:
            raise ValueError(f"Rollout '{_controller(rollout)}' has no step_df.")
        _require(step_df, {"battery_charge_total", "battery_discharge_total", "timestamp"}, _controller(rollout), "Rollout '{label}' is missing battery power columns required for compare plotting.")
        if agent_df.empty or "soc" not in agent_df.columns or "agent_id" not in agent_df.columns:
            raise ValueError(f"Rollout '{_controller(rollout)}' is missing agent_df.soc/agent_id required for compare plotting.")
        net_power = step_df["battery_discharge_total"].to_numpy(dtype=np.float32) - step_df["battery_charge_total"].to_numpy(dtype=np.float32)
        axis.bar(step_df["timestamp"], net_power, width=_WIDTH, color=["#dc2626" if value >= 0.0 else "#2563eb" for value in net_power], alpha=0.82)
        axis.axhline(0.0, color="#475569", linewidth=1.0)
        _compare_style(axis, title=_controller(rollout), ylabel="Battery Power [kW]")
        groups = [column for column in ("episode_idx", "step", "timestamp") if column in agent_df.columns]
        if not groups:
            raise ValueError(f"Rollout '{_controller(rollout)}' is missing grouping columns required to aggregate SoC.")
        soc_axis = axis.twinx()
        summary = agent_df.groupby(groups, as_index=False).agg(mean=("soc", "mean"))
        first = True
        for _, frame in agent_df.sort_values(groups + ["agent_id"]).groupby("agent_id", sort=True):
            soc_axis.plot(frame["timestamp"], frame["soc"], color="#94a3b8", linewidth=1.1, alpha=0.45, label="Agent SoC" if first and idx == 0 else None)
            first = False
        soc_axis.plot(summary["timestamp"], summary["mean"], color="#111827", linewidth=1.7, label="Mean SoC" if idx == 0 else None)
        soc_axis.set_ylabel("SoC", fontsize=_LABEL)
        soc_axis.set_ylim(0.0, 1.0)
        soc_axis.grid(True, alpha=0.25)
        soc_axis.tick_params(axis="both", labelsize=_TICK)
        if idx == 0:
            handles, labels = soc_axis.get_legend_handles_labels()
            axis.legend([Patch(facecolor="#dc2626", alpha=0.82, label="Discharge (+)"), Patch(facecolor="#2563eb", alpha=0.82, label="Charge (-)"), *handles], ["Discharge (+)", "Charge (-)", *labels], loc="upper right", ncol=2, fontsize=_LEGEND)
    axes[-1].set_xlabel("Timestamp", fontsize=_LABEL)
    fig.tight_layout()
    return fig


def plot_global_misocp_validation(rollout: RolloutResult, *, figsize: tuple[float, float] = (18.0, 10.0)):
    validation_df = rollout.meta.get("misocp_validation_df", pd.DataFrame())
    if not isinstance(validation_df, pd.DataFrame) or validation_df.empty:
        raise ValueError(f"Rollout '{_controller(rollout)}' has no MISOCP validation dataframe.")
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for axis, column, title, ylabel in zip(axes, ("max_vm_abs_err_pu", "max_line_loading_abs_err_pct", "trafo_loading_abs_err_pct", "root_p_abs_err_kw"), ("Voltage replay error", "Line loading replay error", "Transformer loading replay error", "Root active-power replay error"), ("p.u.", "%", "%", "kW"), strict=False):
        axis.plot(validation_df["timestamp"], validation_df[column], color="#2563eb", linewidth=1.7)
        _compare_style(axis, title=title, ylabel=ylabel)
    axes[-1].set_xlabel("Timestamp", fontsize=_LABEL)
    fig.suptitle("\n".join((text for text in (_controller(rollout), str(rollout.meta.get("misocp_health_warning", "")).strip()) if text)), fontsize=_LABEL)
    fig.tight_layout()
    return fig
