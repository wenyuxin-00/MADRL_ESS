"""Validation summaries and plotting helpers for the MISOCP notebook flow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def build_misocp_validation_df(diagnostic_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a per-step MISOCP-vs-pandapower validation DataFrame."""

    from controllers.mpc.global_socp_mpc import (
        _LINE_LOADING_ERR_TOL_PCT,
        _ROOT_POWER_ERR_TOL_KW,
        _TRAFO_LOADING_ERR_TOL_PCT,
        _VOLTAGE_ERR_TOL_PU,
    )

    rows: list[dict[str, object]] = []
    for entry in diagnostic_rows:
        misocp_vm = entry.get("misocp_vm_pu")
        pp_vm = entry.get("pp_vm_pu")
        misocp_line = entry.get("misocp_line_loading_pct")
        pp_line = entry.get("pp_line_loading_pct")
        misocp_trafo = entry.get("misocp_trafo_loading_pct")
        pp_trafo = entry.get("pp_trafo_loading_pct")
        pp_root_p_value = entry.get("pp_root_p_kw", np.nan)
        root_p_available = bool(
            entry.get(
                "pp_root_p_available",
                np.isfinite(float(pp_root_p_value)) if pp_root_p_value is not None else False,
            )
        )
        if misocp_vm is None or pp_vm is None or misocp_line is None or pp_line is None or misocp_trafo is None or pp_trafo is None:
            max_vm_abs_err = np.nan
            max_line_abs_err = np.nan
            trafo_abs_err = np.nan
            root_p_abs_err = np.nan
            within_tolerance = False
        else:
            max_vm_abs_err = float(
                np.max(np.abs(np.asarray(misocp_vm, dtype=np.float32) - np.asarray(pp_vm, dtype=np.float32)))
            )
            max_line_abs_err = float(
                np.max(np.abs(np.asarray(misocp_line, dtype=np.float32) - np.asarray(pp_line, dtype=np.float32)))
            )
            trafo_abs_err = float(
                np.max(np.abs(np.asarray(misocp_trafo, dtype=np.float32) - np.asarray(pp_trafo, dtype=np.float32)))
            )
            root_p_abs_err = (
                float(abs(float(entry.get("root_p_kw", np.nan)) - float(pp_root_p_value)))
                if root_p_available
                else float("nan")
            )
            within_tolerance = bool(
                max_vm_abs_err < _VOLTAGE_ERR_TOL_PU
                and max_line_abs_err < _LINE_LOADING_ERR_TOL_PCT
                and trafo_abs_err < _TRAFO_LOADING_ERR_TOL_PCT
                and root_p_available
                and root_p_abs_err < _ROOT_POWER_ERR_TOL_KW
            )
        rows.append(
            {
                "controller": entry.get("controller", "MISOCP Global Oracle"),
                "episode_idx": int(entry.get("episode_idx", 0)),
                "step": int(entry.get("step", entry.get("global_step", 0))),
                "timestamp": entry.get("timestamp"),
                "misocp_fallback": float(entry.get("misocp_fallback", 0.0)),
                "misocp_time_limit_feasible": float(entry.get("misocp_time_limit_feasible", 0.0)),
                "solve_time_sec": float(entry.get("solve_time_sec", np.nan)),
                "max_vm_abs_err_pu": max_vm_abs_err,
                "max_line_loading_abs_err_pct": max_line_abs_err,
                "trafo_loading_abs_err_pct": trafo_abs_err,
                "root_p_abs_err_kw": root_p_abs_err,
                "pp_root_p_available": root_p_available,
                "root_power_validation_unavailable": bool(not root_p_available),
                "misocp_root_p_kw": float(entry.get("root_p_kw", np.nan)),
                "pp_root_p_kw": float(pp_root_p_value),
                "root_q_kvar": float(entry.get("root_q_kvar", np.nan)),
                "misocp_root_s_kva": float(entry.get("misocp_root_s_kva", np.nan)),
                "pp_root_s_kva": float(entry.get("pp_root_s_kva", np.nan)),
                "soc_slack_max": float(entry.get("soc_slack_max", np.nan)),
                "soc_slack_mean": float(entry.get("soc_slack_mean", np.nan)),
                "soc_slack_p95_global": float(entry.get("soc_slack_p95_global", np.nan)),
                "background_q_base_total_kvar": float(entry.get("background_q_base_total_kvar", np.nan)),
                "network_q_loss_proxy_kvar": float(entry.get("network_q_loss_proxy_kvar", np.nan)),
                "root_q_residual_kvar": float(entry.get("root_q_residual_kvar", np.nan)),
                "solver_feeder_gap_kw": float(entry.get("solver_feeder_gap_kw", np.nan)),
                "replay_feeder_gap_kw": float(entry.get("replay_feeder_gap_kw", np.nan)),
                "within_tolerance": within_tolerance,
            }
        )
    return pd.DataFrame(rows)


def summarize_misocp_validation(validation_df: pd.DataFrame) -> pd.Series:
    """Build a compact summary for the notebook validation section."""

    from scripts.utils.misocp_notebook_helpers import (
        _SOC_SLACK_TIGHT_P95_THRESHOLD,
        _interpret_linear_fit,
        _linear_fit_summary,
    )

    if validation_df.empty:
        return pd.Series(
            {
                "validation_rows": 0,
                "max_vm_abs_err_pu": np.nan,
                "max_line_loading_abs_err_pct": np.nan,
                "max_trafo_loading_abs_err_pct": np.nan,
                "max_root_p_abs_err_kw": np.nan,
                "steps_outside_tolerance": 0,
                "root_power_validation_unavailable": True,
                "pp_root_p_available_ratio": 0.0,
                "max_root_q_kvar": np.nan,
                "max_soc_slack": np.nan,
                "mean_soc_slack": np.nan,
                "p95_soc_slack": np.nan,
                "soc_relaxation_is_tight": False,
                "max_solver_feeder_gap_kw": np.nan,
                "mean_abs_solver_feeder_gap_kw": np.nan,
                "max_replay_feeder_gap_kw": np.nan,
                "mean_abs_replay_feeder_gap_kw": np.nan,
                "background_q_base_total_kvar": np.nan,
                "max_network_q_loss_proxy_kvar": np.nan,
                "max_abs_root_q_residual_kvar": np.nan,
                "root_p_fit_k": np.nan,
                "root_p_fit_b": np.nan,
                "root_p_fit_r2": np.nan,
                "root_p_fit_interpretation": "insufficient_data",
                "root_s_fit_k": np.nan,
                "root_s_fit_b": np.nan,
                "root_s_fit_r2": np.nan,
                "root_s_fit_interpretation": "insufficient_data",
            },
            name="misocp_validation_summary",
        )
    available_mask = validation_df["pp_root_p_available"].fillna(False).astype(bool)
    root_p_fit = _linear_fit_summary(
        validation_df.loc[available_mask, "misocp_root_p_kw"].to_numpy(dtype=np.float64),
        validation_df.loc[available_mask, "pp_root_p_kw"].to_numpy(dtype=np.float64),
    )
    root_s_fit = _linear_fit_summary(
        validation_df["misocp_root_s_kva"].to_numpy(dtype=np.float64),
        validation_df["pp_root_s_kva"].to_numpy(dtype=np.float64),
    )
    p95_soc_slack_series = validation_df["soc_slack_p95_global"].dropna()
    p95_soc_slack = float(p95_soc_slack_series.iloc[0]) if not p95_soc_slack_series.empty else float("nan")
    return pd.Series(
        {
            "validation_rows": int(len(validation_df)),
            "max_vm_abs_err_pu": float(validation_df["max_vm_abs_err_pu"].max()),
            "max_line_loading_abs_err_pct": float(validation_df["max_line_loading_abs_err_pct"].max()),
            "max_trafo_loading_abs_err_pct": float(validation_df["trafo_loading_abs_err_pct"].max()),
            "max_root_p_abs_err_kw": float(validation_df.loc[available_mask, "root_p_abs_err_kw"].max())
            if bool(available_mask.any())
            else float("nan"),
            "steps_outside_tolerance": int((~validation_df["within_tolerance"].fillna(False)).sum()),
            "root_power_validation_unavailable": bool(not available_mask.all()),
            "pp_root_p_available_ratio": float(np.mean(available_mask.to_numpy(dtype=np.float32))),
            "max_root_q_kvar": float(validation_df["root_q_kvar"].max()),
            "max_soc_slack": float(validation_df["soc_slack_max"].max()),
            "mean_soc_slack": float(validation_df["soc_slack_mean"].mean()),
            "p95_soc_slack": p95_soc_slack,
            "soc_relaxation_is_tight": bool(np.isfinite(p95_soc_slack) and p95_soc_slack < _SOC_SLACK_TIGHT_P95_THRESHOLD),
            "max_solver_feeder_gap_kw": float(np.abs(validation_df["solver_feeder_gap_kw"]).max()),
            "mean_abs_solver_feeder_gap_kw": float(np.abs(validation_df["solver_feeder_gap_kw"]).mean()),
            "max_replay_feeder_gap_kw": float(np.abs(validation_df["replay_feeder_gap_kw"]).max()),
            "mean_abs_replay_feeder_gap_kw": float(np.abs(validation_df["replay_feeder_gap_kw"]).mean()),
            "background_q_base_total_kvar": float(validation_df["background_q_base_total_kvar"].dropna().iloc[0])
            if not validation_df["background_q_base_total_kvar"].dropna().empty
            else float("nan"),
            "max_network_q_loss_proxy_kvar": float(validation_df["network_q_loss_proxy_kvar"].max()),
            "max_abs_root_q_residual_kvar": float(np.abs(validation_df["root_q_residual_kvar"]).max()),
            "root_p_fit_k": float(root_p_fit["k"]),
            "root_p_fit_b": float(root_p_fit["b"]),
            "root_p_fit_r2": float(root_p_fit["r2"]),
            "root_p_fit_interpretation": _interpret_linear_fit(
                float(root_p_fit["k"]),
                float(root_p_fit["b"]),
                float(root_p_fit["r2"]),
            ),
            "root_s_fit_k": float(root_s_fit["k"]),
            "root_s_fit_b": float(root_s_fit["b"]),
            "root_s_fit_r2": float(root_s_fit["r2"]),
            "root_s_fit_interpretation": _interpret_linear_fit(
                float(root_s_fit["k"]),
                float(root_s_fit["b"]),
                float(root_s_fit["r2"]),
            ),
        },
        name="misocp_validation_summary",
    )


def plot_full_horizon_power_balance(
    step_df: pd.DataFrame,
    *,
    controller_label: str,
    dt_hours: float,
    figsize: tuple[float, float] = (18.0, 4.8),
):
    """Render a MADRL-style single-panel full-horizon power-balance chart."""

    if step_df.empty:
        raise ValueError("step_df is empty; nothing to plot.")

    x_values = step_df["timestamp"]
    demand_specs = [
        ("agent_load_kw", "Agent load", "#111827"),
        ("fixed_load_kw", "Non-agent fixed load", "#4b5563"),
        ("battery_charge_kw", "Battery charge", "#dc2626"),
        ("grid_export_kw", "Grid export", "#f59e0b"),
    ]
    supply_specs = [
        ("pv_effective_kw", "PV effective", "#16a34a"),
        ("fixed_generation_kw", "Non-agent fixed generation", "#15803d"),
        ("grid_import_kw", "Grid import", "#2563eb"),
        ("battery_discharge_kw", "Battery discharge", "#7c3aed"),
    ]

    figure, balance_axis = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    x_index = pd.Index(x_values)
    is_temporal_x = bool(
        pd.api.types.is_datetime64_any_dtype(x_index.dtype)
        or isinstance(x_index.dtype, pd.DatetimeTZDtype)
    )
    if is_temporal_x:
        bar_width = pd.Timedelta(hours=max(float(dt_hours) * 0.8, 1e-3))
    else:
        bar_width = 0.8

    demand_bottom = np.zeros((len(step_df),), dtype=np.float32)
    for column, label, color in demand_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        extra_kwargs = (
            {"hatch": "//", "edgecolor": "#f59e0b"}
            if column == "grid_export_kw"
            else {}
        )
        balance_axis.bar(
            x_values,
            values,
            width=bar_width,
            bottom=demand_bottom,
            color=color,
            alpha=0.82,
            label=label,
            linewidth=0.0,
            zorder=2,
            **extra_kwargs,
        )
        demand_bottom = demand_bottom + values

    supply_bottom = np.zeros((len(step_df),), dtype=np.float32)
    for column, label, color in supply_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        balance_axis.bar(
            x_values,
            -values,
            width=bar_width,
            bottom=supply_bottom,
            color=color,
            alpha=0.82,
            label=label,
            linewidth=0.0,
            zorder=2,
        )
        supply_bottom = supply_bottom - values

    balance_axis.plot(
        x_values,
        step_df["pv_curtail_kw"].to_numpy(dtype=np.float32),
        color="#ef4444",
        linewidth=1.3,
        linestyle="--",
        label="PV curtailment",
        zorder=4,
    )
    balance_axis.plot(
        x_values,
        step_df["balance_residual_kw"].to_numpy(dtype=np.float32),
        color="#0f172a",
        linewidth=1.35,
        label="Balance residual",
        zorder=4,
    )
    balance_axis.axhline(0.0, color="#111827", linewidth=1.0, zorder=5)
    balance_axis.set_ylabel("Power [kW]")
    balance_axis.set_title(f"Power Balance - {controller_label}")
    balance_axis.set_xlabel("Timestamp")
    balance_axis.grid(True, axis="y", alpha=0.25)
    balance_axis.legend(loc="upper right", ncol=3)
    figure._misocp_balance_axes = [balance_axis]
    return figure


def plot_full_horizon_voltage(
    step_df: pd.DataFrame,
    grid_voltage_df: pd.DataFrame,
    *,
    agent_bus_ids: list[int] | tuple[int, ...],
    v_min_pu: float,
    v_max_pu: float,
    controller_label: str,
    figsize: tuple[float, float] = (18.0, 5.4),
):
    """Render the full-horizon bus-voltage chart."""

    if step_df.empty or grid_voltage_df.empty:
        raise ValueError("Voltage plotting requires non-empty step_df and grid_voltage_df.")

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    background_wide = (
        grid_voltage_df.loc[~grid_voltage_df["is_agent_bus"], ["timestamp", "bus_id", "vm_pu"]]
        .pivot_table(index="timestamp", columns="bus_id", values="vm_pu", aggfunc="first")
        .sort_index()
    )
    if background_wide.shape[1] > 0:
        axis.plot(
            background_wide.index.to_numpy(),
            background_wide.to_numpy(dtype=np.float32),
            color="#cbd5e1",
            linewidth=0.9,
            alpha=0.35,
            zorder=1,
        )

    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]
    for color_idx, bus_id in enumerate(agent_bus_ids):
        bus_frame = grid_voltage_df.loc[grid_voltage_df["bus_id"] == int(bus_id)]
        if bus_frame.empty:
            continue
        axis.plot(
            bus_frame["timestamp"],
            bus_frame["vm_pu"],
            color=highlight_palette[color_idx % len(highlight_palette)],
            linewidth=1.9,
            alpha=0.95,
            label=f"Agent bus {int(bus_id)}",
            zorder=3,
        )

    axis.plot(
        step_df["timestamp"],
        step_df["min_vm_pu"],
        color="#0f172a",
        linewidth=1.4,
        linestyle="--",
        label="Min vm",
    )
    axis.plot(
        step_df["timestamp"],
        step_df["mean_vm_pu"],
        color="#475569",
        linewidth=1.3,
        linestyle="-.",
        label="Mean vm",
    )
    axis.plot(
        step_df["timestamp"],
        step_df["max_vm_pu"],
        color="#334155",
        linewidth=1.4,
        linestyle=":",
        label="Max vm",
    )
    axis.axhline(float(v_min_pu), color="#dc2626", linestyle="--", linewidth=1.1, label="V min")
    axis.axhline(float(v_max_pu), color="#ea580c", linestyle="--", linewidth=1.1, label="V max")
    axis.set_title(f"Node Voltage Profile - {controller_label}")
    axis.set_ylabel("Voltage [p.u.]")
    axis.set_xlabel("Timestamp")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", ncol=3)
    figure.tight_layout()
    figure._misocp_voltage_axis = axis
    return figure


def plot_full_horizon_net_load(
    step_df: pd.DataFrame,
    *,
    trafo_limit_kw: float | None,
    controller_label: str,
    figsize: tuple[float, float] = (18.0, 8.2),
):
    """Render feeder-total and agent-only net load with a transformer S-limit reference in active-power view."""

    from scripts.utils.misocp_notebook_helpers import (
        _resolve_agent_net_load_columns,
        _resolve_feeder_net_load_columns,
    )

    if step_df.empty:
        raise ValueError("step_df is empty; nothing to plot.")

    agent_raw, agent_effective, agent_post_action = _resolve_agent_net_load_columns(step_df)
    feeder_raw, feeder_effective, feeder_post_action = _resolve_feeder_net_load_columns(step_df)

    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    feeder_axis, agent_axis = axes

    if feeder_raw is not None and feeder_effective is not None and feeder_post_action is not None:
        feeder_axis.plot(
            step_df["timestamp"],
            feeder_raw,
            color="#111827",
            linewidth=1.6,
            label="Feeder raw net load",
        )
        feeder_axis.plot(
            step_df["timestamp"],
            feeder_effective,
            color="#16a34a",
            linewidth=1.4,
            linestyle="-.",
            label="Feeder post-curtail net load",
        )
        feeder_axis.plot(
            step_df["timestamp"],
            feeder_post_action,
            color="#2563eb",
            linewidth=1.5,
            linestyle="--",
            label="Feeder post-action net load",
        )
    if "root_net_exchange_kw" in step_df.columns:
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["root_net_exchange_kw"],
            color="#7c3aed",
            linewidth=1.7,
            label="MISOCP root net exchange",
        )
    if "pp_root_p_kw" in step_df.columns:
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["pp_root_p_kw"],
            color="#0f766e",
            linewidth=1.4,
            linestyle=":",
            label="Pandapower root exchange",
        )
    if trafo_limit_kw is not None and np.isfinite(float(trafo_limit_kw)):
        limit = float(trafo_limit_kw)
        feeder_axis.axhline(limit, color="#b91c1c", linestyle=":", linewidth=1.2, label="Transformer S-limit ref (+P view)")
        feeder_axis.axhline(-limit, color="#b91c1c", linestyle=":", linewidth=1.2, label="Transformer S-limit ref (-P view)")
    feeder_axis.set_title(f"Net Load - {controller_label}\nFeeder total (transformer S-limit shown as active-power reference)")
    feeder_axis.set_ylabel("Power [kW]")
    feeder_axis.grid(True, alpha=0.25)
    feeder_axis.legend(loc="upper right", ncol=2)

    agent_axis.plot(
        step_df["timestamp"],
        agent_raw,
        color="#111827",
        linewidth=1.6,
        label="Agent raw net load",
    )
    agent_axis.plot(
        step_df["timestamp"],
        agent_effective,
        color="#16a34a",
        linewidth=1.4,
        linestyle="-.",
        label="Agent post-curtail net load",
    )
    agent_axis.plot(
        step_df["timestamp"],
        agent_post_action,
        color="#2563eb",
        linewidth=1.5,
        linestyle="--",
        label="Agent post-action net load",
    )
    agent_axis.set_title("Agent-only aggregate")
    agent_axis.set_ylabel("Power [kW]")
    agent_axis.set_xlabel("Timestamp")
    agent_axis.grid(True, alpha=0.25)
    agent_axis.legend(loc="upper right")
    figure._misocp_net_load_axes = list(axes)
    return figure


def plot_root_exchange_alignment(
    step_df: pd.DataFrame,
    *,
    controller_label: str,
    figsize: tuple[float, float] = (14.0, 4.5),
) -> plt.Figure:
    """Plot solver root exchange, replay root exchange, and feeder post-action net load together."""

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    axis.plot(
        step_df["timestamp"],
        step_df["root_net_exchange_kw"],
        color="#2563eb",
        linewidth=1.6,
        label="MISOCP root net exchange",
    )
    if "pp_root_p_kw" in step_df.columns:
        axis.plot(
            step_df["timestamp"],
            step_df["pp_root_p_kw"],
            color="#f97316",
            linewidth=1.4,
            linestyle="--",
            label="Pandapower root exchange",
        )
    axis.plot(
        step_df["timestamp"],
        step_df["feeder_post_action_net_load_kw"],
        color="#16a34a",
        linewidth=1.4,
        linestyle="-.",
        label="Feeder post-action net load",
    )
    axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.6)
    axis.set_title(f"{controller_label}: Root Exchange Alignment")
    axis.set_ylabel("Power [kW]")
    axis.set_xlabel("Timestamp")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")
    return figure


def plot_misocp_validation_scatter_panel(
    validation_df: pd.DataFrame,
    *,
    controller_label: str,
    figsize: tuple[float, float] = (12.0, 4.8),
) -> plt.Figure:
    """Plot root-P and root-S scatter diagnostics with linear-fit annotations."""

    from scripts.utils.misocp_notebook_helpers import _interpret_linear_fit, _linear_fit_summary

    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    scatter_specs = [
        (
            axes[0],
            "misocp_root_p_kw",
            "pp_root_p_kw",
            "Root P fit",
            "MISOCP root P [kW]",
            "Pandapower root P [kW]",
        ),
        (
            axes[1],
            "misocp_root_s_kva",
            "pp_root_s_kva",
            "Root S fit",
            "MISOCP root S [kVA]",
            "Pandapower root S [kVA]",
        ),
    ]
    for axis, x_col, y_col, title, x_label, y_label in scatter_specs:
        x = validation_df.get(x_col, pd.Series(dtype=np.float32)).to_numpy(dtype=np.float64)
        y = validation_df.get(y_col, pd.Series(dtype=np.float32)).to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        axis.scatter(x[finite_mask], y[finite_mask], s=18, alpha=0.75, color="#2563eb")
        fit = _linear_fit_summary(x[finite_mask], y[finite_mask])
        interpretation = _interpret_linear_fit(float(fit["k"]), float(fit["b"]), float(fit["r2"]))
        if int(np.sum(finite_mask)) >= 2 and np.isfinite(float(fit["k"])) and np.isfinite(float(fit["b"])):
            x_min = float(np.min(x[finite_mask]))
            x_max = float(np.max(x[finite_mask]))
            fit_x = np.linspace(x_min, x_max, num=100, dtype=np.float64)
            fit_y = float(fit["k"]) * fit_x + float(fit["b"])
            axis.plot(fit_x, fit_y, color="#ef4444", linewidth=1.2, label="linear fit")
        if int(np.sum(finite_mask)) >= 1:
            identity_min = float(np.min(np.concatenate([x[finite_mask], y[finite_mask]])))
            identity_max = float(np.max(np.concatenate([x[finite_mask], y[finite_mask]])))
            axis.plot([identity_min, identity_max], [identity_min, identity_max], color="#6b7280", linewidth=1.0, linestyle=":")
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        axis.text(
            0.03,
            0.97,
            f"k={fit['k']:.4g}\nb={fit['b']:.4g}\nR²={fit['r2']:.4g}\n{interpretation}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
        )
    figure.suptitle(f"{controller_label}: Validation Fit Diagnostics", fontsize=12)
    return figure
