"""Internal plotting helpers for MADRL notebook workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.utils.price_protocol import (
    IMPORT_PRICE_COLUMN,
    IMPORT_PRICE_MARKUP_KEY,
    IMPORT_PRICE_PRED_COLUMN,
    WHOLESALE_PRICE_PRED_COLUMN,
    derive_import_price_seq,
)

if TYPE_CHECKING:
    from scripts.utils.grid_notebook_workflow import RolloutResult

NORMAL_PREDICTION_MODE = "normal"

COMPARE_PLOT_TITLE_FONTSIZE = 20
COMPARE_PLOT_LABEL_FONTSIZE = 18
COMPARE_PLOT_TICK_FONTSIZE = 16
COMPARE_PLOT_LEGEND_FONTSIZE = 16
COMPARE_PLOT_AGENT_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]
COMPARE_PLOT_MUTED_COLOR = "#cbd5e1"

def plot_global_misocp_validation(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] = (11.0, 8.0),
):
    validation_df = rollout.meta.get("misocp_validation_df")
    if not isinstance(validation_df, pd.DataFrame) or validation_df.empty:
        raise ValueError("Rollout does not contain MISOCP-vs-pandapower validation data.")

    from controllers.mpc.global_socp_mpc import (
        _LINE_LOADING_ERR_TOL_PCT,
        _ROOT_POWER_ERR_TOL_KW,
        _TRAFO_LOADING_ERR_TOL_PCT,
        _VOLTAGE_ERR_TOL_PU,
    )

    figure, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
    series = [
        ("max_vm_abs_err_pu", _VOLTAGE_ERR_TOL_PU, "Max |V_misocp - V_pp| [p.u.]"),
        ("max_line_loading_abs_err_pct", _LINE_LOADING_ERR_TOL_PCT, "Max |Line Loading| Error [pct-point]"),
        ("trafo_loading_abs_err_pct", _TRAFO_LOADING_ERR_TOL_PCT, "|Trafo Loading| Error [pct-point]"),
        ("root_p_abs_err_kw", _ROOT_POWER_ERR_TOL_KW, "|P_root| Error [kW]"),
    ]
    timestamps = validation_df["timestamp"]
    for axis, (column, threshold, ylabel) in zip(axes, series, strict=False):
        axis.plot(timestamps, validation_df[column], color="#1d4ed8", linewidth=1.8)
        axis.axhline(float(threshold), color="#dc2626", linestyle="--", linewidth=1.2)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    warning = str(rollout.meta.get("misocp_health_warning", "") or "")
    title = f"{rollout.meta.get('controller', 'Global SOCP-MPC')} Validation"
    if warning:
        title = f"{title}\n{warning}"
    axes[0].set_title(title)
    axes[-1].set_xlabel("Timestamp")
    return figure


def compare_purchase_costs(*rollouts: RolloutResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rollout in rollouts:
        total_cost = float(rollout.agent_df["purchase_cost"].sum()) if not rollout.agent_df.empty else 0.0
        rows.append(
            {
                "controller": rollout.meta["controller"],
                "purchase_cost_total": total_cost,
            }
        )
    return pd.DataFrame(rows).sort_values("purchase_cost_total").reset_index(drop=True)


def _prepare_agent_power_balance_frame(
    step_df: pd.DataFrame,
    *,
    controller_label: str,
    tolerance_kw: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    if step_df.empty:
        raise ValueError(f"Rollout '{controller_label}' has no step_df.")
    prepared = step_df.copy()
    required_columns = {
        "load_total",
        "battery_charge_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
        "battery_discharge_total",
    }
    missing = sorted(required_columns.difference(prepared.columns))
    if missing:
        raise ValueError(
            f"Rollout '{controller_label}' is missing required agent power-balance columns: {missing}"
        )
    if "pv_raw_total" not in prepared.columns:
        if "pv_effective_total" not in prepared.columns:
            raise ValueError(
                f"Rollout '{controller_label}' must contain either pv_raw_total or pv_effective_total."
            )
        prepared["pv_raw_total"] = (
            prepared["pv_effective_total"].to_numpy(dtype=np.float32)
            + prepared["pv_curtail_total"].to_numpy(dtype=np.float32)
        )

    positive_total = (
        prepared["load_total"].to_numpy(dtype=np.float32)
        + prepared["battery_charge_total"].to_numpy(dtype=np.float32)
        + prepared["grid_export_total"].to_numpy(dtype=np.float32)
        + prepared["pv_curtail_total"].to_numpy(dtype=np.float32)
    )
    negative_total = (
        prepared["pv_raw_total"].to_numpy(dtype=np.float32)
        + prepared["grid_import_total"].to_numpy(dtype=np.float32)
        + prepared["battery_discharge_total"].to_numpy(dtype=np.float32)
    )
    balance_residual_kw = positive_total - negative_total
    max_abs_balance_residual_kw = float(np.max(np.abs(balance_residual_kw))) if balance_residual_kw.size else 0.0
    if max_abs_balance_residual_kw > float(tolerance_kw):
        raise ValueError(
            f"Rollout '{controller_label}' violates agent-only power balance by "
            f"{max_abs_balance_residual_kw:.3f} kW (tolerance={float(tolerance_kw):.3f} kW)."
        )
    prepared["agent_power_balance_residual_kw"] = balance_residual_kw.astype(np.float32)
    return prepared, max_abs_balance_residual_kw


def _prepare_compare_net_load_frame(
    step_df: pd.DataFrame,
    *,
    controller_label: str,
) -> pd.DataFrame:
    if step_df.empty:
        raise ValueError(f"Rollout '{controller_label}' has no step_df.")
    prepared = step_df.copy()

    if not {"agent_raw_net_load_kw", "agent_effective_net_load_kw", "agent_post_action_net_load_kw"}.issubset(
        prepared.columns
    ):
        if not {"base_net_load_total", "net_load_total"}.issubset(prepared.columns):
            raise ValueError(
                f"Rollout '{controller_label}' is missing agent net-load columns required for compare plotting."
            )
        prepared["agent_raw_net_load_kw"] = prepared["base_net_load_total"].to_numpy(dtype=np.float32)
        prepared["agent_effective_net_load_kw"] = np.asarray(
            prepared.get("base_net_load_effective_total", prepared["base_net_load_total"]),
            dtype=np.float32,
        )
        prepared["agent_post_action_net_load_kw"] = prepared["net_load_total"].to_numpy(dtype=np.float32)

    if not {"feeder_raw_net_load_kw", "feeder_effective_net_load_kw", "feeder_post_action_net_load_kw"}.issubset(
        prepared.columns
    ):
        if not {"fixed_load_kw", "fixed_generation_kw"}.issubset(prepared.columns):
            raise ValueError(
                f"Rollout '{controller_label}' is missing feeder-total net-load columns required for compare plotting."
            )
        fixed_load = prepared["fixed_load_kw"].to_numpy(dtype=np.float32)
        fixed_generation = prepared["fixed_generation_kw"].to_numpy(dtype=np.float32)
        prepared["feeder_raw_net_load_kw"] = (
            prepared["agent_raw_net_load_kw"].to_numpy(dtype=np.float32) + fixed_load - fixed_generation
        )
        prepared["feeder_effective_net_load_kw"] = (
            prepared["agent_effective_net_load_kw"].to_numpy(dtype=np.float32) + fixed_load - fixed_generation
        )
        prepared["feeder_post_action_net_load_kw"] = (
            prepared["agent_post_action_net_load_kw"].to_numpy(dtype=np.float32) + fixed_load - fixed_generation
        )

    return prepared


def _compare_plot_fontsizes() -> tuple[int, int, int, int]:
    return (
        COMPARE_PLOT_TITLE_FONTSIZE,
        COMPARE_PLOT_LABEL_FONTSIZE,
        COMPARE_PLOT_TICK_FONTSIZE,
        COMPARE_PLOT_LEGEND_FONTSIZE,
    )


def _prepare_shared_price_frame(
    step_df: pd.DataFrame,
    *,
    controller_label: str,
    require_predicted: bool = False,
) -> pd.DataFrame:
    required_columns = {"timestamp", "wholesale_price", IMPORT_PRICE_COLUMN}
    if require_predicted:
        required_columns.add(WHOLESALE_PRICE_PRED_COLUMN)
    if step_df.empty or not required_columns.issubset(step_df.columns):
        missing = sorted(required_columns.difference(step_df.columns))
        raise ValueError(
            f"Rollout '{controller_label}' is missing shared price columns required for compare plotting: {missing}"
        )
    columns = ["timestamp", "wholesale_price", IMPORT_PRICE_COLUMN]
    if WHOLESALE_PRICE_PRED_COLUMN in step_df.columns:
        columns.append(WHOLESALE_PRICE_PRED_COLUMN)
    if IMPORT_PRICE_PRED_COLUMN in step_df.columns:
        columns.append(IMPORT_PRICE_PRED_COLUMN)
    frame = step_df.loc[:, columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["timestamp"]).drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return frame


def _resolve_adjusted_price_prediction(step_df: pd.DataFrame, *, price_markup: float) -> np.ndarray:
    if IMPORT_PRICE_PRED_COLUMN in step_df.columns:
        return step_df[IMPORT_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64)
    return derive_import_price_seq(
        step_df[WHOLESALE_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64),
        markup_eur_per_kwh=float(price_markup),
    )


def _prepare_shared_agent_signal_frame(
    agent_df: pd.DataFrame,
    signal_name: str,
    *,
    controller_label: str,
    require_predicted: bool = False,
) -> pd.DataFrame:
    required_columns = {"timestamp", "agent_profile", signal_name}
    predicted_column = f"{signal_name}_pred"
    if require_predicted:
        required_columns.add(predicted_column)
    if agent_df.empty or not required_columns.issubset(agent_df.columns):
        missing = sorted(required_columns.difference(agent_df.columns))
        raise ValueError(
            f"Rollout '{controller_label}' is missing shared {signal_name} columns required for compare plotting: {missing}"
        )
    columns = ["timestamp", "agent_profile", signal_name]
    if predicted_column in agent_df.columns:
        columns.append(predicted_column)
    frame = agent_df.loc[:, columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["agent_profile"] = frame["agent_profile"].astype(str)
    frame = (
        frame.sort_values(["timestamp", "agent_profile"])
        .drop_duplicates(subset=["timestamp", "agent_profile"], keep="first")
        .reset_index(drop=True)
    )
    return frame


def _assert_matching_shared_actual_frame(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    key_columns: list[str],
    value_column: str,
    controller_label: str,
    signal_label: str,
) -> None:
    reference_keys = reference_frame.loc[:, key_columns].to_records(index=False)
    candidate_keys = candidate_frame.loc[:, key_columns].to_records(index=False)
    if reference_keys.shape != candidate_keys.shape or not np.array_equal(reference_keys, candidate_keys):
        raise ValueError(
            f"Rollout '{controller_label}' has different {signal_label} timestamps/profiles; "
            "shared forecast-vs-actual plotting requires matching actual series across rollouts."
        )
    if not np.allclose(
        reference_frame[value_column].to_numpy(dtype=np.float64),
        candidate_frame[value_column].to_numpy(dtype=np.float64),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    ):
        raise ValueError(
            f"Rollout '{controller_label}' has different actual {signal_label} values; "
            "shared forecast-vs-actual plotting requires matching actual series across rollouts."
        )


def _align_shared_actual_frames(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    key_columns: list[str],
    value_column: str,
    controller_label: str,
    signal_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_indexed = reference_frame.set_index(key_columns)
    candidate_indexed = candidate_frame.set_index(key_columns)
    common_index = reference_indexed.index.intersection(candidate_indexed.index, sort=False)
    if len(common_index) == 0:
        raise ValueError(
            f"Rollout '{controller_label}' has no overlapping {signal_label} timestamps/profiles with the "
            "shared forecast-vs-actual reference rollout."
        )

    reference_aligned = reference_indexed.loc[common_index].reset_index()
    candidate_aligned = candidate_indexed.loc[common_index].reset_index()
    _assert_matching_shared_actual_frame(
        reference_aligned,
        candidate_aligned,
        key_columns=key_columns,
        value_column=value_column,
        controller_label=controller_label,
        signal_label=signal_label,
    )
    return reference_aligned, candidate_aligned


def _select_shared_forecast_reference_rollout(*rollouts: RolloutResult) -> RolloutResult:
    for rollout in rollouts:
        if str(rollout.meta.get("prediction_mode", "")).strip().lower() != NORMAL_PREDICTION_MODE:
            continue
        step_df = rollout.step_df
        agent_df = rollout.agent_df
        if {"timestamp", "wholesale_price", IMPORT_PRICE_COLUMN, WHOLESALE_PRICE_PRED_COLUMN}.issubset(step_df.columns) and {
            "timestamp",
            "agent_profile",
            "load",
            "load_pred",
            "pv",
            "pv_pred",
        }.issubset(agent_df.columns):
            return rollout
    return rollouts[0]


def plot_shared_forecast_vs_actual(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize = _compare_plot_fontsizes()
    reference_rollout = rollouts[0]
    reference_label = str(reference_rollout.meta.get("controller", "unknown"))
    price_reference = _prepare_shared_price_frame(
        reference_rollout.step_df,
        controller_label=reference_label,
    )
    load_reference = _prepare_shared_agent_signal_frame(
        reference_rollout.agent_df,
        "load",
        controller_label=reference_label,
    )
    pv_reference = _prepare_shared_agent_signal_frame(
        reference_rollout.agent_df,
        "pv",
        controller_label=reference_label,
    )

    for rollout in rollouts[1:]:
        controller_label = str(rollout.meta.get("controller", "unknown"))
        price_candidate = _prepare_shared_price_frame(
            rollout.step_df,
            controller_label=controller_label,
        )
        load_candidate = _prepare_shared_agent_signal_frame(
            rollout.agent_df,
            "load",
            controller_label=controller_label,
        )
        pv_candidate = _prepare_shared_agent_signal_frame(
            rollout.agent_df,
            "pv",
            controller_label=controller_label,
        )
        price_reference, _ = _align_shared_actual_frames(
            price_reference,
            price_candidate,
            key_columns=["timestamp"],
            value_column=IMPORT_PRICE_COLUMN,
            controller_label=controller_label,
            signal_label="import_price",
        )
        load_reference, _ = _align_shared_actual_frames(
            load_reference,
            load_candidate,
            key_columns=["timestamp", "agent_profile"],
            value_column="load",
            controller_label=controller_label,
            signal_label="load",
        )
        pv_reference, _ = _align_shared_actual_frames(
            pv_reference,
            pv_candidate,
            key_columns=["timestamp", "agent_profile"],
            value_column="pv",
            controller_label=controller_label,
            signal_label="pv",
        )

    prediction_rollout = _select_shared_forecast_reference_rollout(*rollouts)
    prediction_label = str(prediction_rollout.meta.get("controller", "unknown"))
    price_prediction = _prepare_shared_price_frame(
        prediction_rollout.step_df,
        controller_label=prediction_label,
        require_predicted=True,
    )
    load_prediction = _prepare_shared_agent_signal_frame(
        prediction_rollout.agent_df,
        "load",
        controller_label=prediction_label,
        require_predicted=True,
    )
    pv_prediction = _prepare_shared_agent_signal_frame(
        prediction_rollout.agent_df,
        "pv",
        controller_label=prediction_label,
        require_predicted=True,
    )
    price_reference, price_prediction = _align_shared_actual_frames(
        price_reference,
        price_prediction,
        key_columns=["timestamp"],
        value_column=IMPORT_PRICE_COLUMN,
        controller_label=prediction_label,
        signal_label="import_price",
    )
    load_reference, load_prediction = _align_shared_actual_frames(
        load_reference,
        load_prediction,
        key_columns=["timestamp", "agent_profile"],
        value_column="load",
        controller_label=prediction_label,
        signal_label="load",
    )
    pv_reference, pv_prediction = _align_shared_actual_frames(
        pv_reference,
        pv_prediction,
        key_columns=["timestamp", "agent_profile"],
        value_column="pv",
        controller_label=prediction_label,
        signal_label="pv",
    )

    figure, axes = plt.subplots(3, 1, figsize=figsize or (20.0, 12.0), sharex=True)
    price_axis, load_axis, pv_axis = np.atleast_1d(axes)
    price_axis.plot(
        price_reference["timestamp"],
        price_reference[IMPORT_PRICE_COLUMN],
        color="#111827",
        linewidth=1.9,
        label="Actual import price",
    )
    price_axis.plot(
        price_prediction["timestamp"],
        _resolve_adjusted_price_prediction(
            price_prediction,
            price_markup=float(prediction_rollout.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0)),
        ),
        color="#dc2626",
        linewidth=1.8,
        linestyle="--",
        label=f"Predicted import price ({prediction_label})",
    )
    price_axis.set_title("Import Price Forecast vs Actual", fontsize=title_fontsize)
    price_axis.set_ylabel("EUR/kWh", fontsize=label_fontsize)
    price_axis.grid(True, alpha=0.25)
    price_axis.tick_params(axis="both", labelsize=tick_fontsize)
    price_axis.legend(loc="upper right", fontsize=legend_fontsize)

    profile_names = list(dict.fromkeys(load_reference["agent_profile"].astype(str).tolist()))
    figure_handles = [
        Line2D([0], [0], color=COMPARE_PLOT_AGENT_PALETTE[idx % len(COMPARE_PLOT_AGENT_PALETTE)], linewidth=2.0, label=profile)
        for idx, profile in enumerate(profile_names)
    ]
    figure_handles.extend(
        [
            Line2D([0], [0], color="#111827", linewidth=2.0, linestyle="-", label="Actual"),
            Line2D([0], [0], color="#111827", linewidth=2.0, linestyle="--", label="Predicted"),
        ]
    )

    for axis, signal_name, reference_frame, prediction_frame in (
        (load_axis, "load", load_reference, load_prediction),
        (pv_axis, "pv", pv_reference, pv_prediction),
    ):
        for color_idx, profile in enumerate(profile_names):
            color = COMPARE_PLOT_AGENT_PALETTE[color_idx % len(COMPARE_PLOT_AGENT_PALETTE)]
            actual_frame = reference_frame.loc[reference_frame["agent_profile"] == profile]
            predicted_frame = prediction_frame.loc[prediction_frame["agent_profile"] == profile]
            axis.plot(
                actual_frame["timestamp"],
                actual_frame[signal_name],
                color=color,
                linewidth=1.8,
            )
            axis.plot(
                predicted_frame["timestamp"],
                predicted_frame[f"{signal_name}_pred"],
                color=color,
                linewidth=1.6,
                linestyle="--",
            )
        axis.set_title(f"{signal_name.upper()} Forecast vs Actual", fontsize=title_fontsize)
        axis.set_ylabel("kW", fontsize=label_fontsize)
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="both", labelsize=tick_fontsize)

    pv_axis.set_xlabel("Timestamp", fontsize=label_fontsize)
    figure.legend(
        handles=figure_handles,
        loc="upper center",
        ncol=min(len(figure_handles), 7),
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure


def plot_rollout_dashboard(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] | None = None,
):
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    if step_df.empty or agent_df.empty:
        raise ValueError("Rollout is empty; nothing to plot.")

    agent_profiles = list(rollout.meta["agent_profiles"])
    n_agents = len(agent_profiles)
    main_axis_count = 6 + n_agents
    if figsize is None:
        figsize = (18.0, 2.8 * main_axis_count)

    figure, axes = plt.subplots(main_axis_count, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    palette = ["#0f172a", "#2563eb", "#16a34a", "#ea580c", "#dc2626", "#7c3aed"]

    axes[0].plot(
        step_df["timestamp"],
        step_df[IMPORT_PRICE_COLUMN],
        color="#111827",
        linewidth=1.6,
        label="Actual",
    )
    axes[0].plot(
        step_df["timestamp"],
        step_df[IMPORT_PRICE_PRED_COLUMN],
        color="#dc2626",
        linewidth=1.4,
        linestyle="--",
        label="Forecast",
    )
    axes[0].set_ylabel("Import Price")
    axes[0].set_title(f"Test Rollout Dashboard - {rollout.meta['controller']}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    for axis_idx, signal_name in enumerate(["pv", "load"], start=1):
        axis = axes[axis_idx]
        for agent_idx, profile in enumerate(agent_profiles):
            agent_frame = agent_df.loc[agent_df["agent_profile"] == profile]
            color = palette[agent_idx % len(palette)]
            axis.plot(
                agent_frame["timestamp"],
                agent_frame[signal_name],
                color=color,
                linewidth=1.4,
                label=f"{profile} actual",
            )
            axis.plot(
                agent_frame["timestamp"],
                agent_frame[f"{signal_name}_pred"],
                color=color,
                linewidth=1.2,
                linestyle="--",
                label=f"{profile} forecast",
            )
        axis.set_ylabel(signal_name.upper())
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=2)

    voltage_axis = axes[3]
    if grid_df.empty:
        raise ValueError("Rollout does not contain full-grid voltage traces.")

    muted_color = "#cbd5e1"
    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]

    for bus_id, frame in grid_df.loc[~grid_df["is_agent_bus"]].groupby("bus_id"):
        voltage_axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=muted_color,
            linewidth=0.9,
            alpha=0.35,
            zorder=1,
        )

    for color_idx, bus_id in enumerate(rollout.meta["agent_bus_ids"]):
        frame = grid_df.loc[grid_df["bus_id"] == int(bus_id)]
        if frame.empty:
            continue
        voltage_axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=highlight_palette[color_idx % len(highlight_palette)],
            linewidth=2.1,
            alpha=0.95,
            label=f"Agent bus {bus_id}",
            zorder=3,
        )

    voltage_axis.axhline(
        float(rollout.meta["v_min_pu"]),
        color="#dc2626",
        linestyle="--",
        linewidth=1.1,
        label="V min",
    )
    voltage_axis.axhline(
        float(rollout.meta["v_max_pu"]),
        color="#ea580c",
        linestyle="--",
        linewidth=1.1,
        label="V max",
    )
    voltage_axis.set_ylabel("Voltage [p.u.]")
    voltage_axis.grid(True, alpha=0.25)
    voltage_axis.legend(loc="upper right", ncol=2)

    net_load_axis = axes[4]
    net_load_axis.plot(
        step_df["timestamp"],
        step_df["base_net_load_total"],
        color="#111827",
        linewidth=1.6,
        label="Raw net load",
    )
    if "base_net_load_effective_total" in step_df.columns:
        net_load_axis.plot(
            step_df["timestamp"],
            step_df["base_net_load_effective_total"],
            color="#16a34a",
            linewidth=1.4,
            linestyle="-.",
            label="Post-curtail net load",
        )
    net_load_axis.plot(
        step_df["timestamp"],
        step_df["net_load_total"],
        color="#2563eb",
        linewidth=1.5,
        linestyle="--",
        label="Post-action net load",
    )
    trafo_limit_kw = rollout.meta.get("trafo_limit_kw")
    if trafo_limit_kw is not None:
        trafo_limit_value = float(trafo_limit_kw)
        if np.isfinite(trafo_limit_value) and trafo_limit_value > 0.0:
            net_load_axis.axhline(
                trafo_limit_value,
                color="#dc2626",
                linestyle=":",
                linewidth=1.2,
                label=f"Approx trafo +limit ({trafo_limit_value:.1f} kW)",
            )
            net_load_axis.axhline(
                -trafo_limit_value,
                color="#dc2626",
                linestyle=":",
                linewidth=1.2,
                label=f"Approx trafo -limit ({trafo_limit_value:.1f} kW)",
            )
    net_load_axis.set_ylabel("Net load")
    net_load_axis.grid(True, alpha=0.25)
    net_load_axis.legend(loc="upper right")

    pv_axis = axes[5]
    if "pv_effective_total" in step_df.columns:
        pv_axis.plot(
            step_df["timestamp"],
            step_df["pv_raw_total"],
            color="#ea580c",
            linewidth=1.5,
            label="Raw PV",
        )
        pv_axis.plot(
            step_df["timestamp"],
            step_df["pv_effective_total"],
            color="#16a34a",
            linewidth=1.5,
            linestyle="--",
            label="Effective PV",
        )
        pv_axis.bar(
            step_df["timestamp"],
            step_df["pv_curtail_total"],
            width=0.008,
            color="#dc2626",
            alpha=0.35,
            label="Curtailment",
        )
        pv_axis.set_ylabel("PV")
        pv_axis.grid(True, alpha=0.25)
        pv_axis.legend(loc="upper right")

    for agent_offset, profile in enumerate(agent_profiles, start=6):
        axis = axes[agent_offset]
        agent_frame = agent_df.loc[agent_df["agent_profile"] == profile]
        charge = np.clip(agent_frame["e_bat"].to_numpy(dtype=np.float32), 0.0, None)
        discharge = np.clip(agent_frame["e_bat"].to_numpy(dtype=np.float32), None, 0.0)
        axis.bar(agent_frame["timestamp"], charge, width=0.008, color="#dc2626", alpha=0.7, label="Charge")
        axis.bar(agent_frame["timestamp"], discharge, width=0.008, color="#2563eb", alpha=0.7, label="Discharge")
        axis.set_ylabel(f"{profile}\nP_bat")
        axis.grid(True, alpha=0.25)

        soc_axis = axis.twinx()
        soc_axis.plot(agent_frame["timestamp"], agent_frame["soc"], color="#111827", linewidth=1.2, label="SoC")
        soc_axis.set_ylabel("SoC")
        soc_axis.set_ylim(0.0, 1.0)

        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = soc_axis.get_legend_handles_labels()
        axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    axes[-1].set_xlabel("Timestamp")
    figure.tight_layout()
    figure._dashboard_main_axes = list(axes)
    return figure


def plot_power_balance_bars(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] = (18.0, 4.8),
):
    step_df, _ = _prepare_agent_power_balance_frame(
        rollout.step_df,
        controller_label=str(rollout.meta.get("controller", "unknown")),
    )

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    timestamps = step_df["timestamp"]
    width = 0.008

    positive_specs = [
        ("load_total", "Load", "#111827"),
        ("battery_charge_total", "Charge", "#dc2626"),
        ("grid_export_total", "Grid export", "#f59e0b"),
        ("pv_curtail_total", "Curtailment loss", "#fca5a5"),
    ]
    negative_specs = [
        ("pv_raw_total", "PV raw", "#16a34a"),
        ("grid_import_total", "Grid import", "#2563eb"),
        ("battery_discharge_total", "Discharge", "#7c3aed"),
    ]

    positive_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, label, color in positive_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        extra_kwargs = {"hatch": "//", "edgecolor": "#dc2626", "linewidth": 1.0} if column == "pv_curtail_total" else {}
        axis.bar(
            timestamps,
            values,
            width=width,
            bottom=positive_bottom,
            color=color,
            alpha=0.78,
            label=label,
            **extra_kwargs,
        )
        positive_bottom = positive_bottom + values

    negative_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, label, color in negative_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        axis.bar(timestamps, -values, width=width, bottom=negative_bottom, color=color, alpha=0.78, label=label)
        negative_bottom = negative_bottom - values

    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_title(f"Power Balance - {rollout.meta['controller']}")
    axis.set_ylabel("kW")
    axis.set_xlabel("Timestamp")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="upper right", ncol=4)
    figure.tight_layout()
    return figure


def plot_test_rollout(rollout: RolloutResult, *, figsize: tuple[float, float] | None = None):
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    if step_df.empty or agent_df.empty:
        raise ValueError("Rollout is empty; nothing to plot.")

    agent_profiles = list(rollout.meta["agent_profiles"])
    n_agents = len(agent_profiles)
    if figsize is None:
        figsize = (18.0, 2.8 * (3 + n_agents))

    figure, axes = plt.subplots(3 + n_agents, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    palette = ["#0f172a", "#2563eb", "#16a34a", "#ea580c", "#dc2626", "#7c3aed"]

    axes[0].plot(
        step_df["timestamp"],
        step_df[IMPORT_PRICE_COLUMN],
        color="#111827",
        linewidth=1.6,
        label="Actual",
    )
    axes[0].plot(
        step_df["timestamp"],
        step_df[IMPORT_PRICE_PRED_COLUMN],
        color="#dc2626",
        linewidth=1.4,
        linestyle="--",
        label="Forecast",
    )
    axes[0].set_ylabel("Import Price")
    axes[0].set_title(f"Test Rollout - {rollout.meta['controller']}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    for axis_idx, signal_name in enumerate(["pv", "load"], start=1):
        axis = axes[axis_idx]
        for agent_idx, profile in enumerate(agent_profiles):
            agent_frame = agent_df.loc[agent_df["agent_profile"] == profile]
            color = palette[agent_idx % len(palette)]
            axis.plot(
                agent_frame["timestamp"],
                agent_frame[signal_name],
                color=color,
                linewidth=1.4,
                label=f"{profile} actual",
            )
            axis.plot(
                agent_frame["timestamp"],
                agent_frame[f"{signal_name}_pred"],
                color=color,
                linewidth=1.2,
                linestyle="--",
                label=f"{profile} forecast",
            )
        axis.set_ylabel(signal_name.upper())
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=2)

    for agent_offset, profile in enumerate(agent_profiles, start=3):
        axis = axes[agent_offset]
        agent_frame = agent_df.loc[agent_df["agent_profile"] == profile]
        charge = np.clip(agent_frame["e_bat"].to_numpy(dtype=np.float32), 0.0, None)
        discharge = np.clip(agent_frame["e_bat"].to_numpy(dtype=np.float32), None, 0.0)
        axis.bar(agent_frame["timestamp"], charge, width=0.008, color="#dc2626", alpha=0.7, label="Charge")
        axis.bar(agent_frame["timestamp"], discharge, width=0.008, color="#2563eb", alpha=0.7, label="Discharge")
        axis.set_ylabel(f"{profile}\nP_bat")
        axis.grid(True, alpha=0.25)

        soc_axis = axis.twinx()
        soc_axis.plot(agent_frame["timestamp"], agent_frame["soc"], color="#111827", linewidth=1.2, label="SoC")
        soc_axis.set_ylabel("SoC")
        soc_axis.set_ylim(0.0, 1.0)

        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = soc_axis.get_legend_handles_labels()
        axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    axes[-1].set_xlabel("Timestamp")
    figure.tight_layout()
    return figure


def plot_test_voltage_profile(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] = (18.0, 4.8),
):
    grid_df = rollout.grid_df.copy()
    if grid_df.empty:
        raise ValueError("Rollout does not contain full-grid voltage traces.")

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    muted_color = "#cbd5e1"
    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]

    for bus_id, frame in grid_df.loc[~grid_df["is_agent_bus"]].groupby("bus_id"):
        axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=muted_color,
            linewidth=0.9,
            alpha=0.35,
            zorder=1,
        )

    for color_idx, bus_id in enumerate(rollout.meta["agent_bus_ids"]):
        frame = grid_df.loc[grid_df["bus_id"] == int(bus_id)]
        if frame.empty:
            continue
        axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=highlight_palette[color_idx % len(highlight_palette)],
            linewidth=2.1,
            alpha=0.95,
            label=f"Agent bus {bus_id}",
            zorder=3,
        )

    axis.axhline(float(rollout.meta["v_min_pu"]), color="#dc2626", linestyle="--", linewidth=1.1, label="V min")
    axis.axhline(float(rollout.meta["v_max_pu"]), color="#ea580c", linestyle="--", linewidth=1.1, label="V max")
    axis.set_title(f"Node Voltage Profile - {rollout.meta['controller']}")
    axis.set_ylabel("Voltage [p.u.]")
    axis.set_xlabel("Timestamp")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", ncol=2)
    figure.tight_layout()
    return figure


def plot_voltage_profile_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (20.0, max(4.0 * n_rows, 5.6)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    muted_color = COMPARE_PLOT_MUTED_COLOR
    highlight_palette = COMPARE_PLOT_AGENT_PALETTE
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize = _compare_plot_fontsizes()

    for axis_idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        grid_df = rollout.grid_df
        if grid_df.empty:
            raise ValueError(f"Rollout '{rollout.meta.get('controller', 'unknown')}' has no grid_df.")
        background_df = grid_df.loc[~grid_df["is_agent_bus"], ["timestamp", "bus_id", "vm_pu"]].sort_values(
            ["timestamp", "bus_id"]
        )
        if not background_df.empty:
            background_wide = background_df.pivot_table(
                index="timestamp",
                columns="bus_id",
                values="vm_pu",
                aggfunc="first",
            ).sort_index()
            if background_wide.shape[1] > 0:
                axis.plot(
                    background_wide.index.to_numpy(),
                    background_wide.to_numpy(dtype=np.float32),
                    color=muted_color,
                    linewidth=0.9,
                    alpha=0.35,
                    zorder=1,
                )
        for color_idx, bus_id in enumerate(rollout.meta["agent_bus_ids"]):
            frame = grid_df.loc[grid_df["bus_id"] == int(bus_id)]
            if frame.empty:
                continue
            axis.plot(
                frame["timestamp"],
                frame["vm_pu"],
                color=highlight_palette[color_idx % len(highlight_palette)],
                linewidth=2.1,
                alpha=0.95,
                label=f"Agent bus {bus_id}" if axis_idx == 0 else None,
                zorder=3,
            )
        axis.axhline(
            float(rollout.meta["v_min_pu"]),
            color="#dc2626",
            linestyle="--",
            linewidth=1.1,
            label="V min" if axis_idx == 0 else None,
        )
        axis.axhline(
            float(rollout.meta["v_max_pu"]),
            color="#ea580c",
            linestyle="--",
            linewidth=1.1,
            label="V max" if axis_idx == 0 else None,
        )
        axis.set_ylabel("V [p.u.]", fontsize=label_fontsize)
        axis.set_title(str(rollout.meta["controller"]), fontsize=title_fontsize)
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        if axis_idx == 0:
            axis.legend(loc="upper right", ncol=2, fontsize=legend_fontsize)

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
    figure.tight_layout()
    return figure


def plot_price_prediction_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] = (20.0, 5.0),
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize = _compare_plot_fontsizes()

    reference_rollout = next(
        (rollout for rollout in rollouts if str(rollout.meta.get("prediction_mode", "")) == NORMAL_PREDICTION_MODE),
        rollouts[0],
    )
    reference_step_df = reference_rollout.step_df.copy()
    required_columns = {"timestamp", IMPORT_PRICE_COLUMN, WHOLESALE_PRICE_PRED_COLUMN}
    if reference_step_df.empty or not required_columns.issubset(reference_step_df.columns):
        raise ValueError("Rollouts must contain timestamp, import_price, and wholesale_price_pred columns.")

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    axis.plot(
        reference_step_df["timestamp"],
        reference_step_df[IMPORT_PRICE_COLUMN],
        color="#111827",
        linewidth=1.8,
        label="Actual import price",
    )

    reference_price_markup = float(reference_rollout.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0))
    reference_pred = _resolve_adjusted_price_prediction(reference_step_df, price_markup=reference_price_markup)
    prediction_groups: list[dict[str, object]] = []
    for rollout in rollouts:
        if str(rollout.meta.get("prediction_mode", "")) != NORMAL_PREDICTION_MODE:
            continue
        step_df = rollout.step_df.copy()
        if step_df.empty or WHOLESALE_PRICE_PRED_COLUMN not in step_df.columns:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing wholesale_price_pred for compare plotting."
            )
        price_markup = float(rollout.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0))
        candidate_timestamps = pd.to_datetime(step_df["timestamp"]).to_numpy()
        candidate_pred = _resolve_adjusted_price_prediction(step_df, price_markup=price_markup)
        controller_name = str(rollout.meta.get("controller", "unknown"))
        matched_group = None
        for group in prediction_groups:
            group_timestamps = group["timestamps"]
            group_pred = group["pred"]
            if (
                candidate_timestamps.shape == group_timestamps.shape
                and np.array_equal(candidate_timestamps, group_timestamps)
                and candidate_pred.shape == group_pred.shape
                and np.allclose(candidate_pred, group_pred, equal_nan=True)
            ):
                matched_group = group
                break
        if matched_group is None:
            prediction_groups.append(
                {
                    "timestamps": candidate_timestamps,
                    "pred": candidate_pred,
                    "controllers": [controller_name],
                }
            )
        else:
            matched_group["controllers"].append(controller_name)

    if not prediction_groups:
        prediction_groups.append(
            {
                "timestamps": pd.to_datetime(reference_step_df["timestamp"]).to_numpy(),
                "pred": reference_pred,
                "controllers": [str(reference_rollout.meta.get("controller", "unknown"))],
            }
        )

    prediction_palette = ["#dc2626", "#ea580c", "#16a34a", "#7c3aed", "#0891b2"]
    for group_idx, group in enumerate(prediction_groups):
        controllers = list(dict.fromkeys(str(name) for name in group["controllers"]))
        if len(prediction_groups) == 1:
            label = "Predicted import price"
        elif len(controllers) == 1:
            label = f"Predicted import price ({controllers[0]})"
        else:
            label = f"Predicted import price ({controllers[0]} +{len(controllers) - 1})"
        axis.plot(
            group["timestamps"],
            group["pred"],
            color=prediction_palette[group_idx % len(prediction_palette)],
            linewidth=1.8,
            linestyle="--",
            label=label,
        )

    axis.set_title("Import Price And Derived Forecast", fontsize=title_fontsize)
    axis.set_ylabel("EUR/kWh", fontsize=label_fontsize)
    axis.set_xlabel("Timestamp", fontsize=label_fontsize)
    axis.grid(True, alpha=0.25)
    axis.tick_params(axis="both", labelsize=tick_fontsize)
    axis.legend(loc="upper right", ncol=2, fontsize=legend_fontsize)
    figure.tight_layout()
    return figure


def plot_net_load_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (20.0, max(4.2 * n_rows, 5.8)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize = _compare_plot_fontsizes()

    for axis_idx, (feeder_axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df = _prepare_compare_net_load_frame(
            rollout.step_df,
            controller_label=str(rollout.meta.get("controller", "unknown")),
        )
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["feeder_raw_net_load_kw"],
            color="#111827",
            linewidth=1.6,
            label="Feeder raw net load",
        )
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["feeder_effective_net_load_kw"],
            color="#16a34a",
            linewidth=1.4,
            linestyle="-.",
            label="Feeder post-curtail net load",
        )
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["feeder_post_action_net_load_kw"],
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
                label="Root net exchange",
            )
        elif "pp_root_p_kw" in step_df.columns:
            feeder_axis.plot(
                step_df["timestamp"],
                step_df["pp_root_p_kw"],
                color="#7c3aed",
                linewidth=1.7,
                label="Pandapower root exchange",
            )
        trafo_limit_kw = rollout.meta.get("trafo_limit_kw")
        if trafo_limit_kw is not None:
            trafo_limit_value = float(trafo_limit_kw)
            if np.isfinite(trafo_limit_value) and trafo_limit_value > 0.0:
                feeder_axis.axhline(
                    trafo_limit_value,
                    color="#dc2626",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"Transformer S-limit ref (+P view) ({trafo_limit_value:.1f} kW)" if axis_idx == 0 else None,
                )
                feeder_axis.axhline(
                    -trafo_limit_value,
                    color="#dc2626",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"Transformer S-limit ref (-P view) ({trafo_limit_value:.1f} kW)" if axis_idx == 0 else None,
                )
        feeder_axis.set_ylabel("kW", fontsize=label_fontsize)
        feeder_axis.set_title(f"{rollout.meta['controller']} - feeder total", fontsize=title_fontsize)
        feeder_axis.grid(True, alpha=0.25)
        feeder_axis.tick_params(axis="both", labelsize=tick_fontsize)
        if axis_idx == 0:
            feeder_axis.legend(loc="upper right", fontsize=legend_fontsize)

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
    figure.tight_layout()
    return figure


def plot_power_balance_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (20.0, max(4.0 * n_rows, 5.6)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize = _compare_plot_fontsizes()
    positive_specs = [
        ("load_total", "Load", "#111827"),
        ("battery_charge_total", "Charge", "#dc2626"),
        ("grid_export_total", "Grid export", "#f59e0b"),
        ("pv_curtail_total", "Curtailment loss", "#fca5a5"),
    ]
    negative_specs = [
        ("pv_raw_total", "PV raw", "#16a34a"),
        ("grid_import_total", "Grid import", "#2563eb"),
        ("battery_discharge_total", "Discharge", "#7c3aed"),
    ]

    for axis_idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df, max_abs_balance_residual_kw = _prepare_agent_power_balance_frame(
            rollout.step_df,
            controller_label=str(rollout.meta.get("controller", "unknown")),
        )

        timestamps = step_df["timestamp"].to_numpy()
        positive_bottom = np.zeros(len(step_df), dtype=np.float32)
        for column, label, color in positive_specs:
            values = step_df[column].to_numpy(dtype=np.float32)
            extra_kwargs = (
                {"hatch": "//", "edgecolor": "#dc2626", "linewidth": 1.0}
                if column == "pv_curtail_total"
                else {}
            )
            axis.bar(
                timestamps,
                values,
                width=0.008,
                bottom=positive_bottom,
                color=color,
                alpha=0.78,
                label=label if axis_idx == 0 else None,
                **extra_kwargs,
            )
            positive_bottom = positive_bottom + values

        negative_bottom = np.zeros(len(step_df), dtype=np.float32)
        for column, label, color in negative_specs:
            values = step_df[column].to_numpy(dtype=np.float32)
            axis.bar(
                timestamps,
                -values,
                width=0.008,
                bottom=negative_bottom,
                color=color,
                alpha=0.78,
                label=label if axis_idx == 0 else None,
            )
            negative_bottom = negative_bottom - values
        axis.axhline(0.0, color="#111827", linewidth=1.0)
        axis.set_ylabel("kW", fontsize=label_fontsize)
        axis.set_title(
            f"{rollout.meta['controller']} - agent-only balance (residual<={max_abs_balance_residual_kw:.3f} kW)",
            fontsize=title_fontsize,
        )
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        if axis_idx == 0:
            axis.legend(loc="upper right", ncol=4, fontsize=legend_fontsize)

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
    figure.tight_layout()
    return figure


def plot_battery_power_and_soc_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (20.0, max(4.2 * n_rows, 6.5)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    title_fontsize, label_fontsize, tick_fontsize, legend_fontsize = _compare_plot_fontsizes()
    discharge_color = "#dc2626"
    charge_color = "#2563eb"
    soc_agent_color = "#94a3b8"
    soc_mean_color = "#111827"
    zero_line_color = "#475569"

    for rollout_idx, rollout in enumerate(rollouts):
        power_axis = axes[rollout_idx]
        soc_axis = power_axis.twinx()
        step_df = rollout.step_df.copy()
        agent_df = rollout.agent_df.copy()
        if step_df.empty:
            raise ValueError(f"Rollout '{rollout.meta.get('controller', 'unknown')}' has no step_df.")
        if not {"battery_charge_total", "battery_discharge_total", "timestamp"}.issubset(step_df.columns):
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing battery power columns required for compare plotting."
            )
        step_df = step_df.copy()
        step_df["battery_net_power_kw"] = (
            step_df["battery_discharge_total"].to_numpy(dtype=np.float32)
            - step_df["battery_charge_total"].to_numpy(dtype=np.float32)
        )
        net_power = step_df["battery_net_power_kw"].to_numpy(dtype=np.float32)
        bar_colors = [discharge_color if value >= 0.0 else charge_color for value in net_power]
        power_axis.bar(
            step_df["timestamp"],
            net_power,
            width=0.008,
            color=bar_colors,
            alpha=0.82,
        )
        power_axis.axhline(0.0, color=zero_line_color, linewidth=1.0)
        power_axis.set_ylabel("Battery Power [kW]", fontsize=label_fontsize)
        power_axis.set_title(str(rollout.meta["controller"]), fontsize=title_fontsize)
        power_axis.grid(True, alpha=0.25)
        power_axis.tick_params(axis="both", labelsize=tick_fontsize)

        if agent_df.empty or "soc" not in agent_df.columns:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing agent_df.soc required for compare plotting."
            )
        group_columns = [column for column in ["episode_idx", "step", "timestamp"] if column in agent_df.columns]
        if not group_columns:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing grouping columns required to aggregate SoC."
            )
        agent_id_column = "agent_id" if "agent_id" in agent_df.columns else None
        if agent_id_column is None:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing agent_df.agent_id required for compare plotting."
            )
        soc_summary = (
            agent_df.groupby(group_columns, as_index=False)
            .agg(
                mean=("soc", "mean"),
            )
        )
        first_agent = True
        for _, agent_frame in agent_df.sort_values(group_columns + [agent_id_column]).groupby(agent_id_column, sort=True):
            soc_axis.plot(
                agent_frame["timestamp"],
                agent_frame["soc"],
                color=soc_agent_color,
                linewidth=1.1,
                alpha=0.45,
                label="Agent SoC" if first_agent and rollout_idx == 0 else None,
            )
            first_agent = False
        soc_axis.plot(
            soc_summary["timestamp"],
            soc_summary["mean"],
            color=soc_mean_color,
            linewidth=1.7,
            label="Mean SoC" if rollout_idx == 0 else None,
        )
        soc_axis.set_ylabel("SoC", fontsize=label_fontsize)
        soc_axis.set_ylim(0.0, 1.0)
        soc_axis.grid(True, alpha=0.25)
        soc_axis.tick_params(axis="both", labelsize=tick_fontsize)
        if rollout_idx == 0:
            legend_handles = [
                Patch(facecolor=discharge_color, alpha=0.82, label="Discharge (+)"),
                Patch(facecolor=charge_color, alpha=0.82, label="Charge (-)"),
            ]
            handles_2, labels_2 = soc_axis.get_legend_handles_labels()
            power_axis.legend(
                legend_handles + handles_2,
                ["Discharge (+)", "Charge (-)"] + labels_2,
                loc="upper right",
                ncol=2,
                fontsize=legend_fontsize,
            )

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
    figure.tight_layout()
    return figure


def plot_purchase_cost_comparison(cost_df: pd.DataFrame, *, figsize: tuple[float, float] = (10.0, 4.5)):
    if cost_df.empty:
        raise ValueError("cost_df is empty; nothing to plot.")
    figure, axis = plt.subplots(1, 1, figsize=figsize)
    ordered = cost_df.sort_values("purchase_cost_total", ascending=True)
    axis.bar(ordered["controller"], ordered["purchase_cost_total"], color=["#2563eb", "#16a34a", "#dc2626"])
    axis.set_ylabel("Purchase Cost")
    axis.set_title("Purchase Cost Comparison")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_operating_cost_comparison(cost_df: pd.DataFrame, *, figsize: tuple[float, float] = (10.0, 4.5)):
    return plot_purchase_cost_comparison(cost_df, figsize=figsize)


def plot_rollout_comparison_dashboard(
    metrics_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (16.0, 9.0),
):
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; nothing to plot.")

    figure, axes = plt.subplots(2, 3, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    controllers = metrics_df["controller"].astype(str).tolist()
    x = np.arange(len(controllers))
    bar_palette = ["#2563eb", "#16a34a", "#dc2626", "#ea580c", "#7c3aed", "#0891b2"]
    colors = [bar_palette[idx % len(bar_palette)] for idx in range(len(controllers))]

    metric_specs = [
        ("purchase_cost_total", "Purchase Cost"),
        ("export_subsidy_total", "Export Subsidy"),
        ("objective_total", "Objective Total"),
        ("voltage_violation_count", "Voltage Violation Count"),
        ("trafo_penalty_total", "Transformer Penalty"),
        ("line_penalty_total", "Line Penalty"),
    ]
    for axis, (column, title) in zip(axes, metric_specs, strict=False):
        axis.bar(x, metrics_df[column].astype(float).to_numpy(), color=colors)
        axis.set_xticks(x)
        axis.set_xticklabels(controllers, rotation=15, ha="right")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)

    figure.tight_layout()
    return figure

