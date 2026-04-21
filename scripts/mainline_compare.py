from __future__ import annotations
from html import escape
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN, IMPORT_PRICE_PRED_COLUMN
if TYPE_CHECKING:
    from scripts.utils.grid_notebook_workflow import RolloutResult

COMPARE_METRIC_COLUMNS = [
    "controller",
    "soc_mode",
    "purchase_cost_total",
    "purchase_cost_total_eur",
    "export_subsidy_total",
    "export_subsidy_total_eur",
    "total_cost_eur",
    "objective_total",
    "soc_penalty_total",
    "voltage_penalty_total",
    "trafo_penalty_total",
    "line_penalty_total",
    "voltage_violation_count",
    "price_mae",
    "load_mae",
    "pv_mae",
    "voltage_violation_steps",
    "voltage_violation_bus_points",
    "min_vm_pu",
    "max_vm_pu",
    "v_min_pu",
    "v_max_pu",
    "voltage_step_delta_p95_pu",
    "voltage_step_delta_max_pu",
    "voltage_spread_mean_pu",
    "voltage_spread_max_pu",
    "trafo_overload_steps",
    "trafo_loading_max_pct",
    "feeder_netload_ramp_mean_abs_kw",
    "feeder_netload_ramp_max_kw",
    "battery_net_power_kw_mean_abs",
    "returned_primary_objective_eur",
    "high_budget_refinement_warn",
    "formulation_tightening_required",
    "physics_refinement_status",
    "missing_optional_compare_fields",
]
ECONOMIC_TABLE_COLUMNS = [
    "controller",
    "purchase_cost_total_eur",
    "export_subsidy_total_eur",
    "total_cost_eur",
]
SAFETY_TABLE_COLUMNS = [
    "controller",
    "voltage_violation_steps",
    "voltage_violation_bus_points",
    "voltage_step_delta_p95_pu",
    "voltage_step_delta_max_pu",
    "voltage_spread_mean_pu",
    "voltage_spread_max_pu",
    "trafo_overload_steps",
    "trafo_loading_max_pct",
    "feeder_netload_ramp_mean_abs_kw",
    "feeder_netload_ramp_max_kw",
]
def _step_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ["episode_idx", "step"] if column in frame.columns]
    if columns:
        return columns
    if "timestamp" in frame.columns:
        return ["timestamp"]
    return []

def _sort_columns(frame: pd.DataFrame, *preferred: str) -> list[str]:
    return [column for column in preferred if column in frame.columns]

def _summarize_voltage_step_delta(grid_df: pd.DataFrame) -> tuple[float, float]:
    if grid_df.empty or "vm_pu" not in grid_df.columns or "bus_id" not in grid_df.columns:
        return float("nan"), float("nan")
    sort_columns = _sort_columns(grid_df, "episode_idx", "bus_id", "step", "timestamp")
    group_columns = [column for column in ["episode_idx", "bus_id"] if column in grid_df.columns]
    if "bus_id" not in group_columns:
        group_columns.append("bus_id")
    ordered = grid_df.sort_values(sort_columns) if sort_columns else grid_df.copy()
    step_delta = ordered.groupby(group_columns, sort=False)["vm_pu"].diff().abs()
    finite = step_delta.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 95.0)), float(np.max(finite))

def _summarize_voltage_spread(grid_df: pd.DataFrame) -> tuple[float, float]:
    if grid_df.empty or "vm_pu" not in grid_df.columns:
        return float("nan"), float("nan")
    group_columns = _step_group_columns(grid_df)
    if not group_columns:
        return float("nan"), float("nan")
    spread = grid_df.groupby(group_columns, sort=False)["vm_pu"].agg(lambda values: float(np.max(values) - np.min(values)))
    values = spread.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.max(values))

def _summarize_grouped_ramp(step_df: pd.DataFrame, column: str) -> tuple[float, float]:
    if step_df.empty or column not in step_df.columns:
        return float("nan"), float("nan")
    sort_columns = _sort_columns(step_df, "episode_idx", "step", "timestamp")
    ordered = step_df.sort_values(sort_columns) if sort_columns else step_df.copy()
    group_columns = [column_name for column_name in ["episode_idx"] if column_name in ordered.columns]
    if group_columns:
        ramp = ordered.groupby(group_columns, sort=False)[column].diff().abs()
    else:
        ramp = ordered[column].diff().abs()
    values = ramp.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.max(values))

def _collect_compare_optional_missing_fields(rollout: RolloutResult) -> list[str]:
    missing: list[str] = []
    if "trafo_loading_pct_max" not in rollout.step_df.columns:
        missing.append("step_df.trafo_loading_pct_max")
    if "n_trafo_violations" not in rollout.step_df.columns:
        missing.append("step_df.n_trafo_violations")
    if "soc" not in rollout.agent_df.columns:
        missing.append("agent_df.soc")
    return missing

def summarize_rollout_metrics(rollout: RolloutResult) -> dict[str, object]:
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    summary_df = rollout.summary.copy()
    purchase_cost_total = (
        float(step_df["purchase_cost_total"].sum())
        if "purchase_cost_total" in step_df.columns
        else float(rollout.meta.get("agent_purchase_cost_eur", float("nan")))
    )
    export_subsidy_total = (
        float(step_df["export_subsidy_total"].sum())
        if "export_subsidy_total" in step_df.columns
        else float(rollout.meta.get("agent_export_subsidy_eur", float("nan")))
    )
    total_cost_eur = (
        float(purchase_cost_total - export_subsidy_total)
        if np.isfinite(purchase_cost_total) and np.isfinite(export_subsidy_total)
        else float("nan")
    )
    objective_total = float(step_df["objective_total"].sum()) if "objective_total" in step_df.columns else 0.0
    soc_penalty_total = float(step_df["soc_penalty_total"].sum()) if "soc_penalty_total" in step_df.columns else 0.0
    voltage_penalty_total = (
        float(step_df["voltage_penalty_total"].sum()) if "voltage_penalty_total" in step_df.columns else 0.0
    )
    line_penalty_total = float(step_df["line_penalty_total"].sum()) if "line_penalty_total" in step_df.columns else 0.0
    trafo_penalty_total = (
        float(step_df["trafo_penalty_total"].sum()) if "trafo_penalty_total" in step_df.columns else 0.0
    )
    price_mae = (
        float(np.abs(step_df[IMPORT_PRICE_COLUMN] - step_df[IMPORT_PRICE_PRED_COLUMN]).mean())
        if not step_df.empty and {IMPORT_PRICE_COLUMN, IMPORT_PRICE_PRED_COLUMN}.issubset(step_df.columns)
        else float("nan")
    )
    load_mae = float(np.abs(agent_df["load"] - agent_df["load_pred"]).mean()) if not agent_df.empty else float("nan")
    pv_mae = float(np.abs(agent_df["pv"] - agent_df["pv_pred"]).mean()) if not agent_df.empty else float("nan")
    v_min = float(rollout.meta.get("v_min_pu", np.nan))
    v_max = float(rollout.meta.get("v_max_pu", np.nan))
    if grid_df.empty or not np.isfinite(v_min) or not np.isfinite(v_max):
        voltage_violation_bus_points = float("nan")
        voltage_violation_steps = float("nan")
        min_vm_pu = float("nan")
        max_vm_pu = float("nan")
    else:
        violation_mask = (grid_df["vm_pu"] < v_min) | (grid_df["vm_pu"] > v_max)
        voltage_violation_bus_points = int(violation_mask.sum())
        voltage_violation_steps = int(grid_df.loc[violation_mask, ["episode_idx", "step"]].drop_duplicates().shape[0])
        min_vm_pu = float(grid_df["vm_pu"].min())
        max_vm_pu = float(grid_df["vm_pu"].max())

    voltage_step_delta_p95_pu, voltage_step_delta_max_pu = _summarize_voltage_step_delta(grid_df)
    voltage_spread_mean_pu, voltage_spread_max_pu = _summarize_voltage_spread(grid_df)
    feeder_netload_ramp_mean_abs_kw, feeder_netload_ramp_max_kw = _summarize_grouped_ramp(
        step_df,
        "feeder_post_action_net_load_kw",
    )
    trafo_loading_limit_pct = float(
        rollout.meta.get(
            "trafo_loading_limit_pct",
            rollout.meta.get("loading_limit_pct", np.nan),
        )
    )
    trafo_loading_max_pct = (
        float(step_df["trafo_loading_pct_max"].max()) if "trafo_loading_pct_max" in step_df.columns else float("nan")
    )
    if "n_trafo_violations" in step_df.columns:
        trafo_overload_steps = int((step_df["n_trafo_violations"].astype(float) > 0.0).sum())
    elif np.isfinite(trafo_loading_max_pct) and np.isfinite(trafo_loading_limit_pct) and "trafo_loading_pct_max" in step_df.columns:
        trafo_overload_steps = int((step_df["trafo_loading_pct_max"].astype(float) > trafo_loading_limit_pct).sum())
    else:
        trafo_overload_steps = float("nan")
    battery_net_power_kw_mean_abs = (
        float(np.mean(np.abs(step_df["battery_discharge_total"].astype(float) - step_df["battery_charge_total"].astype(float))))
        if {"battery_charge_total", "battery_discharge_total"}.issubset(step_df.columns)
        else float("nan")
    )
    missing_optional_fields = _collect_compare_optional_missing_fields(rollout)
    metrics = {
        "controller": str(rollout.meta.get("controller", "unknown")),
        "soc_mode": str(rollout.meta.get("soc_mode", "reset")),
        "purchase_cost_total": purchase_cost_total,
        "purchase_cost_total_eur": purchase_cost_total,
        "export_subsidy_total": export_subsidy_total,
        "export_subsidy_total_eur": export_subsidy_total,
        "total_cost_eur": total_cost_eur,
        "objective_total": objective_total,
        "soc_penalty_total": soc_penalty_total,
        "voltage_penalty_total": voltage_penalty_total,
        "trafo_penalty_total": trafo_penalty_total,
        "line_penalty_total": line_penalty_total,
        "voltage_violation_count": voltage_violation_bus_points,
        "price_mae": price_mae,
        "load_mae": load_mae,
        "pv_mae": pv_mae,
        "voltage_violation_steps": voltage_violation_steps,
        "voltage_violation_bus_points": voltage_violation_bus_points,
        "min_vm_pu": min_vm_pu,
        "max_vm_pu": max_vm_pu,
        "v_min_pu": v_min,
        "v_max_pu": v_max,
        "voltage_step_delta_p95_pu": voltage_step_delta_p95_pu,
        "voltage_step_delta_max_pu": voltage_step_delta_max_pu,
        "voltage_spread_mean_pu": voltage_spread_mean_pu,
        "voltage_spread_max_pu": voltage_spread_max_pu,
        "trafo_overload_steps": trafo_overload_steps,
        "trafo_loading_max_pct": trafo_loading_max_pct,
        "feeder_netload_ramp_mean_abs_kw": feeder_netload_ramp_mean_abs_kw,
        "feeder_netload_ramp_max_kw": feeder_netload_ramp_max_kw,
        "battery_net_power_kw_mean_abs": battery_net_power_kw_mean_abs,
        "returned_primary_objective_eur": float(rollout.meta.get("returned_primary_objective_eur", np.nan)),
        "high_budget_refinement_warn": bool(rollout.meta.get("high_budget_refinement_warn", False)),
        "formulation_tightening_required": bool(rollout.meta.get("formulation_tightening_required", False)),
        "physics_refinement_status": str(rollout.meta.get("physics_refinement_status", "")),
        "missing_optional_compare_fields": ", ".join(missing_optional_fields),
    }
    if not summary_df.empty:
        for row in summary_df.itertuples(index=False):
            profile = str(getattr(row, "agent_profile", "unknown")).strip().lower()
            safe_profile = "".join(ch if ch.isalnum() else "_" for ch in profile).strip("_") or "unknown"
            metrics[f"purchase_cost_{safe_profile}"] = float(getattr(row, "purchase_cost"))
    return metrics

def compare_rollout_metrics(*rollouts: RolloutResult) -> pd.DataFrame:
    rows = [summarize_rollout_metrics(rollout) for rollout in rollouts]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=COMPARE_METRIC_COLUMNS)

def build_compare_economic_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=ECONOMIC_TABLE_COLUMNS)
    required_columns = ECONOMIC_TABLE_COLUMNS[1:]
    if metrics_df.loc[:, required_columns].isna().any().any():
        incomplete = metrics_df.loc[
            metrics_df.loc[:, required_columns].isna().any(axis=1),
            "controller",
        ].astype(str).tolist()
        raise ValueError(
            "Compare economic table requires final-dispatch purchase/export/total cost fields. "
            f"Missing values detected for: {incomplete}"
        )
    return metrics_df.loc[:, ECONOMIC_TABLE_COLUMNS].copy()

def build_compare_safety_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=SAFETY_TABLE_COLUMNS)
    return metrics_df.loc[:, SAFETY_TABLE_COLUMNS].copy()

def build_compare_warning_banner(*rollouts: RolloutResult):
    try:
        from IPython.display import HTML as _HTML  # type: ignore
    except ModuleNotFoundError:

        class _HTML(str):
            @property
            def data(self) -> str:
                return str(self)

            def _repr_html_(self) -> str:
                return str(self)

    items: list[str] = []
    for rollout in rollouts:
        controller = escape(str(rollout.meta.get("controller", "unknown")))
        if bool(rollout.meta.get("high_budget_refinement_warn", False)):
            items.append(
                f"<li><strong>{controller}</strong>: high-budget MISOCP refinement was used; "
                "compare economics are based on the returned/final dispatch, so the economic comparison should be interpreted with care.</li>"
            )
        if bool(rollout.meta.get("formulation_tightening_required", False)):
            items.append(
                f"<li><strong>{controller}</strong>: formulation tightening is still required; "
                "safety and economic conclusions may be unstable.</li>"
            )
        missing_optional_fields = _collect_compare_optional_missing_fields(rollout)
        if missing_optional_fields:
            joined = ", ".join(escape(field_name) for field_name in missing_optional_fields)
            items.append(
                f"<li><strong>{controller}</strong>: optional compare fields missing "
                f"({joined}); affected table cells will show N/A.</li>"
            )

    if not items:
        html = (
            "<div style='padding:10px 12px;border:1px solid #86efac;background:#f0fdf4;color:#166534;"
            "border-radius:8px;'>Compare warnings: none.</div>"
        )
        return _HTML(html)

    html = (
        "<div style='padding:10px 12px;border:1px solid #facc15;background:#fefce8;color:#854d0e;border-radius:8px;'>"
        "<strong>Compare warnings</strong><ul style='margin:8px 0 0 18px;'>"
        + "".join(items)
        + "</ul></div>"
    )
    return _HTML(html)
