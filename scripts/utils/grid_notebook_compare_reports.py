"""Compare/report helpers extracted from the grid notebook workflow module."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Mapping

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


def _extract_safety_summary(train_result: Mapping[str, object] | None) -> dict[str, object]:
    payload = dict(train_result or {})
    if isinstance(payload.get("safety_summary"), Mapping):
        return dict(payload["safety_summary"])
    nested_result = payload.get("train_result")
    if isinstance(nested_result, Mapping) and isinstance(nested_result.get("safety_summary"), Mapping):
        return dict(nested_result["safety_summary"])
    return {}


def _extract_trafo_penalty_weight(train_result: Mapping[str, object] | None) -> float | None:
    payload = dict(train_result or {})
    if isinstance(payload.get("experiment_controls"), Mapping):
        reward_controls = dict(payload["experiment_controls"].get("reward_controls", {}))
        if "w_trafo_pen" in reward_controls:
            return float(reward_controls["w_trafo_pen"])
    nested_result = payload.get("train_result")
    if isinstance(nested_result, Mapping) and isinstance(nested_result.get("experiment_controls"), Mapping):
        reward_controls = dict(nested_result["experiment_controls"].get("reward_controls", {}))
        if "w_trafo_pen" in reward_controls:
            return float(reward_controls["w_trafo_pen"])
    return None


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


def _safe_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_series = numerator.astype(np.float32)
    denominator_series = denominator.astype(np.float32)
    safe_denominator = denominator_series.where(denominator_series.abs() > 1e-6, np.nan)
    return numerator_series / safe_denominator


def _build_trafo_cause_hint(row: pd.Series) -> str:
    regime = str(row.get("dominant_regime", "mixed_or_balanced"))
    raw_export = max(float(row.get("raw_export_total", 0.0)), 0.0)
    raw_import = max(float(row.get("raw_import_total", 0.0)), 0.0)
    grid_export = max(float(row.get("grid_export_total", 0.0)), 0.0)
    grid_import = max(float(row.get("grid_import_total", 0.0)), 0.0)
    battery_charge = max(float(row.get("battery_charge_total", 0.0)), 0.0)
    battery_discharge = max(float(row.get("battery_discharge_total", 0.0)), 0.0)
    curtailment_ratio = row.get("curtailment_ratio", np.nan)
    projector_adjustment = row.get("projector_adjustment_kw_total", np.nan)

    if regime == "reverse_flow_export":
        stress_kw = max(raw_export, 1.0)
        adjustment_ratio = float(projector_adjustment / stress_kw) if pd.notna(projector_adjustment) else np.nan
        if pd.notna(projector_adjustment) and adjustment_ratio >= 0.20 and grid_export > 0.25 * raw_export:
            return "PV reverse flow dominates; projection changed actions but residual export stayed high."
        if (
            (pd.isna(projector_adjustment) or adjustment_ratio < 0.05)
            and (pd.isna(curtailment_ratio) or float(curtailment_ratio) < 0.10)
            and battery_charge < 0.10 * stress_kw
        ):
            return "PV reverse flow dominates; projection barely added charging or curtailment."
        return "PV reverse flow dominates; check reverse-flow linearization and available flexibility."

    if regime == "grid_import_overload":
        stress_kw = max(raw_import, 1.0)
        adjustment_ratio = float(projector_adjustment / stress_kw) if pd.notna(projector_adjustment) else np.nan
        if pd.notna(projector_adjustment) and adjustment_ratio >= 0.20 and grid_import > 0.25 * raw_import:
            return "Load-driven import dominates; projection changed actions but residual import stayed high."
        if (pd.isna(projector_adjustment) or adjustment_ratio < 0.05) and battery_discharge < 0.10 * stress_kw:
            return "Load-driven import dominates; projection barely added discharge support."
        return "Load-driven import dominates; likely limited discharge headroom or linearization error."

    return "Transformer stress is mixed; inspect export/import swings and action gaps together."


def _prepare_trafo_diagnostic_frame(
    rollout: RolloutResult,
    *,
    train_result: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, str]:
    step_df = rollout.step_df.copy()
    if step_df.empty:
        raise ValueError("Rollout is empty; no trafo diagnostics are available.")

    required_columns = {
        "episode_idx",
        "step",
        "timestamp",
        "base_net_load_total",
        "net_load_total",
        "load_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
        "battery_charge_total",
        "battery_discharge_total",
        "trafo_penalty_total",
    }
    missing = sorted(required_columns.difference(step_df.columns))
    if missing:
        raise ValueError(f"Rollout step_df is missing required trafo diagnostic columns: {missing}")

    if "pv_raw_total" not in step_df.columns:
        if "pv_effective_total" not in step_df.columns:
            raise ValueError("Rollout step_df must contain either pv_raw_total or pv_effective_total.")
        step_df["pv_raw_total"] = (
            step_df["pv_effective_total"].to_numpy(dtype=np.float32)
            + step_df["pv_curtail_total"].to_numpy(dtype=np.float32)
        )

    if "psi_trafo_raw" not in step_df.columns:
        trafo_weight = _extract_trafo_penalty_weight(train_result)
        if trafo_weight is not None and abs(float(trafo_weight)) > 1e-6:
            step_df["psi_trafo_raw"] = step_df["trafo_penalty_total"].astype(np.float32) / float(trafo_weight)

    step_df["raw_export_total"] = np.clip(
        -step_df["base_net_load_total"].to_numpy(dtype=np.float32),
        0.0,
        None,
    )
    step_df["raw_import_total"] = np.clip(
        step_df["base_net_load_total"].to_numpy(dtype=np.float32),
        0.0,
        None,
    )
    step_df["net_load_shift_from_raw_kw"] = (
        step_df["net_load_total"].astype(np.float32) - step_df["base_net_load_total"].astype(np.float32)
    )
    step_df["curtailment_ratio"] = _safe_ratio_series(step_df["pv_curtail_total"], step_df["pv_raw_total"])
    step_df["charge_to_raw_export_ratio"] = _safe_ratio_series(step_df["battery_charge_total"], step_df["raw_export_total"])
    step_df["discharge_to_raw_import_ratio"] = _safe_ratio_series(
        step_df["battery_discharge_total"],
        step_df["raw_import_total"],
    )

    reverse_flow_mask = (
        (step_df["raw_export_total"] > step_df["raw_import_total"])
        & (step_df["grid_export_total"] > 1e-6)
        & (step_df["net_load_total"] < 0.0)
    )
    import_mask = (
        (step_df["raw_import_total"] >= step_df["raw_export_total"])
        & (step_df["grid_import_total"] > 1e-6)
        & (step_df["net_load_total"] >= 0.0)
    )
    step_df["dominant_regime"] = np.select(
        [reverse_flow_mask, import_mask],
        ["reverse_flow_export", "grid_import_overload"],
        default="mixed_or_balanced",
    )

    step_df["mitigation_toward_safe_direction_kw"] = np.where(
        reverse_flow_mask,
        step_df["pv_curtail_total"].astype(np.float32) + step_df["battery_charge_total"].astype(np.float32),
        np.where(
            import_mask,
            step_df["battery_discharge_total"].astype(np.float32),
            np.nan,
        ),
    )
    step_df["mitigation_against_safe_direction_kw"] = np.where(
        reverse_flow_mask,
        step_df["battery_discharge_total"].astype(np.float32),
        np.where(
            import_mask,
            step_df["battery_charge_total"].astype(np.float32) + step_df["pv_curtail_total"].astype(np.float32),
            np.nan,
        ),
    )

    loading_limit_pct = rollout.meta.get("trafo_loading_limit_pct", rollout.meta.get("loading_limit_pct", np.nan))
    if "trafo_loading_pct_max" not in step_df.columns:
        step_df["trafo_loading_pct_max"] = np.nan
    if pd.notna(loading_limit_pct):
        step_df["trafo_over_limit_pct"] = step_df["trafo_loading_pct_max"].astype(np.float32) - float(loading_limit_pct)
    else:
        step_df["trafo_over_limit_pct"] = np.nan

    rank_metric = "psi_trafo_raw" if "psi_trafo_raw" in step_df.columns else "trafo_penalty_total"
    step_df["rank_score"] = step_df[rank_metric].astype(np.float32)
    step_df["cause_hint"] = step_df.apply(_build_trafo_cause_hint, axis=1)
    ordered = step_df.sort_values(
        ["rank_score", "episode_idx", "step"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return ordered, rank_metric


def build_trafo_diagnostic_table(
    rollout: RolloutResult,
    *,
    train_result: Mapping[str, object] | None = None,
    top_k: int = 12,
) -> pd.DataFrame:
    ordered, rank_metric = _prepare_trafo_diagnostic_frame(rollout, train_result=train_result)
    selected = ordered.head(max(1, int(top_k))).copy()
    selected["rank_metric"] = rank_metric

    preferred_columns = [
        "episode_idx",
        "step",
        "timestamp",
        "rank_metric",
        "rank_score",
        "trafo_penalty_total",
        "psi_trafo_raw",
        "trafo_loading_pct_max",
        "trafo_over_limit_pct",
        "n_trafo_violations",
        "dominant_regime",
        "raw_export_total",
        "raw_import_total",
        "net_load_total",
        "grid_export_total",
        "grid_import_total",
        "pv_raw_total",
        "pv_curtail_total",
        "curtailment_ratio",
        "battery_charge_total",
        "battery_discharge_total",
        "mitigation_toward_safe_direction_kw",
        "mitigation_against_safe_direction_kw",
        "projector_adjustment_kw_total",
        "battery_request_gap_kw_total",
        "pv_curtail_request_gap_kw_total",
        "controller_action_gap_total",
        "voltage_penalty_total",
        "line_penalty_total",
        "cause_hint",
    ]
    existing_columns = [column for column in preferred_columns if column in selected.columns]
    return selected.loc[:, existing_columns]


def summarize_trafo_diagnostics(
    rollout: RolloutResult,
    *,
    train_result: Mapping[str, object] | None = None,
    top_k: int = 12,
) -> pd.Series:
    ordered, rank_metric = _prepare_trafo_diagnostic_frame(rollout, train_result=train_result)
    selected = ordered.head(max(1, int(top_k))).copy()
    if selected.empty:
        raise ValueError("Rollout is empty; no trafo diagnostics are available.")

    dominant_regime = (
        str(selected["dominant_regime"].value_counts().idxmax())
        if "dominant_regime" in selected.columns and not selected["dominant_regime"].empty
        else "mixed_or_balanced"
    )
    safety_summary = _extract_safety_summary(train_result)
    pre_violation = safety_summary.get("mean_pre_projection_violation", np.nan)
    post_violation = safety_summary.get("mean_post_projection_violation", np.nan)
    if pd.notna(pre_violation) and abs(float(pre_violation)) > 1e-6 and pd.notna(post_violation):
        violation_reduction_pct = 100.0 * (1.0 - float(post_violation) / float(pre_violation))
    else:
        violation_reduction_pct = np.nan

    projected_fraction = safety_summary.get("projected_fraction", np.nan)
    mean_adjustment_kw = (
        float(selected["projector_adjustment_kw_total"].mean())
        if "projector_adjustment_kw_total" in selected.columns
        else float("nan")
    )
    export_share = float((selected["dominant_regime"] == "reverse_flow_export").mean())
    import_share = float((selected["dominant_regime"] == "grid_import_overload").mean())

    if dominant_regime == "reverse_flow_export":
        if pd.notna(projected_fraction) and float(projected_fraction) < 0.10:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven, and the projector rarely changes the action."
        elif pd.notna(violation_reduction_pct) and float(violation_reduction_pct) >= 50.0:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven; the projector reduces linearized violation, but real trafo stress remains high."
        elif pd.notna(mean_adjustment_kw) and float(mean_adjustment_kw) < 0.25:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven, but the projector response stays small."
        else:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven; inspect reverse-flow sensitivity quality and flexibility limits."
    elif dominant_regime == "grid_import_overload":
        if pd.notna(projected_fraction) and float(projected_fraction) < 0.10:
            diagnosis = "Top trafo steps are mostly import driven, and the projector rarely changes the action."
        elif pd.notna(violation_reduction_pct) and float(violation_reduction_pct) >= 50.0:
            diagnosis = "Top trafo steps are mostly import driven; the projector reduces linearized violation, but real trafo stress remains high."
        else:
            diagnosis = "Top trafo steps are mostly import driven; inspect discharge headroom and linearization quality."
    else:
        diagnosis = "Top trafo steps mix export and import stress; inspect the detailed rows for regime switching."

    return pd.Series(
        {
            "controller": str(rollout.meta.get("controller", "unknown")),
            "rank_metric": rank_metric,
            "analyzed_steps": int(len(ordered)),
            "top_k_steps": int(len(selected)),
            "steps_with_trafo_penalty": int((ordered["trafo_penalty_total"] > 0.0).sum()),
            "trafo_penalty_total": float(ordered["trafo_penalty_total"].sum()),
            "top_k_mean_trafo_penalty": float(selected["trafo_penalty_total"].mean()),
            "top_k_max_trafo_penalty": float(selected["trafo_penalty_total"].max()),
            "dominant_regime_top_k": dominant_regime,
            "reverse_flow_export_share_top_k": export_share,
            "grid_import_share_top_k": import_share,
            "top_k_mean_raw_export_kw": float(selected["raw_export_total"].mean()),
            "top_k_mean_raw_import_kw": float(selected["raw_import_total"].mean()),
            "top_k_mean_grid_export_kw": float(selected["grid_export_total"].mean()),
            "top_k_mean_grid_import_kw": float(selected["grid_import_total"].mean()),
            "top_k_mean_pv_curtail_total": float(selected["pv_curtail_total"].mean()),
            "top_k_mean_battery_charge_total": float(selected["battery_charge_total"].mean()),
            "top_k_mean_battery_discharge_total": float(selected["battery_discharge_total"].mean()),
            "top_k_mean_projector_adjustment_kw_total": mean_adjustment_kw,
            "top_k_mean_trafo_loading_pct_max": (
                float(selected["trafo_loading_pct_max"].mean())
                if "trafo_loading_pct_max" in selected.columns
                else float("nan")
            ),
            "projector_enabled": bool(safety_summary.get("enabled", False)) if safety_summary else False,
            "projected_fraction": float(projected_fraction) if pd.notna(projected_fraction) else float("nan"),
            "mean_pre_projection_violation": float(pre_violation) if pd.notna(pre_violation) else float("nan"),
            "mean_post_projection_violation": float(post_violation) if pd.notna(post_violation) else float("nan"),
            "projector_violation_reduction_pct": (
                float(violation_reduction_pct) if pd.notna(violation_reduction_pct) else float("nan")
            ),
            "diagnosis": diagnosis,
        }
    )


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
