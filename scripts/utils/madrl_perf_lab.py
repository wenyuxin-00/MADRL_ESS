"""Helpers for standalone MADRL performance-lab experiments."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd


_NON_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


def merge_control_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deep-merge notebook control dictionaries without mutating the inputs."""
    merged = deepcopy(dict(base))
    for key, value in dict(overrides or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_control_overrides(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def slugify_candidate_token(value: str, *, default: str = "candidate") -> str:
    token = _NON_TOKEN_PATTERN.sub("_", str(value).strip().lower()).strip("_")
    return token or default


def build_candidate_experiment_name(experiment_name_base: str, candidate_token: str) -> str:
    base_token = slugify_candidate_token(experiment_name_base, default="grid_mainline_perf_lab")
    candidate_part = slugify_candidate_token(candidate_token, default="baseline")
    return f"{base_token}_{candidate_part}"


def summarize_rollout_metrics(rollout) -> dict[str, Any]:
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    summary_df = rollout.summary.copy()

    total_operating_cost = float(agent_df["operating_cost"].sum()) if not agent_df.empty else 0.0
    price_mae = (
        float(np.abs(step_df["price"] - step_df["price_pred"]).mean())
        if not step_df.empty
        else float("nan")
    )
    load_mae = (
        float(np.abs(agent_df["load"] - agent_df["load_pred"]).mean())
        if not agent_df.empty
        else float("nan")
    )
    pv_mae = (
        float(np.abs(agent_df["pv"] - agent_df["pv_pred"]).mean())
        if not agent_df.empty
        else float("nan")
    )

    v_min = float(rollout.meta.get("v_min_pu", np.nan))
    v_max = float(rollout.meta.get("v_max_pu", np.nan))
    if grid_df.empty or not np.isfinite(v_min) or not np.isfinite(v_max):
        voltage_violation_bus_points = 0
        voltage_violation_steps = 0
        min_vm_pu = float("nan")
        max_vm_pu = float("nan")
    else:
        violation_mask = (grid_df["vm_pu"] < v_min) | (grid_df["vm_pu"] > v_max)
        voltage_violation_bus_points = int(violation_mask.sum())
        voltage_violation_steps = int(
            grid_df.loc[violation_mask, ["episode_idx", "step"]].drop_duplicates().shape[0]
        )
        min_vm_pu = float(grid_df["vm_pu"].min())
        max_vm_pu = float(grid_df["vm_pu"].max())

    metrics: dict[str, Any] = {
        "total_operating_cost": total_operating_cost,
        "price_mae": price_mae,
        "load_mae": load_mae,
        "pv_mae": pv_mae,
        "voltage_violation_bus_points": voltage_violation_bus_points,
        "voltage_violation_steps": voltage_violation_steps,
        "min_vm_pu": min_vm_pu,
        "max_vm_pu": max_vm_pu,
    }
    if not summary_df.empty:
        for row in summary_df.itertuples(index=False):
            profile = slugify_candidate_token(getattr(row, "agent_profile", "unknown"), default="unknown")
            metrics[f"operating_cost_{profile}"] = float(getattr(row, "operating_cost"))
    return metrics


def build_perf_leaderboard(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    if "candidate_name" in frame.columns:
        frame = frame.sort_values("candidate_name").reset_index(drop=True)
    if "is_baseline" in frame.columns:
        frame = frame.sort_values(["is_baseline", "candidate_name"], ascending=[False, True]).reset_index(drop=True)

    if "candidate_name" in frame.columns and "total_operating_cost" in frame.columns:
        baseline_rows = frame.loc[frame["candidate_name"] == "baseline"]
        if not baseline_rows.empty:
            baseline = baseline_rows.iloc[0]
            baseline_cost = float(baseline["total_operating_cost"])
            baseline_wall = float(baseline.get("total_wall_time_s", np.nan))
            frame["delta_operating_cost_vs_baseline"] = frame["total_operating_cost"].astype(float) - baseline_cost
            if np.isfinite(baseline_wall):
                frame["delta_wall_time_s_vs_baseline"] = frame["total_wall_time_s"].astype(float) - baseline_wall

    sort_columns = [column for column in ("total_operating_cost", "total_wall_time_s", "steps_per_sec") if column in frame.columns]
    ascending = [True, True, False][: len(sort_columns)]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    return frame


def recommend_perf_candidate(
    leaderboard: pd.DataFrame,
    *,
    baseline_name: str = "baseline",
) -> dict[str, Any]:
    if leaderboard.empty:
        raise ValueError("leaderboard is empty.")

    candidates = leaderboard.copy()
    if "status" in candidates.columns:
        ok_candidates = candidates.loc[candidates["status"] == "ok"].copy()
        if not ok_candidates.empty:
            candidates = ok_candidates

    baseline_rows = candidates.loc[candidates["candidate_name"] == baseline_name]
    if baseline_rows.empty:
        ranked = candidates.sort_values(
            ["total_operating_cost", "total_wall_time_s", "steps_per_sec"],
            ascending=[True, True, False],
        )
        winner = ranked.iloc[0]
        return {
            "recommended_candidate": str(winner["candidate_name"]),
            "reason": "No baseline row was present, so the leaderboard winner was selected directly.",
            "improves_over_baseline": None,
            "row": winner.to_dict(),
        }

    baseline = baseline_rows.iloc[0]
    baseline_cost = float(baseline["total_operating_cost"])
    improving = candidates.loc[candidates["total_operating_cost"].astype(float) <= baseline_cost].copy()
    ranked = improving.sort_values(
        ["total_operating_cost", "total_wall_time_s", "steps_per_sec"],
        ascending=[True, True, False],
    )
    winner = ranked.iloc[0] if not ranked.empty else baseline
    winner_name = str(winner["candidate_name"])
    if winner_name == baseline_name:
        reason = "Baseline remained the best short-run candidate when ranking test operating cost first and wall-clock time second."
    else:
        reason = (
            f"{winner_name} matched or improved baseline test operating cost and achieved the best cost-first ranking "
            "among the short-run candidates."
        )
    return {
        "recommended_candidate": winner_name,
        "reason": reason,
        "improves_over_baseline": bool(winner_name != baseline_name),
        "row": winner.to_dict(),
    }


__all__ = [
    "build_candidate_experiment_name",
    "build_perf_leaderboard",
    "merge_control_overrides",
    "recommend_perf_candidate",
    "slugify_candidate_token",
    "summarize_rollout_metrics",
]
