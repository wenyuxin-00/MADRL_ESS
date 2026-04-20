"""Pure helpers for the MISOCP global-oracle notebook."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.utils.price_protocol import (
    IMPORT_PRICE_COLUMN,
    IMPORT_PRICE_MARKUP_KEY,
    IMPORT_PRICE_PRED_COLUMN,
    PRICE_PROTOCOL_VERSION,
    WHOLESALE_PRICE_PRED_COLUMN,
    WHOLESALE_PRICE_SEQ_FIELD,
    derive_import_price,
)

_PLAN_PACKAGE_VERSION = 12
_CFG_FLOAT_RTOL = 1e-6
_CFG_FLOAT_ATOL = 1e-6
_SOC_SLACK_EPS = 1e-6
_SOC_SLACK_TIGHT_P95_THRESHOLD = 1e-4


def expand_episode_indices(full_input: Any) -> np.ndarray:
    """Expand per-episode lengths into a step-wise episode index array."""

    lengths = np.asarray(full_input.episode_lengths, dtype=np.int32).reshape(-1)
    indices = np.asarray(full_input.episode_indices, dtype=np.int32).reshape(-1)
    if lengths.size != indices.size:
        raise ValueError("episode_lengths and episode_indices must have the same length.")
    pieces = [
        np.full((int(length),), int(episode_idx), dtype=np.int32)
        for length, episode_idx in zip(lengths.tolist(), indices.tolist(), strict=False)
        if int(length) > 0
    ]
    if not pieces:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(pieces, axis=0)


def _require_solution_matrix(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Result is missing required solution field '{name}'.")
    return np.asarray(value, dtype=np.float32)


def validate_misocp_result_schema(result: Any) -> None:
    """Guard notebook helpers against stale MISOCPResult objects kept in a live kernel."""

    required_fields = (
        "agent_import_mw",
        "agent_export_mw",
        "agent_net_grid_mw",
        "agent_purchase_cost_eur",
        "agent_export_subsidy_eur",
        "agent_net_cost_eur",
    )
    missing_fields = [field for field in required_fields if not hasattr(result, field)]
    if missing_fields:
        missing_display = ", ".join(missing_fields)
        raise ValueError(
            "solve_result uses an outdated MISOCPResult schema; missing fields: "
            f"{missing_display}. This usually means the notebook kernel still holds an older "
            "'controllers.mpc.global_socp_mpc' module. Re-run the first import cell and the "
            "solve cell, or restart the kernel before rebuilding notebook diagnostics."
        )


def _timestamps_from_full_input(full_input: Any) -> pd.Index:
    raw = pd.Index([str(value) for value in tuple(full_input.timestamps)], name="timestamp")
    parsed = pd.to_datetime(raw, errors="coerce")
    return parsed if parsed.notna().all() else raw


def _line_loading_max(line_loading_pct: np.ndarray, horizon_steps: int) -> np.ndarray:
    if line_loading_pct.size == 0:
        return np.zeros((horizon_steps,), dtype=np.float32)
    return np.max(line_loading_pct, axis=0).astype(np.float32)


def _agent_bus_ids_from_problem(problem: Any) -> list[int]:
    bus_ids = np.asarray(problem.network.bus_ids, dtype=np.int32).reshape(-1)
    agent_positions = np.asarray(problem.network.agent_bus_positions, dtype=np.int32).reshape(-1)
    return [int(bus_ids[int(position)]) for position in agent_positions.tolist()]


def _fixed_feeder_components_from_problem(problem: Any) -> tuple[float, float]:
    """Return non-agent baseline feeder load/generation from the fixed network injections."""

    fixed_active_kw = np.asarray(problem.network.p_base_mw, dtype=np.float32).reshape(-1) * 1000.0
    fixed_load_kw = float(np.clip(fixed_active_kw, 0.0, None).sum())
    fixed_generation_kw = float(np.clip(-fixed_active_kw, 0.0, None).sum())
    return fixed_load_kw, fixed_generation_kw


def _series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full((len(frame),), float(default), dtype=np.float32)
    return frame[column].to_numpy(dtype=np.float32)


def _resolve_agent_net_load_columns(step_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {"agent_raw_net_load_kw", "agent_effective_net_load_kw", "agent_post_action_net_load_kw"}.issubset(step_df.columns):
        return (
            step_df["agent_raw_net_load_kw"].to_numpy(dtype=np.float32),
            step_df["agent_effective_net_load_kw"].to_numpy(dtype=np.float32),
            step_df["agent_post_action_net_load_kw"].to_numpy(dtype=np.float32),
        )
    return (
        _series_or_default(step_df, "base_net_load_total"),
        _series_or_default(step_df, "base_net_load_effective_total"),
        _series_or_default(step_df, "net_load_total"),
    )


def _resolve_feeder_net_load_columns(step_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if {"feeder_raw_net_load_kw", "feeder_effective_net_load_kw", "feeder_post_action_net_load_kw"}.issubset(step_df.columns):
        return (
            step_df["feeder_raw_net_load_kw"].to_numpy(dtype=np.float32),
            step_df["feeder_effective_net_load_kw"].to_numpy(dtype=np.float32),
            step_df["feeder_post_action_net_load_kw"].to_numpy(dtype=np.float32),
        )
    if {"fixed_load_kw", "fixed_generation_kw"}.issubset(step_df.columns):
        agent_raw, agent_effective, agent_post_action = _resolve_agent_net_load_columns(step_df)
        fixed_load = _series_or_default(step_df, "fixed_load_kw")
        fixed_generation = _series_or_default(step_df, "fixed_generation_kw")
        return (
            (agent_raw + fixed_load - fixed_generation).astype(np.float32),
            (agent_effective + fixed_load - fixed_generation).astype(np.float32),
            (agent_post_action + fixed_load - fixed_generation).astype(np.float32),
        )
    return None, None, None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def _build_cfg_snapshot_from_cfg(cfg: Any) -> dict[str, Any]:
    return {
        "test_start_date": str(getattr(getattr(cfg, "data", None), "test_start_date", "")),
        "test_end_date": str(getattr(getattr(cfg, "data", None), "test_end_date", "")),
        "agent_profiles": [str(value) for value in list(getattr(getattr(cfg, "data", None), "agent_profiles", []))],
        "agent_bus_ids": [int(value) for value in list(getattr(getattr(cfg, "grid", None), "agent_bus_ids", []))],
        "load_scale": [float(value) for value in list(getattr(getattr(cfg, "data", None), "load_scale", []))],
        "pv_scale": [float(value) for value in list(getattr(getattr(cfg, "data", None), "pv_scale", []))],
        "future_horizon": int(getattr(getattr(cfg, "env", None), "future_horizon", 0)),
        "sb_code": str(getattr(getattr(cfg, "grid", None), "sb_code", "")),
        "line_max_loading_pct": float(getattr(getattr(cfg, "grid", None), "line_max_loading_pct", np.nan)),
        "v_min_pu": float(getattr(getattr(cfg, "grid", None), "v_min_pu", np.nan)),
        "v_max_pu": float(getattr(getattr(cfg, "grid", None), "v_max_pu", np.nan)),
        "battery_capacity": [float(value) for value in list(getattr(getattr(cfg, "env", None), "battery_capacity", []))],
        "max_charge_rate": float(getattr(getattr(cfg, "env", None), "max_charge_rate", np.nan)),
        "efficiency": float(getattr(getattr(cfg, "env", None), "efficiency", np.nan)),
        "init_soc": float(getattr(getattr(cfg, "env", None), "init_soc", np.nan)),
        "soc_min": float(getattr(getattr(cfg, "env", None), "soc_min", np.nan)),
        "soc_max": float(getattr(getattr(cfg, "env", None), "soc_max", np.nan)),
        "soc_target": float(getattr(getattr(cfg, "env", None), "soc_target", np.nan)),
        "export_subsidy_eur_per_kwh": float(
            getattr(getattr(cfg, "reward", None), "export_subsidy_eur_per_kwh", np.nan)
        ),
        IMPORT_PRICE_MARKUP_KEY: float(getattr(getattr(cfg, "reward", None), IMPORT_PRICE_MARKUP_KEY, 0.0)),
        "price_protocol_version": int(PRICE_PROTOCOL_VERSION),
        "branch_current_tiebreaker_eur_per_pu_step": float(
            getattr(getattr(cfg, "mpc", None), "branch_current_tiebreaker_eur_per_pu_step", 0.0)
        ),
        "physics_refinement_mode": str(getattr(getattr(cfg, "mpc", None), "physics_refinement_mode", "none")),
        "physics_refinement_slack_ratio": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_slack_ratio", 2e-2)
        ),
        "physics_refinement_slack_abs_floor_eur": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_slack_abs_floor_eur", 2.0)
        ),
        "physics_refinement_slack_ratio_schedule": [
            float(value)
            for value in list(
                getattr(getattr(cfg, "mpc", None), "physics_refinement_slack_ratio_schedule", [2e-2, 5e-2])
            )
        ],
        "physics_refinement_slack_abs_floor_schedule_eur": [
            float(value)
            for value in list(
                getattr(
                    getattr(cfg, "mpc", None),
                    "physics_refinement_slack_abs_floor_schedule_eur",
                    [2.0, 5.0],
                )
            )
        ],
        "physics_refinement_enable_aggressive_third_tier": bool(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_enable_aggressive_third_tier", False)
        ),
        "physics_refinement_aggressive_third_tier_ratio": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_aggressive_third_tier_ratio", 1e-1)
        ),
        "physics_refinement_aggressive_third_tier_abs_floor_eur": float(
            getattr(
                getattr(cfg, "mpc", None),
                "physics_refinement_aggressive_third_tier_abs_floor_eur",
                10.0,
            )
        ),
        "physics_refinement_cap_utilization_trigger": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_cap_utilization_trigger", 0.95)
        ),
        "physics_refinement_branch_l_gap_ratio_trigger": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_branch_l_gap_ratio_trigger", 0.01)
        ),
        "physics_refinement_time_limit_sec": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_time_limit_sec", 20.0)
        ),
        "physics_refinement_total_time_limit_sec": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_total_time_limit_sec", 40.0)
        ),
        "physics_refinement_target_mean_solver_gap_kw": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_target_mean_solver_gap_kw", 3.0)
        ),
        "physics_refinement_target_max_solver_gap_kw": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_target_max_solver_gap_kw", 15.0)
        ),
        "physics_refinement_target_export_gap_ratio": float(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_target_export_gap_ratio", 0.05)
        ),
        "physics_refinement_use_full_start": bool(
            getattr(getattr(cfg, "mpc", None), "physics_refinement_use_full_start", True)
        ),
    }


def _build_problem_snapshot(problem: Any) -> dict[str, Any]:
    line_branch_indices = getattr(problem.network, "line_branch_indices", None)
    return {
        "n_agents": int(problem.n_agents),
        "n_buses": int(problem.n_buses),
        "n_branches": int(problem.n_branches),
        "dt_hours": float(getattr(problem, "dt_hours", np.nan)),
        "capacity_mwh": np.asarray(getattr(problem, "capacity_mwh", np.zeros(0, dtype=np.float32)), dtype=np.float32).tolist(),
        "v_min_sq": float(getattr(problem, "v_min_sq", np.nan)),
        "v_max_sq": float(getattr(problem, "v_max_sq", np.nan)),
        "trafo_limit_mva": float(getattr(problem, "trafo_limit_mva", np.nan)),
        IMPORT_PRICE_MARKUP_KEY: float(getattr(problem, "import_price_markup_eur_per_kwh", 0.0)),
        "price_protocol_version": int(PRICE_PROTOCOL_VERSION),
        "branch_current_tiebreaker_eur_per_pu_step": float(
            getattr(problem, "branch_current_tiebreaker_eur_per_pu_step", 0.0)
        ),
        "physics_refinement_mode": str(getattr(problem, "physics_refinement_mode", "none")),
        "physics_refinement_slack_ratio": float(getattr(problem, "physics_refinement_slack_ratio", 2e-2)),
        "physics_refinement_slack_abs_floor_eur": float(
            getattr(problem, "physics_refinement_slack_abs_floor_eur", 2.0)
        ),
        "physics_refinement_slack_ratio_schedule": [
            float(value) for value in list(getattr(problem, "physics_refinement_slack_ratio_schedule", (2e-2, 5e-2)))
        ],
        "physics_refinement_slack_abs_floor_schedule_eur": [
            float(value)
            for value in list(getattr(problem, "physics_refinement_slack_abs_floor_schedule_eur", (2.0, 5.0)))
        ],
        "physics_refinement_enable_aggressive_third_tier": bool(
            getattr(problem, "physics_refinement_enable_aggressive_third_tier", False)
        ),
        "physics_refinement_aggressive_third_tier_ratio": float(
            getattr(problem, "physics_refinement_aggressive_third_tier_ratio", 1e-1)
        ),
        "physics_refinement_aggressive_third_tier_abs_floor_eur": float(
            getattr(problem, "physics_refinement_aggressive_third_tier_abs_floor_eur", 10.0)
        ),
        "physics_refinement_cap_utilization_trigger": float(
            getattr(problem, "physics_refinement_cap_utilization_trigger", 0.95)
        ),
        "physics_refinement_branch_l_gap_ratio_trigger": float(
            getattr(problem, "physics_refinement_branch_l_gap_ratio_trigger", 0.01)
        ),
        "physics_refinement_time_limit_sec": float(getattr(problem, "physics_refinement_time_limit_sec", 20.0)),
        "physics_refinement_total_time_limit_sec": float(
            getattr(problem, "physics_refinement_total_time_limit_sec", 40.0)
        ),
        "physics_refinement_target_mean_solver_gap_kw": float(
            getattr(problem, "physics_refinement_target_mean_solver_gap_kw", 3.0)
        ),
        "physics_refinement_target_max_solver_gap_kw": float(
            getattr(problem, "physics_refinement_target_max_solver_gap_kw", 15.0)
        ),
        "physics_refinement_target_export_gap_ratio": float(
            getattr(problem, "physics_refinement_target_export_gap_ratio", 0.05)
        ),
        "physics_refinement_use_full_start": bool(getattr(problem, "physics_refinement_use_full_start", True)),
        "agent_q_base_zeroed": bool(
            np.allclose(
                np.asarray(problem.network.q_base_mvar, dtype=np.float32)[
                    np.asarray(problem.network.agent_bus_positions, dtype=np.int32)
                ],
                0.0,
                atol=_CFG_FLOAT_ATOL,
            )
        ),
        "network": {
            "bus_ids": np.asarray(problem.network.bus_ids, dtype=np.int32).tolist(),
            "p_base_mw": np.asarray(problem.network.p_base_mw, dtype=np.float32).tolist(),
            "q_base_mvar": np.asarray(problem.network.q_base_mvar, dtype=np.float32).tolist(),
            "agent_bus_positions": np.asarray(problem.network.agent_bus_positions, dtype=np.int32).tolist(),
            "line_branch_indices": (
                None if line_branch_indices is None else np.asarray(line_branch_indices, dtype=np.int32).tolist()
            ),
            "branch_is_trafo": np.asarray(
                getattr(problem.network, "branch_is_trafo", np.zeros((int(problem.n_branches),), dtype=bool)),
                dtype=bool,
            ).tolist(),
        },
    }


def _build_problem_view(problem_snapshot: dict[str, Any]) -> SimpleNamespace:
    network_snapshot = dict(problem_snapshot["network"])
    network = SimpleNamespace(
        bus_ids=np.asarray(network_snapshot["bus_ids"], dtype=np.int32),
        p_base_mw=np.asarray(network_snapshot["p_base_mw"], dtype=np.float32),
        q_base_mvar=np.asarray(network_snapshot.get("q_base_mvar", np.zeros(0, dtype=np.float32)), dtype=np.float32),
        agent_bus_positions=np.asarray(network_snapshot["agent_bus_positions"], dtype=np.int32),
        line_branch_indices=(
            None
            if network_snapshot.get("line_branch_indices") is None
            else np.asarray(network_snapshot["line_branch_indices"], dtype=np.int32)
        ),
        branch_is_trafo=np.asarray(network_snapshot["branch_is_trafo"], dtype=bool),
    )
    return SimpleNamespace(
        network=network,
        n_agents=int(problem_snapshot["n_agents"]),
        n_buses=int(problem_snapshot["n_buses"]),
        n_branches=int(problem_snapshot["n_branches"]),
        dt_hours=float(problem_snapshot["dt_hours"]),
        capacity_mwh=np.asarray(problem_snapshot["capacity_mwh"], dtype=np.float32),
        v_min_sq=float(problem_snapshot["v_min_sq"]),
        v_max_sq=float(problem_snapshot["v_max_sq"]),
        trafo_limit_mva=float(problem_snapshot["trafo_limit_mva"]),
        import_price_markup_eur_per_kwh=float(problem_snapshot.get(IMPORT_PRICE_MARKUP_KEY, 0.0)),
        branch_current_tiebreaker_eur_per_pu_step=float(
            problem_snapshot.get("branch_current_tiebreaker_eur_per_pu_step", 0.0)
        ),
        physics_refinement_mode=str(problem_snapshot.get("physics_refinement_mode", "none")),
        physics_refinement_slack_ratio=float(problem_snapshot.get("physics_refinement_slack_ratio", 2e-2)),
        physics_refinement_slack_abs_floor_eur=float(
            problem_snapshot.get("physics_refinement_slack_abs_floor_eur", 2.0)
        ),
        physics_refinement_slack_ratio_schedule=[
            float(value)
            for value in list(problem_snapshot.get("physics_refinement_slack_ratio_schedule", [2e-2, 5e-2]))
        ],
        physics_refinement_slack_abs_floor_schedule_eur=[
            float(value)
            for value in list(problem_snapshot.get("physics_refinement_slack_abs_floor_schedule_eur", [2.0, 5.0]))
        ],
        physics_refinement_enable_aggressive_third_tier=bool(
            problem_snapshot.get("physics_refinement_enable_aggressive_third_tier", False)
        ),
        physics_refinement_aggressive_third_tier_ratio=float(
            problem_snapshot.get("physics_refinement_aggressive_third_tier_ratio", 1e-1)
        ),
        physics_refinement_aggressive_third_tier_abs_floor_eur=float(
            problem_snapshot.get("physics_refinement_aggressive_third_tier_abs_floor_eur", 10.0)
        ),
        physics_refinement_cap_utilization_trigger=float(
            problem_snapshot.get("physics_refinement_cap_utilization_trigger", 0.95)
        ),
        physics_refinement_branch_l_gap_ratio_trigger=float(
            problem_snapshot.get("physics_refinement_branch_l_gap_ratio_trigger", 0.01)
        ),
        physics_refinement_time_limit_sec=float(problem_snapshot.get("physics_refinement_time_limit_sec", 20.0)),
        physics_refinement_total_time_limit_sec=float(
            problem_snapshot.get("physics_refinement_total_time_limit_sec", 40.0)
        ),
        physics_refinement_target_mean_solver_gap_kw=float(
            problem_snapshot.get("physics_refinement_target_mean_solver_gap_kw", 3.0)
        ),
        physics_refinement_target_max_solver_gap_kw=float(
            problem_snapshot.get("physics_refinement_target_max_solver_gap_kw", 15.0)
        ),
        physics_refinement_target_export_gap_ratio=float(
            problem_snapshot.get("physics_refinement_target_export_gap_ratio", 0.05)
        ),
        physics_refinement_use_full_start=bool(problem_snapshot.get("physics_refinement_use_full_start", True)),
    )


def _raise_mismatch(field_name: str, expected: Any, actual: Any) -> None:
    raise ValueError(
        f"MISOCP plan package mismatch at '{field_name}': expected={expected!r}, actual={actual!r}."
    )


def _assert_strict_match(field_name: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        _raise_mismatch(field_name, expected, actual)


def _assert_float_match(field_name: str, expected: float, actual: float) -> None:
    if not math.isclose(float(expected), float(actual), rel_tol=_CFG_FLOAT_RTOL, abs_tol=_CFG_FLOAT_ATOL):
        _raise_mismatch(field_name, expected, actual)


def _assert_float_sequence_match(field_name: str, expected: list[float], actual: list[float]) -> None:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if expected_array.shape != actual_array.shape:
        _raise_mismatch(field_name, expected, actual)
    if not np.allclose(expected_array, actual_array, rtol=_CFG_FLOAT_RTOL, atol=_CFG_FLOAT_ATOL):
        _raise_mismatch(field_name, expected, actual)


def _assert_int_sequence_match(field_name: str, expected: list[int], actual: list[int]) -> None:
    if [int(value) for value in expected] != [int(value) for value in actual]:
        _raise_mismatch(field_name, expected, actual)


def _assert_str_sequence_match(field_name: str, expected: list[str], actual: list[str]) -> None:
    if [str(value) for value in expected] != [str(value) for value in actual]:
        _raise_mismatch(field_name, expected, actual)


def _assert_cfg_snapshot_matches(expected_snapshot: dict[str, Any], actual_snapshot: dict[str, Any]) -> None:
    for field_name in (
        "test_start_date",
        "test_end_date",
        "future_horizon",
        "sb_code",
    ):
        _assert_strict_match(field_name, expected_snapshot[field_name], actual_snapshot[field_name])
    _assert_str_sequence_match("agent_profiles", expected_snapshot["agent_profiles"], actual_snapshot["agent_profiles"])
    _assert_int_sequence_match("agent_bus_ids", expected_snapshot["agent_bus_ids"], actual_snapshot["agent_bus_ids"])
    _assert_float_sequence_match("load_scale", expected_snapshot["load_scale"], actual_snapshot["load_scale"])
    _assert_float_sequence_match("pv_scale", expected_snapshot["pv_scale"], actual_snapshot["pv_scale"])
    _assert_float_sequence_match(
        "battery_capacity",
        expected_snapshot["battery_capacity"],
        actual_snapshot["battery_capacity"],
    )
    for field_name in (
        "line_max_loading_pct",
        "v_min_pu",
        "v_max_pu",
        "max_charge_rate",
        "efficiency",
        "init_soc",
        "soc_min",
        "soc_max",
        "soc_target",
        "export_subsidy_eur_per_kwh",
        IMPORT_PRICE_MARKUP_KEY,
        "branch_current_tiebreaker_eur_per_pu_step",
        "physics_refinement_slack_ratio",
        "physics_refinement_slack_abs_floor_eur",
        "physics_refinement_aggressive_third_tier_ratio",
        "physics_refinement_aggressive_third_tier_abs_floor_eur",
        "physics_refinement_cap_utilization_trigger",
        "physics_refinement_branch_l_gap_ratio_trigger",
        "physics_refinement_time_limit_sec",
        "physics_refinement_total_time_limit_sec",
        "physics_refinement_target_mean_solver_gap_kw",
        "physics_refinement_target_max_solver_gap_kw",
        "physics_refinement_target_export_gap_ratio",
    ):
        _assert_float_match(field_name, expected_snapshot[field_name], actual_snapshot[field_name])
    _assert_float_sequence_match(
        "physics_refinement_slack_ratio_schedule",
        expected_snapshot["physics_refinement_slack_ratio_schedule"],
        actual_snapshot["physics_refinement_slack_ratio_schedule"],
    )
    _assert_float_sequence_match(
        "physics_refinement_slack_abs_floor_schedule_eur",
        expected_snapshot["physics_refinement_slack_abs_floor_schedule_eur"],
        actual_snapshot["physics_refinement_slack_abs_floor_schedule_eur"],
    )
    for field_name in (
        "physics_refinement_mode",
        "physics_refinement_enable_aggressive_third_tier",
        "physics_refinement_use_full_start",
    ):
        _assert_strict_match(field_name, expected_snapshot[field_name], actual_snapshot[field_name])


def _assert_full_input_structure_matches(expected_full_input: Any, actual_full_input: Any) -> None:
    _assert_strict_match(
        "horizon_steps",
        int(expected_full_input.horizon_steps),
        int(actual_full_input.horizon_steps),
    )
    _assert_int_sequence_match(
        "episode_offsets",
        np.asarray(expected_full_input.episode_offsets, dtype=np.int32).tolist(),
        np.asarray(actual_full_input.episode_offsets, dtype=np.int32).tolist(),
    )
    _assert_int_sequence_match(
        "episode_lengths",
        np.asarray(expected_full_input.episode_lengths, dtype=np.int32).tolist(),
        np.asarray(actual_full_input.episode_lengths, dtype=np.int32).tolist(),
    )
    _assert_int_sequence_match(
        "episode_indices",
        np.asarray(expected_full_input.episode_indices, dtype=np.int32).tolist(),
        np.asarray(actual_full_input.episode_indices, dtype=np.int32).tolist(),
    )
    _assert_str_sequence_match(
        "timestamps",
        [str(value) for value in tuple(expected_full_input.timestamps)],
        [str(value) for value in tuple(actual_full_input.timestamps)],
    )


def estimate_global_misocp_model_size(
    problem: Any,
    horizon_steps: int,
    *,
    episode_count: int | None = None,
    enforce_terminal_soc: bool = True,
) -> pd.Series:
    """Estimate the exact model size implied by the current full-grid formulation."""

    horizon_steps = int(horizon_steps)
    if horizon_steps < 0:
        raise ValueError("horizon_steps must be non-negative.")

    n_agents = int(problem.n_agents)
    n_buses = int(problem.n_buses)
    n_branches = int(problem.n_branches)
    line_branch_indices = getattr(problem.network, "line_branch_indices", None)
    if line_branch_indices is None:
        branch_is_trafo = np.asarray(getattr(problem.network, "branch_is_trafo", np.zeros((n_branches,), dtype=bool)), dtype=bool)
        n_lines = int(np.sum(~branch_is_trafo))
    else:
        n_lines = int(np.asarray(line_branch_indices, dtype=np.int32).size)

    num_vars_est = (
        5 * n_agents * horizon_steps
        + n_agents
        + 3 * n_branches * horizon_steps
        + n_buses * horizon_steps
        + 3 * horizon_steps
    )
    num_binary_vars_est = horizon_steps
    num_linear_constraints_est = (
        3 * horizon_steps
        + n_agents * (8 * horizon_steps + 3 + int(bool(enforce_terminal_soc)))
        + horizon_steps * (3 * n_branches + n_lines + 1)
    )
    num_quadratic_constraints_est = horizon_steps * (n_branches + 1)

    return pd.Series(
        {
            "horizon_steps": horizon_steps,
            "episode_count": np.nan if episode_count is None else int(episode_count),
            "n_agents": n_agents,
            "n_buses": n_buses,
            "n_branches": n_branches,
            "n_lines": n_lines,
            "num_vars_est": int(num_vars_est),
            "num_binary_vars_est": int(num_binary_vars_est),
            "num_linear_constraints_est": int(num_linear_constraints_est),
            "num_quadratic_constraints_est": int(num_quadratic_constraints_est),
        },
        name="estimated_model_size",
    )


def format_solver_summary(
    result: Any,
    *,
    total_steps: int | None = None,
    episode_count: int | None = None,
) -> pd.Series:
    """Build a compact notebook-friendly summary of a completed solve."""

    horizon_steps = int(total_steps if total_steps is not None else getattr(result, "horizon_steps", 0))
    summary_note = ""
    if bool(getattr(result, "time_limit_feasible", False)):
        summary_note = "Reached time limit with an incumbent; objective is feasible but not proven optimal."
    elif not bool(getattr(result, "has_solution", False)):
        summary_note = "Solver returned no incumbent; inspect debug artifacts and raw log for the failure mode."
    chunk_summaries = list(getattr(result, "chunk_summaries", []) or [])
    chunk_retry_count = int(
        getattr(
            result,
            "chunk_retry_count",
            sum(1 for item in chunk_summaries if str(item.get("attempt_used", "")) == "retry"),
        )
    )
    no_retry_or_fallback_used = bool(
        getattr(result, "no_retry_or_fallback_used", bool(chunk_retry_count == 0))
    )
    total_runtime_sec = float(getattr(result, "solve_time_sec", float("nan")))
    finite_chunk_runtimes = [
        float(item.get("solve_time_sec", float("nan")))
        for item in chunk_summaries
        if np.isfinite(float(item.get("solve_time_sec", float("nan"))))
    ]
    max_chunk_runtime_sec = float(max(finite_chunk_runtimes)) if finite_chunk_runtimes else float("nan")

    return pd.Series(
        {
            "status_code": int(result.status_code),
            "status_label": str(result.status_label),
            "has_solution": bool(result.has_solution),
            "time_limit_feasible": bool(result.time_limit_feasible),
            "solve_mode": str(result.solve_mode),
            "horizon_steps": int(horizon_steps),
            "episode_count": np.nan if episode_count is None else int(episode_count),
            "solve_time_sec": float(result.solve_time_sec),
            "total_runtime_sec": total_runtime_sec,
            "max_chunk_runtime_sec": max_chunk_runtime_sec,
            "mip_gap": float(result.mip_gap),
            "best_bound": float(result.best_bound),
            "objective_value": float(result.objective_value),
            "sol_count": int(result.sol_count),
            "node_count": float(result.node_count),
            "iter_count": float(result.iter_count),
            "bar_iter_count": float(result.bar_iter_count),
            "is_near_optimal": bool(str(getattr(result, "solve_mode", "")) == "chunked_window"),
            "economics_scope": "agent_only",
            "physical_tiebreaker_eur": float(getattr(result, "physical_tiebreaker_eur", float("nan"))),
            "physical_tiebreaker_weight": float(getattr(result, "physical_tiebreaker_weight", float("nan"))),
            "stage1_primary_objective_eur": float(getattr(result, "stage1_primary_objective_eur", float("nan"))),
            "stage2_primary_objective_eur": float(getattr(result, "stage2_primary_objective_eur", float("nan"))),
            "stage2_objective_slack_eur": float(getattr(result, "stage2_objective_slack_eur", float("nan"))),
            "stage2_branch_l_objective": float(getattr(result, "stage2_branch_l_objective", float("nan"))),
            "physics_refinement_mode": str(getattr(result, "physics_refinement_mode", "none")),
            "physics_refinement_status": str(getattr(result, "physics_refinement_status", "not_enabled")),
            "physics_refinement_runtime_sec": float(getattr(result, "physics_refinement_runtime_sec", 0.0)),
            "floor_p95_soc_slack": float(getattr(result, "floor_p95_soc_slack", float("nan"))),
            "floor_mean_abs_solver_feeder_gap_kw": float(
                getattr(result, "floor_mean_abs_solver_feeder_gap_kw", float("nan"))
            ),
            "floor_primary_objective_eur": float(getattr(result, "floor_primary_objective_eur", float("nan"))),
            "floor_primary_delta_signed_eur": float(
                getattr(result, "floor_primary_delta_signed_eur", float("nan"))
            ),
            "floor_primary_delta_positive_eur": float(
                getattr(result, "floor_primary_delta_positive_eur", float("nan"))
            ),
            "physics_refinement_slack_cap_eur": float(
                getattr(result, "physics_refinement_slack_cap_eur", float("nan"))
            ),
            "initial_physics_refinement_slack_cap_eur": float(
                getattr(result, "initial_physics_refinement_slack_cap_eur", float("nan"))
            ),
            "returned_primary_objective_eur": float(
                getattr(result, "returned_primary_objective_eur", float("nan"))
            ),
            "returned_primary_delta_abs_eur": float(
                getattr(result, "returned_primary_delta_abs_eur", float("nan"))
            ),
            "returned_primary_delta_pct": float(getattr(result, "returned_primary_delta_pct", float("nan"))),
            "floor_accepted_tier": getattr(result, "floor_accepted_tier", None),
            "used_physics_refinement_tier": getattr(result, "used_physics_refinement_tier", None),
            "total_tiers_configured": int(getattr(result, "total_tiers_configured", 0)),
            "physics_refinement_attempt_count": int(getattr(result, "physics_refinement_attempt_count", 0)),
            "physics_refinement_attempt_caps_eur": [
                float(value)
                for value in list(getattr(result, "physics_refinement_attempt_caps_eur", None) or [])
            ],
            "physics_refinement_cap_utilization": float(
                getattr(result, "physics_refinement_cap_utilization", float("nan"))
            ),
            "branch_l_gap_ratio_to_floor": float(
                getattr(result, "branch_l_gap_ratio_to_floor", float("nan"))
            ),
            "returned_mean_abs_solver_feeder_gap_kw": float(
                getattr(result, "returned_mean_abs_solver_feeder_gap_kw", float("nan"))
            ),
            "returned_max_solver_feeder_gap_kw": float(
                getattr(result, "returned_max_solver_feeder_gap_kw", float("nan"))
            ),
            "returned_mean_abs_export_gap_ratio": float(
                getattr(result, "returned_mean_abs_export_gap_ratio", float("nan"))
            ),
            "high_budget_refinement_warn": bool(
                getattr(result, "high_budget_refinement_warn", False)
            ),
            "returned_solution_source": str(getattr(result, "returned_solution_source", "stage1")),
            "formulation_tightening_required": bool(
                getattr(result, "formulation_tightening_required", False)
            ),
            "negative_floor_delta_warn": bool(getattr(result, "negative_floor_delta_warn", False)),
            "refinement_status_counts": {
                str(key): int(value)
                for key, value in dict(getattr(result, "refinement_status_counts", None) or {}).items()
            },
            "no_retry_or_fallback_used": no_retry_or_fallback_used,
            "chunk_retry_count": chunk_retry_count,
            "chunk_count": int(len(chunk_summaries)),
            "num_binary_vars": int(result.model_size.num_binary_vars),
            "num_quadratic_constraints": int(result.model_size.num_quadratic_constraints),
            "solver_note": summary_note,
        },
        name="solver_summary",
    )


def build_full_horizon_step_df(problem: Any, full_input: Any, result: Any) -> pd.DataFrame:
    """Build a dense per-step diagnostics table from a solved full-horizon result."""

    validate_misocp_result_schema(result)

    load_seq = np.asarray(full_input.load_seq, dtype=np.float32)
    pv_seq = np.asarray(full_input.pv_seq, dtype=np.float32)
    wholesale_price_seq = np.asarray(full_input.wholesale_price_seq, dtype=np.float32).reshape(-1)
    import_price_seq = np.asarray(full_input.import_price_seq, dtype=np.float32).reshape(-1)
    episode_idx = expand_episode_indices(full_input)
    timestamps = _timestamps_from_full_input(full_input)
    horizon_steps = int(import_price_seq.shape[0])

    if episode_idx.shape[0] != horizon_steps:
        raise ValueError("Expanded episode index series does not match the full-horizon length.")
    if load_seq.shape[1] != horizon_steps or pv_seq.shape[1] != horizon_steps:
        raise ValueError("load_seq and pv_seq must match the full-horizon length.")
    if wholesale_price_seq.shape[0] != horizon_steps:
        raise ValueError("wholesale_price_seq must match the full-horizon length.")

    battery_charge_mw = _require_solution_matrix(result.battery_charge_mw, "battery_charge_mw")
    battery_discharge_mw = _require_solution_matrix(result.battery_discharge_mw, "battery_discharge_mw")
    pv_curtail_mw = _require_solution_matrix(result.pv_curtail_mw, "pv_curtail_mw")
    bus_vm_pu = _require_solution_matrix(result.bus_vm_pu, "bus_vm_pu")
    energy_mwh = _require_solution_matrix(result.energy_mwh, "energy_mwh")
    line_loading_pct = _require_solution_matrix(result.line_loading_pct, "line_loading_pct")
    trafo_loading_pct = _require_solution_matrix(result.trafo_loading_pct, "trafo_loading_pct")
    root_import_mw = _require_solution_matrix(result.root_import_mw, "root_import_mw").reshape(-1)
    root_export_mw = _require_solution_matrix(result.root_export_mw, "root_export_mw").reshape(-1)
    agent_import_mw = _require_solution_matrix(result.agent_import_mw, "agent_import_mw")
    agent_export_mw = _require_solution_matrix(result.agent_export_mw, "agent_export_mw")
    agent_net_grid_mw = _require_solution_matrix(result.agent_net_grid_mw, "agent_net_grid_mw")
    simultaneous_kw = _require_solution_matrix(
        result.simultaneous_charge_discharge_kw,
        "simultaneous_charge_discharge_kw",
    )

    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_from_problem(problem)

    agent_load_kw = load_seq.sum(axis=0).astype(np.float32)
    pv_raw_kw = pv_seq.sum(axis=0).astype(np.float32)
    pv_curtail_kw = (pv_curtail_mw.sum(axis=0) * 1000.0).astype(np.float32)
    pv_effective_kw = np.maximum(pv_raw_kw - pv_curtail_kw, 0.0).astype(np.float32)
    battery_charge_kw = (battery_charge_mw.sum(axis=0) * 1000.0).astype(np.float32)
    battery_discharge_kw = (battery_discharge_mw.sum(axis=0) * 1000.0).astype(np.float32)
    grid_import_kw = (root_import_mw * 1000.0).astype(np.float32)
    grid_export_kw = (root_export_mw * 1000.0).astype(np.float32)
    root_net_exchange_kw = (grid_import_kw - grid_export_kw).astype(np.float32)
    agent_import_kw_total = (agent_import_mw.sum(axis=0) * 1000.0).astype(np.float32)
    agent_export_kw_total = (agent_export_mw.sum(axis=0) * 1000.0).astype(np.float32)
    agent_net_exchange_kw_total = (agent_net_grid_mw.sum(axis=0) * 1000.0).astype(np.float32)

    agent_raw_net_load_kw = (agent_load_kw - pv_raw_kw).astype(np.float32)
    agent_effective_net_load_kw = (agent_load_kw - pv_effective_kw).astype(np.float32)
    agent_post_action_net_load_kw = (
        agent_effective_net_load_kw + battery_charge_kw - battery_discharge_kw
    ).astype(np.float32)
    feeder_raw_net_load_kw = (
        agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw
    ).astype(np.float32)
    feeder_effective_net_load_kw = (
        agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw
    ).astype(np.float32)
    feeder_post_action_net_load_kw = (
        agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw
    ).astype(np.float32)

    balance_supply_kw = (
        pv_effective_kw + fixed_generation_kw + battery_discharge_kw + grid_import_kw
    ).astype(np.float32)
    balance_demand_kw = (
        agent_load_kw + fixed_load_kw + battery_charge_kw + grid_export_kw
    ).astype(np.float32)
    balance_residual_kw = (balance_supply_kw - balance_demand_kw).astype(np.float32)
    agent_export_rate = (
        float(result.agent_export_subsidy_eur)
        / max(float(np.sum(agent_export_kw_total * np.float32(problem.dt_hours))), 1e-6)
        if float(np.sum(agent_export_kw_total)) > 0.0
        else 0.0
    )
    feeder_export_rate = (
        float(result.feeder_export_subsidy_eur)
        / max(float(np.sum(grid_export_kw * np.float32(problem.dt_hours))), 1e-6)
        if float(np.sum(grid_export_kw)) > 0.0
        else 0.0
    )
    agent_purchase_cost_eur_step = (
        agent_import_kw_total * np.float32(problem.dt_hours) * import_price_seq.astype(np.float32)
    ).astype(np.float32)
    agent_export_subsidy_eur_step = (
        agent_export_kw_total * np.float32(problem.dt_hours) * np.float32(agent_export_rate)
    ).astype(np.float32)
    agent_net_cost_eur_step = (agent_purchase_cost_eur_step - agent_export_subsidy_eur_step).astype(np.float32)
    feeder_purchase_cost_eur_step = (
        grid_import_kw * np.float32(problem.dt_hours) * import_price_seq.astype(np.float32)
    ).astype(np.float32)
    feeder_export_subsidy_eur_step = (
        grid_export_kw * np.float32(problem.dt_hours) * np.float32(feeder_export_rate)
    ).astype(np.float32)
    feeder_net_cost_eur_step = (feeder_purchase_cost_eur_step - feeder_export_subsidy_eur_step).astype(np.float32)
    agent_root_gap_kw = (root_net_exchange_kw - agent_post_action_net_load_kw).astype(np.float32)

    step_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "episode_idx": episode_idx,
            "global_step": np.arange(horizon_steps, dtype=np.int32),
            "wholesale_price": wholesale_price_seq.astype(np.float32),
            IMPORT_PRICE_COLUMN: import_price_seq.astype(np.float32),
            "fixed_load_kw": np.full((horizon_steps,), fixed_load_kw, dtype=np.float32),
            "fixed_generation_kw": np.full((horizon_steps,), fixed_generation_kw, dtype=np.float32),
            "agent_load_kw": agent_load_kw,
            "pv_raw_kw": pv_raw_kw,
            "pv_curtail_kw": pv_curtail_kw,
            "pv_effective_kw": pv_effective_kw,
            "battery_charge_kw": battery_charge_kw,
            "battery_discharge_kw": battery_discharge_kw,
            "agent_import_kw_total": agent_import_kw_total,
            "agent_export_kw_total": agent_export_kw_total,
            "agent_net_exchange_kw_total": agent_net_exchange_kw_total,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "agent_purchase_cost_eur_step": agent_purchase_cost_eur_step,
            "agent_export_subsidy_eur_step": agent_export_subsidy_eur_step,
            "agent_net_cost_eur_step": agent_net_cost_eur_step,
            "feeder_purchase_cost_eur_step": feeder_purchase_cost_eur_step,
            "feeder_export_subsidy_eur_step": feeder_export_subsidy_eur_step,
            "feeder_net_cost_eur_step": feeder_net_cost_eur_step,
            "agent_raw_net_load_kw": agent_raw_net_load_kw,
            "agent_effective_net_load_kw": agent_effective_net_load_kw,
            "agent_post_action_net_load_kw": agent_post_action_net_load_kw,
            "feeder_raw_net_load_kw": feeder_raw_net_load_kw,
            "feeder_effective_net_load_kw": feeder_effective_net_load_kw,
            "feeder_post_action_net_load_kw": feeder_post_action_net_load_kw,
            "raw_net_load_kw": feeder_raw_net_load_kw,
            "effective_net_load_kw": feeder_effective_net_load_kw,
            "post_action_net_load_kw": feeder_post_action_net_load_kw,
            "root_net_exchange_kw": root_net_exchange_kw,
            "agent_root_gap_kw": agent_root_gap_kw,
            "balance_supply_kw": balance_supply_kw,
            "balance_demand_kw": balance_demand_kw,
            "balance_residual_kw": balance_residual_kw,
            "network_loss_kw_estimate": balance_residual_kw,
            "aggregate_stored_energy_kwh": np.sum(energy_mwh[:, 1:], axis=0).astype(np.float32) * 1000.0,
            "trafo_limit_reference_kw": np.full((horizon_steps,), float(problem.trafo_limit_mva) * 1000.0, dtype=np.float32),
            "min_vm_pu": np.min(bus_vm_pu, axis=0).astype(np.float32),
            "mean_vm_pu": np.mean(bus_vm_pu, axis=0).astype(np.float32),
            "max_vm_pu": np.max(bus_vm_pu, axis=0).astype(np.float32),
            "max_line_loading_pct": _line_loading_max(line_loading_pct, horizon_steps),
            "trafo_loading_pct": np.asarray(trafo_loading_pct, dtype=np.float32).reshape(-1)[:horizon_steps],
            "simultaneous_kw_total": np.sum(simultaneous_kw, axis=0).astype(np.float32),
        }
    )
    return step_df


def build_voltage_df(problem: Any, full_input: Any, result: Any) -> pd.DataFrame:
    """Expand the bus-voltage tensor into a long-form DataFrame."""

    bus_vm_pu = _require_solution_matrix(result.bus_vm_pu, "bus_vm_pu")
    episode_idx = expand_episode_indices(full_input)
    timestamps = _timestamps_from_full_input(full_input)
    bus_ids = np.asarray(problem.network.bus_ids, dtype=np.int32).reshape(-1)
    horizon_steps = int(bus_vm_pu.shape[1])

    if episode_idx.shape[0] != horizon_steps:
        raise ValueError("Expanded episode index series does not match bus_vm_pu horizon.")
    if bus_ids.shape[0] != bus_vm_pu.shape[0]:
        raise ValueError("bus_vm_pu bus dimension does not match network bus_ids.")

    agent_bus_set = set(_agent_bus_ids_from_problem(problem))
    step_repeats = np.repeat(np.arange(horizon_steps, dtype=np.int32), bus_ids.shape[0])
    episode_repeats = np.repeat(episode_idx.astype(np.int32), bus_ids.shape[0])
    timestamp_repeats = np.repeat(np.asarray(timestamps, dtype=object), bus_ids.shape[0])
    bus_tile = np.tile(bus_ids.astype(np.int32), horizon_steps)
    vm_flat = bus_vm_pu.T.reshape(-1).astype(np.float32)
    is_agent_bus = np.asarray([int(bus_id) in agent_bus_set for bus_id in bus_tile], dtype=bool)

    return pd.DataFrame(
        {
            "timestamp": timestamp_repeats,
            "episode_idx": episode_repeats,
            "global_step": step_repeats,
            "bus_id": bus_tile,
            "vm_pu": vm_flat,
            "is_agent_bus": is_agent_bus,
        }
    )


def build_debug_tables(
    problem: Any,
    full_input: Any,
    result: Any,
    *,
    step_df: pd.DataFrame | None = None,
    agent_profiles: list[str] | tuple[str, ...] | None = None,
    agent_bus_ids: list[int] | tuple[int, ...] | None = None,
    top_k: int = 12,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build notebook-friendly diagnostic summary tables."""

    step_df = build_full_horizon_step_df(problem, full_input, result) if step_df is None else step_df.copy()
    bus_vm_pu = _require_solution_matrix(result.bus_vm_pu, "bus_vm_pu")
    battery_charge_mw = _require_solution_matrix(result.battery_charge_mw, "battery_charge_mw")
    battery_discharge_mw = _require_solution_matrix(result.battery_discharge_mw, "battery_discharge_mw")
    pv_curtail_mw = _require_solution_matrix(result.pv_curtail_mw, "pv_curtail_mw")
    energy_mwh = _require_solution_matrix(result.energy_mwh, "energy_mwh")

    resolved_profiles = list(agent_profiles or [f"agent_{idx}" for idx in range(energy_mwh.shape[0])])
    if len(resolved_profiles) != energy_mwh.shape[0]:
        raise ValueError("agent_profiles length must match the number of agents in the solution.")
    resolved_bus_ids = list(agent_bus_ids or _agent_bus_ids_from_problem(problem))
    if len(resolved_bus_ids) != energy_mwh.shape[0]:
        raise ValueError("agent_bus_ids length must match the number of agents in the solution.")

    solve_summary = pd.Series(
        {
            **format_solver_summary(
                result,
                total_steps=int(step_df.shape[0]),
                episode_count=int(np.asarray(full_input.episode_lengths, dtype=np.int32).size),
            ).to_dict(),
            "agent_purchase_cost_eur": float(result.agent_purchase_cost_eur),
            "agent_export_subsidy_eur": float(result.agent_export_subsidy_eur),
            "agent_net_cost_eur": float(result.agent_net_cost_eur),
            "feeder_purchase_cost_eur": float(result.feeder_purchase_cost_eur),
            "feeder_export_subsidy_eur": float(result.feeder_export_subsidy_eur),
            "feeder_net_cost_eur": float(result.feeder_net_cost_eur),
            "throughput_regularization_eur": float(result.throughput_regularization_eur),
            "num_vars": int(result.model_size.num_vars),
            "num_binary_vars": int(result.model_size.num_binary_vars),
            "num_linear_constraints": int(result.model_size.num_linear_constraints),
            "num_quadratic_constraints": int(result.model_size.num_quadratic_constraints),
            "simultaneous_agent_steps": int(result.simultaneous_agent_steps),
            "simultaneous_step_ratio": float(result.simultaneous_step_ratio),
            "max_simultaneous_kw": float(result.max_simultaneous_kw),
        },
        name="solve_summary",
    )

    worst_balance_df = (
        step_df.assign(abs_balance_residual_kw=lambda frame: frame["balance_residual_kw"].abs())
        .sort_values(["abs_balance_residual_kw", "global_step"], ascending=[False, True])
        .loc[
            :,
            [
                "timestamp",
                "episode_idx",
                "global_step",
                "balance_supply_kw",
                "balance_demand_kw",
                "balance_residual_kw",
                "network_loss_kw_estimate",
                "pv_curtail_kw",
                "post_action_net_load_kw",
                "root_net_exchange_kw",
                "max_line_loading_pct",
                "trafo_loading_pct",
                "simultaneous_kw_total",
                "abs_balance_residual_kw",
            ],
        ]
        .head(int(max(top_k, 1)))
        .reset_index(drop=True)
    )

    v_min_pu = float(np.sqrt(float(problem.v_min_sq)))
    v_max_pu = float(np.sqrt(float(problem.v_max_sq)))
    voltage_violation_count = (
        (bus_vm_pu < v_min_pu - 1e-6) | (bus_vm_pu > v_max_pu + 1e-6)
    ).sum(axis=0).astype(np.int32)
    under_voltage_pu = np.maximum(v_min_pu - step_df["min_vm_pu"].to_numpy(dtype=np.float32), 0.0)
    over_voltage_pu = np.maximum(step_df["max_vm_pu"].to_numpy(dtype=np.float32) - v_max_pu, 0.0)
    violation_magnitude_pu = np.maximum(under_voltage_pu, over_voltage_pu).astype(np.float32)
    voltage_issue_df = (
        step_df.assign(
            voltage_violation_count=voltage_violation_count,
            under_voltage_pu=under_voltage_pu,
            over_voltage_pu=over_voltage_pu,
            violation_magnitude_pu=violation_magnitude_pu,
        )
        .loc[
            lambda frame: frame["violation_magnitude_pu"] > 0.0,
            [
                "timestamp",
                "episode_idx",
                "global_step",
                "min_vm_pu",
                "mean_vm_pu",
                "max_vm_pu",
                "under_voltage_pu",
                "over_voltage_pu",
                "violation_magnitude_pu",
                "voltage_violation_count",
                "max_line_loading_pct",
                "trafo_loading_pct",
            ],
        ]
        .sort_values(["violation_magnitude_pu", "global_step"], ascending=[False, True])
        .head(int(max(top_k, 1)))
        .reset_index(drop=True)
    )

    capacity_kwh = np.asarray(problem.capacity_mwh, dtype=np.float32).reshape(-1) * 1000.0
    soc_init = np.asarray(full_input.soc_init, dtype=np.float32).reshape(-1)
    soc_final = (
        np.asarray(result.energy_mwh, dtype=np.float32)[:, -1] / np.maximum(np.asarray(problem.capacity_mwh, dtype=np.float32), 1e-6)
    ).astype(np.float32)
    battery_summary_df = pd.DataFrame(
        {
            "agent_profile": resolved_profiles,
            "agent_bus_id": np.asarray(resolved_bus_ids, dtype=np.int32),
            "capacity_kwh": capacity_kwh.astype(np.float32),
            "soc_init": soc_init.astype(np.float32),
            "soc_final": soc_final.astype(np.float32),
            "max_charge_kw": np.max(battery_charge_mw, axis=1).astype(np.float32) * 1000.0,
            "max_discharge_kw": np.max(battery_discharge_mw, axis=1).astype(np.float32) * 1000.0,
            "total_charge_kwh": np.sum(battery_charge_mw, axis=1).astype(np.float32) * 1000.0 * float(problem.dt_hours),
            "total_discharge_kwh": np.sum(battery_discharge_mw, axis=1).astype(np.float32) * 1000.0 * float(problem.dt_hours),
            "total_pv_curtail_kwh": np.sum(pv_curtail_mw, axis=1).astype(np.float32) * 1000.0 * float(problem.dt_hours),
            "throughput_kwh": (
                np.sum(battery_charge_mw + battery_discharge_mw, axis=1).astype(np.float32)
                * 1000.0
                * float(problem.dt_hours)
            ),
        }
    )
    return solve_summary, worst_balance_df, voltage_issue_df, battery_summary_df


def build_simultaneous_diagnostic_tables(
    problem: Any,
    full_input: Any,
    result: Any,
    *,
    agent_profiles: list[str] | tuple[str, ...] | None = None,
    top_k: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build step-level and agent-level tables for simultaneous charge/discharge diagnostics."""

    battery_charge_mw = _require_solution_matrix(result.battery_charge_mw, "battery_charge_mw")
    battery_discharge_mw = _require_solution_matrix(result.battery_discharge_mw, "battery_discharge_mw")
    pv_curtail_mw = _require_solution_matrix(result.pv_curtail_mw, "pv_curtail_mw")
    simultaneous_kw = _require_solution_matrix(result.simultaneous_charge_discharge_kw, "simultaneous_charge_discharge_kw")
    energy_mwh = _require_solution_matrix(result.energy_mwh, "energy_mwh")
    import_price_seq = np.asarray(full_input.import_price_seq, dtype=np.float32).reshape(-1)
    timestamps = _timestamps_from_full_input(full_input)
    episode_idx = expand_episode_indices(full_input)
    resolved_profiles = list(agent_profiles or [f"agent_{idx}" for idx in range(battery_charge_mw.shape[0])])
    if len(resolved_profiles) != battery_charge_mw.shape[0]:
        raise ValueError("agent_profiles length must match the number of agents in the solution.")

    step_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "episode_idx": episode_idx,
            "global_step": np.arange(import_price_seq.size, dtype=np.int32),
            IMPORT_PRICE_COLUMN: import_price_seq.astype(np.float32),
            "charge_kw_total": np.sum(battery_charge_mw, axis=0).astype(np.float32) * 1000.0,
            "discharge_kw_total": np.sum(battery_discharge_mw, axis=0).astype(np.float32) * 1000.0,
            "pv_curtail_kw_total": np.sum(pv_curtail_mw, axis=0).astype(np.float32) * 1000.0,
            "grid_import_kw": np.asarray(result.root_import_mw, dtype=np.float32).reshape(-1) * 1000.0,
            "grid_export_kw": np.asarray(result.root_export_mw, dtype=np.float32).reshape(-1) * 1000.0,
            "simultaneous_kw_total": np.sum(simultaneous_kw, axis=0).astype(np.float32),
            "simultaneous_agent_count": np.sum(simultaneous_kw > 0.0, axis=0).astype(np.int32),
        }
    )
    simultaneous_step_df = (
        step_df.loc[lambda frame: frame["simultaneous_kw_total"] > 0.0]
        .sort_values(["simultaneous_kw_total", "global_step"], ascending=[False, True])
        .head(int(max(top_k, 1)))
        .reset_index(drop=True)
    )

    capacity_mwh = np.asarray(problem.capacity_mwh, dtype=np.float32).reshape(-1)
    rows: list[dict[str, object]] = []
    for agent_idx, profile in enumerate(resolved_profiles):
        for step_idx in range(import_price_seq.size):
            simultaneous_value = float(simultaneous_kw[agent_idx, step_idx])
            if simultaneous_value <= 0.0:
                continue
            rows.append(
                {
                    "timestamp": timestamps[step_idx],
                    "episode_idx": int(episode_idx[step_idx]),
                    "global_step": int(step_idx),
                    "agent_id": int(agent_idx),
                    "agent_profile": str(profile),
                    IMPORT_PRICE_COLUMN: float(import_price_seq[step_idx]),
                    "charge_kw": float(battery_charge_mw[agent_idx, step_idx] * 1000.0),
                    "discharge_kw": float(battery_discharge_mw[agent_idx, step_idx] * 1000.0),
                    "simultaneous_kw": simultaneous_value,
                    "soc_before": float(energy_mwh[agent_idx, step_idx] / max(float(capacity_mwh[agent_idx]), 1e-6)),
                    "soc_after": float(energy_mwh[agent_idx, step_idx + 1] / max(float(capacity_mwh[agent_idx]), 1e-6)),
                    "pv_curtail_kw": float(pv_curtail_mw[agent_idx, step_idx] * 1000.0),
                    "grid_import_kw": float(np.asarray(result.root_import_mw, dtype=np.float32).reshape(-1)[step_idx] * 1000.0),
                    "grid_export_kw": float(np.asarray(result.root_export_mw, dtype=np.float32).reshape(-1)[step_idx] * 1000.0),
                }
            )
    simultaneous_agent_df = (
        pd.DataFrame(rows)
        .sort_values(["simultaneous_kw", "global_step", "agent_id"], ascending=[False, True, True])
        .head(int(max(top_k, 1)))
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(
            columns=[
                "timestamp",
                "episode_idx",
                "global_step",
                "agent_id",
                "agent_profile",
                IMPORT_PRICE_COLUMN,
                "charge_kw",
                "discharge_kw",
                "simultaneous_kw",
                "soc_before",
                "soc_after",
                "pv_curtail_kw",
                "grid_import_kw",
                "grid_export_kw",
            ]
        )
    )
    return simultaneous_step_df, simultaneous_agent_df


def build_chunk_boundary_soc_df(
    problem: Any,
    full_input: Any,
    result: Any,
) -> pd.DataFrame:
    """Summarize SoC slope kinks at chunk boundaries for near-optimal diagnostics."""

    chunk_summaries = list(getattr(result, "chunk_summaries", []) or [])
    if len(chunk_summaries) < 2:
        return pd.DataFrame(
            columns=[
                "boundary_step",
                "boundary_timestamp",
                "chunk_left",
                "chunk_right",
                "max_abs_soc_kink",
                "mean_abs_soc_kink",
            ]
        )

    energy_mwh = _require_solution_matrix(result.energy_mwh, "energy_mwh")
    capacity_mwh = np.asarray(problem.capacity_mwh, dtype=np.float32).reshape(-1, 1)
    soc = energy_mwh / np.maximum(capacity_mwh, 1e-6)
    timestamps = _timestamps_from_full_input(full_input)
    horizon_steps = int(np.asarray(full_input.import_price_seq, dtype=np.float32).reshape(-1).size)

    rows: list[dict[str, object]] = []
    for left_summary, right_summary in zip(chunk_summaries[:-1], chunk_summaries[1:], strict=False):
        boundary_step = int(left_summary["end_step"])
        if boundary_step <= 0 or boundary_step >= horizon_steps:
            continue
        delta_before = soc[:, boundary_step] - soc[:, boundary_step - 1]
        delta_after = soc[:, boundary_step + 1] - soc[:, boundary_step]
        kink = np.abs(delta_after - delta_before).astype(np.float32)
        rows.append(
            {
                "boundary_step": boundary_step,
                "boundary_timestamp": timestamps[boundary_step],
                "chunk_left": int(left_summary["chunk_idx"]),
                "chunk_right": int(right_summary["chunk_idx"]),
                "max_abs_soc_kink": float(np.max(kink)) if kink.size else 0.0,
                "mean_abs_soc_kink": float(np.mean(kink)) if kink.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _linear_fit_summary(x_values: Any, y_values: Any) -> dict[str, float]:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y_values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
    if int(np.sum(finite_mask)) < 2:
        return {"k": float("nan"), "b": float("nan"), "r2": float("nan")}
    x = x_array[finite_mask]
    y = y_array[finite_mask]
    if np.allclose(x, x[0], atol=1e-12, rtol=0.0):
        return {"k": float("nan"), "b": float("nan"), "r2": float("nan")}
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-12:
        r2 = 1.0 if ss_res <= 1e-12 else float("nan")
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    return {"k": float(slope), "b": float(intercept), "r2": float(r2)}


def _interpret_linear_fit(k: float, b: float, r2: float) -> str:
    if not (np.isfinite(k) and np.isfinite(b) and np.isfinite(r2)):
        return "insufficient_data"
    if abs(k - 1.0) <= 0.05 and abs(b) <= 1.0 and r2 > 0.99:
        return "consistent"
    if abs(k + 1.0) <= 0.05:
        return "sign_error"
    if abs(k) >= 900.0 or (abs(k) > 0.0 and abs(k) <= 0.002):
        return "unit_mismatch"
    if r2 < 0.9:
        return "nonlinear_or_noise_dominated"
    return "systematic_linear_bias"


def build_soc_relaxation_diagnostics(
    problem: Any,
    full_input: Any,
    result: Any,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Diagnose SOCP relaxation tightness using solver-internal per-unit variables."""

    branch_p_pu = _require_solution_matrix(result.branch_p_pu, "branch_p_pu")
    branch_q_pu = _require_solution_matrix(result.branch_q_pu, "branch_q_pu")
    branch_i2_pu = _require_solution_matrix(result.branch_i2_pu, "branch_i2_pu")
    bus_v_sq = _require_solution_matrix(result.bus_v_sq, "bus_v_sq")
    parent_pos = np.asarray(problem.network.branch_parent_pos, dtype=np.int32).reshape(-1)
    child_pos = np.asarray(problem.network.branch_child_pos, dtype=np.int32).reshape(-1)
    bus_ids = np.asarray(problem.network.bus_ids, dtype=np.int32).reshape(-1)
    timestamps = _timestamps_from_full_input(full_input)
    episode_idx = expand_episode_indices(full_input)

    if branch_p_pu.shape != branch_q_pu.shape or branch_p_pu.shape != branch_i2_pu.shape:
        raise ValueError("branch_p_pu, branch_q_pu, and branch_i2_pu must share the same shape.")
    if branch_p_pu.shape[0] != parent_pos.shape[0]:
        raise ValueError("Branch solution arrays do not match network.branch_parent_pos length.")
    if bus_v_sq.shape[0] != bus_ids.shape[0]:
        raise ValueError("bus_v_sq bus dimension does not match network.bus_ids length.")

    v_parent_sq = np.asarray(bus_v_sq[parent_pos, :], dtype=np.float32)
    soc_slack = np.asarray(
        branch_i2_pu - ((branch_p_pu ** 2 + branch_q_pu ** 2) / np.maximum(v_parent_sq, _SOC_SLACK_EPS)),
        dtype=np.float32,
    )
    slack_max = np.max(soc_slack, axis=0).astype(np.float32)
    slack_mean = np.mean(soc_slack, axis=0).astype(np.float32)
    finite_slack = soc_slack[np.isfinite(soc_slack)]
    p95_soc_slack = float(np.percentile(finite_slack, 95.0)) if finite_slack.size else float("nan")
    summary = pd.Series(
        {
            "max_soc_slack": float(np.max(finite_slack)) if finite_slack.size else float("nan"),
            "mean_soc_slack": float(np.mean(finite_slack)) if finite_slack.size else float("nan"),
            "p95_soc_slack": p95_soc_slack,
            "soc_relaxation_is_tight": bool(np.isfinite(p95_soc_slack) and p95_soc_slack < _SOC_SLACK_TIGHT_P95_THRESHOLD),
        },
        name="soc_relaxation_summary",
    )
    step_df = pd.DataFrame(
        {
            "global_step": np.arange(branch_p_pu.shape[1], dtype=np.int32),
            "timestamp": timestamps,
            "episode_idx": episode_idx.astype(np.int32),
            "soc_slack_max": slack_max,
            "soc_slack_mean": slack_mean,
            "soc_slack_p95_global": np.full((branch_p_pu.shape[1],), p95_soc_slack, dtype=np.float32),
        }
    )

    flat_indices = np.argsort(soc_slack.reshape(-1))[::-1]
    worst_rows: list[dict[str, object]] = []
    for flat_idx in flat_indices[: max(int(top_k), 0)]:
        branch_idx, step_idx = np.unravel_index(int(flat_idx), soc_slack.shape)
        worst_rows.append(
            {
                "branch_idx": int(branch_idx),
                "global_step": int(step_idx),
                "timestamp": timestamps[int(step_idx)],
                "episode_idx": int(episode_idx[int(step_idx)]),
                "parent_bus_id": int(bus_ids[int(parent_pos[int(branch_idx)])]),
                "child_bus_id": int(bus_ids[int(child_pos[int(branch_idx)])]),
                "v_parent_sq_pu": float(v_parent_sq[int(branch_idx), int(step_idx)]),
                "branch_p_pu": float(branch_p_pu[int(branch_idx), int(step_idx)]),
                "branch_q_pu": float(branch_q_pu[int(branch_idx), int(step_idx)]),
                "branch_i2_pu": float(branch_i2_pu[int(branch_idx), int(step_idx)]),
                "soc_slack": float(soc_slack[int(branch_idx), int(step_idx)]),
            }
        )
    return {
        "step_df": step_df,
        "worst_df": pd.DataFrame(worst_rows),
        "summary": summary,
        "soc_slack": soc_slack,
    }


def build_root_q_diagnostic_df(problem: Any, full_input: Any, result: Any) -> pd.DataFrame:
    """Decompose root reactive power into background-q, x*l loss proxy, and residual."""

    branch_i2_pu = _require_solution_matrix(result.branch_i2_pu, "branch_i2_pu")
    root_q_kvar = _require_solution_matrix(result.root_q_kvar, "root_q_kvar").reshape(-1)
    branch_x_pu = np.asarray(problem.network.branch_x_pu, dtype=np.float32).reshape(-1)
    s_base_mva = float(problem.network.s_base_mva)
    timestamps = _timestamps_from_full_input(full_input)
    episode_idx = expand_episode_indices(full_input)

    if branch_i2_pu.shape[0] != branch_x_pu.shape[0]:
        raise ValueError("branch_i2_pu and network.branch_x_pu must share the branch dimension.")
    if branch_i2_pu.shape[1] != root_q_kvar.shape[0]:
        raise ValueError("branch_i2_pu and root_q_kvar must share the horizon length.")

    background_q_base_total_kvar = float(np.sum(np.asarray(problem.network.q_base_mvar, dtype=np.float32)) * 1000.0)
    network_q_loss_proxy_kvar = (
        np.sum(branch_x_pu[:, None] * branch_i2_pu, axis=0) * np.float32(s_base_mva * 1000.0)
    ).astype(np.float32)
    root_q_residual_kvar = (root_q_kvar.astype(np.float32) - background_q_base_total_kvar - network_q_loss_proxy_kvar).astype(
        np.float32
    )
    return pd.DataFrame(
        {
            "global_step": np.arange(root_q_kvar.shape[0], dtype=np.int32),
            "timestamp": timestamps,
            "episode_idx": episode_idx.astype(np.int32),
            "background_q_base_total_kvar": np.full((root_q_kvar.shape[0],), background_q_base_total_kvar, dtype=np.float32),
            "network_q_loss_proxy_kvar": network_q_loss_proxy_kvar,
            "root_q_kvar": root_q_kvar.astype(np.float32),
            "root_q_residual_kvar": root_q_residual_kvar,
        }
    )


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


def build_misocp_validation_artifacts(
    env: Any,
    problem: Any,
    full_input: Any,
    result: Any,
    *,
    controller_label: str,
    export_subsidy: float,
    agent_profiles: list[str] | tuple[str, ...] | None = None,
    agent_bus_ids: list[int] | tuple[int, ...] | None = None,
    v_min_pu: float | None = None,
    v_max_pu: float | None = None,
) -> dict[str, Any]:
    """Replay a solved MISOCP plan in GridEnv and build validation artifacts without re-solving."""

    from controllers.action_feasibility import (
        build_safety_local_numpy,
        compute_action_gap_metrics_numpy,
        merge_action_info_into_step_info,
    )
    from controllers.mpc.global_socp_mpc import _SIMULTANEOUS_THRESHOLD_RATIO
    from scripts.utils.grid_notebook_workflow import (
        RolloutResult,
        _aligned_prediction,
        _assemble_global_oracle_actions,
        _export_subsidy_per_agent,
        _mean_component_total,
        _purchase_cost_per_agent,
        _step_timestamp,
    )

    resolved_profiles = list(agent_profiles or [f"agent_{idx}" for idx in range(int(env.n))])
    resolved_agent_bus_ids = list(agent_bus_ids or getattr(getattr(env, "_grid_core", None), "agent_bus_ids", []))
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_set = set(resolved_agent_bus_ids)
    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_from_problem(problem)
    trafo_limit_reference_kw = float(problem.trafo_limit_mva) * 1000.0
    loading_limit_pct = float(getattr(getattr(env, "_grid_cfg", None), "line_max_loading_pct", 100.0))
    threshold_kw = (_SIMULTANEOUS_THRESHOLD_RATIO * np.asarray(env.agent_p_max, dtype=np.float32)).astype(np.float32)

    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    carried_soc = np.asarray(full_input.soc_init, dtype=np.float32).copy()
    timestamps = _timestamps_from_full_input(full_input)
    soc_relaxation_artifacts = build_soc_relaxation_diagnostics(problem, full_input, result, top_k=20)
    soc_relaxation_step_df = soc_relaxation_artifacts["step_df"]
    soc_relaxation_worst_df = soc_relaxation_artifacts["worst_df"]
    soc_relaxation_summary = soc_relaxation_artifacts["summary"]
    root_q_diagnostic_df = build_root_q_diagnostic_df(problem, full_input, result)
    soc_relaxation_lookup = soc_relaxation_step_df.set_index("global_step", drop=False) if not soc_relaxation_step_df.empty else pd.DataFrame()
    root_q_lookup = root_q_diagnostic_df.set_index("global_step", drop=False) if not root_q_diagnostic_df.empty else pd.DataFrame()

    for episode_list_idx, episode_idx in enumerate(np.asarray(full_input.episode_indices, dtype=np.int32).tolist()):
        obs, reset_info = env.reset(episode_idx=int(episode_idx))
        if episode_list_idx > 0:
            env.soc = carried_soc.copy()
            obs = env.obs_builder.build(env)
        raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
        previous_raw_obs = None
        episode_offset = int(np.asarray(full_input.episode_offsets, dtype=np.int32)[episode_list_idx])
        episode_length = int(np.asarray(full_input.episode_lengths, dtype=np.int32)[episode_list_idx])

        for step_in_episode in range(episode_length):
            global_step = episode_offset + step_in_episode
            timestamp = timestamps[global_step] if global_step < len(timestamps) else _step_timestamp(reset_info, step_in_episode)
            wholesale_price_pred = _aligned_prediction(previous_raw_obs, raw_obs, WHOLESALE_PRICE_SEQ_FIELD)
            import_price_pred = float(
                derive_import_price(
                    wholesale_price_pred,
                    markup_eur_per_kwh=float(getattr(getattr(env, "_reward_cfg", None), IMPORT_PRICE_MARKUP_KEY, 0.0)),
                )
            )
            load_pred = _aligned_prediction(previous_raw_obs, raw_obs, "load_seq")
            pv_pred = _aligned_prediction(previous_raw_obs, raw_obs, "pv_seq")

            battery_power_kw = (
                (
                    np.asarray(result.battery_charge_mw[:, global_step], dtype=np.float32)
                    - np.asarray(result.battery_discharge_mw[:, global_step], dtype=np.float32)
                )
                * 1000.0
            ).astype(np.float32)
            pv_curtail_kw = (np.asarray(result.pv_curtail_mw[:, global_step], dtype=np.float32) * 1000.0).astype(np.float32)
            actions, action_array = _assemble_global_oracle_actions(env, battery_power_kw, pv_curtail_kw)

            action_info = compute_action_gap_metrics_numpy(
                build_safety_local_numpy(
                    soc=np.asarray(env.soc, dtype=np.float32),
                    load_raw=np.asarray(env.get_signal_step("load"), dtype=np.float32),
                    pv_raw=np.asarray(env.get_signal_step("pv"), dtype=np.float32),
                    battery_capacity_kwh=np.asarray(env.agent_c_bat, dtype=np.float32),
                    p_max_kw=np.asarray(env.agent_p_max, dtype=np.float32),
                ),
                action_array,
                action_array,
            )
            simultaneous_kw = np.asarray(result.simultaneous_charge_discharge_kw[:, global_step], dtype=np.float32)
            simultaneous_mask = simultaneous_kw > threshold_kw
            action_info.update(
                {
                    "misocp_fallback": np.asarray(0.0, dtype=np.float32),
                    "misocp_time_limit_feasible": np.asarray(float(result.time_limit_feasible), dtype=np.float32),
                    "solve_time_sec": np.asarray(float(result.solve_time_sec), dtype=np.float32),
                    "root_import_kw": np.asarray(float(result.root_import_mw[global_step] * 1000.0), dtype=np.float32),
                    "root_export_kw": np.asarray(float(result.root_export_mw[global_step] * 1000.0), dtype=np.float32),
                    "simultaneous_charge_discharge_kw_total": np.asarray(float(np.sum(simultaneous_kw)), dtype=np.float32),
                    "simultaneous_agent_count": np.asarray(float(np.sum(simultaneous_mask)), dtype=np.float32),
                    "simultaneous_step_flag": np.asarray(float(np.any(simultaneous_mask)), dtype=np.float32),
                }
            )

            next_obs, reward, terminated, truncated, info = env.step(actions)
            reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
            info, action_penalty = merge_action_info_into_step_info(
                info,
                action_info,
                soc_pen_weight=float(getattr(getattr(env, "_reward_cfg", None), "w_soc_pen", 0.0)),
                apply_action_penalty=False,
            )
            reward_array = reward_array - np.asarray(action_penalty, dtype=np.float32)
            info["reward"] = reward_array.astype(np.float32)
            del terminated, truncated
            raw_next_obs = (
                env.obs_builder.build_raw(env)
                if hasattr(env.obs_builder, "build_raw") and not bool(info.get("episode_done", False))
                else next_obs
            )

            purchase_cost_per_agent = _purchase_cost_per_agent(info, float(env.dt))
            base_net_load = np.asarray(
                info.get("base_net_load", np.asarray(info["load"], dtype=np.float32) - np.asarray(info["pv"], dtype=np.float32)),
                dtype=np.float32,
            )
            base_net_load_effective = np.asarray(info.get("base_net_load_effective", base_net_load), dtype=np.float32)
            net_load = np.asarray(
                info.get("net_load", base_net_load_effective + np.asarray(info["e_bat"], dtype=np.float32)),
                dtype=np.float32,
            )
            pv_raw = np.asarray(info.get("pv_raw", info["pv"]), dtype=np.float32)
            pv_effective = np.asarray(info.get("pv_effective", pv_raw), dtype=np.float32)
            pv_curtail = np.asarray(info.get("pv_curtail", pv_raw - pv_effective), dtype=np.float32)
            grid_import = np.asarray(info.get("grid_import_kw", np.maximum(net_load, 0.0)), dtype=np.float32)
            grid_export = np.asarray(info.get("grid_export_kw", np.maximum(-net_load, 0.0)), dtype=np.float32)
            battery_power = np.asarray(info["e_bat"], dtype=np.float32)
            battery_charge = np.clip(battery_power, 0.0, None).astype(np.float32)
            battery_discharge = np.maximum(-battery_power, 0.0).astype(np.float32)
            line_loading_pct = np.asarray(info.get("line_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
            trafo_loading_pct = np.asarray(info.get("trafo_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
            pp_root_p_raw = info.get("trafo_p_signed_kw")
            pp_root_p_array = (
                np.asarray(pp_root_p_raw, dtype=np.float32).reshape(-1)
                if pp_root_p_raw is not None
                else np.zeros(0, dtype=np.float32)
            )
            pp_root_p_available = bool(pp_root_p_array.size > 0)
            pp_root_p_kw = (
                float(pp_root_p_array.sum())
                if pp_root_p_available
                else float("nan")
            )
            export_subsidy_per_agent = _export_subsidy_per_agent(info, float(env.dt), float(export_subsidy))
            voltage_penalty_total = _mean_component_total(info, "r_safe_v", env.n)
            line_penalty_total = _mean_component_total(info, "r_safe_line", env.n)
            trafo_penalty_total = _mean_component_total(info, "r_safe_trafo", env.n)
            soc_penalty_total = float(np.sum(np.asarray(info.get("r_soc_pen", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)))
            purchase_cost_total = float(np.sum(purchase_cost_per_agent))
            export_subsidy_total = float(np.sum(export_subsidy_per_agent))
            agent_net_cost_total = float(purchase_cost_total - export_subsidy_total)
            agent_raw_net_load_kw = float(np.sum(base_net_load))
            agent_effective_net_load_kw = float(np.sum(base_net_load_effective))
            agent_post_action_net_load_kw = float(np.sum(net_load))
            feeder_raw_net_load_kw = float(agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw)
            feeder_effective_net_load_kw = float(agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw)
            feeder_post_action_net_load_kw = float(agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw)
            agent_root_gap_kw = float(result.root_p_kw[global_step] - agent_post_action_net_load_kw)
            solver_feeder_gap_kw = float(result.root_p_kw[global_step] - feeder_post_action_net_load_kw)
            replay_feeder_gap_kw = (
                float(pp_root_p_kw - feeder_post_action_net_load_kw)
                if pp_root_p_available
                else float("nan")
            )
            misocp_root_s_kva = float(np.hypot(float(result.root_p_kw[global_step]), float(result.root_q_kvar[global_step])))
            # Assumes pandapower trafo_loading_pct is apparent-power based; current-based loading would make
            # this back-calculated pp_root_s_kva only an approximation.
            pp_root_s_kva = float(
                np.max(trafo_loading_pct) / 100.0 * float(problem.network.s_base_mva) * 1000.0
            ) if trafo_loading_pct.size else float("nan")
            soc_row = soc_relaxation_lookup.loc[int(global_step)] if not soc_relaxation_lookup.empty else None
            root_q_row = root_q_lookup.loc[int(global_step)] if not root_q_lookup.empty else None
            soc_slack_max = float(soc_row["soc_slack_max"]) if soc_row is not None else float("nan")
            soc_slack_mean = float(soc_row["soc_slack_mean"]) if soc_row is not None else float("nan")
            soc_slack_p95_global = (
                float(soc_relaxation_summary.get("p95_soc_slack", float("nan")))
                if not soc_relaxation_summary.empty
                else float("nan")
            )
            background_q_base_total_kvar = (
                float(root_q_row["background_q_base_total_kvar"]) if root_q_row is not None else float("nan")
            )
            network_q_loss_proxy_kvar = (
                float(root_q_row["network_q_loss_proxy_kvar"]) if root_q_row is not None else float("nan")
            )
            root_q_residual_kvar = float(root_q_row["root_q_residual_kvar"]) if root_q_row is not None else float("nan")

            step_rows.append(
                {
                    "controller": controller_label,
                    "episode_idx": int(episode_idx),
                    "step": step_in_episode,
                    "global_step": int(global_step),
                    "timestamp": timestamp,
                    "wholesale_price": float(info["wholesale_price"]),
                    IMPORT_PRICE_COLUMN: float(info[IMPORT_PRICE_COLUMN]),
                    WHOLESALE_PRICE_PRED_COLUMN: float(wholesale_price_pred),
                    IMPORT_PRICE_PRED_COLUMN: float(import_price_pred),
                    "base_net_load_total": agent_raw_net_load_kw,
                    "base_net_load_effective_total": agent_effective_net_load_kw,
                    "net_load_total": agent_post_action_net_load_kw,
                    "agent_raw_net_load_kw": agent_raw_net_load_kw,
                    "agent_effective_net_load_kw": agent_effective_net_load_kw,
                    "agent_post_action_net_load_kw": agent_post_action_net_load_kw,
                    "fixed_load_kw": fixed_load_kw,
                    "fixed_generation_kw": fixed_generation_kw,
                    "feeder_raw_net_load_kw": feeder_raw_net_load_kw,
                    "feeder_effective_net_load_kw": feeder_effective_net_load_kw,
                    "feeder_post_action_net_load_kw": feeder_post_action_net_load_kw,
                    "root_net_exchange_kw": float(result.root_p_kw[global_step]),
                    "agent_root_gap_kw": agent_root_gap_kw,
                    "solver_feeder_gap_kw": solver_feeder_gap_kw,
                    "replay_feeder_gap_kw": replay_feeder_gap_kw,
                    "pp_root_p_kw": pp_root_p_kw,
                    "pp_root_p_available": bool(pp_root_p_available),
                    "load_total": float(np.sum(np.asarray(info["load"], dtype=np.float32))),
                    "pv_raw_total": float(np.sum(pv_raw)),
                    "pv_effective_total": float(np.sum(pv_effective)),
                    "pv_curtail_total": float(np.sum(pv_curtail)),
                    "grid_import_total": float(np.sum(grid_import)),
                    "grid_export_total": float(np.sum(grid_export)),
                    "battery_charge_total": float(np.sum(battery_charge)),
                    "battery_discharge_total": float(np.sum(battery_discharge)),
                    "purchase_cost_total": purchase_cost_total,
                    "export_subsidy_total": export_subsidy_total,
                    "agent_net_cost_total": agent_net_cost_total,
                    "soc_penalty_total": soc_penalty_total,
                    "voltage_penalty_total": voltage_penalty_total,
                    "line_penalty_total": line_penalty_total,
                    "trafo_penalty_total": trafo_penalty_total,
                    "line_loading_pct_max": float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0,
                    "trafo_loading_pct_max": float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0,
                    "n_line_violations": int(np.any(line_loading_pct > loading_limit_pct)),
                    "n_trafo_violations": int(np.any(trafo_loading_pct > loading_limit_pct)),
                    "misocp_fallback": 0.0,
                    "misocp_time_limit_feasible": float(result.time_limit_feasible),
                    "solve_time_sec": float(result.solve_time_sec),
                    "root_import_kw": float(result.root_import_mw[global_step] * 1000.0),
                    "root_export_kw": float(result.root_export_mw[global_step] * 1000.0),
                    "simultaneous_charge_discharge_kw_total": float(np.sum(simultaneous_kw)),
                    "simultaneous_agent_count": int(np.sum(simultaneous_mask)),
                    "simultaneous_step_flag": float(np.any(simultaneous_mask)),
                    "trafo_limit_reference_kw": trafo_limit_reference_kw,
                    "misocp_root_s_kva": misocp_root_s_kva,
                    "pp_root_s_kva": pp_root_s_kva,
                    "soc_slack_max": soc_slack_max,
                    "soc_slack_mean": soc_slack_mean,
                    "soc_slack_p95_global": soc_slack_p95_global,
                    "background_q_base_total_kvar": background_q_base_total_kvar,
                    "network_q_loss_proxy_kvar": network_q_loss_proxy_kvar,
                    "root_q_residual_kvar": root_q_residual_kvar,
                }
            )
            diagnostic_rows.append(
                {
                    "solver_type": "gurobi_misocp",
                    "status_code": int(result.status_code),
                    "status_label": str(result.status_label),
                    "solve_mode": str(result.solve_mode),
                    "solve_time_sec": float(result.solve_time_sec),
                    "misocp_fallback": 0.0,
                    "misocp_time_limit_feasible": float(result.time_limit_feasible),
                    "mip_gap": float(result.mip_gap),
                    "best_bound": float(result.best_bound),
                    "objective_value": float(result.objective_value),
                    "agent_purchase_cost_eur": float(result.agent_purchase_cost_eur),
                    "agent_export_subsidy_eur": float(result.agent_export_subsidy_eur),
                    "agent_net_cost_eur": float(result.agent_net_cost_eur),
                    "feeder_purchase_cost_eur": float(result.feeder_purchase_cost_eur),
                    "feeder_export_subsidy_eur": float(result.feeder_export_subsidy_eur),
                    "feeder_net_cost_eur": float(result.feeder_net_cost_eur),
                    "throughput_regularization_eur": float(result.throughput_regularization_eur),
                    "throughput_regularization_weight": float(result.throughput_regularization_weight),
                    "root_import_kw": float(result.root_import_mw[global_step] * 1000.0),
                    "root_export_kw": float(result.root_export_mw[global_step] * 1000.0),
                    "root_p_kw": float(result.root_p_kw[global_step]),
                    "root_q_kvar": float(result.root_q_kvar[global_step]),
                    "misocp_vm_pu": np.asarray(result.bus_vm_pu[:, global_step], dtype=np.float32).copy(),
                    "misocp_line_loading_pct": np.asarray(result.line_loading_pct[:, global_step], dtype=np.float32).copy(),
                    "misocp_trafo_loading_pct": np.asarray(result.trafo_loading_pct[:, global_step], dtype=np.float32).copy(),
                    "pp_vm_pu": np.asarray(info.get("vm_pu", np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32).copy(),
                    "pp_line_loading_pct": line_loading_pct.copy(),
                    "pp_trafo_loading_pct": trafo_loading_pct.copy(),
                    "pp_root_p_kw": pp_root_p_kw,
                    "pp_root_p_available": bool(pp_root_p_available),
                    "misocp_root_s_kva": misocp_root_s_kva,
                    "pp_root_s_kva": pp_root_s_kva,
                    "soc_slack_max": soc_slack_max,
                    "soc_slack_mean": soc_slack_mean,
                    "soc_slack_p95_global": soc_slack_p95_global,
                    "background_q_base_total_kvar": background_q_base_total_kvar,
                    "network_q_loss_proxy_kvar": network_q_loss_proxy_kvar,
                    "root_q_residual_kvar": root_q_residual_kvar,
                    "solver_feeder_gap_kw": solver_feeder_gap_kw,
                    "replay_feeder_gap_kw": replay_feeder_gap_kw,
                    "controller": controller_label,
                    "episode_idx": int(episode_idx),
                    "step": int(step_in_episode),
                    "timestamp": timestamp,
                }
            )

            for agent_idx, profile in enumerate(resolved_profiles):
                agent_rows.append(
                    {
                        "controller": controller_label,
                        "episode_idx": int(episode_idx),
                        "step": step_in_episode,
                        "global_step": int(global_step),
                        "timestamp": timestamp,
                        "agent_id": agent_idx,
                        "agent_profile": str(profile),
                        "load": float(np.asarray(info["load"], dtype=np.float32)[agent_idx]),
                        "load_pred": float(np.asarray(load_pred, dtype=np.float32)[agent_idx]),
                        "pv": float(np.asarray(info["pv"], dtype=np.float32)[agent_idx]),
                        "pv_raw": float(pv_raw[agent_idx]),
                        "pv_effective": float(pv_effective[agent_idx]),
                        "pv_curtail": float(pv_curtail[agent_idx]),
                        "pv_pred": float(np.asarray(pv_pred, dtype=np.float32)[agent_idx]),
                        "base_net_load": float(base_net_load[agent_idx]),
                        "base_net_load_effective": float(base_net_load_effective[agent_idx]),
                        "net_load": float(net_load[agent_idx]),
                        "grid_import_kw": float(grid_import[agent_idx]),
                        "grid_export_kw": float(grid_export[agent_idx]),
                        "e_bat": float(battery_power[agent_idx]),
                        "soc": float(np.asarray(info["soc_next"], dtype=np.float32)[agent_idx]),
                    }
                )

            vm_pu = np.asarray(info.get("vm_pu", np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32)
            if vm_pu.shape[0] == len(bus_ids):
                for bus_id, vm_value in zip(bus_ids, vm_pu, strict=False):
                    grid_rows.append(
                        {
                            "controller": controller_label,
                            "episode_idx": int(episode_idx),
                            "step": step_in_episode,
                            "global_step": int(global_step),
                            "timestamp": timestamp,
                            "bus_id": int(bus_id),
                            "vm_pu": float(vm_value),
                            "is_agent_bus": bool(int(bus_id) in agent_bus_set),
                        }
                    )

            previous_raw_obs = raw_obs
            obs = next_obs
            raw_obs = raw_next_obs

        carried_soc = np.asarray(env.soc, dtype=np.float32).copy()

    step_df = pd.DataFrame(step_rows).sort_values(["episode_idx", "step"]).reset_index(drop=True)
    agent_df = pd.DataFrame(agent_rows).sort_values(["episode_idx", "step", "agent_id"]).reset_index(drop=True)
    grid_df = pd.DataFrame(grid_rows).sort_values(["episode_idx", "step", "bus_id"]).reset_index(drop=True)
    validation_df = build_misocp_validation_df(diagnostic_rows)
    validation_summary = summarize_misocp_validation(validation_df)
    health_warning = ""
    if not validation_df.empty and bool((~validation_df["within_tolerance"]).any()):
        health_warning = "MISOCP-vs-pandapower validation exceeded at least one tolerance."
    if bool(validation_summary.get("root_power_validation_unavailable", False)):
        health_warning = (
            (health_warning + " " if health_warning else "")
            + "Root-power validation is unavailable for at least one step because GridEnv did not expose trafo_p_signed_kw."
        )
    rollout = RolloutResult(
        step_df=step_df,
        agent_df=agent_df,
        grid_df=grid_df,
        summary=pd.DataFrame(),
        meta={
            "controller": controller_label,
            "agent_profiles": resolved_profiles,
            "agent_bus_ids": resolved_agent_bus_ids,
            "bus_ids": bus_ids,
            "v_min_pu": np.nan if v_min_pu is None else float(v_min_pu),
            "v_max_pu": np.nan if v_max_pu is None else float(v_max_pu),
            "trafo_limit_kw": trafo_limit_reference_kw,
            "trafo_limit_note": "Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.",
            "misocp_validation_df": validation_df,
            "misocp_validation_summary": validation_summary,
            "misocp_health_warning": health_warning,
            "controller_diagnostic_log": diagnostic_rows,
            "economics_scope": "agent_only",
            "validation_root_power_source": "trafo_p_signed_kw",
            "soc_relaxation_summary": soc_relaxation_summary,
        },
    )
    return {
        "rollout": rollout,
        "step_df": step_df,
        "agent_df": agent_df,
        "grid_df": grid_df,
        "diagnostic_rows": diagnostic_rows,
        "validation_df": validation_df,
        "validation_summary": validation_summary,
        "soc_relaxation_step_df": soc_relaxation_step_df,
        "soc_relaxation_worst_df": soc_relaxation_worst_df,
        "soc_relaxation_summary": soc_relaxation_summary,
        "root_q_diagnostic_df": root_q_diagnostic_df,
    }


def build_misocp_plan_package(
    problem: Any,
    full_input: Any,
    result: Any,
    *,
    controller_label: str,
    export_subsidy: float,
    cfg: Any | None = None,
    cfg_snapshot: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    diagnostic_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a disk-friendly MISOCP plan package without replay data."""

    if cfg_snapshot is None:
        if cfg is None:
            raise ValueError("Provide either cfg or cfg_snapshot when building a MISOCP plan package.")
        cfg_snapshot = _build_cfg_snapshot_from_cfg(cfg)
    if not bool(getattr(result, "has_solution", False)):
        raise ValueError("Cannot package a MISOCP result without a feasible incumbent.")

    solved_result = result
    manifest = {
        "plan_package_version": int(_PLAN_PACKAGE_VERSION),
        "controller_label": str(controller_label),
        "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
        "price_protocol_version": int(PRICE_PROTOCOL_VERSION),
        "horizon_steps": int(full_input.horizon_steps),
        "episode_offsets": np.asarray(full_input.episode_offsets, dtype=np.int32).tolist(),
        "episode_lengths": np.asarray(full_input.episode_lengths, dtype=np.int32).tolist(),
        "episode_indices": np.asarray(full_input.episode_indices, dtype=np.int32).tolist(),
        "cfg_snapshot": dict(cfg_snapshot),
        "problem_snapshot": _build_problem_snapshot(problem),
        "extra_meta": dict(extra_meta or {}),
        "diagnostics_stage": "commit4_adaptive_refinement_ladder",
        "validation_root_power_source": "trafo_p_signed_kw",
        "agent_q_base_zeroed": bool(
            np.allclose(
                np.asarray(problem.network.q_base_mvar, dtype=np.float32)[
                    np.asarray(problem.network.agent_bus_positions, dtype=np.int32)
                ],
                0.0,
                atol=_CFG_FLOAT_ATOL,
            )
        ),
        "physical_tiebreaker_weight": float(getattr(solved_result, "physical_tiebreaker_weight", 0.0)),
        "validation_fix_stage": "commit4_adaptive_refinement_ladder",
        "physics_refinement_mode": str(getattr(solved_result, "physics_refinement_mode", "none")),
        "physics_refinement_status": str(getattr(solved_result, "physics_refinement_status", "not_enabled")),
        "physics_refinement_slack_ratio": float(cfg_snapshot.get("physics_refinement_slack_ratio", 2e-2)),
        "physics_refinement_slack_abs_floor_eur": float(
            cfg_snapshot.get("physics_refinement_slack_abs_floor_eur", 2.0)
        ),
        "physics_refinement_slack_ratio_schedule": list(
            cfg_snapshot.get("physics_refinement_slack_ratio_schedule", [2e-2, 5e-2])
        ),
        "physics_refinement_slack_abs_floor_schedule_eur": list(
            cfg_snapshot.get("physics_refinement_slack_abs_floor_schedule_eur", [2.0, 5.0])
        ),
        "physics_refinement_enable_aggressive_third_tier": bool(
            cfg_snapshot.get("physics_refinement_enable_aggressive_third_tier", False)
        ),
        "physics_refinement_time_limit_sec": float(cfg_snapshot.get("physics_refinement_time_limit_sec", 20.0)),
        "physics_refinement_total_time_limit_sec": float(
            cfg_snapshot.get("physics_refinement_total_time_limit_sec", 40.0)
        ),
        "diagnostics_floor_stage": "min_sum_branch_l",
    }
    solve_summary = {
        **format_solver_summary(
            solved_result,
            total_steps=int(full_input.horizon_steps),
            episode_count=int(np.asarray(full_input.episode_lengths, dtype=np.int32).size),
        ).to_dict(),
        "agent_purchase_cost_eur": float(solved_result.agent_purchase_cost_eur),
        "agent_export_subsidy_eur": float(solved_result.agent_export_subsidy_eur),
        "agent_net_cost_eur": float(solved_result.agent_net_cost_eur),
        "feeder_purchase_cost_eur": float(solved_result.feeder_purchase_cost_eur),
        "feeder_export_subsidy_eur": float(solved_result.feeder_export_subsidy_eur),
        "feeder_net_cost_eur": float(solved_result.feeder_net_cost_eur),
        "throughput_regularization_eur": float(solved_result.throughput_regularization_eur),
        "throughput_regularization_weight": float(solved_result.throughput_regularization_weight),
        "physical_tiebreaker_eur": float(getattr(solved_result, "physical_tiebreaker_eur", 0.0)),
        "physical_tiebreaker_weight": float(getattr(solved_result, "physical_tiebreaker_weight", 0.0)),
        "num_vars": int(solved_result.model_size.num_vars),
        "num_binary_vars": int(solved_result.model_size.num_binary_vars),
        "num_linear_constraints": int(solved_result.model_size.num_linear_constraints),
        "num_quadratic_constraints": int(solved_result.model_size.num_quadratic_constraints),
        "simultaneous_agent_steps": int(solved_result.simultaneous_agent_steps),
        "simultaneous_step_ratio": float(solved_result.simultaneous_step_ratio),
        "max_simultaneous_kw": float(solved_result.max_simultaneous_kw),
        "sanity_warning": str(getattr(solved_result, "sanity_warning", "")),
        "debug_artifacts": dict(getattr(solved_result, "debug_artifacts", {})),
        "export_subsidy_eur_per_kwh": float(export_subsidy),
        IMPORT_PRICE_MARKUP_KEY: float(cfg_snapshot.get(IMPORT_PRICE_MARKUP_KEY, 0.0)),
        "branch_current_tiebreaker_eur_per_pu_step": float(
            cfg_snapshot.get("branch_current_tiebreaker_eur_per_pu_step", 0.0)
        ),
        "physics_refinement_mode": str(getattr(solved_result, "physics_refinement_mode", "none")),
        "physics_refinement_status": str(getattr(solved_result, "physics_refinement_status", "not_enabled")),
        "physics_refinement_slack_ratio": float(cfg_snapshot.get("physics_refinement_slack_ratio", 2e-2)),
        "physics_refinement_slack_abs_floor_eur": float(
            cfg_snapshot.get("physics_refinement_slack_abs_floor_eur", 2.0)
        ),
        "physics_refinement_slack_ratio_schedule": list(
            cfg_snapshot.get("physics_refinement_slack_ratio_schedule", [2e-2, 5e-2])
        ),
        "physics_refinement_slack_abs_floor_schedule_eur": list(
            cfg_snapshot.get("physics_refinement_slack_abs_floor_schedule_eur", [2.0, 5.0])
        ),
        "physics_refinement_enable_aggressive_third_tier": bool(
            cfg_snapshot.get("physics_refinement_enable_aggressive_third_tier", False)
        ),
        "physics_refinement_time_limit_sec": float(cfg_snapshot.get("physics_refinement_time_limit_sec", 20.0)),
        "physics_refinement_total_time_limit_sec": float(
            cfg_snapshot.get("physics_refinement_total_time_limit_sec", 40.0)
        ),
        "stage1_primary_objective_eur": float(getattr(solved_result, "stage1_primary_objective_eur", float("nan"))),
        "stage2_primary_objective_eur": float(getattr(solved_result, "stage2_primary_objective_eur", float("nan"))),
        "stage2_objective_slack_eur": float(getattr(solved_result, "stage2_objective_slack_eur", float("nan"))),
        "stage2_branch_l_objective": float(getattr(solved_result, "stage2_branch_l_objective", float("nan"))),
        "physics_refinement_runtime_sec": float(getattr(solved_result, "physics_refinement_runtime_sec", 0.0)),
        "floor_p95_soc_slack": float(getattr(solved_result, "floor_p95_soc_slack", float("nan"))),
        "floor_mean_abs_solver_feeder_gap_kw": float(
            getattr(solved_result, "floor_mean_abs_solver_feeder_gap_kw", float("nan"))
        ),
        "floor_primary_objective_eur": float(getattr(solved_result, "floor_primary_objective_eur", float("nan"))),
        "floor_primary_delta_signed_eur": float(
            getattr(solved_result, "floor_primary_delta_signed_eur", float("nan"))
        ),
        "floor_primary_delta_positive_eur": float(
            getattr(solved_result, "floor_primary_delta_positive_eur", float("nan"))
        ),
        "physics_refinement_slack_cap_eur": float(
            getattr(solved_result, "physics_refinement_slack_cap_eur", float("nan"))
        ),
        "initial_physics_refinement_slack_cap_eur": float(
            getattr(solved_result, "initial_physics_refinement_slack_cap_eur", float("nan"))
        ),
        "returned_primary_objective_eur": float(
            getattr(solved_result, "returned_primary_objective_eur", float("nan"))
        ),
        "returned_primary_delta_abs_eur": float(
            getattr(solved_result, "returned_primary_delta_abs_eur", float("nan"))
        ),
        "returned_primary_delta_pct": float(getattr(solved_result, "returned_primary_delta_pct", float("nan"))),
        "floor_accepted_tier": getattr(solved_result, "floor_accepted_tier", None),
        "used_physics_refinement_tier": getattr(solved_result, "used_physics_refinement_tier", None),
        "total_tiers_configured": int(getattr(solved_result, "total_tiers_configured", 0)),
        "physics_refinement_attempt_count": int(getattr(solved_result, "physics_refinement_attempt_count", 0)),
        "physics_refinement_attempt_caps_eur": [
            float(value)
            for value in list(getattr(solved_result, "physics_refinement_attempt_caps_eur", None) or [])
        ],
        "physics_refinement_cap_utilization": float(
            getattr(solved_result, "physics_refinement_cap_utilization", float("nan"))
        ),
        "branch_l_gap_ratio_to_floor": float(
            getattr(solved_result, "branch_l_gap_ratio_to_floor", float("nan"))
        ),
        "returned_mean_abs_solver_feeder_gap_kw": float(
            getattr(solved_result, "returned_mean_abs_solver_feeder_gap_kw", float("nan"))
        ),
        "returned_max_solver_feeder_gap_kw": float(
            getattr(solved_result, "returned_max_solver_feeder_gap_kw", float("nan"))
        ),
        "returned_mean_abs_export_gap_ratio": float(
            getattr(solved_result, "returned_mean_abs_export_gap_ratio", float("nan"))
        ),
        "high_budget_refinement_warn": bool(getattr(solved_result, "high_budget_refinement_warn", False)),
        "returned_solution_source": str(getattr(solved_result, "returned_solution_source", "stage1")),
        "formulation_tightening_required": bool(
            getattr(solved_result, "formulation_tightening_required", False)
        ),
        "negative_floor_delta_warn": bool(getattr(solved_result, "negative_floor_delta_warn", False)),
        "refinement_status_counts": {
            str(key): int(value)
            for key, value in dict(getattr(solved_result, "refinement_status_counts", None) or {}).items()
        },
        "economics_scope": "agent_only",
        "diagnostics_stage": "commit4_adaptive_refinement_ladder",
        "diagnostics_floor_stage": "min_sum_branch_l",
        "chunk_summaries": [dict(item) for item in list(getattr(solved_result, "chunk_summaries", []) or [])],
    }
    if diagnostic_summary:
        solve_summary.update(dict(diagnostic_summary))
    return {
        "manifest": manifest,
        "full_input": full_input,
        "result": solved_result,
        "solve_summary": solve_summary,
    }


def save_misocp_plan_package(package: dict[str, Any], target_dir: str | Path) -> Path:
    """Persist a MISOCP plan package to a deterministic directory layout."""

    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    manifest = dict(package["manifest"])
    full_input = package["full_input"]
    result = package["result"]
    solve_summary = dict(package["solve_summary"])

    (target_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (target_path / "solve_summary.json").write_text(
        json.dumps(solve_summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    np.savez_compressed(
        target_path / "full_input.npz",
        wholesale_price_seq=np.asarray(full_input.wholesale_price_seq, dtype=np.float32),
        import_price_seq=np.asarray(full_input.import_price_seq, dtype=np.float32),
        load_seq=np.asarray(full_input.load_seq, dtype=np.float32),
        pv_seq=np.asarray(full_input.pv_seq, dtype=np.float32),
        soc_init=np.asarray(full_input.soc_init, dtype=np.float32),
        timestamps=np.asarray(tuple(full_input.timestamps), dtype=np.str_),
        episode_offsets=np.asarray(full_input.episode_offsets, dtype=np.int32),
        episode_lengths=np.asarray(full_input.episode_lengths, dtype=np.int32),
        episode_indices=np.asarray(full_input.episode_indices, dtype=np.int32),
    )
    np.savez_compressed(
        target_path / "solve_result.npz",
        agent_net_grid_mw=np.asarray(result.agent_net_grid_mw, dtype=np.float32),
        agent_import_mw=np.asarray(result.agent_import_mw, dtype=np.float32),
        agent_export_mw=np.asarray(result.agent_export_mw, dtype=np.float32),
        battery_charge_mw=np.asarray(result.battery_charge_mw, dtype=np.float32),
        battery_discharge_mw=np.asarray(result.battery_discharge_mw, dtype=np.float32),
        pv_curtail_mw=np.asarray(result.pv_curtail_mw, dtype=np.float32),
        energy_mwh=np.asarray(result.energy_mwh, dtype=np.float32),
        branch_p_pu=np.asarray(result.branch_p_pu, dtype=np.float32),
        branch_q_pu=np.asarray(result.branch_q_pu, dtype=np.float32),
        branch_i2_pu=np.asarray(result.branch_i2_pu, dtype=np.float32),
        bus_v_sq=np.asarray(result.bus_v_sq, dtype=np.float32),
        root_import_mw=np.asarray(result.root_import_mw, dtype=np.float32),
        root_export_mw=np.asarray(result.root_export_mw, dtype=np.float32),
        root_p_kw=np.asarray(result.root_p_kw, dtype=np.float32),
        root_q_kvar=np.asarray(result.root_q_kvar, dtype=np.float32),
        bus_vm_pu=np.asarray(result.bus_vm_pu, dtype=np.float32),
        line_loading_pct=np.asarray(result.line_loading_pct, dtype=np.float32),
        trafo_loading_pct=np.asarray(result.trafo_loading_pct, dtype=np.float32),
        simultaneous_charge_discharge_kw=np.asarray(result.simultaneous_charge_discharge_kw, dtype=np.float32),
    )
    return target_path


def resolve_latest_compatible_misocp_plan_package_dir(target_dir: str | Path) -> Path:
    """Resolve the newest complete cached-plan directory sharing the requested prefix."""

    target_path = Path(target_dir).expanduser().resolve()
    parent_dir = target_path.parent
    prefix = target_path.name
    required_names = ("manifest.json", "full_input.npz", "solve_result.npz", "solve_summary.json")

    if not parent_dir.exists():
        raise FileNotFoundError(
            f"Cannot resolve cached MISOCP plan prefix '{prefix}' because parent directory does not exist: {parent_dir}"
        )

    compatible_candidates: list[dict[str, Any]] = []
    discovered_candidates: list[str] = []
    for candidate_dir in parent_dir.iterdir():
        if not candidate_dir.is_dir():
            continue
        if candidate_dir.name != prefix and not candidate_dir.name.startswith(f"{prefix}_"):
            continue
        manifest_path = candidate_dir / "manifest.json"
        if not manifest_path.exists():
            discovered_candidates.append(f"{candidate_dir.name} (missing manifest)")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            discovered_candidates.append(f"{candidate_dir.name} (invalid manifest)")
            continue
        version = int(manifest.get("plan_package_version", -1))
        missing_required = [
            name for name in required_names if not (candidate_dir / name).exists()
        ]
        saved_at_raw = manifest.get("saved_at_utc")
        try:
            saved_at = pd.Timestamp(saved_at_raw)
            if saved_at.tzinfo is None:
                saved_at = saved_at.tz_localize("UTC")
            else:
                saved_at = saved_at.tz_convert("UTC")
            saved_at_key = int(saved_at.value)
        except Exception:
            saved_at_key = -1
        if version == int(_PLAN_PACKAGE_VERSION) and not missing_required:
            compatible_candidates.append(
                {
                    "path": candidate_dir,
                    "saved_at_key": saved_at_key,
                    "mtime_ns": int(candidate_dir.stat().st_mtime_ns),
                }
            )
        else:
            missing_display = (
                f", missing={','.join(missing_required)}" if missing_required else ""
            )
            discovered_candidates.append(
                f"{candidate_dir.name} (version={version}{missing_display})"
            )

    if compatible_candidates:
        compatible_candidates.sort(
            key=lambda item: (int(item["saved_at_key"]), int(item["mtime_ns"]), str(item["path"].name)),
            reverse=True,
        )
        return Path(compatible_candidates[0]["path"])

    if discovered_candidates:
        raise ValueError(
            "No compatible MISOCP plan package found for prefix "
            f"'{prefix}' under {parent_dir}. Expected plan_package_version={_PLAN_PACKAGE_VERSION}. "
            f"Discovered candidates: {', '.join(discovered_candidates)}. "
            "Re-run MISOCP_global.ipynb or point MISOCP_PLAN_INPUT_DIR to a compatible package."
        )

    raise FileNotFoundError(
        f"No cached MISOCP plan package directories matching prefix '{prefix}' were found under {parent_dir}."
    )


def load_misocp_plan_package(target_dir: str | Path) -> dict[str, Any]:
    """Load a saved MISOCP plan package and reconstruct real dataclasses."""

    from controllers.mpc.global_socp_mpc import FullHorizonProblemInput, MISOCPResult, ModelSize

    target_path = Path(target_dir).resolve()
    manifest_path = target_path / "manifest.json"
    full_input_path = target_path / "full_input.npz"
    solve_result_path = target_path / "solve_result.npz"
    solve_summary_path = target_path / "solve_summary.json"

    missing_files = [
        str(path.name)
        for path in (manifest_path, full_input_path, solve_result_path, solve_summary_path)
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"MISOCP plan package is incomplete at {target_path}: missing {', '.join(missing_files)}."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("plan_package_version", -1)) != int(_PLAN_PACKAGE_VERSION):
        raise ValueError(
            "Unsupported MISOCP plan package version at "
            f"{target_path}: expected={_PLAN_PACKAGE_VERSION}, actual={manifest.get('plan_package_version')!r}. "
            "This cache predates the adaptive slack ladder / tiered floor accept / floor-distance gate / time-budget refinement semantics. "
            "Re-run MISOCP_global.ipynb to regenerate it."
        )
    solve_summary = json.loads(solve_summary_path.read_text(encoding="utf-8"))

    with np.load(full_input_path, allow_pickle=False) as full_input_archive:
        full_input = FullHorizonProblemInput(
            wholesale_price_seq=np.asarray(full_input_archive["wholesale_price_seq"], dtype=np.float32),
            import_price_seq=np.asarray(full_input_archive["import_price_seq"], dtype=np.float32),
            load_seq=np.asarray(full_input_archive["load_seq"], dtype=np.float32),
            pv_seq=np.asarray(full_input_archive["pv_seq"], dtype=np.float32),
            soc_init=np.asarray(full_input_archive["soc_init"], dtype=np.float32),
            timestamps=tuple(str(value) for value in np.asarray(full_input_archive["timestamps"]).tolist()),
            episode_offsets=np.asarray(full_input_archive["episode_offsets"], dtype=np.int32),
            episode_lengths=np.asarray(full_input_archive["episode_lengths"], dtype=np.int32),
            episode_indices=np.asarray(full_input_archive["episode_indices"], dtype=np.int32),
        )
    with np.load(solve_result_path, allow_pickle=False) as result_archive:
        model_size = ModelSize(
            num_vars=int(solve_summary["num_vars"]),
            num_binary_vars=int(solve_summary["num_binary_vars"]),
            num_linear_constraints=int(solve_summary["num_linear_constraints"]),
            num_quadratic_constraints=int(solve_summary["num_quadratic_constraints"]),
        )
        result = MISOCPResult(
            status_code=int(solve_summary["status_code"]),
            status_label=str(solve_summary["status_label"]),
            has_solution=bool(solve_summary["has_solution"]),
            time_limit_feasible=bool(solve_summary["time_limit_feasible"]),
            solve_time_sec=float(solve_summary["solve_time_sec"]),
            mip_gap=float(solve_summary["mip_gap"]),
            best_bound=float(solve_summary["best_bound"]),
            objective_value=float(solve_summary["objective_value"]),
            agent_purchase_cost_eur=float(solve_summary["agent_purchase_cost_eur"]),
            agent_export_subsidy_eur=float(solve_summary["agent_export_subsidy_eur"]),
            agent_net_cost_eur=float(solve_summary["agent_net_cost_eur"]),
            feeder_purchase_cost_eur=float(solve_summary["feeder_purchase_cost_eur"]),
            feeder_export_subsidy_eur=float(solve_summary["feeder_export_subsidy_eur"]),
            feeder_net_cost_eur=float(solve_summary["feeder_net_cost_eur"]),
            throughput_regularization_eur=float(solve_summary["throughput_regularization_eur"]),
            throughput_regularization_weight=float(solve_summary["throughput_regularization_weight"]),
            physical_tiebreaker_eur=float(solve_summary.get("physical_tiebreaker_eur", 0.0)),
            physical_tiebreaker_weight=float(solve_summary.get("physical_tiebreaker_weight", 0.0)),
            model_size=model_size,
            debug_artifacts=dict(solve_summary.get("debug_artifacts", {})),
            agent_net_grid_mw=np.asarray(result_archive["agent_net_grid_mw"], dtype=np.float32),
            agent_import_mw=np.asarray(result_archive["agent_import_mw"], dtype=np.float32),
            agent_export_mw=np.asarray(result_archive["agent_export_mw"], dtype=np.float32),
            battery_charge_mw=np.asarray(result_archive["battery_charge_mw"], dtype=np.float32),
            battery_discharge_mw=np.asarray(result_archive["battery_discharge_mw"], dtype=np.float32),
            pv_curtail_mw=np.asarray(result_archive["pv_curtail_mw"], dtype=np.float32),
            energy_mwh=np.asarray(result_archive["energy_mwh"], dtype=np.float32),
            branch_p_pu=np.asarray(result_archive["branch_p_pu"], dtype=np.float32),
            branch_q_pu=np.asarray(result_archive["branch_q_pu"], dtype=np.float32),
            branch_i2_pu=np.asarray(result_archive["branch_i2_pu"], dtype=np.float32),
            bus_v_sq=np.asarray(result_archive["bus_v_sq"], dtype=np.float32),
            root_import_mw=np.asarray(result_archive["root_import_mw"], dtype=np.float32),
            root_export_mw=np.asarray(result_archive["root_export_mw"], dtype=np.float32),
            root_p_kw=np.asarray(result_archive["root_p_kw"], dtype=np.float32),
            root_q_kvar=np.asarray(result_archive["root_q_kvar"], dtype=np.float32),
            bus_vm_pu=np.asarray(result_archive["bus_vm_pu"], dtype=np.float32),
            line_loading_pct=np.asarray(result_archive["line_loading_pct"], dtype=np.float32),
            trafo_loading_pct=np.asarray(result_archive["trafo_loading_pct"], dtype=np.float32),
            simultaneous_charge_discharge_kw=np.asarray(
                result_archive["simultaneous_charge_discharge_kw"],
                dtype=np.float32,
            ),
            simultaneous_agent_steps=int(solve_summary["simultaneous_agent_steps"]),
            simultaneous_step_ratio=float(solve_summary["simultaneous_step_ratio"]),
            max_simultaneous_kw=float(solve_summary["max_simultaneous_kw"]),
            sol_count=int(solve_summary["sol_count"]),
            node_count=float(solve_summary["node_count"]),
            iter_count=float(solve_summary["iter_count"]),
            bar_iter_count=float(solve_summary["bar_iter_count"]),
            sanity_warning=str(solve_summary.get("sanity_warning", "")),
            horizon_steps=int(manifest["horizon_steps"]),
            solve_mode=str(solve_summary["solve_mode"]),
            episode_offsets=np.asarray(manifest["episode_offsets"], dtype=np.int32),
            episode_lengths=np.asarray(manifest["episode_lengths"], dtype=np.int32),
            chunk_summaries=[dict(item) for item in list(solve_summary.get("chunk_summaries", []) or [])],
            no_retry_or_fallback_used=bool(solve_summary.get("no_retry_or_fallback_used", True)),
            chunk_retry_count=int(solve_summary.get("chunk_retry_count", 0)),
            stage1_primary_objective_eur=float(solve_summary.get("stage1_primary_objective_eur", float("nan"))),
            stage2_primary_objective_eur=float(solve_summary.get("stage2_primary_objective_eur", float("nan"))),
            stage2_objective_slack_eur=float(solve_summary.get("stage2_objective_slack_eur", float("nan"))),
            stage2_branch_l_objective=float(solve_summary.get("stage2_branch_l_objective", float("nan"))),
            physics_refinement_mode=str(solve_summary.get("physics_refinement_mode", "none")),
            physics_refinement_status=str(solve_summary.get("physics_refinement_status", "not_enabled")),
            physics_refinement_runtime_sec=float(solve_summary.get("physics_refinement_runtime_sec", 0.0)),
            floor_p95_soc_slack=float(solve_summary.get("floor_p95_soc_slack", float("nan"))),
            floor_mean_abs_solver_feeder_gap_kw=float(
                solve_summary.get("floor_mean_abs_solver_feeder_gap_kw", float("nan"))
            ),
            floor_primary_objective_eur=float(solve_summary.get("floor_primary_objective_eur", float("nan"))),
            floor_primary_delta_signed_eur=float(
                solve_summary.get("floor_primary_delta_signed_eur", float("nan"))
            ),
            floor_primary_delta_positive_eur=float(
                solve_summary.get("floor_primary_delta_positive_eur", float("nan"))
            ),
            physics_refinement_slack_cap_eur=float(
                solve_summary.get("physics_refinement_slack_cap_eur", float("nan"))
            ),
            initial_physics_refinement_slack_cap_eur=float(
                solve_summary.get("initial_physics_refinement_slack_cap_eur", float("nan"))
            ),
            returned_primary_objective_eur=float(
                solve_summary.get("returned_primary_objective_eur", float("nan"))
            ),
            returned_primary_delta_abs_eur=float(
                solve_summary.get("returned_primary_delta_abs_eur", float("nan"))
            ),
            returned_primary_delta_pct=float(solve_summary.get("returned_primary_delta_pct", float("nan"))),
            floor_accepted_tier=solve_summary.get("floor_accepted_tier"),
            used_physics_refinement_tier=solve_summary.get("used_physics_refinement_tier"),
            total_tiers_configured=int(solve_summary.get("total_tiers_configured", 0)),
            physics_refinement_attempt_count=int(solve_summary.get("physics_refinement_attempt_count", 0)),
            physics_refinement_attempt_caps_eur=[
                float(value) for value in list(solve_summary.get("physics_refinement_attempt_caps_eur", []) or [])
            ],
            physics_refinement_cap_utilization=float(
                solve_summary.get("physics_refinement_cap_utilization", float("nan"))
            ),
            branch_l_gap_ratio_to_floor=float(
                solve_summary.get("branch_l_gap_ratio_to_floor", float("nan"))
            ),
            returned_mean_abs_solver_feeder_gap_kw=float(
                solve_summary.get("returned_mean_abs_solver_feeder_gap_kw", float("nan"))
            ),
            returned_max_solver_feeder_gap_kw=float(
                solve_summary.get("returned_max_solver_feeder_gap_kw", float("nan"))
            ),
            returned_mean_abs_export_gap_ratio=float(
                solve_summary.get("returned_mean_abs_export_gap_ratio", float("nan"))
            ),
            high_budget_refinement_warn=bool(solve_summary.get("high_budget_refinement_warn", False)),
            returned_solution_source=str(solve_summary.get("returned_solution_source", "stage1")),
            formulation_tightening_required=bool(
                solve_summary.get("formulation_tightening_required", False)
            ),
            negative_floor_delta_warn=bool(solve_summary.get("negative_floor_delta_warn", False)),
            refinement_status_counts={
                str(key): int(value)
                for key, value in dict(solve_summary.get("refinement_status_counts", {}) or {}).items()
            },
        )

    problem_view = _build_problem_view(manifest["problem_snapshot"])
    try:
        step_df = build_full_horizon_step_df(problem_view, full_input, result)
        build_debug_tables(
            problem_view,
            full_input,
            result,
            step_df=step_df,
            agent_profiles=list(manifest["cfg_snapshot"]["agent_profiles"]),
            agent_bus_ids=list(manifest["cfg_snapshot"]["agent_bus_ids"]),
        )
    except Exception as exc:  # pragma: no cover - exercised through round-trip tests
        raise RuntimeError(
            f"Loaded MISOCP plan package at {target_path} failed compatibility self-check: {exc}"
        ) from exc

    return {
        "manifest": manifest,
        "full_input": full_input,
        "result": result,
        "solve_summary": solve_summary,
        "problem_view": problem_view,
    }


def replay_misocp_plan_package(
    cfg: Any,
    target_dir: str | Path,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Replay a cached MISOCP plan package through pandapower without re-solving."""

    from controllers.mpc import GlobalMISOCPProblem
    from scripts.builder import build_env
    from scripts.utils.grid_notebook_workflow import PERFECT_PREDICTION_MODE, build_comparison_cfg

    package = load_misocp_plan_package(target_dir)
    comparison_cfg = build_comparison_cfg(cfg, prediction_mode=PERFECT_PREDICTION_MODE)
    current_cfg_snapshot = _build_cfg_snapshot_from_cfg(comparison_cfg)
    _assert_cfg_snapshot_matches(package["manifest"]["cfg_snapshot"], current_cfg_snapshot)

    env = build_env(comparison_cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, comparison_cfg)
        current_full_input = problem.build_full_horizon_input(env)
        _assert_full_input_structure_matches(package["full_input"], current_full_input)
        controller_label = str(label or package["manifest"]["controller_label"])
        replay_artifacts = build_misocp_validation_artifacts(
            env,
            problem,
            package["full_input"],
            package["result"],
            controller_label=controller_label,
            export_subsidy=float(package["solve_summary"]["export_subsidy_eur_per_kwh"]),
            agent_profiles=list(current_cfg_snapshot["agent_profiles"]),
            agent_bus_ids=list(current_cfg_snapshot["agent_bus_ids"]),
            v_min_pu=float(current_cfg_snapshot["v_min_pu"]),
            v_max_pu=float(current_cfg_snapshot["v_max_pu"]),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to replay cached MISOCP plan package at {Path(target_dir).resolve()}: {exc}"
        ) from exc
    finally:
        env.close()

    rollout = replay_artifacts["rollout"]
    if (
        not replay_artifacts["agent_df"].empty
        and {"purchase_cost", "export_subsidy", "objective_total"}.issubset(replay_artifacts["agent_df"].columns)
    ):
        rollout.summary = (
            replay_artifacts["agent_df"]
            .groupby(["controller", "agent_profile"], as_index=False)[["purchase_cost", "export_subsidy", "objective_total"]]
            .sum()
        )
    rollout.meta.update(
        {
            "controller": controller_label,
            "soc_mode": "continuous",
            "solve_mode": str(package["result"].solve_mode),
            "global_oracle_runtime_sec": float(
                package["solve_summary"].get("total_runtime_sec", package["solve_summary"]["solve_time_sec"])
            ),
            "global_oracle_gap": float(package["solve_summary"]["mip_gap"]),
            "global_misocp_gap": float(package["solve_summary"]["mip_gap"]),
            "global_oracle_status_label": str(package["solve_summary"]["status_label"]),
            "global_oracle_debug_artifacts": dict(package["solve_summary"].get("debug_artifacts", {})),
            "is_near_optimal": bool(str(package["result"].solve_mode) == "chunked_window"),
            "no_retry_or_fallback_used": bool(package["solve_summary"].get("no_retry_or_fallback_used", True)),
            "chunk_retry_count": int(package["solve_summary"].get("chunk_retry_count", 0)),
            "plan_package_dir": str(Path(target_dir).resolve()),
            "plan_package_version": int(package["manifest"]["plan_package_version"]),
            "loaded_from_cached_plan": True,
            "economics_scope": str(package["solve_summary"].get("economics_scope", "agent_only")),
            "agent_purchase_cost_eur": float(package["solve_summary"]["agent_purchase_cost_eur"]),
            "agent_export_subsidy_eur": float(package["solve_summary"]["agent_export_subsidy_eur"]),
            "agent_net_cost_eur": float(package["solve_summary"]["agent_net_cost_eur"]),
            "feeder_purchase_cost_eur": float(package["solve_summary"]["feeder_purchase_cost_eur"]),
            "feeder_export_subsidy_eur": float(package["solve_summary"]["feeder_export_subsidy_eur"]),
            "feeder_net_cost_eur": float(package["solve_summary"]["feeder_net_cost_eur"]),
            "export_subsidy_eur_per_kwh": float(package["solve_summary"]["export_subsidy_eur_per_kwh"]),
            IMPORT_PRICE_MARKUP_KEY: float(
                package["manifest"]["cfg_snapshot"].get(IMPORT_PRICE_MARKUP_KEY, 0.0)
            ),
            "physical_tiebreaker_weight": float(package["manifest"].get("physical_tiebreaker_weight", 0.0)),
            "physics_refinement_mode": str(package["solve_summary"].get("physics_refinement_mode", "none")),
            "physics_refinement_status": str(package["solve_summary"].get("physics_refinement_status", "not_enabled")),
            "physics_refinement_runtime_sec": float(package["solve_summary"].get("physics_refinement_runtime_sec", 0.0)),
            "stage1_primary_objective_eur": float(
                package["solve_summary"].get("stage1_primary_objective_eur", float("nan"))
            ),
            "stage2_primary_objective_eur": float(
                package["solve_summary"].get("stage2_primary_objective_eur", float("nan"))
            ),
            "stage2_objective_slack_eur": float(
                package["solve_summary"].get("stage2_objective_slack_eur", float("nan"))
            ),
            "stage2_branch_l_objective": float(
                package["solve_summary"].get("stage2_branch_l_objective", float("nan"))
            ),
            "floor_p95_soc_slack": float(package["solve_summary"].get("floor_p95_soc_slack", float("nan"))),
            "floor_mean_abs_solver_feeder_gap_kw": float(
                package["solve_summary"].get("floor_mean_abs_solver_feeder_gap_kw", float("nan"))
            ),
            "floor_primary_objective_eur": float(
                package["solve_summary"].get("floor_primary_objective_eur", float("nan"))
            ),
            "floor_primary_delta_signed_eur": float(
                package["solve_summary"].get("floor_primary_delta_signed_eur", float("nan"))
            ),
            "floor_primary_delta_positive_eur": float(
                package["solve_summary"].get("floor_primary_delta_positive_eur", float("nan"))
            ),
            "physics_refinement_slack_cap_eur": float(
                package["solve_summary"].get("physics_refinement_slack_cap_eur", float("nan"))
            ),
            "initial_physics_refinement_slack_cap_eur": float(
                package["solve_summary"].get("initial_physics_refinement_slack_cap_eur", float("nan"))
            ),
            "returned_primary_objective_eur": float(
                package["solve_summary"].get("returned_primary_objective_eur", float("nan"))
            ),
            "returned_primary_delta_abs_eur": float(
                package["solve_summary"].get("returned_primary_delta_abs_eur", float("nan"))
            ),
            "returned_primary_delta_pct": float(
                package["solve_summary"].get("returned_primary_delta_pct", float("nan"))
            ),
            "floor_accepted_tier": package["solve_summary"].get("floor_accepted_tier"),
            "used_physics_refinement_tier": package["solve_summary"].get("used_physics_refinement_tier"),
            "total_tiers_configured": int(package["solve_summary"].get("total_tiers_configured", 0)),
            "physics_refinement_attempt_count": int(
                package["solve_summary"].get("physics_refinement_attempt_count", 0)
            ),
            "physics_refinement_attempt_caps_eur": [
                float(value)
                for value in list(package["solve_summary"].get("physics_refinement_attempt_caps_eur", []) or [])
            ],
            "physics_refinement_cap_utilization": float(
                package["solve_summary"].get("physics_refinement_cap_utilization", float("nan"))
            ),
            "branch_l_gap_ratio_to_floor": float(
                package["solve_summary"].get("branch_l_gap_ratio_to_floor", float("nan"))
            ),
            "returned_mean_abs_solver_feeder_gap_kw": float(
                package["solve_summary"].get("returned_mean_abs_solver_feeder_gap_kw", float("nan"))
            ),
            "returned_max_solver_feeder_gap_kw": float(
                package["solve_summary"].get("returned_max_solver_feeder_gap_kw", float("nan"))
            ),
            "returned_mean_abs_export_gap_ratio": float(
                package["solve_summary"].get("returned_mean_abs_export_gap_ratio", float("nan"))
            ),
            "high_budget_refinement_warn": bool(
                package["solve_summary"].get("high_budget_refinement_warn", False)
            ),
            "returned_solution_source": str(package["solve_summary"].get("returned_solution_source", "stage1")),
            "formulation_tightening_required": bool(
                package["solve_summary"].get("formulation_tightening_required", False)
            ),
            "negative_floor_delta_warn": bool(package["solve_summary"].get("negative_floor_delta_warn", False)),
            "refinement_status_counts": {
                str(key): int(value)
                for key, value in dict(package["solve_summary"].get("refinement_status_counts", {}) or {}).items()
            },
            "validation_root_power_source": str(package["manifest"].get("validation_root_power_source", "")),
            "agent_q_base_zeroed": bool(package["manifest"].get("agent_q_base_zeroed", False)),
            "diagnostics_stage": str(package["manifest"].get("diagnostics_stage", "")),
            "diagnostics_floor_stage": str(package["manifest"].get("diagnostics_floor_stage", "")),
            "chunk_summaries": [dict(item) for item in list(package["solve_summary"].get("chunk_summaries", []) or [])],
        }
    )
    replay_artifacts["package"] = package
    replay_artifacts["rollout"] = rollout
    return replay_artifacts

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


__all__ = [
    "build_chunk_boundary_soc_df",
    "build_root_q_diagnostic_df",
    "build_soc_relaxation_diagnostics",
    "build_misocp_plan_package",
    "build_misocp_validation_artifacts",
    "build_misocp_validation_df",
    "build_debug_tables",
    "build_full_horizon_step_df",
    "build_simultaneous_diagnostic_tables",
    "build_voltage_df",
    "estimate_global_misocp_model_size",
    "expand_episode_indices",
    "format_solver_summary",
    "load_misocp_plan_package",
    "plot_full_horizon_net_load",
    "plot_full_horizon_power_balance",
    "plot_full_horizon_voltage",
    "plot_misocp_validation_scatter_panel",
    "plot_root_exchange_alignment",
    "replay_misocp_plan_package",
    "resolve_latest_compatible_misocp_plan_package_dir",
    "save_misocp_plan_package",
    "summarize_misocp_validation",
    "validate_misocp_result_schema",
]
