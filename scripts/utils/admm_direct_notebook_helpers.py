"""Direct day-ahead optimization helpers for the ADMM notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.builder import build_env
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.grid_surrogate_notebook_helpers import (
    GridTrafoSurrogate,
    _battery_to_netload_sensitivity,
    _get_grid_projector,
    _projector_to_numpy,
)


_THROUGHPUT_TIEBREAKER_EUR_PER_KWH = 1e-8
_NPOS_TIEBREAKER_EUR_PER_KWH = 1e-9
ADMM_DIRECT_PERFECT_LABEL = "ADMM Direct + Perfect Forecast"


def _load_gurobi():
    import gurobipy as gp

    return gp, gp.GRB


def _create_model(gp: Any, name: str):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    return model


def _wrap_gurobi_error(detail: str, exc: Exception) -> RuntimeError:
    error = RuntimeError(f"Direct day optimization requires a working Gurobi installation/license: {detail}: {exc}")
    error.__cause__ = exc
    return error


@dataclass(frozen=True)
class DirectDayProblemData:
    timestamps: tuple[str, ...]
    wholesale_price_eur_per_kwh: np.ndarray
    import_price_eur_per_kwh: np.ndarray
    load_kw: np.ndarray
    pv_kw: np.ndarray
    battery_capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    eff_charge: np.ndarray
    eff_discharge: np.ndarray
    energy_init_kwh: np.ndarray
    energy_min_kwh: np.ndarray
    energy_max_kwh: np.ndarray
    export_subsidy_eur_per_kwh: float
    import_price_adder_eur_per_kwh: float
    dt_hours: float

    @property
    def horizon(self) -> int:
        return int(self.import_price_eur_per_kwh.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.load_kw.shape[0])


@dataclass(frozen=True)
class DirectDaySolution:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    pv_effective_kw: np.ndarray
    net_load_kw: np.ndarray
    grid_import_kw: np.ndarray
    grid_export_kw: np.ndarray
    energy_kwh: np.ndarray
    objective_eur: float
    max_export_violation_kw: float
    solver_status: str
    solver_metadata: dict[str, object] = field(default_factory=dict)
    surrogate_root_p_kw: np.ndarray | None = None
    pp_root_p_kw: np.ndarray | None = None
    reference_only: bool = False
    note: str = ""


@dataclass(frozen=True)
class DirectDayAdmmResult:
    solution: DirectDaySolution
    history_df: pd.DataFrame
    converged: bool
    final_primal_residual: float
    final_dual_residual: float
    iterations: int
    objective_gap_vs_centralized_eur: float
    objective_gap_vs_centralized_pct: float


@dataclass(frozen=True)
class _LocalAgentSolveResult:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    net_load_kw: np.ndarray
    n_pos_kw: np.ndarray
    energy_kwh: np.ndarray
    contribution_kw: np.ndarray
    solve_time_sec: float
    status: str


def _normalize_date(value: str | int | None) -> str:
    if value is None:
        raise ValueError("test_start_date/test_end_date must not be None.")
    return str(pd.Timestamp(str(value)).date())


def _day_episode_length(dt_hours: float) -> int:
    resolved = int(round(24.0 / float(dt_hours)))
    if abs(float(dt_hours) * resolved - 24.0) > 1e-9:
        raise ValueError(
            "Direct day optimization requires dt in hours with one full day per episode, "
            f"got dt_hours={dt_hours} and resolved episode_length={resolved}."
        )
    return resolved


def _coerce_1d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array.astype(np.float32, copy=False)


def _coerce_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


def _objective_from_net_load(
    net_load_kw: np.ndarray,
    import_price_eur_per_kwh: np.ndarray,
    export_subsidy_eur_per_kwh: float,
    dt_hours: float,
) -> float:
    net_load = _coerce_2d(net_load_kw, name="net_load_kw")
    import_price = _coerce_1d(import_price_eur_per_kwh, name="import_price_eur_per_kwh")
    if net_load.shape[1] != import_price.size:
        raise ValueError(
            f"net_load_kw horizon {net_load.shape[1]} does not match import_price horizon {import_price.size}."
        )
    grid_import_kw = np.maximum(net_load, 0.0)
    grid_export_kw = np.maximum(-net_load, 0.0)
    return float(
        np.sum(
            float(dt_hours)
            * (
                import_price.reshape(1, -1) * grid_import_kw
                - float(export_subsidy_eur_per_kwh) * grid_export_kw
            )
        )
    )


def _baseline_net_load_kw(data: DirectDayProblemData) -> np.ndarray:
    return (np.asarray(data.load_kw, dtype=np.float32) - np.asarray(data.pv_kw, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )


def _surrogate_root_offset_kw(data: DirectDayProblemData, surrogate: GridTrafoSurrogate) -> np.ndarray:
    baseline_net_load = _baseline_net_load_kw(data)
    alpha_window = _coerce_2d(surrogate.alpha_netload_window_kw, name="alpha_netload_window_kw")
    baseline_root = _coerce_1d(surrogate.baseline_root_p_kw, name="baseline_root_p_kw")
    return (
        baseline_root - np.sum(alpha_window * baseline_net_load, axis=0)
    ).astype(np.float32, copy=False)


def compute_surrogate_root_p_kw(
    net_load_kw: np.ndarray,
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
) -> np.ndarray:
    net_load = _coerce_2d(net_load_kw, name="net_load_kw")
    alpha_window = _coerce_2d(surrogate.alpha_netload_window_kw, name="alpha_netload_window_kw")
    if net_load.shape != alpha_window.shape:
        raise ValueError(f"net_load_kw shape {net_load.shape} must match alpha_netload_window_kw {alpha_window.shape}.")
    offset_kw = _surrogate_root_offset_kw(data, surrogate)
    return (offset_kw + np.sum(alpha_window * net_load, axis=0)).astype(np.float32, copy=False)


def _build_power_balance_step_df(
    solution: DirectDaySolution,
    data: DirectDayProblemData,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(list(data.timestamps))
    load_total_kw = np.sum(np.asarray(data.load_kw, dtype=np.float32), axis=0)
    pv_raw_total_kw = np.sum(np.asarray(data.pv_kw, dtype=np.float32), axis=0)
    pv_curtail_total_kw = np.sum(np.asarray(solution.pv_curtail_kw, dtype=np.float32), axis=0)
    pv_effective_total_kw = np.sum(np.asarray(solution.pv_effective_kw, dtype=np.float32), axis=0)
    battery_charge_total_kw = np.sum(np.asarray(solution.charge_kw, dtype=np.float32), axis=0)
    battery_discharge_total_kw = np.sum(np.asarray(solution.discharge_kw, dtype=np.float32), axis=0)
    grid_import_total_kw = np.sum(np.asarray(solution.grid_import_kw, dtype=np.float32), axis=0)
    grid_export_total_kw = np.sum(np.asarray(solution.grid_export_kw, dtype=np.float32), axis=0)
    baseline_net_load_total_kw = np.sum(_baseline_net_load_kw(data), axis=0)
    post_curtail_net_load_total_kw = np.sum(
        np.asarray(data.load_kw, dtype=np.float32) - np.asarray(solution.pv_effective_kw, dtype=np.float32),
        axis=0,
    )
    post_action_net_load_total_kw = np.sum(np.asarray(solution.net_load_kw, dtype=np.float32), axis=0)
    power_balance_residual_kw = (
        load_total_kw
        + battery_charge_total_kw
        + grid_export_total_kw
        + pv_curtail_total_kw
        - pv_raw_total_kw
        - grid_import_total_kw
        - battery_discharge_total_kw
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_total_kw": load_total_kw.astype(np.float32, copy=False),
            "pv_raw_total_kw": pv_raw_total_kw.astype(np.float32, copy=False),
            "pv_effective_total_kw": pv_effective_total_kw.astype(np.float32, copy=False),
            "pv_curtail_total_kw": pv_curtail_total_kw.astype(np.float32, copy=False),
            "battery_charge_total_kw": battery_charge_total_kw.astype(np.float32, copy=False),
            "battery_discharge_total_kw": battery_discharge_total_kw.astype(np.float32, copy=False),
            "grid_import_total_kw": grid_import_total_kw.astype(np.float32, copy=False),
            "grid_export_total_kw": grid_export_total_kw.astype(np.float32, copy=False),
            "baseline_net_load_total_kw": baseline_net_load_total_kw.astype(np.float32, copy=False),
            "post_curtail_net_load_total_kw": post_curtail_net_load_total_kw.astype(np.float32, copy=False),
            "post_action_net_load_total_kw": post_action_net_load_total_kw.astype(np.float32, copy=False),
            "power_balance_residual_kw": power_balance_residual_kw.astype(np.float32, copy=False),
        }
    )


def _rollout_step_indices(data: DirectDayProblemData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_length = _day_episode_length(float(data.dt_hours))
    if data.horizon % episode_length != 0:
        raise ValueError(
            "Direct-day compare replay expects the horizon to contain whole-day episodes, "
            f"got horizon={data.horizon} and day_episode_length={episode_length}."
        )
    global_step = np.arange(data.horizon, dtype=np.int32)
    episode_idx = (global_step // episode_length).astype(np.int32, copy=False)
    step_in_episode = (global_step % episode_length).astype(np.int32, copy=False)
    return global_step, episode_idx, step_in_episode


def _build_direct_day_replay_rollout(
    cfg,
    env,
    data: DirectDayProblemData,
    solution: DirectDaySolution,
    *,
    label: str,
) -> grid_nb.RolloutResult:
    grid_core = getattr(env, "_grid_core", None)
    if grid_core is None:
        raise ValueError("Environment does not expose _grid_core for direct-day compare replay.")
    reward_fn = getattr(env, "reward_fn", None)
    if reward_fn is None or not hasattr(reward_fn, "compute"):
        raise ValueError("Environment must expose reward_fn.compute for direct-day compare replay.")
    if int(getattr(env, "n", data.n_agents)) != data.n_agents:
        raise ValueError(f"env.n should match data.n_agents={data.n_agents}.")

    power_balance_df = _build_power_balance_step_df(solution, data)
    timestamps = pd.to_datetime(list(data.timestamps))
    global_step, episode_idx_arr, step_arr = _rollout_step_indices(data)
    bus_ids = [int(bus_id) for bus_id in getattr(getattr(grid_core, "net", None), "bus").index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(grid_core, "agent_bus_ids", getattr(cfg.grid, "agent_bus_ids", []))]
    agent_bus_set = set(agent_bus_ids)
    agent_profiles = [str(profile) for profile in list(getattr(cfg.data, "agent_profiles", []))]
    if len(agent_profiles) != data.n_agents:
        raise ValueError(
            f"cfg.data.agent_profiles should contain {data.n_agents} entries, got {len(agent_profiles)}."
        )

    fixed_load_kw, fixed_generation_kw = grid_nb._fixed_feeder_components_kw(env)
    loading_limit_pct = float(cfg.grid.line_max_loading_pct)
    trafo_limit_kw = grid_nb._approx_trafo_limit_kw(env, loading_limit_pct=loading_limit_pct)
    export_subsidy = float(data.export_subsidy_eur_per_kwh)
    dt_hours = float(data.dt_hours)
    p_max_kw = np.maximum(np.asarray(data.p_max_kw, dtype=np.float32), 1e-6)
    battery_capacity_kwh = np.maximum(np.asarray(data.battery_capacity_kwh, dtype=np.float32), 1e-6)

    grid_core.reset(
        np.asarray(data.load_kw[:, 0] - solution.pv_effective_kw[:, 0], dtype=np.float32),
        np.asarray(solution.pv_effective_kw[:, 0], dtype=np.float32),
    )

    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    for step_idx in range(data.horizon):
        timestamp = timestamps[step_idx]
        load_t = np.asarray(data.load_kw[:, step_idx], dtype=np.float32)
        pv_raw_t = np.asarray(data.pv_kw[:, step_idx], dtype=np.float32)
        pv_effective_t = np.asarray(solution.pv_effective_kw[:, step_idx], dtype=np.float32)
        pv_curtail_t = np.asarray(solution.pv_curtail_kw[:, step_idx], dtype=np.float32)
        charge_t = np.asarray(solution.charge_kw[:, step_idx], dtype=np.float32)
        discharge_t = np.asarray(solution.discharge_kw[:, step_idx], dtype=np.float32)
        net_load_t = np.asarray(solution.net_load_kw[:, step_idx], dtype=np.float32)
        battery_power_t = (charge_t - discharge_t).astype(np.float32, copy=False)
        grid_import_t = np.maximum(net_load_t, 0.0).astype(np.float32, copy=False)
        grid_export_t = np.maximum(-net_load_t, 0.0).astype(np.float32, copy=False)
        base_net_load_raw_t = (load_t - pv_raw_t).astype(np.float32, copy=False)
        base_net_load_effective_t = (load_t - pv_effective_t).astype(np.float32, copy=False)
        pv_utilization_t = np.ones_like(pv_raw_t, dtype=np.float32)
        valid_pv_mask = pv_raw_t > 1e-6
        pv_utilization_t[valid_pv_mask] = (pv_effective_t[valid_pv_mask] / pv_raw_t[valid_pv_mask]).astype(
            np.float32,
            copy=False,
        )
        battery_action_t = np.clip(battery_power_t / p_max_kw, -1.0, 1.0).astype(np.float32, copy=False)
        pv_action_t = np.clip(2.0 * pv_utilization_t - 1.0, -1.0, 1.0).astype(np.float32, copy=False)

        pf_result = grid_core.step(
            p_batt_kw=battery_power_t,
            base_load_kw=base_net_load_effective_t,
        )
        line_loading_pct = np.asarray(getattr(pf_result, "line_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        trafo_loading_pct = np.asarray(getattr(pf_result, "trafo_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
        vm_pu = np.asarray(getattr(pf_result, "vm_pu", np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32)
        if vm_pu.shape[0] != len(bus_ids):
            raise ValueError(f"pf_result.vm_pu length {vm_pu.shape[0]} should match n_buses={len(bus_ids)}.")
        trafo_p_signed_kw = np.asarray(getattr(pf_result, "trafo_p_signed_kw", np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)
        pp_root_p_kw = float(np.sum(trafo_p_signed_kw)) if trafo_p_signed_kw.size else 0.0

        reward_state = {
            "price_t": float(data.import_price_eur_per_kwh[step_idx]),
            "wholesale_price_t": float(data.wholesale_price_eur_per_kwh[step_idx]),
            "import_price_t": float(data.import_price_eur_per_kwh[step_idx]),
            "net_load_t": base_net_load_raw_t,
            "actual_grid_power_t": net_load_t,
            "dt": dt_hours,
            "v_violation": np.asarray(getattr(pf_result, "v_violation", np.zeros(data.n_agents, dtype=np.float32)), dtype=np.float32),
            "psi_v_raw": float(getattr(pf_result, "psi_v_raw", 0.0)),
            "psi_line_raw": float(getattr(pf_result, "psi_line_raw", 0.0)),
            "psi_trafo_raw": float(getattr(pf_result, "psi_trafo_raw", 0.0)),
        }
        _, reward_components = reward_fn.compute(reward_state)
        voltage_penalty_per_agent = np.asarray(
            reward_components.get("r_safe_v", np.zeros(data.n_agents, dtype=np.float32)),
            dtype=np.float32,
        )
        line_penalty_per_agent = np.asarray(
            reward_components.get("r_safe_line", np.zeros(data.n_agents, dtype=np.float32)),
            dtype=np.float32,
        )
        trafo_penalty_per_agent = np.asarray(
            reward_components.get("r_safe_trafo", np.zeros(data.n_agents, dtype=np.float32)),
            dtype=np.float32,
        )

        purchase_cost_per_agent = (
            grid_import_t * np.float32(dt_hours) * np.float32(data.import_price_eur_per_kwh[step_idx])
        ).astype(np.float32, copy=False)
        export_subsidy_per_agent = (
            grid_export_t * np.float32(dt_hours) * np.float32(export_subsidy)
        ).astype(np.float32, copy=False)
        objective_per_agent = (purchase_cost_per_agent - export_subsidy_per_agent).astype(np.float32, copy=False)

        simultaneous_kw = np.minimum(charge_t, discharge_t).astype(np.float32, copy=False)
        step_rows.append(
            {
                "controller": label,
                "episode_idx": int(episode_idx_arr[step_idx]),
                "step": int(step_arr[step_idx]),
                "global_step": int(global_step[step_idx]),
                "timestamp": timestamp,
                "price": float(data.import_price_eur_per_kwh[step_idx]),
                "price_pred": float(data.import_price_eur_per_kwh[step_idx]),
                "wholesale_price": float(data.wholesale_price_eur_per_kwh[step_idx]),
                "import_price": float(data.import_price_eur_per_kwh[step_idx]),
                "base_net_load_total": float(np.sum(base_net_load_raw_t)),
                "base_net_load_effective_total": float(np.sum(base_net_load_effective_t)),
                "net_load_total": float(np.sum(net_load_t)),
                "agent_raw_net_load_kw": float(np.sum(base_net_load_raw_t)),
                "agent_effective_net_load_kw": float(np.sum(base_net_load_effective_t)),
                "agent_post_action_net_load_kw": float(np.sum(net_load_t)),
                "fixed_load_kw": float(fixed_load_kw),
                "fixed_generation_kw": float(fixed_generation_kw),
                "feeder_raw_net_load_kw": float(np.sum(base_net_load_raw_t) + fixed_load_kw - fixed_generation_kw),
                "feeder_effective_net_load_kw": float(np.sum(base_net_load_effective_t) + fixed_load_kw - fixed_generation_kw),
                "feeder_post_action_net_load_kw": float(np.sum(net_load_t) + fixed_load_kw - fixed_generation_kw),
                "load_total": float(np.sum(load_t)),
                "pv_raw_total": float(np.sum(pv_raw_t)),
                "pv_effective_total": float(np.sum(pv_effective_t)),
                "pv_curtail_total": float(np.sum(pv_curtail_t)),
                "grid_import_total": float(np.sum(grid_import_t)),
                "grid_export_total": float(np.sum(grid_export_t)),
                "battery_charge_total": float(np.sum(charge_t)),
                "battery_charge_req_total": float(np.sum(charge_t)),
                "battery_discharge_total": float(np.sum(discharge_t)),
                "battery_discharge_req_total": float(np.sum(discharge_t)),
                "pv_curtail_req_total": float(np.sum(pv_curtail_t)),
                "battery_request_gap_kw_total": 0.0,
                "pv_curtail_request_gap_kw_total": 0.0,
                "projector_adjustment_kw_total": 0.0,
                "purchase_cost_total": float(np.sum(purchase_cost_per_agent)),
                "export_subsidy_total": float(np.sum(export_subsidy_per_agent)),
                "objective_total": float(np.sum(objective_per_agent)),
                "soc_penalty_total": 0.0,
                "voltage_penalty_total": float(np.mean(voltage_penalty_per_agent)) if voltage_penalty_per_agent.size else 0.0,
                "line_penalty_total": float(np.mean(line_penalty_per_agent)) if line_penalty_per_agent.size else 0.0,
                "trafo_penalty_total": float(np.mean(trafo_penalty_per_agent)) if trafo_penalty_per_agent.size else 0.0,
                "psi_v_raw": float(getattr(pf_result, "psi_v_raw", 0.0)),
                "psi_line_raw": float(getattr(pf_result, "psi_line_raw", 0.0)),
                "psi_trafo_raw": float(getattr(pf_result, "psi_trafo_raw", 0.0)),
                "line_loading_pct_max": float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0,
                "trafo_loading_pct_max": float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0,
                "line_violation": float(getattr(pf_result, "line_violation", 0.0)),
                "trafo_violation": float(getattr(pf_result, "trafo_violation", 0.0)),
                "n_line_violations": int(np.any(line_loading_pct > loading_limit_pct)) if line_loading_pct.size else 0,
                "n_trafo_violations": int(np.any(trafo_loading_pct > loading_limit_pct)) if trafo_loading_pct.size else 0,
                "voltage_violation_count": int(((vm_pu < float(cfg.grid.v_min_pu)) | (vm_pu > float(cfg.grid.v_max_pu))).sum()),
                "controller_action_gap_total": 0.0,
                "soc_penalty_step_total": 0.0,
                "misocp_fallback": 0.0,
                "misocp_time_limit_feasible": 0.0,
                "solve_time_sec": float("nan"),
                "root_import_kw": float(max(pp_root_p_kw, 0.0)),
                "root_export_kw": float(max(-pp_root_p_kw, 0.0)),
                "pp_root_p_kw": float(pp_root_p_kw),
                "simultaneous_charge_discharge_kw_total": float(np.sum(simultaneous_kw)),
                "simultaneous_agent_count": int(np.sum(simultaneous_kw > 1e-6)),
                "simultaneous_step_flag": float(np.any(simultaneous_kw > 1e-6)),
                "trafo_limit_reference_kw": float(trafo_limit_kw) if trafo_limit_kw is not None else float("nan"),
                "pf_converged": bool(getattr(pf_result, "converged", True)),
            }
        )

        for agent_idx in range(data.n_agents):
            agent_rows.append(
                {
                    "controller": label,
                    "episode_idx": int(episode_idx_arr[step_idx]),
                    "step": int(step_arr[step_idx]),
                    "global_step": int(global_step[step_idx]),
                    "timestamp": timestamp,
                    "agent_id": int(agent_idx),
                    "agent_profile": agent_profiles[agent_idx],
                    "load": float(load_t[agent_idx]),
                    "load_pred": float(load_t[agent_idx]),
                    "pv": float(pv_raw_t[agent_idx]),
                    "pv_raw": float(pv_raw_t[agent_idx]),
                    "pv_effective": float(pv_effective_t[agent_idx]),
                    "pv_curtail": float(pv_curtail_t[agent_idx]),
                    "pv_curtail_req": float(pv_curtail_t[agent_idx]),
                    "pv_utilization": float(pv_utilization_t[agent_idx]),
                    "pv_pred": float(pv_raw_t[agent_idx]),
                    "base_net_load": float(base_net_load_raw_t[agent_idx]),
                    "base_net_load_effective": float(base_net_load_effective_t[agent_idx]),
                    "net_load": float(net_load_t[agent_idx]),
                    "grid_import_kw": float(grid_import_t[agent_idx]),
                    "grid_export_kw": float(grid_export_t[agent_idx]),
                    "e_bat": float(battery_power_t[agent_idx]),
                    "e_bat_req": float(battery_power_t[agent_idx]),
                    "battery_action_req": float(battery_action_t[agent_idx]),
                    "battery_action_exec": float(battery_action_t[agent_idx]),
                    "pv_action_req": float(pv_action_t[agent_idx]),
                    "pv_action_exec": float(pv_action_t[agent_idx]),
                    "controller_action_gap": 0.0,
                    "battery_request_gap_kw": 0.0,
                    "pv_curtail_request_gap_kw": 0.0,
                    "soc_penalty_unweighted": 0.0,
                    "r_soc_pen": 0.0,
                    "r_safe_v": float(voltage_penalty_per_agent[agent_idx]),
                    "r_safe_line": float(line_penalty_per_agent[agent_idx]),
                    "r_safe_trafo": float(trafo_penalty_per_agent[agent_idx]),
                    "soc": float(solution.energy_kwh[agent_idx, step_idx + 1] / battery_capacity_kwh[agent_idx]),
                    "purchase_cost": float(purchase_cost_per_agent[agent_idx]),
                    "export_subsidy": float(export_subsidy_per_agent[agent_idx]),
                    "objective_total": float(objective_per_agent[agent_idx]),
                }
            )

        for bus_id, vm_value in zip(bus_ids, vm_pu, strict=False):
            grid_rows.append(
                {
                    "controller": label,
                    "episode_idx": int(episode_idx_arr[step_idx]),
                    "step": int(step_arr[step_idx]),
                    "global_step": int(global_step[step_idx]),
                    "timestamp": timestamp,
                    "bus_id": int(bus_id),
                    "vm_pu": float(vm_value),
                    "is_agent_bus": bool(int(bus_id) in agent_bus_set),
                }
            )

    step_df = pd.DataFrame(step_rows).sort_values(["global_step"]).reset_index(drop=True)
    agent_df = pd.DataFrame(agent_rows).sort_values(["global_step", "agent_id"]).reset_index(drop=True)
    grid_df = pd.DataFrame(grid_rows).sort_values(["global_step", "bus_id"]).reset_index(drop=True)
    summary = (
        agent_df.groupby(["controller", "agent_profile"], as_index=False)[["purchase_cost", "export_subsidy", "objective_total"]].sum()
        if not agent_df.empty
        else pd.DataFrame(columns=["controller", "agent_profile", "purchase_cost", "export_subsidy", "objective_total"])
    )
    meta = {
        "controller": label,
        "n_agents": int(data.n_agents),
        "agent_profiles": agent_profiles,
        "agent_bus_ids": agent_bus_ids,
        "bus_ids": bus_ids,
        "v_min_pu": float(cfg.grid.v_min_pu),
        "v_max_pu": float(cfg.grid.v_max_pu),
        "future_horizon": int(cfg.env.future_horizon),
        "prediction_mode": grid_nb.PERFECT_PREDICTION_MODE,
        "evaluation_mode": grid_nb.resolve_evaluation_mode(grid_nb.PERFECT_PREDICTION_MODE),
        "dt_hours": float(data.dt_hours),
        "trafo_limit_kw": trafo_limit_kw,
        "trafo_limit_note": "Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.",
        "loading_limit_pct": loading_limit_pct,
        "trafo_loading_limit_pct": loading_limit_pct,
        "export_subsidy_eur_per_kwh": export_subsidy,
        "import_price_adder_eur_per_kwh": float(data.import_price_adder_eur_per_kwh),
        "soc_mode": "continuous",
        "economics_scope": "agent_only",
        "penalty_source": "replay_derived",
    }
    return grid_nb.RolloutResult(
        step_df=step_df,
        agent_df=agent_df,
        grid_df=grid_df,
        summary=summary,
        meta=meta,
    )


def compute_direct_day_grid_profile(
    env,
    data: DirectDayProblemData,
    solution: DirectDaySolution,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_core = getattr(env, "_grid_core", None)
    if grid_core is None:
        raise ValueError("Environment does not expose _grid_core for pandapower evaluation.")
    if not hasattr(getattr(grid_core, "net", None), "bus"):
        raise ValueError("grid_core.net.bus is required to build direct-day grid diagnostics.")

    step_df = _build_power_balance_step_df(solution, data)
    bus_ids = [int(bus_id) for bus_id in grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(grid_core, "agent_bus_ids", [])]
    agent_bus_set = set(agent_bus_ids)

    base_load0 = np.asarray(data.load_kw[:, 0] - solution.pv_effective_kw[:, 0], dtype=np.float32)
    grid_core.reset(base_load0, np.asarray(solution.pv_effective_kw[:, 0], dtype=np.float32))

    root_p_kw = np.zeros((data.horizon,), dtype=np.float32)
    grid_rows: list[dict[str, object]] = []
    for step_idx in range(data.horizon):
        base_load_kw = np.asarray(data.load_kw[:, step_idx] - solution.pv_effective_kw[:, step_idx], dtype=np.float32)
        p_batt_kw = np.asarray(solution.charge_kw[:, step_idx] - solution.discharge_kw[:, step_idx], dtype=np.float32)
        pf_result = grid_core.step(p_batt_kw=p_batt_kw, base_load_kw=base_load_kw)
        trafo_p_signed_kw = np.asarray(getattr(pf_result, "trafo_p_signed_kw", np.zeros(1, dtype=np.float32)), dtype=np.float32)
        root_p_kw[step_idx] = float(trafo_p_signed_kw.reshape(-1)[0]) if trafo_p_signed_kw.size else 0.0

        vm_pu = np.asarray(getattr(pf_result, "vm_pu", np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32)
        if vm_pu.shape[0] != len(bus_ids):
            raise ValueError(
                f"pf_result.vm_pu should have one value per bus, got {vm_pu.shape[0]} for {len(bus_ids)} buses."
            )
        for bus_id, vm_value in zip(bus_ids, vm_pu, strict=False):
            grid_rows.append(
                {
                    "timestamp": step_df.loc[step_idx, "timestamp"],
                    "bus_id": int(bus_id),
                    "vm_pu": float(vm_value),
                    "is_agent_bus": bool(int(bus_id) in agent_bus_set),
                }
            )

    step_df = step_df.copy()
    step_df["pp_root_p_kw"] = root_p_kw.astype(np.float32, copy=False)
    grid_df = pd.DataFrame(grid_rows).sort_values(["timestamp", "bus_id"]).reset_index(drop=True)
    return step_df, grid_df


def compute_direct_day_pp_root_p_kw(
    env,
    data: DirectDayProblemData,
    solution: DirectDaySolution,
) -> np.ndarray:
    step_df, _ = compute_direct_day_grid_profile(env, data, solution)
    return step_df["pp_root_p_kw"].to_numpy(dtype=np.float32, copy=True)


def _max_export_violation_kw(root_p_kw: np.ndarray, trafo_limit_kw: float) -> float:
    root_p = _coerce_1d(root_p_kw, name="root_p_kw")
    return float(np.max(np.maximum(-root_p - float(trafo_limit_kw), 0.0))) if root_p.size else 0.0


def _status_label(model, grb) -> str:
    mapping = {
        int(grb.OPTIMAL): "optimal",
        int(grb.SUBOPTIMAL): "suboptimal",
        int(grb.INFEASIBLE): "infeasible",
        int(grb.INF_OR_UNBD): "inf_or_unbd",
        int(grb.UNBOUNDED): "unbounded",
        int(grb.TIME_LIMIT): "time_limit",
    }
    return mapping.get(int(model.Status), f"status_{int(model.Status)}")


def _build_solution(
    *,
    charge_kw: np.ndarray,
    discharge_kw: np.ndarray,
    pv_curtail_kw: np.ndarray,
    energy_kwh: np.ndarray,
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
    solver_status: str,
    solver_metadata: dict[str, object] | None = None,
    reference_only: bool = False,
    note: str = "",
    pp_root_p_kw: np.ndarray | None = None,
) -> DirectDaySolution:
    charge = _coerce_2d(charge_kw, name="charge_kw")
    discharge = _coerce_2d(discharge_kw, name="discharge_kw")
    pv_curtail = _coerce_2d(pv_curtail_kw, name="pv_curtail_kw")
    energy = _coerce_2d(energy_kwh, name="energy_kwh")
    if charge.shape != (data.n_agents, data.horizon):
        raise ValueError(f"charge_kw must have shape {(data.n_agents, data.horizon)}, got {charge.shape}.")
    if discharge.shape != charge.shape or pv_curtail.shape != charge.shape:
        raise ValueError("discharge_kw and pv_curtail_kw must match charge_kw shape.")
    if energy.shape != (data.n_agents, data.horizon + 1):
        raise ValueError(f"energy_kwh must have shape {(data.n_agents, data.horizon + 1)}, got {energy.shape}.")
    pv_effective = (np.asarray(data.pv_kw, dtype=np.float32) - pv_curtail).astype(np.float32, copy=False)
    net_load = (np.asarray(data.load_kw, dtype=np.float32) - pv_effective + charge - discharge).astype(
        np.float32,
        copy=False,
    )
    root_p_kw = compute_surrogate_root_p_kw(net_load, data, surrogate)
    objective_eur = _objective_from_net_load(
        net_load,
        import_price_eur_per_kwh=data.import_price_eur_per_kwh,
        export_subsidy_eur_per_kwh=data.export_subsidy_eur_per_kwh,
        dt_hours=data.dt_hours,
    )
    metadata = dict(solver_metadata or {})
    metadata.setdefault("dt_hours", float(data.dt_hours))
    return DirectDaySolution(
        charge_kw=charge,
        discharge_kw=discharge,
        pv_curtail_kw=pv_curtail,
        pv_effective_kw=pv_effective,
        net_load_kw=net_load,
        grid_import_kw=np.maximum(net_load, 0.0).astype(np.float32, copy=False),
        grid_export_kw=np.maximum(-net_load, 0.0).astype(np.float32, copy=False),
        energy_kwh=energy,
        objective_eur=float(objective_eur),
        max_export_violation_kw=_max_export_violation_kw(root_p_kw, surrogate.trafo_limit_kw),
        solver_status=str(solver_status),
        solver_metadata=metadata,
        surrogate_root_p_kw=root_p_kw,
        pp_root_p_kw=None if pp_root_p_kw is None else np.asarray(pp_root_p_kw, dtype=np.float32).copy(),
        reference_only=bool(reference_only),
        note=str(note),
    )


def _validate_prices(data: DirectDayProblemData) -> None:
    price_gap = np.asarray(data.import_price_eur_per_kwh, dtype=np.float32) - np.float32(data.export_subsidy_eur_per_kwh)
    min_gap = float(np.min(price_gap))
    if min_gap < -1e-9:
        raise ValueError(
            "Direct day hinge economics requires import_price_eur_per_kwh >= export_subsidy_eur_per_kwh, "
            f"got min(import_price - subsidy)={min_gap:.6f}."
        )
    if min_gap < 1e-3:
        warnings.warn(
            "import_price_eur_per_kwh is very close to export_subsidy_eur_per_kwh; "
            "the hinge auxiliary n_pos may become numerically weakly identified.",
            RuntimeWarning,
            stacklevel=2,
        )


def build_direct_day_problem_data(
    cfg,
    *,
    test_start_date: str | int,
    test_end_date: str | int,
) -> DirectDayProblemData:
    start_date = _normalize_date(test_start_date)
    end_date = _normalize_date(test_end_date)
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError(f"test_start_date must be <= test_end_date, got {start_date} > {end_date}.")
    expected_episode_length = _day_episode_length(float(cfg.env.dt))
    if int(cfg.env.episode_limit) != expected_episode_length:
        raise ValueError(
            "Direct day optimization expects cfg.env.episode_limit to match one full day, "
            f"got episode_limit={cfg.env.episode_limit}, expected {expected_episode_length}."
        )

    env = build_env(cfg, mode="test")
    try:
        if int(getattr(env, "num_available_episodes", 0)) < 1:
            raise ValueError("Current test window does not contain any episodes.")
        selected: list[tuple[pd.Timestamp, list[str], np.ndarray, np.ndarray, np.ndarray]] = []
        for episode_idx in range(int(env.num_available_episodes)):
            _, reset_info = env.reset(episode_idx=episode_idx)
            episode_meta = dict(reset_info.get("episode_meta", {}))
            timestamps = [str(value) for value in list(episode_meta.get("timestamps") or [])]
            if len(timestamps) != expected_episode_length:
                raise ValueError(
                    f"Episode {episode_idx} should contain {expected_episode_length} timestamps, got {len(timestamps)}."
                )
            first_timestamp = pd.Timestamp(timestamps[0])
            first_date = str(first_timestamp.date())
            if not (start_date <= first_date <= end_date):
                continue
            load_kw = np.asarray(env.ep_load, dtype=np.float32)
            pv_kw = np.asarray(env.ep_pv, dtype=np.float32)
            wholesale_price = np.asarray(env.ep_price, dtype=np.float32).reshape(-1)
            if load_kw.shape != (expected_episode_length, int(env.n)):
                raise ValueError(f"env.ep_load must have shape {(expected_episode_length, int(env.n))}, got {load_kw.shape}.")
            if pv_kw.shape != load_kw.shape:
                raise ValueError(f"env.ep_pv shape {pv_kw.shape} must match env.ep_load shape {load_kw.shape}.")
            if wholesale_price.size != expected_episode_length:
                raise ValueError(
                    f"env.ep_price must have length {expected_episode_length}, got {wholesale_price.size}."
                )
            selected.append(
                (
                    first_timestamp,
                    timestamps,
                    wholesale_price.astype(np.float32, copy=True),
                    load_kw.T.astype(np.float32, copy=True),
                    pv_kw.T.astype(np.float32, copy=True),
                )
            )

        if not selected:
            raise ValueError(f"No test episodes found between {start_date} and {end_date}.")

        selected.sort(key=lambda item: item[0])
        selected_dates = [str(item[0].date()) for item in selected]
        expected_dates = [str(value.date()) for value in pd.date_range(start_date, end_date, freq="D")]
        if selected_dates != expected_dates:
            raise ValueError(
                "Selected episodes do not cover the requested date range exactly, "
                f"got {selected_dates}, expected {expected_dates}."
            )

        timestamps = tuple(timestamp for _, episode_timestamps, _, _, _ in selected for timestamp in episode_timestamps)
        wholesale_price = np.concatenate([episode_price for _, _, episode_price, _, _ in selected], axis=0).astype(
            np.float32,
            copy=False,
        )
        load_kw = np.concatenate([episode_load for _, _, _, episode_load, _ in selected], axis=1).astype(
            np.float32,
            copy=False,
        )
        pv_kw = np.concatenate([episode_pv for _, _, _, _, episode_pv in selected], axis=1).astype(
            np.float32,
            copy=False,
        )
        import_price_adder = float(getattr(getattr(cfg, "reward", None), "import_price_adder_eur_per_kwh", 0.0))
        export_subsidy = float(getattr(getattr(cfg, "reward", None), "export_subsidy_eur_per_kwh", 0.079))
        data = DirectDayProblemData(
            timestamps=timestamps,
            wholesale_price_eur_per_kwh=wholesale_price,
            import_price_eur_per_kwh=(wholesale_price + np.float32(import_price_adder)).astype(np.float32, copy=False),
            load_kw=load_kw,
            pv_kw=pv_kw,
            battery_capacity_kwh=np.asarray(env.agent_c_bat, dtype=np.float32).reshape(-1).copy(),
            p_max_kw=np.asarray(env.agent_p_max, dtype=np.float32).reshape(-1).copy(),
            eff_charge=np.full((int(env.n),), float(env.eff), dtype=np.float32),
            eff_discharge=np.full((int(env.n),), float(env.eff), dtype=np.float32),
            energy_init_kwh=(float(env.init_soc) * np.asarray(env.agent_c_bat, dtype=np.float32)).astype(np.float32),
            energy_min_kwh=(float(env.soc_min) * np.asarray(env.agent_c_bat, dtype=np.float32)).astype(np.float32),
            energy_max_kwh=(float(env.soc_max) * np.asarray(env.agent_c_bat, dtype=np.float32)).astype(np.float32),
            export_subsidy_eur_per_kwh=export_subsidy,
            import_price_adder_eur_per_kwh=import_price_adder,
            dt_hours=float(env.dt),
        )
        if len(data.timestamps) != data.horizon or data.load_kw.shape[1] != data.horizon or data.pv_kw.shape[1] != data.horizon:
            raise ValueError("timestamps, price, load, and pv must share the same horizon.")
        _validate_prices(data)
        return data
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def build_direct_day_trafo_surrogate(
    cfg,
    env,
    data: DirectDayProblemData,
) -> GridTrafoSurrogate:
    projector = _get_grid_projector(cfg, env)
    alpha_netload_kw = _battery_to_netload_sensitivity(projector)
    alpha_window = np.repeat(alpha_netload_kw[:, None], data.horizon, axis=1).astype(np.float32)
    trafo_base_kw = _projector_to_numpy(getattr(projector, "trafo_power_base_kw")).reshape(-1)
    if trafo_base_kw.size == 0:
        raise ValueError("Direct day surrogate requires transformer baseline data.")
    baseline_net_load = _baseline_net_load_kw(data)
    baseline_root_p_kw = (float(trafo_base_kw[0]) + np.sum(alpha_window * baseline_net_load, axis=0)).astype(
        np.float32,
        copy=False,
    )
    trafo_limit_kw = grid_nb._approx_trafo_limit_kw(env, loading_limit_pct=float(cfg.grid.line_max_loading_pct))
    if trafo_limit_kw is None or not np.isfinite(float(trafo_limit_kw)):
        raise ValueError("Unable to resolve a transformer power reference limit for direct day optimization.")
    return GridTrafoSurrogate(
        trafo_limit_kw=float(trafo_limit_kw),
        alpha_netload_kw=alpha_netload_kw.astype(np.float32),
        alpha_netload_window_kw=alpha_window,
        baseline_root_p_kw=baseline_root_p_kw.astype(np.float32),
        export_overload_mask=(baseline_root_p_kw < -float(trafo_limit_kw) - 1e-6),
        import_overload_mask=(baseline_root_p_kw > float(trafo_limit_kw) + 1e-6),
    )


def compute_direct_day_baseline(
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
) -> DirectDaySolution:
    """Return the no-battery/no-curtailment reference.

    The baseline uses pv_curtail=0 and therefore may violate transformer export
    limits severely. Its objective is a comparison reference only, not a legal
    operating cost under export constraints.
    """

    charge_kw = np.zeros((data.n_agents, data.horizon), dtype=np.float32)
    discharge_kw = np.zeros_like(charge_kw)
    pv_curtail_kw = np.zeros_like(charge_kw)
    energy_kwh = np.repeat(data.energy_init_kwh[:, None], data.horizon + 1, axis=1).astype(np.float32)
    return _build_solution(
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        pv_curtail_kw=pv_curtail_kw,
        energy_kwh=energy_kwh,
        data=data,
        surrogate=surrogate,
        solver_status="baseline_reference_only",
        solver_metadata={"num_binary_vars": 0, "num_integer_vars": 0},
        reference_only=True,
        note="Baseline uses pv_curtail=0 and may violate export limits; economic objective is reference-only.",
    )


def _build_centralized_model(
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
    *,
    export_only: bool,
):
    try:
        gp, grb = _load_gurobi()
        model = _create_model(gp, "direct_day_centralized")
    except Exception as exc:
        raise _wrap_gurobi_error("unable to create centralized direct-day model", exc)

    T = data.horizon
    N = data.n_agents
    alpha_window = np.asarray(surrogate.alpha_netload_window_kw, dtype=np.float32)
    offset_kw = _surrogate_root_offset_kw(data, surrogate)

    charge = model.addVars(N, T, lb=0.0, name="charge_kw")
    discharge = model.addVars(N, T, lb=0.0, name="discharge_kw")
    pv_curtail = model.addVars(N, T, lb=0.0, name="pv_curtail_kw")
    net_load = model.addVars(N, T, lb=-grb.INFINITY, name="net_load_kw")
    n_pos = model.addVars(N, T, lb=0.0, name="n_pos_kw")
    energy = model.addVars(N, T + 1, lb=0.0, name="energy_kwh")

    objective = gp.LinExpr()
    for agent_idx in range(N):
        model.addConstr(energy[agent_idx, 0] == float(data.energy_init_kwh[agent_idx]), name=f"energy_init_{agent_idx}")
        model.addConstr(energy[agent_idx, T] == float(data.energy_init_kwh[agent_idx]), name=f"energy_terminal_{agent_idx}")
        for step_idx in range(T):
            model.addConstr(charge[agent_idx, step_idx] <= float(data.p_max_kw[agent_idx]), name=f"charge_ub_{agent_idx}_{step_idx}")
            model.addConstr(discharge[agent_idx, step_idx] <= float(data.p_max_kw[agent_idx]), name=f"discharge_ub_{agent_idx}_{step_idx}")
            model.addConstr(pv_curtail[agent_idx, step_idx] <= float(data.pv_kw[agent_idx, step_idx]), name=f"curtail_ub_{agent_idx}_{step_idx}")
            model.addConstr(
                net_load[agent_idx, step_idx]
                == float(data.load_kw[agent_idx, step_idx])
                - float(data.pv_kw[agent_idx, step_idx])
                + pv_curtail[agent_idx, step_idx]
                + charge[agent_idx, step_idx]
                - discharge[agent_idx, step_idx],
                name=f"net_load_balance_{agent_idx}_{step_idx}",
            )
            model.addConstr(n_pos[agent_idx, step_idx] >= net_load[agent_idx, step_idx], name=f"n_pos_lb_net_{agent_idx}_{step_idx}")
            model.addConstr(
                energy[agent_idx, step_idx + 1]
                == energy[agent_idx, step_idx]
                + float(data.eff_charge[agent_idx]) * float(data.dt_hours) * charge[agent_idx, step_idx]
                - float(data.dt_hours / max(float(data.eff_discharge[agent_idx]), 1e-6)) * discharge[agent_idx, step_idx],
                name=f"energy_balance_{agent_idx}_{step_idx}",
            )
            energy[agent_idx, step_idx].LB = float(data.energy_min_kwh[agent_idx])
            energy[agent_idx, step_idx].UB = float(data.energy_max_kwh[agent_idx])
            objective += float(data.dt_hours * data.export_subsidy_eur_per_kwh) * net_load[agent_idx, step_idx]
            objective += float(
                data.dt_hours
                * max(float(data.import_price_eur_per_kwh[step_idx] - data.export_subsidy_eur_per_kwh), 0.0)
            ) * n_pos[agent_idx, step_idx]
            objective += float(data.dt_hours * _THROUGHPUT_TIEBREAKER_EUR_PER_KWH) * (
                charge[agent_idx, step_idx] + discharge[agent_idx, step_idx]
            )
            objective += float(data.dt_hours * _NPOS_TIEBREAKER_EUR_PER_KWH) * n_pos[agent_idx, step_idx]
        energy[agent_idx, T].LB = float(data.energy_min_kwh[agent_idx])
        energy[agent_idx, T].UB = float(data.energy_max_kwh[agent_idx])

    if export_only:
        for step_idx in range(T):
            model.addConstr(
                float(offset_kw[step_idx])
                + gp.quicksum(float(alpha_window[agent_idx, step_idx]) * net_load[agent_idx, step_idx] for agent_idx in range(N))
                >= -float(surrogate.trafo_limit_kw),
                name=f"trafo_export_limit_{step_idx}",
            )

    model.setObjective(objective, grb.MINIMIZE)
    model.update()
    return gp, grb, model, charge, discharge, pv_curtail, net_load, n_pos, energy


def solve_direct_day_centralized(
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
    *,
    export_only: bool = True,
) -> DirectDaySolution:
    _validate_prices(data)
    gp, grb, model, charge, discharge, pv_curtail, net_load, n_pos, energy = _build_centralized_model(
        data,
        surrogate,
        export_only=bool(export_only),
    )
    del gp, net_load, n_pos
    try:
        model.optimize()
    except Exception as exc:
        raise _wrap_gurobi_error("centralized direct-day optimize failed", exc)

    status = _status_label(model, grb)
    if int(model.Status) not in {int(grb.OPTIMAL), int(grb.SUBOPTIMAL)}:
        raise RuntimeError(f"Centralized direct day model did not return a usable solution, got status={status}.")

    charge_kw = np.asarray([[charge[i, t].X for t in range(data.horizon)] for i in range(data.n_agents)], dtype=np.float32)
    discharge_kw = np.asarray([[discharge[i, t].X for t in range(data.horizon)] for i in range(data.n_agents)], dtype=np.float32)
    pv_curtail_kw = np.asarray([[pv_curtail[i, t].X for t in range(data.horizon)] for i in range(data.n_agents)], dtype=np.float32)
    energy_kwh = np.asarray([[energy[i, t].X for t in range(data.horizon + 1)] for i in range(data.n_agents)], dtype=np.float32)
    solution = _build_solution(
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        pv_curtail_kw=pv_curtail_kw,
        energy_kwh=energy_kwh,
        data=data,
        surrogate=surrogate,
        solver_status=status,
        solver_metadata={
            "num_vars": int(model.NumVars),
            "num_binary_vars": int(getattr(model, "NumBinVars", 0)),
            "num_integer_vars": int(getattr(model, "NumIntVars", 0)),
            "num_constraints": int(model.NumConstrs),
            "objective_value": float(model.ObjVal),
        },
    )
    dispose = getattr(model, "dispose", None)
    if callable(dispose):
        dispose()
    return solution


class _DirectDayLocalQPSolver:
    def __init__(self, data: DirectDayProblemData, surrogate: GridTrafoSurrogate, agent_idx: int) -> None:
        self.data = data
        self.surrogate = surrogate
        self.agent_idx = int(agent_idx)
        self.alpha = np.asarray(surrogate.alpha_netload_window_kw[self.agent_idx], dtype=np.float32)
        self.T = int(data.horizon)
        try:
            self.gp, self.grb = _load_gurobi()
            self.model = _create_model(self.gp, f"direct_day_local_agent_{agent_idx}")
        except Exception as exc:
            raise _wrap_gurobi_error(f"unable to create local ADMM model for agent {agent_idx}", exc)

        self.charge = self.model.addVars(self.T, lb=0.0, name="charge_kw")
        self.discharge = self.model.addVars(self.T, lb=0.0, name="discharge_kw")
        self.pv_curtail = self.model.addVars(self.T, lb=0.0, name="pv_curtail_kw")
        self.net_load = self.model.addVars(self.T, lb=-self.grb.INFINITY, name="net_load_kw")
        self.n_pos = self.model.addVars(self.T, lb=0.0, name="n_pos_kw")
        self.energy = self.model.addVars(self.T + 1, lb=0.0, name="energy_kwh")

        self.model.addConstr(self.energy[0] == float(data.energy_init_kwh[self.agent_idx]), name="energy_init")
        self.model.addConstr(self.energy[self.T] == float(data.energy_init_kwh[self.agent_idx]), name="energy_terminal")
        for step_idx in range(self.T):
            self.model.addConstr(self.charge[step_idx] <= float(data.p_max_kw[self.agent_idx]), name=f"charge_ub_{step_idx}")
            self.model.addConstr(self.discharge[step_idx] <= float(data.p_max_kw[self.agent_idx]), name=f"discharge_ub_{step_idx}")
            self.model.addConstr(self.pv_curtail[step_idx] <= float(data.pv_kw[self.agent_idx, step_idx]), name=f"curtail_ub_{step_idx}")
            self.model.addConstr(
                self.net_load[step_idx]
                == float(data.load_kw[self.agent_idx, step_idx])
                - float(data.pv_kw[self.agent_idx, step_idx])
                + self.pv_curtail[step_idx]
                + self.charge[step_idx]
                - self.discharge[step_idx],
                name=f"net_load_balance_{step_idx}",
            )
            self.model.addConstr(self.n_pos[step_idx] >= self.net_load[step_idx], name=f"n_pos_lb_net_{step_idx}")
            self.model.addConstr(
                self.energy[step_idx + 1]
                == self.energy[step_idx]
                + float(data.eff_charge[self.agent_idx]) * float(data.dt_hours) * self.charge[step_idx]
                - float(data.dt_hours / max(float(data.eff_discharge[self.agent_idx]), 1e-6)) * self.discharge[step_idx],
                name=f"energy_balance_{step_idx}",
            )
            self.energy[step_idx].LB = float(data.energy_min_kwh[self.agent_idx])
            self.energy[step_idx].UB = float(data.energy_max_kwh[self.agent_idx])
        self.energy[self.T].LB = float(data.energy_min_kwh[self.agent_idx])
        self.energy[self.T].UB = float(data.energy_max_kwh[self.agent_idx])
        self.model.update()

    def dispose(self) -> None:
        dispose = getattr(self.model, "dispose", None)
        if callable(dispose):
            dispose()

    def solve(self, *, z_kw: np.ndarray, u_kw: np.ndarray, rho: float) -> _LocalAgentSolveResult:
        z = np.asarray(z_kw, dtype=np.float32).reshape(-1)
        u = np.asarray(u_kw, dtype=np.float32).reshape(-1)
        if z.size != self.T or u.size != self.T:
            raise ValueError(f"z/u must have horizon {self.T}, got {z.size} and {u.size}.")

        objective = self.gp.QuadExpr()
        for step_idx in range(self.T):
            objective += float(self.data.dt_hours * self.data.export_subsidy_eur_per_kwh) * self.net_load[step_idx]
            objective += float(
                self.data.dt_hours
                * max(float(self.data.import_price_eur_per_kwh[step_idx] - self.data.export_subsidy_eur_per_kwh), 0.0)
            ) * self.n_pos[step_idx]
            objective += float(self.data.dt_hours * _THROUGHPUT_TIEBREAKER_EUR_PER_KWH) * (
                self.charge[step_idx] + self.discharge[step_idx]
            )
            objective += float(self.data.dt_hours * _NPOS_TIEBREAKER_EUR_PER_KWH) * self.n_pos[step_idx]
            if abs(float(self.alpha[step_idx])) > 1e-12 and float(rho) > 0.0:
                objective += float(0.5 * rho * self.alpha[step_idx] * self.alpha[step_idx]) * (
                    self.net_load[step_idx] * self.net_load[step_idx]
                )
                objective += float(rho * self.alpha[step_idx] * (u[step_idx] - z[step_idx])) * self.net_load[step_idx]
        self.model.setObjective(objective, self.grb.MINIMIZE)
        start = perf_counter()
        try:
            self.model.optimize()
        except Exception as exc:
            raise _wrap_gurobi_error(f"local ADMM optimize failed for agent {self.agent_idx}", exc)
        solve_time_sec = float(perf_counter() - start)
        status = _status_label(self.model, self.grb)
        if int(self.model.Status) not in {int(self.grb.OPTIMAL), int(self.grb.SUBOPTIMAL)}:
            raise RuntimeError(f"Local ADMM model for agent {self.agent_idx} failed with status={status}.")

        charge_kw = np.asarray([self.charge[t].X for t in range(self.T)], dtype=np.float32)
        discharge_kw = np.asarray([self.discharge[t].X for t in range(self.T)], dtype=np.float32)
        pv_curtail_kw = np.asarray([self.pv_curtail[t].X for t in range(self.T)], dtype=np.float32)
        net_load_kw = np.asarray([self.net_load[t].X for t in range(self.T)], dtype=np.float32)
        n_pos_kw = np.asarray([self.n_pos[t].X for t in range(self.T)], dtype=np.float32)
        energy_kwh = np.asarray([self.energy[t].X for t in range(self.T + 1)], dtype=np.float32)
        return _LocalAgentSolveResult(
            charge_kw=charge_kw,
            discharge_kw=discharge_kw,
            pv_curtail_kw=pv_curtail_kw,
            net_load_kw=net_load_kw,
            n_pos_kw=n_pos_kw,
            energy_kwh=energy_kwh,
            contribution_kw=(self.alpha * net_load_kw).astype(np.float32, copy=False),
            solve_time_sec=solve_time_sec,
            status=status,
        )


def _project_contribution_copies(v_kw: np.ndarray, lower_bound_kw: np.ndarray) -> np.ndarray:
    v = _coerce_2d(v_kw, name="v_kw")
    lower_bound = _coerce_1d(lower_bound_kw, name="lower_bound_kw")
    if v.shape[1] != lower_bound.size:
        raise ValueError(f"v_kw horizon {v.shape[1]} must match lower_bound horizon {lower_bound.size}.")
    projected = np.asarray(v, dtype=np.float32).copy()
    n_agents = int(v.shape[0])
    if n_agents <= 0:
        return projected
    for step_idx in range(v.shape[1]):
        total_value = float(np.sum(v[:, step_idx]))
        if total_value >= float(lower_bound[step_idx]):
            continue
        projected[:, step_idx] += np.float32((float(lower_bound[step_idx]) - total_value) / n_agents)
    return projected.astype(np.float32, copy=False)


def _compute_gap_metrics(admm_objective_eur: float, centralized_objective_eur: float | None) -> tuple[float, float]:
    if centralized_objective_eur is None or not np.isfinite(float(centralized_objective_eur)):
        return float("nan"), float("nan")
    gap_eur = float(admm_objective_eur - float(centralized_objective_eur))
    denom = max(abs(float(centralized_objective_eur)), 1e-3)
    return gap_eur, float(100.0 * gap_eur / denom)


def solve_direct_day_admm(
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
    *,
    rho_init: float,
    rho_adaptation: str | None,
    max_iters: int,
    primal_tol: float,
    dual_tol: float,
    export_only: bool = True,
    centralized_objective_eur: float | None = None,
) -> DirectDayAdmmResult:
    _validate_prices(data)
    if int(max_iters) <= 0:
        raise ValueError("max_iters must be positive.")
    rho = float(max(rho_init, 1e-9))
    mu = 10.0
    tau = 2.0
    alpha_window = np.asarray(surrogate.alpha_netload_window_kw, dtype=np.float32)
    offset_kw = _surrogate_root_offset_kw(data, surrogate)
    lower_bound_kw = (-float(surrogate.trafo_limit_kw) - offset_kw).astype(np.float32, copy=False)
    baseline_contrib_kw = (alpha_window * _baseline_net_load_kw(data)).astype(np.float32, copy=False)
    z_kw = (
        _project_contribution_copies(baseline_contrib_kw, lower_bound_kw)
        if export_only
        else baseline_contrib_kw.copy()
    )
    u_kw = np.zeros_like(z_kw, dtype=np.float32)
    solvers = [_DirectDayLocalQPSolver(data, surrogate, agent_idx) for agent_idx in range(data.n_agents)]

    history_rows: list[dict[str, float | int]] = []
    final_local_results: list[_LocalAgentSolveResult] = []
    converged = False
    final_primal = float("inf")
    final_dual = float("inf")

    for iteration in range(1, int(max_iters) + 1):
        iter_start = perf_counter()
        local_results = [
            solvers[agent_idx].solve(z_kw=z_kw[agent_idx], u_kw=u_kw[agent_idx], rho=rho)
            for agent_idx in range(data.n_agents)
        ]
        c_kw = np.asarray([result.contribution_kw for result in local_results], dtype=np.float32)
        v_kw = (c_kw + u_kw).astype(np.float32, copy=False)
        z_next_kw = _project_contribution_copies(v_kw, lower_bound_kw) if export_only else v_kw.copy()
        u_next_kw = (u_kw + c_kw - z_next_kw).astype(np.float32, copy=False)
        primal_residual = float(np.linalg.norm((c_kw - z_next_kw).reshape(-1), ord=2))
        dual_residual = float(rho * np.linalg.norm((z_next_kw - z_kw).reshape(-1), ord=2))

        charge_kw = np.asarray([result.charge_kw for result in local_results], dtype=np.float32)
        discharge_kw = np.asarray([result.discharge_kw for result in local_results], dtype=np.float32)
        pv_curtail_kw = np.asarray([result.pv_curtail_kw for result in local_results], dtype=np.float32)
        energy_kwh = np.asarray([result.energy_kwh for result in local_results], dtype=np.float32)
        candidate_solution = _build_solution(
            charge_kw=charge_kw,
            discharge_kw=discharge_kw,
            pv_curtail_kw=pv_curtail_kw,
            energy_kwh=energy_kwh,
            data=data,
            surrogate=surrogate,
            solver_status="admm_iterate",
            solver_metadata={"rho": rho},
        )
        iter_runtime_sec = float(perf_counter() - iter_start)
        history_rows.append(
            {
                "iteration": int(iteration),
                "objective_estimate": float(candidate_solution.objective_eur),
                "primal_residual": primal_residual,
                "dual_residual": dual_residual,
                "export_violation_kw_max": float(candidate_solution.max_export_violation_kw),
                "iter_runtime_sec": iter_runtime_sec,
                "rho_value": float(rho),
            }
        )

        z_kw = z_next_kw
        u_kw = u_next_kw
        final_local_results = local_results
        final_primal = primal_residual
        final_dual = dual_residual
        if primal_residual <= float(primal_tol) and dual_residual <= float(dual_tol):
            converged = True
            break

        if str(rho_adaptation or "").strip().lower() == "residual_balancing":
            if primal_residual > mu * dual_residual and rho > 0.0:
                rho *= tau
                u_kw /= np.float32(tau)
            elif dual_residual > mu * primal_residual and rho > 0.0:
                rho /= tau
                u_kw *= np.float32(tau)

    if not final_local_results:
        raise RuntimeError("ADMM did not produce any iterate.")

    final_solution = _build_solution(
        charge_kw=np.asarray([result.charge_kw for result in final_local_results], dtype=np.float32),
        discharge_kw=np.asarray([result.discharge_kw for result in final_local_results], dtype=np.float32),
        pv_curtail_kw=np.asarray([result.pv_curtail_kw for result in final_local_results], dtype=np.float32),
        energy_kwh=np.asarray([result.energy_kwh for result in final_local_results], dtype=np.float32),
        data=data,
        surrogate=surrogate,
        solver_status="admm_converged" if converged else "admm_not_converged",
        solver_metadata={"rho_final": float(rho), "rho_adaptation": str(rho_adaptation or "none")},
    )
    for solver in solvers:
        solver.dispose()
    history_df = pd.DataFrame(history_rows)
    gap_eur, gap_pct = _compute_gap_metrics(final_solution.objective_eur, centralized_objective_eur)
    return DirectDayAdmmResult(
        solution=final_solution,
        history_df=history_df,
        converged=bool(converged),
        final_primal_residual=float(final_primal),
        final_dual_residual=float(final_dual),
        iterations=int(len(history_rows)),
        objective_gap_vs_centralized_eur=float(gap_eur),
        objective_gap_vs_centralized_pct=float(gap_pct),
    )


def summarize_direct_day_solution(
    solution: DirectDaySolution,
    surrogate: GridTrafoSurrogate,
) -> pd.Series:
    root_p_kw = (
        np.asarray(solution.surrogate_root_p_kw, dtype=np.float32)
        if solution.surrogate_root_p_kw is not None
        else np.zeros((0,), dtype=np.float32)
    )
    dt_hours = float(solution.solver_metadata.get("dt_hours", 1.0))
    return pd.Series(
        {
            "objective_eur": float(solution.objective_eur),
            "solver_status": str(solution.solver_status),
            "max_export_violation_kw": float(solution.max_export_violation_kw),
            "max_root_export_kw": float(np.max(np.maximum(-root_p_kw, 0.0))) if root_p_kw.size else 0.0,
            "trafo_limit_kw": float(surrogate.trafo_limit_kw),
            "total_charge_kwh": float(np.sum(solution.charge_kw) * dt_hours),
            "total_discharge_kwh": float(np.sum(solution.discharge_kw) * dt_hours),
            "total_curtail_kwh": float(np.sum(solution.pv_curtail_kw) * dt_hours),
            "reference_only": bool(solution.reference_only),
            "note": str(solution.note),
        }
    )


def plot_direct_day_convergence(admm_result: DirectDayAdmmResult, *, figsize: tuple[float, float] = (14.0, 8.0)):
    history_df = admm_result.history_df.copy()
    if history_df.empty:
        raise ValueError("ADMM history is empty.")
    figure, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes[0].plot(history_df["iteration"], history_df["primal_residual"], label="Primal residual", color="#1d4ed8")
    axes[0].plot(history_df["iteration"], history_df["dual_residual"], label="Dual residual", color="#dc2626")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[0].set_title("ADMM residual convergence")

    axes[1].plot(history_df["iteration"], history_df["objective_estimate"], color="#111827")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_ylabel("EUR")
    axes[1].set_title("Objective estimate")

    axes[2].plot(history_df["iteration"], history_df["export_violation_kw_max"], color="#059669", label="Export violation")
    axes[2].plot(history_df["iteration"], history_df["rho_value"], color="#7c3aed", label="rho")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("Constraint violation and rho")
    figure.tight_layout()
    return figure, axes


def plot_direct_day_root_p_comparison(
    solution: DirectDaySolution,
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
    *,
    baseline_solution: DirectDaySolution | None = None,
    env=None,
    label: str = "Solution",
    figsize: tuple[float, float] = (16.0, 5.0),
):
    timestamps = pd.to_datetime(list(data.timestamps))
    surrogate_root = (
        np.asarray(solution.surrogate_root_p_kw, dtype=np.float32)
        if solution.surrogate_root_p_kw is not None
        else compute_surrogate_root_p_kw(solution.net_load_kw, data, surrogate)
    )
    baseline_root = None
    if baseline_solution is not None:
        baseline_root = (
            np.asarray(baseline_solution.surrogate_root_p_kw, dtype=np.float32)
            if baseline_solution.surrogate_root_p_kw is not None
            else compute_surrogate_root_p_kw(baseline_solution.net_load_kw, data, surrogate)
        )
    pp_root = solution.pp_root_p_kw if solution.pp_root_p_kw is not None else (
        compute_direct_day_pp_root_p_kw(env, data, solution) if env is not None else None
    )
    figure, axis = plt.subplots(1, 1, figsize=figsize)
    if baseline_root is not None:
        axis.plot(timestamps, baseline_root, label="Baseline root P", color="#94a3b8", linewidth=1.6)
    axis.plot(timestamps, surrogate_root, label=f"{label} surrogate root P", color="#2563eb", linewidth=1.8)
    if pp_root is not None:
        axis.plot(timestamps, pp_root, label=f"{label} pandapower root P", color="#dc2626", linewidth=1.4, linestyle="--")
    axis.axhline(-float(surrogate.trafo_limit_kw), color="#16a34a", linestyle=":", linewidth=1.2, label="Export limit")
    axis.axhline(float(surrogate.trafo_limit_kw), color="#16a34a", linestyle=":", linewidth=1.2)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")
    axis.set_title(f"Transformer root P comparison - {label}")
    axis.set_ylabel("kW")
    figure.tight_layout()
    return figure, axis


def plot_direct_day_power_energy_balance(
    solution: DirectDaySolution,
    data: DirectDayProblemData,
    *,
    baseline_solution: DirectDaySolution | None = None,
    label: str = "ADMM",
    figsize: tuple[float, float] = (18.0, 8.5),
):
    del baseline_solution
    step_df = _build_power_balance_step_df(solution, data)
    timestamps = step_df["timestamp"].to_numpy()
    width = 0.008
    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    positive_specs = [
        ("load_total_kw", "Load", "#111827"),
        ("battery_charge_total_kw", "Battery charge", "#dc2626"),
        ("grid_export_total_kw", "Grid export", "#f59e0b"),
        ("pv_curtail_total_kw", "PV curtailment", "#fca5a5"),
    ]
    negative_specs = [
        ("pv_raw_total_kw", "PV raw", "#16a34a"),
        ("grid_import_total_kw", "Grid import", "#2563eb"),
        ("battery_discharge_total_kw", "Battery discharge", "#7c3aed"),
    ]

    power_axis = axes[0]
    positive_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, legend_label, color in positive_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        extra_kwargs = {"hatch": "//", "edgecolor": "#dc2626", "linewidth": 1.0} if column == "pv_curtail_total_kw" else {}
        power_axis.bar(
            timestamps,
            values,
            width=width,
            bottom=positive_bottom,
            color=color,
            alpha=0.78,
            label=legend_label,
            **extra_kwargs,
        )
        positive_bottom = positive_bottom + values

    negative_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, legend_label, color in negative_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        power_axis.bar(
            timestamps,
            -values,
            width=width,
            bottom=negative_bottom,
            color=color,
            alpha=0.78,
            label=legend_label,
        )
        negative_bottom = negative_bottom - values

    max_residual_kw = float(np.max(np.abs(step_df["power_balance_residual_kw"].to_numpy(dtype=np.float32))))
    power_axis.axhline(0.0, color="#111827", linewidth=1.0)
    power_axis.set_title(f"{label} Power / Energy Balance (aggregate residual <= {max_residual_kw:.3e} kW)")
    power_axis.set_ylabel("kW")
    power_axis.grid(True, axis="y", alpha=0.25)
    power_axis.legend(loc="upper right", ncol=4)

    energy_axis = axes[1]
    timestamps_dt = pd.to_datetime(list(data.timestamps))
    energy_timestamps = pd.Index(
        list(timestamps_dt) + [timestamps_dt[-1] + pd.to_timedelta(float(data.dt_hours), unit="h")]
    )
    energy_axis.plot(
        energy_timestamps,
        np.sum(np.asarray(solution.energy_kwh, dtype=np.float32), axis=0),
        color="#059669",
        linewidth=1.6,
        label="Aggregate energy",
    )
    energy_axis.set_ylabel("kWh")
    energy_axis.set_xlabel("Timestamp")
    energy_axis.grid(True, alpha=0.25)
    energy_axis.legend(loc="upper right")
    figure.tight_layout()
    return figure, axes


def plot_direct_day_voltage_profile(
    solution: DirectDaySolution,
    data: DirectDayProblemData,
    env,
    cfg,
    *,
    label: str = "ADMM",
    grid_df: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (18.0, 5.4),
):
    if grid_df is None:
        _, grid_df = compute_direct_day_grid_profile(env, data, solution)
    if grid_df.empty:
        raise ValueError("Grid profile is empty; cannot plot direct-day voltages.")

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    muted_color = "#cbd5e1"
    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]

    for bus_id, frame in grid_df.loc[~grid_df["is_agent_bus"]].groupby("bus_id", sort=True):
        axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=muted_color,
            linewidth=0.9,
            alpha=0.35,
            zorder=1,
        )

    agent_bus_ids = [int(bus_id) for bus_id in getattr(getattr(env, "_grid_core", None), "agent_bus_ids", [])]
    for color_idx, bus_id in enumerate(agent_bus_ids):
        frame = grid_df.loc[grid_df["bus_id"] == int(bus_id)]
        if frame.empty:
            continue
        axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=highlight_palette[color_idx % len(highlight_palette)],
            linewidth=2.0,
            alpha=0.95,
            label=f"Agent bus {bus_id}",
            zorder=3,
        )

    axis.axhline(float(cfg.grid.v_min_pu), color="#dc2626", linestyle="--", linewidth=1.1, label="V min")
    axis.axhline(float(cfg.grid.v_max_pu), color="#ea580c", linestyle="--", linewidth=1.1, label="V max")
    axis.set_title(f"Voltage Profile - {label}")
    axis.set_ylabel("Voltage [p.u.]")
    axis.set_xlabel("Timestamp")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", ncol=2)
    figure.tight_layout()
    return figure, axis


def plot_direct_day_net_load(
    solution: DirectDaySolution,
    data: DirectDayProblemData,
    surrogate: GridTrafoSurrogate,
    *,
    baseline_solution: DirectDaySolution | None = None,
    centralized_solution: DirectDaySolution | None = None,
    label: str = "ADMM",
    figsize: tuple[float, float] = (16.0, 5.0),
):
    del surrogate
    timestamps = pd.to_datetime(list(data.timestamps))
    figure, axis = plt.subplots(1, 1, figsize=figsize)
    baseline_raw_total_kw = (
        np.sum(np.asarray(baseline_solution.net_load_kw, dtype=np.float32), axis=0)
        if baseline_solution is not None
        else np.sum(_baseline_net_load_kw(data), axis=0)
    )
    axis.plot(
        timestamps,
        baseline_raw_total_kw,
        label="Baseline raw net load",
        color="#94a3b8",
        linewidth=1.5,
    )
    axis.plot(
        timestamps,
        np.sum(np.asarray(data.load_kw, dtype=np.float32) - np.asarray(solution.pv_effective_kw, dtype=np.float32), axis=0),
        label=f"{label} post-curtail net load",
        color="#16a34a",
        linewidth=1.5,
        linestyle="-.",
    )
    axis.plot(
        timestamps,
        np.sum(np.asarray(solution.net_load_kw, dtype=np.float32), axis=0),
        label=f"{label} post-action net load",
        color="#2563eb",
        linewidth=1.8,
    )
    if centralized_solution is not None:
        axis.plot(
            timestamps,
            np.sum(np.asarray(centralized_solution.net_load_kw, dtype=np.float32), axis=0),
            label="Centralized post-action net load",
            color="#7c3aed",
            linewidth=1.6,
            linestyle="--",
        )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")
    axis.set_title(f"Net Load Comparison - {label}")
    axis.set_ylabel("kW")
    axis.set_xlabel("Timestamp")
    axis.text(
        0.01,
        0.02,
        "Transformer capacity is enforced on root P, not directly on aggregate net load; see root-P diagnostics.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#475569",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.0},
    )
    figure.tight_layout()
    return figure, axis


def plot_direct_day_soc_and_battery(
    solution: DirectDaySolution,
    data: DirectDayProblemData,
    *,
    figsize: tuple[float, float] = (16.0, 8.0),
):
    timestamps = pd.to_datetime(list(data.timestamps))
    energy_timestamps = pd.Index(list(timestamps) + [timestamps[-1] + pd.to_timedelta(float(data.dt_hours), unit="h")])
    figure, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
    axes[0].plot(timestamps, np.sum(solution.charge_kw, axis=0), label="Aggregate charge", color="#2563eb")
    axes[0].plot(timestamps, np.sum(solution.discharge_kw, axis=0), label="Aggregate discharge", color="#dc2626")
    axes[0].plot(timestamps, np.sum(solution.pv_curtail_kw, axis=0), label="Aggregate curtail", color="#f59e0b")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("kW")

    axes[1].plot(energy_timestamps, np.sum(solution.energy_kwh, axis=0), label="Aggregate energy", color="#059669")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("kWh")
    axes[1].set_title("Battery power and aggregate energy")
    figure.tight_layout()
    return figure, axes


def collect_admm_direct_rollout(
    cfg,
    *,
    prediction_mode: str = "perfect",
    rho_init: float | None = None,
    rho_adaptation: str | None = "residual_balancing",
    max_iters: int = 300,
    primal_tol: float = 1e-3,
    dual_tol: float = 1e-3,
) -> grid_nb.RolloutResult:
    resolved_mode = grid_nb.normalize_prediction_mode(prediction_mode)
    if resolved_mode != grid_nb.PERFECT_PREDICTION_MODE:
        raise ValueError("ADMM compare v1 only supports perfect forecasts.")

    comparison_cfg = grid_nb.build_comparison_cfg(cfg, prediction_mode=resolved_mode)
    # The direct-day solver is defined on daily episodes even when compare uses a
    # longer episode_limit elsewhere (for example 192-step rollout episodes).
    comparison_cfg.env.episode_limit = _day_episode_length(float(comparison_cfg.env.dt))
    test_start_date = getattr(getattr(comparison_cfg, "data", None), "test_start_date", None)
    test_end_date = getattr(getattr(comparison_cfg, "data", None), "test_end_date", None)
    if not test_start_date or not test_end_date:
        raise ValueError("ADMM direct compare rollout requires cfg.data.test_start_date and cfg.data.test_end_date.")

    data = build_direct_day_problem_data(
        comparison_cfg,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
    )
    env = build_env(comparison_cfg, mode="test")
    try:
        surrogate = build_direct_day_trafo_surrogate(comparison_cfg, env, data)
        if rho_init is None:
            baseline_net_load = _baseline_net_load_kw(data)
            alpha_window = np.asarray(surrogate.alpha_netload_window_kw, dtype=np.float32)
            rho_value = float(
                np.mean(np.asarray(data.import_price_eur_per_kwh, dtype=np.float32)) * float(data.dt_hours)
                / max(1.0, float(np.mean(np.abs(alpha_window * baseline_net_load))))
            )
        else:
            rho_value = float(rho_init)
        admm_result = solve_direct_day_admm(
            data,
            surrogate,
            rho_init=rho_value,
            rho_adaptation=rho_adaptation,
            max_iters=max_iters,
            primal_tol=primal_tol,
            dual_tol=dual_tol,
            export_only=True,
        )
        rollout = _build_direct_day_replay_rollout(
            comparison_cfg,
            env,
            data,
            admm_result.solution,
            label=ADMM_DIRECT_PERFECT_LABEL,
        )
        rollout.meta.update(
            {
                "admm_converged": bool(admm_result.converged),
                "admm_iterations": int(admm_result.iterations),
                "admm_final_primal_residual": float(admm_result.final_primal_residual),
                "admm_final_dual_residual": float(admm_result.final_dual_residual),
                "admm_objective_gap_vs_centralized_eur": float(admm_result.objective_gap_vs_centralized_eur),
                "admm_objective_gap_vs_centralized_pct": float(admm_result.objective_gap_vs_centralized_pct),
                "solver_status": str(admm_result.solution.solver_status),
                "prediction_mode": grid_nb.PERFECT_PREDICTION_MODE,
            }
        )
        return rollout
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


__all__ = [
    "ADMM_DIRECT_PERFECT_LABEL",
    "DirectDayAdmmResult",
    "DirectDayProblemData",
    "DirectDaySolution",
    "GridTrafoSurrogate",
    "build_direct_day_problem_data",
    "build_direct_day_trafo_surrogate",
    "collect_admm_direct_rollout",
    "compute_direct_day_baseline",
    "compute_direct_day_grid_profile",
    "compute_direct_day_pp_root_p_kw",
    "compute_surrogate_root_p_kw",
    "plot_direct_day_convergence",
    "plot_direct_day_net_load",
    "plot_direct_day_power_energy_balance",
    "plot_direct_day_root_p_comparison",
    "plot_direct_day_soc_and_battery",
    "plot_direct_day_voltage_profile",
    "solve_direct_day_admm",
    "solve_direct_day_centralized",
    "summarize_direct_day_solution",
]
