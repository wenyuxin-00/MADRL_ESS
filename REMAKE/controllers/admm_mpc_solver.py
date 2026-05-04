from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

from REMAKE.envs.grid_env import project_action_to_soc

_THROUGHPUT_TIEBREAKER_EUR_PER_KWH = 1e-8
_RHO_BALANCE_MU = 1e1
_RHO_BALANCE_TAU = 2.0
_TRAFO_ACTIVE_GUARD = 0.97
_ADMM_RESERVE_POWER_FRACTION = 0.30


@dataclass(frozen=True)
class AdmmMpcWindowData:
    import_price_eur_per_kwh: np.ndarray
    load_seq: np.ndarray
    pv_seq: np.ndarray
    battery_capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    efficiency: float
    energy_init_kwh: np.ndarray
    energy_min_kwh: np.ndarray
    energy_max_kwh: np.ndarray
    dt_hours: float


@dataclass(frozen=True)
class AdmmMpcCoordinationCache:
    trafo_limit_kw: float
    trafo_base_kw: float
    alpha_netload_window_kw: np.ndarray


@dataclass(frozen=True)
class AdmmMpcStepResult:
    executed_action_array: np.ndarray
    converged: bool
    iterations: int
    final_primal_residual: float
    final_dual_residual: float
    solve_time_sec: float
    rho_final: float


def build_admm_mpc_coordination_cache(cfg, env, *, horizon_steps: int) -> AdmmMpcCoordinationCache:
    alpha_window = np.ones((int(cfg.env.num_agents), int(horizon_steps)), dtype=np.float32)
    return AdmmMpcCoordinationCache(float(env.grid_core.trafo_limit_kw) * _TRAFO_ACTIVE_GUARD, 0.0, alpha_window)


def compute_default_rho(window_data: AdmmMpcWindowData, coordination_cache: AdmmMpcCoordinationCache) -> float:
    alpha_window = np.asarray(coordination_cache.alpha_netload_window_kw, dtype=np.float32)
    baseline_net_load = (np.asarray(window_data.load_seq, dtype=np.float32) - np.asarray(window_data.pv_seq, dtype=np.float32)).astype(np.float32)
    denominator = max(1.0, float(np.mean(np.abs(alpha_window * baseline_net_load))))
    return float(np.mean(np.asarray(window_data.import_price_eur_per_kwh, dtype=np.float32)) * float(window_data.dt_hours) / denominator)


def _project_contribution_copies(values: np.ndarray, lower_bound_kw: np.ndarray, upper_bound_kw: np.ndarray) -> np.ndarray:
    projected = np.asarray(values, dtype=np.float32).copy()
    projected_sum = projected.sum(axis=0)
    clipped_sum = np.clip(projected_sum, lower_bound_kw, upper_bound_kw).astype(np.float32)
    projected += ((clipped_sum - projected_sum) / np.float32(projected.shape[0])).astype(np.float32)
    return projected.astype(np.float32)


class ReusableAdmmLocalSolver:
    def __init__(self, *, agent_idx: int, horizon: int, p_max_kw: float, dt_hours: float, efficiency: float, energy_min_kwh: float, energy_max_kwh: float) -> None:
        import gurobipy as gp

        self.agent_idx = int(agent_idx)
        self.horizon = int(horizon)
        self.gp = gp
        self.grb = gp.GRB
        self.model = gp.Model(f"remake_admm_mpc_agent_{self.agent_idx}")
        self.model.Params.OutputFlag = 0
        self.charge = self.model.addVars(self.horizon, lb=0.0, ub=float(p_max_kw), name="charge_kw")
        self.discharge = self.model.addVars(self.horizon, lb=0.0, ub=float(p_max_kw), name="discharge_kw")
        self.pv_curtail = self.model.addVars(self.horizon, lb=0.0, name="pv_curtail_kw")
        self.net_load = self.model.addVars(self.horizon, lb=-self.grb.INFINITY, name="net_load_kw")
        self.energy = self.model.addVars(self.horizon + 1, lb=float(energy_min_kwh), ub=float(energy_max_kwh), name="energy_kwh")
        self.energy_init = self.model.addConstr(self.energy[0] == 0.0, name="energy_init")
        self.pv_curtail_upper = []
        self.net_load_balance = []
        self.energy_reserve_floor = []
        for step_idx in range(self.horizon):
            self.pv_curtail_upper.append(self.model.addConstr(self.pv_curtail[step_idx] <= 0.0, name=f"curtail_ub_{step_idx}"))
            self.net_load_balance.append(self.model.addConstr(self.net_load[step_idx] - self.pv_curtail[step_idx] - self.charge[step_idx] + self.discharge[step_idx] == 0.0, name=f"net_load_balance_{step_idx}"))
            self.model.addConstr(self.energy[step_idx + 1] == self.energy[step_idx] + float(efficiency) * float(dt_hours) * self.charge[step_idx] - float(dt_hours) / float(efficiency) * self.discharge[step_idx], name=f"energy_balance_{step_idx}")
            self.energy_reserve_floor.append(self.model.addConstr(self.energy[step_idx + 1] >= float(energy_min_kwh), name=f"energy_reserve_floor_{step_idx}"))
        self.model.ModelSense = self.grb.MINIMIZE
        self.model.update()

    def dispose(self) -> None:
        self.model.dispose()

    def solve(self, *, window_data: AdmmMpcWindowData, alpha_kw: np.ndarray, z_kw: np.ndarray, u_kw: np.ndarray, rho: float) -> tuple[np.ndarray, np.float32, np.float32]:
        alpha = np.asarray(alpha_kw, dtype=np.float32).reshape(-1)
        z = np.asarray(z_kw, dtype=np.float32).reshape(-1)
        u = np.asarray(u_kw, dtype=np.float32).reshape(-1)
        self.energy_init.RHS = float(window_data.energy_init_kwh[self.agent_idx])
        reserve_floor = float(_planning_energy_min_kwh(window_data)[self.agent_idx])
        objective = self.gp.QuadExpr()
        for step_idx in range(self.horizon):
            price_t = float(window_data.import_price_eur_per_kwh[step_idx])
            self.pv_curtail_upper[step_idx].RHS = float(window_data.pv_seq[self.agent_idx, step_idx])
            self.net_load_balance[step_idx].RHS = float(window_data.load_seq[self.agent_idx, step_idx] - window_data.pv_seq[self.agent_idx, step_idx])
            self.energy_reserve_floor[step_idx].RHS = reserve_floor
            objective += float(window_data.dt_hours * price_t) * self.charge[step_idx]
            objective += float(window_data.dt_hours * -price_t) * self.discharge[step_idx]
            objective += float(window_data.dt_hours * _THROUGHPUT_TIEBREAKER_EUR_PER_KWH) * (self.charge[step_idx] + self.discharge[step_idx])
            objective += float(0.5 * rho * alpha[step_idx] * alpha[step_idx]) * (self.net_load[step_idx] * self.net_load[step_idx])
            objective += float(rho * alpha[step_idx] * (u[step_idx] - z[step_idx])) * self.net_load[step_idx]
        self.model.setObjective(objective, self.grb.MINIMIZE)
        self.model.optimize()
        status = int(self.model.Status)
        if status not in {int(self.grb.OPTIMAL), int(self.grb.SUBOPTIMAL)}:
            raise RuntimeError(f"ADMM MPC local model for agent {self.agent_idx} ended with status={status}.")
        net_load_kw = np.asarray([self.net_load[t].X for t in range(self.horizon)], dtype=np.float32)
        return ((alpha * net_load_kw).astype(np.float32), np.float32(float(self.charge[0].X) - float(self.discharge[0].X)), np.float32(float(self.pv_curtail[0].X)))


def _first_step_action(env, *, battery_power_kw: np.ndarray, pv_curtail_kw: np.ndarray, pv_step_kw: np.ndarray) -> np.ndarray:
    requested_battery_power_kw = np.asarray(battery_power_kw, dtype=np.float32).reshape(-1)
    battery_action = project_action_to_soc(env.cfg, env.soc, requested_battery_power_kw / np.maximum(np.asarray(env.pmax, dtype=np.float32), 1e-6))
    pv_raw_kw = np.maximum(np.asarray(pv_step_kw, dtype=np.float32).reshape(-1), 0.0)
    pv_effective_kw = np.maximum(pv_raw_kw - np.asarray(pv_curtail_kw, dtype=np.float32).reshape(-1), 0.0).astype(np.float32)
    pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
    valid_mask = pv_raw_kw > 1e-6
    pv_utilization[valid_mask] = pv_effective_kw[valid_mask] / pv_raw_kw[valid_mask]
    return np.stack([battery_action, np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0).astype(np.float32)], axis=-1).astype(np.float32)


def _planning_energy_min_kwh(window_data: AdmmMpcWindowData) -> np.ndarray:
    one_step_reserve = np.asarray(window_data.p_max_kw, dtype=np.float32) * np.float32(_ADMM_RESERVE_POWER_FRACTION * window_data.dt_hours / window_data.efficiency)
    return np.minimum(np.asarray(window_data.energy_min_kwh, dtype=np.float32) + one_step_reserve, np.asarray(window_data.energy_max_kwh, dtype=np.float32)).astype(np.float32)


def _project_first_step_action(env, window_data: AdmmMpcWindowData, coordination_cache: AdmmMpcCoordinationCache, battery_power_kw: np.ndarray, pv_curtail_kw: np.ndarray, target_contribution_kw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import gurobipy as gp

    n = int(window_data.load_seq.shape[0]); grb = gp.GRB
    model = gp.Model("remake_admm_first_step")
    model.Params.OutputFlag = 0
    charge = model.addVars(n, lb=0.0, name="charge_kw")
    discharge = model.addVars(n, lb=0.0, name="discharge_kw")
    curtail = model.addVars(n, lb=0.0, name="pv_curtail_kw")
    net = model.addVars(n, lb=-grb.INFINITY, name="net_load_kw")
    objective = gp.QuadExpr()
    baseline = np.asarray(window_data.load_seq[:, 0] - window_data.pv_seq[:, 0], dtype=np.float32)
    desired_power = np.asarray(battery_power_kw, dtype=np.float32).reshape(n)
    desired_curtail = np.asarray(pv_curtail_kw, dtype=np.float32).reshape(n)
    target = np.asarray(target_contribution_kw, dtype=np.float32).reshape(n)
    lower = -float(coordination_cache.trafo_limit_kw) - float(coordination_cache.trafo_base_kw)
    upper = float(coordination_cache.trafo_limit_kw) - float(coordination_cache.trafo_base_kw)
    for idx in range(n):
        model.addConstr(charge[idx] <= float(window_data.p_max_kw[idx]), name=f"charge_ub_{idx}")
        model.addConstr(discharge[idx] <= float(window_data.p_max_kw[idx]), name=f"discharge_ub_{idx}")
        model.addConstr(curtail[idx] <= float(max(window_data.pv_seq[idx, 0], 0.0)), name=f"curtail_ub_{idx}")
        next_energy = float(window_data.energy_init_kwh[idx]) + float(window_data.efficiency * window_data.dt_hours) * charge[idx] - float(window_data.dt_hours / window_data.efficiency) * discharge[idx]
        model.addConstr(next_energy >= float(window_data.energy_min_kwh[idx]), name=f"energy_lb_{idx}")
        model.addConstr(next_energy <= float(window_data.energy_max_kwh[idx]), name=f"energy_ub_{idx}")
        model.addConstr(net[idx] == float(baseline[idx]) + curtail[idx] + charge[idx] - discharge[idx], name=f"net_balance_{idx}")
        objective += (charge[idx] - discharge[idx] - float(desired_power[idx])) * (charge[idx] - discharge[idx] - float(desired_power[idx]))
        objective += (curtail[idx] - float(desired_curtail[idx])) * (curtail[idx] - float(desired_curtail[idx]))
        objective += np.float32(1e-6) * (net[idx] - float(target[idx])) * (net[idx] - float(target[idx]))
    model.addConstr(gp.quicksum(net[idx] for idx in range(n)) >= lower, name="trafo_lower")
    model.addConstr(gp.quicksum(net[idx] for idx in range(n)) <= upper, name="trafo_upper")
    model.setObjective(objective, grb.MINIMIZE); model.optimize()
    status = int(model.Status)
    if status != int(grb.OPTIMAL):
        raise RuntimeError(f"ADMM MPC first-step feasibility projection ended with status={status}.")
    signed = np.asarray([charge[idx].X - discharge[idx].X for idx in range(n)], dtype=np.float32)
    curtailed = np.asarray([curtail[idx].X for idx in range(n)], dtype=np.float32)
    model.dispose()
    return signed, curtailed


def run_admm_mpc_step(
    env,
    window_data: AdmmMpcWindowData,
    *,
    coordination_cache: AdmmMpcCoordinationCache,
    rho_init: float,
    rho_min: float,
    rho_max: float,
    rho_adaptation: str | None,
    max_iters: int,
    max_iters_first_step: int,
    primal_tol: float,
    dual_tol: float,
    progress_desc: str | None = None,
) -> AdmmMpcStepResult:
    n_agents, horizon = window_data.load_seq.shape
    alpha_window = np.asarray(coordination_cache.alpha_netload_window_kw, dtype=np.float32)
    contribution_lower_bound_kw = np.full((horizon,), -float(coordination_cache.trafo_limit_kw) - float(coordination_cache.trafo_base_kw), dtype=np.float32)
    contribution_upper_bound_kw = np.full((horizon,), float(coordination_cache.trafo_limit_kw) - float(coordination_cache.trafo_base_kw), dtype=np.float32)
    baseline_contrib = (alpha_window * (np.asarray(window_data.load_seq, dtype=np.float32) - np.asarray(window_data.pv_seq, dtype=np.float32))).astype(np.float32)
    z_kw = _project_contribution_copies(baseline_contrib, contribution_lower_bound_kw, contribution_upper_bound_kw)
    u_kw = np.zeros_like(z_kw, dtype=np.float32)
    rho = float(np.clip(float(rho_init), float(rho_min), float(rho_max)))
    iteration_budget = int(max_iters_first_step) if int(getattr(env, "cur_step", 0)) == 0 else int(max_iters)
    use_residual_balancing = str(rho_adaptation or "").strip().lower() == "residual_balancing"
    local_solvers = tuple(
        ReusableAdmmLocalSolver(
            agent_idx=agent_idx,
            horizon=horizon,
            p_max_kw=float(window_data.p_max_kw[agent_idx]),
            dt_hours=float(window_data.dt_hours),
            efficiency=float(window_data.efficiency),
            energy_min_kwh=float(window_data.energy_min_kwh[agent_idx]),
            energy_max_kwh=float(window_data.energy_max_kwh[agent_idx]),
        )
        for agent_idx in range(n_agents)
    )
    final_battery_power = np.zeros((n_agents,), dtype=np.float32)
    final_pv_curtail = np.zeros((n_agents,), dtype=np.float32)
    final_primal = final_dual = float("inf")
    converged = False
    started_at = perf_counter()
    iterations = tqdm(range(1, iteration_budget + 1), desc=progress_desc, unit="iter", leave=False, ascii=True, disable=progress_desc is None)
    for iteration in iterations:
        local_results = [solver.solve(window_data=window_data, alpha_kw=alpha_window[agent_idx], z_kw=z_kw[agent_idx], u_kw=u_kw[agent_idx], rho=float(rho)) for agent_idx, solver in enumerate(local_solvers)]
        contribution_kw = np.asarray([result[0] for result in local_results], dtype=np.float32)
        final_battery_power = np.asarray([result[1] for result in local_results], dtype=np.float32)
        final_pv_curtail = np.asarray([result[2] for result in local_results], dtype=np.float32)
        z_next = _project_contribution_copies(contribution_kw + u_kw, contribution_lower_bound_kw, contribution_upper_bound_kw)
        u_kw = (u_kw + contribution_kw - z_next).astype(np.float32)
        final_primal = float(np.linalg.norm((contribution_kw - z_next).reshape(-1), ord=2))
        final_dual = float(rho * np.linalg.norm((z_next - z_kw).reshape(-1), ord=2))
        iterations.set_postfix(primal=f"{final_primal:.2e}", dual=f"{final_dual:.2e}", rho=f"{rho:.2e}", refresh=False)
        if final_primal <= float(primal_tol) and final_dual <= float(dual_tol):
            converged = True
            break
        if use_residual_balancing and rho > 0.0:
            rho_before = float(rho)
            if final_primal > _RHO_BALANCE_MU * final_dual:
                rho = float(np.clip(rho_before * _RHO_BALANCE_TAU, float(rho_min), float(rho_max)))
            elif final_dual > _RHO_BALANCE_MU * final_primal:
                rho = float(np.clip(rho_before / _RHO_BALANCE_TAU, float(rho_min), float(rho_max)))
            if not np.isclose(rho, rho_before):
                u_kw = (u_kw * np.float32(rho_before / max(rho, 1e-9))).astype(np.float32)
        z_kw = z_next
    for solver in local_solvers:
        solver.dispose()
    first_target = z_kw[:, 0].astype(np.float32)
    final_battery_power, final_pv_curtail = _project_first_step_action(env, window_data, coordination_cache, final_battery_power, final_pv_curtail, first_target)
    return AdmmMpcStepResult(
        executed_action_array=_first_step_action(env, battery_power_kw=final_battery_power, pv_curtail_kw=final_pv_curtail, pv_step_kw=window_data.pv_seq[:, 0]),
        converged=bool(converged),
        iterations=int(iteration),
        final_primal_residual=float(final_primal),
        final_dual_residual=float(final_dual),
        solve_time_sec=float(perf_counter() - started_at),
        rho_final=float(rho),
    )
