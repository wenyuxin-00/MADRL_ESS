from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd
from controllers.action_feasibility import build_safety_local_numpy, compute_action_gap_metrics_numpy
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.grid_surrogate_notebook_helpers import _battery_to_netload_sensitivity, _get_grid_projector, _projector_to_numpy
from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN, IMPORT_PRICE_MARKUP_KEY, IMPORT_PRICE_PRED_COLUMN, WHOLESALE_PRICE_PRED_COLUMN, WHOLESALE_PRICE_SEQ_FIELD, derive_import_price_seq, get_import_price_markup
_THROUGHPUT_TIEBREAKER_EUR_PER_KWH = 1e-08
_NPOS_TIEBREAKER_EUR_PER_KWH = 1e-09
_ADMM_RESIDUAL_BALANCING_MU = 10.0
_ADMM_RESIDUAL_BALANCING_TAU = 2.0
def _load_gurobi():
    import gurobipy as gp
    return (gp, gp.GRB)
def _create_model(gp: Any, name: str):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    return model
def _wrap_gurobi_error(detail: str, exc: Exception) -> RuntimeError:
    error = RuntimeError(f'Direct day optimization requires a working Gurobi installation/license: {detail}: {exc}')
    error.__cause__ = exc
    return error
@dataclass(frozen=True)
class AdmmMpcWindowData:
    wholesale_price_seq: np.ndarray
    wholesale_price_eur_per_kwh: np.ndarray
    import_price_eur_per_kwh: np.ndarray
    load_seq: np.ndarray
    pv_seq: np.ndarray
    battery_capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    eff_charge: np.ndarray
    eff_discharge: np.ndarray
    energy_init_kwh: np.ndarray
    energy_ref_kwh: np.ndarray
    energy_min_kwh: np.ndarray
    energy_max_kwh: np.ndarray
    export_subsidy_eur_per_kwh: float
    import_price_markup_eur_per_kwh: float
    dt_hours: float
    @property
    def horizon(self) -> int:
        return int(self.import_price_eur_per_kwh.shape[0])
    @property
    def n_agents(self) -> int:
        return int(self.load_seq.shape[0])
@dataclass(frozen=True)
class AdmmMpcSurrogateCache:
    trafo_limit_kw: float
    trafo_base_kw: float
    alpha_netload_kw: np.ndarray
    alpha_netload_window_kw: np.ndarray
    horizon_steps: int
@dataclass(frozen=True)
class _AdmmMpcWindowSurrogate:
    baseline_root_p_kw: np.ndarray
    export_overload_mask: np.ndarray
    import_overload_mask: np.ndarray
@dataclass(frozen=True)
class _AdmmMpcLocalWarmStart:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    net_load_kw: np.ndarray
    n_pos_kw: np.ndarray
    energy_kwh: np.ndarray
@dataclass
class AdmmMpcWarmStartCache:
    horizon_steps: int
    charge_kw: np.ndarray | None = None
    discharge_kw: np.ndarray | None = None
    pv_curtail_kw: np.ndarray | None = None
    net_load_kw: np.ndarray | None = None
    n_pos_kw: np.ndarray | None = None
    energy_kwh: np.ndarray | None = None
    z_kw: np.ndarray | None = None
    u_kw: np.ndarray | None = None
    rho_final: float | None = None
    local_solvers: tuple[Any, ...] | None = None
@dataclass(frozen=True)
class _AdmmMpcLocalSolveResult:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    net_load_kw: np.ndarray
    n_pos_kw: np.ndarray
    energy_kwh: np.ndarray
    contribution_kw: np.ndarray
    solve_time_sec: float
    status: str
@dataclass(frozen=True)
class AdmmMpcWindowResult:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    pv_effective_kw: np.ndarray
    net_load_kw: np.ndarray
    grid_import_kw: np.ndarray
    grid_export_kw: np.ndarray
    energy_kwh: np.ndarray
    surrogate_root_p_kw: np.ndarray
    baseline_root_p_kw: np.ndarray
    objective_eur: float
    converged: bool
    iterations: int
    final_primal_residual: float
    final_dual_residual: float
    solve_time_sec: float
    rho_final: float
    solver_status: str
    history_df: pd.DataFrame
@dataclass(frozen=True)
class AdmmMpcStepResult:
    executed_charge_kw: np.ndarray
    executed_discharge_kw: np.ndarray
    executed_pv_curtail_kw: np.ndarray
    executed_net_load_kw: np.ndarray
    executed_action_array: np.ndarray
    full_horizon_solution: AdmmMpcWindowResult
    converged: bool
    iterations: int
    final_primal_residual: float
    final_dual_residual: float
    solve_time_sec: float
    rho_final: float
    warm_start_cache: AdmmMpcWarmStartCache
def _coerce_1d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f'{name} must not be empty.')
    return array.astype(np.float32, copy=False)
def _coerce_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f'{name} must be 2D, got shape {array.shape}.')
    return array.astype(np.float32, copy=False)
def _clamp_rho(value: float, *, rho_min: float, rho_max: float) -> float:
    return float(np.clip(float(value), float(rho_min), float(rho_max)))
def _duplicate_last_shift_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f'values must be 2D, got shape {array.shape}.')
    if array.shape[1] == 0:
        return array.copy()
    shifted = np.empty_like(array)
    if array.shape[1] > 1:
        shifted[:, :-1] = array[:, 1:]
    shifted[:, -1] = array[:, -1]
    return shifted.astype(np.float32, copy=False)
def _shift_energy_warm_start(values: np.ndarray, *, current_energy_kwh: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    current = np.asarray(current_energy_kwh, dtype=np.float32).reshape(-1)
    if array.ndim != 2:
        raise ValueError(f'energy warm start must be 2D, got {array.shape}.')
    if array.shape[0] != current.size:
        raise ValueError(f'current_energy_kwh must have {array.shape[0]} values, got {current.size}.')
    if array.shape[1] == 0:
        return array.copy()
    shifted = np.empty_like(array)
    shifted[:, 0] = current
    if array.shape[1] > 2:
        shifted[:, 1:-1] = array[:, 2:]
    if array.shape[1] > 1:
        shifted[:, -1] = array[:, -1]
    return shifted.astype(np.float32, copy=False)
def _status_label(model, grb) -> str:
    mapping = {int(grb.OPTIMAL): 'optimal', int(grb.SUBOPTIMAL): 'suboptimal', int(grb.INFEASIBLE): 'infeasible', int(grb.INF_OR_UNBD): 'inf_or_unbd', int(grb.UNBOUNDED): 'unbounded', int(grb.TIME_LIMIT): 'time_limit'}
    return mapping.get(int(model.Status), f'status_{int(model.Status)}')
def _validate_prices(window_data: AdmmMpcWindowData) -> None:
    price_gap = np.asarray(window_data.import_price_eur_per_kwh, dtype=np.float32) - np.float32(window_data.export_subsidy_eur_per_kwh)
    min_gap = float(np.min(price_gap))
    if min_gap < -1e-09:
        raise ValueError(f'Rolling ADMM-MPC requires import_price_eur_per_kwh >= export_subsidy_eur_per_kwh, got min(import_price - subsidy)={min_gap:.6f}.')
def _day_episode_length(dt_hours: float) -> int:
    resolved = int(round(24.0 / float(dt_hours)))
    if abs(float(dt_hours) * resolved - 24.0) > 1e-09:
        raise ValueError(f'ADMM MPC requires dt in hours with one full day per episode, got dt_hours={dt_hours} and resolved episode_length={resolved}.')
    return resolved
def _objective_from_net_load(net_load_kw: np.ndarray, import_price_eur_per_kwh: np.ndarray, export_subsidy_eur_per_kwh: float, dt_hours: float) -> float:
    net_load = _coerce_2d(net_load_kw, name='net_load_kw')
    import_price = _coerce_1d(import_price_eur_per_kwh, name='import_price_eur_per_kwh')
    if net_load.shape[1] != import_price.size:
        raise ValueError(f'net_load_kw horizon {net_load.shape[1]} does not match import_price horizon {import_price.size}.')
    grid_import_kw = np.maximum(net_load, 0.0)
    grid_export_kw = np.maximum(-net_load, 0.0)
    return float(np.sum(float(dt_hours) * (import_price.reshape(1, -1) * grid_import_kw - float(export_subsidy_eur_per_kwh) * grid_export_kw)))
def _compute_surrogate_root_p_kw(net_load_kw: np.ndarray, surrogate_cache: AdmmMpcSurrogateCache) -> np.ndarray:
    net_load = _coerce_2d(net_load_kw, name='net_load_kw')
    alpha_window = _coerce_2d(surrogate_cache.alpha_netload_window_kw, name='alpha_netload_window_kw')
    if net_load.shape != alpha_window.shape:
        raise ValueError(f'net_load_kw shape {net_load.shape} must match alpha_netload_window_kw {alpha_window.shape}.')
    return (float(surrogate_cache.trafo_base_kw) + np.sum(alpha_window * net_load, axis=0)).astype(np.float32, copy=False)
def build_admm_mpc_surrogate_cache(cfg, env, *, horizon_steps: int) -> AdmmMpcSurrogateCache:
    projector = _get_grid_projector(cfg, env)
    alpha_netload_kw = _battery_to_netload_sensitivity(projector).astype(np.float32, copy=False)
    alpha_window = np.repeat(alpha_netload_kw[:, None], int(horizon_steps), axis=1).astype(np.float32, copy=False)
    trafo_base_kw = _projector_to_numpy(getattr(projector, 'trafo_power_base_kw')).reshape(-1)
    if trafo_base_kw.size == 0:
        raise ValueError('ADMM MPC surrogate requires transformer baseline data.')
    trafo_limit_kw = grid_nb._approx_trafo_limit_kw(env, loading_limit_pct=float(cfg.grid.line_max_loading_pct))
    if trafo_limit_kw is None or not np.isfinite(float(trafo_limit_kw)):
        raise ValueError('Unable to resolve a transformer power reference limit for ADMM MPC.')
    return AdmmMpcSurrogateCache(trafo_limit_kw=float(trafo_limit_kw), trafo_base_kw=float(trafo_base_kw[0]), alpha_netload_kw=alpha_netload_kw, alpha_netload_window_kw=alpha_window, horizon_steps=int(horizon_steps))
def build_admm_mpc_window_data(cfg, env, raw_obs) -> AdmmMpcWindowData:
    wholesale_price = _coerce_1d(np.asarray(raw_obs[WHOLESALE_PRICE_SEQ_FIELD], dtype=np.float32), name=WHOLESALE_PRICE_SEQ_FIELD)
    load_seq = _coerce_2d(np.asarray(raw_obs['load_seq'], dtype=np.float32), name='load_seq')
    pv_seq = _coerce_2d(np.asarray(raw_obs['pv_seq'], dtype=np.float32), name='pv_seq')
    if load_seq.shape != pv_seq.shape:
        raise ValueError(f'load_seq shape {load_seq.shape} must match pv_seq shape {pv_seq.shape}.')
    if load_seq.shape[1] != wholesale_price.size:
        raise ValueError(f'load_seq horizon {load_seq.shape[1]} must match {WHOLESALE_PRICE_SEQ_FIELD} horizon {wholesale_price.size}.')
    battery_capacity_kwh = _coerce_1d(np.asarray(env.agent_c_bat, dtype=np.float32), name='agent_c_bat')
    p_max_kw = _coerce_1d(np.asarray(env.agent_p_max, dtype=np.float32), name='agent_p_max')
    energy_init_kwh = (np.asarray(env.soc, dtype=np.float32).reshape(-1) * battery_capacity_kwh).astype(np.float32, copy=False)
    if energy_init_kwh.size != load_seq.shape[0]:
        raise ValueError(f'env.soc agent dimension {energy_init_kwh.size} must match load_seq agents {load_seq.shape[0]}.')
    import_price_markup = float(get_import_price_markup(cfg))
    import_price = derive_import_price_seq(wholesale_price, markup_eur_per_kwh=import_price_markup)
    energy_ref_kwh = (float(cfg.env.soc_target) * battery_capacity_kwh).astype(np.float32, copy=False)
    data = AdmmMpcWindowData(wholesale_price_seq=wholesale_price.copy(), wholesale_price_eur_per_kwh=wholesale_price.copy(), import_price_eur_per_kwh=import_price.copy(), load_seq=load_seq.copy(), pv_seq=pv_seq.copy(), battery_capacity_kwh=battery_capacity_kwh.copy(), p_max_kw=p_max_kw.copy(), eff_charge=np.full((load_seq.shape[0],), float(env.eff), dtype=np.float32), eff_discharge=np.full((load_seq.shape[0],), float(env.eff), dtype=np.float32), energy_init_kwh=energy_init_kwh.copy(), energy_ref_kwh=energy_ref_kwh.copy(), energy_min_kwh=(float(env.soc_min) * battery_capacity_kwh).astype(np.float32, copy=False), energy_max_kwh=(float(env.soc_max) * battery_capacity_kwh).astype(np.float32, copy=False), export_subsidy_eur_per_kwh=float(getattr(cfg.reward, 'export_subsidy_eur_per_kwh', 0.079)), import_price_markup_eur_per_kwh=import_price_markup, dt_hours=float(env.dt))
    _validate_prices(data)
    return data
def resolve_default_terminal_cost_weight(window_data: AdmmMpcWindowData, *, multiplier: float=1.0) -> np.ndarray:
    mean_import_price = float(np.mean(np.asarray(window_data.import_price_eur_per_kwh, dtype=np.float32)))
    weights = float(multiplier) * 2.0 * mean_import_price / np.maximum(np.asarray(window_data.battery_capacity_kwh, dtype=np.float32), 1.0)
    return np.asarray(weights, dtype=np.float32)
def _build_window_surrogate(window_data: AdmmMpcWindowData, surrogate_cache: AdmmMpcSurrogateCache) -> _AdmmMpcWindowSurrogate:
    baseline_net_load = (np.asarray(window_data.load_seq, dtype=np.float32) - np.asarray(window_data.pv_seq, dtype=np.float32)).astype(np.float32, copy=False)
    baseline_root_p_kw = (float(surrogate_cache.trafo_base_kw) + np.sum(np.asarray(surrogate_cache.alpha_netload_window_kw, dtype=np.float32) * baseline_net_load, axis=0)).astype(np.float32, copy=False)
    return _AdmmMpcWindowSurrogate(baseline_root_p_kw=baseline_root_p_kw, export_overload_mask=baseline_root_p_kw < -float(surrogate_cache.trafo_limit_kw) - 1e-06, import_overload_mask=baseline_root_p_kw > float(surrogate_cache.trafo_limit_kw) + 1e-06)
def _project_contribution_copies(v_kw: np.ndarray, lower_bound_kw: np.ndarray) -> np.ndarray:
    v = _coerce_2d(v_kw, name='v_kw')
    lower_bound = _coerce_1d(lower_bound_kw, name='lower_bound_kw')
    if v.shape[1] != lower_bound.size:
        raise ValueError(f'v_kw horizon {v.shape[1]} must match lower_bound horizon {lower_bound.size}.')
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
def _build_default_projected_copies(window_data: AdmmMpcWindowData, surrogate_cache: AdmmMpcSurrogateCache) -> np.ndarray:
    baseline_net_load = (np.asarray(window_data.load_seq, dtype=np.float32) - np.asarray(window_data.pv_seq, dtype=np.float32)).astype(np.float32, copy=False)
    baseline_contrib = (np.asarray(surrogate_cache.alpha_netload_window_kw, dtype=np.float32) * baseline_net_load).astype(np.float32, copy=False)
    lower_bound = np.full((window_data.horizon,), -float(surrogate_cache.trafo_limit_kw) - float(surrogate_cache.trafo_base_kw), dtype=np.float32)
    return _project_contribution_copies(baseline_contrib, lower_bound)
def _compute_default_rho(window_data: AdmmMpcWindowData, surrogate_cache: AdmmMpcSurrogateCache) -> float:
    baseline_net_load = (np.asarray(window_data.load_seq, dtype=np.float32) - np.asarray(window_data.pv_seq, dtype=np.float32)).astype(np.float32, copy=False)
    alpha_window = np.asarray(surrogate_cache.alpha_netload_window_kw, dtype=np.float32)
    return float(np.mean(np.asarray(window_data.import_price_eur_per_kwh, dtype=np.float32)) * float(window_data.dt_hours) / max(1.0, float(np.mean(np.abs(alpha_window * baseline_net_load)))))
class _ReusableAdmmMpcLocalQPSolver:
    def __init__(self, *, agent_idx: int, horizon_steps: int, battery_capacity_kwh: float, p_max_kw: float, dt_hours: float, eff_charge: float, eff_discharge: float, energy_min_kwh: float, energy_max_kwh: float) -> None:
        self.agent_idx = int(agent_idx)
        self.T = int(horizon_steps)
        self.capacity = float(battery_capacity_kwh)
        self.power_limit = float(p_max_kw)
        self.dt = float(dt_hours)
        self.eff_charge = max(float(eff_charge), 1e-06)
        self.eff_discharge = max(float(eff_discharge), 1e-06)
        self.energy_min = float(energy_min_kwh)
        self.energy_max = float(energy_max_kwh)
        try:
            self.gp, self.grb = _load_gurobi()
            self.model = _create_model(self.gp, f'admm_mpc_local_agent_{self.agent_idx}')
        except Exception as exc:
            raise _wrap_gurobi_error(f'unable to create ADMM MPC model for agent {self.agent_idx}', exc)
        self.charge = self.model.addVars(self.T, lb=0.0, ub=self.power_limit, name='charge_kw')
        self.discharge = self.model.addVars(self.T, lb=0.0, ub=self.power_limit, name='discharge_kw')
        self.pv_curtail = self.model.addVars(self.T, lb=0.0, name='pv_curtail_kw')
        self.net_load = self.model.addVars(self.T, lb=-self.grb.INFINITY, name='net_load_kw')
        self.n_pos = self.model.addVars(self.T, lb=0.0, name='n_pos_kw')
        self.energy = self.model.addVars(self.T + 1, lb=self.energy_min, ub=self.energy_max, name='energy_kwh')
        self._energy_init_constr = self.model.addConstr(self.energy[0] == 0.0, name='energy_init')
        self._pv_curtail_ub_constrs: list[Any] = []
        self._net_load_balance_constrs: list[Any] = []
        for step_idx in range(self.T):
            self._pv_curtail_ub_constrs.append(self.model.addConstr(self.pv_curtail[step_idx] <= 0.0, name=f'curtail_ub_{step_idx}'))
            self._net_load_balance_constrs.append(self.model.addConstr(self.net_load[step_idx] - self.pv_curtail[step_idx] - self.charge[step_idx] + self.discharge[step_idx] == 0.0, name=f'net_load_balance_{step_idx}'))
            self.model.addConstr(self.n_pos[step_idx] >= self.net_load[step_idx], name=f'n_pos_lb_net_{step_idx}')
            self.model.addConstr(self.energy[step_idx + 1] == self.energy[step_idx] + self.eff_charge * self.dt * self.charge[step_idx] - self.dt / self.eff_discharge * self.discharge[step_idx], name=f'energy_balance_{step_idx}')
        self.model.ModelSense = self.grb.MINIMIZE
        self.model.update()
    def dispose(self) -> None:
        dispose = getattr(self.model, 'dispose', None)
        if callable(dispose):
            dispose()
    def _apply_window_data(self, window_data: AdmmMpcWindowData) -> None:
        self._energy_init_constr.RHS = float(window_data.energy_init_kwh[self.agent_idx])
        for step_idx in range(self.T):
            self._pv_curtail_ub_constrs[step_idx].RHS = float(window_data.pv_seq[self.agent_idx, step_idx])
            self._net_load_balance_constrs[step_idx].RHS = float(window_data.load_seq[self.agent_idx, step_idx] - window_data.pv_seq[self.agent_idx, step_idx])
    def _apply_warm_start(self, warm_start: _AdmmMpcLocalWarmStart | None) -> None:
        if warm_start is None:
            return
        for step_idx in range(self.T):
            self.charge[step_idx].Start = float(warm_start.charge_kw[step_idx])
            self.discharge[step_idx].Start = float(warm_start.discharge_kw[step_idx])
            self.pv_curtail[step_idx].Start = float(warm_start.pv_curtail_kw[step_idx])
            self.net_load[step_idx].Start = float(warm_start.net_load_kw[step_idx])
            self.n_pos[step_idx].Start = float(warm_start.n_pos_kw[step_idx])
        for step_idx in range(self.T + 1):
            self.energy[step_idx].Start = float(warm_start.energy_kwh[step_idx])
    def solve(self, *, window_data: AdmmMpcWindowData, alpha_kw: np.ndarray, z_kw: np.ndarray, u_kw: np.ndarray, rho: float, terminal_ref_kwh: float, terminal_cost_weight_eur_per_kwh2: float, warm_start: _AdmmMpcLocalWarmStart | None) -> _AdmmMpcLocalSolveResult:
        alpha = _coerce_1d(alpha_kw, name='alpha_kw')
        z = _coerce_1d(z_kw, name='z_kw')
        u = _coerce_1d(u_kw, name='u_kw')
        if alpha.size != self.T or z.size != self.T or u.size != self.T:
            raise ValueError(f'alpha/z/u must have horizon {self.T}, got {alpha.size}, {z.size}, {u.size}.')
        self._apply_window_data(window_data)
        self._apply_warm_start(warm_start)
        objective = self.gp.QuadExpr()
        for step_idx in range(self.T):
            objective += float(window_data.dt_hours * window_data.export_subsidy_eur_per_kwh) * self.net_load[step_idx]
            objective += float(window_data.dt_hours * max(float(window_data.import_price_eur_per_kwh[step_idx] - window_data.export_subsidy_eur_per_kwh), 0.0)) * self.n_pos[step_idx]
            objective += float(window_data.dt_hours * _THROUGHPUT_TIEBREAKER_EUR_PER_KWH) * (self.charge[step_idx] + self.discharge[step_idx])
            objective += float(window_data.dt_hours * _NPOS_TIEBREAKER_EUR_PER_KWH) * self.n_pos[step_idx]
            if abs(float(alpha[step_idx])) > 1e-12 and float(rho) > 0.0:
                objective += float(0.5 * rho * alpha[step_idx] * alpha[step_idx]) * (self.net_load[step_idx] * self.net_load[step_idx])
                objective += float(rho * alpha[step_idx] * (u[step_idx] - z[step_idx])) * self.net_load[step_idx]
        if float(terminal_cost_weight_eur_per_kwh2) > 0.0:
            objective += float(terminal_cost_weight_eur_per_kwh2) * (self.energy[self.T] * self.energy[self.T] - 2.0 * float(terminal_ref_kwh) * self.energy[self.T])
        self.model.setObjective(objective, self.grb.MINIMIZE)
        started_at = perf_counter()
        try:
            self.model.optimize()
        except Exception as exc:
            raise _wrap_gurobi_error(f'ADMM MPC optimize failed for agent {self.agent_idx}', exc)
        solve_time_sec = float(perf_counter() - started_at)
        status = _status_label(self.model, self.grb)
        if int(self.model.Status) not in {int(self.grb.OPTIMAL), int(self.grb.SUBOPTIMAL)}:
            raise RuntimeError(f'ADMM MPC local model for agent {self.agent_idx} failed with status={status}.')
        charge_kw = np.asarray([self.charge[t].X for t in range(self.T)], dtype=np.float32)
        discharge_kw = np.asarray([self.discharge[t].X for t in range(self.T)], dtype=np.float32)
        pv_curtail_kw = np.asarray([self.pv_curtail[t].X for t in range(self.T)], dtype=np.float32)
        net_load_kw = np.asarray([self.net_load[t].X for t in range(self.T)], dtype=np.float32)
        n_pos_kw = np.asarray([self.n_pos[t].X for t in range(self.T)], dtype=np.float32)
        energy_kwh = np.asarray([self.energy[t].X for t in range(self.T + 1)], dtype=np.float32)
        return _AdmmMpcLocalSolveResult(charge_kw=charge_kw, discharge_kw=discharge_kw, pv_curtail_kw=pv_curtail_kw, net_load_kw=net_load_kw, n_pos_kw=n_pos_kw, energy_kwh=energy_kwh, contribution_kw=(alpha * net_load_kw).astype(np.float32, copy=False), solve_time_sec=solve_time_sec, status=status)
def _make_local_warm_start_cache(warm_start_cache: AdmmMpcWarmStartCache | None, *, window_data: AdmmMpcWindowData) -> list[_AdmmMpcLocalWarmStart | None]:
    if warm_start_cache is None:
        return [None] * window_data.n_agents
    required = [warm_start_cache.charge_kw, warm_start_cache.discharge_kw, warm_start_cache.pv_curtail_kw, warm_start_cache.net_load_kw, warm_start_cache.n_pos_kw, warm_start_cache.energy_kwh]
    if any((value is None for value in required)):
        return [None] * window_data.n_agents
    charge = _duplicate_last_shift_2d(np.asarray(warm_start_cache.charge_kw, dtype=np.float32))
    discharge = _duplicate_last_shift_2d(np.asarray(warm_start_cache.discharge_kw, dtype=np.float32))
    pv_curtail = _duplicate_last_shift_2d(np.asarray(warm_start_cache.pv_curtail_kw, dtype=np.float32))
    net_load = _duplicate_last_shift_2d(np.asarray(warm_start_cache.net_load_kw, dtype=np.float32))
    n_pos = _duplicate_last_shift_2d(np.asarray(warm_start_cache.n_pos_kw, dtype=np.float32))
    energy = _shift_energy_warm_start(np.asarray(warm_start_cache.energy_kwh, dtype=np.float32), current_energy_kwh=np.asarray(window_data.energy_init_kwh, dtype=np.float32))
    return [_AdmmMpcLocalWarmStart(charge_kw=charge[agent_idx], discharge_kw=discharge[agent_idx], pv_curtail_kw=pv_curtail[agent_idx], net_load_kw=net_load[agent_idx], n_pos_kw=n_pos[agent_idx], energy_kwh=energy[agent_idx]) for agent_idx in range(window_data.n_agents)]
def _copy_local_solvers(warm_start_cache: AdmmMpcWarmStartCache | None, *, window_data: AdmmMpcWindowData) -> tuple[_ReusableAdmmMpcLocalQPSolver, ...] | None:
    if warm_start_cache is None or warm_start_cache.local_solvers is None:
        return None
    if int(warm_start_cache.horizon_steps) != int(window_data.horizon):
        return None
    local_solvers = tuple(warm_start_cache.local_solvers)
    if len(local_solvers) != window_data.n_agents:
        return None
    return local_solvers
def _build_or_reuse_local_solvers(warm_start_cache: AdmmMpcWarmStartCache | None, *, window_data: AdmmMpcWindowData) -> tuple[_ReusableAdmmMpcLocalQPSolver, ...]:
    reused = _copy_local_solvers(warm_start_cache, window_data=window_data)
    if reused is not None:
        return reused
    if warm_start_cache is not None and warm_start_cache.local_solvers is not None:
        for solver in warm_start_cache.local_solvers:
            dispose = getattr(solver, 'dispose', None)
            if callable(dispose):
                dispose()
    return tuple((_ReusableAdmmMpcLocalQPSolver(agent_idx=agent_idx, horizon_steps=window_data.horizon, battery_capacity_kwh=float(window_data.battery_capacity_kwh[agent_idx]), p_max_kw=float(window_data.p_max_kw[agent_idx]), dt_hours=float(window_data.dt_hours), eff_charge=float(window_data.eff_charge[agent_idx]), eff_discharge=float(window_data.eff_discharge[agent_idx]), energy_min_kwh=float(window_data.energy_min_kwh[agent_idx]), energy_max_kwh=float(window_data.energy_max_kwh[agent_idx])) for agent_idx in range(window_data.n_agents)))
def solve_admm_mpc_window(window_data: AdmmMpcWindowData, surrogate_cache: AdmmMpcSurrogateCache, *, terminal_cost_weight_eur_per_kwh2: np.ndarray, warm_start_cache: AdmmMpcWarmStartCache | None, rho_init: float, rho_min: float, rho_max: float, rho_adaptation: str | None, max_iters: int, max_iters_first_step: int, primal_tol: float, dual_tol: float) -> AdmmMpcWindowResult:
    window_result, _ = _solve_admm_mpc_window_with_cache(window_data, surrogate_cache, terminal_cost_weight_eur_per_kwh2=terminal_cost_weight_eur_per_kwh2, warm_start_cache=warm_start_cache, rho_init=rho_init, rho_min=rho_min, rho_max=rho_max, rho_adaptation=rho_adaptation, max_iters=max_iters, max_iters_first_step=max_iters_first_step, primal_tol=primal_tol, dual_tol=dual_tol)
    return window_result
def _build_next_warm_start_cache(window_result: AdmmMpcWindowResult, *, z_kw: np.ndarray, u_kw: np.ndarray, rho_final: float, local_solvers: tuple[_ReusableAdmmMpcLocalQPSolver, ...]) -> AdmmMpcWarmStartCache:
    n_agents, horizon = window_result.charge_kw.shape
    n_pos_kw = np.maximum(window_result.net_load_kw, 0.0).astype(np.float32, copy=False)
    return AdmmMpcWarmStartCache(horizon_steps=int(horizon), charge_kw=np.asarray(window_result.charge_kw, dtype=np.float32).copy(), discharge_kw=np.asarray(window_result.discharge_kw, dtype=np.float32).copy(), pv_curtail_kw=np.asarray(window_result.pv_curtail_kw, dtype=np.float32).copy(), net_load_kw=np.asarray(window_result.net_load_kw, dtype=np.float32).copy(), n_pos_kw=np.asarray(n_pos_kw, dtype=np.float32).copy(), energy_kwh=np.asarray(window_result.energy_kwh, dtype=np.float32).copy(), z_kw=np.asarray(z_kw, dtype=np.float32).reshape(n_agents, horizon).copy(), u_kw=np.asarray(u_kw, dtype=np.float32).reshape(n_agents, horizon).copy(), rho_final=float(rho_final), local_solvers=tuple(local_solvers))
def _solve_admm_mpc_window_with_cache(window_data: AdmmMpcWindowData, surrogate_cache: AdmmMpcSurrogateCache, *, terminal_cost_weight_eur_per_kwh2: np.ndarray, warm_start_cache: AdmmMpcWarmStartCache | None, rho_init: float, rho_min: float, rho_max: float, rho_adaptation: str | None, max_iters: int, max_iters_first_step: int, primal_tol: float, dual_tol: float) -> tuple[AdmmMpcWindowResult, AdmmMpcWarmStartCache]:
    _validate_prices(window_data)
    if int(surrogate_cache.horizon_steps) != int(window_data.horizon):
        raise ValueError(f'surrogate_cache.horizon_steps={surrogate_cache.horizon_steps} must match window horizon {window_data.horizon}.')
    terminal_weight = _coerce_1d(np.asarray(terminal_cost_weight_eur_per_kwh2, dtype=np.float32), name='terminal_cost_weight_eur_per_kwh2')
    if terminal_weight.size != window_data.n_agents:
        raise ValueError(f'terminal_cost_weight_eur_per_kwh2 must provide {window_data.n_agents} values, got {terminal_weight.size}.')
    cold_start = warm_start_cache is None or int(warm_start_cache.horizon_steps) != int(window_data.horizon) or warm_start_cache.z_kw is None or (warm_start_cache.u_kw is None)
    local_solvers = _build_or_reuse_local_solvers(warm_start_cache, window_data=window_data)
    local_warm_starts = _make_local_warm_start_cache(warm_start_cache, window_data=window_data)
    window_surrogate = _build_window_surrogate(window_data, surrogate_cache)
    lower_bound_kw = np.full((window_data.horizon,), -float(surrogate_cache.trafo_limit_kw) - float(surrogate_cache.trafo_base_kw), dtype=np.float32)
    default_rho = _compute_default_rho(window_data, surrogate_cache)
    if cold_start or warm_start_cache is None or warm_start_cache.rho_final is None:
        rho = _clamp_rho(rho_init if rho_init is not None else default_rho, rho_min=rho_min, rho_max=rho_max)
        z_kw = _build_default_projected_copies(window_data, surrogate_cache)
        u_kw = np.zeros_like(z_kw, dtype=np.float32)
        iteration_budget = int(max_iters_first_step)
    else:
        rho = _clamp_rho(float(warm_start_cache.rho_final), rho_min=rho_min, rho_max=rho_max)
        z_kw = _duplicate_last_shift_2d(np.asarray(warm_start_cache.z_kw, dtype=np.float32))
        u_kw = _duplicate_last_shift_2d(np.asarray(warm_start_cache.u_kw, dtype=np.float32))
        iteration_budget = int(max_iters)
    history_rows: list[dict[str, float | int]] = []
    final_local_results: list[_AdmmMpcLocalSolveResult] = []
    converged = False
    final_primal = float('inf')
    final_dual = float('inf')
    started_at = perf_counter()
    for iteration in range(1, iteration_budget + 1):
        iteration_started_at = perf_counter()
        local_results = [local_solvers[agent_idx].solve(window_data=window_data, alpha_kw=np.asarray(surrogate_cache.alpha_netload_window_kw[agent_idx], dtype=np.float32), z_kw=z_kw[agent_idx], u_kw=u_kw[agent_idx], rho=float(rho), terminal_ref_kwh=float(window_data.energy_ref_kwh[agent_idx]), terminal_cost_weight_eur_per_kwh2=float(terminal_weight[agent_idx]), warm_start=local_warm_starts[agent_idx] if iteration == 1 else None) for agent_idx in range(window_data.n_agents)]
        c_kw = np.asarray([result.contribution_kw for result in local_results], dtype=np.float32)
        v_kw = (c_kw + u_kw).astype(np.float32, copy=False)
        z_next_kw = _project_contribution_copies(v_kw, lower_bound_kw)
        u_next_kw = (u_kw + c_kw - z_next_kw).astype(np.float32, copy=False)
        primal_residual = float(np.linalg.norm((c_kw - z_next_kw).reshape(-1), ord=2))
        dual_residual = float(rho * np.linalg.norm((z_next_kw - z_kw).reshape(-1), ord=2))
        charge_kw = np.asarray([result.charge_kw for result in local_results], dtype=np.float32)
        discharge_kw = np.asarray([result.discharge_kw for result in local_results], dtype=np.float32)
        pv_curtail_kw = np.asarray([result.pv_curtail_kw for result in local_results], dtype=np.float32)
        energy_kwh = np.asarray([result.energy_kwh for result in local_results], dtype=np.float32)
        pv_effective_kw = (np.asarray(window_data.pv_seq, dtype=np.float32) - pv_curtail_kw).astype(np.float32, copy=False)
        net_load_kw = (np.asarray(window_data.load_seq, dtype=np.float32) - pv_effective_kw + charge_kw - discharge_kw).astype(np.float32, copy=False)
        candidate_root_p_kw = _compute_surrogate_root_p_kw(net_load_kw, surrogate_cache)
        candidate_objective_eur = _objective_from_net_load(net_load_kw, import_price_eur_per_kwh=window_data.import_price_eur_per_kwh, export_subsidy_eur_per_kwh=window_data.export_subsidy_eur_per_kwh, dt_hours=window_data.dt_hours)
        history_rows.append({'iteration': int(iteration), 'objective_estimate': float(candidate_objective_eur), 'primal_residual': float(primal_residual), 'dual_residual': float(dual_residual), 'export_violation_kw_max': float(np.max(np.maximum(-candidate_root_p_kw - float(surrogate_cache.trafo_limit_kw), 0.0))), 'iter_runtime_sec': float(perf_counter() - iteration_started_at), 'rho_value': float(rho)})
        final_local_results = local_results
        final_primal = float(primal_residual)
        final_dual = float(dual_residual)
        z_kw = z_next_kw
        u_kw = u_next_kw
        if primal_residual <= float(primal_tol) and dual_residual <= float(dual_tol):
            converged = True
            break
        if str(rho_adaptation or '').strip().lower() == 'residual_balancing':
            rho_before = float(rho)
            rho_after = rho_before
            if primal_residual > _ADMM_RESIDUAL_BALANCING_MU * dual_residual and rho_before > 0.0:
                rho_after = _clamp_rho(rho_before * _ADMM_RESIDUAL_BALANCING_TAU, rho_min=rho_min, rho_max=rho_max)
            elif dual_residual > _ADMM_RESIDUAL_BALANCING_MU * primal_residual and rho_before > 0.0:
                rho_after = _clamp_rho(rho_before / _ADMM_RESIDUAL_BALANCING_TAU, rho_min=rho_min, rho_max=rho_max)
            if not np.isclose(rho_after, rho_before):
                scale = float(rho_before / max(rho_after, 1e-09))
                u_kw = (u_kw * np.float32(scale)).astype(np.float32, copy=False)
                rho = float(rho_after)
    if not final_local_results:
        raise RuntimeError('ADMM MPC did not produce any iterate.')
    charge_kw = np.asarray([result.charge_kw for result in final_local_results], dtype=np.float32)
    discharge_kw = np.asarray([result.discharge_kw for result in final_local_results], dtype=np.float32)
    pv_curtail_kw = np.asarray([result.pv_curtail_kw for result in final_local_results], dtype=np.float32)
    energy_kwh = np.asarray([result.energy_kwh for result in final_local_results], dtype=np.float32)
    pv_effective_kw = (np.asarray(window_data.pv_seq, dtype=np.float32) - pv_curtail_kw).astype(np.float32, copy=False)
    net_load_kw = (np.asarray(window_data.load_seq, dtype=np.float32) - pv_effective_kw + charge_kw - discharge_kw).astype(np.float32, copy=False)
    window_result = AdmmMpcWindowResult(charge_kw=charge_kw, discharge_kw=discharge_kw, pv_curtail_kw=pv_curtail_kw, pv_effective_kw=pv_effective_kw, net_load_kw=net_load_kw, grid_import_kw=np.maximum(net_load_kw, 0.0).astype(np.float32, copy=False), grid_export_kw=np.maximum(-net_load_kw, 0.0).astype(np.float32, copy=False), energy_kwh=energy_kwh, surrogate_root_p_kw=_compute_surrogate_root_p_kw(net_load_kw, surrogate_cache), baseline_root_p_kw=window_surrogate.baseline_root_p_kw, objective_eur=float(_objective_from_net_load(net_load_kw, import_price_eur_per_kwh=window_data.import_price_eur_per_kwh, export_subsidy_eur_per_kwh=window_data.export_subsidy_eur_per_kwh, dt_hours=window_data.dt_hours)), converged=bool(converged), iterations=int(len(history_rows)), final_primal_residual=float(final_primal), final_dual_residual=float(final_dual), solve_time_sec=float(perf_counter() - started_at), rho_final=float(_clamp_rho(rho, rho_min=rho_min, rho_max=rho_max)), solver_status='admm_mpc_converged' if converged else 'admm_mpc_not_converged', history_df=pd.DataFrame(history_rows))
    next_cache = _build_next_warm_start_cache(window_result, z_kw=z_kw, u_kw=u_kw, rho_final=float(_clamp_rho(rho, rho_min=rho_min, rho_max=rho_max)), local_solvers=local_solvers)
    return (window_result, next_cache)
def _clip_first_step_battery_power_kw(env, requested_battery_power_kw: np.ndarray) -> np.ndarray:
    battery_power_kw = np.asarray(requested_battery_power_kw, dtype=np.float32)
    e_t = np.asarray(env.soc, dtype=np.float32) * np.asarray(env.agent_c_bat, dtype=np.float32)
    e_min = float(env.soc_min) * np.asarray(env.agent_c_bat, dtype=np.float32)
    e_max = float(env.soc_max) * np.asarray(env.agent_c_bat, dtype=np.float32)
    p_max = np.asarray(env.agent_p_max, dtype=np.float32)
    eff = max(float(env.eff), 1e-06)
    dt = float(env.dt)
    p_max_charge = np.minimum(p_max, np.maximum(0.0, (e_max - e_t) / (eff * dt)))
    p_max_discharge = np.minimum(p_max, np.maximum(0.0, (e_t - e_min) * eff / dt))
    return np.clip(battery_power_kw, -p_max_discharge, p_max_charge).astype(np.float32)
def _first_step_action_array(env, window_result: AdmmMpcWindowResult) -> np.ndarray:
    requested_battery_power_kw = (np.asarray(window_result.charge_kw[:, 0], dtype=np.float32) - np.asarray(window_result.discharge_kw[:, 0], dtype=np.float32)).astype(np.float32, copy=False)
    clipped_battery_power_kw = _clip_first_step_battery_power_kw(env, requested_battery_power_kw)
    p_max_kw = np.maximum(np.asarray(env.agent_p_max, dtype=np.float32), 1e-06)
    battery_action = np.clip(clipped_battery_power_kw / p_max_kw, -1.0, 1.0).astype(np.float32, copy=False)
    pv_raw_kw = np.maximum(np.asarray(window_result.pv_effective_kw[:, 0] + window_result.pv_curtail_kw[:, 0], dtype=np.float32), 0.0)
    pv_effective_kw = np.asarray(window_result.pv_effective_kw[:, 0], dtype=np.float32)
    pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
    valid_mask = pv_raw_kw > 1e-06
    pv_utilization[valid_mask] = (pv_effective_kw[valid_mask] / pv_raw_kw[valid_mask]).astype(np.float32, copy=False)
    pv_action = np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0).astype(np.float32, copy=False)
    return np.stack([battery_action, pv_action], axis=-1).astype(np.float32, copy=False)
def run_admm_mpc_step(env, raw_obs, cfg, *, surrogate_cache: AdmmMpcSurrogateCache, warm_start_cache: AdmmMpcWarmStartCache | None, rho_init: float, rho_min: float, rho_max: float, rho_adaptation: str | None, max_iters: int, max_iters_first_step: int, primal_tol: float, dual_tol: float, terminal_cost_weight_eur_per_kwh2: np.ndarray) -> AdmmMpcStepResult:
    window_data = build_admm_mpc_window_data(cfg, env, raw_obs)
    window_result, next_cache = _solve_admm_mpc_window_with_cache(window_data, surrogate_cache, terminal_cost_weight_eur_per_kwh2=terminal_cost_weight_eur_per_kwh2, warm_start_cache=warm_start_cache, rho_init=float(rho_init), rho_min=float(rho_min), rho_max=float(rho_max), rho_adaptation=rho_adaptation, max_iters=int(max_iters), max_iters_first_step=int(max_iters_first_step), primal_tol=float(primal_tol), dual_tol=float(dual_tol))
    action_array = _first_step_action_array(env, window_result)
    return AdmmMpcStepResult(executed_charge_kw=np.asarray(window_result.charge_kw[:, 0], dtype=np.float32), executed_discharge_kw=np.asarray(window_result.discharge_kw[:, 0], dtype=np.float32), executed_pv_curtail_kw=np.asarray(window_result.pv_curtail_kw[:, 0], dtype=np.float32), executed_net_load_kw=np.asarray(window_result.net_load_kw[:, 0], dtype=np.float32), executed_action_array=action_array, full_horizon_solution=window_result, converged=bool(window_result.converged), iterations=int(window_result.iterations), final_primal_residual=float(window_result.final_primal_residual), final_dual_residual=float(window_result.final_dual_residual), solve_time_sec=float(window_result.solve_time_sec), rho_final=float(window_result.rho_final), warm_start_cache=next_cache)
