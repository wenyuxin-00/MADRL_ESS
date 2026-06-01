from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from configs.cfg import Cfg
from envs.grid_env import action_array_from_power
from utils.price import derive_import_price_seq, get_import_price_markup

LOCAL_MPC_TIME_LIMIT_SEC = 2.0
LOCAL_MPC_MIP_GAP = 0.01
BATTERY_THROUGHPUT_TIEBREAKER_EPS_EUR_PER_KWH = 5e-5


@dataclass(frozen=True)
class LocalMPCFullHorizonResult:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    pv_effective_kw: np.ndarray
    pv_utilization: np.ndarray
    signed_battery_kw: np.ndarray
    ev_charge_kw: np.ndarray
    net_load_kw: np.ndarray
    energy_kwh: np.ndarray
    objective_eur: float
    solve_time_sec: float
    grid_import_kw: np.ndarray
    grid_export_kw: np.ndarray


@dataclass(frozen=True)
class _LocalMPCPrimalSolution:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    ev_charge_kw: np.ndarray
    energy_kwh: np.ndarray
    ev_energy_kwh: np.ndarray


def _shift_primal_solution(solution: _LocalMPCPrimalSolution, current_energy_kwh: float) -> _LocalMPCPrimalSolution:
    energy = np.asarray(solution.energy_kwh, dtype=np.float32).reshape(-1).copy()
    energy[0] = np.float32(current_energy_kwh)
    if energy.size > 2:
        energy[1:-1] = np.asarray(solution.energy_kwh, dtype=np.float32).reshape(-1)[2:]
    if energy.size > 1:
        energy[-1] = np.asarray(solution.energy_kwh, dtype=np.float32).reshape(-1)[-2]

    def shift(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        return np.concatenate([values[1:], np.zeros((1,), dtype=np.float32)]).astype(np.float32)

    ev_energy = np.asarray(solution.ev_energy_kwh, dtype=np.float32).reshape(-1).copy()
    if ev_energy.size > 2:
        ev_energy[1:-1] = np.asarray(solution.ev_energy_kwh, dtype=np.float32).reshape(-1)[2:]
    if ev_energy.size > 1:
        ev_energy[-1] = np.asarray(solution.ev_energy_kwh, dtype=np.float32).reshape(-1)[-2]
    return _LocalMPCPrimalSolution(shift(solution.charge_kw), shift(solution.discharge_kw), shift(solution.pv_curtail_kw), shift(solution.ev_charge_kw), energy, ev_energy)


def _ev_available(cfg: Cfg, step: int) -> bool:
    daily_step = int(step) % int(cfg.env.episode_steps)
    arr, dep = int(cfg.env.ev_arrival_step), int(cfg.env.ev_departure_step)
    return bool(daily_step >= arr or daily_step < dep) if arr > dep else bool(arr <= daily_step < dep)


def _ev_departure_offset(cfg: Cfg, start_step: int, horizon: int) -> int | None:
    dep = int(cfg.env.ev_departure_step)
    for offset in range(1, int(horizon) + 1):
        if (int(start_step) + offset) % int(cfg.env.episode_steps) == dep:
            return int(offset)
    return None


class ReusableLocalMPCSolver:
    def __init__(self, *, cfg: Cfg, horizon: int, battery_capacity_kwh: float, p_max_kw: float, dt_hours: float, efficiency: float, soc_min: float, soc_max: float, ev_capacity_kwh: float, ev_p_max_kw: float) -> None:
        import gurobipy as gp

        self.gp, self.grb = gp, gp.GRB
        self.cfg = cfg
        self.horizon = int(horizon)
        self.capacity = float(battery_capacity_kwh)
        self.power_limit = float(p_max_kw)
        self.dt = float(dt_hours)
        self.eff = float(efficiency)
        self.energy_min = float(soc_min) * self.capacity
        self.energy_max = float(soc_max) * self.capacity
        self.ev_capacity = float(ev_capacity_kwh)
        self.ev_power_limit = float(ev_p_max_kw)
        self.ev_energy_min = float(cfg.env.ev_soc_min) * self.ev_capacity
        self.ev_energy_max = float(cfg.env.ev_soc_max) * self.ev_capacity
        self.last_solution: _LocalMPCPrimalSolution | None = None

        self.model = gp.Model("remake_local_mpc")
        self.model.Params.OutputFlag = 0
        self.model.Params.TimeLimit = LOCAL_MPC_TIME_LIMIT_SEC
        self.model.Params.MIPGap = LOCAL_MPC_MIP_GAP
        self.charge = self.model.addVars(self.horizon, lb=0.0, ub=self.power_limit, name="p_charge")
        self.discharge = self.model.addVars(self.horizon, lb=0.0, ub=self.power_limit, name="p_discharge")
        self.pv_curtail = self.model.addVars(self.horizon, lb=0.0, name="pv_curtail")
        self.ev_charge = self.model.addVars(self.horizon, lb=0.0, ub=self.ev_power_limit, name="ev_charge_kw")
        self.energy = self.model.addVars(self.horizon + 1, lb=self.energy_min, ub=self.energy_max, name="energy")
        self.ev_energy = self.model.addVars(self.horizon + 1, lb=self.ev_energy_min, ub=self.ev_energy_max, name="ev_energy_kwh")
        self.ev_departure_gap = self.model.addVar(lb=0.0, ub=1.0, name="ev_departure_gap_soc")
        self.grid_import = self.model.addVars(self.horizon, lb=0.0, name="grid_import")
        self.grid_export = self.model.addVars(self.horizon, lb=0.0, name="grid_export")
        self.grid_mode = self.model.addVars(self.horizon, vtype=self.grb.BINARY, name="grid_mode")
        self.energy_init_constr = self.model.addConstr(self.energy[0] == 0.0, name="energy_init")
        self.ev_energy_init_constr = self.model.addConstr(self.ev_energy[0] == 0.0, name="ev_energy_init")
        self.ev_departure_gap_constr = self.model.addConstr(self.ev_departure_gap >= 0.0, name="ev_departure_gap")
        self.ev_departure_energy_coeffs = np.zeros((self.horizon + 1,), dtype=np.float32)
        self.net_load_floor_constrs: list[Any] = []
        self.pv_curtail_upper_constrs: list[Any] = []
        self.ev_charge_upper_constrs: list[Any] = []
        self.grid_balance_constrs: list[Any] = []
        self.grid_import_ub_constrs: list[Any] = []
        self.grid_export_ub_constrs: list[Any] = []
        self.grid_import_gate_constrs: list[Any] = []
        self.grid_export_gate_constrs: list[Any] = []
        for step_idx in range(self.horizon):
            self.model.addConstr(self.energy[step_idx + 1] == self.energy[step_idx] + self.eff * self.charge[step_idx] * self.dt - self.discharge[step_idx] / self.eff * self.dt, name=f"energy_balance_{step_idx}")
            self.model.addConstr(self.ev_energy[step_idx + 1] == self.ev_energy[step_idx] + float(cfg.env.ev_efficiency) * self.ev_charge[step_idx] * self.dt, name=f"ev_energy_balance_{step_idx}")
            self.net_load_floor_constrs.append(self.model.addConstr(self.charge[step_idx] - self.discharge[step_idx] + self.pv_curtail[step_idx] >= -2.0 * self.power_limit, name=f"net_load_floor_{step_idx}"))
            self.pv_curtail_upper_constrs.append(self.model.addConstr(self.pv_curtail[step_idx] <= 0.0, name=f"pv_curtail_ub_{step_idx}"))
            self.ev_charge_upper_constrs.append(self.model.addConstr(self.ev_charge[step_idx] <= 0.0, name=f"ev_charge_ub_{step_idx}"))
            self.grid_balance_constrs.append(self.model.addConstr(self.grid_import[step_idx] - self.grid_export[step_idx] - self.charge[step_idx] + self.discharge[step_idx] - self.pv_curtail[step_idx] - self.ev_charge[step_idx] == 0.0, name=f"grid_balance_{step_idx}"))
            self.grid_import_ub_constrs.append(self.model.addConstr(self.grid_import[step_idx] <= 0.0, name=f"grid_import_ub_{step_idx}"))
            self.grid_export_ub_constrs.append(self.model.addConstr(self.grid_export[step_idx] <= 0.0, name=f"grid_export_ub_{step_idx}"))
            self.grid_import_gate_constrs.append(self.model.addConstr(self.grid_import[step_idx] - self.grid_mode[step_idx] <= 0.0, name=f"grid_import_gate_{step_idx}"))
            self.grid_export_gate_constrs.append(self.model.addConstr(self.grid_export[step_idx] + self.grid_mode[step_idx] <= 1.0, name=f"grid_export_gate_{step_idx}"))
        self.model.ModelSense = self.grb.MINIMIZE
        self.model.update()

    def dispose(self) -> None:
        self.model.dispose()

    def _apply_problem_data(self, *, prices: np.ndarray, net_load: np.ndarray, pv_available_kw: np.ndarray, energy_now: float, ev_energy_now: float, start_step: int) -> None:
        self.energy_init_constr.RHS = float(energy_now)
        self.ev_energy_init_constr.RHS = float(ev_energy_now)
        throughput_penalty = BATTERY_THROUGHPUT_TIEBREAKER_EPS_EUR_PER_KWH * self.dt
        curtail_upper = np.maximum(np.asarray(pv_available_kw, dtype=np.float32).reshape(-1)[: self.horizon], 0.0)
        objective = self.gp.QuadExpr()
        objective += float(getattr(self.cfg.reward, "ev_departure_penalty_weight", 0.0)) * self.ev_departure_gap * self.ev_departure_gap
        dep_offset = _ev_departure_offset(self.cfg, int(start_step), self.horizon)
        for idx, coeff in enumerate(self.ev_departure_energy_coeffs):
            if abs(float(coeff)) > 0.0:
                self.model.chgCoeff(self.ev_departure_gap_constr, self.ev_energy[idx], 0.0)
                self.ev_departure_energy_coeffs[idx] = np.float32(0.0)
        if bool(getattr(self.cfg.env, "ev_enabled", False)) and dep_offset is not None:
            self.ev_departure_gap_constr.RHS = float(self.cfg.env.ev_departure_soc_req)
            coeff = np.float32(1.0 / max(self.ev_capacity, 1e-6))
            self.model.chgCoeff(self.ev_departure_gap_constr, self.ev_energy[int(dep_offset)], float(coeff))
            self.ev_departure_energy_coeffs[int(dep_offset)] = coeff
        else:
            self.ev_departure_gap_constr.RHS = 0.0
        for step_idx in range(self.horizon):
            gross_flow_limit_kw = abs(float(net_load[step_idx])) + self.power_limit + float(curtail_upper[step_idx])
            self.net_load_floor_constrs[step_idx].RHS = -2.0 * self.power_limit
            self.pv_curtail_upper_constrs[step_idx].RHS = float(curtail_upper[step_idx])
            self.ev_charge_upper_constrs[step_idx].RHS = self.ev_power_limit if bool(getattr(self.cfg.env, "ev_enabled", False)) and _ev_available(self.cfg, int(start_step) + step_idx) else 0.0
            self.grid_balance_constrs[step_idx].RHS = float(net_load[step_idx])
            gross_flow_limit_kw += self.ev_power_limit
            self.grid_import_ub_constrs[step_idx].RHS = gross_flow_limit_kw
            self.grid_export_ub_constrs[step_idx].RHS = gross_flow_limit_kw
            self.model.chgCoeff(self.grid_import_gate_constrs[step_idx], self.grid_mode[step_idx], -gross_flow_limit_kw)
            self.grid_import_gate_constrs[step_idx].RHS = 0.0
            self.model.chgCoeff(self.grid_export_gate_constrs[step_idx], self.grid_mode[step_idx], gross_flow_limit_kw)
            self.grid_export_gate_constrs[step_idx].RHS = gross_flow_limit_kw
            objective += (float(prices[step_idx]) * self.dt + throughput_penalty) * self.charge[step_idx]
            objective += (-float(prices[step_idx]) * self.dt + throughput_penalty) * self.discharge[step_idx]
            objective += float(prices[step_idx]) * self.dt * self.ev_charge[step_idx]
        self.model.setObjective(objective, self.grb.MINIMIZE)

    def _apply_warm_start(self, energy_now: float) -> None:
        if self.last_solution is None:
            return
        shifted = _shift_primal_solution(self.last_solution, energy_now)
        for step_idx in range(self.horizon):
            self.charge[step_idx].Start = float(shifted.charge_kw[step_idx])
            self.discharge[step_idx].Start = float(shifted.discharge_kw[step_idx])
            self.pv_curtail[step_idx].Start = float(shifted.pv_curtail_kw[step_idx])
            self.ev_charge[step_idx].Start = float(shifted.ev_charge_kw[step_idx])
        for step_idx in range(self.horizon + 1):
            self.energy[step_idx].Start = float(shifted.energy_kwh[step_idx])
            self.ev_energy[step_idx].Start = float(shifted.ev_energy_kwh[step_idx])

    def _extract_solution(self) -> _LocalMPCPrimalSolution:
        return _LocalMPCPrimalSolution(
            charge_kw=np.asarray([self.charge[idx].X for idx in range(self.horizon)], dtype=np.float32),
            discharge_kw=np.asarray([self.discharge[idx].X for idx in range(self.horizon)], dtype=np.float32),
            pv_curtail_kw=np.asarray([self.pv_curtail[idx].X for idx in range(self.horizon)], dtype=np.float32),
            ev_charge_kw=np.asarray([self.ev_charge[idx].X for idx in range(self.horizon)], dtype=np.float32),
            energy_kwh=np.asarray([self.energy[idx].X for idx in range(self.horizon + 1)], dtype=np.float32),
            ev_energy_kwh=np.asarray([self.ev_energy[idx].X for idx in range(self.horizon + 1)], dtype=np.float32),
        )

    def solve_full_horizon(self, *, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc: float, ev_soc: float, start_step: int) -> LocalMPCFullHorizonResult:
        prices = np.asarray(import_price_seq, dtype=np.float32).reshape(-1)[: self.horizon]
        load = np.asarray(load_seq, dtype=np.float32).reshape(-1)[: self.horizon]
        pv = np.asarray(pv_seq, dtype=np.float32).reshape(-1)[: self.horizon]
        energy_now = float(soc) * self.capacity
        ev_energy_now = float(ev_soc) * self.ev_capacity
        net_load = (load - pv).astype(np.float32)
        pv_available_kw = np.maximum(pv, 0.0).astype(np.float32)
        self._apply_problem_data(prices=prices, net_load=net_load, pv_available_kw=pv_available_kw, energy_now=energy_now, ev_energy_now=ev_energy_now, start_step=int(start_step))
        self._apply_warm_start(energy_now)
        self.model.optimize()
        solve_time_sec = float(self.model.Runtime)
        solution = self._extract_solution()
        self.last_solution = solution
        pv_curtail_kw = np.minimum(np.maximum(solution.pv_curtail_kw, 0.0), pv_available_kw).astype(np.float32)
        pv_effective_kw = np.maximum(pv_available_kw - pv_curtail_kw, 0.0).astype(np.float32)
        pv_utilization = np.ones_like(pv_available_kw, dtype=np.float32)
        valid_pv_mask = pv_available_kw > 1e-6
        pv_utilization[valid_pv_mask] = np.clip(pv_effective_kw[valid_pv_mask] / pv_available_kw[valid_pv_mask], 0.0, 1.0).astype(np.float32)
        signed_battery_kw = (solution.charge_kw - solution.discharge_kw).astype(np.float32)
        ev_charge_kw = np.maximum(solution.ev_charge_kw, 0.0).astype(np.float32)
        net_grid_kw = (net_load + pv_curtail_kw + signed_battery_kw + ev_charge_kw).astype(np.float32)
        return LocalMPCFullHorizonResult(
            charge_kw=solution.charge_kw.copy(),
            discharge_kw=solution.discharge_kw.copy(),
            pv_curtail_kw=pv_curtail_kw.copy(),
            pv_effective_kw=pv_effective_kw.copy(),
            pv_utilization=pv_utilization.copy(),
            signed_battery_kw=signed_battery_kw.copy(),
            ev_charge_kw=ev_charge_kw.copy(),
            net_load_kw=net_grid_kw.copy(),
            energy_kwh=solution.energy_kwh.copy(),
            objective_eur=float(np.sum((prices * solution.charge_kw - prices * solution.discharge_kw + prices * ev_charge_kw) * self.dt)),
            solve_time_sec=solve_time_sec,
            grid_import_kw=np.maximum(net_grid_kw, 0.0).astype(np.float32),
            grid_export_kw=np.maximum(-net_grid_kw, 0.0).astype(np.float32),
        )


class LocalMPCController:
    name = "LOCAL_MPC"

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg; self.cache: dict[tuple[Any, ...], ReusableLocalMPCSolver] = {}

    def reset(self, env_state: dict) -> None:
        pass

    def __call__(self, env, window: dict[str, np.ndarray]) -> tuple[list[np.ndarray], dict[str, np.ndarray], dict[str, object]]:
        import_price_seq = derive_import_price_seq(window["price_seq"], markup_eur_per_kwh=float(get_import_price_markup(env)))
        load_seq, pv_seq = np.asarray(window["load_seq"], dtype=np.float32), np.asarray(window["pv_seq"], dtype=np.float32)
        battery_power_kw = np.zeros((int(env.n),), dtype=np.float32); pv_curtail_kw = np.zeros((int(env.n),), dtype=np.float32); ev_charge_kw = np.zeros((int(env.n),), dtype=np.float32); solve_time_sec = 0.0
        for agent_idx in range(int(env.n)):
            key = (agent_idx, int(import_price_seq.size), float(env.cap[agent_idx]), float(env.pmax[agent_idx]), float(env.cfg.env.dt_hours), float(env.cfg.env.efficiency), float(env.cfg.env.soc_min), float(env.cfg.env.soc_max), bool(getattr(env.cfg.env, "ev_enabled", False)), float(env.ev_cap[agent_idx]), float(env.ev_pmax[agent_idx]), str(getattr(env.cfg.env, "ev_departure_constraint_mode", "soft")))
            solver = self.cache.get(key)
            if solver is None:
                solver = ReusableLocalMPCSolver(cfg=env.cfg, horizon=int(import_price_seq.size), battery_capacity_kwh=float(env.cap[agent_idx]), p_max_kw=float(env.pmax[agent_idx]), dt_hours=float(env.cfg.env.dt_hours), efficiency=float(env.cfg.env.efficiency), soc_min=float(env.cfg.env.soc_min), soc_max=float(env.cfg.env.soc_max), ev_capacity_kwh=float(env.ev_cap[agent_idx]), ev_p_max_kw=float(env.ev_pmax[agent_idx]))
                self.cache[key] = solver
            result = solver.solve_full_horizon(import_price_seq=import_price_seq, load_seq=load_seq[agent_idx], pv_seq=pv_seq[agent_idx], soc=float(env.soc[agent_idx]), ev_soc=float(env.ev_soc[agent_idx]), start_step=int(env.cur_step))
            battery_power_kw[agent_idx] = np.float32(result.signed_battery_kw[0]); pv_curtail_kw[agent_idx] = np.float32(result.pv_curtail_kw[0]); solve_time_sec += float(result.solve_time_sec)
            ev_charge_kw[agent_idx] = np.float32(result.ev_charge_kw[0])
        actions, _, action_info = action_array_from_power(env, battery_power_kw, pv_curtail_kw, ev_charge_kw)
        action_info["solve_time_sec"] = np.asarray(solve_time_sec, dtype=np.float32)
        return actions, action_info, {"solve_time_sec": solve_time_sec}

    def close(self) -> None:
        for solver in self.cache.values():
            solver.dispose()
        self.cache.clear()
