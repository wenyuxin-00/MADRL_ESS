from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

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
    energy_kwh: np.ndarray


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

    return _LocalMPCPrimalSolution(shift(solution.charge_kw), shift(solution.discharge_kw), shift(solution.pv_curtail_kw), energy)


class ReusableLocalMPCSolver:
    def __init__(self, *, horizon: int, battery_capacity_kwh: float, p_max_kw: float, dt_hours: float, efficiency: float, soc_min: float, soc_max: float) -> None:
        import gurobipy as gp

        self.gp, self.grb = gp, gp.GRB
        self.horizon = int(horizon)
        self.capacity = float(battery_capacity_kwh)
        self.power_limit = float(p_max_kw)
        self.dt = float(dt_hours)
        self.eff = float(efficiency)
        self.energy_min = float(soc_min) * self.capacity
        self.energy_max = float(soc_max) * self.capacity
        self.last_solution: _LocalMPCPrimalSolution | None = None

        self.model = gp.Model("remake_local_mpc")
        self.model.Params.OutputFlag = 0
        self.model.Params.TimeLimit = LOCAL_MPC_TIME_LIMIT_SEC
        self.model.Params.MIPGap = LOCAL_MPC_MIP_GAP
        self.charge = self.model.addVars(self.horizon, lb=0.0, ub=self.power_limit, name="p_charge")
        self.discharge = self.model.addVars(self.horizon, lb=0.0, ub=self.power_limit, name="p_discharge")
        self.pv_curtail = self.model.addVars(self.horizon, lb=0.0, name="pv_curtail")
        self.energy = self.model.addVars(self.horizon + 1, lb=self.energy_min, ub=self.energy_max, name="energy")
        self.grid_import = self.model.addVars(self.horizon, lb=0.0, name="grid_import")
        self.grid_export = self.model.addVars(self.horizon, lb=0.0, name="grid_export")
        self.grid_mode = self.model.addVars(self.horizon, vtype=self.grb.BINARY, name="grid_mode")
        self.energy_init_constr = self.model.addConstr(self.energy[0] == 0.0, name="energy_init")
        self.net_load_floor_constrs: list[Any] = []
        self.pv_curtail_upper_constrs: list[Any] = []
        self.grid_balance_constrs: list[Any] = []
        self.grid_import_ub_constrs: list[Any] = []
        self.grid_export_ub_constrs: list[Any] = []
        self.grid_import_gate_constrs: list[Any] = []
        self.grid_export_gate_constrs: list[Any] = []
        for step_idx in range(self.horizon):
            self.model.addConstr(self.energy[step_idx + 1] == self.energy[step_idx] + self.eff * self.charge[step_idx] * self.dt - self.discharge[step_idx] / self.eff * self.dt, name=f"energy_balance_{step_idx}")
            self.net_load_floor_constrs.append(self.model.addConstr(self.charge[step_idx] - self.discharge[step_idx] + self.pv_curtail[step_idx] >= -2.0 * self.power_limit, name=f"net_load_floor_{step_idx}"))
            self.pv_curtail_upper_constrs.append(self.model.addConstr(self.pv_curtail[step_idx] <= 0.0, name=f"pv_curtail_ub_{step_idx}"))
            self.grid_balance_constrs.append(self.model.addConstr(self.grid_import[step_idx] - self.grid_export[step_idx] - self.charge[step_idx] + self.discharge[step_idx] - self.pv_curtail[step_idx] == 0.0, name=f"grid_balance_{step_idx}"))
            self.grid_import_ub_constrs.append(self.model.addConstr(self.grid_import[step_idx] <= 0.0, name=f"grid_import_ub_{step_idx}"))
            self.grid_export_ub_constrs.append(self.model.addConstr(self.grid_export[step_idx] <= 0.0, name=f"grid_export_ub_{step_idx}"))
            self.grid_import_gate_constrs.append(self.model.addConstr(self.grid_import[step_idx] - self.grid_mode[step_idx] <= 0.0, name=f"grid_import_gate_{step_idx}"))
            self.grid_export_gate_constrs.append(self.model.addConstr(self.grid_export[step_idx] + self.grid_mode[step_idx] <= 1.0, name=f"grid_export_gate_{step_idx}"))
        self.model.ModelSense = self.grb.MINIMIZE
        self.model.update()

    def dispose(self) -> None:
        self.model.dispose()

    def _apply_problem_data(self, *, prices: np.ndarray, net_load: np.ndarray, pv_available_kw: np.ndarray, energy_now: float) -> None:
        self.energy_init_constr.RHS = float(energy_now)
        throughput_penalty = BATTERY_THROUGHPUT_TIEBREAKER_EPS_EUR_PER_KWH * self.dt
        curtail_upper = np.maximum(np.asarray(pv_available_kw, dtype=np.float32).reshape(-1)[: self.horizon], 0.0)
        for step_idx in range(self.horizon):
            gross_flow_limit_kw = abs(float(net_load[step_idx])) + self.power_limit + float(curtail_upper[step_idx])
            self.net_load_floor_constrs[step_idx].RHS = -2.0 * self.power_limit
            self.pv_curtail_upper_constrs[step_idx].RHS = float(curtail_upper[step_idx])
            self.grid_balance_constrs[step_idx].RHS = float(net_load[step_idx])
            self.grid_import_ub_constrs[step_idx].RHS = gross_flow_limit_kw
            self.grid_export_ub_constrs[step_idx].RHS = gross_flow_limit_kw
            self.model.chgCoeff(self.grid_import_gate_constrs[step_idx], self.grid_mode[step_idx], -gross_flow_limit_kw)
            self.grid_import_gate_constrs[step_idx].RHS = 0.0
            self.model.chgCoeff(self.grid_export_gate_constrs[step_idx], self.grid_mode[step_idx], gross_flow_limit_kw)
            self.grid_export_gate_constrs[step_idx].RHS = gross_flow_limit_kw
            self.charge[step_idx].Obj = float(prices[step_idx]) * self.dt + throughput_penalty
            self.discharge[step_idx].Obj = -float(prices[step_idx]) * self.dt + throughput_penalty

    def _apply_warm_start(self, energy_now: float) -> None:
        if self.last_solution is None:
            return
        shifted = _shift_primal_solution(self.last_solution, energy_now)
        for step_idx in range(self.horizon):
            self.charge[step_idx].Start = float(shifted.charge_kw[step_idx])
            self.discharge[step_idx].Start = float(shifted.discharge_kw[step_idx])
            self.pv_curtail[step_idx].Start = float(shifted.pv_curtail_kw[step_idx])
        for step_idx in range(self.horizon + 1):
            self.energy[step_idx].Start = float(shifted.energy_kwh[step_idx])

    def _extract_solution(self) -> _LocalMPCPrimalSolution:
        return _LocalMPCPrimalSolution(
            charge_kw=np.asarray([self.charge[idx].X for idx in range(self.horizon)], dtype=np.float32),
            discharge_kw=np.asarray([self.discharge[idx].X for idx in range(self.horizon)], dtype=np.float32),
            pv_curtail_kw=np.asarray([self.pv_curtail[idx].X for idx in range(self.horizon)], dtype=np.float32),
            energy_kwh=np.asarray([self.energy[idx].X for idx in range(self.horizon + 1)], dtype=np.float32),
        )

    def solve_full_horizon(self, *, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc: float) -> LocalMPCFullHorizonResult:
        prices = np.asarray(import_price_seq, dtype=np.float32).reshape(-1)[: self.horizon]
        load = np.asarray(load_seq, dtype=np.float32).reshape(-1)[: self.horizon]
        pv = np.asarray(pv_seq, dtype=np.float32).reshape(-1)[: self.horizon]
        energy_now = float(soc) * self.capacity
        net_load = (load - pv).astype(np.float32)
        pv_available_kw = np.maximum(pv, 0.0).astype(np.float32)
        self._apply_problem_data(prices=prices, net_load=net_load, pv_available_kw=pv_available_kw, energy_now=energy_now)
        self._apply_warm_start(energy_now)
        started_at = perf_counter()
        self.model.optimize()
        solve_time_sec = float(getattr(self.model, "Runtime", perf_counter() - started_at))
        status = int(self.model.Status)
        if status not in {int(self.grb.OPTIMAL), int(self.grb.SUBOPTIMAL), int(self.grb.TIME_LIMIT)} or int(getattr(self.model, "SolCount", 0)) <= 0:
            raise RuntimeError(f"Local MPC Gurobi solve ended with status={status}.")
        solution = self._extract_solution()
        self.last_solution = solution
        pv_curtail_kw = np.minimum(np.maximum(solution.pv_curtail_kw, 0.0), pv_available_kw).astype(np.float32)
        pv_effective_kw = np.maximum(pv_available_kw - pv_curtail_kw, 0.0).astype(np.float32)
        pv_utilization = np.ones_like(pv_available_kw, dtype=np.float32)
        valid_pv_mask = pv_available_kw > 1e-6
        pv_utilization[valid_pv_mask] = np.clip(pv_effective_kw[valid_pv_mask] / pv_available_kw[valid_pv_mask], 0.0, 1.0).astype(np.float32)
        signed_battery_kw = (solution.charge_kw - solution.discharge_kw).astype(np.float32)
        net_grid_kw = (net_load + pv_curtail_kw + signed_battery_kw).astype(np.float32)
        return LocalMPCFullHorizonResult(
            charge_kw=solution.charge_kw.copy(),
            discharge_kw=solution.discharge_kw.copy(),
            pv_curtail_kw=pv_curtail_kw.copy(),
            pv_effective_kw=pv_effective_kw.copy(),
            pv_utilization=pv_utilization.copy(),
            signed_battery_kw=signed_battery_kw.copy(),
            net_load_kw=net_grid_kw.copy(),
            energy_kwh=solution.energy_kwh.copy(),
            objective_eur=float(np.sum((prices * solution.charge_kw - prices * solution.discharge_kw) * self.dt)),
            solve_time_sec=solve_time_sec,
            grid_import_kw=np.maximum(net_grid_kw, 0.0).astype(np.float32),
            grid_export_kw=np.maximum(-net_grid_kw, 0.0).astype(np.float32),
        )
