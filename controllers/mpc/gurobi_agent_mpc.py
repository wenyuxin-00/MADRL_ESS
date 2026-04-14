"""Single-agent rolling MPC solved with Gurobi."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

_GUROBI_ERROR_PREFIX = "MPC rollout requires a working Gurobi installation/license"
GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH = 1e-6
GRID_DIRECTION_GUARD_TIME_LIMIT_SEC = 5.0
GRID_DIRECTION_GUARD_MIP_GAP = 1e-4


def _load_gurobi():
    import gurobipy as gp

    return gp, gp.GRB


def _create_model(gp: Any):
    model = gp.Model("single_agent_mpc")
    model.Params.OutputFlag = 0
    return model


def _optimize_model(model: Any) -> None:
    model.optimize()


def _wrap_gurobi_error(detail: str, exc: Exception) -> RuntimeError:
    error = RuntimeError(f"{_GUROBI_ERROR_PREFIX}: {detail}: {exc}")
    error.__cause__ = exc
    return error


def _requires_grid_direction_binary(prices: np.ndarray, export_subsidy_eur_per_kwh: float) -> bool:
    price_seq = np.asarray(prices, dtype=np.float32).reshape(-1)
    if price_seq.size == 0:
        return False
    export_subsidy = float(export_subsidy_eur_per_kwh)
    return bool(
        np.any(
            price_seq - export_subsidy + np.float32(GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH)
            <= np.float32(0.0)
        )
    )


@dataclass(frozen=True)
class _SingleAgentMPCPrimalSolution:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    charge_mode: np.ndarray
    energy_kwh: np.ndarray
    export_kw: np.ndarray | None = None
    grid_import_kw: np.ndarray | None = None
    grid_export_kw: np.ndarray | None = None
    grid_mode: np.ndarray | None = None


@dataclass(frozen=True)
class _SingleAgentMPCSolveResult:
    power_kw: float
    objective_eur: float
    solve_time_sec: float
    used_guarded_fallback: bool
    net_grid_kw: np.ndarray
    solution: _SingleAgentMPCPrimalSolution


def _sanitize_problem_inputs(
    *,
    price_seq: np.ndarray,
    load_seq: np.ndarray,
    pv_seq: np.ndarray,
    soc: float,
    battery_capacity_kwh: float,
    p_max_kw: float,
    dt_hours: float,
    efficiency: float,
    soc_min: float,
    soc_max: float,
) -> dict[str, Any] | None:
    capacity = float(battery_capacity_kwh)
    power_limit = float(p_max_kw)
    dt = float(dt_hours)
    if capacity <= 0.0 or power_limit <= 0.0 or dt <= 0.0:
        return None

    prices = np.asarray(price_seq, dtype=np.float32).reshape(-1)
    load = np.asarray(load_seq, dtype=np.float32).reshape(-1)
    pv = np.asarray(pv_seq, dtype=np.float32).reshape(-1)
    horizon = int(min(len(prices), len(load), len(pv)))
    if horizon <= 0:
        return None

    prices = prices[:horizon].astype(np.float32, copy=False)
    net_load = (load[:horizon] - pv[:horizon]).astype(np.float32, copy=False)
    eff = max(float(efficiency), 1e-6)
    energy_min = float(np.clip(soc_min, 0.0, 1.0)) * capacity
    energy_max = float(np.clip(soc_max, 0.0, 1.0)) * capacity
    if energy_max <= energy_min + 1e-9:
        return None
    energy_now = float(np.clip(soc, soc_min, soc_max)) * capacity
    return {
        "prices": prices,
        "net_load": net_load,
        "horizon": horizon,
        "capacity": capacity,
        "power_limit": power_limit,
        "dt": dt,
        "eff": eff,
        "energy_min": energy_min,
        "energy_max": energy_max,
        "energy_now": energy_now,
    }


def _shift_control_start(values: np.ndarray, fill_value: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return array.copy()
    shifted = np.empty_like(array)
    if array.size > 1:
        shifted[:-1] = array[1:]
    shifted[-1] = np.float32(fill_value)
    return shifted


def _shift_energy_start(values: np.ndarray, current_energy_kwh: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return array.copy()
    shifted = np.empty_like(array)
    shifted[0] = np.float32(current_energy_kwh)
    if array.size > 2:
        shifted[1:-1] = array[2:]
    if array.size > 1:
        shifted[-1] = array[-2]
    return shifted


def _shift_primal_solution_start(
    solution: _SingleAgentMPCPrimalSolution,
    *,
    current_energy_kwh: float,
    use_guarded_fallback: bool,
) -> _SingleAgentMPCPrimalSolution:
    shifted = _SingleAgentMPCPrimalSolution(
        charge_kw=_shift_control_start(solution.charge_kw, 0.0),
        discharge_kw=_shift_control_start(solution.discharge_kw, 0.0),
        charge_mode=_shift_control_start(solution.charge_mode, 0.0),
        energy_kwh=_shift_energy_start(solution.energy_kwh, current_energy_kwh),
        export_kw=(
            None
            if solution.export_kw is None
            else _shift_control_start(solution.export_kw, 0.0)
        ),
        grid_import_kw=(
            None
            if solution.grid_import_kw is None
            else _shift_control_start(solution.grid_import_kw, 0.0)
        ),
        grid_export_kw=(
            None
            if solution.grid_export_kw is None
            else _shift_control_start(solution.grid_export_kw, 0.0)
        ),
        grid_mode=(
            None
            if solution.grid_mode is None
            else _shift_control_start(
                solution.grid_mode,
                float(solution.grid_mode[-1]) if solution.grid_mode.size else 0.0,
            )
        ),
    )
    if use_guarded_fallback:
        return shifted
    return shifted


class _ReusableSingleAgentMPCSolver:
    """Reuse a fixed-horizon single-agent MPC model across rollout steps."""

    def __init__(
        self,
        *,
        horizon: int,
        battery_capacity_kwh: float,
        p_max_kw: float,
        dt_hours: float,
        efficiency: float,
        soc_min: float,
        soc_max: float,
        export_subsidy_eur_per_kwh: float,
        use_guarded_fallback: bool,
    ) -> None:
        self.horizon = int(horizon)
        self.capacity = float(battery_capacity_kwh)
        self.power_limit = float(p_max_kw)
        self.dt = float(dt_hours)
        self.eff = max(float(efficiency), 1e-6)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.energy_min = float(np.clip(self.soc_min, 0.0, 1.0)) * self.capacity
        self.energy_max = float(np.clip(self.soc_max, 0.0, 1.0)) * self.capacity
        self.export_subsidy_eur_per_kwh = float(export_subsidy_eur_per_kwh)
        self.use_guarded_fallback = bool(use_guarded_fallback)
        self._last_solution: _SingleAgentMPCPrimalSolution | None = None

        try:
            self._gp, self._grb = _load_gurobi()
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"{_GUROBI_ERROR_PREFIX}: gurobipy import failed") from exc

        try:
            self.model = _create_model(self._gp)
        except Exception as exc:  # pragma: no cover - exercised by monkeypatch tests
            raise _wrap_gurobi_error("unable to create Gurobi model", exc)

        if self.use_guarded_fallback:
            self.model.Params.TimeLimit = GRID_DIRECTION_GUARD_TIME_LIMIT_SEC
            self.model.Params.MIPGap = GRID_DIRECTION_GUARD_MIP_GAP

        self.charge = self.model.addVars(self.horizon, lb=0.0, ub=self.power_limit, name="p_charge")
        self.discharge = self.model.addVars(self.horizon, lb=0.0, ub=self.power_limit, name="p_discharge")
        self.charge_mode = self.model.addVars(self.horizon, vtype=self._grb.BINARY, name="charge_mode")
        self.energy = self.model.addVars(
            self.horizon + 1,
            lb=self.energy_min,
            ub=self.energy_max,
            name="energy",
        )

        self.export = None
        self.grid_import = None
        self.grid_export = None
        self.grid_mode = None
        self._fast_export_lower_bound_constrs: list[Any] = []
        self._fallback_grid_balance_constrs: list[Any] = []
        self._fallback_import_gate_constrs: list[Any] = []
        self._fallback_export_gate_constrs: list[Any] = []

        self._energy_init_constr = self.model.addConstr(self.energy[0] == 0.0, name="energy_init")
        for step_idx in range(self.horizon):
            self.model.addConstr(
                self.charge[step_idx] <= self.power_limit * self.charge_mode[step_idx],
                name=f"charge_limit_{step_idx}",
            )
            self.model.addConstr(
                self.discharge[step_idx] <= self.power_limit * (1.0 - self.charge_mode[step_idx]),
                name=f"discharge_limit_{step_idx}",
            )
            self.model.addConstr(
                self.energy[step_idx + 1]
                == self.energy[step_idx]
                + self.eff * self.charge[step_idx] * self.dt
                - (self.discharge[step_idx] / self.eff) * self.dt,
                name=f"energy_balance_{step_idx}",
            )

        if self.use_guarded_fallback:
            self.grid_import = self.model.addVars(self.horizon, lb=0.0, name="grid_import")
            self.grid_export = self.model.addVars(self.horizon, lb=0.0, name="grid_export")
            self.grid_mode = self.model.addVars(self.horizon, vtype=self._grb.BINARY, name="grid_mode")
            for step_idx in range(self.horizon):
                self._fallback_grid_balance_constrs.append(
                    self.model.addConstr(
                        self.grid_import[step_idx]
                        - self.grid_export[step_idx]
                        - self.charge[step_idx]
                        + self.discharge[step_idx]
                        == 0.0,
                        name=f"grid_balance_{step_idx}",
                    )
                )
                self._fallback_import_gate_constrs.append(
                    self.model.addConstr(
                        self.grid_import[step_idx] <= 0.0,
                        name=f"grid_import_gate_{step_idx}",
                    )
                )
                self._fallback_export_gate_constrs.append(
                    self.model.addConstr(
                        self.grid_export[step_idx] <= 0.0,
                        name=f"grid_export_gate_{step_idx}",
                    )
                )
        else:
            self.export = self.model.addVars(self.horizon, lb=0.0, name="grid_export")
            for step_idx in range(self.horizon):
                self._fast_export_lower_bound_constrs.append(
                    self.model.addConstr(
                        self.export[step_idx] + self.charge[step_idx] - self.discharge[step_idx] >= 0.0,
                        name=f"export_lb_{step_idx}",
                    )
                )

        self.model.ModelSense = self._grb.MINIMIZE
        self.model.update()

    def dispose(self) -> None:
        dispose = getattr(self.model, "dispose", None)
        if callable(dispose):
            dispose()

    def _apply_problem_data(
        self,
        *,
        prices: np.ndarray,
        net_load: np.ndarray,
        energy_now: float,
    ) -> None:
        self._energy_init_constr.RHS = float(energy_now)
        for step_idx in range(self.horizon):
            price_t = float(prices[step_idx])
            self.charge[step_idx].Obj = price_t * self.dt
            self.discharge[step_idx].Obj = -price_t * self.dt
            if self.use_guarded_fallback:
                gross_flow_limit_kw = abs(float(net_load[step_idx])) + self.power_limit
                self._fallback_grid_balance_constrs[step_idx].RHS = float(net_load[step_idx])
                self._fallback_import_gate_constrs[step_idx].RHS = gross_flow_limit_kw
                self._fallback_export_gate_constrs[step_idx].RHS = gross_flow_limit_kw
                self.grid_import[step_idx].Obj = (
                    price_t + GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH
                ) * self.dt
                self.grid_export[step_idx].Obj = (
                    -self.export_subsidy_eur_per_kwh + GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH
                ) * self.dt
                self.grid_mode[step_idx].Obj = 0.0
            else:
                self._fast_export_lower_bound_constrs[step_idx].RHS = -float(net_load[step_idx])
                self.export[step_idx].Obj = (
                    price_t
                    - self.export_subsidy_eur_per_kwh
                    + GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH
                ) * self.dt

    def _apply_warm_start(self, *, energy_now: float) -> None:
        if self._last_solution is None:
            return
        shifted = _shift_primal_solution_start(
            self._last_solution,
            current_energy_kwh=energy_now,
            use_guarded_fallback=self.use_guarded_fallback,
        )
        for step_idx in range(self.horizon):
            self.charge[step_idx].Start = float(shifted.charge_kw[step_idx])
            self.discharge[step_idx].Start = float(shifted.discharge_kw[step_idx])
            self.charge_mode[step_idx].Start = float(shifted.charge_mode[step_idx])
            if self.use_guarded_fallback:
                self.grid_import[step_idx].Start = float(shifted.grid_import_kw[step_idx])
                self.grid_export[step_idx].Start = float(shifted.grid_export_kw[step_idx])
                self.grid_mode[step_idx].Start = float(shifted.grid_mode[step_idx])
            else:
                self.export[step_idx].Start = float(shifted.export_kw[step_idx])
        for step_idx in range(self.horizon + 1):
            self.energy[step_idx].Start = float(shifted.energy_kwh[step_idx])

    def _status_has_usable_solution(self) -> bool:
        status = int(self.model.Status)
        if status == int(self._grb.OPTIMAL):
            return True
        if not self.use_guarded_fallback:
            return False
        sol_count = int(getattr(self.model, "SolCount", 0))
        return sol_count > 0 and status in {
            int(self._grb.TIME_LIMIT),
            int(self._grb.SUBOPTIMAL),
        }

    def _extract_solution(self) -> _SingleAgentMPCPrimalSolution:
        charge_kw = np.asarray([self.charge[idx].X for idx in range(self.horizon)], dtype=np.float32)
        discharge_kw = np.asarray([self.discharge[idx].X for idx in range(self.horizon)], dtype=np.float32)
        charge_mode = np.asarray([self.charge_mode[idx].X for idx in range(self.horizon)], dtype=np.float32)
        energy_kwh = np.asarray([self.energy[idx].X for idx in range(self.horizon + 1)], dtype=np.float32)
        if self.use_guarded_fallback:
            return _SingleAgentMPCPrimalSolution(
                charge_kw=charge_kw,
                discharge_kw=discharge_kw,
                charge_mode=charge_mode,
                energy_kwh=energy_kwh,
                grid_import_kw=np.asarray(
                    [self.grid_import[idx].X for idx in range(self.horizon)],
                    dtype=np.float32,
                ),
                grid_export_kw=np.asarray(
                    [self.grid_export[idx].X for idx in range(self.horizon)],
                    dtype=np.float32,
                ),
                grid_mode=np.asarray([self.grid_mode[idx].X for idx in range(self.horizon)], dtype=np.float32),
            )
        return _SingleAgentMPCPrimalSolution(
            charge_kw=charge_kw,
            discharge_kw=discharge_kw,
            charge_mode=charge_mode,
            energy_kwh=energy_kwh,
            export_kw=np.asarray([self.export[idx].X for idx in range(self.horizon)], dtype=np.float32),
        )

    def solve(
        self,
        *,
        price_seq: np.ndarray,
        load_seq: np.ndarray,
        pv_seq: np.ndarray,
        soc: float,
    ) -> _SingleAgentMPCSolveResult:
        prepared = _sanitize_problem_inputs(
            price_seq=price_seq,
            load_seq=load_seq,
            pv_seq=pv_seq,
            soc=soc,
            battery_capacity_kwh=self.capacity,
            p_max_kw=self.power_limit,
            dt_hours=self.dt,
            efficiency=self.eff,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
        )
        if prepared is None:
            zero_solution = _SingleAgentMPCPrimalSolution(
                charge_kw=np.zeros((max(self.horizon, 0),), dtype=np.float32),
                discharge_kw=np.zeros((max(self.horizon, 0),), dtype=np.float32),
                charge_mode=np.zeros((max(self.horizon, 0),), dtype=np.float32),
                energy_kwh=np.zeros((max(self.horizon + 1, 0),), dtype=np.float32),
            )
            return _SingleAgentMPCSolveResult(
                power_kw=0.0,
                objective_eur=0.0,
                solve_time_sec=0.0,
                used_guarded_fallback=self.use_guarded_fallback,
                net_grid_kw=np.zeros((max(self.horizon, 0),), dtype=np.float32),
                solution=zero_solution,
            )

        if int(prepared["horizon"]) != self.horizon:
            raise ValueError(
                "Reusable single-agent MPC solver was built with a different horizon: "
                f"expected {self.horizon}, got {prepared['horizon']}."
            )

        prices = prepared["prices"]
        net_load = prepared["net_load"]
        energy_now = float(prepared["energy_now"])
        self._apply_problem_data(prices=prices, net_load=net_load, energy_now=energy_now)
        self._apply_warm_start(energy_now=energy_now)

        started_at = perf_counter()
        try:
            _optimize_model(self.model)
        except Exception as exc:  # pragma: no cover - exercised by monkeypatch tests
            raise _wrap_gurobi_error("Gurobi optimize failed", exc)
        solve_time_sec = float(getattr(self.model, "Runtime", perf_counter() - started_at))

        if not self._status_has_usable_solution():
            raise RuntimeError(
                f"{_GUROBI_ERROR_PREFIX}: optimization did not reach a usable status "
                f"(status={int(self.model.Status)})"
            )

        solution = self._extract_solution()
        self._last_solution = solution
        net_grid_kw = (net_load + solution.charge_kw - solution.discharge_kw).astype(np.float32)
        grid_import_kw = np.maximum(net_grid_kw, 0.0).astype(np.float32)
        grid_export_kw = np.maximum(-net_grid_kw, 0.0).astype(np.float32)
        objective_eur = float(
            np.sum((prices * grid_import_kw - self.export_subsidy_eur_per_kwh * grid_export_kw) * self.dt)
        )
        power_kw = float(solution.charge_kw[0] - solution.discharge_kw[0])
        if abs(power_kw) < 1e-8:
            power_kw = 0.0
        return _SingleAgentMPCSolveResult(
            power_kw=power_kw,
            objective_eur=objective_eur,
            solve_time_sec=solve_time_sec,
            used_guarded_fallback=self.use_guarded_fallback,
            net_grid_kw=net_grid_kw,
            solution=solution,
        )


def _solve_single_agent_gurobi_mpc(
    *,
    price_seq: np.ndarray,
    load_seq: np.ndarray,
    pv_seq: np.ndarray,
    soc: float,
    battery_capacity_kwh: float,
    p_max_kw: float,
    dt_hours: float,
    efficiency: float,
    soc_min: float,
    soc_max: float,
    export_subsidy_eur_per_kwh: float,
    force_guarded_fallback: bool | None = None,
) -> _SingleAgentMPCSolveResult:
    prepared = _sanitize_problem_inputs(
        price_seq=price_seq,
        load_seq=load_seq,
        pv_seq=pv_seq,
        soc=soc,
        battery_capacity_kwh=battery_capacity_kwh,
        p_max_kw=p_max_kw,
        dt_hours=dt_hours,
        efficiency=efficiency,
        soc_min=soc_min,
        soc_max=soc_max,
    )
    if prepared is None:
        zero_solution = _SingleAgentMPCPrimalSolution(
            charge_kw=np.zeros((0,), dtype=np.float32),
            discharge_kw=np.zeros((0,), dtype=np.float32),
            charge_mode=np.zeros((0,), dtype=np.float32),
            energy_kwh=np.zeros((0,), dtype=np.float32),
        )
        return _SingleAgentMPCSolveResult(
            power_kw=0.0,
            objective_eur=0.0,
            solve_time_sec=0.0,
            used_guarded_fallback=bool(force_guarded_fallback),
            net_grid_kw=np.zeros((0,), dtype=np.float32),
            solution=zero_solution,
        )

    use_guarded_fallback = (
        _requires_grid_direction_binary(prepared["prices"], export_subsidy_eur_per_kwh)
        if force_guarded_fallback is None
        else bool(force_guarded_fallback)
    )
    solver = _ReusableSingleAgentMPCSolver(
        horizon=int(prepared["horizon"]),
        battery_capacity_kwh=float(prepared["capacity"]),
        p_max_kw=float(prepared["power_limit"]),
        dt_hours=float(prepared["dt"]),
        efficiency=float(prepared["eff"]),
        soc_min=float(soc_min),
        soc_max=float(soc_max),
        export_subsidy_eur_per_kwh=float(export_subsidy_eur_per_kwh),
        use_guarded_fallback=use_guarded_fallback,
    )
    try:
        return solver.solve(
            price_seq=prepared["prices"],
            load_seq=np.asarray(load_seq, dtype=np.float32)[: int(prepared["horizon"])],
            pv_seq=np.asarray(pv_seq, dtype=np.float32)[: int(prepared["horizon"])],
            soc=soc,
        )
    finally:
        solver.dispose()


def solve_single_agent_gurobi_mpc_action(
    *,
    price_seq: np.ndarray,
    load_seq: np.ndarray,
    pv_seq: np.ndarray,
    soc: float,
    battery_capacity_kwh: float,
    p_max_kw: float,
    dt_hours: float,
    efficiency: float,
    soc_min: float,
    soc_max: float,
    export_subsidy_eur_per_kwh: float,
) -> float:
    """Return the first-step battery power for one agent in kW.

    Positive power means charging, negative power means discharging.
    """

    result = _solve_single_agent_gurobi_mpc(
        price_seq=price_seq,
        load_seq=load_seq,
        pv_seq=pv_seq,
        soc=soc,
        battery_capacity_kwh=battery_capacity_kwh,
        p_max_kw=p_max_kw,
        dt_hours=dt_hours,
        efficiency=efficiency,
        soc_min=soc_min,
        soc_max=soc_max,
        export_subsidy_eur_per_kwh=export_subsidy_eur_per_kwh,
    )
    return float(result.power_kw)
