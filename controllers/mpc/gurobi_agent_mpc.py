"""Single-agent rolling MPC solved with Gurobi."""

from __future__ import annotations

from typing import Any

import numpy as np

_GUROBI_ERROR_PREFIX = "MPC rollout requires a working Gurobi installation/license"


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
) -> float:
    """Return the first-step battery power for one agent in kW.

    Positive power means charging, negative power means discharging.
    """

    capacity = float(battery_capacity_kwh)
    power_limit = float(p_max_kw)
    dt = float(dt_hours)
    if capacity <= 0.0 or power_limit <= 0.0 or dt <= 0.0:
        return 0.0

    prices = np.asarray(price_seq, dtype=np.float32).reshape(-1)
    load = np.asarray(load_seq, dtype=np.float32).reshape(-1)
    pv = np.asarray(pv_seq, dtype=np.float32).reshape(-1)
    horizon = int(min(len(prices), len(load), len(pv)))
    if horizon <= 0:
        return 0.0

    prices = prices[:horizon]
    net_load = (load[:horizon] - pv[:horizon]).astype(np.float32)

    eff = max(float(efficiency), 1e-6)
    energy_min = float(np.clip(soc_min, 0.0, 1.0)) * capacity
    energy_max = float(np.clip(soc_max, 0.0, 1.0)) * capacity
    if energy_max <= energy_min + 1e-9:
        return 0.0
    energy_now = float(np.clip(soc, soc_min, soc_max)) * capacity

    try:
        gp, grb = _load_gurobi()
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{_GUROBI_ERROR_PREFIX}: gurobipy import failed") from exc

    try:
        model = _create_model(gp)
    except Exception as exc:  # pragma: no cover - exercised by monkeypatch tests
        raise _wrap_gurobi_error("unable to create Gurobi model", exc)

    try:
        charge = model.addVars(horizon, lb=0.0, ub=power_limit, name="p_charge")
        discharge = model.addVars(horizon, lb=0.0, ub=power_limit, name="p_discharge")
        charge_mode = model.addVars(horizon, vtype=grb.BINARY, name="charge_mode")
        energy = model.addVars(horizon + 1, lb=energy_min, ub=energy_max, name="energy")

        model.addConstr(energy[0] == energy_now, name="energy_init")
        for step_idx in range(horizon):
            model.addConstr(
                charge[step_idx] <= power_limit * charge_mode[step_idx],
                name=f"charge_limit_{step_idx}",
            )
            model.addConstr(
                discharge[step_idx] <= power_limit * (1.0 - charge_mode[step_idx]),
                name=f"discharge_limit_{step_idx}",
            )
            model.addConstr(
                energy[step_idx + 1]
                == energy[step_idx]
                + eff * charge[step_idx] * dt
                - (discharge[step_idx] / eff) * dt,
                name=f"energy_balance_{step_idx}",
            )

        model.setObjective(
            gp.quicksum(
                (float(net_load[step_idx]) + charge[step_idx] - discharge[step_idx])
                * dt
                * float(prices[step_idx])
                for step_idx in range(horizon)
            ),
            grb.MINIMIZE,
        )

        try:
            _optimize_model(model)
        except Exception as exc:  # pragma: no cover - exercised by monkeypatch tests
            raise _wrap_gurobi_error("Gurobi optimize failed", exc)

        if int(model.Status) != int(grb.OPTIMAL):
            raise RuntimeError(
                f"{_GUROBI_ERROR_PREFIX}: optimization did not reach OPTIMAL status "
                f"(status={int(model.Status)})"
            )

        power_kw = float(charge[0].X - discharge[0].X)
        if abs(power_kw) < 1e-8:
            return 0.0
        return power_kw
    finally:
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()


