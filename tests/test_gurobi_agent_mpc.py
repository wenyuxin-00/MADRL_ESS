from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from controllers.mpc import gurobi_agent_mpc

HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


def _has_working_gurobi_license() -> bool:
    if not HAS_GUROBI:
        return False
    try:
        gp, _ = gurobi_agent_mpc._load_gurobi()
        model = gurobi_agent_mpc._create_model(gp)
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
        return True
    except Exception:
        return False


HAS_WORKING_GUROBI_LICENSE = _has_working_gurobi_license()


def test_single_agent_gurobi_solver_raises_clear_error_when_dependency_missing(monkeypatch):
    def _raise_missing():
        raise ModuleNotFoundError("No module named 'gurobipy'")

    monkeypatch.setattr(gurobi_agent_mpc, "_load_gurobi", _raise_missing)

    with pytest.raises(RuntimeError, match="MPC rollout requires a working Gurobi installation/license"):
        gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
            price_seq=np.array([0.2], dtype=np.float32),
            load_seq=np.array([1.0], dtype=np.float32),
            pv_seq=np.array([0.0], dtype=np.float32),
            soc=0.5,
            battery_capacity_kwh=4.0,
            p_max_kw=2.0,
            dt_hours=1.0,
            efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9,
        )


@pytest.mark.skipif(not HAS_GUROBI, reason="gurobipy is required for the model creation failure test")
def test_single_agent_gurobi_solver_raises_clear_error_when_model_creation_fails(monkeypatch):
    gp, _ = gurobi_agent_mpc._load_gurobi()

    def _raise_model(_gp):
        raise gp.GurobiError(10009, "license failure")

    monkeypatch.setattr(gurobi_agent_mpc, "_create_model", _raise_model)

    with pytest.raises(RuntimeError, match="working Gurobi installation/license"):
        gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
            price_seq=np.array([0.2], dtype=np.float32),
            load_seq=np.array([1.0], dtype=np.float32),
            pv_seq=np.array([0.0], dtype=np.float32),
            soc=0.5,
            battery_capacity_kwh=4.0,
            p_max_kw=2.0,
            dt_hours=1.0,
            efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9,
        )


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_single_agent_gurobi_solver_prefers_discharge_for_positive_prices():
    power = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([0.3, 0.3], dtype=np.float32),
        load_seq=np.array([1.0, 1.0], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        soc=0.8,
        battery_capacity_kwh=4.0,
        p_max_kw=2.0,
        dt_hours=1.0,
        efficiency=1.0,
        soc_min=0.1,
        soc_max=0.9,
    )

    assert power < 0.0
    assert abs(power) <= 2.0 + 1e-6


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_single_agent_gurobi_solver_prefers_charge_for_negative_prices():
    power = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([-0.2, -0.2], dtype=np.float32),
        load_seq=np.array([1.0, 1.0], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        soc=0.5,
        battery_capacity_kwh=4.0,
        p_max_kw=2.0,
        dt_hours=1.0,
        efficiency=1.0,
        soc_min=0.1,
        soc_max=0.9,
    )

    assert power > 0.0
    assert power <= 2.0 + 1e-6


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_single_agent_gurobi_solver_respects_soc_bounds():
    power_at_max_soc = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([-0.2, -0.2], dtype=np.float32),
        load_seq=np.array([1.0, 1.0], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        soc=0.9,
        battery_capacity_kwh=4.0,
        p_max_kw=2.0,
        dt_hours=1.0,
        efficiency=1.0,
        soc_min=0.1,
        soc_max=0.9,
    )
    power_at_min_soc = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([0.2, 0.2], dtype=np.float32),
        load_seq=np.array([1.0, 1.0], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        soc=0.1,
        battery_capacity_kwh=4.0,
        p_max_kw=2.0,
        dt_hours=1.0,
        efficiency=1.0,
        soc_min=0.1,
        soc_max=0.9,
    )

    assert power_at_max_soc == pytest.approx(0.0, abs=1e-6)
    assert power_at_min_soc == pytest.approx(0.0, abs=1e-6)
