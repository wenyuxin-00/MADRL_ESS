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


def _base_solver_kwargs() -> dict[str, float]:
    return {
        "soc": 0.5,
        "battery_capacity_kwh": 4.0,
        "p_max_kw": 2.0,
        "dt_hours": 1.0,
        "efficiency": 1.0,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "export_subsidy_eur_per_kwh": 0.079,
    }


def test_single_agent_gurobi_solver_raises_clear_error_when_dependency_missing(monkeypatch):
    def _raise_missing():
        raise ModuleNotFoundError("No module named 'gurobipy'")

    monkeypatch.setattr(gurobi_agent_mpc, "_load_gurobi", _raise_missing)

    with pytest.raises(RuntimeError, match="MPC rollout requires a working Gurobi installation/license"):
        gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
            price_seq=np.array([0.2], dtype=np.float32),
            load_seq=np.array([1.0], dtype=np.float32),
            pv_seq=np.array([0.0], dtype=np.float32),
            **{**_base_solver_kwargs(), "efficiency": 0.95},
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
            **{**_base_solver_kwargs(), "efficiency": 0.95},
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
        **{**_base_solver_kwargs(), "soc": 0.8},
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
        **_base_solver_kwargs(),
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
        **{**_base_solver_kwargs(), "soc": 0.9},
    )
    power_at_min_soc = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([0.2, 0.2], dtype=np.float32),
        load_seq=np.array([1.0, 1.0], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.1},
    )

    assert power_at_max_soc == pytest.approx(0.0, abs=1e-6)
    assert power_at_min_soc == pytest.approx(0.0, abs=1e-6)


def test_requires_grid_direction_binary_only_when_price_below_subsidy():
    assert gurobi_agent_mpc._requires_grid_direction_binary(
        np.array([0.11, 0.20], dtype=np.float32),
        0.079,
    ) is False
    assert gurobi_agent_mpc._requires_grid_direction_binary(
        np.array([0.079], dtype=np.float32),
        0.079,
    ) is False
    assert gurobi_agent_mpc._requires_grid_direction_binary(
        np.array([0.05, 0.20], dtype=np.float32),
        0.079,
    ) is True


def test_shift_primal_solution_start_shifts_controls_and_reuses_penultimate_terminal_energy():
    previous = gurobi_agent_mpc._SingleAgentMPCPrimalSolution(
        charge_kw=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        discharge_kw=np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        charge_mode=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        energy_kwh=np.asarray([7.0, 8.0, 9.0, 10.0], dtype=np.float32),
        export_kw=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )

    shifted = gurobi_agent_mpc._shift_primal_solution_start(
        previous,
        current_energy_kwh=6.5,
        use_guarded_fallback=False,
    )

    assert np.allclose(shifted.charge_kw, np.asarray([2.0, 3.0, 0.0], dtype=np.float32))
    assert np.allclose(shifted.discharge_kw, np.asarray([5.0, 6.0, 0.0], dtype=np.float32))
    assert np.allclose(shifted.charge_mode, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(shifted.export_kw, np.asarray([0.2, 0.3, 0.0], dtype=np.float32))
    assert shifted.energy_kwh[0] == pytest.approx(6.5)
    assert shifted.energy_kwh[-1] == pytest.approx(previous.energy_kwh[-2])


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_single_agent_gurobi_solver_exports_when_subsidy_makes_last_step_valuable():
    power = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([0.0], dtype=np.float32),
        load_seq=np.array([0.0], dtype=np.float32),
        pv_seq=np.array([0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.8, "export_subsidy_eur_per_kwh": 0.079},
    )

    assert power < 0.0


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_single_agent_gurobi_solver_does_not_export_without_subsidy_or_load():
    power = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([0.0], dtype=np.float32),
        load_seq=np.array([0.0], dtype=np.float32),
        pv_seq=np.array([0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.8, "export_subsidy_eur_per_kwh": 0.0},
    )

    assert power == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_single_agent_gurobi_solver_handles_price_below_subsidy_guard_path():
    power = gurobi_agent_mpc.solve_single_agent_gurobi_mpc_action(
        price_seq=np.array([0.05, 0.05], dtype=np.float32),
        load_seq=np.array([0.0, 0.0], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.6, "export_subsidy_eur_per_kwh": 0.079},
    )

    assert np.isfinite(power)
    assert abs(power) <= 2.0 + 1e-6


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_fast_path_export_variable_matches_negative_grid_flow():
    result = gurobi_agent_mpc._solve_single_agent_gurobi_mpc(
        price_seq=np.array([0.20], dtype=np.float32),
        load_seq=np.array([0.0], dtype=np.float32),
        pv_seq=np.array([0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.8, "force_guarded_fallback": False},
    )

    assert result.used_guarded_fallback is False
    assert result.net_grid_kw[0] < 0.0
    assert result.solution.export_kw[0] == pytest.approx(abs(result.net_grid_kw[0]), abs=1e-6)


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_fast_path_export_variable_is_zero_for_import_case():
    result = gurobi_agent_mpc._solve_single_agent_gurobi_mpc(
        price_seq=np.array([0.20], dtype=np.float32),
        load_seq=np.array([1.0], dtype=np.float32),
        pv_seq=np.array([0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.1, "force_guarded_fallback": False},
    )

    assert result.net_grid_kw[0] > 0.0
    assert result.solution.export_kw[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(
    not HAS_WORKING_GUROBI_LICENSE,
    reason="a working Gurobi license is required for the real MPC solver behavior tests",
)
def test_fast_path_and_guarded_fallback_match_economic_objective_in_overlap_domain():
    fast_result = gurobi_agent_mpc._solve_single_agent_gurobi_mpc(
        price_seq=np.array([0.20, 0.25], dtype=np.float32),
        load_seq=np.array([0.3, 0.4], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.7, "force_guarded_fallback": False},
    )
    fallback_result = gurobi_agent_mpc._solve_single_agent_gurobi_mpc(
        price_seq=np.array([0.20, 0.25], dtype=np.float32),
        load_seq=np.array([0.3, 0.4], dtype=np.float32),
        pv_seq=np.array([0.0, 0.0], dtype=np.float32),
        **{**_base_solver_kwargs(), "soc": 0.7, "force_guarded_fallback": True},
    )

    assert fast_result.objective_eur == pytest.approx(fallback_result.objective_eur, abs=1e-4)
