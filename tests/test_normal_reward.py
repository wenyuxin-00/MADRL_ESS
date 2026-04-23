"""Unit tests for NormalReward."""

from __future__ import annotations

import numpy as np
import pytest

from envs.rewards.NormalReward import NormalReward


def _make_cfg():
    class _Reward:
        w_soc_pen = 0.0
        export_subsidy_eur_per_kwh = 0.079
        import_price_markup_eur_per_kwh = 0.20
        storage_objective_mode = "max_storage_profit"
        storage_price_mode = "real_time_price"
        storage_profit_weight = 1.0
        w_voltage_pen = 10.0
        w_line_pen = 3.0
        w_trafo_pen = 7.0

    class _Env:
        max_charge_rate = 0.5

    class _Cfg:
        reward = _Reward()
        env = _Env()

    return _Cfg()


def _make_env_state(
    n_agents: int = 3,
    *,
    storage_price_t: float = 0.15,
    battery_power_t=None,
    dt: float = 0.25,
    v_violation=None,
    psi_v_raw: float = 0.0,
    psi_line_raw: float = 0.0,
    psi_trafo_raw: float = 0.0,
):
    if battery_power_t is None:
        battery_power_t = np.zeros(n_agents, dtype=np.float32)
    if v_violation is None:
        v_violation = np.zeros(n_agents, dtype=np.float32)

    return {
        "storage_price_t": float(storage_price_t),
        "battery_power_t": np.asarray(battery_power_t, dtype=np.float32),
        "dt": float(dt),
        "v_violation": np.asarray(v_violation, dtype=np.float32),
        "psi_v_raw": float(psi_v_raw),
        "psi_line_raw": float(psi_line_raw),
        "psi_trafo_raw": float(psi_trafo_raw),
    }


def test_component_meta_has_expected_keys() -> None:
    rf = NormalReward(_make_cfg())
    meta_keys = [meta.key for meta in rf.component_meta]
    assert meta_keys == [
        "r_storage_discharge_revenue",
        "r_storage_charge_cost",
        "r_storage_profit",
        "r_soc_pen",
        "r_safe_v",
        "r_safe_line",
        "r_safe_trafo",
    ]


def test_storage_profit_follows_battery_power_direction() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(battery_power_t=np.array([2.0, -1.0, 0.0], dtype=np.float32))

    total, components = rf.compute(env_state)

    expected_charge_cost = np.array([0.075, 0.0, 0.0], dtype=np.float32)
    expected_revenue = np.array([0.0, 0.0375, 0.0], dtype=np.float32)
    expected_profit = expected_revenue - expected_charge_cost
    np.testing.assert_allclose(components["r_storage_charge_cost"], expected_charge_cost, rtol=1e-6)
    np.testing.assert_allclose(components["r_storage_discharge_revenue"], expected_revenue, rtol=1e-6)
    np.testing.assert_allclose(components["r_storage_profit"], expected_profit, rtol=1e-6)
    np.testing.assert_allclose(total, components["r_storage_profit"], rtol=1e-6)


def test_negative_price_makes_charging_profitable() -> None:
    rf = NormalReward(_make_cfg())
    total, components = rf.compute(
        _make_env_state(storage_price_t=-0.3, battery_power_t=np.array([2.0, 0.0, -1.0], dtype=np.float32))
    )

    assert components["r_storage_charge_cost"][0] == pytest.approx(-0.15)
    assert components["r_storage_discharge_revenue"][2] == pytest.approx(-0.075)
    np.testing.assert_allclose(total, components["r_storage_profit"], rtol=1e-6)


def test_voltage_penalty_splits_by_local_violation_proportion() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        v_violation=np.array([0.02, 0.0, 0.01], dtype=np.float32),
        psi_v_raw=0.005,
    )

    _, components = rf.compute(env_state)

    voltage_total = 10.0 * 0.005
    expected = np.array([0.1, 0.0, 0.05], dtype=np.float32)
    np.testing.assert_allclose(components["r_safe_v"], expected, rtol=1e-5)
    assert np.isclose(np.mean(components["r_safe_v"]), voltage_total)


def test_voltage_penalty_falls_back_to_uniform_split_for_non_agent_violations() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        v_violation=np.zeros(3, dtype=np.float32),
        psi_v_raw=0.005,
    )

    _, components = rf.compute(env_state)

    expected = np.full(3, 10.0 * 0.005, dtype=np.float32)
    np.testing.assert_allclose(components["r_safe_v"], expected, rtol=1e-5)


def test_transformer_penalty_is_shared() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(psi_trafo_raw=0.02)

    _, components = rf.compute(env_state)

    expected = np.full(3, 7.0 * 0.02, dtype=np.float32)
    np.testing.assert_allclose(components["r_safe_trafo"], expected, rtol=1e-5)


def test_line_penalty_is_shared() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(psi_line_raw=0.5)

    _, components = rf.compute(env_state)

    expected = np.full(3, 3.0 * 0.5, dtype=np.float32)
    np.testing.assert_allclose(components["r_safe_line"], expected, rtol=1e-5)


def test_objective_combines_storage_profit_and_penalties() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        battery_power_t=np.array([2.0, -1.0, 0.5], dtype=np.float32),
        v_violation=np.array([0.01, 0.0, 0.0], dtype=np.float32),
        psi_v_raw=0.002,
        psi_line_raw=0.1,
        psi_trafo_raw=0.05,
    )

    total, components = rf.compute(env_state)

    expected = (
        components["r_storage_profit"]
        - components["r_safe_v"]
        - components["r_safe_line"]
        - components["r_safe_trafo"]
    )
    np.testing.assert_allclose(total, expected, rtol=1e-6)


def test_old_grid_power_contract_is_rejected() -> None:
    rf = NormalReward(_make_cfg())
    with pytest.raises(KeyError, match="battery_power_t"):
        rf.compute({"storage_price_t": 0.2, "actual_grid_power_t": np.zeros(3, dtype=np.float32)})
