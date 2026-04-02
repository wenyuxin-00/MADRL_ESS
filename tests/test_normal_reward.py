"""Unit tests for NormalReward."""

from __future__ import annotations

import numpy as np

from envs.rewards import NormalReward


def _make_cfg():
    class _Reward:
        w_soc_pen = 0.0
        export_subsidy_eur_per_kwh = 0.079
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
    price_t: float = 0.15,
    actual_grid_power_t=None,
    dt: float = 0.25,
    v_violation=None,
    psi_v_raw: float = 0.0,
    psi_line_raw: float = 0.0,
    psi_trafo_raw: float = 0.0,
):
    if actual_grid_power_t is None:
        actual_grid_power_t = np.zeros(n_agents, dtype=np.float32)
    if v_violation is None:
        v_violation = np.zeros(n_agents, dtype=np.float32)

    return {
        "price_t": float(price_t),
        "actual_grid_power_t": np.asarray(actual_grid_power_t, dtype=np.float32),
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
        "r_purchase_cost",
        "r_export_subsidy",
        "r_soc_pen",
        "r_safe_v",
        "r_safe_line",
        "r_safe_trafo",
    ]


def test_purchase_cost_and_export_subsidy_follow_grid_power_direction() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(actual_grid_power_t=np.array([2.0, -1.0, 0.0], dtype=np.float32))

    total, components = rf.compute(env_state)

    expected_purchase = np.array([0.075, 0.0, 0.0], dtype=np.float32)
    expected_subsidy = np.array([0.0, 0.01975, 0.0], dtype=np.float32)
    np.testing.assert_allclose(components["r_purchase_cost"], expected_purchase, rtol=1e-6)
    np.testing.assert_allclose(components["r_export_subsidy"], expected_subsidy, rtol=1e-6)
    np.testing.assert_allclose(
        total,
        -components["r_purchase_cost"] + components["r_export_subsidy"],
        rtol=1e-6,
    )


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


def test_objective_combines_cost_subsidy_and_penalties() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        actual_grid_power_t=np.array([2.0, -1.0, 0.5], dtype=np.float32),
        v_violation=np.array([0.01, 0.0, 0.0], dtype=np.float32),
        psi_v_raw=0.002,
        psi_line_raw=0.1,
        psi_trafo_raw=0.05,
    )

    total, components = rf.compute(env_state)

    expected = (
        -components["r_purchase_cost"]
        + components["r_export_subsidy"]
        - components["r_safe_v"]
        - components["r_safe_line"]
        - components["r_safe_trafo"]
    )
    np.testing.assert_allclose(total, expected, rtol=1e-6)
