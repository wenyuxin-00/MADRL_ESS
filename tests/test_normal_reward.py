"""Unit tests for NormalReward."""

from __future__ import annotations

import numpy as np

from envs.rewards import NormalReward


def _make_cfg():
    class _Reward:
        w_action_pen = 5.0
        lambda_throughput = 0.01
        w_voltage_pen = 10.0
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
    e_bat_req=None,
    e_bat=None,
    price_t: float = 0.15,
    net_load_t=None,
    p_max=None,
    dt: float = 0.25,
    v_violation=None,
    psi_v_raw: float = 0.0,
    psi_line_raw: float = 0.0,
    psi_trafo_raw: float = 0.0,
):
    if e_bat_req is None:
        e_bat_req = np.zeros(n_agents, dtype=np.float32)
    if e_bat is None:
        e_bat = np.zeros(n_agents, dtype=np.float32)
    if net_load_t is None:
        net_load_t = np.ones(n_agents, dtype=np.float32)
    if p_max is None:
        p_max = np.full(n_agents, 0.5, dtype=np.float32)
    if v_violation is None:
        v_violation = np.zeros(n_agents, dtype=np.float32)

    return {
        "e_bat_req": np.asarray(e_bat_req, dtype=np.float32),
        "e_bat": np.asarray(e_bat, dtype=np.float32),
        "price_t": float(price_t),
        "net_load_t": np.asarray(net_load_t, dtype=np.float32),
        "p_max": np.asarray(p_max, dtype=np.float32),
        "dt": float(dt),
        "v_violation": np.asarray(v_violation, dtype=np.float32),
        "psi_v_raw": float(psi_v_raw),
        "psi_line_raw": float(psi_line_raw),
        "psi_trafo_raw": float(psi_trafo_raw),
    }


def test_component_meta_has_expected_keys() -> None:
    rf = NormalReward(_make_cfg())
    meta_keys = [meta.key for meta in rf.component_meta]
    assert meta_keys == ["r_cost", "r_throughput", "r_action_pen", "r_safe_v", "r_safe_trafo"]


def test_cost_only_behavior_without_violations() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(e_bat=np.array([0.2, -0.2, 0.0], dtype=np.float32))

    total, components = rf.compute(env_state)

    expected_cost = -env_state["e_bat"] * env_state["dt"] * env_state["price_t"]
    np.testing.assert_allclose(components["r_cost"], expected_cost, rtol=1e-5)
    np.testing.assert_allclose(components["r_safe_v"], 0.0, atol=1e-7)
    np.testing.assert_allclose(components["r_safe_trafo"], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        total,
        components["r_cost"] + components["r_throughput"] - components["r_action_pen"],
        rtol=1e-5,
    )


def test_throughput_reward_matches_abs_battery_power() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(e_bat=np.array([0.4, -0.2, 0.0], dtype=np.float32))

    _, components = rf.compute(env_state)

    expected = 0.01 * np.abs(env_state["e_bat"]) * env_state["dt"]
    np.testing.assert_allclose(components["r_throughput"], expected, rtol=1e-5)


def test_storage_action_limit_penalty_uses_request_execution_gap() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        e_bat_req=np.array([0.5, -0.5, 0.1], dtype=np.float32),
        e_bat=np.array([0.2, -0.1, 0.1], dtype=np.float32),
    )

    _, components = rf.compute(env_state)

    expected = 5.0 * np.abs(env_state["e_bat_req"] - env_state["e_bat"]) / env_state["p_max"]
    np.testing.assert_allclose(components["r_action_pen"], expected, rtol=1e-5)


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


def test_line_violations_do_not_affect_reward() -> None:
    rf = NormalReward(_make_cfg())
    with_line = _make_env_state(psi_line_raw=0.5)
    without_line = _make_env_state(psi_line_raw=0.0)

    total_with_line, components_with_line = rf.compute(with_line)
    total_without_line, components_without_line = rf.compute(without_line)

    np.testing.assert_allclose(components_with_line["r_safe_v"], components_without_line["r_safe_v"])
    np.testing.assert_allclose(
        components_with_line["r_safe_trafo"],
        components_without_line["r_safe_trafo"],
    )
    np.testing.assert_allclose(total_with_line, total_without_line)
