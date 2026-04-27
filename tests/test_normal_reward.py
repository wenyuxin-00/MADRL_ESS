"""Unit tests for NormalReward."""

from __future__ import annotations

import numpy as np
import pytest

from envs.rewards.NormalReward import NormalReward


def _make_cfg():
    class _Reward:
        action_boundary_penalty_weight = 0.05
        soc_boundary_regularization_weight = 2.0
        throughput_bonus_eur_per_kwh_max = 0.002
        soc_boundary_epsilon = 0.02
        soc_boundary_margin = 0.10
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
    soc_t=None,
    dt: float = 0.25,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    v_violation=None,
    psi_v_raw: float = 0.0,
    psi_line_raw: float = 0.0,
    psi_trafo_raw: float = 0.0,
    training_progress: float = 1.0,
):
    if battery_power_t is None:
        battery_power_t = np.zeros(n_agents, dtype=np.float32)
    if soc_t is None:
        soc_t = np.full(n_agents, 0.5, dtype=np.float32)
    if v_violation is None:
        v_violation = np.zeros(n_agents, dtype=np.float32)

    env_state = {
        "storage_price_t": float(storage_price_t),
        "battery_power_t": np.asarray(battery_power_t, dtype=np.float32),
        "soc_t": np.asarray(soc_t, dtype=np.float32),
        "soc_min": float(soc_min),
        "soc_max": float(soc_max),
        "dt": float(dt),
        "v_violation": np.asarray(v_violation, dtype=np.float32),
        "psi_v_raw": float(psi_v_raw),
        "psi_line_raw": float(psi_line_raw),
        "psi_trafo_raw": float(psi_trafo_raw),
        "training_progress": float(training_progress),
    }
    return env_state


def test_component_meta_has_expected_keys() -> None:
    rf = NormalReward(_make_cfg())
    meta_keys = [meta.key for meta in rf.component_meta]
    assert meta_keys == [
        "madrl_r_inc",
        "madrl_r_action_penalty",
        "madrl_r_soc_regularization",
        "madrl_r_throughput_bonus",
        "madrl_r_safe_v",
        "madrl_r_safe_line",
        "madrl_r_safe_trafo",
        "madrl_r_safe_total",
        "madrl_r_total_internal",
    ]


def test_incremental_profit_follows_battery_power_direction() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        battery_power_t=np.array([2.0, -1.0, 0.0], dtype=np.float32),
        training_progress=1.0,
    )

    total, components = rf.compute(env_state)

    expected_profit = np.array([-0.075, 0.0375, 0.0], dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_inc"], expected_profit, rtol=1e-6)
    np.testing.assert_allclose(components["madrl_r_action_penalty"], 0.0, rtol=1e-6)
    np.testing.assert_allclose(components["madrl_r_soc_regularization"], 0.0, rtol=1e-6)
    np.testing.assert_allclose(components["madrl_r_throughput_bonus"], 0.0, rtol=1e-6)
    np.testing.assert_allclose(total, components["madrl_r_inc"], rtol=1e-6)


def test_negative_price_makes_charging_profitable() -> None:
    rf = NormalReward(_make_cfg())
    total, components = rf.compute(
        _make_env_state(
            storage_price_t=-0.3,
            battery_power_t=np.array([2.0, 0.0, -1.0], dtype=np.float32),
            training_progress=1.0,
        )
    )

    expected_profit = np.array([0.15, 0.0, -0.075], dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_inc"], expected_profit, rtol=1e-6)
    np.testing.assert_allclose(total, expected_profit, rtol=1e-6)


def test_missing_reward_contract_field_is_rejected() -> None:
    cfg = _make_cfg()
    delattr(cfg.reward.__class__, "action_boundary_penalty_weight")

    with pytest.raises(AttributeError, match="action_boundary_penalty_weight"):
        NormalReward(cfg)


def test_action_boundary_penalty_uses_executed_power_at_soc_edges() -> None:
    rf = NormalReward(_make_cfg())
    total, components = rf.compute(
        _make_env_state(
            battery_power_t=np.array([-1.0, 0.5, 1.5], dtype=np.float32),
            soc_t=np.array([0.01, 0.50, 0.99], dtype=np.float32),
            training_progress=1.0,
        )
    )

    expected_penalty = np.array([0.05, 0.0, 0.075], dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_action_penalty"], expected_penalty, rtol=1e-6)
    np.testing.assert_allclose(
        total,
        components["madrl_r_inc"] - expected_penalty - components["madrl_r_soc_regularization"],
        rtol=1e-6,
    )


def test_reward_does_not_expose_action_feasibility_postprocessor() -> None:
    assert not hasattr(NormalReward(_make_cfg()), "apply_action_feasibility_regularization")


def test_soc_regularization_is_flat_inside_soft_band() -> None:
    rf = NormalReward(_make_cfg())
    _, components = rf.compute(
        _make_env_state(
            battery_power_t=np.zeros(3, dtype=np.float32),
            soc_t=np.array([0.05, 0.50, 0.95], dtype=np.float32),
            training_progress=1.0,
        )
    )

    expected = np.array([0.005, 0.0, 0.005], dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_soc_regularization"], expected, rtol=1e-6)


def test_throughput_bonus_anneals_with_training_progress() -> None:
    rf = NormalReward(_make_cfg())
    battery_power_t = np.array([2.0], dtype=np.float32)

    early_total, early_components = rf.compute(
        _make_env_state(
            n_agents=1,
            battery_power_t=battery_power_t,
            training_progress=0.10,
        )
    )
    mid_total, mid_components = rf.compute(
        _make_env_state(
            n_agents=1,
            battery_power_t=battery_power_t,
            training_progress=0.50,
        )
    )
    late_total, late_components = rf.compute(
        _make_env_state(
            n_agents=1,
            battery_power_t=battery_power_t,
            training_progress=0.90,
        )
    )

    assert early_components["madrl_r_throughput_bonus"][0] == pytest.approx(0.001)
    assert mid_components["madrl_r_throughput_bonus"][0] == pytest.approx(0.0005)
    assert late_components["madrl_r_throughput_bonus"][0] == pytest.approx(0.0)
    assert early_total[0] > mid_total[0] > late_total[0]


def test_voltage_penalty_splits_by_local_violation_proportion() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        v_violation=np.array([0.02, 0.0, 0.01], dtype=np.float32),
        psi_v_raw=0.005,
        training_progress=1.0,
    )

    _, components = rf.compute(env_state)

    voltage_total = 10.0 * 0.005
    expected = np.array([0.1, 0.0, 0.05], dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_safe_v"], expected, rtol=1e-5)
    assert np.isclose(np.mean(components["madrl_r_safe_v"]), voltage_total)


def test_voltage_penalty_falls_back_to_uniform_split_for_non_agent_violations() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        v_violation=np.zeros(3, dtype=np.float32),
        psi_v_raw=0.005,
        training_progress=1.0,
    )

    _, components = rf.compute(env_state)

    expected = np.full(3, 10.0 * 0.005, dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_safe_v"], expected, rtol=1e-5)


def test_transformer_penalty_is_shared() -> None:
    rf = NormalReward(_make_cfg())
    _, components = rf.compute(_make_env_state(psi_trafo_raw=0.02, training_progress=1.0))

    expected = np.full(3, 7.0 * 0.02, dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_safe_trafo"], expected, rtol=1e-5)


def test_line_penalty_is_shared() -> None:
    rf = NormalReward(_make_cfg())
    _, components = rf.compute(_make_env_state(psi_line_raw=0.5, training_progress=1.0))

    expected = np.full(3, 3.0 * 0.5, dtype=np.float32)
    np.testing.assert_allclose(components["madrl_r_safe_line"], expected, rtol=1e-5)


def test_objective_combines_increment_penalty_regularization_bonus_and_safety() -> None:
    rf = NormalReward(_make_cfg())
    env_state = _make_env_state(
        battery_power_t=np.array([2.0, -1.0, 0.5], dtype=np.float32),
        soc_t=np.array([0.99, 0.01, 0.05], dtype=np.float32),
        training_progress=0.5,
        v_violation=np.array([0.01, 0.0, 0.0], dtype=np.float32),
        psi_v_raw=0.002,
        psi_line_raw=0.1,
        psi_trafo_raw=0.05,
    )

    total, components = rf.compute(env_state)

    expected = (
        components["madrl_r_inc"]
        - components["madrl_r_action_penalty"]
        - components["madrl_r_soc_regularization"]
        + components["madrl_r_throughput_bonus"]
        - components["madrl_r_safe_total"]
    )
    np.testing.assert_allclose(total, expected, rtol=1e-6)


def test_old_grid_power_contract_is_rejected() -> None:
    rf = NormalReward(_make_cfg())
    with pytest.raises(KeyError, match="battery_power_t"):
        rf.compute({"storage_price_t": 0.2, "actual_grid_power_t": np.zeros(3, dtype=np.float32)})
