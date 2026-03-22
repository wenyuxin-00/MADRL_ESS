"""Unit tests for GridCompositeReward."""

from __future__ import annotations

import numpy as np


def _make_cfg(
    w_v_pen: float = 10.0,
    w_l_pen: float = 5.0,
    w_line_pen: float | None = None,
    w_trafo_pen: float | None = None,
):
    class _Reward:
        w_pen = 5.0
        w_soc = 0.1
        lambda_bonus = 0.01

    class _Env:
        max_charge_rate = 0.5
        soc_target = 0.5

    class _Cfg:
        reward = _Reward()
        env = _Env()

    cfg = _Cfg()

    class _Grid:
        pass

    grid = _Grid()
    grid.w_v_pen = w_v_pen
    grid.w_l_pen = w_l_pen
    grid.w_line_pen = w_l_pen if w_line_pen is None else w_line_pen
    grid.w_trafo_pen = w_l_pen if w_trafo_pen is None else w_trafo_pen
    cfg.grid = grid
    return cfg


def _make_env_state(
    n_agents: int = 3,
    *,
    v_violation=None,
    line_violation: float = 0.0,
    trafo_violation: float = 0.0,
):
    rng = np.random.default_rng(0)
    soc = rng.uniform(0.3, 0.7, n_agents).astype(np.float32)
    e_t = soc * 5.0
    e_next = np.clip(e_t + 0.01, 0.05 * 5.0, 0.95 * 5.0).astype(np.float32)
    e_bat = np.zeros(n_agents, dtype=np.float32)

    if v_violation is None:
        v_violation = np.zeros(n_agents, dtype=np.float32)

    return {
        "e_bat_req": e_bat,
        "e_bat": e_bat,
        "soc_t": soc,
        "soc_next": soc,
        "e_t": e_t,
        "e_next": e_next,
        "e_min": np.full(n_agents, 0.05 * 5.0, dtype=np.float32),
        "e_max": np.full(n_agents, 0.95 * 5.0, dtype=np.float32),
        "p_max": np.full(n_agents, 0.5, dtype=np.float32),
        "price_t": 0.15,
        "net_load_t": np.full(n_agents, 1.0, dtype=np.float32),
        "mu_t": 0.15,
        "mu_next": 0.16,
        "gamma": 0.999,
        "dt": 0.25,
        "v_violation": np.asarray(v_violation, dtype=np.float32),
        "line_violation": float(line_violation),
        "trafo_violation": float(trafo_violation),
        "l_violation": float(max(line_violation, trafo_violation)),
    }


def test_component_meta_length() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    meta_keys = {m.key for m in rf.component_meta}
    assert meta_keys >= {"r_v_pen", "r_line_pen", "r_trafo_pen"}
    assert len(rf.component_meta) == 8


def test_component_meta_signs() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    sign_map = {m.key: m.sign for m in rf.component_meta}
    assert sign_map["r_v_pen"] == -1
    assert sign_map["r_line_pen"] == -1
    assert sign_map["r_trafo_pen"] == -1


def test_no_violation_zero_penalty() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    _, components = rf.compute(_make_env_state())
    assert np.allclose(components["r_v_pen"], 0.0)
    assert np.allclose(components["r_line_pen"], 0.0)
    assert np.allclose(components["r_trafo_pen"], 0.0)


def test_voltage_violation_penalty_is_local() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    w_v = 10.0
    rf = GridCompositeReward(_make_cfg(w_v_pen=w_v))
    v_viol = np.array([0.02, 0.0, 0.05], dtype=np.float32)
    _, components = rf.compute(_make_env_state(v_violation=v_viol))
    np.testing.assert_allclose(components["r_v_pen"], w_v * v_viol, rtol=1e-5)


def test_line_and_transformer_penalties_are_shared() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg(w_line_pen=7.0, w_trafo_pen=11.0))
    _, components = rf.compute(_make_env_state(line_violation=0.2, trafo_violation=0.1))
    assert np.allclose(components["r_line_pen"], np.full(3, 1.4, dtype=np.float32))
    assert np.allclose(components["r_trafo_pen"], np.full(3, 1.1, dtype=np.float32))


def test_total_subtracts_all_grid_penalties() -> None:
    from envs.rewards.composite import CompositeReward
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg(w_v_pen=10.0, w_line_pen=5.0, w_trafo_pen=9.0)
    rf_base = CompositeReward(cfg)
    rf_grid = GridCompositeReward(cfg)
    env_state = _make_env_state(
        v_violation=np.array([0.01, 0.02, 0.0], dtype=np.float32),
        line_violation=0.1,
        trafo_violation=0.05,
    )

    base_total, _ = rf_base.compute(env_state)
    grid_total, grid_components = rf_grid.compute(env_state)
    expected_total = (
        base_total
        - grid_components["r_v_pen"]
        - grid_components["r_line_pen"]
        - grid_components["r_trafo_pen"]
    )
    np.testing.assert_allclose(grid_total, expected_total, rtol=1e-5)


def test_component_keys_match_meta() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    _, components = rf.compute(_make_env_state())
    meta_keys = {m.key for m in rf.component_meta}
    assert meta_keys == set(components.keys())
