"""Unit tests for GridCompositeReward（全局安全势函数 + 灵敏度 credit 版本）。"""

from __future__ import annotations

import numpy as np


def _make_cfg(
    w_v_pen: float = 10.0,
    w_line_pen: float = 5.0,
    w_trafo_pen: float = 5.0,
    w_global_safe: float = 1.0,
    w_sens_credit: float = 0.2,
    sens_credit_scale: float = 0.05,
):
    class _Reward:
        pass

    reward = _Reward()
    reward.w_pen = 5.0
    reward.w_soc = 0.1
    reward.lambda_bonus = 0.01
    reward.w_global_safe = w_global_safe
    reward.w_sens_credit = w_sens_credit
    reward.sens_credit_scale = sens_credit_scale

    class _Env:
        max_charge_rate = 0.5
        soc_target = 0.5

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.reward = reward
    cfg.env = _Env()

    class _Grid:
        pass

    grid = _Grid()
    grid.w_v_pen = w_v_pen
    grid.w_line_pen = w_line_pen
    grid.w_trafo_pen = w_trafo_pen
    cfg.grid = grid
    return cfg


def _make_env_state(
    n_agents: int = 3,
    *,
    psi_v_raw: float = 0.0,
    psi_line_raw: float = 0.0,
    psi_trafo_raw: float = 0.0,
    bus_v_excess=None,
    bus_v_signed_indicator=None,
    line_excess=None,
    trafo_excess=None,
    sensitivity_snapshot=None,
    e_bat=None,
):
    rng = np.random.default_rng(0)
    soc = rng.uniform(0.3, 0.7, n_agents).astype(np.float32)
    e_t = soc * 5.0
    e_next = np.clip(e_t + 0.01, 0.05 * 5.0, 0.95 * 5.0).astype(np.float32)
    if e_bat is None:
        e_bat = np.zeros(n_agents, dtype=np.float32)
    n_buses = 15
    n_lines = 20
    n_trafos = 2

    if bus_v_excess is None:
        bus_v_excess = np.zeros(n_buses, dtype=np.float32)
    if bus_v_signed_indicator is None:
        bus_v_signed_indicator = np.zeros(n_buses, dtype=np.float32)
    if line_excess is None:
        line_excess = np.zeros(n_lines, dtype=np.float32)
    if trafo_excess is None:
        trafo_excess = np.zeros(n_trafos, dtype=np.float32)

    return {
        "e_bat_req": e_bat.copy(),
        "e_bat": np.asarray(e_bat, dtype=np.float32),
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
        # 全局安全势函数字段
        "psi_v_raw": psi_v_raw,
        "psi_line_raw": psi_line_raw,
        "psi_trafo_raw": psi_trafo_raw,
        "bus_v_excess": np.asarray(bus_v_excess, dtype=np.float32),
        "bus_v_signed_indicator": np.asarray(bus_v_signed_indicator, dtype=np.float32),
        "line_excess": np.asarray(line_excess, dtype=np.float32),
        "trafo_excess": np.asarray(trafo_excess, dtype=np.float32),
        "sensitivity_snapshot": sensitivity_snapshot,
        # 旧兼容字段
        "v_violation": np.zeros(n_agents, dtype=np.float32),
        "line_violation": 0.0,
        "trafo_violation": 0.0,
        "l_violation": 0.0,
    }


def test_component_meta_length() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    meta_keys = {m.key for m in rf.component_meta}
    assert meta_keys >= {"r_safe_v_global", "r_safe_line_global", "r_safe_trafo_global", "r_sens_credit"}
    # 5 base + 4 grid = 9
    assert len(rf.component_meta) == 9


def test_component_meta_signs() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    sign_map = {m.key: m.sign for m in rf.component_meta}
    assert sign_map["r_safe_v_global"] == -1
    assert sign_map["r_safe_line_global"] == -1
    assert sign_map["r_safe_trafo_global"] == -1
    assert sign_map["r_sens_credit"] == +1


def test_no_violation_zero_penalty() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    _, components = rf.compute(_make_env_state())
    assert np.allclose(components["r_safe_v_global"], 0.0)
    assert np.allclose(components["r_safe_line_global"], 0.0)
    assert np.allclose(components["r_safe_trafo_global"], 0.0)
    assert np.allclose(components["r_sens_credit"], 0.0)


def test_global_voltage_penalty_shared() -> None:
    """全局电压惩罚所有 agent 应相同。"""
    from envs.rewards.grid_composite import GridCompositeReward

    w_v = 10.0
    psi_v = 0.04  # 例如两个 bus 各超 0.1 pu => 2*0.01=0.02... 这里直接给定
    rf = GridCompositeReward(_make_cfg(w_v_pen=w_v, w_global_safe=1.0))
    _, components = rf.compute(_make_env_state(psi_v_raw=psi_v))
    expected = w_v * psi_v
    np.testing.assert_allclose(components["r_safe_v_global"], np.full(3, expected), rtol=1e-5)
    # 所有 agent 应完全相同
    assert components["r_safe_v_global"][0] == components["r_safe_v_global"][1]
    assert components["r_safe_v_global"][1] == components["r_safe_v_global"][2]


def test_line_and_trafo_penalties_are_shared() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg(w_line_pen=7.0, w_trafo_pen=11.0))
    _, components = rf.compute(
        _make_env_state(psi_line_raw=0.04, psi_trafo_raw=0.01)
    )
    np.testing.assert_allclose(
        components["r_safe_line_global"], np.full(3, 7.0 * 0.04), rtol=1e-5
    )
    np.testing.assert_allclose(
        components["r_safe_trafo_global"], np.full(3, 11.0 * 0.01), rtol=1e-5
    )


def test_total_subtracts_all_grid_penalties() -> None:
    from envs.rewards.composite import CompositeReward
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg(w_v_pen=10.0, w_line_pen=5.0, w_trafo_pen=9.0)
    rf_base = CompositeReward(cfg)
    rf_grid = GridCompositeReward(cfg)
    env_state = _make_env_state(psi_v_raw=0.01, psi_line_raw=0.02, psi_trafo_raw=0.005)

    base_total, _ = rf_base.compute(env_state)
    grid_total, gc = rf_grid.compute(env_state)
    expected_total = (
        base_total
        - gc["r_safe_v_global"]
        - gc["r_safe_line_global"]
        - gc["r_safe_trafo_global"]
        + gc["r_sens_credit"]
    )
    np.testing.assert_allclose(grid_total, expected_total, rtol=1e-5)


def test_component_keys_match_meta() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    rf = GridCompositeReward(_make_cfg())
    _, components = rf.compute(_make_env_state())
    meta_keys = {m.key for m in rf.component_meta}
    assert meta_keys == set(components.keys())
