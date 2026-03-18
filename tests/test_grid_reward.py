"""Unit tests for GridCompositeReward.

These tests do not require pandapower or GPU — they exercise only the
reward computation logic with hand-crafted env_state dicts.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(w_v_pen: float = 10.0, w_l_pen: float = 5.0):
    """Return a minimal mock config accepted by GridCompositeReward."""

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
    cfg.grid = grid
    return cfg


def _make_env_state(n_agents: int = 3, v_violation=None, l_violation: float = 0.0):
    """Build a minimal env_state dict sufficient for GridCompositeReward."""
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
        "l_violation": float(l_violation),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_component_meta_length() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    meta_keys = {m.key for m in rf.component_meta}
    assert "r_v_pen" in meta_keys
    assert "r_l_pen" in meta_keys
    assert len(rf.component_meta) == 7   # 5 base + 2 grid


def test_component_meta_signs() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    sign_map = {m.key: m.sign for m in rf.component_meta}
    assert sign_map["r_v_pen"] == -1, "voltage penalty should subtract from total"
    assert sign_map["r_l_pen"] == -1, "line penalty should subtract from total"


def test_no_violation_zero_penalty() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    env_state = _make_env_state(v_violation=np.zeros(3), l_violation=0.0)
    total, components = rf.compute(env_state)

    assert np.allclose(components["r_v_pen"], 0.0), "v_pen should be 0 with no violations"
    assert np.allclose(components["r_l_pen"], 0.0), "l_pen should be 0 with no violations"


def test_voltage_violation_penalty_magnitude() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    w_v = 10.0
    cfg = _make_cfg(w_v_pen=w_v)
    rf = GridCompositeReward(cfg)
    v_viol = np.array([0.02, 0.0, 0.05], dtype=np.float32)
    env_state = _make_env_state(v_violation=v_viol, l_violation=0.0)
    total, components = rf.compute(env_state)

    expected = w_v * v_viol
    np.testing.assert_allclose(components["r_v_pen"], expected, rtol=1e-5)


def test_line_violation_penalty_broadcast() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    w_l = 5.0
    cfg = _make_cfg(w_l_pen=w_l)
    rf = GridCompositeReward(cfg)
    l_viol = 0.2   # worst line 20% beyond limit
    env_state = _make_env_state(v_violation=np.zeros(3), l_violation=l_viol)
    total, components = rf.compute(env_state)

    expected_scalar = w_l * l_viol
    # All agents should see the same line penalty.
    assert np.allclose(components["r_l_pen"], expected_scalar, rtol=1e-5)
    assert components["r_l_pen"].shape == (3,)


def test_total_subtracts_penalties() -> None:
    from envs.rewards.grid_composite import GridCompositeReward
    from envs.rewards.composite import CompositeReward

    cfg = _make_cfg(w_v_pen=10.0, w_l_pen=5.0)
    rf_grid = GridCompositeReward(cfg)
    rf_base = CompositeReward(cfg)

    v_viol = np.array([0.01, 0.02, 0.0], dtype=np.float32)
    l_viol = 0.1
    env_state = _make_env_state(v_violation=v_viol, l_violation=l_viol)

    base_total, _ = rf_base.compute(env_state)
    grid_total, grid_components = rf_grid.compute(env_state)

    expected_total = base_total - grid_components["r_v_pen"] - grid_components["r_l_pen"]
    np.testing.assert_allclose(grid_total, expected_total, rtol=1e-5)


def test_component_keys_match_meta() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    env_state = _make_env_state()
    _, components = rf.compute(env_state)
    meta_keys = {m.key for m in rf.component_meta}
    assert meta_keys == set(components.keys()), (
        f"component_meta keys {meta_keys} != compute() keys {set(components.keys())}"
    )
