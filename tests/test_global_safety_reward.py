"""全局安全势函数 + 灵敏度 credit 的专项测试。

覆盖要求：
1. 非 agent bus 的电压越限也会进入 psi_v_raw
2. line / trafo penalty 是 sum of squared excess，不是 max
3. sensitivity credit 方向正确
4. train compact info 与 test full info 行为区分正确
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _make_cfg(**overrides):
    class _Reward:
        w_pen = 5.0
        w_soc = 0.1
        lambda_bonus = 0.01
        w_global_safe = 1.0
        w_sens_credit = 0.2
        sens_credit_scale = 0.05

    class _Env:
        max_charge_rate = 0.5
        soc_target = 0.5

    class _Grid:
        w_v_pen = 10.0
        w_line_pen = 5.0
        w_trafo_pen = 5.0

    class _Cfg:
        reward = _Reward()
        env = _Env()
        grid = _Grid()

    cfg = _Cfg()
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], v)
    return cfg


def _base_env_state(n_agents: int = 3):
    """最小可运行的 env_state。"""
    rng = np.random.default_rng(42)
    soc = rng.uniform(0.3, 0.7, n_agents).astype(np.float32)
    e_t = soc * 5.0
    return {
        "e_bat_req": np.zeros(n_agents, dtype=np.float32),
        "e_bat": np.zeros(n_agents, dtype=np.float32),
        "soc_t": soc,
        "soc_next": soc,
        "e_t": e_t,
        "e_next": e_t,
        "e_min": np.full(n_agents, 0.25, dtype=np.float32),
        "e_max": np.full(n_agents, 4.75, dtype=np.float32),
        "p_max": np.full(n_agents, 0.5, dtype=np.float32),
        "price_t": 0.15,
        "net_load_t": np.ones(n_agents, dtype=np.float32),
        "mu_t": 0.15,
        "mu_next": 0.16,
        "gamma": 0.999,
        "dt": 0.25,
        # 默认无违规
        "psi_v_raw": 0.0,
        "psi_line_raw": 0.0,
        "psi_trafo_raw": 0.0,
        "bus_v_excess": np.zeros(15, dtype=np.float32),
        "bus_v_signed_indicator": np.zeros(15, dtype=np.float32),
        "line_excess": np.zeros(20, dtype=np.float32),
        "trafo_excess": np.zeros(2, dtype=np.float32),
        "sensitivity_snapshot": None,
        "v_violation": np.zeros(n_agents, dtype=np.float32),
        "line_violation": 0.0,
        "trafo_violation": 0.0,
        "l_violation": 0.0,
    }


# ------------------------------------------------------------------
# 1. 非 agent bus 电压越限进入 psi_v_raw
# ------------------------------------------------------------------

def test_non_agent_bus_voltage_excess_enters_psi_v() -> None:
    """即使 agent bus 无越限，其他 bus 越限仍应产生全局安全惩罚。"""
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    state = _base_env_state()

    # agent bus 无越限，但 bus #5（非 agent）有 0.05 pu 越限
    bus_v_excess = np.zeros(15, dtype=np.float32)
    bus_v_excess[5] = 0.05
    bus_v_signed = np.zeros(15, dtype=np.float32)
    bus_v_signed[5] = 1.0  # 过压
    psi_v = float(np.sum(bus_v_excess ** 2))  # 0.0025

    state["bus_v_excess"] = bus_v_excess
    state["bus_v_signed_indicator"] = bus_v_signed
    state["psi_v_raw"] = psi_v

    _, components = rf.compute(state)
    # 全局电压惩罚应为 w_global_safe * w_v_pen * psi_v = 1.0 * 10.0 * 0.0025 = 0.025
    expected = 10.0 * psi_v
    np.testing.assert_allclose(components["r_safe_v_global"][0], expected, rtol=1e-5)
    assert components["r_safe_v_global"][0] > 0.0


# ------------------------------------------------------------------
# 2. psi_line_raw 是 sum-of-squared-excess，不是 max
# ------------------------------------------------------------------

def test_psi_line_is_sum_of_squared_not_max() -> None:
    """两条线路各越限 0.05 和 0.10，psi = 0.05² + 0.10² = 0.0125。"""
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    state = _base_env_state()

    line_excess = np.zeros(20, dtype=np.float32)
    line_excess[2] = 0.05
    line_excess[7] = 0.10
    psi_line = float(np.sum(line_excess ** 2))  # 0.0025 + 0.01 = 0.0125

    state["line_excess"] = line_excess
    state["psi_line_raw"] = psi_line

    _, components = rf.compute(state)
    expected = 5.0 * psi_line  # w_line_pen * psi_line_raw
    np.testing.assert_allclose(components["r_safe_line_global"][0], expected, rtol=1e-5)

    # 如果是 max-only，psi 只会是 0.10² = 0.01，小于 0.0125
    assert psi_line > 0.01


# ------------------------------------------------------------------
# 3. sensitivity credit 方向正确性
# ------------------------------------------------------------------

def test_sensitivity_credit_direction() -> None:
    """agent 增加出力（充电）降低过压时，credit 应为正。"""
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg(**{"reward.w_sens_credit": 1.0, "reward.sens_credit_scale": 1.0})
    rf = GridCompositeReward(cfg)
    state = _base_env_state()

    n_buses = 15
    n_agents = 3
    # 仅 bus #5 过压 0.1 pu
    bus_v_excess = np.zeros(n_buses, dtype=np.float32)
    bus_v_excess[5] = 0.1
    bus_v_signed = np.zeros(n_buses, dtype=np.float32)
    bus_v_signed[5] = 1.0  # 过压
    psi_v = float(np.sum(bus_v_excess ** 2))

    # 灵敏度：agent 0 增加充电 → bus #5 电压下降（dvm/dp < 0）
    dvm_dp = np.zeros((n_buses, n_agents), dtype=np.float32)
    dvm_dp[5, 0] = -0.001  # 每 kW 充电降低 0.001 pu 电压

    state["bus_v_excess"] = bus_v_excess
    state["bus_v_signed_indicator"] = bus_v_signed
    state["psi_v_raw"] = psi_v
    state["sensitivity_snapshot"] = {
        "dvm_dp": dvm_dp,
        "dline_loading_dp": np.zeros((20, n_agents), dtype=np.float32),
        "dtrafo_loading_dp": np.zeros((2, n_agents), dtype=np.float32),
    }
    # agent 0 执行正功率（充电）
    state["e_bat"] = np.array([5.0, 0.0, 0.0], dtype=np.float32)

    _, components = rf.compute(state)

    # agent 0 的动作方向减少了过压 psi → credit 应为正
    assert components["r_sens_credit"][0] > 0.0
    # agent 1, 2 没有动作 → credit 应为 0
    np.testing.assert_allclose(components["r_sens_credit"][1], 0.0, atol=1e-7)
    np.testing.assert_allclose(components["r_sens_credit"][2], 0.0, atol=1e-7)


def test_sensitivity_credit_zero_when_no_violations() -> None:
    """无越限时，灵敏度 credit 应为零（无论灵敏度矩阵如何）。"""
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    state = _base_env_state()

    n_buses = 15
    n_agents = 3
    # 无越限
    state["bus_v_excess"] = np.zeros(n_buses, dtype=np.float32)
    state["bus_v_signed_indicator"] = np.zeros(n_buses, dtype=np.float32)
    state["line_excess"] = np.zeros(20, dtype=np.float32)
    state["trafo_excess"] = np.zeros(2, dtype=np.float32)

    # 灵敏度非零
    state["sensitivity_snapshot"] = {
        "dvm_dp": np.random.default_rng(0).standard_normal((n_buses, n_agents)).astype(np.float32),
        "dline_loading_dp": np.random.default_rng(1).standard_normal((20, n_agents)).astype(np.float32),
        "dtrafo_loading_dp": np.random.default_rng(2).standard_normal((2, n_agents)).astype(np.float32),
    }
    state["e_bat"] = np.array([5.0, -3.0, 2.0], dtype=np.float32)

    _, components = rf.compute(state)
    # 因为所有 excess = 0，梯度乘积也为 0 → credit 为 0
    np.testing.assert_allclose(components["r_sens_credit"], 0.0, atol=1e-7)


# ------------------------------------------------------------------
# 4. 全局惩罚所有 agent 相同
# ------------------------------------------------------------------

def test_global_penalties_identical_across_agents() -> None:
    from envs.rewards.grid_composite import GridCompositeReward

    cfg = _make_cfg()
    rf = GridCompositeReward(cfg)
    state = _base_env_state()
    state["psi_v_raw"] = 0.01
    state["psi_line_raw"] = 0.02
    state["psi_trafo_raw"] = 0.005

    _, components = rf.compute(state)

    for key in ("r_safe_v_global", "r_safe_line_global", "r_safe_trafo_global"):
        arr = components[key]
        assert arr[0] == arr[1] == arr[2], f"{key} should be identical across agents"
