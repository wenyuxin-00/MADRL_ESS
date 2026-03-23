"""Unit tests for the GridCore physics layer."""

from __future__ import annotations

import numpy as np
import pytest


def test_grid_step_result_import() -> None:
    from envs.grid.core.grid_types import GridStepResult

    result = GridStepResult(
        converged=True,
        vm_pu=np.ones(5, dtype=np.float32),
        va_degree=np.zeros(5, dtype=np.float32),
        line_loading_pct=np.zeros(8, dtype=np.float32),
        trafo_loading_pct=np.zeros(2, dtype=np.float32),
        p_mw_from=np.zeros(8, dtype=np.float32),
        agent_vm_pu=np.ones(3, dtype=np.float32),
        v_violation=np.zeros(3, dtype=np.float32),
        line_violation=0.0,
        trafo_violation=0.0,
        l_violation=0.0,
        n_buses=5,
        n_lines=8,
        n_trafos=2,
    )
    assert result.converged
    assert result.agent_vm_pu.shape == (3,)
    assert result.trafo_loading_pct.shape == (2,)
    assert result.n_buses == 5


SB_CODE = "1-LV-rural1--0-sw"
N_AGENTS = 3


@pytest.fixture(scope="module")
def agent_deployments():
    from envs.grid.config.grid_config import AgentDeployment
    from envs.grid.core.net_builder import build_simbench_net

    net = build_simbench_net(SB_CODE)
    all_bus_ids = list(net.bus.index)
    assert len(all_bus_ids) >= 3
    buses_with_load = list(net.load["bus"].unique())
    chosen = buses_with_load[:N_AGENTS]
    return [
        AgentDeployment(bus_id=int(bus_id), battery_capacity_kwh=5.0, battery_power_kw=2.5)
        for bus_id in chosen
    ]


@pytest.fixture(scope="module")
def grid_cfg():
    from configs.experiment_config import GridConfig

    return GridConfig(
        sb_code=SB_CODE,
        pf_solver="nr",
        v_min_pu=0.95,
        v_max_pu=1.05,
        line_max_loading_pct=100.0,
        w_v_pen=10.0,
    )


@pytest.mark.slow
def test_simbench_net_loads(agent_deployments) -> None:
    from envs.grid.core.net_builder import build_simbench_net

    net = build_simbench_net(SB_CODE)
    assert len(net.bus) > 0
    assert len(net.line) > 0
    for deployment in agent_deployments:
        assert deployment.bus_id in net.bus.index


@pytest.mark.slow
def test_grid_core_zero_injection(agent_deployments, grid_cfg) -> None:
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.zeros(N_AGENTS, dtype=np.float32)
    base_pv = np.zeros(N_AGENTS, dtype=np.float32)
    core.reset(base_load, base_pv)

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert result.converged
    assert result.vm_pu.shape == (core.n_buses,)
    assert result.line_loading_pct.shape == (core.n_lines,)
    assert result.trafo_loading_pct.shape == (core.n_trafos,)
    assert result.agent_vm_pu.shape == (N_AGENTS,)
    assert result.v_violation.shape == (N_AGENTS,)
    assert np.all(result.vm_pu > 0.8)
    assert np.all(result.vm_pu < 1.2)


@pytest.mark.slow
def test_grid_core_shapes(agent_deployments, grid_cfg) -> None:
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.ones(N_AGENTS, dtype=np.float32) * 0.5
    core.reset(base_load, np.zeros(N_AGENTS))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert result.vm_pu.shape == (core.n_buses,)
    assert result.va_degree.shape == (core.n_buses,)
    assert result.line_loading_pct.shape == (core.n_lines,)
    assert result.trafo_loading_pct.shape == (core.n_trafos,)
    assert result.p_mw_from.shape == (core.n_lines,)
    assert result.agent_vm_pu.shape == (N_AGENTS,)
    assert result.v_violation.shape == (N_AGENTS,)
    assert isinstance(result.line_violation, float)
    assert isinstance(result.trafo_violation, float)
    assert isinstance(result.l_violation, float)
    assert result.n_buses == core.n_buses
    assert result.n_lines == core.n_lines
    assert result.n_trafos == core.n_trafos


@pytest.mark.slow
def test_grid_core_violation_nonneg(agent_deployments, grid_cfg) -> None:
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.zeros(N_AGENTS, dtype=np.float32)
    core.reset(base_load, np.zeros(N_AGENTS))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert np.all(result.v_violation >= 0.0)
    assert result.line_violation >= 0.0
    assert result.trafo_violation >= 0.0
    assert result.l_violation >= 0.0


# ------------------------------------------------------------------
# Phase 2 新增：全局安全势函数字段 + 灵敏度快照
# ------------------------------------------------------------------

@pytest.mark.slow
def test_psi_fields_shapes_and_nonneg(agent_deployments, grid_cfg) -> None:
    """新 psi 字段应为标量且非负，excess 数组形状正确。"""
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.ones(N_AGENTS, dtype=np.float32) * 0.5
    core.reset(base_load, np.zeros(N_AGENTS))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    # psi 标量非负
    assert isinstance(result.psi_v_raw, float) and result.psi_v_raw >= 0.0
    assert isinstance(result.psi_line_raw, float) and result.psi_line_raw >= 0.0
    assert isinstance(result.psi_trafo_raw, float) and result.psi_trafo_raw >= 0.0

    # excess 数组形状
    assert result.bus_v_excess.shape == (core.n_buses,)
    assert result.bus_v_signed_indicator.shape == (core.n_buses,)
    assert result.line_excess.shape == (core.n_lines,)
    assert result.trafo_excess.shape == (core.n_trafos,)

    # excess 元素非负
    assert np.all(result.bus_v_excess >= 0.0)
    assert np.all(result.line_excess >= 0.0)
    assert np.all(result.trafo_excess >= 0.0)

    # signed indicator 只包含 -1, 0, +1
    unique_vals = set(np.unique(result.bus_v_signed_indicator))
    assert unique_vals <= {-1.0, 0.0, 1.0}

    # psi_v_raw == sum(bus_v_excess²)
    np.testing.assert_allclose(result.psi_v_raw, float(np.sum(result.bus_v_excess ** 2)), rtol=1e-5)
    np.testing.assert_allclose(result.psi_line_raw, float(np.sum(result.line_excess ** 2)), rtol=1e-5)
    np.testing.assert_allclose(result.psi_trafo_raw, float(np.sum(result.trafo_excess ** 2)), rtol=1e-5)


@pytest.mark.slow
def test_compute_sensitivity_snapshot_shapes(agent_deployments, grid_cfg) -> None:
    """compute_sensitivity_snapshot 返回正确形状的灵敏度矩阵。"""
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.ones(N_AGENTS, dtype=np.float32) * 0.5
    base_pv = np.zeros(N_AGENTS, dtype=np.float32)
    core.reset(base_load, base_pv)

    # 先跑一步让 net 有结果
    core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    snapshot = core.compute_sensitivity_snapshot(
        p_batt_kw=np.zeros(N_AGENTS, dtype=np.float32),
        base_load_kw=base_load,
        delta_kw=1.0,
    )

    assert "dvm_dp" in snapshot
    assert "dline_loading_dp" in snapshot
    assert "dtrafo_loading_dp" in snapshot

    assert snapshot["dvm_dp"].shape == (core.n_buses, N_AGENTS)
    assert snapshot["dline_loading_dp"].shape == (core.n_lines, N_AGENTS)
    assert snapshot["dtrafo_loading_dp"].shape == (core.n_trafos, N_AGENTS)

    # 灵敏度不应全为零（非退化网络）
    assert np.any(snapshot["dvm_dp"] != 0.0)
