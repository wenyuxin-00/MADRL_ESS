"""Unit tests for the GridCore physics layer."""

from __future__ import annotations

import numpy as np
import pytest


def test_grid_step_result_import() -> None:
    from envs.grid.core.grid_core import GridStepResult

    result = GridStepResult(
        converged=True,
        vm_pu=np.ones(5, dtype=np.float32),
        line_loading_pct=np.zeros(8, dtype=np.float32),
        trafo_loading_pct=np.zeros(2, dtype=np.float32),
        v_violation=np.zeros(3, dtype=np.float32),
    )
    assert result.converged
    assert result.v_violation.shape == (3,)
    assert result.trafo_loading_pct.shape == (2,)
    assert result.trafo_p_signed_kw.shape == (0,)
    assert result.psi_v_raw == 0.0


SB_CODE = "1-LV-rural1--0-sw"
N_AGENTS = 3


@pytest.fixture(scope="module")
def agent_deployments():
    from envs.grid.core.net_builder import build_simbench_net
    from envs.grid.deployments import AgentDeployment

    net = build_simbench_net(SB_CODE)
    assert len(net.bus) >= 3
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
    )


def _assert_non_agent_power_zero(core) -> None:
    agent_bus_set = {int(bus_id) for bus_id in core.agent_bus_ids}
    for table_name in ("load", "sgen"):
        table = getattr(core.net, table_name)
        non_agent_mask = ~table["bus"].isin(agent_bus_set)
        for column in ("p_mw", "q_mvar"):
            if column in table.columns:
                values = table.loc[non_agent_mask, column].to_numpy(dtype=np.float64)
                assert np.allclose(values, 0.0)


@pytest.mark.slow
def test_simbench_net_loads(agent_deployments) -> None:
    from envs.grid.core.net_builder import build_simbench_net

    net = build_simbench_net(SB_CODE)
    assert len(net.bus) > 0
    assert len(net.line) > 0
    for deployment in agent_deployments:
        assert deployment.bus_id in net.bus.index


@pytest.mark.slow
def test_grid_core_zeroes_non_agent_static_power(agent_deployments, grid_cfg) -> None:
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    _assert_non_agent_power_zero(core)

    base_load = np.ones(N_AGENTS, dtype=np.float32)
    base_pv = np.ones(N_AGENTS, dtype=np.float32) * 0.25
    core.reset(base_load, base_pv)

    _assert_non_agent_power_zero(core)


@pytest.mark.slow
def test_grid_core_step_power_exists_only_on_agent_buses(agent_deployments, grid_cfg) -> None:
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.ones(N_AGENTS, dtype=np.float32) * 0.5
    core.reset(base_load, np.zeros(N_AGENTS, dtype=np.float32))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS, dtype=np.float32), base_load_kw=base_load)

    assert result.converged
    _assert_non_agent_power_zero(core)
    agent_load = core.net.load.loc[core.net.load["bus"].isin(core.agent_bus_ids), "p_mw"]
    assert float(agent_load.sum()) > 0.0


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
    assert result.trafo_p_signed_kw.shape == (core.n_trafos,)
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
    assert result.line_loading_pct.shape == (core.n_lines,)
    assert result.trafo_loading_pct.shape == (core.n_trafos,)
    assert result.trafo_p_signed_kw.shape == (core.n_trafos,)
    assert result.v_violation.shape == (N_AGENTS,)


@pytest.mark.slow
def test_grid_core_violation_nonneg(agent_deployments, grid_cfg) -> None:
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.zeros(N_AGENTS, dtype=np.float32)
    core.reset(base_load, np.zeros(N_AGENTS))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert np.all(result.v_violation >= 0.0)
    assert result.psi_line_raw >= 0.0
    assert result.psi_trafo_raw >= 0.0


@pytest.mark.slow
def test_psi_fields_shapes_and_nonneg(agent_deployments, grid_cfg) -> None:
    from envs.grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.ones(N_AGENTS, dtype=np.float32) * 0.5
    core.reset(base_load, np.zeros(N_AGENTS))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert isinstance(result.psi_v_raw, float) and result.psi_v_raw >= 0.0
    assert isinstance(result.psi_line_raw, float) and result.psi_line_raw >= 0.0
    assert isinstance(result.psi_trafo_raw, float) and result.psi_trafo_raw >= 0.0
