"""Unit tests for the GridCore physics layer.

Tests that do NOT require pandapower / simbench are fast and run always.
Tests decorated with ``@pytest.mark.slow`` load the real SimBench network
(~3-5 s on first call) and are skipped in fast CI runs.

Run fast tests only::

    pytest tests/test_grid_core.py -v -m "not slow"

Run all including slow::

    pytest tests/test_grid_core.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Lightweight dataclass smoke test (no pandapower needed)
# ---------------------------------------------------------------------------


def test_grid_step_result_import() -> None:
    from grid.core.grid_types import GridStepResult

    r = GridStepResult(
        converged=True,
        vm_pu=np.ones(5, dtype=np.float32),
        va_degree=np.zeros(5, dtype=np.float32),
        line_loading_pct=np.zeros(8, dtype=np.float32),
        p_mw_from=np.zeros(8, dtype=np.float32),
        agent_vm_pu=np.ones(3, dtype=np.float32),
        v_violation=np.zeros(3, dtype=np.float32),
        l_violation=0.0,
        n_buses=5,
        n_lines=8,
    )
    assert r.converged
    assert r.agent_vm_pu.shape == (3,)
    assert r.n_buses == 5


# ---------------------------------------------------------------------------
# Slow tests — require real simbench + pandapower
# ---------------------------------------------------------------------------

SB_CODE = "1-LV-rural1--0-sw"
N_AGENTS = 3


@pytest.fixture(scope="module")
def agent_deployments():
    """Return 3 AgentDeployment objects with validated bus IDs."""
    from grid.config.grid_config import AgentDeployment
    from grid.core.net_builder import build_simbench_net

    net = build_simbench_net(SB_CODE)
    all_bus_ids = list(net.bus.index)
    assert len(all_bus_ids) >= 3, "Network has fewer than 3 buses?"
    # Pick buses that have a load (prosumer-like nodes).
    buses_with_load = list(net.load["bus"].unique())
    chosen = buses_with_load[:N_AGENTS]
    return [
        AgentDeployment(bus_id=int(b), battery_capacity_kwh=5.0, battery_power_kw=2.5)
        for b in chosen
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
        w_l_pen=5.0,
    )


@pytest.mark.slow
def test_simbench_net_loads(agent_deployments) -> None:
    """Verify that the SimBench net can be loaded and has expected structure."""
    from grid.core.net_builder import build_simbench_net

    net = build_simbench_net(SB_CODE)
    assert len(net.bus) > 0, "Network has no buses"
    assert len(net.line) > 0, "Network has no lines"
    # Verify our chosen bus IDs are valid.
    for d in agent_deployments:
        assert d.bus_id in net.bus.index, f"bus_id {d.bus_id} not in network"


@pytest.mark.slow
def test_grid_core_zero_injection(agent_deployments, grid_cfg) -> None:
    """With zero battery injection, power flow should converge and voltages be sane."""
    from grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.zeros(N_AGENTS, dtype=np.float32)
    base_pv = np.zeros(N_AGENTS, dtype=np.float32)
    core.reset(base_load, base_pv)

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert result.converged, "Power flow should converge for zero injection"
    assert result.vm_pu.shape == (core.n_buses,)
    assert result.line_loading_pct.shape == (core.n_lines,)
    assert result.agent_vm_pu.shape == (N_AGENTS,)
    assert result.v_violation.shape == (N_AGENTS,)
    # Voltages for a lightly loaded network should be in a sane range.
    assert np.all(result.vm_pu > 0.8), "Some voltages are unrealistically low"
    assert np.all(result.vm_pu < 1.2), "Some voltages are unrealistically high"


@pytest.mark.slow
def test_grid_core_shapes(agent_deployments, grid_cfg) -> None:
    """Shape assertions across all result fields."""
    from grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.ones(N_AGENTS, dtype=np.float32) * 0.5
    core.reset(base_load, np.zeros(N_AGENTS))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert result.vm_pu.shape == (core.n_buses,)
    assert result.va_degree.shape == (core.n_buses,)
    assert result.line_loading_pct.shape == (core.n_lines,)
    assert result.p_mw_from.shape == (core.n_lines,)
    assert result.agent_vm_pu.shape == (N_AGENTS,)
    assert result.v_violation.shape == (N_AGENTS,)
    assert isinstance(result.l_violation, float)
    assert result.n_buses == core.n_buses
    assert result.n_lines == core.n_lines


@pytest.mark.slow
def test_grid_core_violation_nonneg(agent_deployments, grid_cfg) -> None:
    """Violation metrics must always be >= 0."""
    from grid.core.grid_core import GridCore

    core = GridCore(agent_deployments, grid_cfg)
    base_load = np.zeros(N_AGENTS, dtype=np.float32)
    core.reset(base_load, np.zeros(N_AGENTS))

    result = core.step(p_batt_kw=np.zeros(N_AGENTS), base_load_kw=base_load)

    assert np.all(result.v_violation >= 0.0), "v_violation has negative values"
    assert result.l_violation >= 0.0, "l_violation is negative"
