from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from controllers.madrl.safety_projector import JointGridSafetyProjector
from envs.grid.core.grid_types import GridStepResult


def _make_projector(*, base_power_kw: float, limit_kw: float, mode: str = "joint_linearized") -> JointGridSafetyProjector:
    return JointGridSafetyProjector(
        n_agents=1,
        voltage_sensitivity=np.zeros((0, 2), dtype=np.float32),
        line_loading_sensitivity=np.zeros((0, 2), dtype=np.float32),
        trafo_power_sensitivity=np.array([[1.0, 1.0]], dtype=np.float32),
        voltage_base=np.zeros(0, dtype=np.float32),
        line_loading_base=np.zeros(0, dtype=np.float32),
        trafo_power_base_kw=np.array([base_power_kw], dtype=np.float32),
        voltage_min_pu=0.95,
        voltage_max_pu=1.05,
        line_limit_pct=100.0,
        trafo_limit_pct=100.0,
        trafo_rating_kw=np.array([limit_kw], dtype=np.float32),
        efficiency=0.95,
        dt_hours=1.0,
        soc_min=0.0,
        soc_max=1.0,
        projector_mode=mode,
        projection_iters=6,
        voltage_margin_pu=0.0,
        line_margin_pct=0.0,
        trafo_margin_pct=0.0,
        linearization_delta_kw=0.25,
    )


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        grid=SimpleNamespace(v_min_pu=0.95, v_max_pu=1.05, line_max_loading_pct=100.0),
        env=SimpleNamespace(efficiency=0.95, dt=1.0, soc_min=0.0, soc_max=1.0),
        safety=SimpleNamespace(
            linearization_delta_kw=0.25,
            projector_mode="joint_linearized_fast",
            projection_iters=4,
            voltage_margin_pu=0.0,
            line_margin_pct=0.0,
            trafo_margin_pct=5.0,
        ),
    )


def _fake_grid_result(*, p_hv_mw: float, loading_percent: float = 0.0) -> GridStepResult:
    return GridStepResult(
        converged=True,
        vm_pu=np.zeros(0, dtype=np.float32),
        va_degree=np.zeros(0, dtype=np.float32),
        line_loading_pct=np.zeros(0, dtype=np.float32),
        trafo_loading_pct=np.array([loading_percent], dtype=np.float32),
        p_mw_from=np.zeros(0, dtype=np.float32),
        agent_vm_pu=np.zeros(0, dtype=np.float32),
        v_violation=np.zeros(0, dtype=np.float32),
        line_violation=0.0,
        trafo_violation=max(loading_percent - 100.0, 0.0) / 100.0,
        l_violation=max(loading_percent - 100.0, 0.0) / 100.0,
        n_buses=0,
        n_lines=0,
        n_trafos=1,
        trafo_p_signed_kw=np.array([p_hv_mw * 1000.0], dtype=np.float32),
    )


def test_projector_reduces_export_side_trafo_violation() -> None:
    projector = _make_projector(base_power_kw=0.0, limit_kw=4.0)
    safety_local = torch.tensor([[0.5, 0.0, 5.0, 10.0, 1.0]], dtype=torch.float32)
    raw_actions = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)

    projected_actions, diagnostics = projector.project_actions_from_safety_local(
        safety_local,
        raw_actions,
        return_diagnostics=True,
    )

    assert diagnostics["post_violation"] <= diagnostics["pre_violation"] + 1e-6
    assert diagnostics["post_trafo_export_violation_kw"] <= diagnostics["pre_trafo_export_violation_kw"] + 1e-6
    assert projected_actions[0, 0].item() > raw_actions[0, 0].item()
    assert projected_actions[0, 1].item() < raw_actions[0, 1].item()


def test_projector_reduces_import_side_trafo_violation() -> None:
    projector = _make_projector(base_power_kw=0.0, limit_kw=4.0)
    safety_local = torch.tensor([[0.5, 5.0, 0.0, 10.0, 2.0]], dtype=torch.float32)
    raw_actions = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    projected_actions, diagnostics = projector.project_actions_from_safety_local(
        safety_local,
        raw_actions,
        return_diagnostics=True,
    )

    assert diagnostics["post_violation"] <= diagnostics["pre_violation"] + 1e-6
    assert diagnostics["post_trafo_import_violation_kw"] <= diagnostics["pre_trafo_import_violation_kw"] + 1e-6
    assert projected_actions[0, 0].item() < raw_actions[0, 0].item()
    assert projected_actions[0, 0].item() < 0.0


def test_joint_linearized_fast_combines_import_and_export_trafo_rows() -> None:
    projector = _make_projector(base_power_kw=0.0, limit_kw=4.0, mode="joint_linearized_fast")

    rows = projector.combined_constraint_rows.cpu().numpy()

    np.testing.assert_allclose(rows, np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32))
    np.testing.assert_allclose(projector.combined_constraint_row_norm_sq.cpu().numpy(), np.array([2.0, 2.0]))


def test_from_cfg_builds_signed_power_sensitivity_with_center_difference(monkeypatch) -> None:
    cfg = _make_cfg()
    call_log: list[float] = []

    class FakeGridCore:
        def __init__(self, deployments, grid_cfg) -> None:
            del deployments, grid_cfg
            self.n_buses = 0
            self.n_lines = 0
            self.n_trafos = 1
            self.net = SimpleNamespace(
                trafo=pd.DataFrame({"sn_mva": [0.4]}),
                res_trafo=pd.DataFrame({"p_hv_mw": [0.0]}),
            )

        def reset(self, base_load_kw, base_pv_kw) -> None:
            del base_load_kw, base_pv_kw

        def step(self, p_batt_kw, base_load_kw) -> GridStepResult:
            del base_load_kw
            value = float(np.asarray(p_batt_kw, dtype=np.float32)[0])
            call_log.append(value)
            return _fake_grid_result(p_hv_mw=2.0 * value / 1000.0)

    monkeypatch.setattr("controllers.madrl.safety_projector.build_agent_deployments", lambda cfg: [object()])
    monkeypatch.setattr("controllers.madrl.safety_projector.GridCore", FakeGridCore)

    projector = JointGridSafetyProjector.from_cfg(cfg)

    np.testing.assert_allclose(projector.trafo_power_sensitivity.cpu().numpy(), np.array([[2.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(projector.trafo_import_limit_kw.cpu().numpy(), np.array([380.0], dtype=np.float32))
    assert call_log == pytest.approx([0.0, 0.25, -0.25])


def test_from_cfg_disables_trafo_rows_without_p_hv_column(monkeypatch) -> None:
    cfg = _make_cfg()

    class FakeGridCoreNoPHv:
        def __init__(self, deployments, grid_cfg) -> None:
            del deployments, grid_cfg
            self.n_buses = 0
            self.n_lines = 0
            self.n_trafos = 1
            self.net = SimpleNamespace(
                trafo=pd.DataFrame({"sn_mva": [0.4]}),
                res_trafo=pd.DataFrame({"loading_percent": [0.0]}),
            )

        def reset(self, base_load_kw, base_pv_kw) -> None:
            del base_load_kw, base_pv_kw

        def step(self, p_batt_kw, base_load_kw) -> GridStepResult:
            del p_batt_kw, base_load_kw
            return _fake_grid_result(p_hv_mw=0.0)

    monkeypatch.setattr("controllers.madrl.safety_projector.build_agent_deployments", lambda cfg: [object()])
    monkeypatch.setattr("controllers.madrl.safety_projector.GridCore", FakeGridCoreNoPHv)

    projector = JointGridSafetyProjector.from_cfg(cfg)

    assert projector.trafo_power_sensitivity.shape == (0, 2)
    assert projector.combined_constraint_rows.shape == (0, 2)
    assert projector.trafo_import_limit_kw.numel() == 0
