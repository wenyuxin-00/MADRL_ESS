from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.utils import dual_distributed_notebook_helpers as dual_nb
from tests.support.helpers import make_case_dir, make_smoke_config


def _make_bundle(
    *,
    net_load_kw: np.ndarray,
    charge_kw: np.ndarray,
    discharge_kw: np.ndarray,
    p_max_kw: np.ndarray,
    pv_seq: np.ndarray | None = None,
    pv_curtail_kw: np.ndarray | None = None,
) -> dual_nb.LocalMPCBundle:
    net_load_kw = np.asarray(net_load_kw, dtype=np.float32)
    charge_kw = np.asarray(charge_kw, dtype=np.float32)
    discharge_kw = np.asarray(discharge_kw, dtype=np.float32)
    p_max_kw = np.asarray(p_max_kw, dtype=np.float32)
    n_agents, horizon = net_load_kw.shape
    pv_seq_array = (
        np.zeros((n_agents, horizon), dtype=np.float32)
        if pv_seq is None
        else np.asarray(pv_seq, dtype=np.float32)
    )
    pv_curtail_array = (
        np.zeros((n_agents, horizon), dtype=np.float32)
        if pv_curtail_kw is None
        else np.asarray(pv_curtail_kw, dtype=np.float32)
    )
    pv_effective_kw = np.maximum(pv_seq_array - pv_curtail_array, 0.0).astype(np.float32)
    pv_utilization = np.ones((n_agents, horizon), dtype=np.float32)
    valid_mask = pv_seq_array > 1e-6
    pv_utilization[valid_mask] = np.clip(
        pv_effective_kw[valid_mask] / pv_seq_array[valid_mask],
        0.0,
        1.0,
    ).astype(np.float32)
    return dual_nb.LocalMPCBundle(
        price_seq=np.zeros((horizon,), dtype=np.float32),
        load_seq=np.zeros((n_agents, horizon), dtype=np.float32),
        pv_seq=pv_seq_array,
        soc_init=np.full((n_agents,), 0.5, dtype=np.float32),
        p_max_kw=p_max_kw,
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        pv_curtail_kw=pv_curtail_array,
        pv_effective_kw=pv_effective_kw,
        pv_utilization=pv_utilization,
        signed_battery_kw=charge_kw - discharge_kw,
        net_load_kw=net_load_kw,
        energy_kwh=np.zeros((n_agents, horizon + 1), dtype=np.float32),
        objective_eur=np.zeros((n_agents,), dtype=np.float32),
        solve_time_sec=np.zeros((n_agents,), dtype=np.float32),
        feasible=np.ones((n_agents,), dtype=bool),
        net_load_floor_kw=None,
    )


def test_battery_to_netload_sensitivity_uses_battery_columns():
    projector = SimpleNamespace(
        n_agents=2,
        trafo_power_sensitivity=np.array([[1.5, 2.5, 1.5, 2.5]], dtype=np.float32),
    )

    alpha = dual_nb._battery_to_netload_sensitivity(projector)

    np.testing.assert_allclose(alpha, np.array([1.5, 2.5], dtype=np.float32))


def test_compute_netload_delta_max_kw_covers_discharge_and_charge_headroom():
    bundle = _make_bundle(
        net_load_kw=np.array([[0.0], [0.0]], dtype=np.float32),
        charge_kw=np.array([[1.0], [0.5]], dtype=np.float32),
        discharge_kw=np.array([[0.0], [1.5]], dtype=np.float32),
        p_max_kw=np.array([2.0, 3.0], dtype=np.float32),
        pv_seq=np.array([[0.3], [0.7]], dtype=np.float32),
    )

    delta_max_kw = dual_nb._compute_netload_delta_max_kw(bundle)

    np.testing.assert_allclose(delta_max_kw[:, 0], np.array([1.3, 4.7], dtype=np.float32))


def test_evaluate_first_step_pf_accounts_for_pv_curtailment():
    captured: dict[str, np.ndarray] = {}

    class _DummyGridCore:
        def step(self, *, p_batt_kw, base_load_kw):
            captured["p_batt_kw"] = np.asarray(p_batt_kw, dtype=np.float32)
            captured["base_load_kw"] = np.asarray(base_load_kw, dtype=np.float32)
            return SimpleNamespace(
                line_loading_pct=np.array([12.0], dtype=np.float32),
                trafo_loading_pct=np.array([23.0], dtype=np.float32),
                trafo_p_signed_kw=np.array([1.5], dtype=np.float32),
            )

    env = SimpleNamespace(
        _grid_core=_DummyGridCore(),
        get_signal_step=lambda name: {
            "load": np.array([2.0], dtype=np.float32),
            "pv": np.array([1.5], dtype=np.float32),
        }[name],
    )

    evaluation = dual_nb._evaluate_first_step_pf(
        env,
        np.array([0.4], dtype=np.float32),
        np.array([0.6], dtype=np.float32),
        trafo_limit_kw=2.0,
        loading_limit_pct=100.0,
    )

    np.testing.assert_allclose(captured["base_load_kw"], np.array([1.1], dtype=np.float32))
    np.testing.assert_allclose(captured["p_batt_kw"], np.array([0.4], dtype=np.float32))
    np.testing.assert_allclose(evaluation.pv_curtail_kw, np.array([0.6], dtype=np.float32))


def test_solve_first_pass_local_mpc_bundle_raises_on_infeasible_local_solver(monkeypatch):
    env = SimpleNamespace(
        n=1,
        soc=np.array([0.5], dtype=np.float32),
        agent_c_bat=np.array([4.0], dtype=np.float32),
        agent_p_max=np.array([2.0], dtype=np.float32),
        reward_fn=SimpleNamespace(export_subsidy_eur_per_kwh=0.079),
        dt=1.0,
        eff=1.0,
        soc_min=0.1,
        soc_max=0.9,
    )
    raw_obs = {
        "price_seq": np.array([0.1, 0.2], dtype=np.float32),
        "load_seq": np.array([[1.0, 1.0]], dtype=np.float32),
        "pv_seq": np.array([[0.0, 0.0]], dtype=np.float32),
    }

    infeasible_result = SimpleNamespace(
        charge_kw=np.zeros((2,), dtype=np.float32),
        discharge_kw=np.zeros((2,), dtype=np.float32),
        pv_curtail_kw=np.zeros((2,), dtype=np.float32),
        pv_effective_kw=np.zeros((2,), dtype=np.float32),
        pv_utilization=np.ones((2,), dtype=np.float32),
        signed_battery_kw=np.zeros((2,), dtype=np.float32),
        net_load_kw=np.zeros((2,), dtype=np.float32),
        energy_kwh=np.zeros((3,), dtype=np.float32),
        objective_eur=0.0,
        solve_time_sec=0.01,
        feasible=False,
    )

    monkeypatch.setattr(
        dual_nb.grid_nb,
        "_get_single_agent_mpc_solver",
        lambda *args, **kwargs: (SimpleNamespace(solve_full_horizon=lambda **solve_kwargs: infeasible_result), {}),
    )

    with pytest.raises(RuntimeError, match="Round-1 local MPC returned infeasible or unbounded solutions"):
        dual_nb.solve_first_pass_local_mpc_bundle(env, raw_obs)


def test_build_dual_trafo_surrogate_aggregates_round1_window():
    bundle = _make_bundle(
        net_load_kw=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        charge_kw=np.zeros((2, 2), dtype=np.float32),
        discharge_kw=np.zeros((2, 2), dtype=np.float32),
        p_max_kw=np.array([2.0, 2.0], dtype=np.float32),
    )
    projector = SimpleNamespace(
        n_agents=2,
        trafo_power_sensitivity=np.array([[2.0, 3.0, 2.0, 3.0]], dtype=np.float32),
        trafo_power_base_kw=np.array([10.0], dtype=np.float32),
    )
    env = SimpleNamespace(
        _grid_cfg=SimpleNamespace(line_max_loading_pct=100.0),
        _grid_core=SimpleNamespace(net=SimpleNamespace(trafo=pd.DataFrame({"sn_mva": [0.4]}))),
    )
    cfg = SimpleNamespace(grid=SimpleNamespace(line_max_loading_pct=100.0))

    surrogate = dual_nb.build_dual_trafo_surrogate(cfg, env, bundle, projector=projector)

    np.testing.assert_allclose(surrogate.alpha_netload_kw, np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(surrogate.baseline_root_p_kw, np.array([21.0, 26.0], dtype=np.float32))
    assert surrogate.trafo_limit_kw == 400.0


def test_per_step_netload_qp_allocation_respects_box_and_transformer_constraint():
    bundle = _make_bundle(
        net_load_kw=np.array([[-5.0], [-3.0]], dtype=np.float32),
        charge_kw=np.array([[0.2], [0.1]], dtype=np.float32),
        discharge_kw=np.array([[0.0], [0.0]], dtype=np.float32),
        p_max_kw=np.array([3.0, 3.0], dtype=np.float32),
    )
    surrogate = dual_nb.DualTrafoSurrogate(
        trafo_limit_kw=6.0,
        alpha_netload_kw=np.array([1.0, 1.0], dtype=np.float32),
        alpha_netload_window_kw=np.array([[1.0], [1.0]], dtype=np.float32),
        baseline_root_p_kw=np.array([-10.0], dtype=np.float32),
        export_overload_mask=np.array([True]),
        import_overload_mask=np.array([False]),
    )

    allocation = dual_nb.solve_per_step_netload_qp_allocation(bundle, surrogate)

    delivered = float(np.sum(allocation.delta_netload_kw[:, 0]))
    assert delivered >= 4.0 - 1e-3
    assert np.all(allocation.delta_netload_kw[:, 0] <= allocation.delta_netload_max_kw[:, 0] + 1e-6)
    assert allocation.dual_lambda.shape == (1,)
    assert allocation.dual_lambda[0] > 0.0
    np.testing.assert_allclose(
        allocation.surrogate_trafo_relief_kw,
        np.array([float(np.dot(surrogate.alpha_netload_kw, allocation.delta_netload_kw[:, 0]))], dtype=np.float32),
    )


def test_single_step_identity_qp_zero_deficit_returns_zero_lambda():
    delta_step, dual_lambda_step, surrogate_relief_kw_step, feasible = dual_nb._solve_single_step_identity_qp(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([2.0, 3.0], dtype=np.float32),
        0.0,
    )

    np.testing.assert_allclose(delta_step, np.zeros((2,), dtype=np.float32))
    assert dual_lambda_step == pytest.approx(0.0)
    assert surrogate_relief_kw_step == pytest.approx(0.0)
    assert feasible is True


def test_single_step_identity_qp_infeasible_returns_nan_lambda():
    delta_step, dual_lambda_step, surrogate_relief_kw_step, feasible = dual_nb._solve_single_step_identity_qp(
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.5, 0.5], dtype=np.float32),
        3.0,
    )

    np.testing.assert_allclose(delta_step, np.array([1.5, 0.5], dtype=np.float32))
    assert np.isnan(dual_lambda_step)
    assert surrogate_relief_kw_step == pytest.approx(2.0)
    assert feasible is False


def test_run_dual_two_pass_step_falls_back_to_round1_when_all_backoffs_fail(monkeypatch):
    first_pass = _make_bundle(
        net_load_kw=np.array([[-3.0, -2.0]], dtype=np.float32),
        charge_kw=np.array([[0.0, 0.0]], dtype=np.float32),
        discharge_kw=np.array([[1.0, 0.5]], dtype=np.float32),
        p_max_kw=np.array([2.0], dtype=np.float32),
    )
    surrogate = dual_nb.DualTrafoSurrogate(
        trafo_limit_kw=2.0,
        alpha_netload_kw=np.array([1.0], dtype=np.float32),
        alpha_netload_window_kw=np.array([[1.0, 1.0]], dtype=np.float32),
        baseline_root_p_kw=np.array([-4.0, -3.0], dtype=np.float32),
        export_overload_mask=np.array([True, True]),
        import_overload_mask=np.array([False, False]),
    )
    allocation = dual_nb.PerStepNetLoadAllocation(
        delta_netload_max_kw=np.array([[3.0, 2.5]], dtype=np.float32),
        delta_battery_headroom_kw=np.array([[3.0, 2.5]], dtype=np.float32),
        delta_curtail_headroom_kw=np.zeros((1, 2), dtype=np.float32),
        export_deficit_kw=np.array([2.0, 1.0], dtype=np.float32),
        delta_netload_kw=np.array([[2.0, 1.0]], dtype=np.float32),
        dual_lambda=np.array([1.0, 0.5], dtype=np.float32),
        surrogate_trafo_relief_kw=np.array([2.0, 1.0], dtype=np.float32),
        net_load_floor_kw=np.array([[-1.0, -1.0]], dtype=np.float32),
        infeasible_export_mask=np.array([False, False]),
        import_only_unsupported=False,
    )
    round1_pf = dual_nb.FirstStepPFEvaluation(
        battery_power_kw=np.array([-1.0], dtype=np.float32),
        pv_curtail_kw=np.array([0.0], dtype=np.float32),
        line_loading_pct=np.array([50.0], dtype=np.float32),
        trafo_loading_pct=np.array([110.0], dtype=np.float32),
        trafo_p_signed_kw=np.array([-4.0], dtype=np.float32),
        root_import_kw=0.0,
        root_export_kw=4.0,
        max_line_loading_pct=50.0,
        max_trafo_loading_pct=110.0,
        export_over_limit_kw=2.0,
        import_over_limit_kw=0.0,
    )
    fake_env = SimpleNamespace(
        n=1,
        soc=np.array([0.5], dtype=np.float32),
        agent_c_bat=np.array([4.0], dtype=np.float32),
        agent_p_max=np.array([2.0], dtype=np.float32),
        get_signal_step=lambda name: np.array([0.0], dtype=np.float32),
    )
    cfg = SimpleNamespace(grid=SimpleNamespace(line_max_loading_pct=100.0))

    monkeypatch.setattr(dual_nb, "solve_first_pass_local_mpc_bundle", lambda env, raw_obs, **kwargs: first_pass)
    monkeypatch.setattr(dual_nb, "build_dual_trafo_surrogate", lambda cfg, env, bundle: surrogate)
    monkeypatch.setattr(dual_nb, "solve_per_step_netload_qp_allocation", lambda bundle, surrogate, **kwargs: allocation)
    monkeypatch.setattr(
        dual_nb,
        "resolve_second_pass_local_mpc_bundle",
        lambda env, bundle, net_load_floor_kw, **kwargs: dual_nb.LocalMPCBundle(
            **{**first_pass.__dict__, "feasible": np.array([False], dtype=bool), "net_load_floor_kw": np.asarray(net_load_floor_kw, dtype=np.float32)}
        ),
    )
    monkeypatch.setattr(dual_nb, "_evaluate_first_step_pf", lambda env, battery_power_kw, pv_curtail_kw, trafo_limit_kw, loading_limit_pct: round1_pf)
    monkeypatch.setattr(
        dual_nb,
        "_bundle_to_actions",
        lambda env, battery_power_kw, pv_curtail_kw: ([np.array([-0.5, 1.0], dtype=np.float32)], np.array([[-0.5, 1.0]], dtype=np.float32)),
    )

    result = dual_nb.run_dual_two_pass_step(fake_env, {"price_seq": np.zeros((2,), dtype=np.float32)}, cfg, max_bisect_iters=2)

    assert result.flexibility_insufficient is True
    assert result.beta == 0.0
    np.testing.assert_allclose(result.second_pass_bundle.signed_battery_kw, first_pass.signed_battery_kw)


def test_collect_dual_two_pass_rollout_records_dual_columns(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "dual_rollout_stub")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.env.num_agents = 1
    cfg.data.agent_profiles = [cfg.data.agent_profiles[0]]
    cfg.grid.agent_bus_ids = [cfg.grid.agent_bus_ids[0]]

    class _DummyObsBuilder:
        def __init__(self, raw_obs):
            self._raw_obs = raw_obs

        def build_raw(self, env):
            del env
            return self._raw_obs

    class _DummyEnv:
        def __init__(self):
            self.n = 1
            self.dt = 1.0
            self.eff = 1.0
            self.soc_min = 0.1
            self.soc_max = 0.9
            self.soc = np.array([0.5], dtype=np.float32)
            self.agent_c_bat = np.array([4.0], dtype=np.float32)
            self.agent_p_max = np.array([2.0], dtype=np.float32)
            self.reward_fn = SimpleNamespace(export_subsidy_eur_per_kwh=0.079)
            self._single_agent_mpc_stats = {}
            self._grid_core = SimpleNamespace(
                net=SimpleNamespace(bus=SimpleNamespace(index=np.array([10], dtype=np.int32))),
                agent_bus_ids=[10],
            )
            self.num_available_episodes = 1
            self.obs_builder = _DummyObsBuilder(
                {
                    "price_seq": np.array([0.2, 0.2], dtype=np.float32),
                    "load_seq": np.array([[1.0, 1.0]], dtype=np.float32),
                    "pv_seq": np.array([[0.0, 0.0]], dtype=np.float32),
                }
            )

        def reset(self, episode_idx=0):
            del episode_idx
            return self.obs_builder._raw_obs, {"episode_meta": {"timestamps": ["2020-01-01T00:00:00"]}}

        def step(self, actions):
            del actions
            info = {
                "episode_done": True,
                "price": 0.2,
                "load": np.array([1.0], dtype=np.float32),
                "pv": np.array([0.0], dtype=np.float32),
                "pv_raw": np.array([0.0], dtype=np.float32),
                "pv_effective": np.array([0.0], dtype=np.float32),
                "pv_curtail": np.array([0.0], dtype=np.float32),
                "pv_curtail_req": np.array([0.0], dtype=np.float32),
                "pv_utilization": np.array([1.0], dtype=np.float32),
                "base_net_load": np.array([1.0], dtype=np.float32),
                "base_net_load_effective": np.array([1.0], dtype=np.float32),
                "net_load": np.array([0.5], dtype=np.float32),
                "grid_import_kw": np.array([0.5], dtype=np.float32),
                "grid_export_kw": np.array([0.0], dtype=np.float32),
                "e_bat": np.array([-0.5], dtype=np.float32),
                "e_bat_req": np.array([-0.5], dtype=np.float32),
                "soc_next": np.array([0.4], dtype=np.float32),
                "vm_pu": np.array([1.0], dtype=np.float32),
                "line_loading_pct": np.array([30.0], dtype=np.float32),
                "trafo_loading_pct": np.array([40.0], dtype=np.float32),
                "trafo_p_signed_kw": np.array([0.5], dtype=np.float32),
                "r_safe_v": np.array([0.0], dtype=np.float32),
                "r_safe_line": np.array([0.0], dtype=np.float32),
                "r_safe_trafo": np.array([0.0], dtype=np.float32),
                "reward": np.array([0.0], dtype=np.float32),
            }
            return self.obs_builder._raw_obs, [0.0], [True], [False], info

        def close(self):
            return None

    bundle = _make_bundle(
        net_load_kw=np.array([[0.5, 0.5]], dtype=np.float32),
        charge_kw=np.array([[0.0, 0.0]], dtype=np.float32),
        discharge_kw=np.array([[0.5, 0.0]], dtype=np.float32),
        p_max_kw=np.array([2.0], dtype=np.float32),
    )
    pf_eval = dual_nb.FirstStepPFEvaluation(
        battery_power_kw=np.array([-0.5], dtype=np.float32),
        pv_curtail_kw=np.array([0.0], dtype=np.float32),
        line_loading_pct=np.array([30.0], dtype=np.float32),
        trafo_loading_pct=np.array([40.0], dtype=np.float32),
        trafo_p_signed_kw=np.array([0.5], dtype=np.float32),
        root_import_kw=0.5,
        root_export_kw=0.0,
        max_line_loading_pct=30.0,
        max_trafo_loading_pct=40.0,
        export_over_limit_kw=0.0,
        import_over_limit_kw=0.0,
    )
    surrogate = dual_nb.DualTrafoSurrogate(
        trafo_limit_kw=400.0,
        alpha_netload_kw=np.array([1.0], dtype=np.float32),
        alpha_netload_window_kw=np.array([[1.0, 1.0]], dtype=np.float32),
        baseline_root_p_kw=np.array([0.5, 0.5], dtype=np.float32),
        export_overload_mask=np.array([False, False]),
        import_overload_mask=np.array([False, False]),
    )
    allocation = dual_nb.PerStepNetLoadAllocation(
        delta_netload_max_kw=np.array([[2.5, 2.0]], dtype=np.float32),
        delta_battery_headroom_kw=np.array([[2.5, 2.0]], dtype=np.float32),
        delta_curtail_headroom_kw=np.zeros((1, 2), dtype=np.float32),
        export_deficit_kw=np.array([0.0, 0.0], dtype=np.float32),
        delta_netload_kw=np.zeros((1, 2), dtype=np.float32),
        dual_lambda=np.zeros((2,), dtype=np.float32),
        surrogate_trafo_relief_kw=np.zeros((2,), dtype=np.float32),
        net_load_floor_kw=bundle.net_load_kw.copy(),
        infeasible_export_mask=np.array([False, False]),
        import_only_unsupported=False,
    )
    action_array = np.array([[-0.25, 1.0]], dtype=np.float32)
    action_info = {
        "requested_action": action_array.copy(),
        "executed_action": action_array.copy(),
        "action_gap": np.zeros_like(action_array, dtype=np.float32),
        "controller_action_gap": np.zeros((1,), dtype=np.float32),
        "battery_action_req": np.array([-0.25], dtype=np.float32),
        "battery_action_exec": np.array([-0.25], dtype=np.float32),
        "pv_action_req": np.array([1.0], dtype=np.float32),
        "pv_action_exec": np.array([1.0], dtype=np.float32),
        "battery_power_req_kw": np.array([-0.5], dtype=np.float32),
        "battery_power_exec_kw": np.array([-0.5], dtype=np.float32),
        "pv_effective_req_kw": np.array([0.0], dtype=np.float32),
        "pv_effective_exec_kw": np.array([0.0], dtype=np.float32),
        "pv_curtail_req_kw": np.array([0.0], dtype=np.float32),
        "pv_curtail_exec_kw": np.array([0.0], dtype=np.float32),
        "soc_penalty_unweighted": np.array([0.0], dtype=np.float32),
        "action_penalty_unweighted": np.array([0.0], dtype=np.float32),
    }
    step_result = dual_nb.DualTwoPassStepResult(
        actions=[action_array[0].copy()],
        action_array=action_array.copy(),
        action_info=action_info,
        first_pass_bundle=bundle,
        second_pass_bundle=bundle,
        surrogate=surrogate,
        allocation=allocation,
        round1_pf=pf_eval,
        final_pf=pf_eval,
        last_candidate_pf=None,
        beta=0.0,
        backoff_iterations=0,
        flexibility_insufficient=False,
        import_only_unsupported=False,
    )

    monkeypatch.setattr("scripts.builder.build_env", lambda cfg, mode: _DummyEnv())
    monkeypatch.setattr(dual_nb, "_get_dual_projector", lambda cfg, env: object())
    monkeypatch.setattr(
        dual_nb,
        "run_dual_two_pass_step",
        lambda env, raw_obs, cfg, max_bisect_iters=6, show_progress=False, progress_cb=None: step_result,
    )

    rollout = dual_nb.collect_dual_two_pass_rollout(
        cfg,
        prediction_mode="perfect",
        label="dual stub",
        max_bisect_iters=2,
    )

    assert not rollout.step_df.empty
    assert "dual_beta" in rollout.step_df.columns
    assert "dual_delta_first_step_kw" in rollout.agent_df.columns
    assert rollout.meta["dual_diagnostic_log"]


def test_build_dual_window_and_summary_frames_expand_dual_payload():
    round1_bundle = _make_bundle(
        net_load_kw=np.array([[0.5, 0.5]], dtype=np.float32),
        charge_kw=np.array([[0.0, 0.0]], dtype=np.float32),
        discharge_kw=np.array([[0.5, 0.0]], dtype=np.float32),
        p_max_kw=np.array([2.0], dtype=np.float32),
        pv_seq=np.array([[2.0, 1.0]], dtype=np.float32),
        pv_curtail_kw=np.array([[0.0, 0.0]], dtype=np.float32),
    )
    round2_bundle = _make_bundle(
        net_load_kw=np.array([[1.0, 0.8]], dtype=np.float32),
        charge_kw=np.array([[0.5, 0.1]], dtype=np.float32),
        discharge_kw=np.array([[0.0, 0.0]], dtype=np.float32),
        p_max_kw=np.array([2.0], dtype=np.float32),
        pv_seq=np.array([[2.0, 1.0]], dtype=np.float32),
        pv_curtail_kw=np.array([[0.5, 0.0]], dtype=np.float32),
    )
    pf_eval = dual_nb.FirstStepPFEvaluation(
        battery_power_kw=np.array([-0.5], dtype=np.float32),
        pv_curtail_kw=np.array([0.0], dtype=np.float32),
        line_loading_pct=np.array([30.0], dtype=np.float32),
        trafo_loading_pct=np.array([40.0], dtype=np.float32),
        trafo_p_signed_kw=np.array([-3.0], dtype=np.float32),
        root_import_kw=0.0,
        root_export_kw=3.0,
        max_line_loading_pct=30.0,
        max_trafo_loading_pct=40.0,
        export_over_limit_kw=1.0,
        import_over_limit_kw=0.0,
    )
    final_pf = dual_nb.FirstStepPFEvaluation(
        battery_power_kw=np.array([1.0], dtype=np.float32),
        pv_curtail_kw=np.array([0.5], dtype=np.float32),
        line_loading_pct=np.array([35.0], dtype=np.float32),
        trafo_loading_pct=np.array([45.0], dtype=np.float32),
        trafo_p_signed_kw=np.array([-1.5], dtype=np.float32),
        root_import_kw=0.0,
        root_export_kw=1.5,
        max_line_loading_pct=35.0,
        max_trafo_loading_pct=45.0,
        export_over_limit_kw=0.0,
        import_over_limit_kw=0.0,
    )
    surrogate = dual_nb.DualTrafoSurrogate(
        trafo_limit_kw=2.0,
        alpha_netload_kw=np.array([1.0], dtype=np.float32),
        alpha_netload_window_kw=np.array([[1.0, 1.0]], dtype=np.float32),
        baseline_root_p_kw=np.array([-3.0, -2.5], dtype=np.float32),
        export_overload_mask=np.array([True, True]),
        import_overload_mask=np.array([False, False]),
    )
    allocation = dual_nb.PerStepNetLoadAllocation(
        delta_netload_max_kw=np.array([[3.0, 2.0]], dtype=np.float32),
        delta_battery_headroom_kw=np.array([[3.0, 2.0]], dtype=np.float32),
        delta_curtail_headroom_kw=np.zeros((1, 2), dtype=np.float32),
        export_deficit_kw=np.array([1.0, 0.5], dtype=np.float32),
        delta_netload_kw=np.array([[1.0, 0.5]], dtype=np.float32),
        dual_lambda=np.array([1.0, np.nan], dtype=np.float32),
        surrogate_trafo_relief_kw=np.array([1.0, 0.5], dtype=np.float32),
        net_load_floor_kw=np.array([[1.5, 1.0]], dtype=np.float32),
        infeasible_export_mask=np.array([False, True]),
        import_only_unsupported=False,
    )
    step_result = dual_nb.DualTwoPassStepResult(
        actions=[np.array([-0.5, 1.0], dtype=np.float32)],
        action_array=np.array([[-0.5, 1.0]], dtype=np.float32),
        action_info={},
        first_pass_bundle=round1_bundle,
        second_pass_bundle=round2_bundle,
        surrogate=surrogate,
        allocation=allocation,
        round1_pf=pf_eval,
        final_pf=final_pf,
        last_candidate_pf=None,
        beta=1.0,
        backoff_iterations=0,
        flexibility_insufficient=False,
        import_only_unsupported=False,
    )
    payload = step_result.to_diagnostic_payload()
    payload["controller"] = "dual stub"
    payload["episode_idx"] = 0
    payload["step"] = 3
    payload["timestamp"] = "2020-01-01T03:00:00"
    rollout = SimpleNamespace(meta={"controller": "dual stub", "dual_diagnostic_log": [payload]})

    window_df = dual_nb.build_dual_window_diagnostic_frame(rollout)
    summary_df = dual_nb.build_dual_first_step_summary_frame(rollout)

    assert list(window_df["horizon_step"]) == [0, 1]
    assert "dual_lambda" in window_df.columns
    assert "dual_surrogate_trafo_relief_kw" in window_df.columns
    assert "dual_surrogate_pf_relief_error_kw" in window_df.columns
    assert summary_df.loc[0, "dual_lambda_first_step"] == pytest.approx(1.0)
    assert summary_df.loc[0, "dual_delta_first_step_total_kw"] == pytest.approx(1.0)
    assert summary_df.loc[0, "dual_pf_trafo_relief_kw"] == pytest.approx(1.5)
    assert summary_df.loc[0, "dual_surrogate_pf_relief_error_kw"] == pytest.approx(-0.5)
    assert summary_df.loc[0, "dual_round1_pv_utilization_first_step"] == pytest.approx(1.0)
    assert summary_df.loc[0, "dual_round2_pv_utilization_first_step"] == pytest.approx(0.75)


def test_collect_dual_two_pass_rollout_progress_callback_order(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "dual_progress_stub")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.env.num_agents = 1
    cfg.data.agent_profiles = [cfg.data.agent_profiles[0]]
    cfg.grid.agent_bus_ids = [cfg.grid.agent_bus_ids[0]]

    class _DummyObsBuilder:
        def __init__(self, raw_obs):
            self._raw_obs = raw_obs

        def build_raw(self, env):
            del env
            return self._raw_obs

    class _DummyEnv:
        def __init__(self):
            self.n = 1
            self.dt = 1.0
            self.eff = 1.0
            self.soc_min = 0.1
            self.soc_max = 0.9
            self.soc = np.array([0.5], dtype=np.float32)
            self.agent_c_bat = np.array([4.0], dtype=np.float32)
            self.agent_p_max = np.array([2.0], dtype=np.float32)
            self.reward_fn = SimpleNamespace(export_subsidy_eur_per_kwh=0.079)
            self._single_agent_mpc_stats = {}
            self._grid_core = SimpleNamespace(
                net=SimpleNamespace(bus=SimpleNamespace(index=np.array([10], dtype=np.int32))),
                agent_bus_ids=[10],
            )
            self.num_available_episodes = 1
            self.obs_builder = _DummyObsBuilder(
                {
                    "price_seq": np.array([0.2, 0.2], dtype=np.float32),
                    "load_seq": np.array([[1.0, 1.0]], dtype=np.float32),
                    "pv_seq": np.array([[0.0, 0.0]], dtype=np.float32),
                }
            )

        def reset(self, episode_idx=0):
            del episode_idx
            return self.obs_builder._raw_obs, {"episode_meta": {"timestamps": ["2020-01-01T00:00:00"]}}

        def step(self, actions):
            del actions
            info = {
                "episode_done": True,
                "price": 0.2,
                "load": np.array([1.0], dtype=np.float32),
                "pv": np.array([0.0], dtype=np.float32),
                "pv_raw": np.array([0.0], dtype=np.float32),
                "pv_effective": np.array([0.0], dtype=np.float32),
                "pv_curtail": np.array([0.0], dtype=np.float32),
                "pv_curtail_req": np.array([0.0], dtype=np.float32),
                "pv_utilization": np.array([1.0], dtype=np.float32),
                "base_net_load": np.array([1.0], dtype=np.float32),
                "base_net_load_effective": np.array([1.0], dtype=np.float32),
                "net_load": np.array([0.5], dtype=np.float32),
                "grid_import_kw": np.array([0.5], dtype=np.float32),
                "grid_export_kw": np.array([0.0], dtype=np.float32),
                "e_bat": np.array([-0.5], dtype=np.float32),
                "e_bat_req": np.array([-0.5], dtype=np.float32),
                "soc_next": np.array([0.4], dtype=np.float32),
                "vm_pu": np.array([1.0], dtype=np.float32),
                "line_loading_pct": np.array([30.0], dtype=np.float32),
                "trafo_loading_pct": np.array([40.0], dtype=np.float32),
                "trafo_p_signed_kw": np.array([0.5], dtype=np.float32),
                "r_safe_v": np.array([0.0], dtype=np.float32),
                "r_safe_line": np.array([0.0], dtype=np.float32),
                "r_safe_trafo": np.array([0.0], dtype=np.float32),
                "reward": np.array([0.0], dtype=np.float32),
            }
            return self.obs_builder._raw_obs, [0.0], [True], [False], info

        def close(self):
            return None

    bundle = _make_bundle(
        net_load_kw=np.array([[0.5, 0.5]], dtype=np.float32),
        charge_kw=np.array([[0.0, 0.0]], dtype=np.float32),
        discharge_kw=np.array([[0.5, 0.0]], dtype=np.float32),
        p_max_kw=np.array([2.0], dtype=np.float32),
    )
    pf_eval = dual_nb.FirstStepPFEvaluation(
        battery_power_kw=np.array([-0.5], dtype=np.float32),
        pv_curtail_kw=np.array([0.0], dtype=np.float32),
        line_loading_pct=np.array([30.0], dtype=np.float32),
        trafo_loading_pct=np.array([40.0], dtype=np.float32),
        trafo_p_signed_kw=np.array([0.5], dtype=np.float32),
        root_import_kw=0.5,
        root_export_kw=0.0,
        max_line_loading_pct=30.0,
        max_trafo_loading_pct=40.0,
        export_over_limit_kw=0.0,
        import_over_limit_kw=0.0,
    )
    surrogate = dual_nb.DualTrafoSurrogate(
        trafo_limit_kw=400.0,
        alpha_netload_kw=np.array([1.0], dtype=np.float32),
        alpha_netload_window_kw=np.array([[1.0, 1.0]], dtype=np.float32),
        baseline_root_p_kw=np.array([0.5, 0.5], dtype=np.float32),
        export_overload_mask=np.array([False, False]),
        import_overload_mask=np.array([False, False]),
    )
    allocation = dual_nb.PerStepNetLoadAllocation(
        delta_netload_max_kw=np.array([[2.5, 2.0]], dtype=np.float32),
        delta_battery_headroom_kw=np.array([[2.5, 2.0]], dtype=np.float32),
        delta_curtail_headroom_kw=np.zeros((1, 2), dtype=np.float32),
        export_deficit_kw=np.array([0.0, 0.0], dtype=np.float32),
        delta_netload_kw=np.zeros((1, 2), dtype=np.float32),
        dual_lambda=np.zeros((2,), dtype=np.float32),
        surrogate_trafo_relief_kw=np.zeros((2,), dtype=np.float32),
        net_load_floor_kw=bundle.net_load_kw.copy(),
        infeasible_export_mask=np.array([False, False]),
        import_only_unsupported=False,
    )
    step_result = dual_nb.DualTwoPassStepResult(
        actions=[np.array([-0.25, 1.0], dtype=np.float32)],
        action_array=np.array([[-0.25, 1.0]], dtype=np.float32),
        action_info={
            "requested_action": np.array([[-0.25, 1.0]], dtype=np.float32),
            "executed_action": np.array([[-0.25, 1.0]], dtype=np.float32),
            "action_gap": np.zeros((1, 2), dtype=np.float32),
            "controller_action_gap": np.zeros((1,), dtype=np.float32),
            "battery_action_req": np.array([-0.25], dtype=np.float32),
            "battery_action_exec": np.array([-0.25], dtype=np.float32),
            "pv_action_req": np.array([1.0], dtype=np.float32),
            "pv_action_exec": np.array([1.0], dtype=np.float32),
            "battery_power_req_kw": np.array([-0.5], dtype=np.float32),
            "battery_power_exec_kw": np.array([-0.5], dtype=np.float32),
            "pv_effective_req_kw": np.array([0.0], dtype=np.float32),
            "pv_effective_exec_kw": np.array([0.0], dtype=np.float32),
            "pv_curtail_req_kw": np.array([0.0], dtype=np.float32),
            "pv_curtail_exec_kw": np.array([0.0], dtype=np.float32),
            "soc_penalty_unweighted": np.array([0.0], dtype=np.float32),
            "action_penalty_unweighted": np.array([0.0], dtype=np.float32),
        },
        first_pass_bundle=bundle,
        second_pass_bundle=bundle,
        surrogate=surrogate,
        allocation=allocation,
        round1_pf=pf_eval,
        final_pf=pf_eval,
        last_candidate_pf=None,
        beta=0.0,
        backoff_iterations=0,
        flexibility_insufficient=False,
        import_only_unsupported=False,
    )

    events: list[str] = []

    def _stub_run_dual_two_pass_step(env, raw_obs, cfg, max_bisect_iters=6, show_progress=False, progress_cb=None):
        del env, raw_obs, cfg, max_bisect_iters, show_progress
        assert progress_cb is not None
        progress_cb("round1", {"agent_idx": 1, "total_agents": 1})
        progress_cb("dual", {"horizon_step": 1, "total_horizon_steps": 2})
        progress_cb("round2", {"agent_idx": 1, "total_agents": 1})
        return step_result

    monkeypatch.setattr("scripts.builder.build_env", lambda cfg, mode: _DummyEnv())
    monkeypatch.setattr(dual_nb, "_get_dual_projector", lambda cfg, env: object())
    monkeypatch.setattr(dual_nb, "run_dual_two_pass_step", _stub_run_dual_two_pass_step)

    def _progress_cb(stage: str, payload):
        del payload
        events.append(stage)

    dual_nb.collect_dual_two_pass_rollout(
        cfg,
        prediction_mode="perfect",
        label="dual progress stub",
        max_bisect_iters=2,
        show_progress=False,
        progress_cb=_progress_cb,
    )

    assert events == ["step_start", "round1", "dual", "round2", "step_done"]
