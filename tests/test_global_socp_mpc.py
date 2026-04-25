import importlib.util

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from controllers.mpc.global_socp_mpc import (
    FullHorizonProblemInput,
    GlobalMISOCPProblem,
    GlobalSOCPMPCController,
    MISOCPResult,
    ModelSize,
    NetworkModel,
    _compute_agent_net_grid_mw,
)
from scripts.builder import build_env
from tests.support.helpers import make_case_dir, make_smoke_config


HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


def _has_working_gurobi_license() -> bool:
    if not HAS_GUROBI:
        return False
    try:
        import gurobipy as gp

        model = gp.Model("global_misocp_test")
        model.Params.OutputFlag = 0
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
        return True
    except Exception:
        return False


HAS_WORKING_GUROBI_LICENSE = _has_working_gurobi_license()


def _make_global_mpc_cfg(tmp_path, label: str):
    case_dir = make_case_dir(tmp_path, label)
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.forecast.type = "perfect"
    cfg.forecast.target_signals = ["wholesale_price", "load", "pv"]
    cfg.obs.sequence_features = ["wholesale_price", "load", "pv"]
    return cfg


def _no_solution_result() -> MISOCPResult:
    return MISOCPResult(
        status_code=3,
        status_label="infeasible",
        has_solution=False,
        time_limit_feasible=False,
        solve_time_sec=0.25,
        mip_gap=np.nan,
        best_bound=np.nan,
        objective_value=np.nan,
        agent_purchase_cost_eur=np.nan,
        agent_export_subsidy_eur=np.nan,
        agent_net_cost_eur=np.nan,
        feeder_purchase_cost_eur=np.nan,
        feeder_export_subsidy_eur=np.nan,
        feeder_net_cost_eur=np.nan,
        throughput_regularization_eur=np.nan,
        throughput_regularization_weight=1e-4,
        physical_tiebreaker_eur=np.nan,
        physical_tiebreaker_weight=1e-6,
        model_size=ModelSize(0, 0, 0, 0),
        agent_net_grid_mw=None,
        agent_import_mw=None,
        agent_export_mw=None,
        battery_charge_mw=None,
        battery_discharge_mw=None,
        pv_curtail_mw=None,
        energy_mwh=None,
        branch_p_pu=None,
        branch_q_pu=None,
        branch_i2_pu=None,
        bus_v_sq=None,
        root_import_mw=None,
        root_export_mw=None,
        root_p_kw=None,
        root_q_kvar=None,
        bus_vm_pu=None,
        line_loading_pct=None,
        trafo_loading_pct=None,
        simultaneous_charge_discharge_kw=None,
        simultaneous_agent_steps=0,
        simultaneous_step_ratio=0.0,
        max_simultaneous_kw=0.0,
        sol_count=0,
        node_count=0.0,
        iter_count=0.0,
        bar_iter_count=0.0,
    )


def _mock_window_input(problem: GlobalMISOCPProblem, horizon_steps: int) -> FullHorizonProblemInput:
    return FullHorizonProblemInput(
        wholesale_price_seq=np.zeros((horizon_steps,), dtype=np.float32),
        import_price_seq=np.zeros((horizon_steps,), dtype=np.float32),
        load_seq=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        pv_seq=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        soc_init=np.full((problem.n_agents,), 0.5, dtype=np.float32),
        timestamps=tuple(f"2020-01-01T00:{step_idx:02d}:00" for step_idx in range(horizon_steps)),
        episode_offsets=np.asarray([0], dtype=np.int32),
        episode_lengths=np.asarray([horizon_steps], dtype=np.int32),
        episode_indices=np.asarray([0], dtype=np.int32),
    )


def test_network_model_extracts_tree_and_agent_reactive_background_is_zeroed(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_network")
    cfg.grid.agent_bus_ids = [10, 4]

    network = NetworkModel.from_cfg(cfg, agent_bus_ids=cfg.grid.agent_bus_ids)

    assert network.root_bus_id not in cfg.grid.agent_bus_ids
    assert len(network.branch_parent_pos) == len(network.bus_ids) - 1
    assert bool(network.branch_is_trafo[network.trafo_branch_index])
    assert int(network.root_outgoing_branches.size) == 1
    assert int(network.root_outgoing_branches[0]) == int(network.trafo_branch_index)

    assert np.allclose(network.p_base_mw, 0.0)
    assert np.allclose(network.q_base_mvar, 0.0)


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_global_misocp_problem_uses_only_grid_binaries_and_returns_solution(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_problem")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        obs, _ = env.reset(episode_idx=0)
        raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
        result = problem.solve(
            import_price_seq=problem.apply_import_price_markup(raw_obs["wholesale_price_seq"]),
            load_seq=raw_obs["load_seq"],
            pv_seq=raw_obs["pv_seq"],
            soc_init=np.asarray(env.soc, dtype=np.float32),
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
        )

        assert problem.uses_nonconvex is False
        assert result.has_solution
        assert result.agent_import_mw is not None
        assert result.agent_export_mw is not None
        assert result.agent_net_grid_mw is not None
        assert result.agent_net_cost_eur == pytest.approx(
            result.agent_purchase_cost_eur - result.agent_export_subsidy_eur
        )
        expected_storage_charge_cost = float(
            np.sum(1e3 * env.dt * problem.apply_import_price_markup(raw_obs["wholesale_price_seq"]).reshape(1, -1) * result.battery_charge_mw)
        )
        expected_storage_discharge_revenue = float(
            np.sum(1e3 * env.dt * problem.apply_import_price_markup(raw_obs["wholesale_price_seq"]).reshape(1, -1) * result.battery_discharge_mw)
        )
        assert result.storage_charge_cost_eur == pytest.approx(expected_storage_charge_cost)
        assert result.storage_discharge_revenue_eur == pytest.approx(expected_storage_discharge_revenue)
        assert result.storage_profit_eur == pytest.approx(
            result.storage_discharge_revenue_eur - result.storage_charge_cost_eur
        )
        assert not hasattr(result, "storage_purchase_cost_eur")
        assert not hasattr(result, "storage_sale_revenue_eur")
        assert not hasattr(result, "storage_total_profit_eur")
        assert result.total_eur == pytest.approx(result.storage_profit_eur - result.system_other_cost_eur)
        assert result.storage_objective_eur == pytest.approx(-result.storage_profit_eur)
        assert result.stage1_primary_objective_eur == pytest.approx(
            result.storage_objective_eur + result.throughput_regularization_eur,
            abs=1e-5,
        )
        assert result.model_size.num_binary_vars == problem.horizon
        assert result.model_size.num_quadratic_constraints > 0
        assert result.throughput_regularization_weight == pytest.approx(1e-4)
        assert result.physical_tiebreaker_weight == pytest.approx(
            cfg.mpc.branch_current_tiebreaker_eur_per_pu_step
        )
        assert result.physical_tiebreaker_eur >= 0.0
        assert result.physical_tiebreaker_eur / max(abs(result.agent_net_cost_eur), 1.0) < 1e-3
        assert result.sol_count >= 1
        assert result.node_count >= 0.0
        assert result.iter_count >= 0.0
    finally:
        env.close()


def test_build_full_horizon_input_stitches_all_test_episodes(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_full_input")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = problem.build_full_horizon_input(env)
        raw_episode = env._dataset.get_episode(0)
        raw_wholesale_price_seq = np.asarray(raw_episode["signals"]["wholesale_price"], dtype=np.float32)

        assert isinstance(full_input, FullHorizonProblemInput)
        assert full_input.horizon_steps > problem.horizon
        assert full_input.load_seq.shape == (env.n, full_input.horizon_steps)
        assert full_input.pv_seq.shape == (env.n, full_input.horizon_steps)
        expected_offsets = tuple(range(0, env.num_available_episodes * cfg.env.episode_limit, cfg.env.episode_limit))
        expected_lengths = tuple([cfg.env.episode_limit] * env.num_available_episodes)
        assert tuple(full_input.episode_offsets.tolist()) == expected_offsets
        assert tuple(full_input.episode_lengths.tolist()) == expected_lengths
        np.testing.assert_allclose(
            full_input.wholesale_price_seq[: raw_wholesale_price_seq.size],
            raw_wholesale_price_seq,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            full_input.import_price_seq[: raw_wholesale_price_seq.size],
            raw_wholesale_price_seq + np.float32(cfg.reward.import_price_markup_eur_per_kwh),
            rtol=1e-6,
        )
    finally:
        env.close()


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_storage_profit_objective_allows_prices_below_export_subsidy(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_storage_profit_price_guard")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=2)
        result = problem.solve_full_horizon(
            import_price_seq=np.asarray([0.05, 0.06], dtype=np.float32),
            load_seq=full_input.load_seq,
            pv_seq=full_input.pv_seq,
            soc_init=full_input.soc_init,
            export_subsidy=0.079,
            timestamps=full_input.timestamps,
            episode_offsets=full_input.episode_offsets,
            episode_lengths=full_input.episode_lengths,
            time_limit_sec=5.0,
        )

        assert result.has_solution
        assert np.isfinite(result.storage_profit_eur)
    finally:
        env.close()


def test_compute_agent_net_grid_mw_uses_pv_raw_and_curtailment_once():
    load_seq = np.asarray([[10.0, 10.0]], dtype=np.float32)
    pv_raw_seq = np.asarray([[4.0, 4.0]], dtype=np.float32)
    battery_charge_mw = np.asarray([[0.001, 0.001]], dtype=np.float32)
    battery_discharge_mw = np.asarray([[0.0, 0.0]], dtype=np.float32)
    pv_curtail_mw = np.asarray([[0.0, 0.004]], dtype=np.float32)

    agent_net_grid_mw = _compute_agent_net_grid_mw(
        load_seq,
        pv_raw_seq,
        battery_charge_mw,
        battery_discharge_mw,
        pv_curtail_mw,
    )

    assert agent_net_grid_mw[0, 0] == pytest.approx((10.0 - 4.0) / 1000.0 + 0.001)
    assert agent_net_grid_mw[0, 1] == pytest.approx(10.0 / 1000.0 + 0.001)


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_global_misocp_full_horizon_solve_returns_long_plan(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_full_solve")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = problem.build_full_horizon_input(env)
        result = problem.solve_full_horizon(
            import_price_seq=full_input.import_price_seq,
            load_seq=full_input.load_seq,
            pv_seq=full_input.pv_seq,
            soc_init=full_input.soc_init,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            timestamps=full_input.timestamps,
            episode_offsets=full_input.episode_offsets,
            episode_lengths=full_input.episode_lengths,
            time_limit_sec=30.0,
        )

        assert result.has_solution
        assert result.solve_mode == "single_window"
        assert result.horizon_steps == full_input.horizon_steps
        assert result.episode_offsets is not None
        assert tuple(result.episode_offsets.tolist()) == tuple(full_input.episode_offsets.tolist())
        assert result.sol_count >= 1
    finally:
        env.close()
@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_global_misocp_controller_returns_bounded_actions_and_diagnostics(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_actions")
    env = build_env(cfg, mode="test")
    try:
        controller = GlobalSOCPMPCController(env, cfg)
        obs, _ = env.reset(episode_idx=0)
        actions = controller.act(obs)

        assert len(actions) == env.n
        for action in actions:
            action_array = np.asarray(action, dtype=np.float32)
            assert action_array.shape == (2,)
            assert np.all(action_array >= -1.0 - 1e-6)
            assert np.all(action_array <= 1.0 + 1e-6)

        assert controller.last_action_info is not None
        assert float(controller.last_action_info["misocp_fallback"]) == 0.0
        assert "solve_time_sec" in controller.last_action_info
        assert controller.last_diagnostic is not None
        assert controller.last_diagnostic["solver_type"] == "gurobi_misocp"
        assert controller.last_diagnostic["model_size_num_binary_vars"] == pytest.approx(controller.horizon)
    finally:
        env.close()


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_global_misocp_root_trade_is_exclusive_and_simultaneous_metrics_are_small(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_exclusive")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        obs, _ = env.reset(episode_idx=0)
        raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
        result = problem.solve(
            import_price_seq=problem.apply_import_price_markup(raw_obs["wholesale_price_seq"]),
            load_seq=raw_obs["load_seq"],
            pv_seq=raw_obs["pv_seq"],
            soc_init=np.asarray(env.soc, dtype=np.float32),
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
        )

        assert result.has_solution
        assert result.root_import_mw is not None and result.root_export_mw is not None
        assert not np.any((result.root_import_mw > 1e-6) & (result.root_export_mw > 1e-6))
        assert result.simultaneous_step_ratio <= 0.05
    finally:
        env.close()


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_global_misocp_enforces_terminal_soc_target(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_terminal_soc")
    cfg.env.soc_target = 0.7
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        obs, _ = env.reset(episode_idx=0)
        raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
        result = problem.solve(
            import_price_seq=problem.apply_import_price_markup(raw_obs["wholesale_price_seq"]),
            load_seq=raw_obs["load_seq"],
            pv_seq=raw_obs["pv_seq"],
            soc_init=np.asarray(env.soc, dtype=np.float32),
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
        )

        assert result.has_solution
        expected_target = cfg.env.soc_target * (np.asarray(env.agent_c_bat, dtype=np.float32) / 1000.0)
        assert result.energy_mwh is not None
        assert np.all(np.asarray(result.energy_mwh[:, -1], dtype=np.float32) >= expected_target - 1e-5)
    finally:
        env.close()


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_global_misocp_infeasible_case_returns_no_solution(tmp_path):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_infeasible")
    cfg.env.soc_target = 0.95
    env = build_env(cfg, mode="test")
    try:
        env.agent_p_max = np.zeros_like(np.asarray(env.agent_p_max, dtype=np.float32))
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        obs, _ = env.reset(episode_idx=0)
        raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
        result = problem.solve(
            import_price_seq=problem.apply_import_price_markup(raw_obs["wholesale_price_seq"]),
            load_seq=raw_obs["load_seq"],
            pv_seq=raw_obs["pv_seq"],
            soc_init=np.asarray(env.soc, dtype=np.float32),
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
        )

        assert not result.has_solution
        assert result.sol_count == 0
    finally:
        env.close()


def test_global_misocp_fallback_returns_noop(tmp_path, monkeypatch):
    cfg = _make_global_mpc_cfg(tmp_path, "global_misocp_fallback")
    env = build_env(cfg, mode="test")
    try:
        controller = GlobalSOCPMPCController(env, cfg)
        obs, _ = env.reset(episode_idx=0)
        monkeypatch.setattr(controller, "_solve_step", lambda _raw_obs: _no_solution_result())

        actions = controller.act(obs)

        assert controller.last_action_info is not None
        assert float(controller.last_action_info["misocp_fallback"]) == 1.0
        for action in actions:
            assert np.allclose(np.asarray(action, dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32))
    finally:
        env.close()
