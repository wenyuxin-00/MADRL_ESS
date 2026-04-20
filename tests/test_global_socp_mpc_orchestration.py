from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from controllers.mpc.global_socp_mpc import (
    FullHorizonProblemInput,
    GlobalMISOCPProblem,
    GurobiSolveConfig,
    MISOCPResult,
    ModelSize,
)
from scripts.builder import build_env
from tests.support.helpers import make_case_dir, make_smoke_config


def _make_cfg(tmp_path, label: str):
    case_dir = make_case_dir(tmp_path, label)
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.forecast.type = "perfect"
    cfg.forecast.target_signals = ["wholesale_price", "load", "pv"]
    cfg.mpc.physics_refinement_mode = "none"
    return cfg


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
        debug_artifacts={},
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
        horizon_steps=0,
        solve_mode="single_window",
    )


def _solution_result(problem: GlobalMISOCPProblem, horizon_steps: int, *, solve_mode: str) -> MISOCPResult:
    n_lines = int(np.asarray(problem.network.line_branch_indices, dtype=np.int32).size)
    energy = np.repeat((problem.capacity_mwh * 0.5).astype(np.float32)[:, None], horizon_steps + 1, axis=1)
    return MISOCPResult(
        status_code=2,
        status_label="optimal",
        has_solution=True,
        time_limit_feasible=False,
        solve_time_sec=1.0,
        mip_gap=0.0,
        best_bound=0.0,
        objective_value=0.0,
        agent_purchase_cost_eur=0.0,
        agent_export_subsidy_eur=0.0,
        agent_net_cost_eur=0.0,
        feeder_purchase_cost_eur=0.0,
        feeder_export_subsidy_eur=0.0,
        feeder_net_cost_eur=0.0,
        throughput_regularization_eur=0.0,
        throughput_regularization_weight=1e-4,
        physical_tiebreaker_eur=0.01 * horizon_steps,
        physical_tiebreaker_weight=float(problem.branch_current_tiebreaker_eur_per_pu_step),
        model_size=ModelSize(
            num_vars=1,
            num_binary_vars=horizon_steps,
            num_linear_constraints=1,
            num_quadratic_constraints=1,
        ),
        debug_artifacts={},
        agent_net_grid_mw=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        agent_import_mw=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        agent_export_mw=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        battery_charge_mw=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        battery_discharge_mw=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        pv_curtail_mw=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        energy_mwh=energy,
        branch_p_pu=np.zeros((problem.n_branches, horizon_steps), dtype=np.float32),
        branch_q_pu=np.zeros((problem.n_branches, horizon_steps), dtype=np.float32),
        branch_i2_pu=np.zeros((problem.n_branches, horizon_steps), dtype=np.float32),
        bus_v_sq=np.ones((problem.n_buses, horizon_steps), dtype=np.float32),
        root_import_mw=np.zeros((horizon_steps,), dtype=np.float32),
        root_export_mw=np.zeros((horizon_steps,), dtype=np.float32),
        root_p_kw=np.zeros((horizon_steps,), dtype=np.float32),
        root_q_kvar=np.zeros((horizon_steps,), dtype=np.float32),
        bus_vm_pu=np.ones((problem.n_buses, horizon_steps), dtype=np.float32),
        line_loading_pct=np.zeros((n_lines, horizon_steps), dtype=np.float32),
        trafo_loading_pct=np.zeros((1, horizon_steps), dtype=np.float32),
        simultaneous_charge_discharge_kw=np.zeros((problem.n_agents, horizon_steps), dtype=np.float32),
        simultaneous_agent_steps=0,
        simultaneous_step_ratio=0.0,
        max_simultaneous_kw=0.0,
        sol_count=1,
        node_count=0.0,
        iter_count=0.0,
        bar_iter_count=0.0,
        horizon_steps=horizon_steps,
        solve_mode=solve_mode,
    )


def test_solve_adaptive_full_horizon_keeps_single_window_when_primary_has_incumbent(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_single_nogurobi")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=8)
        call_log: list[dict[str, object]] = []

        def _fake_solve_sequences(*, import_price_seq, solve_config, solve_mode, **kwargs):
            call_log.append(
                {
                    "horizon_steps": int(np.asarray(import_price_seq).shape[0]),
                    "solve_mode": str(solve_mode),
                    "cuts": solve_config.cuts,
                    "heuristics": solve_config.heuristics,
                }
            )
            return _solution_result(problem, int(np.asarray(import_price_seq).shape[0]), solve_mode=str(solve_mode))

        monkeypatch.setattr(problem, "_solve_sequences", _fake_solve_sequences)

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert result.solve_mode == "single_window"
        assert result.has_solution
        assert result.no_retry_or_fallback_used is True
        assert result.chunk_retry_count == 0
        assert result.physical_tiebreaker_eur == pytest.approx(0.08)
        assert len(call_log) == 1
        assert call_log[0]["solve_mode"] == "single_window"
        assert call_log[0]["cuts"] == 2
    finally:
        env.close()


def test_solve_adaptive_full_horizon_falls_back_to_chunked_window_and_uses_retry_profile(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_chunked_nogurobi")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=5)
        call_log: list[dict[str, object]] = []
        scripted_outcomes = iter(
            [
                _no_solution_result(),
                _no_solution_result(),
                _no_solution_result(),
                _solution_result(problem, 2, solve_mode="chunked_window"),
                _solution_result(problem, 2, solve_mode="chunked_window"),
                _solution_result(problem, 1, solve_mode="chunked_window"),
            ]
        )

        def _fake_solve_sequences(*, import_price_seq, solve_config, solve_mode, **kwargs):
            horizon_steps = int(np.asarray(import_price_seq).shape[0])
            call_log.append(
                {
                    "horizon_steps": horizon_steps,
                    "solve_mode": str(solve_mode),
                    "cuts": solve_config.cuts,
                    "heuristics": solve_config.heuristics,
                }
            )
            result = next(scripted_outcomes)
            result.horizon_steps = horizon_steps
            result.solve_mode = str(solve_mode)
            return result

        monkeypatch.setattr(problem, "_solve_sequences", _fake_solve_sequences)

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert result.solve_mode == "chunked_window"
        assert result.has_solution
        assert result.no_retry_or_fallback_used is False
        assert result.chunk_retry_count == 1
        assert result.physical_tiebreaker_eur == pytest.approx(0.05)
        assert result.chunk_summaries is not None
        assert len(result.chunk_summaries) == 3
        assert result.chunk_summaries[0]["attempt_used"] == "retry"
        assert [entry["solve_mode"] for entry in call_log[:2]] == ["single_window", "single_window"]
        assert call_log[0]["cuts"] == 2
        assert call_log[1]["cuts"] == 1
        assert call_log[2]["solve_mode"] == "chunked_window"
    finally:
        env.close()


def test_solve_adaptive_full_horizon_runs_floor_check_and_stage2_refinement(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_refine_nogurobi")
    cfg.mpc.physics_refinement_mode = "two_stage_min_branch_l"
    cfg.mpc.physics_refinement_time_limit_sec = 20.0
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=4)
        call_log: list[dict[str, object]] = []

        stage1_result = _solution_result(problem, 4, solve_mode="single_window")
        floor_result = replace(
            _solution_result(problem, 4, solve_mode="single_window"),
            agent_net_cost_eur=6.0,
            throughput_regularization_eur=0.0,
            branch_i2_pu=np.full((problem.n_branches, 4), 0.5, dtype=np.float32),
        )
        stage2_result = _solution_result(problem, 4, solve_mode="single_window")
        stage2_result.agent_net_cost_eur = 0.5
        stage2_result.branch_i2_pu = np.full((problem.n_branches, 4), 0.505, dtype=np.float32)
        scripted_outcomes = iter([stage1_result, floor_result, stage2_result])

        def _fake_solve_sequences(*, objective_mode="primary", partial_mip_start=None, strict_mip_start=False, **kwargs):
            call_log.append(
                {
                    "objective_mode": str(objective_mode),
                    "has_start": partial_mip_start is not None,
                    "strict_mip_start": bool(strict_mip_start),
                    "start_keys": tuple(sorted((partial_mip_start or {}).keys())),
                }
            )
            return next(scripted_outcomes)

        monkeypatch.setattr(problem, "_solve_sequences", _fake_solve_sequences)
        monkeypatch.setattr(problem, "_soc_relaxation_summary_from_result", lambda result: (0.0, 0.0))
        monkeypatch.setattr(problem, "_solver_feeder_gap_summary_from_result", lambda result, **kwargs: (0.0, 0.0))

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert [entry["objective_mode"] for entry in call_log] == ["primary", "min_branch_l", "min_branch_l"]
        assert call_log[2]["has_start"] is True
        assert call_log[2]["strict_mip_start"] is True
        assert {"agent_abs_grid", "branch_l", "branch_p", "branch_q", "bus_v", "energy", "p_charge", "p_discharge", "p_export", "p_import", "pv_curtail", "u_grid"}.issubset(call_log[2]["start_keys"])
        assert result.physics_refinement_mode == "two_stage_min_branch_l"
        assert result.physics_refinement_status == "refined"
        assert result.stage1_primary_objective_eur == pytest.approx(0.0)
        assert result.stage2_objective_slack_eur == pytest.approx(2.0)
        assert result.physics_refinement_slack_cap_eur == pytest.approx(2.0)
        assert result.initial_physics_refinement_slack_cap_eur == pytest.approx(2.0)
        assert result.floor_p95_soc_slack == pytest.approx(0.0)
        assert result.floor_mean_abs_solver_feeder_gap_kw == pytest.approx(0.0)
        assert result.stage2_branch_l_objective == pytest.approx(float(np.sum(stage2_result.branch_i2_pu)))
        assert result.used_physics_refinement_tier == 0
        assert result.total_tiers_configured == 2
        assert result.physics_refinement_attempt_count == 1
        assert result.physics_refinement_attempt_caps_eur == [2.0]
        assert result.returned_solution_source == "stage2"
        assert result.refinement_status_counts == {"refined": 1}
    finally:
        env.close()


def test_solve_adaptive_full_horizon_accepts_floor_when_within_slack_cap(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_floor_accept_nogurobi")
    cfg.mpc.physics_refinement_mode = "two_stage_min_branch_l"
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=4)
        call_log: list[str] = []

        stage1_result = _solution_result(problem, 4, solve_mode="single_window")
        floor_result = replace(
            _solution_result(problem, 4, solve_mode="single_window"),
            agent_net_cost_eur=3.0,
            throughput_regularization_eur=0.0,
            branch_i2_pu=np.full((problem.n_branches, 4), 0.25, dtype=np.float32),
        )
        scripted_outcomes = iter([stage1_result, floor_result])

        def _fake_solve_sequences(*, objective_mode="primary", **kwargs):
            call_log.append(str(objective_mode))
            return next(scripted_outcomes)

        monkeypatch.setattr(problem, "_solve_sequences", _fake_solve_sequences)
        monkeypatch.setattr(problem, "_soc_relaxation_summary_from_result", lambda result: (0.0, 0.0))
        monkeypatch.setattr(problem, "_solver_feeder_gap_summary_from_result", lambda result, **kwargs: (0.0, 0.0))

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert call_log == ["primary", "min_branch_l"]
        assert result.physics_refinement_status == "floor_accepted"
        assert result.returned_solution_source == "floor"
        assert result.returned_primary_objective_eur == pytest.approx(3.0)
        assert result.floor_primary_delta_signed_eur == pytest.approx(3.0)
        assert result.floor_primary_delta_positive_eur == pytest.approx(3.0)
        assert result.physics_refinement_slack_cap_eur == pytest.approx(5.0)
        assert result.floor_accepted_tier == 1
        assert result.used_physics_refinement_tier == 1
        assert result.total_tiers_configured == 2
        assert result.refinement_status_counts == {"floor_accepted": 1}
    finally:
        env.close()


def test_solve_adaptive_full_horizon_accepts_floor_at_aggressive_third_tier_when_enabled(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_floor_accept_tier3_nogurobi")
    cfg.mpc.physics_refinement_mode = "two_stage_min_branch_l"
    cfg.mpc.physics_refinement_enable_aggressive_third_tier = True
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=4)
        call_log: list[str] = []

        stage1_result = _solution_result(problem, 4, solve_mode="single_window")
        floor_result = replace(
            _solution_result(problem, 4, solve_mode="single_window"),
            agent_net_cost_eur=7.0,
            throughput_regularization_eur=0.0,
            branch_i2_pu=np.full((problem.n_branches, 4), 0.2, dtype=np.float32),
        )
        scripted_outcomes = iter([stage1_result, floor_result])

        def _fake_solve_sequences(*, objective_mode="primary", **kwargs):
            call_log.append(str(objective_mode))
            return next(scripted_outcomes)

        monkeypatch.setattr(problem, "_solve_sequences", _fake_solve_sequences)
        monkeypatch.setattr(problem, "_soc_relaxation_summary_from_result", lambda result: (0.0, 0.0))
        monkeypatch.setattr(problem, "_solver_feeder_gap_summary_from_result", lambda result, **kwargs: (0.0, 0.0))

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert call_log == ["primary", "min_branch_l"]
        assert result.physics_refinement_status == "floor_accepted"
        assert result.floor_accepted_tier == 2
        assert result.used_physics_refinement_tier == 2
        assert result.total_tiers_configured == 3
        assert result.returned_primary_objective_eur == pytest.approx(7.0)
        assert result.physics_refinement_slack_cap_eur == pytest.approx(10.0)
    finally:
        env.close()


def test_solve_adaptive_full_horizon_marks_formulation_tightening_when_floor_gate_fails(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_floor_fail_nogurobi")
    cfg.mpc.physics_refinement_mode = "two_stage_min_branch_l"
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=4)

        stage1_result = _solution_result(problem, 4, solve_mode="single_window")
        floor_result = _solution_result(problem, 4, solve_mode="single_window")
        scripted_outcomes = iter([stage1_result, floor_result])

        monkeypatch.setattr(problem, "_solve_sequences", lambda **kwargs: next(scripted_outcomes))
        monkeypatch.setattr(problem, "_soc_relaxation_summary_from_result", lambda result: (1.0, 1.0))
        monkeypatch.setattr(problem, "_solver_feeder_gap_summary_from_result", lambda result, **kwargs: (6.0, 6.0))

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert result.physics_refinement_status == "floor_failed_gate"
        assert result.returned_solution_source == "stage1"
        assert result.formulation_tightening_required is True
        assert result.refinement_status_counts == {"floor_failed_gate": 1}
    finally:
        env.close()


def test_solve_adaptive_full_horizon_marks_refined_nonbudget_limited_when_gap_remains_but_branch_l_is_close_to_floor(
    tmp_path,
    monkeypatch,
):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_nonbudget_nogurobi")
    cfg.mpc.physics_refinement_mode = "two_stage_min_branch_l"
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=4)

        stage1_result = _solution_result(problem, 4, solve_mode="single_window")
        floor_result = replace(
            _solution_result(problem, 4, solve_mode="single_window"),
            agent_net_cost_eur=6.0,
            throughput_regularization_eur=0.0,
            branch_i2_pu=np.full((problem.n_branches, 4), 0.5, dtype=np.float32),
        )
        stage2_result = replace(
            _solution_result(problem, 4, solve_mode="single_window"),
            agent_net_cost_eur=0.25,
            throughput_regularization_eur=0.0,
            branch_i2_pu=np.full((problem.n_branches, 4), 0.504, dtype=np.float32),
            root_import_mw=np.full((4,), 0.02, dtype=np.float32),
        )
        scripted_outcomes = iter([stage1_result, floor_result, stage2_result])

        monkeypatch.setattr(problem, "_solve_sequences", lambda **kwargs: next(scripted_outcomes))
        monkeypatch.setattr(problem, "_soc_relaxation_summary_from_result", lambda result: (0.0, 0.0))
        monkeypatch.setattr(
            problem,
            "_solver_feeder_gap_summary_from_result",
            lambda result, **kwargs: (10.0, 12.0) if result is stage2_result else (0.0, 0.0),
        )

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert result.physics_refinement_status == "refined_nonbudget_limited"
        assert result.used_physics_refinement_tier == 0
        assert result.branch_l_gap_ratio_to_floor < 0.01
        assert result.returned_solution_source == "stage2"
    finally:
        env.close()


def test_solve_adaptive_full_horizon_marks_refined_time_budget_limited_when_total_budget_is_hit(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_time_budget_nogurobi")
    cfg.mpc.physics_refinement_mode = "two_stage_min_branch_l"
    cfg.mpc.physics_refinement_time_limit_sec = 20.0
    cfg.mpc.physics_refinement_total_time_limit_sec = 5.0
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=4)

        stage1_result = _solution_result(problem, 4, solve_mode="single_window")
        floor_result = replace(
            _solution_result(problem, 4, solve_mode="single_window"),
            agent_net_cost_eur=6.0,
            throughput_regularization_eur=0.0,
            branch_i2_pu=np.full((problem.n_branches, 4), 0.5, dtype=np.float32),
        )
        stage2_result = replace(
            _solution_result(problem, 4, solve_mode="single_window"),
            agent_net_cost_eur=0.5,
            throughput_regularization_eur=0.0,
            solve_time_sec=6.0,
            branch_i2_pu=np.full((problem.n_branches, 4), 0.6, dtype=np.float32),
        )
        scripted_outcomes = iter([stage1_result, floor_result, stage2_result])

        monkeypatch.setattr(problem, "_solve_sequences", lambda **kwargs: next(scripted_outcomes))
        monkeypatch.setattr(problem, "_soc_relaxation_summary_from_result", lambda result: (0.0, 0.0))
        monkeypatch.setattr(
            problem,
            "_solver_feeder_gap_summary_from_result",
            lambda result, **kwargs: (10.0, 12.0) if result is stage2_result else (0.0, 0.0),
        )

        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            primary_window_steps=384,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
            retry_solve_config=GurobiSolveConfig(time_limit_sec=60.0, mip_gap=5e-3, cuts=1, heuristics=0.20, mip_focus=1),
        )

        assert result.physics_refinement_status == "refined_time_budget_limited"
        assert result.used_physics_refinement_tier == 0
        assert result.physics_refinement_attempt_count == 1
    finally:
        env.close()


def test_refinement_metrics_guard_branch_l_gap_ratio_when_floor_sum_is_near_zero(tmp_path):
    cfg = _make_cfg(tmp_path, "global_misocp_branch_l_gap_guard_nogurobi")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        result = replace(
            _solution_result(problem, 2, solve_mode="single_window"),
            branch_i2_pu=np.full((problem.n_branches, 2), 1e-7, dtype=np.float32),
        )

        metrics = problem._refinement_metrics_from_result(
            result,
            load_seq=np.zeros((problem.n_agents, 2), dtype=np.float32),
            pv_seq=np.zeros((problem.n_agents, 2), dtype=np.float32),
            floor_branch_l_objective=0.0,
        )

        assert np.isfinite(metrics["branch_l_gap_ratio"])
        assert metrics["branch_l_gap_ratio"] >= 0.0
    finally:
        env.close()


def test_export_gap_ratio_only_counts_meaningful_export_steps(tmp_path):
    cfg = _make_cfg(tmp_path, "global_misocp_export_gap_mask_nogurobi")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        result = replace(
            _solution_result(problem, 3, solve_mode="single_window"),
            root_p_kw=np.asarray([-2.0, -0.5, 5.0], dtype=np.float32),
        )
        mean_ratio, step_count = problem._export_gap_ratio_summary_from_result(
            result,
            load_seq=np.zeros((problem.n_agents, 3), dtype=np.float32),
            pv_seq=np.zeros((problem.n_agents, 3), dtype=np.float32),
        )

        assert step_count == 1
        assert np.isfinite(mean_ratio)
    finally:
        env.close()


def test_concatenate_chunk_results_preserves_refinement_status_counts_and_formulation_flag(tmp_path):
    cfg = _make_cfg(tmp_path, "global_misocp_chunk_status_nogurobi")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        chunk_a = replace(
            _solution_result(problem, 2, solve_mode="chunked_window"),
            stage1_primary_objective_eur=1.0,
            returned_primary_objective_eur=1.2,
            returned_primary_delta_abs_eur=0.2,
            returned_primary_delta_pct=20.0,
            returned_solution_source="floor",
            physics_refinement_status="floor_accepted",
            refinement_status_counts={"floor_accepted": 1},
            floor_accepted_tier=1,
            used_physics_refinement_tier=1,
            total_tiers_configured=2,
            floor_primary_objective_eur=1.2,
            floor_primary_delta_signed_eur=0.2,
            floor_primary_delta_positive_eur=0.2,
            physics_refinement_slack_cap_eur=2.0,
        )
        chunk_b = replace(
            _solution_result(problem, 2, solve_mode="chunked_window"),
            stage1_primary_objective_eur=2.0,
            returned_primary_objective_eur=2.0,
            returned_primary_delta_abs_eur=0.0,
            returned_primary_delta_pct=0.0,
            returned_solution_source="stage1",
            physics_refinement_status="floor_failed_gate",
            formulation_tightening_required=True,
            refinement_status_counts={"floor_failed_gate": 1},
            physics_refinement_slack_cap_eur=2.0,
        )

        result = problem._concatenate_chunk_results(
            [chunk_a, chunk_b],
            episode_offsets=np.asarray([0], dtype=np.int32),
            episode_lengths=np.asarray([4], dtype=np.int32),
            debug_artifacts={},
            solve_mode="chunked_window",
            chunk_summaries=[{"attempt_used": "primary"}, {"attempt_used": "primary"}],
        )

        assert result.physics_refinement_status == "refined"
        assert result.formulation_tightening_required is True
        assert result.refinement_status_counts == {"floor_accepted": 1, "floor_failed_gate": 1}
        assert result.floor_accepted_tier == 1
        assert result.used_physics_refinement_tier == 1
        assert result.returned_primary_objective_eur == pytest.approx(3.2)
        assert result.returned_primary_delta_abs_eur == pytest.approx(0.2)
        assert result.returned_primary_delta_pct == pytest.approx(100.0 * (3.2 - 3.0) / 3.0)
    finally:
        env.close()
