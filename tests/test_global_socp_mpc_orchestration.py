from __future__ import annotations

import numpy as np

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


def _solution_result(
    problem: GlobalMISOCPProblem,
    horizon_steps: int,
    *,
    solve_mode: str,
    objective_value: float = 0.0,
    stage1_primary_objective_eur: float | None = None,
) -> MISOCPResult:
    n_lines = int(np.asarray(problem.network.line_branch_indices, dtype=np.int32).size)
    energy = np.repeat((problem.capacity_mwh * 0.5).astype(np.float32)[:, None], horizon_steps + 1, axis=1)
    stage1_primary = float(objective_value) if stage1_primary_objective_eur is None else float(stage1_primary_objective_eur)
    return MISOCPResult(
        status_code=2,
        status_label="optimal",
        has_solution=True,
        time_limit_feasible=False,
        solve_time_sec=1.0,
        mip_gap=0.0,
        best_bound=0.0,
        objective_value=float(objective_value),
        agent_purchase_cost_eur=float(objective_value),
        agent_export_subsidy_eur=0.0,
        agent_net_cost_eur=float(objective_value),
        feeder_purchase_cost_eur=float(objective_value),
        feeder_export_subsidy_eur=0.0,
        feeder_net_cost_eur=float(objective_value),
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
        stage1_primary_objective_eur=stage1_primary,
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
        )

        assert result.solve_mode == "single_window"
        assert result.has_solution
        assert result.horizon_steps == 8
        assert result.stage1_primary_objective_eur == 0.0
        assert len(call_log) == 1
        assert call_log[0]["solve_mode"] == "single_window"
        assert call_log[0]["cuts"] == 2
    finally:
        env.close()


def test_solve_adaptive_full_horizon_falls_back_to_chunked_window_when_single_has_no_solution(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_adaptive_chunked_nogurobi")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=5)
        call_log: list[dict[str, object]] = []
        scripted_outcomes = iter(
            [
                _no_solution_result(),
                _solution_result(problem, 2, solve_mode="chunked_window", objective_value=1.0, stage1_primary_objective_eur=1.0),
                _solution_result(problem, 2, solve_mode="chunked_window", objective_value=2.0, stage1_primary_objective_eur=2.0),
                _solution_result(problem, 1, solve_mode="chunked_window", objective_value=3.0, stage1_primary_objective_eur=3.0),
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
            primary_window_steps=8,
            fallback_window_steps=2,
            solve_config=GurobiSolveConfig(time_limit_sec=30.0, mip_gap=5e-3, cuts=2, heuristics=0.10, mip_focus=1),
        )

        assert result.solve_mode == "chunked_window"
        assert result.has_solution
        assert result.horizon_steps == 5
        assert result.objective_value == 6.0
        assert result.stage1_primary_objective_eur == 6.0
        assert tuple(result.episode_offsets.tolist()) == (0,)
        assert tuple(result.episode_lengths.tolist()) == (5,)
        assert result.energy_mwh.shape == (problem.n_agents, 6)
        assert [entry["solve_mode"] for entry in call_log] == [
            "single_window",
            "chunked_window",
            "chunked_window",
            "chunked_window",
        ]
        assert [entry["horizon_steps"] for entry in call_log] == [5, 2, 2, 1]
    finally:
        env.close()


def test_solve_chunked_windows_returns_failed_chunk_without_legacy_retry_metadata(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, "global_misocp_chunked_failure_nogurobi")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = _mock_window_input(problem, horizon_steps=4)
        monkeypatch.setattr(problem, "_solve_sequences", lambda **kwargs: _no_solution_result())

        result = problem.solve_chunked_windows(
            full_input,
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            chunk_steps=2,
        )

        assert result.solve_mode == "chunked_window"
        assert not result.has_solution
        assert result.horizon_steps == 4
        assert tuple(result.episode_offsets.tolist()) == (0,)
        assert tuple(result.episode_lengths.tolist()) == (4,)
        assert not hasattr(result, "chunk_summaries")
        assert not hasattr(result, "chunk_retry_count")
    finally:
        env.close()
