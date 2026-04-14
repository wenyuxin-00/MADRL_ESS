from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from controllers.mpc import FullHorizonProblemInput, MISOCPResult
from controllers.mpc.global_socp_mpc import GlobalMISOCPProblem, ModelSize
from scripts.builder import build_env
from scripts.utils.misocp_notebook_helpers import (
    build_chunk_boundary_soc_df,
    build_misocp_plan_package,
    build_debug_tables,
    build_full_horizon_step_df,
    build_root_q_diagnostic_df,
    build_soc_relaxation_diagnostics,
    build_misocp_validation_df,
    build_simultaneous_diagnostic_tables,
    build_voltage_df,
    estimate_global_misocp_model_size,
    format_solver_summary,
    load_misocp_plan_package,
    plot_full_horizon_net_load,
    plot_misocp_validation_scatter_panel,
    plot_full_horizon_power_balance,
    plot_full_horizon_voltage,
    plot_root_exchange_alignment,
    replay_misocp_plan_package,
    resolve_latest_compatible_misocp_plan_package_dir,
    save_misocp_plan_package,
    summarize_misocp_validation,
    validate_misocp_result_schema,
)
from tests.support.helpers import make_case_dir, make_smoke_config, write_prosumer_processed_dataset


def _make_mock_problem() -> SimpleNamespace:
    network = SimpleNamespace(
        bus_ids=np.asarray([0, 1, 2, 3], dtype=np.int32),
        bus_pos={0: 0, 1: 1, 2: 2, 3: 3},
        s_base_mva=0.1,
        p_base_mw=np.asarray([0.005, 0.0, -0.002, 0.003], dtype=np.float32),
        q_base_mvar=np.asarray([0.0, 0.0, 0.004, 0.0], dtype=np.float32),
        agent_bus_positions=np.asarray([1, 3], dtype=np.int32),
        branch_parent_pos=np.asarray([0, 1, 1], dtype=np.int32),
        branch_child_pos=np.asarray([1, 2, 3], dtype=np.int32),
        branch_r_pu=np.asarray([0.05, 0.04, 0.02], dtype=np.float32),
        branch_x_pu=np.asarray([0.10, 0.20, 0.30], dtype=np.float32),
        line_branch_indices=np.asarray([1, 2], dtype=np.int32),
        branch_is_trafo=np.asarray([True, False, False], dtype=bool),
        root_outgoing_branches=np.asarray([0], dtype=np.int32),
    )
    return SimpleNamespace(
        network=network,
        n_agents=2,
        n_buses=4,
        n_branches=3,
        dt_hours=0.25,
        capacity_mwh=np.asarray([0.02, 0.03], dtype=np.float32),
        v_min_sq=0.95 ** 2,
        v_max_sq=1.05 ** 2,
        trafo_limit_mva=0.1,
    )


def _make_mock_full_input() -> SimpleNamespace:
    timestamps = pd.date_range("2020-06-01", periods=2, freq="15min")
    return SimpleNamespace(
        price_seq=np.asarray([0.10, 0.20], dtype=np.float32),
        load_seq=np.asarray([[10.0, 8.0], [6.0, 7.0]], dtype=np.float32),
        pv_seq=np.asarray([[4.0, 3.0], [1.0, 2.0]], dtype=np.float32),
        soc_init=np.asarray([0.5, 0.4], dtype=np.float32),
        timestamps=tuple(timestamp.isoformat() for timestamp in timestamps),
        episode_offsets=np.asarray([0], dtype=np.int32),
        episode_lengths=np.asarray([2], dtype=np.int32),
        episode_indices=np.asarray([0], dtype=np.int32),
    )


def _make_mock_result() -> SimpleNamespace:
    return SimpleNamespace(
        status_code=2,
        status_label="optimal",
        has_solution=True,
        time_limit_feasible=False,
        solve_time_sec=1.25,
        mip_gap=0.0,
        best_bound=12.0,
        objective_value=12.0,
        agent_purchase_cost_eur=0.875,
        agent_export_subsidy_eur=0.0,
        agent_net_cost_eur=0.875,
        feeder_purchase_cost_eur=1.3,
        feeder_export_subsidy_eur=0.0,
        feeder_net_cost_eur=1.3,
        throughput_regularization_eur=0.01,
        throughput_regularization_weight=1e-4,
        physical_tiebreaker_eur=0.0015,
        physical_tiebreaker_weight=1e-6,
        model_size=SimpleNamespace(
            num_vars=26,
            num_binary_vars=2,
            num_linear_constraints=34,
            num_quadratic_constraints=4,
        ),
        debug_artifacts={},
        agent_net_grid_mw=np.asarray([[0.008, 0.004], [0.003, 0.008]], dtype=np.float32),
        agent_import_mw=np.asarray([[0.008, 0.004], [0.003, 0.008]], dtype=np.float32),
        agent_export_mw=np.zeros((2, 2), dtype=np.float32),
        battery_charge_mw=np.asarray([[0.001, 0.0], [0.0, 0.002]], dtype=np.float32),
        battery_discharge_mw=np.asarray([[0.0, 0.001], [0.002, 0.0]], dtype=np.float32),
        pv_curtail_mw=np.asarray([[0.001, 0.0], [0.0, 0.001]], dtype=np.float32),
        energy_mwh=np.asarray([[0.010, 0.011, 0.010], [0.012, 0.0115, 0.012]], dtype=np.float32),
        branch_p_pu=np.asarray([[0.30, 0.20], [0.20, 0.12], [0.10, 0.08]], dtype=np.float32),
        branch_q_pu=np.asarray([[0.40, 0.10], [0.10, 0.08], [0.05, 0.04]], dtype=np.float32),
        branch_i2_pu=np.asarray([[0.30, 0.06], [0.07, 0.03], [0.015, 0.01]], dtype=np.float32),
        bus_v_sq=np.asarray(
            [
                [1.00, 1.00],
                [0.98, 0.96],
                [0.97, 0.95],
                [0.99, 0.97],
            ],
            dtype=np.float32,
        ),
        root_import_mw=np.asarray([0.018, 0.017], dtype=np.float32),
        root_export_mw=np.asarray([0.0, 0.0], dtype=np.float32),
        root_p_kw=np.asarray([18.0, 17.0], dtype=np.float32),
        root_q_kvar=np.asarray([9.0, 5.5], dtype=np.float32),
        bus_vm_pu=np.asarray(
            [
                [1.00, 1.01],
                [0.97, 0.94],
                [1.02, 1.00],
                [1.04, 1.06],
            ],
            dtype=np.float32,
        ),
        line_loading_pct=np.asarray([[55.0, 65.0], [40.0, 80.0]], dtype=np.float32),
        trafo_loading_pct=np.asarray([[60.0, 70.0]], dtype=np.float32),
        simultaneous_charge_discharge_kw=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32),
        simultaneous_agent_steps=1,
        simultaneous_step_ratio=0.5,
        max_simultaneous_kw=0.5,
        sol_count=1,
        node_count=12.0,
        iter_count=58.0,
        bar_iter_count=7.0,
        solve_mode="single_window",
        chunk_summaries=None,
        stage1_primary_objective_eur=0.885,
        stage2_primary_objective_eur=0.89,
        stage2_objective_slack_eur=0.1,
        stage2_branch_l_objective=0.375,
        physics_refinement_mode="two_stage_min_branch_l",
        physics_refinement_status="refined",
        physics_refinement_runtime_sec=1.0,
        floor_p95_soc_slack=0.02,
        floor_mean_abs_solver_feeder_gap_kw=3.0,
        floor_primary_objective_eur=0.88,
        floor_primary_delta_signed_eur=-0.005,
        floor_primary_delta_positive_eur=0.0,
        physics_refinement_slack_cap_eur=2.0,
        initial_physics_refinement_slack_cap_eur=2.0,
        returned_primary_objective_eur=0.89,
        returned_primary_delta_abs_eur=0.005,
        returned_primary_delta_pct=0.5649717514,
        floor_accepted_tier=None,
        used_physics_refinement_tier=0,
        total_tiers_configured=2,
        physics_refinement_attempt_count=1,
        physics_refinement_attempt_caps_eur=[2.0],
        physics_refinement_cap_utilization=0.25,
        branch_l_gap_ratio_to_floor=0.015,
        returned_mean_abs_solver_feeder_gap_kw=3.0,
        returned_max_solver_feeder_gap_kw=6.0,
        returned_mean_abs_export_gap_ratio=0.03,
        high_budget_refinement_warn=False,
        returned_solution_source="stage2",
        formulation_tightening_required=False,
        negative_floor_delta_warn=False,
        refinement_status_counts={"refined": 1},
    )


def _make_real_full_input() -> FullHorizonProblemInput:
    mock = _make_mock_full_input()
    return FullHorizonProblemInput(
        price_seq=np.asarray(mock.price_seq, dtype=np.float32),
        load_seq=np.asarray(mock.load_seq, dtype=np.float32),
        pv_seq=np.asarray(mock.pv_seq, dtype=np.float32),
        soc_init=np.asarray(mock.soc_init, dtype=np.float32),
        timestamps=tuple(str(value) for value in mock.timestamps),
        episode_offsets=np.asarray(mock.episode_offsets, dtype=np.int32),
        episode_lengths=np.asarray(mock.episode_lengths, dtype=np.int32),
        episode_indices=np.asarray(mock.episode_indices, dtype=np.int32),
    )


def _make_real_result() -> MISOCPResult:
    mock = _make_mock_result()
    return MISOCPResult(
        status_code=int(mock.status_code),
        status_label=str(mock.status_label),
        has_solution=bool(mock.has_solution),
        time_limit_feasible=bool(mock.time_limit_feasible),
        solve_time_sec=float(mock.solve_time_sec),
        mip_gap=float(mock.mip_gap),
        best_bound=float(mock.best_bound),
        objective_value=float(mock.objective_value),
        agent_purchase_cost_eur=float(mock.agent_purchase_cost_eur),
        agent_export_subsidy_eur=float(mock.agent_export_subsidy_eur),
        agent_net_cost_eur=float(mock.agent_net_cost_eur),
        feeder_purchase_cost_eur=float(mock.feeder_purchase_cost_eur),
        feeder_export_subsidy_eur=float(mock.feeder_export_subsidy_eur),
        feeder_net_cost_eur=float(mock.feeder_net_cost_eur),
        throughput_regularization_eur=float(mock.throughput_regularization_eur),
        throughput_regularization_weight=float(mock.throughput_regularization_weight),
        physical_tiebreaker_eur=0.0015,
        physical_tiebreaker_weight=1e-6,
        model_size=ModelSize(
            num_vars=int(mock.model_size.num_vars),
            num_binary_vars=int(mock.model_size.num_binary_vars),
            num_linear_constraints=int(mock.model_size.num_linear_constraints),
            num_quadratic_constraints=int(mock.model_size.num_quadratic_constraints),
        ),
        debug_artifacts=dict(mock.debug_artifacts),
        agent_net_grid_mw=np.asarray(mock.agent_net_grid_mw, dtype=np.float32),
        agent_import_mw=np.asarray(mock.agent_import_mw, dtype=np.float32),
        agent_export_mw=np.asarray(mock.agent_export_mw, dtype=np.float32),
        battery_charge_mw=np.asarray(mock.battery_charge_mw, dtype=np.float32),
        battery_discharge_mw=np.asarray(mock.battery_discharge_mw, dtype=np.float32),
        pv_curtail_mw=np.asarray(mock.pv_curtail_mw, dtype=np.float32),
        energy_mwh=np.asarray(mock.energy_mwh, dtype=np.float32),
        branch_p_pu=np.asarray(mock.branch_p_pu, dtype=np.float32),
        branch_q_pu=np.asarray(mock.branch_q_pu, dtype=np.float32),
        branch_i2_pu=np.asarray(mock.branch_i2_pu, dtype=np.float32),
        bus_v_sq=np.asarray(mock.bus_v_sq, dtype=np.float32),
        root_import_mw=np.asarray(mock.root_import_mw, dtype=np.float32),
        root_export_mw=np.asarray(mock.root_export_mw, dtype=np.float32),
        root_p_kw=np.asarray(mock.root_p_kw, dtype=np.float32),
        root_q_kvar=np.asarray(mock.root_q_kvar, dtype=np.float32),
        bus_vm_pu=np.asarray(mock.bus_vm_pu, dtype=np.float32),
        line_loading_pct=np.asarray(mock.line_loading_pct, dtype=np.float32),
        trafo_loading_pct=np.asarray(mock.trafo_loading_pct, dtype=np.float32),
        simultaneous_charge_discharge_kw=np.asarray(mock.simultaneous_charge_discharge_kw, dtype=np.float32),
        simultaneous_agent_steps=int(mock.simultaneous_agent_steps),
        simultaneous_step_ratio=float(mock.simultaneous_step_ratio),
        max_simultaneous_kw=float(mock.max_simultaneous_kw),
        sol_count=int(mock.sol_count),
        node_count=float(mock.node_count),
        iter_count=float(mock.iter_count),
        bar_iter_count=float(mock.bar_iter_count),
        sanity_warning="",
        horizon_steps=2,
        solve_mode=str(mock.solve_mode),
        episode_offsets=np.asarray([0], dtype=np.int32),
        episode_lengths=np.asarray([2], dtype=np.int32),
        chunk_summaries=None,
        stage1_primary_objective_eur=float(mock.stage1_primary_objective_eur),
        stage2_primary_objective_eur=float(mock.stage2_primary_objective_eur),
        stage2_objective_slack_eur=float(mock.stage2_objective_slack_eur),
        stage2_branch_l_objective=float(mock.stage2_branch_l_objective),
        physics_refinement_mode=str(mock.physics_refinement_mode),
        physics_refinement_status=str(mock.physics_refinement_status),
        physics_refinement_runtime_sec=float(mock.physics_refinement_runtime_sec),
        floor_p95_soc_slack=float(mock.floor_p95_soc_slack),
        floor_mean_abs_solver_feeder_gap_kw=float(mock.floor_mean_abs_solver_feeder_gap_kw),
        floor_primary_objective_eur=float(mock.floor_primary_objective_eur),
        floor_primary_delta_signed_eur=float(mock.floor_primary_delta_signed_eur),
        floor_primary_delta_positive_eur=float(mock.floor_primary_delta_positive_eur),
        physics_refinement_slack_cap_eur=float(mock.physics_refinement_slack_cap_eur),
        initial_physics_refinement_slack_cap_eur=float(mock.initial_physics_refinement_slack_cap_eur),
        returned_primary_objective_eur=float(mock.returned_primary_objective_eur),
        returned_primary_delta_abs_eur=float(mock.returned_primary_delta_abs_eur),
        returned_primary_delta_pct=float(mock.returned_primary_delta_pct),
        floor_accepted_tier=mock.floor_accepted_tier,
        used_physics_refinement_tier=mock.used_physics_refinement_tier,
        total_tiers_configured=int(mock.total_tiers_configured),
        physics_refinement_attempt_count=int(mock.physics_refinement_attempt_count),
        physics_refinement_attempt_caps_eur=[float(value) for value in list(mock.physics_refinement_attempt_caps_eur)],
        physics_refinement_cap_utilization=float(mock.physics_refinement_cap_utilization),
        branch_l_gap_ratio_to_floor=float(mock.branch_l_gap_ratio_to_floor),
        returned_mean_abs_solver_feeder_gap_kw=float(mock.returned_mean_abs_solver_feeder_gap_kw),
        returned_max_solver_feeder_gap_kw=float(mock.returned_max_solver_feeder_gap_kw),
        returned_mean_abs_export_gap_ratio=float(mock.returned_mean_abs_export_gap_ratio),
        high_budget_refinement_warn=bool(mock.high_budget_refinement_warn),
        returned_solution_source=str(mock.returned_solution_source),
        formulation_tightening_required=bool(mock.formulation_tightening_required),
        negative_floor_delta_warn=bool(mock.negative_floor_delta_warn),
        refinement_status_counts=dict(mock.refinement_status_counts or {}),
    )


def _make_real_problem_fixture(tmp_path: Path):
    case_dir = make_case_dir(tmp_path, "misocp_plan_package")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        full_input = problem.build_full_horizon_input(env)
    finally:
        env.close()

    horizon_steps = int(full_input.horizon_steps)
    n_agents = int(problem.n_agents)
    n_buses = int(problem.n_buses)
    n_branches = int(problem.n_branches)
    line_branch_indices = getattr(problem.network, "line_branch_indices", None)
    n_lines = (
        int(np.asarray(line_branch_indices, dtype=np.int32).size)
        if line_branch_indices is not None
        else int(np.sum(~np.asarray(problem.network.branch_is_trafo, dtype=bool)))
    )
    root_import_kw = np.maximum(
        (np.asarray(full_input.load_seq, dtype=np.float32) - np.asarray(full_input.pv_seq, dtype=np.float32)).sum(axis=0),
        0.0,
    ).astype(np.float32)
    agent_net_grid_mw = (
        (np.asarray(full_input.load_seq, dtype=np.float32) - np.asarray(full_input.pv_seq, dtype=np.float32)) / 1000.0
    ).astype(np.float32)
    agent_import_mw = np.clip(agent_net_grid_mw, 0.0, None).astype(np.float32)
    agent_export_mw = np.clip(-agent_net_grid_mw, 0.0, None).astype(np.float32)
    agent_purchase_cost_eur = float(
        np.sum(
            1000.0
            * float(problem.dt_hours)
            * np.asarray(full_input.price_seq, dtype=np.float32).reshape(1, -1)
            * agent_import_mw
        )
    )
    agent_export_subsidy_eur = 0.0
    feeder_purchase_cost_eur = float(
        np.sum(
            1000.0
            * float(problem.dt_hours)
            * np.asarray(full_input.price_seq, dtype=np.float32)
            * (root_import_kw / 1000.0)
        )
    )
    initial_energy_mwh = (
        np.asarray(problem.capacity_mwh, dtype=np.float32) * np.asarray(full_input.soc_init, dtype=np.float32)
    ).astype(np.float32)
    energy_mwh = np.repeat(initial_energy_mwh[:, None], horizon_steps + 1, axis=1).astype(np.float32)
    result = MISOCPResult(
        status_code=2,
        status_label="optimal",
        has_solution=True,
        time_limit_feasible=False,
        solve_time_sec=0.25,
        mip_gap=0.0,
        best_bound=0.0,
        objective_value=0.0,
        agent_purchase_cost_eur=agent_purchase_cost_eur,
        agent_export_subsidy_eur=agent_export_subsidy_eur,
        agent_net_cost_eur=agent_purchase_cost_eur - agent_export_subsidy_eur,
        feeder_purchase_cost_eur=feeder_purchase_cost_eur,
        feeder_export_subsidy_eur=0.0,
        feeder_net_cost_eur=feeder_purchase_cost_eur,
        throughput_regularization_eur=0.0,
        throughput_regularization_weight=1e-4,
        physical_tiebreaker_eur=0.0,
        physical_tiebreaker_weight=float(cfg.mpc.branch_current_tiebreaker_eur_per_pu_step),
        model_size=ModelSize(
            num_vars=10 + n_agents * horizon_steps,
            num_binary_vars=horizon_steps,
            num_linear_constraints=10 + 2 * n_agents * horizon_steps,
            num_quadratic_constraints=10,
        ),
        debug_artifacts={},
        agent_net_grid_mw=agent_net_grid_mw,
        agent_import_mw=agent_import_mw,
        agent_export_mw=agent_export_mw,
        battery_charge_mw=np.zeros((n_agents, horizon_steps), dtype=np.float32),
        battery_discharge_mw=np.zeros((n_agents, horizon_steps), dtype=np.float32),
        pv_curtail_mw=np.zeros((n_agents, horizon_steps), dtype=np.float32),
        energy_mwh=energy_mwh,
        branch_p_pu=np.zeros((n_branches, horizon_steps), dtype=np.float32),
        branch_q_pu=np.zeros((n_branches, horizon_steps), dtype=np.float32),
        branch_i2_pu=np.zeros((n_branches, horizon_steps), dtype=np.float32),
        bus_v_sq=np.ones((n_buses, horizon_steps), dtype=np.float32),
        root_import_mw=(root_import_kw / 1000.0).astype(np.float32),
        root_export_mw=np.zeros((horizon_steps,), dtype=np.float32),
        root_p_kw=root_import_kw.astype(np.float32),
        root_q_kvar=np.zeros((horizon_steps,), dtype=np.float32),
        bus_vm_pu=np.ones((n_buses, horizon_steps), dtype=np.float32),
        line_loading_pct=np.zeros((n_lines, horizon_steps), dtype=np.float32),
        trafo_loading_pct=np.zeros((1, horizon_steps), dtype=np.float32),
        simultaneous_charge_discharge_kw=np.zeros((n_agents, horizon_steps), dtype=np.float32),
        simultaneous_agent_steps=0,
        simultaneous_step_ratio=0.0,
        max_simultaneous_kw=0.0,
        sol_count=1,
        node_count=0.0,
        iter_count=0.0,
        bar_iter_count=0.0,
        sanity_warning="",
        horizon_steps=horizon_steps,
        solve_mode="single_window",
        stage1_primary_objective_eur=agent_purchase_cost_eur,
        stage2_primary_objective_eur=agent_purchase_cost_eur,
        stage2_objective_slack_eur=0.1,
        stage2_branch_l_objective=0.0,
        physics_refinement_mode="two_stage_min_branch_l",
        physics_refinement_status="refined",
        physics_refinement_runtime_sec=0.25,
        floor_p95_soc_slack=0.0,
        floor_mean_abs_solver_feeder_gap_kw=0.0,
        floor_primary_objective_eur=agent_purchase_cost_eur,
        floor_primary_delta_signed_eur=0.0,
        floor_primary_delta_positive_eur=0.0,
        physics_refinement_slack_cap_eur=2.0,
        initial_physics_refinement_slack_cap_eur=2.0,
        returned_primary_objective_eur=agent_purchase_cost_eur,
        returned_primary_delta_abs_eur=0.0,
        returned_primary_delta_pct=0.0,
        floor_accepted_tier=None,
        used_physics_refinement_tier=0,
        total_tiers_configured=2,
        physics_refinement_attempt_count=1,
        physics_refinement_attempt_caps_eur=[2.0],
        physics_refinement_cap_utilization=0.0,
        branch_l_gap_ratio_to_floor=0.0,
        returned_mean_abs_solver_feeder_gap_kw=0.0,
        returned_max_solver_feeder_gap_kw=0.0,
        returned_mean_abs_export_gap_ratio=0.0,
        high_budget_refinement_warn=False,
        returned_solution_source="stage2",
        formulation_tightening_required=False,
        negative_floor_delta_warn=False,
        refinement_status_counts={"refined": 1},
        episode_offsets=np.asarray(full_input.episode_offsets, dtype=np.int32),
        episode_lengths=np.asarray(full_input.episode_lengths, dtype=np.int32),
        chunk_summaries=None,
    )
    return cfg, problem, full_input, result


def _load_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def test_build_full_horizon_step_df_returns_expected_columns_and_values():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()

    step_df = build_full_horizon_step_df(problem, full_input, result)

    expected_columns = {
        "timestamp",
        "episode_idx",
        "global_step",
        "price",
        "fixed_load_kw",
        "fixed_generation_kw",
        "agent_load_kw",
        "pv_raw_kw",
        "pv_curtail_kw",
        "pv_effective_kw",
        "battery_charge_kw",
        "battery_discharge_kw",
        "agent_import_kw_total",
        "agent_export_kw_total",
        "agent_purchase_cost_eur_step",
        "agent_export_subsidy_eur_step",
        "agent_net_cost_eur_step",
        "feeder_purchase_cost_eur_step",
        "feeder_export_subsidy_eur_step",
        "feeder_net_cost_eur_step",
        "grid_import_kw",
        "grid_export_kw",
        "agent_raw_net_load_kw",
        "agent_effective_net_load_kw",
        "agent_post_action_net_load_kw",
        "feeder_raw_net_load_kw",
        "feeder_effective_net_load_kw",
        "feeder_post_action_net_load_kw",
        "raw_net_load_kw",
        "effective_net_load_kw",
        "post_action_net_load_kw",
        "root_net_exchange_kw",
        "agent_root_gap_kw",
        "balance_supply_kw",
        "balance_demand_kw",
        "balance_residual_kw",
        "min_vm_pu",
        "mean_vm_pu",
        "max_vm_pu",
        "max_line_loading_pct",
        "trafo_loading_pct",
        "simultaneous_kw_total",
        "aggregate_stored_energy_kwh",
    }
    assert expected_columns.issubset(step_df.columns)
    assert step_df["fixed_load_kw"].tolist() == [8.0, 8.0]
    assert step_df["fixed_generation_kw"].tolist() == [2.0, 2.0]
    assert step_df["agent_raw_net_load_kw"].tolist() == [11.0, 10.0]
    assert step_df["feeder_post_action_net_load_kw"].tolist() == [17.0, 18.0]
    assert step_df["agent_import_kw_total"].tolist() == [11.0, 12.0]
    assert step_df["agent_export_kw_total"].tolist() == [0.0, 0.0]
    assert step_df["agent_root_gap_kw"].tolist() == [7.0, 5.0]
    assert step_df["balance_residual_kw"].tolist() == [1.0, -1.0]
    assert step_df["root_net_exchange_kw"].tolist() == [18.0, 17.0]
    assert step_df["agent_purchase_cost_eur_step"].sum() == pytest.approx(result.agent_purchase_cost_eur)
    assert step_df["agent_export_subsidy_eur_step"].sum() == pytest.approx(result.agent_export_subsidy_eur)
    assert step_df["agent_net_cost_eur_step"].sum() == pytest.approx(result.agent_net_cost_eur)
    assert step_df["aggregate_stored_energy_kwh"].tolist() == [22.5, 22.0]


def test_validate_misocp_result_schema_rejects_stale_result_objects():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    stale_payload = dict(_make_mock_result().__dict__)
    stale_payload.pop("agent_import_mw")
    stale_result = SimpleNamespace(**stale_payload)

    with pytest.raises(ValueError, match="outdated MISOCPResult schema"):
        validate_misocp_result_schema(stale_result)

    with pytest.raises(ValueError, match="outdated MISOCPResult schema"):
        build_full_horizon_step_df(problem, full_input, stale_result)


def test_build_voltage_df_expands_bus_traces_and_agent_flags():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()

    voltage_df = build_voltage_df(problem, full_input, result)

    assert list(voltage_df.columns) == ["timestamp", "episode_idx", "global_step", "bus_id", "vm_pu", "is_agent_bus"]
    assert len(voltage_df) == 8
    assert sorted(voltage_df.loc[voltage_df["is_agent_bus"], "bus_id"].unique().tolist()) == [1, 3]
    assert voltage_df["global_step"].tolist()[:4] == [0, 0, 0, 0]


def test_build_debug_tables_surface_summary_balance_and_voltage_details():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()
    step_df = build_full_horizon_step_df(problem, full_input, result)

    solve_summary, worst_balance_df, voltage_issue_df, battery_summary_df = build_debug_tables(
        problem,
        full_input,
        result,
        step_df=step_df,
        agent_profiles=["A", "B"],
        agent_bus_ids=[1, 3],
    )

    assert solve_summary["status_label"] == "optimal"
    assert solve_summary["num_binary_vars"] == 2
    assert solve_summary["sol_count"] == 1
    assert solve_summary["economics_scope"] == "agent_only"
    assert solve_summary["agent_purchase_cost_eur"] == pytest.approx(result.agent_purchase_cost_eur)
    assert solve_summary["physical_tiebreaker_eur"] == pytest.approx(0.0015)
    assert "abs_balance_residual_kw" in worst_balance_df.columns
    assert not worst_balance_df.empty
    assert voltage_issue_df["global_step"].tolist() == [1]
    assert battery_summary_df["agent_profile"].tolist() == ["A", "B"]
    assert np.allclose(battery_summary_df["soc_final"].to_numpy(dtype=np.float32), np.asarray([0.5, 0.4], dtype=np.float32))


def test_agent_and_feeder_costs_diverge_with_background_and_match_without_background():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()

    step_df = build_full_horizon_step_df(problem, full_input, result)
    assert float(step_df["feeder_purchase_cost_eur_step"].sum()) > float(step_df["agent_purchase_cost_eur_step"].sum())

    zero_background_problem = _make_mock_problem()
    zero_background_problem.network.p_base_mw = np.zeros_like(zero_background_problem.network.p_base_mw)
    zero_background_result = _make_mock_result()
    zero_background_result.root_import_mw = np.asarray(np.sum(zero_background_result.agent_import_mw, axis=0), dtype=np.float32)
    zero_background_result.root_export_mw = np.asarray(np.sum(zero_background_result.agent_export_mw, axis=0), dtype=np.float32)
    zero_background_result.root_p_kw = (
        (zero_background_result.root_import_mw - zero_background_result.root_export_mw) * 1000.0
    ).astype(np.float32)
    zero_background_result.feeder_purchase_cost_eur = float(zero_background_result.agent_purchase_cost_eur)
    zero_background_result.feeder_export_subsidy_eur = float(zero_background_result.agent_export_subsidy_eur)
    zero_background_result.feeder_net_cost_eur = float(zero_background_result.agent_net_cost_eur)

    zero_background_step_df = build_full_horizon_step_df(zero_background_problem, full_input, zero_background_result)
    assert float(zero_background_step_df["feeder_purchase_cost_eur_step"].sum()) == pytest.approx(
        float(zero_background_step_df["agent_purchase_cost_eur_step"].sum())
    )
    assert float(zero_background_step_df["feeder_net_cost_eur_step"].sum()) == pytest.approx(
        float(zero_background_step_df["agent_net_cost_eur_step"].sum())
    )


def test_estimate_model_size_and_solver_summary_helpers_return_expected_fields():
    problem = _make_mock_problem()
    result = _make_mock_result()

    estimate = estimate_global_misocp_model_size(problem, 2, episode_count=1)
    solver_summary = format_solver_summary(result, total_steps=2, episode_count=1)

    assert estimate["n_agents"] == 2
    assert estimate["n_lines"] == 2
    assert estimate["num_vars_est"] == 54
    assert estimate["num_binary_vars_est"] == 2
    assert estimate["num_linear_constraints_est"] == 70
    assert estimate["num_quadratic_constraints_est"] == 8
    assert solver_summary["sol_count"] == 1
    assert solver_summary["node_count"] == 12.0
    assert solver_summary["num_quadratic_constraints"] == result.model_size.num_quadratic_constraints
    assert solver_summary["economics_scope"] == "agent_only"
    assert solver_summary["physical_tiebreaker_weight"] == pytest.approx(1e-6)
    assert solver_summary["physics_refinement_mode"] == "two_stage_min_branch_l"
    assert solver_summary["physics_refinement_status"] == "refined"
    assert solver_summary["stage1_primary_objective_eur"] == pytest.approx(0.885)
    assert solver_summary["floor_primary_objective_eur"] == pytest.approx(0.88)
    assert solver_summary["returned_primary_objective_eur"] == pytest.approx(0.89)
    assert solver_summary["physics_refinement_slack_cap_eur"] == pytest.approx(2.0)
    assert solver_summary["returned_solution_source"] == "stage2"
    assert solver_summary["refinement_status_counts"] == {"refined": 1}
    assert solver_summary["floor_p95_soc_slack"] == pytest.approx(0.02)
    assert solver_summary["no_retry_or_fallback_used"]
    assert solver_summary["chunk_retry_count"] == 0
    assert solver_summary["total_runtime_sec"] == pytest.approx(result.solve_time_sec)
    assert np.isnan(float(solver_summary["max_chunk_runtime_sec"]))


def test_soc_relaxation_diagnostics_use_parent_voltage_and_internal_pu_units():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()

    diagnostics = build_soc_relaxation_diagnostics(problem, full_input, result, top_k=6)
    summary = diagnostics["summary"]
    step_df = diagnostics["step_df"]
    worst_df = diagnostics["worst_df"]

    expected_branch1_step1 = 0.03 - ((0.12 ** 2 + 0.08 ** 2) / 0.96)
    assert step_df.loc[0, "soc_slack_max"] == pytest.approx(0.05, rel=1e-5)
    assert step_df.loc[1, "soc_slack_max"] == pytest.approx(0.01, rel=1e-5)
    assert worst_df.loc[0, "branch_idx"] == 0
    assert worst_df.loc[0, "global_step"] == 0
    assert worst_df.loc[0, "soc_slack"] == pytest.approx(0.05, rel=1e-5)
    assert worst_df.loc[1, "branch_idx"] == 1
    assert worst_df.loc[1, "global_step"] == 0
    branch1_step1 = worst_df.loc[
        (worst_df["branch_idx"] == 1) & (worst_df["global_step"] == 1),
        "soc_slack",
    ].iloc[0]
    assert branch1_step1 == pytest.approx(expected_branch1_step1, rel=1e-5)
    assert summary["p95_soc_slack"] > 1e-4
    assert bool(summary["soc_relaxation_is_tight"]) is False


def test_root_q_diagnostic_df_decomposes_background_and_xl_proxy():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()

    root_q_df = build_root_q_diagnostic_df(problem, full_input, result)

    assert root_q_df.loc[0, "background_q_base_total_kvar"] == pytest.approx(4.0)
    assert root_q_df.loc[0, "network_q_loss_proxy_kvar"] == pytest.approx(4.85, rel=1e-5)
    assert root_q_df.loc[0, "root_q_residual_kvar"] == pytest.approx(0.15, rel=1e-5)
    assert root_q_df.loc[1, "network_q_loss_proxy_kvar"] == pytest.approx(1.5, rel=1e-5)
    assert root_q_df.loc[1, "root_q_residual_kvar"] == pytest.approx(0.0, abs=1e-6)


def test_validation_and_simultaneous_helpers_return_expected_tables():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()

    validation_df = build_misocp_validation_df(
        [
            {
                "controller": "Mock",
                "episode_idx": 0,
                "step": 0,
                "timestamp": pd.Timestamp("2020-06-01 00:00:00"),
                "solve_time_sec": 1.0,
                "misocp_vm_pu": np.asarray([1.00, 0.97], dtype=np.float32),
                "pp_vm_pu": np.asarray([0.999, 0.971], dtype=np.float32),
                "misocp_line_loading_pct": np.asarray([55.0, 40.0], dtype=np.float32),
                "pp_line_loading_pct": np.asarray([55.2, 40.3], dtype=np.float32),
                "misocp_trafo_loading_pct": np.asarray([60.0], dtype=np.float32),
                "pp_trafo_loading_pct": np.asarray([60.5], dtype=np.float32),
                "root_p_kw": 18.0,
                "pp_root_p_kw": 18.04,
            }
        ]
    )
    validation_summary = summarize_misocp_validation(validation_df)
    simultaneous_step_df, simultaneous_agent_df = build_simultaneous_diagnostic_tables(
        problem,
        full_input,
        result,
        agent_profiles=["A", "B"],
    )

    assert validation_df.loc[0, "within_tolerance"]
    assert validation_summary["steps_outside_tolerance"] == 0
    assert validation_summary["pp_root_p_available_ratio"] == pytest.approx(1.0)
    assert not simultaneous_step_df.empty
    assert simultaneous_step_df.loc[0, "simultaneous_kw_total"] == 0.5
    assert simultaneous_agent_df.loc[0, "agent_profile"] == "B"
    assert {"charge_kw", "discharge_kw", "soc_before", "soc_after"}.issubset(simultaneous_agent_df.columns)


def test_validation_summary_reports_fit_and_gap_metrics():
    validation_df = build_misocp_validation_df(
        [
            {
                "controller": "Mock",
                "episode_idx": 0,
                "step": idx,
                "timestamp": pd.Timestamp("2020-06-01 00:00:00") + pd.Timedelta(minutes=15 * idx),
                "solve_time_sec": 1.0,
                "misocp_vm_pu": np.asarray([1.0], dtype=np.float32),
                "pp_vm_pu": np.asarray([1.0], dtype=np.float32),
                "misocp_line_loading_pct": np.asarray([10.0], dtype=np.float32),
                "pp_line_loading_pct": np.asarray([10.0], dtype=np.float32),
                "misocp_trafo_loading_pct": np.asarray([20.0], dtype=np.float32),
                "pp_trafo_loading_pct": np.asarray([20.0], dtype=np.float32),
                "root_p_kw": float(root_p),
                "pp_root_p_kw": float(pp_root_p),
                "pp_root_p_available": True,
                "misocp_root_s_kva": float(root_s),
                "pp_root_s_kva": float(pp_root_s),
                "root_q_kvar": 4.0,
                "soc_slack_max": float(slack_max),
                "soc_slack_mean": float(slack_mean),
                "soc_slack_p95_global": 0.025,
                "background_q_base_total_kvar": 4.0,
                "network_q_loss_proxy_kvar": float(idx + 1),
                "root_q_residual_kvar": float(idx - 1),
                "solver_feeder_gap_kw": float(solver_gap),
                "replay_feeder_gap_kw": float(replay_gap),
            }
            for idx, (root_p, pp_root_p, root_s, pp_root_s, slack_max, slack_mean, solver_gap, replay_gap) in enumerate(
                [
                    (1.0, 3.0, 2.0, 1.0, 0.01, 0.005, 1.0, 0.5),
                    (2.0, 5.0, 4.0, 2.0, 0.02, 0.01, -2.0, -0.5),
                    (3.0, 7.0, 6.0, 3.0, 0.03, 0.015, 3.0, 1.0),
                ]
            )
        ]
    )

    summary = summarize_misocp_validation(validation_df)

    assert summary["root_p_fit_k"] == pytest.approx(2.0)
    assert summary["root_p_fit_b"] == pytest.approx(1.0)
    assert summary["root_p_fit_r2"] == pytest.approx(1.0)
    assert summary["root_s_fit_k"] == pytest.approx(0.5)
    assert summary["root_s_fit_b"] == pytest.approx(0.0)
    assert summary["root_s_fit_r2"] == pytest.approx(1.0)
    assert summary["max_solver_feeder_gap_kw"] == pytest.approx(3.0)
    assert summary["mean_abs_solver_feeder_gap_kw"] == pytest.approx(2.0)
    assert summary["max_replay_feeder_gap_kw"] == pytest.approx(1.0)
    assert summary["mean_abs_replay_feeder_gap_kw"] == pytest.approx((0.5 + 0.5 + 1.0) / 3.0)
    assert summary["p95_soc_slack"] == pytest.approx(0.025)
    assert summary["root_p_fit_interpretation"] == "systematic_linear_bias"


def test_validation_marks_root_power_unavailable_without_silent_zero_fallback():
    validation_df = build_misocp_validation_df(
        [
            {
                "controller": "Mock",
                "episode_idx": 0,
                "step": 0,
                "timestamp": pd.Timestamp("2020-06-01 00:00:00"),
                "solve_time_sec": 1.0,
                "misocp_vm_pu": np.asarray([1.0, 0.98], dtype=np.float32),
                "pp_vm_pu": np.asarray([1.0, 0.98], dtype=np.float32),
                "misocp_line_loading_pct": np.asarray([10.0, 20.0], dtype=np.float32),
                "pp_line_loading_pct": np.asarray([10.0, 20.0], dtype=np.float32),
                "misocp_trafo_loading_pct": np.asarray([30.0], dtype=np.float32),
                "pp_trafo_loading_pct": np.asarray([30.0], dtype=np.float32),
                "root_p_kw": 18.0,
                "pp_root_p_kw": np.nan,
                "pp_root_p_available": False,
                "root_q_kvar": 4.0,
                "misocp_root_s_kva": 18.4,
                "pp_root_s_kva": 18.4,
            }
        ]
    )
    validation_summary = summarize_misocp_validation(validation_df)

    assert bool(validation_df.loc[0, "pp_root_p_available"]) is False
    assert np.isnan(float(validation_df.loc[0, "root_p_abs_err_kw"]))
    assert bool(validation_df.loc[0, "root_power_validation_unavailable"]) is True
    assert bool(validation_summary["root_power_validation_unavailable"]) is True
    assert validation_summary["pp_root_p_available_ratio"] == pytest.approx(0.0)


def test_chunk_boundary_soc_df_reports_soc_kinks_for_chunked_results():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()
    result.solve_mode = "chunked_window"
    result.chunk_summaries = [
        {"chunk_idx": 0, "start_step": 0, "end_step": 1, "window_steps": 1, "status_label": "optimal", "has_solution": True, "solve_time_sec": 1.0, "mip_gap": 0.0, "objective_value": 1.0, "attempt_used": "primary"},
        {"chunk_idx": 1, "start_step": 1, "end_step": 2, "window_steps": 1, "status_label": "optimal", "has_solution": True, "solve_time_sec": 1.0, "mip_gap": 0.0, "objective_value": 1.0, "attempt_used": "primary"},
    ]

    chunk_boundary_soc_df = build_chunk_boundary_soc_df(problem, full_input, result)

    assert list(chunk_boundary_soc_df.columns) == [
        "boundary_step",
        "boundary_timestamp",
        "chunk_left",
        "chunk_right",
        "max_abs_soc_kink",
        "mean_abs_soc_kink",
    ]
    assert chunk_boundary_soc_df.loc[0, "boundary_step"] == 1
    assert chunk_boundary_soc_df.loc[0, "chunk_left"] == 0
    assert chunk_boundary_soc_df.loc[0, "chunk_right"] == 1


def test_plan_package_round_trip_reconstructs_real_dataclasses(tmp_path):
    cfg, problem, full_input, result = _make_real_problem_fixture(tmp_path)

    package = build_misocp_plan_package(
        problem,
        full_input,
        result,
        controller_label="Global MISOCP (single_window)",
        export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
        cfg=cfg,
    )
    saved_dir = save_misocp_plan_package(package, tmp_path / "cached_plan")
    loaded = load_misocp_plan_package(saved_dir)

    assert isinstance(loaded["full_input"], FullHorizonProblemInput)
    assert isinstance(loaded["result"], MISOCPResult)
    assert loaded["manifest"]["plan_package_version"] == 11
    assert loaded["manifest"]["diagnostics_stage"] == "commit4_adaptive_refinement_ladder"
    assert loaded["manifest"]["cfg_snapshot"]["import_price_adder_eur_per_kwh"] == pytest.approx(
        cfg.reward.import_price_adder_eur_per_kwh
    )
    assert loaded["manifest"]["validation_root_power_source"] == "trafo_p_signed_kw"
    assert loaded["manifest"]["agent_q_base_zeroed"] is True
    assert loaded["full_input"].horizon_steps == full_input.horizon_steps
    assert np.allclose(loaded["result"].battery_charge_mw, result.battery_charge_mw)
    assert loaded["solve_summary"]["status_label"] == "optimal"
    assert loaded["solve_summary"]["economics_scope"] == "agent_only"
    assert loaded["solve_summary"]["diagnostics_stage"] == "commit4_adaptive_refinement_ladder"
    assert loaded["solve_summary"]["physics_refinement_mode"] == "two_stage_min_branch_l"
    assert loaded["solve_summary"]["physics_refinement_status"] == "refined"
    assert loaded["solve_summary"]["total_tiers_configured"] == 2
    assert loaded["solve_summary"]["used_physics_refinement_tier"] == 0
    assert loaded["solve_summary"]["physics_refinement_attempt_caps_eur"] == [2.0]
    assert loaded["solve_summary"]["returned_primary_objective_eur"] == pytest.approx(
        result.returned_primary_objective_eur
    )
    assert loaded["solve_summary"]["refinement_status_counts"] == {"refined": 1}
    assert loaded["solve_summary"]["floor_p95_soc_slack"] == pytest.approx(0.0)
    assert loaded["solve_summary"]["no_retry_or_fallback_used"] is True
    assert loaded["solve_summary"]["agent_purchase_cost_eur"] == pytest.approx(result.agent_purchase_cost_eur)
    assert loaded["solve_summary"]["physical_tiebreaker_weight"] == pytest.approx(
        cfg.mpc.branch_current_tiebreaker_eur_per_pu_step
    )


def test_replay_misocp_plan_package_returns_compare_ready_rollout(tmp_path):
    cfg, problem, full_input, result = _make_real_problem_fixture(tmp_path)
    household_csv = Path(cfg.data.data_dir) / "processed" / "prosumer" / "household.csv"
    if not household_csv.exists():
        write_prosumer_processed_dataset(
            cfg.data.data_dir,
            agent_profiles=list(cfg.data.agent_profiles),
            train_year=int(cfg.data.train_year),
            test_year=int(cfg.data.test_year),
            train_steps=int(cfg.env.episode_limit * 4),
            test_steps=int(cfg.env.episode_limit * 4),
        )
    saved_dir = save_misocp_plan_package(
        build_misocp_plan_package(
            problem,
            full_input,
            result,
            controller_label="Global MISOCP (single_window)",
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            cfg=cfg,
        ),
        tmp_path / "cached_plan",
    )

    replay = replay_misocp_plan_package(
        cfg,
        saved_dir,
        label="Global MISOCP (cached replay)",
    )

    rollout = replay["rollout"]
    assert not rollout.step_df.empty
    assert not rollout.agent_df.empty
    assert not rollout.grid_df.empty
    assert rollout.meta["loaded_from_cached_plan"] is True
    assert rollout.meta["plan_package_version"] == 11
    assert rollout.meta["diagnostics_stage"] == "commit4_adaptive_refinement_ladder"
    assert rollout.meta["physics_refinement_mode"] == "two_stage_min_branch_l"
    assert rollout.meta["no_retry_or_fallback_used"] is True
    assert rollout.meta["chunk_retry_count"] == 0
    assert rollout.meta["economics_scope"] == "agent_only"
    assert rollout.meta["validation_root_power_source"] == "trafo_p_signed_kw"
    assert rollout.meta["agent_q_base_zeroed"] is True
    assert rollout.meta["returned_solution_source"] == "stage2"
    assert rollout.meta["total_tiers_configured"] == 2
    assert rollout.meta["used_physics_refinement_tier"] == 0
    assert rollout.meta["physics_refinement_attempt_caps_eur"] == [2.0]
    assert rollout.meta["refinement_status_counts"] == {"refined": 1}
    assert "is_near_optimal" in rollout.meta
    assert rollout.meta["import_price_adder_eur_per_kwh"] == pytest.approx(
        cfg.reward.import_price_adder_eur_per_kwh
    )
    assert replay["validation_df"].shape[0] == rollout.step_df.shape[0]


def test_misocp_plan_package_rejects_version_and_cfg_mismatches(tmp_path):
    cfg, problem, full_input, result = _make_real_problem_fixture(tmp_path)
    saved_dir = save_misocp_plan_package(
        build_misocp_plan_package(
            problem,
            full_input,
            result,
            controller_label="Global MISOCP (single_window)",
            export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
            cfg=cfg,
        ),
        tmp_path / "cached_plan",
    )

    manifest_path = saved_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plan_package_version"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="adaptive slack ladder / tiered floor accept / floor-distance gate / time-budget refinement semantics"):
        load_misocp_plan_package(saved_dir)

    manifest["plan_package_version"] = 11
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cfg.grid.agent_bus_ids = list(reversed(cfg.grid.agent_bus_ids))
    with pytest.raises(ValueError, match="agent_bus_ids"):
        replay_misocp_plan_package(cfg, saved_dir)


def test_resolve_latest_compatible_misocp_plan_package_dir_prefers_newest_matching_prefix(tmp_path):
    cfg, problem, full_input, result = _make_real_problem_fixture(tmp_path)
    package = build_misocp_plan_package(
        problem,
        full_input,
        result,
        controller_label="Global MISOCP (single_window)",
        export_subsidy=float(cfg.reward.export_subsidy_eur_per_kwh),
        cfg=cfg,
    )
    plans_root = tmp_path / "cached_plans"
    base_dir = save_misocp_plan_package(package, plans_root / "2020-06-01_2020-06-05_agents5")
    newer_dir = save_misocp_plan_package(package, plans_root / "2020-06-01_2020-06-05_agents5_7ad699")

    base_manifest_path = base_dir / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_manifest["plan_package_version"] = 3
    base_manifest["saved_at_utc"] = "2026-01-01T00:00:00+00:00"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    newer_manifest_path = newer_dir / "manifest.json"
    newer_manifest = json.loads(newer_manifest_path.read_text(encoding="utf-8"))
    newer_manifest["saved_at_utc"] = "2026-04-15T12:00:00+00:00"
    newer_manifest_path.write_text(json.dumps(newer_manifest, indent=2), encoding="utf-8")

    resolved = resolve_latest_compatible_misocp_plan_package_dir(plans_root / "2020-06-01_2020-06-05_agents5")

    assert resolved == newer_dir.resolve()


def test_misocp_notebook_plot_helpers_return_figures():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()
    step_df = build_full_horizon_step_df(problem, full_input, result)
    step_df["timestamp"] = pd.date_range("2020-06-01", periods=len(step_df), freq="15min", tz="Europe/Berlin")
    step_df["pp_root_p_kw"] = np.asarray([17.5, 16.8], dtype=np.float32)
    voltage_df = build_voltage_df(problem, full_input, result)
    validation_df = build_misocp_validation_df(
        [
            {
                "controller": "Mock",
                "episode_idx": 0,
                "step": idx,
                "timestamp": timestamp,
                "solve_time_sec": 1.0,
                "misocp_vm_pu": np.asarray([1.0], dtype=np.float32),
                "pp_vm_pu": np.asarray([1.0], dtype=np.float32),
                "misocp_line_loading_pct": np.asarray([10.0], dtype=np.float32),
                "pp_line_loading_pct": np.asarray([10.0], dtype=np.float32),
                "misocp_trafo_loading_pct": np.asarray([20.0], dtype=np.float32),
                "pp_trafo_loading_pct": np.asarray([20.0], dtype=np.float32),
                "root_p_kw": float(10.0 + idx),
                "pp_root_p_kw": float(9.5 + idx),
                "pp_root_p_available": True,
                "misocp_root_s_kva": float(12.0 + idx),
                "pp_root_s_kva": float(11.5 + idx),
                "root_q_kvar": 4.0,
                "soc_slack_max": 0.01,
                "soc_slack_mean": 0.005,
                "soc_slack_p95_global": 0.02,
                "background_q_base_total_kvar": 4.0,
                "network_q_loss_proxy_kvar": 1.0,
                "root_q_residual_kvar": 0.0,
                "solver_feeder_gap_kw": 1.0,
                "replay_feeder_gap_kw": 0.5,
            }
            for idx, timestamp in enumerate(step_df["timestamp"])
        ]
    )

    balance_figure = plot_full_horizon_power_balance(
        step_df,
        controller_label="Global MISOCP (single_window)",
        dt_hours=problem.dt_hours,
    )
    voltage_figure = plot_full_horizon_voltage(
        step_df,
        voltage_df,
        agent_bus_ids=[1, 3],
        v_min_pu=0.95,
        v_max_pu=1.05,
        controller_label="Global MISOCP (single_window)",
    )
    net_load_figure = plot_full_horizon_net_load(
        step_df,
        trafo_limit_kw=problem.trafo_limit_mva * 1000.0,
        controller_label="Global MISOCP (single_window)",
    )
    root_alignment_figure = plot_root_exchange_alignment(
        step_df,
        controller_label="Global MISOCP (single_window)",
    )
    validation_scatter_figure = plot_misocp_validation_scatter_panel(
        validation_df,
        controller_label="Global MISOCP (single_window)",
    )

    assert len(balance_figure.axes) == 1
    assert len(balance_figure.axes[0].lines) >= 2
    assert len(balance_figure.axes[0].patches) > 0
    assert len(voltage_figure.axes) == 1
    assert len(net_load_figure.axes) == 2
    assert len(net_load_figure.axes[0].lines) >= 4
    assert len(net_load_figure.axes[1].lines) == 3
    assert len(root_alignment_figure.axes) == 1
    assert len(root_alignment_figure.axes[0].lines) >= 3
    assert len(validation_scatter_figure.axes) == 2
    plt.close(balance_figure)
    plt.close(voltage_figure)
    plt.close(net_load_figure)
    plt.close(root_alignment_figure)
    plt.close(validation_scatter_figure)


def test_misocp_global_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "MISOCP_global.ipynb"
    code_cells = _load_code_cells(notebook_path)
    notebook_text = notebook_path.read_text(encoding="utf-8")

    assert len(code_cells) >= 4
    notebook_source = "\n".join(code_cells)
    assert "AGENT_PROFILES =" not in notebook_source
    assert "RUN_FULL_HORIZON_BENCHMARK" not in notebook_source
    assert "RUN_SINGLE_WINDOW_BENCHMARK_AFTER_CHUNKED" in notebook_source
    assert "no_retry_or_fallback_used" in notebook_source
    assert "agent_purchase_cost_eur" in notebook_source
    assert "branch_current_tiebreaker_eur_per_pu_step" in notebook_source
    assert "max_root_p_abs_err_kw ~= 148.8" in notebook_text
    assert "max_root_p_abs_err_kw ~= 48" in notebook_text
    assert "agent_bus_q_base_nonzero_count" in notebook_source
    assert "p95_soc_slack" in notebook_source
    assert "root_p_fit_k" in notebook_source
    assert "root_s_fit_k" in notebook_source
    assert "build_soc_relaxation_diagnostics = misocp_nb.build_soc_relaxation_diagnostics" in notebook_source
    assert "plot_root_exchange_alignment = misocp_nb.plot_root_exchange_alignment" in notebook_source
    assert "plot_misocp_validation_scatter_panel = misocp_nb.plot_misocp_validation_scatter_panel" in notebook_source
    assert "import configs as configs_pkg" in notebook_source
    assert "config_profiles = importlib.reload(config_profiles)" in notebook_source
    assert "from controllers import mpc as mpc_pkg" in notebook_source
    assert "from controllers.mpc import global_socp_mpc as misocp_core" in notebook_source
    assert "misocp_core = importlib.reload(misocp_core)" in notebook_source
    assert "validate_misocp_result_schema = misocp_nb.validate_misocp_result_schema" in notebook_source
    assert "build_floor_diagnostic_summary" in notebook_source
    assert "build_refinement_summary" in notebook_source
    assert "floor_diagnostic_summary" in notebook_source
    assert "refinement_summary" in notebook_source
    assert "physics_refinement_slack_abs_floor_eur" in notebook_source
    assert "physics_refinement_slack_ratio_schedule" in notebook_source
    assert "physics_refinement_total_time_limit_sec" in notebook_source
    assert "physics_refinement_status" in notebook_source
    assert "returned_primary_objective_eur" in notebook_source
    assert "floor_accepted_tier" in notebook_source
    assert "used_physics_refinement_tier" in notebook_source
    assert "total_tiers_configured" in notebook_source
    assert "branch_l_gap_ratio_to_floor" in notebook_source
    assert "returned_mean_abs_export_gap_ratio" in notebook_source
    assert "high_budget_refinement_warn" in notebook_source
    assert "formulation_tightening_required" in notebook_source
    assert "refinement_status_counts" in notebook_source
    assert "from controllers.mpc import FullHorizonProblemInput, GlobalMISOCPProblem, GurobiSolveConfig" not in notebook_source
    assert "from controllers.mpc.global_socp_mpc import default_primary_solve_config, default_retry_solve_config" not in notebook_source
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_compare_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "compare.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 4
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")
