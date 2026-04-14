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
    build_misocp_validation_df,
    build_simultaneous_diagnostic_tables,
    build_voltage_df,
    estimate_global_misocp_model_size,
    format_solver_summary,
    load_misocp_plan_package,
    plot_full_horizon_net_load,
    plot_full_horizon_power_balance,
    plot_full_horizon_voltage,
    replay_misocp_plan_package,
    save_misocp_plan_package,
    summarize_misocp_validation,
    validate_misocp_result_schema,
)
from tests.support.helpers import make_case_dir, make_smoke_config


def _make_mock_problem() -> SimpleNamespace:
    network = SimpleNamespace(
        bus_ids=np.asarray([0, 1, 2, 3], dtype=np.int32),
        p_base_mw=np.asarray([0.005, 0.0, -0.002, 0.003], dtype=np.float32),
        agent_bus_positions=np.asarray([1, 3], dtype=np.int32),
        line_branch_indices=np.asarray([1, 2], dtype=np.int32),
        branch_is_trafo=np.asarray([True, False, False], dtype=bool),
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
        branch_p_pu=np.zeros((2, 2), dtype=np.float32),
        branch_q_pu=np.zeros((2, 2), dtype=np.float32),
        branch_i2_pu=np.asarray([[0.25, 0.42], [0.16, 0.64]], dtype=np.float32),
        bus_v_sq=np.zeros((4, 2), dtype=np.float32),
        root_import_mw=np.asarray([0.018, 0.017], dtype=np.float32),
        root_export_mw=np.asarray([0.0, 0.0], dtype=np.float32),
        root_p_kw=np.asarray([18.0, 17.0], dtype=np.float32),
        root_q_kvar=np.asarray([2.0, 3.0], dtype=np.float32),
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
    assert solver_summary["no_retry_or_fallback_used"]
    assert solver_summary["chunk_retry_count"] == 0
    assert solver_summary["total_runtime_sec"] == pytest.approx(result.solve_time_sec)
    assert np.isnan(float(solver_summary["max_chunk_runtime_sec"]))


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
    assert loaded["manifest"]["plan_package_version"] == 7
    assert loaded["manifest"]["cfg_snapshot"]["import_price_adder_eur_per_kwh"] == pytest.approx(
        cfg.reward.import_price_adder_eur_per_kwh
    )
    assert loaded["manifest"]["validation_root_power_source"] == "trafo_p_signed_kw"
    assert loaded["manifest"]["agent_q_base_zeroed"] is True
    assert loaded["full_input"].horizon_steps == full_input.horizon_steps
    assert np.allclose(loaded["result"].battery_charge_mw, result.battery_charge_mw)
    assert loaded["solve_summary"]["status_label"] == "optimal"
    assert loaded["solve_summary"]["economics_scope"] == "agent_only"
    assert loaded["solve_summary"]["no_retry_or_fallback_used"] is True
    assert loaded["solve_summary"]["agent_purchase_cost_eur"] == pytest.approx(result.agent_purchase_cost_eur)
    assert loaded["solve_summary"]["physical_tiebreaker_weight"] == pytest.approx(
        cfg.mpc.branch_current_tiebreaker_eur_per_pu_step
    )


def test_replay_misocp_plan_package_returns_compare_ready_rollout(tmp_path):
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
    assert rollout.meta["plan_package_version"] == 7
    assert rollout.meta["no_retry_or_fallback_used"] is True
    assert rollout.meta["chunk_retry_count"] == 0
    assert rollout.meta["economics_scope"] == "agent_only"
    assert rollout.meta["validation_root_power_source"] == "trafo_p_signed_kw"
    assert rollout.meta["agent_q_base_zeroed"] is True
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
    with pytest.raises(ValueError, match="root-power validation fix"):
        load_misocp_plan_package(saved_dir)

    manifest["plan_package_version"] = 7
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cfg.grid.agent_bus_ids = list(reversed(cfg.grid.agent_bus_ids))
    with pytest.raises(ValueError, match="agent_bus_ids"):
        replay_misocp_plan_package(cfg, saved_dir)


def test_misocp_notebook_plot_helpers_return_figures():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()
    step_df = build_full_horizon_step_df(problem, full_input, result)
    step_df["timestamp"] = pd.date_range("2020-06-01", periods=len(step_df), freq="15min", tz="Europe/Berlin")
    voltage_df = build_voltage_df(problem, full_input, result)

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

    assert len(balance_figure.axes) == 1
    assert len(balance_figure.axes[0].lines) >= 2
    assert len(balance_figure.axes[0].patches) > 0
    assert len(voltage_figure.axes) == 1
    assert len(net_load_figure.axes) == 2
    assert len(net_load_figure.axes[0].lines) >= 4
    assert len(net_load_figure.axes[1].lines) == 3
    plt.close(balance_figure)
    plt.close(voltage_figure)
    plt.close(net_load_figure)


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
    assert "agent_bus_q_base_nonzero_count" in notebook_source
    assert "from controllers import mpc as mpc_pkg" in notebook_source
    assert "from controllers.mpc import global_socp_mpc as misocp_core" in notebook_source
    assert "misocp_core = importlib.reload(misocp_core)" in notebook_source
    assert "validate_misocp_result_schema = misocp_nb.validate_misocp_result_schema" in notebook_source
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
