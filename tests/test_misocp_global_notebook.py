from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from controllers.mpc.global_socp_mpc import GlobalMISOCPProblem
from scripts.builder import build_env
from scripts.mainline_compare import (
    build_chunk_boundary_soc_df,
    build_full_horizon_step_df,
    build_root_q_diagnostic_df,
    build_soc_relaxation_diagnostics,
    build_misocp_validation_df,
    format_solver_summary,
    summarize_misocp_validation,
    validate_misocp_result_schema,
)
from tests.support.helpers import make_case_dir, make_smoke_config


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
        wholesale_price_seq=np.asarray([-0.10, 0.00], dtype=np.float32),
        import_price_seq=np.asarray([0.10, 0.20], dtype=np.float32),
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
        storage_purchase_cost_eur=0.125,
        storage_sale_revenue_eur=0.1,
        storage_total_profit_eur=-0.025,
        storage_charge_cost_eur=0.125,
        storage_discharge_revenue_eur=0.1,
        storage_profit_eur=-0.025,
        storage_objective_eur=0.025,
        system_other_cost_eur=1.275,
        total_eur=-1.3,
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
        stage1_primary_objective_eur=0.885,
    )
def test_build_full_horizon_input_supports_contiguous_subranges_and_rejects_noncontiguous_indices(tmp_path):
    case_dir = make_case_dir(tmp_path, "misocp_full_horizon_indices")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    cfg.data.test_start_date = "2020-01-01"
    cfg.data.test_end_date = "2020-01-01"
    env = build_env(cfg, mode="test")
    try:
        problem = GlobalMISOCPProblem.from_env(env, cfg)
        contiguous = problem.build_full_horizon_input(env, episode_indices=[1, 2])
        assert contiguous.episode_indices.tolist() == [1, 2]
        assert contiguous.episode_offsets.tolist() == [0, cfg.env.episode_limit]

        with pytest.raises(ValueError, match="contiguous episode index ranges"):
            problem.build_full_horizon_input(env, episode_indices=[0, 2])
    finally:
        env.close()


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
        "wholesale_price",
        "import_price",
        "fixed_load_kw",
        "fixed_generation_kw",
        "agent_load_kw",
        "pv_raw_kw",
        "pv_curtail_kw",
        "pv_effective_kw",
        "battery_charge_kw",
        "battery_discharge_kw",
        "storage_purchase_cost_eur_step",
        "storage_sale_revenue_eur_step",
        "storage_total_profit_eur_step",
        "system_other_cost_eur_step",
        "total_eur_step",
        "agent_import_kw_total",
        "agent_export_kw_total",
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
    assert step_df["storage_purchase_cost_eur_step"].sum() == pytest.approx(result.storage_purchase_cost_eur)
    assert step_df["storage_sale_revenue_eur_step"].sum() == pytest.approx(result.storage_sale_revenue_eur)
    assert step_df["storage_total_profit_eur_step"].sum() == pytest.approx(result.storage_total_profit_eur)
    assert step_df["system_other_cost_eur_step"].sum() == pytest.approx(result.system_other_cost_eur)
    assert step_df["total_eur_step"].sum() == pytest.approx(result.total_eur)
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


def test_system_other_cost_reconciles_storage_profit_with_total():
    problem = _make_mock_problem()
    full_input = _make_mock_full_input()
    result = _make_mock_result()

    step_df = build_full_horizon_step_df(problem, full_input, result)
    assert "agent_purchase_cost_eur_step" not in step_df.columns
    assert "feeder_purchase_cost_eur_step" not in step_df.columns
    assert float(step_df["total_eur_step"].sum()) == pytest.approx(
        float(step_df["storage_total_profit_eur_step"].sum() - step_df["system_other_cost_eur_step"].sum())
    )


def test_solver_summary_helper_returns_expected_fields():
    result = _make_mock_result()

    solver_summary = format_solver_summary(result, total_steps=2, episode_count=1)

    assert solver_summary["status_label"] == "optimal"
    assert solver_summary["has_solution"] is True
    assert solver_summary["sol_count"] == 1
    assert solver_summary["node_count"] == 12.0
    assert solver_summary["num_quadratic_constraints"] == result.model_size.num_quadratic_constraints
    assert solver_summary["economics_scope"] == "storage_only"
    assert solver_summary["storage_purchase_cost_eur"] == pytest.approx(result.storage_purchase_cost_eur)
    assert solver_summary["storage_sale_revenue_eur"] == pytest.approx(result.storage_sale_revenue_eur)
    assert solver_summary["storage_total_profit_eur"] == pytest.approx(result.storage_total_profit_eur)
    assert solver_summary["system_other_cost_eur"] == pytest.approx(result.system_other_cost_eur)
    assert solver_summary["total_eur"] == pytest.approx(result.total_eur)
    assert solver_summary["physical_tiebreaker_weight"] == pytest.approx(1e-6)
    assert solver_summary["physical_tiebreaker_eur"] == pytest.approx(0.0015)
    assert solver_summary["stage1_primary_objective_eur"] == pytest.approx(0.885)
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


def test_validation_helpers_return_expected_tables():
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
    assert validation_df.loc[0, "within_tolerance"]
    assert validation_summary["steps_outside_tolerance"] == 0
    assert validation_summary["pp_root_p_available_ratio"] == pytest.approx(1.0)


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
    assert chunk_boundary_soc_df.empty


def test_misocp_global_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "global_MISOCP.ipynb"
    code_cells = _load_code_cells(notebook_path)
    notebook_source = "\n".join(code_cells)

    assert len(code_cells) >= 2
    assert "collect_global_full_horizon_rollout" in notebook_source
    assert "collect_global_full_horizon_rollout(cfg, label='Global MISOCP')" in notebook_source
    assert "save_rollout_record(" in notebook_source
    assert "scheme_name='global_misocp'" in notebook_source
    assert "compare_rollout_metrics(rollout)" in notebook_source
    assert "plot_global_misocp_validation(rollout)" in notebook_source
    assert "RUN_FULL_HORIZON_BENCHMARK" not in notebook_source
    assert "RUN_SINGLE_WINDOW_BENCHMARK_AFTER_CHUNKED" not in notebook_source
    assert "SAVE_MISOCP_PLAN" not in notebook_source
    assert "build_misocp_plan_package" not in notebook_source
    assert "importlib.reload(" not in notebook_source
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


