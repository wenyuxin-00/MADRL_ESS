from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN, IMPORT_PRICE_MARKUP_KEY, IMPORT_PRICE_PRED_COLUMN, PRICE_PROTOCOL_VERSION, WHOLESALE_PRICE_PRED_COLUMN, WHOLESALE_PRICE_SEQ_FIELD, derive_import_price
from scripts.utils.misocp_plan_packages import build_misocp_plan_package, load_misocp_plan_package, replay_misocp_plan_package, resolve_exact_misocp_plan_package_dir, save_misocp_plan_package
from scripts.utils.misocp_notebook_diagnostics import build_misocp_validation_df, plot_full_horizon_net_load, plot_full_horizon_power_balance, plot_full_horizon_voltage, plot_misocp_validation_scatter_panel, plot_root_exchange_alignment, summarize_misocp_validation
_SOC_SLACK_EPS = 1e-06
_SOC_SLACK_TIGHT_P95_THRESHOLD = 0.0001
def expand_episode_indices(full_input: Any) -> np.ndarray:
    lengths = np.asarray(full_input.episode_lengths, dtype=np.int32).reshape(-1)
    indices = np.asarray(full_input.episode_indices, dtype=np.int32).reshape(-1)
    if lengths.size != indices.size:
        raise ValueError('episode_lengths and episode_indices must have the same length.')
    pieces = [np.full((int(length),), int(episode_idx), dtype=np.int32) for length, episode_idx in zip(lengths.tolist(), indices.tolist(), strict=False) if int(length) > 0]
    if not pieces:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(pieces, axis=0)
def _require_solution_matrix(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Result is missing required solution field '{name}'.")
    return np.asarray(value, dtype=np.float32)
def validate_misocp_result_schema(result: Any) -> None:
    required_fields = ('agent_import_mw', 'agent_export_mw', 'agent_net_grid_mw', 'agent_purchase_cost_eur', 'agent_export_subsidy_eur', 'agent_net_cost_eur')
    missing_fields = [field for field in required_fields if not hasattr(result, field)]
    if missing_fields:
        missing_display = ', '.join(missing_fields)
        raise ValueError(f"solve_result uses an outdated MISOCPResult schema; missing fields: {missing_display}. This usually means the notebook kernel still holds an older 'controllers.mpc.global_socp_mpc' module. Re-run the first import cell and the solve cell, or restart the kernel before rebuilding notebook diagnostics.")
def _timestamps_from_full_input(full_input: Any) -> pd.Index:
    raw = pd.Index([str(value) for value in tuple(full_input.timestamps)], name='timestamp')
    parsed = pd.to_datetime(raw, errors='coerce')
    return parsed if parsed.notna().all() else raw
def _line_loading_max(line_loading_pct: np.ndarray, horizon_steps: int) -> np.ndarray:
    if line_loading_pct.size == 0:
        return np.zeros((horizon_steps,), dtype=np.float32)
    return np.max(line_loading_pct, axis=0).astype(np.float32)
def _agent_bus_ids_from_problem(problem: Any) -> list[int]:
    bus_ids = np.asarray(problem.network.bus_ids, dtype=np.int32).reshape(-1)
    agent_positions = np.asarray(problem.network.agent_bus_positions, dtype=np.int32).reshape(-1)
    return [int(bus_ids[int(position)]) for position in agent_positions.tolist()]
def _fixed_feeder_components_from_problem(problem: Any) -> tuple[float, float]:
    fixed_active_kw = np.asarray(problem.network.p_base_mw, dtype=np.float32).reshape(-1) * 1000.0
    fixed_load_kw = float(np.clip(fixed_active_kw, 0.0, None).sum())
    fixed_generation_kw = float(np.clip(-fixed_active_kw, 0.0, None).sum())
    return (fixed_load_kw, fixed_generation_kw)
def _series_or_default(frame: pd.DataFrame, column: str, default: float=0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full((len(frame),), float(default), dtype=np.float32)
    return frame[column].to_numpy(dtype=np.float32)
def _resolve_agent_net_load_columns(step_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {'agent_raw_net_load_kw', 'agent_effective_net_load_kw', 'agent_post_action_net_load_kw'}.issubset(step_df.columns):
        return (step_df['agent_raw_net_load_kw'].to_numpy(dtype=np.float32), step_df['agent_effective_net_load_kw'].to_numpy(dtype=np.float32), step_df['agent_post_action_net_load_kw'].to_numpy(dtype=np.float32))
    return (_series_or_default(step_df, 'base_net_load_total'), _series_or_default(step_df, 'base_net_load_effective_total'), _series_or_default(step_df, 'net_load_total'))
def _resolve_feeder_net_load_columns(step_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if {'feeder_raw_net_load_kw', 'feeder_effective_net_load_kw', 'feeder_post_action_net_load_kw'}.issubset(step_df.columns):
        return (step_df['feeder_raw_net_load_kw'].to_numpy(dtype=np.float32), step_df['feeder_effective_net_load_kw'].to_numpy(dtype=np.float32), step_df['feeder_post_action_net_load_kw'].to_numpy(dtype=np.float32))
    if {'fixed_load_kw', 'fixed_generation_kw'}.issubset(step_df.columns):
        agent_raw, agent_effective, agent_post_action = _resolve_agent_net_load_columns(step_df)
        fixed_load = _series_or_default(step_df, 'fixed_load_kw')
        fixed_generation = _series_or_default(step_df, 'fixed_generation_kw')
        return ((agent_raw + fixed_load - fixed_generation).astype(np.float32), (agent_effective + fixed_load - fixed_generation).astype(np.float32), (agent_post_action + fixed_load - fixed_generation).astype(np.float32))
    return (None, None, None)
def estimate_global_misocp_model_size(problem: Any, horizon_steps: int, *, episode_count: int | None=None, enforce_terminal_soc: bool=True) -> pd.Series:
    horizon_steps = int(horizon_steps)
    if horizon_steps < 0:
        raise ValueError('horizon_steps must be non-negative.')
    n_agents = int(problem.n_agents)
    n_buses = int(problem.n_buses)
    n_branches = int(problem.n_branches)
    line_branch_indices = getattr(problem.network, 'line_branch_indices', None)
    if line_branch_indices is None:
        branch_is_trafo = np.asarray(getattr(problem.network, 'branch_is_trafo', np.zeros((n_branches,), dtype=bool)), dtype=bool)
        n_lines = int(np.sum(~branch_is_trafo))
    else:
        n_lines = int(np.asarray(line_branch_indices, dtype=np.int32).size)
    num_vars_est = 5 * n_agents * horizon_steps + n_agents + 3 * n_branches * horizon_steps + n_buses * horizon_steps + 3 * horizon_steps
    num_binary_vars_est = horizon_steps
    num_linear_constraints_est = 3 * horizon_steps + n_agents * (8 * horizon_steps + 3 + int(bool(enforce_terminal_soc))) + horizon_steps * (3 * n_branches + n_lines + 1)
    num_quadratic_constraints_est = horizon_steps * (n_branches + 1)
    return pd.Series({'horizon_steps': horizon_steps, 'episode_count': np.nan if episode_count is None else int(episode_count), 'n_agents': n_agents, 'n_buses': n_buses, 'n_branches': n_branches, 'n_lines': n_lines, 'num_vars_est': int(num_vars_est), 'num_binary_vars_est': int(num_binary_vars_est), 'num_linear_constraints_est': int(num_linear_constraints_est), 'num_quadratic_constraints_est': int(num_quadratic_constraints_est)}, name='estimated_model_size')
def format_solver_summary(result: Any, *, total_steps: int | None=None, episode_count: int | None=None) -> pd.Series:
    horizon_steps = int(total_steps if total_steps is not None else getattr(result, 'horizon_steps', 0))
    summary_note = ''
    if bool(getattr(result, 'time_limit_feasible', False)):
        summary_note = 'Reached time limit with an incumbent; objective is feasible but not proven optimal.'
    elif not bool(getattr(result, 'has_solution', False)):
        summary_note = 'Solver returned no incumbent; inspect debug artifacts and raw log for the failure mode.'
    chunk_summaries = list(getattr(result, 'chunk_summaries', []) or [])
    chunk_retry_count = int(getattr(result, 'chunk_retry_count', sum((1 for item in chunk_summaries if str(item.get('attempt_used', '')) == 'retry'))))
    no_retry_or_fallback_used = bool(getattr(result, 'no_retry_or_fallback_used', bool(chunk_retry_count == 0)))
    total_runtime_sec = float(getattr(result, 'solve_time_sec', float('nan')))
    finite_chunk_runtimes = [float(item.get('solve_time_sec', float('nan'))) for item in chunk_summaries if np.isfinite(float(item.get('solve_time_sec', float('nan'))))]
    max_chunk_runtime_sec = float(max(finite_chunk_runtimes)) if finite_chunk_runtimes else float('nan')
    return pd.Series({'status_code': int(result.status_code), 'status_label': str(result.status_label), 'has_solution': bool(result.has_solution), 'time_limit_feasible': bool(result.time_limit_feasible), 'solve_mode': str(result.solve_mode), 'horizon_steps': int(horizon_steps), 'episode_count': np.nan if episode_count is None else int(episode_count), 'solve_time_sec': float(result.solve_time_sec), 'total_runtime_sec': total_runtime_sec, 'max_chunk_runtime_sec': max_chunk_runtime_sec, 'mip_gap': float(result.mip_gap), 'best_bound': float(result.best_bound), 'objective_value': float(result.objective_value), 'sol_count': int(result.sol_count), 'node_count': float(result.node_count), 'iter_count': float(result.iter_count), 'bar_iter_count': float(result.bar_iter_count), 'is_near_optimal': bool(str(getattr(result, 'solve_mode', '')) == 'chunked_window'), 'economics_scope': 'agent_only', 'physical_tiebreaker_eur': float(getattr(result, 'physical_tiebreaker_eur', float('nan'))), 'physical_tiebreaker_weight': float(getattr(result, 'physical_tiebreaker_weight', float('nan'))), 'stage1_primary_objective_eur': float(getattr(result, 'stage1_primary_objective_eur', float('nan'))), 'stage2_primary_objective_eur': float(getattr(result, 'stage2_primary_objective_eur', float('nan'))), 'stage2_objective_slack_eur': float(getattr(result, 'stage2_objective_slack_eur', float('nan'))), 'stage2_branch_l_objective': float(getattr(result, 'stage2_branch_l_objective', float('nan'))), 'physics_refinement_mode': str(getattr(result, 'physics_refinement_mode', 'none')), 'physics_refinement_status': str(getattr(result, 'physics_refinement_status', 'not_enabled')), 'physics_refinement_runtime_sec': float(getattr(result, 'physics_refinement_runtime_sec', 0.0)), 'floor_p95_soc_slack': float(getattr(result, 'floor_p95_soc_slack', float('nan'))), 'floor_mean_abs_solver_feeder_gap_kw': float(getattr(result, 'floor_mean_abs_solver_feeder_gap_kw', float('nan'))), 'floor_primary_objective_eur': float(getattr(result, 'floor_primary_objective_eur', float('nan'))), 'floor_primary_delta_signed_eur': float(getattr(result, 'floor_primary_delta_signed_eur', float('nan'))), 'floor_primary_delta_positive_eur': float(getattr(result, 'floor_primary_delta_positive_eur', float('nan'))), 'physics_refinement_slack_cap_eur': float(getattr(result, 'physics_refinement_slack_cap_eur', float('nan'))), 'initial_physics_refinement_slack_cap_eur': float(getattr(result, 'initial_physics_refinement_slack_cap_eur', float('nan'))), 'returned_primary_objective_eur': float(getattr(result, 'returned_primary_objective_eur', float('nan'))), 'returned_primary_delta_abs_eur': float(getattr(result, 'returned_primary_delta_abs_eur', float('nan'))), 'returned_primary_delta_pct': float(getattr(result, 'returned_primary_delta_pct', float('nan'))), 'floor_accepted_tier': getattr(result, 'floor_accepted_tier', None), 'used_physics_refinement_tier': getattr(result, 'used_physics_refinement_tier', None), 'total_tiers_configured': int(getattr(result, 'total_tiers_configured', 0)), 'physics_refinement_attempt_count': int(getattr(result, 'physics_refinement_attempt_count', 0)), 'physics_refinement_attempt_caps_eur': [float(value) for value in list(getattr(result, 'physics_refinement_attempt_caps_eur', None) or [])], 'physics_refinement_cap_utilization': float(getattr(result, 'physics_refinement_cap_utilization', float('nan'))), 'branch_l_gap_ratio_to_floor': float(getattr(result, 'branch_l_gap_ratio_to_floor', float('nan'))), 'returned_mean_abs_solver_feeder_gap_kw': float(getattr(result, 'returned_mean_abs_solver_feeder_gap_kw', float('nan'))), 'returned_max_solver_feeder_gap_kw': float(getattr(result, 'returned_max_solver_feeder_gap_kw', float('nan'))), 'returned_mean_abs_export_gap_ratio': float(getattr(result, 'returned_mean_abs_export_gap_ratio', float('nan'))), 'high_budget_refinement_warn': bool(getattr(result, 'high_budget_refinement_warn', False)), 'returned_solution_source': str(getattr(result, 'returned_solution_source', 'stage1')), 'formulation_tightening_required': bool(getattr(result, 'formulation_tightening_required', False)), 'negative_floor_delta_warn': bool(getattr(result, 'negative_floor_delta_warn', False)), 'refinement_status_counts': {str(key): int(value) for key, value in dict(getattr(result, 'refinement_status_counts', None) or {}).items()}, 'no_retry_or_fallback_used': no_retry_or_fallback_used, 'chunk_retry_count': chunk_retry_count, 'chunk_count': int(len(chunk_summaries)), 'num_binary_vars': int(result.model_size.num_binary_vars), 'num_quadratic_constraints': int(result.model_size.num_quadratic_constraints), 'solver_note': summary_note}, name='solver_summary')
def build_full_horizon_step_df(problem: Any, full_input: Any, result: Any) -> pd.DataFrame:
    validate_misocp_result_schema(result)
    load_seq = np.asarray(full_input.load_seq, dtype=np.float32)
    pv_seq = np.asarray(full_input.pv_seq, dtype=np.float32)
    wholesale_price_seq = np.asarray(full_input.wholesale_price_seq, dtype=np.float32).reshape(-1)
    import_price_seq = np.asarray(full_input.import_price_seq, dtype=np.float32).reshape(-1)
    episode_idx = expand_episode_indices(full_input)
    timestamps = _timestamps_from_full_input(full_input)
    horizon_steps = int(import_price_seq.shape[0])
    if episode_idx.shape[0] != horizon_steps:
        raise ValueError('Expanded episode index series does not match the full-horizon length.')
    if load_seq.shape[1] != horizon_steps or pv_seq.shape[1] != horizon_steps:
        raise ValueError('load_seq and pv_seq must match the full-horizon length.')
    if wholesale_price_seq.shape[0] != horizon_steps:
        raise ValueError('wholesale_price_seq must match the full-horizon length.')
    battery_charge_mw = _require_solution_matrix(result.battery_charge_mw, 'battery_charge_mw')
    battery_discharge_mw = _require_solution_matrix(result.battery_discharge_mw, 'battery_discharge_mw')
    pv_curtail_mw = _require_solution_matrix(result.pv_curtail_mw, 'pv_curtail_mw')
    bus_vm_pu = _require_solution_matrix(result.bus_vm_pu, 'bus_vm_pu')
    energy_mwh = _require_solution_matrix(result.energy_mwh, 'energy_mwh')
    line_loading_pct = _require_solution_matrix(result.line_loading_pct, 'line_loading_pct')
    trafo_loading_pct = _require_solution_matrix(result.trafo_loading_pct, 'trafo_loading_pct')
    root_import_mw = _require_solution_matrix(result.root_import_mw, 'root_import_mw').reshape(-1)
    root_export_mw = _require_solution_matrix(result.root_export_mw, 'root_export_mw').reshape(-1)
    agent_import_mw = _require_solution_matrix(result.agent_import_mw, 'agent_import_mw')
    agent_export_mw = _require_solution_matrix(result.agent_export_mw, 'agent_export_mw')
    agent_net_grid_mw = _require_solution_matrix(result.agent_net_grid_mw, 'agent_net_grid_mw')
    simultaneous_kw = _require_solution_matrix(result.simultaneous_charge_discharge_kw, 'simultaneous_charge_discharge_kw')
    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_from_problem(problem)
    agent_load_kw = load_seq.sum(axis=0).astype(np.float32)
    pv_raw_kw = pv_seq.sum(axis=0).astype(np.float32)
    pv_curtail_kw = (pv_curtail_mw.sum(axis=0) * 1000.0).astype(np.float32)
    pv_effective_kw = np.maximum(pv_raw_kw - pv_curtail_kw, 0.0).astype(np.float32)
    battery_charge_kw = (battery_charge_mw.sum(axis=0) * 1000.0).astype(np.float32)
    battery_discharge_kw = (battery_discharge_mw.sum(axis=0) * 1000.0).astype(np.float32)
    grid_import_kw = (root_import_mw * 1000.0).astype(np.float32)
    grid_export_kw = (root_export_mw * 1000.0).astype(np.float32)
    root_net_exchange_kw = (grid_import_kw - grid_export_kw).astype(np.float32)
    agent_import_kw_total = (agent_import_mw.sum(axis=0) * 1000.0).astype(np.float32)
    agent_export_kw_total = (agent_export_mw.sum(axis=0) * 1000.0).astype(np.float32)
    agent_net_exchange_kw_total = (agent_net_grid_mw.sum(axis=0) * 1000.0).astype(np.float32)
    agent_raw_net_load_kw = (agent_load_kw - pv_raw_kw).astype(np.float32)
    agent_effective_net_load_kw = (agent_load_kw - pv_effective_kw).astype(np.float32)
    agent_post_action_net_load_kw = (agent_effective_net_load_kw + battery_charge_kw - battery_discharge_kw).astype(np.float32)
    feeder_raw_net_load_kw = (agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw).astype(np.float32)
    feeder_effective_net_load_kw = (agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw).astype(np.float32)
    feeder_post_action_net_load_kw = (agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw).astype(np.float32)
    balance_supply_kw = (pv_effective_kw + fixed_generation_kw + battery_discharge_kw + grid_import_kw).astype(np.float32)
    balance_demand_kw = (agent_load_kw + fixed_load_kw + battery_charge_kw + grid_export_kw).astype(np.float32)
    balance_residual_kw = (balance_supply_kw - balance_demand_kw).astype(np.float32)
    agent_export_rate = float(result.agent_export_subsidy_eur) / max(float(np.sum(agent_export_kw_total * np.float32(problem.dt_hours))), 1e-06) if float(np.sum(agent_export_kw_total)) > 0.0 else 0.0
    feeder_export_rate = float(result.feeder_export_subsidy_eur) / max(float(np.sum(grid_export_kw * np.float32(problem.dt_hours))), 1e-06) if float(np.sum(grid_export_kw)) > 0.0 else 0.0
    agent_purchase_cost_eur_step = (agent_import_kw_total * np.float32(problem.dt_hours) * import_price_seq.astype(np.float32)).astype(np.float32)
    agent_export_subsidy_eur_step = (agent_export_kw_total * np.float32(problem.dt_hours) * np.float32(agent_export_rate)).astype(np.float32)
    agent_net_cost_eur_step = (agent_purchase_cost_eur_step - agent_export_subsidy_eur_step).astype(np.float32)
    feeder_purchase_cost_eur_step = (grid_import_kw * np.float32(problem.dt_hours) * import_price_seq.astype(np.float32)).astype(np.float32)
    feeder_export_subsidy_eur_step = (grid_export_kw * np.float32(problem.dt_hours) * np.float32(feeder_export_rate)).astype(np.float32)
    feeder_net_cost_eur_step = (feeder_purchase_cost_eur_step - feeder_export_subsidy_eur_step).astype(np.float32)
    agent_root_gap_kw = (root_net_exchange_kw - agent_post_action_net_load_kw).astype(np.float32)
    step_df = pd.DataFrame({'timestamp': timestamps, 'episode_idx': episode_idx, 'global_step': np.arange(horizon_steps, dtype=np.int32), 'wholesale_price': wholesale_price_seq.astype(np.float32), IMPORT_PRICE_COLUMN: import_price_seq.astype(np.float32), 'fixed_load_kw': np.full((horizon_steps,), fixed_load_kw, dtype=np.float32), 'fixed_generation_kw': np.full((horizon_steps,), fixed_generation_kw, dtype=np.float32), 'agent_load_kw': agent_load_kw, 'pv_raw_kw': pv_raw_kw, 'pv_curtail_kw': pv_curtail_kw, 'pv_effective_kw': pv_effective_kw, 'battery_charge_kw': battery_charge_kw, 'battery_discharge_kw': battery_discharge_kw, 'agent_import_kw_total': agent_import_kw_total, 'agent_export_kw_total': agent_export_kw_total, 'agent_net_exchange_kw_total': agent_net_exchange_kw_total, 'grid_import_kw': grid_import_kw, 'grid_export_kw': grid_export_kw, 'agent_purchase_cost_eur_step': agent_purchase_cost_eur_step, 'agent_export_subsidy_eur_step': agent_export_subsidy_eur_step, 'agent_net_cost_eur_step': agent_net_cost_eur_step, 'feeder_purchase_cost_eur_step': feeder_purchase_cost_eur_step, 'feeder_export_subsidy_eur_step': feeder_export_subsidy_eur_step, 'feeder_net_cost_eur_step': feeder_net_cost_eur_step, 'agent_raw_net_load_kw': agent_raw_net_load_kw, 'agent_effective_net_load_kw': agent_effective_net_load_kw, 'agent_post_action_net_load_kw': agent_post_action_net_load_kw, 'feeder_raw_net_load_kw': feeder_raw_net_load_kw, 'feeder_effective_net_load_kw': feeder_effective_net_load_kw, 'feeder_post_action_net_load_kw': feeder_post_action_net_load_kw, 'raw_net_load_kw': feeder_raw_net_load_kw, 'effective_net_load_kw': feeder_effective_net_load_kw, 'post_action_net_load_kw': feeder_post_action_net_load_kw, 'root_net_exchange_kw': root_net_exchange_kw, 'agent_root_gap_kw': agent_root_gap_kw, 'balance_supply_kw': balance_supply_kw, 'balance_demand_kw': balance_demand_kw, 'balance_residual_kw': balance_residual_kw, 'network_loss_kw_estimate': balance_residual_kw, 'aggregate_stored_energy_kwh': np.sum(energy_mwh[:, 1:], axis=0).astype(np.float32) * 1000.0, 'trafo_limit_reference_kw': np.full((horizon_steps,), float(problem.trafo_limit_mva) * 1000.0, dtype=np.float32), 'min_vm_pu': np.min(bus_vm_pu, axis=0).astype(np.float32), 'mean_vm_pu': np.mean(bus_vm_pu, axis=0).astype(np.float32), 'max_vm_pu': np.max(bus_vm_pu, axis=0).astype(np.float32), 'max_line_loading_pct': _line_loading_max(line_loading_pct, horizon_steps), 'trafo_loading_pct': np.asarray(trafo_loading_pct, dtype=np.float32).reshape(-1)[:horizon_steps], 'simultaneous_kw_total': np.sum(simultaneous_kw, axis=0).astype(np.float32)})
    return step_df
def build_voltage_df(problem: Any, full_input: Any, result: Any) -> pd.DataFrame:
    bus_vm_pu = _require_solution_matrix(result.bus_vm_pu, 'bus_vm_pu')
    episode_idx = expand_episode_indices(full_input)
    timestamps = _timestamps_from_full_input(full_input)
    bus_ids = np.asarray(problem.network.bus_ids, dtype=np.int32).reshape(-1)
    horizon_steps = int(bus_vm_pu.shape[1])
    if episode_idx.shape[0] != horizon_steps:
        raise ValueError('Expanded episode index series does not match bus_vm_pu horizon.')
    if bus_ids.shape[0] != bus_vm_pu.shape[0]:
        raise ValueError('bus_vm_pu bus dimension does not match network bus_ids.')
    agent_bus_set = set(_agent_bus_ids_from_problem(problem))
    step_repeats = np.repeat(np.arange(horizon_steps, dtype=np.int32), bus_ids.shape[0])
    episode_repeats = np.repeat(episode_idx.astype(np.int32), bus_ids.shape[0])
    timestamp_repeats = np.repeat(np.asarray(timestamps, dtype=object), bus_ids.shape[0])
    bus_tile = np.tile(bus_ids.astype(np.int32), horizon_steps)
    vm_flat = bus_vm_pu.T.reshape(-1).astype(np.float32)
    is_agent_bus = np.asarray([int(bus_id) in agent_bus_set for bus_id in bus_tile], dtype=bool)
    return pd.DataFrame({'timestamp': timestamp_repeats, 'episode_idx': episode_repeats, 'global_step': step_repeats, 'bus_id': bus_tile, 'vm_pu': vm_flat, 'is_agent_bus': is_agent_bus})
def build_debug_tables(problem: Any, full_input: Any, result: Any, *, step_df: pd.DataFrame | None=None, agent_profiles: list[str] | tuple[str, ...] | None=None, agent_bus_ids: list[int] | tuple[int, ...] | None=None, top_k: int=12) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step_df = build_full_horizon_step_df(problem, full_input, result) if step_df is None else step_df.copy()
    bus_vm_pu = _require_solution_matrix(result.bus_vm_pu, 'bus_vm_pu')
    battery_charge_mw = _require_solution_matrix(result.battery_charge_mw, 'battery_charge_mw')
    battery_discharge_mw = _require_solution_matrix(result.battery_discharge_mw, 'battery_discharge_mw')
    pv_curtail_mw = _require_solution_matrix(result.pv_curtail_mw, 'pv_curtail_mw')
    energy_mwh = _require_solution_matrix(result.energy_mwh, 'energy_mwh')
    resolved_profiles = list(agent_profiles or [f'agent_{idx}' for idx in range(energy_mwh.shape[0])])
    if len(resolved_profiles) != energy_mwh.shape[0]:
        raise ValueError('agent_profiles length must match the number of agents in the solution.')
    resolved_bus_ids = list(agent_bus_ids or _agent_bus_ids_from_problem(problem))
    if len(resolved_bus_ids) != energy_mwh.shape[0]:
        raise ValueError('agent_bus_ids length must match the number of agents in the solution.')
    solve_summary = pd.Series({**format_solver_summary(result, total_steps=int(step_df.shape[0]), episode_count=int(np.asarray(full_input.episode_lengths, dtype=np.int32).size)).to_dict(), 'agent_purchase_cost_eur': float(result.agent_purchase_cost_eur), 'agent_export_subsidy_eur': float(result.agent_export_subsidy_eur), 'agent_net_cost_eur': float(result.agent_net_cost_eur), 'feeder_purchase_cost_eur': float(result.feeder_purchase_cost_eur), 'feeder_export_subsidy_eur': float(result.feeder_export_subsidy_eur), 'feeder_net_cost_eur': float(result.feeder_net_cost_eur), 'throughput_regularization_eur': float(result.throughput_regularization_eur), 'num_vars': int(result.model_size.num_vars), 'num_binary_vars': int(result.model_size.num_binary_vars), 'num_linear_constraints': int(result.model_size.num_linear_constraints), 'num_quadratic_constraints': int(result.model_size.num_quadratic_constraints), 'simultaneous_agent_steps': int(result.simultaneous_agent_steps), 'simultaneous_step_ratio': float(result.simultaneous_step_ratio), 'max_simultaneous_kw': float(result.max_simultaneous_kw)}, name='solve_summary')
    worst_balance_df = step_df.assign(abs_balance_residual_kw=lambda frame: frame['balance_residual_kw'].abs()).sort_values(['abs_balance_residual_kw', 'global_step'], ascending=[False, True]).loc[:, ['timestamp', 'episode_idx', 'global_step', 'balance_supply_kw', 'balance_demand_kw', 'balance_residual_kw', 'network_loss_kw_estimate', 'pv_curtail_kw', 'post_action_net_load_kw', 'root_net_exchange_kw', 'max_line_loading_pct', 'trafo_loading_pct', 'simultaneous_kw_total', 'abs_balance_residual_kw']].head(int(max(top_k, 1))).reset_index(drop=True)
    v_min_pu = float(np.sqrt(float(problem.v_min_sq)))
    v_max_pu = float(np.sqrt(float(problem.v_max_sq)))
    voltage_violation_count = ((bus_vm_pu < v_min_pu - 1e-06) | (bus_vm_pu > v_max_pu + 1e-06)).sum(axis=0).astype(np.int32)
    under_voltage_pu = np.maximum(v_min_pu - step_df['min_vm_pu'].to_numpy(dtype=np.float32), 0.0)
    over_voltage_pu = np.maximum(step_df['max_vm_pu'].to_numpy(dtype=np.float32) - v_max_pu, 0.0)
    violation_magnitude_pu = np.maximum(under_voltage_pu, over_voltage_pu).astype(np.float32)
    voltage_issue_df = step_df.assign(voltage_violation_count=voltage_violation_count, under_voltage_pu=under_voltage_pu, over_voltage_pu=over_voltage_pu, violation_magnitude_pu=violation_magnitude_pu).loc[lambda frame: frame['violation_magnitude_pu'] > 0.0, ['timestamp', 'episode_idx', 'global_step', 'min_vm_pu', 'mean_vm_pu', 'max_vm_pu', 'under_voltage_pu', 'over_voltage_pu', 'violation_magnitude_pu', 'voltage_violation_count', 'max_line_loading_pct', 'trafo_loading_pct']].sort_values(['violation_magnitude_pu', 'global_step'], ascending=[False, True]).head(int(max(top_k, 1))).reset_index(drop=True)
    capacity_kwh = np.asarray(problem.capacity_mwh, dtype=np.float32).reshape(-1) * 1000.0
    soc_init = np.asarray(full_input.soc_init, dtype=np.float32).reshape(-1)
    soc_final = (np.asarray(result.energy_mwh, dtype=np.float32)[:, -1] / np.maximum(np.asarray(problem.capacity_mwh, dtype=np.float32), 1e-06)).astype(np.float32)
    battery_summary_df = pd.DataFrame({'agent_profile': resolved_profiles, 'agent_bus_id': np.asarray(resolved_bus_ids, dtype=np.int32), 'capacity_kwh': capacity_kwh.astype(np.float32), 'soc_init': soc_init.astype(np.float32), 'soc_final': soc_final.astype(np.float32), 'max_charge_kw': np.max(battery_charge_mw, axis=1).astype(np.float32) * 1000.0, 'max_discharge_kw': np.max(battery_discharge_mw, axis=1).astype(np.float32) * 1000.0, 'total_charge_kwh': np.sum(battery_charge_mw, axis=1).astype(np.float32) * 1000.0 * float(problem.dt_hours), 'total_discharge_kwh': np.sum(battery_discharge_mw, axis=1).astype(np.float32) * 1000.0 * float(problem.dt_hours), 'total_pv_curtail_kwh': np.sum(pv_curtail_mw, axis=1).astype(np.float32) * 1000.0 * float(problem.dt_hours), 'throughput_kwh': np.sum(battery_charge_mw + battery_discharge_mw, axis=1).astype(np.float32) * 1000.0 * float(problem.dt_hours)})
    return (solve_summary, worst_balance_df, voltage_issue_df, battery_summary_df)
def build_simultaneous_diagnostic_tables(problem: Any, full_input: Any, result: Any, *, agent_profiles: list[str] | tuple[str, ...] | None=None, top_k: int=12) -> tuple[pd.DataFrame, pd.DataFrame]:
    battery_charge_mw = _require_solution_matrix(result.battery_charge_mw, 'battery_charge_mw')
    battery_discharge_mw = _require_solution_matrix(result.battery_discharge_mw, 'battery_discharge_mw')
    pv_curtail_mw = _require_solution_matrix(result.pv_curtail_mw, 'pv_curtail_mw')
    simultaneous_kw = _require_solution_matrix(result.simultaneous_charge_discharge_kw, 'simultaneous_charge_discharge_kw')
    energy_mwh = _require_solution_matrix(result.energy_mwh, 'energy_mwh')
    import_price_seq = np.asarray(full_input.import_price_seq, dtype=np.float32).reshape(-1)
    timestamps = _timestamps_from_full_input(full_input)
    episode_idx = expand_episode_indices(full_input)
    resolved_profiles = list(agent_profiles or [f'agent_{idx}' for idx in range(battery_charge_mw.shape[0])])
    if len(resolved_profiles) != battery_charge_mw.shape[0]:
        raise ValueError('agent_profiles length must match the number of agents in the solution.')
    step_df = pd.DataFrame({'timestamp': timestamps, 'episode_idx': episode_idx, 'global_step': np.arange(import_price_seq.size, dtype=np.int32), IMPORT_PRICE_COLUMN: import_price_seq.astype(np.float32), 'charge_kw_total': np.sum(battery_charge_mw, axis=0).astype(np.float32) * 1000.0, 'discharge_kw_total': np.sum(battery_discharge_mw, axis=0).astype(np.float32) * 1000.0, 'pv_curtail_kw_total': np.sum(pv_curtail_mw, axis=0).astype(np.float32) * 1000.0, 'grid_import_kw': np.asarray(result.root_import_mw, dtype=np.float32).reshape(-1) * 1000.0, 'grid_export_kw': np.asarray(result.root_export_mw, dtype=np.float32).reshape(-1) * 1000.0, 'simultaneous_kw_total': np.sum(simultaneous_kw, axis=0).astype(np.float32), 'simultaneous_agent_count': np.sum(simultaneous_kw > 0.0, axis=0).astype(np.int32)})
    simultaneous_step_df = step_df.loc[lambda frame: frame['simultaneous_kw_total'] > 0.0].sort_values(['simultaneous_kw_total', 'global_step'], ascending=[False, True]).head(int(max(top_k, 1))).reset_index(drop=True)
    capacity_mwh = np.asarray(problem.capacity_mwh, dtype=np.float32).reshape(-1)
    rows: list[dict[str, object]] = []
    for agent_idx, profile in enumerate(resolved_profiles):
        for step_idx in range(import_price_seq.size):
            simultaneous_value = float(simultaneous_kw[agent_idx, step_idx])
            if simultaneous_value <= 0.0:
                continue
            rows.append({'timestamp': timestamps[step_idx], 'episode_idx': int(episode_idx[step_idx]), 'global_step': int(step_idx), 'agent_id': int(agent_idx), 'agent_profile': str(profile), IMPORT_PRICE_COLUMN: float(import_price_seq[step_idx]), 'charge_kw': float(battery_charge_mw[agent_idx, step_idx] * 1000.0), 'discharge_kw': float(battery_discharge_mw[agent_idx, step_idx] * 1000.0), 'simultaneous_kw': simultaneous_value, 'soc_before': float(energy_mwh[agent_idx, step_idx] / max(float(capacity_mwh[agent_idx]), 1e-06)), 'soc_after': float(energy_mwh[agent_idx, step_idx + 1] / max(float(capacity_mwh[agent_idx]), 1e-06)), 'pv_curtail_kw': float(pv_curtail_mw[agent_idx, step_idx] * 1000.0), 'grid_import_kw': float(np.asarray(result.root_import_mw, dtype=np.float32).reshape(-1)[step_idx] * 1000.0), 'grid_export_kw': float(np.asarray(result.root_export_mw, dtype=np.float32).reshape(-1)[step_idx] * 1000.0)})
    simultaneous_agent_df = pd.DataFrame(rows).sort_values(['simultaneous_kw', 'global_step', 'agent_id'], ascending=[False, True, True]).head(int(max(top_k, 1))).reset_index(drop=True) if rows else pd.DataFrame(columns=['timestamp', 'episode_idx', 'global_step', 'agent_id', 'agent_profile', IMPORT_PRICE_COLUMN, 'charge_kw', 'discharge_kw', 'simultaneous_kw', 'soc_before', 'soc_after', 'pv_curtail_kw', 'grid_import_kw', 'grid_export_kw'])
    return (simultaneous_step_df, simultaneous_agent_df)
def build_chunk_boundary_soc_df(problem: Any, full_input: Any, result: Any) -> pd.DataFrame:
    chunk_summaries = list(getattr(result, 'chunk_summaries', []) or [])
    if len(chunk_summaries) < 2:
        return pd.DataFrame(columns=['boundary_step', 'boundary_timestamp', 'chunk_left', 'chunk_right', 'max_abs_soc_kink', 'mean_abs_soc_kink'])
    energy_mwh = _require_solution_matrix(result.energy_mwh, 'energy_mwh')
    capacity_mwh = np.asarray(problem.capacity_mwh, dtype=np.float32).reshape(-1, 1)
    soc = energy_mwh / np.maximum(capacity_mwh, 1e-06)
    timestamps = _timestamps_from_full_input(full_input)
    horizon_steps = int(np.asarray(full_input.import_price_seq, dtype=np.float32).reshape(-1).size)
    rows: list[dict[str, object]] = []
    for left_summary, right_summary in zip(chunk_summaries[:-1], chunk_summaries[1:], strict=False):
        boundary_step = int(left_summary['end_step'])
        if boundary_step <= 0 or boundary_step >= horizon_steps:
            continue
        delta_before = soc[:, boundary_step] - soc[:, boundary_step - 1]
        delta_after = soc[:, boundary_step + 1] - soc[:, boundary_step]
        kink = np.abs(delta_after - delta_before).astype(np.float32)
        rows.append({'boundary_step': boundary_step, 'boundary_timestamp': timestamps[boundary_step], 'chunk_left': int(left_summary['chunk_idx']), 'chunk_right': int(right_summary['chunk_idx']), 'max_abs_soc_kink': float(np.max(kink)) if kink.size else 0.0, 'mean_abs_soc_kink': float(np.mean(kink)) if kink.size else 0.0})
    return pd.DataFrame(rows)
def _linear_fit_summary(x_values: Any, y_values: Any) -> dict[str, float]:
    x_array = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_array = np.asarray(y_values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
    if int(np.sum(finite_mask)) < 2:
        return {'k': float('nan'), 'b': float('nan'), 'r2': float('nan')}
    x = x_array[finite_mask]
    y = y_array[finite_mask]
    if np.allclose(x, x[0], atol=1e-12, rtol=0.0):
        return {'k': float('nan'), 'b': float('nan'), 'r2': float('nan')}
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-12:
        r2 = 1.0 if ss_res <= 1e-12 else float('nan')
    else:
        r2 = 1.0 - ss_res / ss_tot
    return {'k': float(slope), 'b': float(intercept), 'r2': float(r2)}
def _interpret_linear_fit(k: float, b: float, r2: float) -> str:
    if not (np.isfinite(k) and np.isfinite(b) and np.isfinite(r2)):
        return 'insufficient_data'
    if abs(k - 1.0) <= 0.05 and abs(b) <= 1.0 and (r2 > 0.99):
        return 'consistent'
    if abs(k + 1.0) <= 0.05:
        return 'sign_error'
    if abs(k) >= 900.0 or (abs(k) > 0.0 and abs(k) <= 0.002):
        return 'unit_mismatch'
    if r2 < 0.9:
        return 'nonlinear_or_noise_dominated'
    return 'systematic_linear_bias'
def build_soc_relaxation_diagnostics(problem: Any, full_input: Any, result: Any, *, top_k: int=10) -> dict[str, Any]:
    branch_p_pu = _require_solution_matrix(result.branch_p_pu, 'branch_p_pu')
    branch_q_pu = _require_solution_matrix(result.branch_q_pu, 'branch_q_pu')
    branch_i2_pu = _require_solution_matrix(result.branch_i2_pu, 'branch_i2_pu')
    bus_v_sq = _require_solution_matrix(result.bus_v_sq, 'bus_v_sq')
    parent_pos = np.asarray(problem.network.branch_parent_pos, dtype=np.int32).reshape(-1)
    child_pos = np.asarray(problem.network.branch_child_pos, dtype=np.int32).reshape(-1)
    bus_ids = np.asarray(problem.network.bus_ids, dtype=np.int32).reshape(-1)
    timestamps = _timestamps_from_full_input(full_input)
    episode_idx = expand_episode_indices(full_input)
    if branch_p_pu.shape != branch_q_pu.shape or branch_p_pu.shape != branch_i2_pu.shape:
        raise ValueError('branch_p_pu, branch_q_pu, and branch_i2_pu must share the same shape.')
    if branch_p_pu.shape[0] != parent_pos.shape[0]:
        raise ValueError('Branch solution arrays do not match network.branch_parent_pos length.')
    if bus_v_sq.shape[0] != bus_ids.shape[0]:
        raise ValueError('bus_v_sq bus dimension does not match network.bus_ids length.')
    v_parent_sq = np.asarray(bus_v_sq[parent_pos, :], dtype=np.float32)
    soc_slack = np.asarray(branch_i2_pu - (branch_p_pu ** 2 + branch_q_pu ** 2) / np.maximum(v_parent_sq, _SOC_SLACK_EPS), dtype=np.float32)
    slack_max = np.max(soc_slack, axis=0).astype(np.float32)
    slack_mean = np.mean(soc_slack, axis=0).astype(np.float32)
    finite_slack = soc_slack[np.isfinite(soc_slack)]
    p95_soc_slack = float(np.percentile(finite_slack, 95.0)) if finite_slack.size else float('nan')
    summary = pd.Series({'max_soc_slack': float(np.max(finite_slack)) if finite_slack.size else float('nan'), 'mean_soc_slack': float(np.mean(finite_slack)) if finite_slack.size else float('nan'), 'p95_soc_slack': p95_soc_slack, 'soc_relaxation_is_tight': bool(np.isfinite(p95_soc_slack) and p95_soc_slack < _SOC_SLACK_TIGHT_P95_THRESHOLD)}, name='soc_relaxation_summary')
    step_df = pd.DataFrame({'global_step': np.arange(branch_p_pu.shape[1], dtype=np.int32), 'timestamp': timestamps, 'episode_idx': episode_idx.astype(np.int32), 'soc_slack_max': slack_max, 'soc_slack_mean': slack_mean, 'soc_slack_p95_global': np.full((branch_p_pu.shape[1],), p95_soc_slack, dtype=np.float32)})
    flat_indices = np.argsort(soc_slack.reshape(-1))[::-1]
    worst_rows: list[dict[str, object]] = []
    for flat_idx in flat_indices[:max(int(top_k), 0)]:
        branch_idx, step_idx = np.unravel_index(int(flat_idx), soc_slack.shape)
        worst_rows.append({'branch_idx': int(branch_idx), 'global_step': int(step_idx), 'timestamp': timestamps[int(step_idx)], 'episode_idx': int(episode_idx[int(step_idx)]), 'parent_bus_id': int(bus_ids[int(parent_pos[int(branch_idx)])]), 'child_bus_id': int(bus_ids[int(child_pos[int(branch_idx)])]), 'v_parent_sq_pu': float(v_parent_sq[int(branch_idx), int(step_idx)]), 'branch_p_pu': float(branch_p_pu[int(branch_idx), int(step_idx)]), 'branch_q_pu': float(branch_q_pu[int(branch_idx), int(step_idx)]), 'branch_i2_pu': float(branch_i2_pu[int(branch_idx), int(step_idx)]), 'soc_slack': float(soc_slack[int(branch_idx), int(step_idx)])})
    return {'step_df': step_df, 'worst_df': pd.DataFrame(worst_rows), 'summary': summary, 'soc_slack': soc_slack}
def build_root_q_diagnostic_df(problem: Any, full_input: Any, result: Any) -> pd.DataFrame:
    branch_i2_pu = _require_solution_matrix(result.branch_i2_pu, 'branch_i2_pu')
    root_q_kvar = _require_solution_matrix(result.root_q_kvar, 'root_q_kvar').reshape(-1)
    branch_x_pu = np.asarray(problem.network.branch_x_pu, dtype=np.float32).reshape(-1)
    s_base_mva = float(problem.network.s_base_mva)
    timestamps = _timestamps_from_full_input(full_input)
    episode_idx = expand_episode_indices(full_input)
    if branch_i2_pu.shape[0] != branch_x_pu.shape[0]:
        raise ValueError('branch_i2_pu and network.branch_x_pu must share the branch dimension.')
    if branch_i2_pu.shape[1] != root_q_kvar.shape[0]:
        raise ValueError('branch_i2_pu and root_q_kvar must share the horizon length.')
    background_q_base_total_kvar = float(np.sum(np.asarray(problem.network.q_base_mvar, dtype=np.float32)) * 1000.0)
    network_q_loss_proxy_kvar = (np.sum(branch_x_pu[:, None] * branch_i2_pu, axis=0) * np.float32(s_base_mva * 1000.0)).astype(np.float32)
    root_q_residual_kvar = (root_q_kvar.astype(np.float32) - background_q_base_total_kvar - network_q_loss_proxy_kvar).astype(np.float32)
    return pd.DataFrame({'global_step': np.arange(root_q_kvar.shape[0], dtype=np.int32), 'timestamp': timestamps, 'episode_idx': episode_idx.astype(np.int32), 'background_q_base_total_kvar': np.full((root_q_kvar.shape[0],), background_q_base_total_kvar, dtype=np.float32), 'network_q_loss_proxy_kvar': network_q_loss_proxy_kvar, 'root_q_kvar': root_q_kvar.astype(np.float32), 'root_q_residual_kvar': root_q_residual_kvar})
def build_misocp_validation_artifacts(env: Any, problem: Any, full_input: Any, result: Any, *, controller_label: str, export_subsidy: float, agent_profiles: list[str] | tuple[str, ...] | None=None, agent_bus_ids: list[int] | tuple[int, ...] | None=None, v_min_pu: float | None=None, v_max_pu: float | None=None) -> dict[str, Any]:
    from controllers.action_feasibility import build_safety_local_numpy, compute_action_gap_metrics_numpy, merge_action_info_into_step_info
    from controllers.mpc.global_socp_mpc import _SIMULTANEOUS_THRESHOLD_RATIO
    from scripts.utils.grid_notebook_workflow import RolloutResult, _aligned_prediction, _assemble_global_oracle_actions, _build_rollout_records, _step_timestamp
    resolved_profiles = list(agent_profiles or [f'agent_{idx}' for idx in range(int(env.n))])
    resolved_agent_bus_ids = list(agent_bus_ids or getattr(getattr(env, '_grid_core', None), 'agent_bus_ids', []))
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_set = set(resolved_agent_bus_ids)
    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_from_problem(problem)
    trafo_limit_reference_kw = float(problem.trafo_limit_mva) * 1000.0
    loading_limit_pct = float(getattr(getattr(env, '_grid_cfg', None), 'line_max_loading_pct', 100.0))
    threshold_kw = (_SIMULTANEOUS_THRESHOLD_RATIO * np.asarray(env.agent_p_max, dtype=np.float32)).astype(np.float32)
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    carried_soc = np.asarray(full_input.soc_init, dtype=np.float32).copy()
    timestamps = _timestamps_from_full_input(full_input)
    soc_relaxation_artifacts = build_soc_relaxation_diagnostics(problem, full_input, result, top_k=20)
    soc_relaxation_step_df = soc_relaxation_artifacts['step_df']
    soc_relaxation_worst_df = soc_relaxation_artifacts['worst_df']
    soc_relaxation_summary = soc_relaxation_artifacts['summary']
    root_q_diagnostic_df = build_root_q_diagnostic_df(problem, full_input, result)
    soc_relaxation_lookup = soc_relaxation_step_df.set_index('global_step', drop=False) if not soc_relaxation_step_df.empty else pd.DataFrame()
    root_q_lookup = root_q_diagnostic_df.set_index('global_step', drop=False) if not root_q_diagnostic_df.empty else pd.DataFrame()
    for episode_list_idx, episode_idx in enumerate(np.asarray(full_input.episode_indices, dtype=np.int32).tolist()):
        obs, reset_info = env.reset(episode_idx=int(episode_idx))
        if episode_list_idx > 0:
            env.soc = carried_soc.copy()
            obs = env.obs_builder.build(env)
        raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, 'build_raw') else obs
        previous_raw_obs = None
        episode_offset = int(np.asarray(full_input.episode_offsets, dtype=np.int32)[episode_list_idx])
        episode_length = int(np.asarray(full_input.episode_lengths, dtype=np.int32)[episode_list_idx])
        for step_in_episode in range(episode_length):
            global_step = episode_offset + step_in_episode
            timestamp = timestamps[global_step] if global_step < len(timestamps) else _step_timestamp(reset_info, step_in_episode)
            wholesale_price_pred = _aligned_prediction(previous_raw_obs, raw_obs, WHOLESALE_PRICE_SEQ_FIELD)
            import_price_pred = float(derive_import_price(wholesale_price_pred, markup_eur_per_kwh=float(getattr(getattr(env, '_reward_cfg', None), IMPORT_PRICE_MARKUP_KEY, 0.0))))
            load_pred = _aligned_prediction(previous_raw_obs, raw_obs, 'load_seq')
            pv_pred = _aligned_prediction(previous_raw_obs, raw_obs, 'pv_seq')
            battery_power_kw = ((np.asarray(result.battery_charge_mw[:, global_step], dtype=np.float32) - np.asarray(result.battery_discharge_mw[:, global_step], dtype=np.float32)) * 1000.0).astype(np.float32)
            pv_curtail_kw = (np.asarray(result.pv_curtail_mw[:, global_step], dtype=np.float32) * 1000.0).astype(np.float32)
            actions, action_array = _assemble_global_oracle_actions(env, battery_power_kw, pv_curtail_kw)
            action_info = compute_action_gap_metrics_numpy(build_safety_local_numpy(soc=np.asarray(env.soc, dtype=np.float32), load_raw=np.asarray(env.get_signal_step('load'), dtype=np.float32), pv_raw=np.asarray(env.get_signal_step('pv'), dtype=np.float32), battery_capacity_kwh=np.asarray(env.agent_c_bat, dtype=np.float32), p_max_kw=np.asarray(env.agent_p_max, dtype=np.float32)), action_array, action_array)
            simultaneous_kw = np.asarray(result.simultaneous_charge_discharge_kw[:, global_step], dtype=np.float32)
            simultaneous_mask = simultaneous_kw > threshold_kw
            action_info.update({'misocp_fallback': np.asarray(0.0, dtype=np.float32), 'misocp_time_limit_feasible': np.asarray(float(result.time_limit_feasible), dtype=np.float32), 'solve_time_sec': np.asarray(float(result.solve_time_sec), dtype=np.float32), 'root_import_kw': np.asarray(float(result.root_import_mw[global_step] * 1000.0), dtype=np.float32), 'root_export_kw': np.asarray(float(result.root_export_mw[global_step] * 1000.0), dtype=np.float32), 'simultaneous_charge_discharge_kw_total': np.asarray(float(np.sum(simultaneous_kw)), dtype=np.float32), 'simultaneous_agent_count': np.asarray(float(np.sum(simultaneous_mask)), dtype=np.float32), 'simultaneous_step_flag': np.asarray(float(np.any(simultaneous_mask)), dtype=np.float32)})
            next_obs, reward, terminated, truncated, info = env.step(actions)
            reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
            info, action_penalty = merge_action_info_into_step_info(info, action_info, soc_pen_weight=float(getattr(getattr(env, '_reward_cfg', None), 'w_soc_pen', 0.0)), apply_action_penalty=False)
            reward_array = reward_array - np.asarray(action_penalty, dtype=np.float32)
            info['reward'] = reward_array.astype(np.float32)
            del terminated, truncated
            raw_next_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, 'build_raw') and (not bool(info.get('episode_done', False))) else next_obs
            soc_row = soc_relaxation_lookup.loc[int(global_step)] if not soc_relaxation_lookup.empty else None
            root_q_row = root_q_lookup.loc[int(global_step)] if not root_q_lookup.empty else None
            soc_slack_max = float(soc_row['soc_slack_max']) if soc_row is not None else float('nan')
            soc_slack_mean = float(soc_row['soc_slack_mean']) if soc_row is not None else float('nan')
            soc_slack_p95_global = float(soc_relaxation_summary.get('p95_soc_slack', float('nan'))) if not soc_relaxation_summary.empty else float('nan')
            background_q_base_total_kvar = float(root_q_row['background_q_base_total_kvar']) if root_q_row is not None else float('nan')
            network_q_loss_proxy_kvar = float(root_q_row['network_q_loss_proxy_kvar']) if root_q_row is not None else float('nan')
            root_q_residual_kvar = float(root_q_row['root_q_residual_kvar']) if root_q_row is not None else float('nan')
            rollout_records = _build_rollout_records(controller_label=controller_label, episode_idx=int(episode_idx), step_in_episode=step_in_episode, timestamp=timestamp, info=info, load_pred=load_pred, pv_pred=pv_pred, wholesale_price_pred=float(wholesale_price_pred), import_price_pred=float(import_price_pred), agent_profiles=resolved_profiles, bus_ids=bus_ids, agent_bus_set=agent_bus_set, dt_hours=float(env.dt), export_subsidy=float(export_subsidy), fixed_load_kw=fixed_load_kw, fixed_generation_kw=fixed_generation_kw, loading_limit_pct=loading_limit_pct, trafo_limit_kw=trafo_limit_reference_kw, v_min_pu=v_min_pu, v_max_pu=v_max_pu, global_step=int(global_step))
            step_row = rollout_records['step_row']
            pp_root_p_kw = float(rollout_records['pp_root_p_kw'])
            pp_root_p_available = bool(step_row['pp_root_p_available'])
            agent_post_action_net_load_kw = float(step_row['agent_post_action_net_load_kw'])
            feeder_post_action_net_load_kw = float(step_row['feeder_post_action_net_load_kw'])
            agent_root_gap_kw = float(result.root_p_kw[global_step] - agent_post_action_net_load_kw)
            solver_feeder_gap_kw = float(result.root_p_kw[global_step] - feeder_post_action_net_load_kw)
            replay_feeder_gap_kw = float(pp_root_p_kw - feeder_post_action_net_load_kw) if pp_root_p_available else float('nan')
            pp_trafo_loading_pct = rollout_records['pp_trafo_loading_pct']
            misocp_root_s_kva = float(np.hypot(float(result.root_p_kw[global_step]), float(result.root_q_kvar[global_step])))
            pp_root_s_kva = float(np.max(pp_trafo_loading_pct) / 100.0 * float(problem.network.s_base_mva) * 1000.0) if pp_trafo_loading_pct.size else float('nan')
            step_row.update({'root_net_exchange_kw': float(result.root_p_kw[global_step]), 'agent_root_gap_kw': agent_root_gap_kw, 'solver_feeder_gap_kw': solver_feeder_gap_kw, 'replay_feeder_gap_kw': replay_feeder_gap_kw, 'misocp_root_s_kva': misocp_root_s_kva, 'pp_root_s_kva': pp_root_s_kva, 'soc_slack_max': soc_slack_max, 'soc_slack_mean': soc_slack_mean, 'soc_slack_p95_global': soc_slack_p95_global, 'background_q_base_total_kvar': background_q_base_total_kvar, 'network_q_loss_proxy_kvar': network_q_loss_proxy_kvar, 'root_q_residual_kvar': root_q_residual_kvar})
            step_rows.append(step_row)
            diagnostic_rows.append({'solver_type': 'gurobi_misocp', 'status_code': int(result.status_code), 'status_label': str(result.status_label), 'solve_mode': str(result.solve_mode), 'solve_time_sec': float(result.solve_time_sec), 'misocp_fallback': 0.0, 'misocp_time_limit_feasible': float(result.time_limit_feasible), 'mip_gap': float(result.mip_gap), 'best_bound': float(result.best_bound), 'objective_value': float(result.objective_value), 'agent_purchase_cost_eur': float(result.agent_purchase_cost_eur), 'agent_export_subsidy_eur': float(result.agent_export_subsidy_eur), 'agent_net_cost_eur': float(result.agent_net_cost_eur), 'feeder_purchase_cost_eur': float(result.feeder_purchase_cost_eur), 'feeder_export_subsidy_eur': float(result.feeder_export_subsidy_eur), 'feeder_net_cost_eur': float(result.feeder_net_cost_eur), 'throughput_regularization_eur': float(result.throughput_regularization_eur), 'throughput_regularization_weight': float(result.throughput_regularization_weight), 'root_import_kw': float(result.root_import_mw[global_step] * 1000.0), 'root_export_kw': float(result.root_export_mw[global_step] * 1000.0), 'root_p_kw': float(result.root_p_kw[global_step]), 'root_q_kvar': float(result.root_q_kvar[global_step]), 'misocp_vm_pu': np.asarray(result.bus_vm_pu[:, global_step], dtype=np.float32).copy(), 'misocp_line_loading_pct': np.asarray(result.line_loading_pct[:, global_step], dtype=np.float32).copy(), 'misocp_trafo_loading_pct': np.asarray(result.trafo_loading_pct[:, global_step], dtype=np.float32).copy(), 'pp_vm_pu': rollout_records['pp_vm_pu'], 'pp_line_loading_pct': rollout_records['pp_line_loading_pct'], 'pp_trafo_loading_pct': pp_trafo_loading_pct, 'pp_root_p_kw': pp_root_p_kw, 'pp_root_p_available': pp_root_p_available, 'misocp_root_s_kva': misocp_root_s_kva, 'pp_root_s_kva': pp_root_s_kva, 'soc_slack_max': soc_slack_max, 'soc_slack_mean': soc_slack_mean, 'soc_slack_p95_global': soc_slack_p95_global, 'background_q_base_total_kvar': background_q_base_total_kvar, 'network_q_loss_proxy_kvar': network_q_loss_proxy_kvar, 'root_q_residual_kvar': root_q_residual_kvar, 'solver_feeder_gap_kw': solver_feeder_gap_kw, 'replay_feeder_gap_kw': replay_feeder_gap_kw, 'controller': controller_label, 'episode_idx': int(episode_idx), 'step': int(step_in_episode), 'timestamp': timestamp})
            agent_rows.extend(rollout_records['agent_rows'])
            grid_rows.extend(rollout_records['grid_rows'])
            previous_raw_obs = raw_obs
            obs = next_obs
            raw_obs = raw_next_obs
        carried_soc = np.asarray(env.soc, dtype=np.float32).copy()
    step_df = pd.DataFrame(step_rows).sort_values(['episode_idx', 'step']).reset_index(drop=True)
    agent_df = pd.DataFrame(agent_rows).sort_values(['episode_idx', 'step', 'agent_id']).reset_index(drop=True)
    grid_df = pd.DataFrame(grid_rows).sort_values(['episode_idx', 'step', 'bus_id']).reset_index(drop=True)
    summary = agent_df.groupby(['controller', 'agent_profile'], as_index=False)[['purchase_cost', 'export_subsidy', 'objective_total']].sum() if {'purchase_cost', 'export_subsidy', 'objective_total'}.issubset(agent_df.columns) else pd.DataFrame()
    validation_df = build_misocp_validation_df(diagnostic_rows)
    validation_summary = summarize_misocp_validation(validation_df)
    health_warning = ''
    if not validation_df.empty and bool((~validation_df['within_tolerance']).any()):
        health_warning = 'MISOCP-vs-pandapower validation exceeded at least one tolerance.'
    if bool(validation_summary.get('root_power_validation_unavailable', False)):
        health_warning = (health_warning + ' ' if health_warning else '') + 'Root-power validation is unavailable for at least one step because GridEnv did not expose trafo_p_signed_kw.'
    rollout = RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=summary, meta={'controller': controller_label, 'agent_profiles': resolved_profiles, 'agent_bus_ids': resolved_agent_bus_ids, 'bus_ids': bus_ids, 'v_min_pu': np.nan if v_min_pu is None else float(v_min_pu), 'v_max_pu': np.nan if v_max_pu is None else float(v_max_pu), 'trafo_limit_kw': trafo_limit_reference_kw, 'trafo_limit_note': 'Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.', 'misocp_validation_df': validation_df, 'misocp_validation_summary': validation_summary, 'misocp_health_warning': health_warning, 'controller_diagnostic_log': diagnostic_rows, 'economics_scope': 'agent_only', 'validation_root_power_source': 'trafo_p_signed_kw', 'soc_relaxation_summary': soc_relaxation_summary})
    return {'rollout': rollout, 'step_df': step_df, 'agent_df': agent_df, 'grid_df': grid_df, 'diagnostic_rows': diagnostic_rows, 'validation_df': validation_df, 'validation_summary': validation_summary, 'soc_relaxation_step_df': soc_relaxation_step_df, 'soc_relaxation_worst_df': soc_relaxation_worst_df, 'soc_relaxation_summary': soc_relaxation_summary, 'root_q_diagnostic_df': root_q_diagnostic_df}
