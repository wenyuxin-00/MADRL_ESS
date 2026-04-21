from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY, IMPORT_PRICE_PRED_COLUMN, WHOLESALE_PRICE_PRED_COLUMN, derive_import_price_seq
from scripts.utils.rollout_package_utils import ROLLOUT_PACKAGE_VERSION as SHARED_ROLLOUT_PACKAGE_VERSION, assert_rollout_cfg_snapshot_matches, assert_solver_fingerprint_matches, build_rollout_cfg_snapshot_from_cfg, load_rollout_package, relabel_rollout_dataframe, resolve_exact_rollout_package_dir, save_rollout_package
_ADMM_RESIDUAL_BALANCING_MU = 10.0
_ADMM_RESIDUAL_BALANCING_TAU = 2.0
_ROLLOUT_PACKAGE_VERSION = SHARED_ROLLOUT_PACKAGE_VERSION

def _build_admm_solver_fingerprint(*, rho_init: float | None, rho_min: float, rho_max: float, rho_adaptation: str | None, max_iters: int, max_iters_first_step: int, primal_tol: float, dual_tol: float, terminal_cost_multiplier: float) -> dict[str, Any]:
    return {'rho_init': None if rho_init is None else float(rho_init), 'rho_min': float(rho_min), 'rho_max': float(rho_max), 'rho_adaptation': None if rho_adaptation is None else str(rho_adaptation), 'max_iters': int(max_iters), 'max_iters_first_step': int(max_iters_first_step), 'primal_tol': float(primal_tol), 'dual_tol': float(dual_tol), 'terminal_cost_multiplier': float(terminal_cost_multiplier), 'residual_balance_ratio': float(_ADMM_RESIDUAL_BALANCING_MU), 'residual_balance_tau': float(_ADMM_RESIDUAL_BALANCING_TAU)}

def _expected_rollout_contract(cfg: Any, *, prediction_mode: str | None=None, rho_init: float | None, rho_min: float, rho_max: float, rho_adaptation: str | None, max_iters: int, max_iters_first_step: int, primal_tol: float, dual_tol: float, terminal_cost_multiplier: float) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    resolved_prediction_mode = grid_nb.normalize_prediction_mode(prediction_mode) if prediction_mode is not None else grid_nb.resolve_prediction_mode_from_forecast_backend(str(getattr(cfg.forecast, 'type', 'perfect')))
    comparison_cfg = grid_nb.build_comparison_cfg(cfg, prediction_mode=resolved_prediction_mode)
    cfg_snapshot = build_rollout_cfg_snapshot_from_cfg(comparison_cfg, prediction_mode=resolved_prediction_mode)
    solver_fingerprint = _build_admm_solver_fingerprint(rho_init=rho_init, rho_min=rho_min, rho_max=rho_max, rho_adaptation=rho_adaptation, max_iters=max_iters, max_iters_first_step=max_iters_first_step, primal_tol=primal_tol, dual_tol=dual_tol, terminal_cost_multiplier=terminal_cost_multiplier)
    return (comparison_cfg, cfg_snapshot, solver_fingerprint)

def _augment_rollout_with_diagnostics(rollout: grid_nb.RolloutResult, diagnostic_rows: list[dict[str, object]], *, forecast_backend: str) -> grid_nb.RolloutResult:
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    if 'global_step' not in step_df.columns:
        step_df['global_step'] = np.arange(len(step_df), dtype=np.int64)
    step_index = step_df[['episode_idx', 'step', 'global_step']].drop_duplicates()
    if not agent_df.empty and 'global_step' not in agent_df.columns:
        agent_df = agent_df.merge(step_index, on=['episode_idx', 'step'], how='left')
    if not grid_df.empty and 'global_step' not in grid_df.columns:
        grid_df = grid_df.merge(step_index, on=['episode_idx', 'step'], how='left')
    if not diagnostic_df.empty:
        merge_keys = [column for column in ('episode_idx', 'step', 'global_step') if column in diagnostic_df.columns]
        step_df = step_df.merge(diagnostic_df, on=merge_keys, how='left')
    for column, default_value in [('admm_converged', False), ('admm_iterations', 0), ('admm_final_primal_residual', np.nan), ('admm_final_dual_residual', np.nan), ('admm_solve_time_sec', np.nan), ('admm_rho_final', np.nan)]:
        if column not in step_df.columns:
            step_df[column] = default_value
    step_df['admm_converged'] = step_df['admm_converged'].astype(bool)
    import_price_markup = float(rollout.meta.get(IMPORT_PRICE_MARKUP_KEY, 0.0))
    if WHOLESALE_PRICE_PRED_COLUMN in step_df.columns and IMPORT_PRICE_PRED_COLUMN not in step_df.columns:
        step_df[IMPORT_PRICE_PRED_COLUMN] = derive_import_price_seq(step_df[WHOLESALE_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64), markup_eur_per_kwh=import_price_markup)
    step_df['objective_total'] = step_df['purchase_cost_total'].astype(float) - step_df['export_subsidy_total'].astype(float)
    meta = dict(rollout.meta)
    meta['forecast_backend'] = str(forecast_backend)
    meta['admm_terminal_cost_mode'] = 'quadratic_to_soc_target'
    return grid_nb.RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=rollout.summary.copy(), meta=meta)

def build_admm_mpc_rollout_package(rollout: grid_nb.RolloutResult, *, controller_label: str, cfg: Any | None=None, cfg_snapshot: dict[str, Any] | None=None, prediction_mode: str | None=None, rho_init: float | None, rho_min: float, rho_max: float, rho_adaptation: str | None, max_iters: int, max_iters_first_step: int, primal_tol: float, dual_tol: float, terminal_cost_multiplier: float, extra_meta: dict[str, Any] | None=None, diagnostic_summary: dict[str, Any] | None=None) -> dict[str, Any]:
    if cfg_snapshot is None:
        if cfg is None:
            raise ValueError('Provide either cfg or cfg_snapshot when building an ADMM MPC rollout package.')
        _, cfg_snapshot, _ = _expected_rollout_contract(cfg, prediction_mode=prediction_mode, rho_init=rho_init, rho_min=rho_min, rho_max=rho_max, rho_adaptation=rho_adaptation, max_iters=max_iters, max_iters_first_step=max_iters_first_step, primal_tol=primal_tol, dual_tol=dual_tol, terminal_cost_multiplier=terminal_cost_multiplier)
    resolved_prediction_mode = str(grid_nb.normalize_prediction_mode(prediction_mode if prediction_mode is not None else cfg_snapshot.get('prediction_mode', rollout.meta.get('prediction_mode', 'normal'))))
    forecast_backend = str(cfg_snapshot.get('forecast_backend', rollout.meta.get('forecast_backend', 'perfect')))
    manifest = {'rollout_package_version': int(_ROLLOUT_PACKAGE_VERSION), 'controller_label': str(controller_label), 'saved_at_utc': pd.Timestamp.utcnow().isoformat(), 'prediction_mode': resolved_prediction_mode, 'forecast_backend': forecast_backend, 'cfg_snapshot': dict(cfg_snapshot), 'admm_solver_fingerprint': _build_admm_solver_fingerprint(rho_init=rho_init, rho_min=rho_min, rho_max=rho_max, rho_adaptation=rho_adaptation, max_iters=max_iters, max_iters_first_step=max_iters_first_step, primal_tol=primal_tol, dual_tol=dual_tol, terminal_cost_multiplier=terminal_cost_multiplier), 'extra_meta': dict(extra_meta or {})}
    return {'manifest': manifest, 'diagnostics': dict(diagnostic_summary or {}), 'step_df': rollout.step_df.copy(), 'agent_df': rollout.agent_df.copy(), 'grid_df': rollout.grid_df.copy(), 'summary_df': rollout.summary.copy()}

def save_admm_mpc_rollout_package(package: dict[str, Any], target_dir: str | Path) -> Path:
    return save_rollout_package(package, target_dir)

def load_admm_mpc_rollout_package(target_dir: str | Path) -> dict[str, Any]:
    return load_rollout_package(target_dir, package_label='ADMM MPC rollout package', version=_ROLLOUT_PACKAGE_VERSION, regenerate_hint='Re-run notebooks/madrl/ADMM_mpc.ipynb to regenerate it.')

def _validate_admm_mpc_rollout_package_dir(target_path: Path, *, expected_cfg_snapshot: dict[str, Any] | None=None, expected_solver_fingerprint: dict[str, Any] | None=None) -> None:
    try:
        manifest = load_admm_mpc_rollout_package(target_path)['manifest']
    except json.JSONDecodeError as exc:
        raise ValueError(f'ADMM MPC rollout package manifest is invalid JSON at {target_path}.') from exc
    if expected_cfg_snapshot is not None:
        assert_rollout_cfg_snapshot_matches(expected_cfg_snapshot, manifest['cfg_snapshot'], mismatch_prefix='ADMM MPC rollout package')
    if expected_solver_fingerprint is not None:
        assert_solver_fingerprint_matches(expected_solver_fingerprint, manifest['admm_solver_fingerprint'], name='admm_solver_fingerprint')

def resolve_exact_admm_mpc_rollout_package_dir(target_dir: str | Path, *, cfg: Any | None=None, prediction_mode: str | None=None, rho_init: float | None=None, rho_min: float=0.001, rho_max: float=1000.0, rho_adaptation: str | None='residual_balancing', max_iters: int=100, max_iters_first_step: int=300, primal_tol: float=0.001, dual_tol: float=0.001, terminal_cost_multiplier: float=1.0) -> Path:
    expected_cfg_snapshot = None
    expected_solver_fingerprint = None
    if cfg is not None:
        _, expected_cfg_snapshot, expected_solver_fingerprint = _expected_rollout_contract(cfg, prediction_mode=prediction_mode, rho_init=rho_init, rho_min=rho_min, rho_max=rho_max, rho_adaptation=rho_adaptation, max_iters=max_iters, max_iters_first_step=max_iters_first_step, primal_tol=primal_tol, dual_tol=dual_tol, terminal_cost_multiplier=terminal_cost_multiplier)

    def _validate(candidate_dir: Path) -> None:
        _validate_admm_mpc_rollout_package_dir(candidate_dir, expected_cfg_snapshot=expected_cfg_snapshot, expected_solver_fingerprint=expected_solver_fingerprint)
    return resolve_exact_rollout_package_dir(target_dir, package_label='ADMM MPC cached rollout', missing_dir_hint='Please run notebooks/madrl/ADMM_mpc.ipynb first and pass the exact rollout package directory.', validate_package_dir=_validate)

def replay_admm_mpc_rollout_package(cfg: Any, target_dir: str | Path, *, label: str | None=None, prediction_mode: str | None=None, rho_init: float | None=None, rho_min: float=0.001, rho_max: float=1000.0, rho_adaptation: str | None='residual_balancing', max_iters: int=100, max_iters_first_step: int=300, primal_tol: float=0.001, dual_tol: float=0.001, terminal_cost_multiplier: float=1.0) -> dict[str, Any]:
    package = load_admm_mpc_rollout_package(target_dir)
    comparison_cfg, current_cfg_snapshot, current_solver_fingerprint = _expected_rollout_contract(cfg, prediction_mode=prediction_mode or package['manifest'].get('prediction_mode'), rho_init=rho_init, rho_min=rho_min, rho_max=rho_max, rho_adaptation=rho_adaptation, max_iters=max_iters, max_iters_first_step=max_iters_first_step, primal_tol=primal_tol, dual_tol=dual_tol, terminal_cost_multiplier=terminal_cost_multiplier)
    assert_rollout_cfg_snapshot_matches(current_cfg_snapshot, package['manifest']['cfg_snapshot'], mismatch_prefix='ADMM MPC rollout package')
    assert_solver_fingerprint_matches(current_solver_fingerprint, package['manifest']['admm_solver_fingerprint'], name='admm_solver_fingerprint')
    controller_label = str(label or package['manifest']['controller_label'])
    step_df = relabel_rollout_dataframe(package['step_df'], controller_label)
    agent_df = relabel_rollout_dataframe(package['agent_df'], controller_label)
    grid_df = relabel_rollout_dataframe(package['grid_df'], controller_label)
    summary_df = relabel_rollout_dataframe(package['summary_df'], controller_label)
    meta = dict(package['manifest'].get('extra_meta', {}).get('rollout_meta', {}))
    meta.update({'controller': controller_label, 'prediction_mode': str(current_cfg_snapshot['prediction_mode']), 'forecast_backend': str(current_cfg_snapshot['forecast_backend']), 'future_horizon': int(current_cfg_snapshot['future_horizon']), 'agent_profiles': list(current_cfg_snapshot['agent_profiles']), 'agent_bus_ids': list(current_cfg_snapshot['agent_bus_ids']), 'v_min_pu': float(current_cfg_snapshot['v_min_pu']), 'v_max_pu': float(current_cfg_snapshot['v_max_pu']), IMPORT_PRICE_MARKUP_KEY: float(current_cfg_snapshot[IMPORT_PRICE_MARKUP_KEY]), 'export_subsidy_eur_per_kwh': float(current_cfg_snapshot['export_subsidy_eur_per_kwh']), 'loaded_from_cached_rollout': True, 'rollout_package_dir': str(Path(target_dir).resolve()), 'rollout_package_version': int(package['manifest']['rollout_package_version']), 'cached_rollout_diagnostics': dict(package['diagnostics']), 'cached_admm_solver_fingerprint': dict(package['manifest']['admm_solver_fingerprint']), 'cached_rollout_prediction_mode': str(package['manifest']['prediction_mode']), 'cached_rollout_forecast_backend': str(package['manifest']['forecast_backend']), 'comparison_forecast_backend': str(comparison_cfg.forecast.type)})
    rollout = grid_nb.RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=summary_df, meta=meta)
    return {'manifest': package['manifest'], 'diagnostics': package['diagnostics'], 'rollout': rollout}
