from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY
from scripts.utils.rollout_package_utils import ROLLOUT_PACKAGE_VERSION, assert_rollout_cfg_snapshot_matches, assert_solver_fingerprint_matches, build_rollout_cfg_snapshot_from_cfg, load_rollout_package, relabel_rollout_dataframe, resolve_exact_rollout_package_dir, save_rollout_package
DEFAULT_LOCAL_MPC_PRICE_MODE = 'import_adjusted'
DEFAULT_LOCAL_MPC_OBJECTIVE_MODE = 'economic_only'
LOCAL_MPC_LSTM_LABEL = 'Local MPC + LSTM Forecast'
LOCAL_MPC_PERFECT_LABEL = 'Local MPC + Perfect Forecast'

def _build_local_mpc_solver_fingerprint(*, price_mode: str=DEFAULT_LOCAL_MPC_PRICE_MODE, objective_mode: str=DEFAULT_LOCAL_MPC_OBJECTIVE_MODE) -> dict[str, Any]:
    return {'price_mode': str(price_mode), 'objective_mode': str(objective_mode)}

def _expected_rollout_contract(cfg: Any, *, prediction_mode: str | None=None, price_mode: str=DEFAULT_LOCAL_MPC_PRICE_MODE, objective_mode: str=DEFAULT_LOCAL_MPC_OBJECTIVE_MODE) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    resolved_prediction_mode = grid_nb.normalize_prediction_mode(prediction_mode) if prediction_mode is not None else grid_nb.resolve_prediction_mode_from_forecast_backend(str(getattr(cfg.forecast, 'type', 'perfect')))
    comparison_cfg = grid_nb.build_comparison_cfg(cfg, prediction_mode=resolved_prediction_mode)
    cfg_snapshot = build_rollout_cfg_snapshot_from_cfg(comparison_cfg, prediction_mode=resolved_prediction_mode)
    solver_fingerprint = _build_local_mpc_solver_fingerprint(price_mode=price_mode, objective_mode=objective_mode)
    return (comparison_cfg, cfg_snapshot, solver_fingerprint)

def build_local_mpc_rollout_package(rollout: grid_nb.RolloutResult, *, controller_label: str, cfg: Any | None=None, cfg_snapshot: dict[str, Any] | None=None, prediction_mode: str | None=None, price_mode: str=DEFAULT_LOCAL_MPC_PRICE_MODE, objective_mode: str=DEFAULT_LOCAL_MPC_OBJECTIVE_MODE, extra_meta: dict[str, Any] | None=None, diagnostic_summary: dict[str, Any] | None=None) -> dict[str, Any]:
    if cfg_snapshot is None:
        if cfg is None:
            raise ValueError('Provide either cfg or cfg_snapshot when building a local MPC rollout package.')
        _, cfg_snapshot, _ = _expected_rollout_contract(cfg, prediction_mode=prediction_mode, price_mode=price_mode, objective_mode=objective_mode)
    resolved_prediction_mode = str(grid_nb.normalize_prediction_mode(prediction_mode if prediction_mode is not None else cfg_snapshot.get('prediction_mode', rollout.meta.get('prediction_mode', 'normal'))))
    forecast_backend = str(cfg_snapshot.get('forecast_backend', rollout.meta.get('forecast_backend', 'perfect')))
    manifest = {'rollout_package_version': int(ROLLOUT_PACKAGE_VERSION), 'controller_label': str(controller_label), 'saved_at_utc': pd.Timestamp.utcnow().isoformat(), 'prediction_mode': resolved_prediction_mode, 'forecast_backend': forecast_backend, 'cfg_snapshot': dict(cfg_snapshot), 'local_mpc_solver_fingerprint': _build_local_mpc_solver_fingerprint(price_mode=price_mode, objective_mode=objective_mode), 'extra_meta': dict(extra_meta or {})}
    return {'manifest': manifest, 'diagnostics': dict(diagnostic_summary or {}), 'step_df': rollout.step_df.copy(), 'agent_df': rollout.agent_df.copy(), 'grid_df': rollout.grid_df.copy(), 'summary_df': rollout.summary.copy()}

def save_local_mpc_rollout_package(package: dict[str, Any], target_dir: str | Path) -> Path:
    return save_rollout_package(package, target_dir)

def load_local_mpc_rollout_package(target_dir: str | Path) -> dict[str, Any]:
    return load_rollout_package(target_dir, package_label='Local MPC rollout package', version=ROLLOUT_PACKAGE_VERSION, regenerate_hint='Re-run notebooks/madrl/local_MPC.ipynb to regenerate it.')

def _validate_local_mpc_rollout_package_dir(target_path: Path, *, expected_cfg_snapshot: dict[str, Any] | None=None, expected_solver_fingerprint: dict[str, Any] | None=None) -> None:
    try:
        manifest = load_local_mpc_rollout_package(target_path)['manifest']
    except json.JSONDecodeError as exc:
        raise ValueError(f'Local MPC rollout package manifest is invalid JSON at {target_path}.') from exc
    if expected_cfg_snapshot is not None:
        assert_rollout_cfg_snapshot_matches(expected_cfg_snapshot, manifest['cfg_snapshot'], mismatch_prefix='Local MPC rollout package')
    if expected_solver_fingerprint is not None:
        assert_solver_fingerprint_matches(expected_solver_fingerprint, manifest['local_mpc_solver_fingerprint'], name='local_mpc_solver_fingerprint')

def resolve_exact_local_mpc_rollout_package_dir(target_dir: str | Path, *, cfg: Any | None=None, prediction_mode: str | None=None, price_mode: str=DEFAULT_LOCAL_MPC_PRICE_MODE, objective_mode: str=DEFAULT_LOCAL_MPC_OBJECTIVE_MODE) -> Path:
    expected_cfg_snapshot = None
    expected_solver_fingerprint = None
    if cfg is not None:
        _, expected_cfg_snapshot, expected_solver_fingerprint = _expected_rollout_contract(cfg, prediction_mode=prediction_mode, price_mode=price_mode, objective_mode=objective_mode)

    def _validate(candidate_dir: Path) -> None:
        _validate_local_mpc_rollout_package_dir(candidate_dir, expected_cfg_snapshot=expected_cfg_snapshot, expected_solver_fingerprint=expected_solver_fingerprint)
    return resolve_exact_rollout_package_dir(target_dir, package_label='Local MPC cached rollout', missing_dir_hint='Please run notebooks/madrl/local_MPC.ipynb first and pass the exact rollout package directory.', validate_package_dir=_validate)

def replay_local_mpc_rollout_package(cfg: Any, target_dir: str | Path, *, label: str | None=None, prediction_mode: str | None=None) -> dict[str, Any]:
    package = load_local_mpc_rollout_package(target_dir)
    comparison_cfg, current_cfg_snapshot, current_solver_fingerprint = _expected_rollout_contract(cfg, prediction_mode=prediction_mode or package['manifest'].get('prediction_mode'))
    assert_rollout_cfg_snapshot_matches(current_cfg_snapshot, package['manifest']['cfg_snapshot'], mismatch_prefix='Local MPC rollout package')
    assert_solver_fingerprint_matches(current_solver_fingerprint, package['manifest']['local_mpc_solver_fingerprint'], name='local_mpc_solver_fingerprint')
    controller_label = str(label or package['manifest']['controller_label'])
    meta = dict(package['manifest'].get('extra_meta', {}).get('rollout_meta', {}))
    meta.update({'controller': controller_label, 'prediction_mode': str(current_cfg_snapshot['prediction_mode']), 'forecast_backend': str(current_cfg_snapshot['forecast_backend']), 'future_horizon': int(current_cfg_snapshot['future_horizon']), 'agent_profiles': list(current_cfg_snapshot['agent_profiles']), 'agent_bus_ids': list(current_cfg_snapshot['agent_bus_ids']), 'v_min_pu': float(current_cfg_snapshot['v_min_pu']), 'v_max_pu': float(current_cfg_snapshot['v_max_pu']), IMPORT_PRICE_MARKUP_KEY: float(current_cfg_snapshot[IMPORT_PRICE_MARKUP_KEY]), 'export_subsidy_eur_per_kwh': float(current_cfg_snapshot['export_subsidy_eur_per_kwh']), 'loaded_from_cached_rollout': True, 'rollout_package_dir': str(Path(target_dir).resolve()), 'rollout_package_version': int(package['manifest']['rollout_package_version']), 'cached_rollout_diagnostics': dict(package['diagnostics']), 'cached_local_mpc_solver_fingerprint': dict(package['manifest']['local_mpc_solver_fingerprint']), 'cached_rollout_prediction_mode': str(package['manifest']['prediction_mode']), 'cached_rollout_forecast_backend': str(package['manifest']['forecast_backend']), 'comparison_forecast_backend': str(comparison_cfg.forecast.type)})
    rollout = grid_nb.RolloutResult(step_df=relabel_rollout_dataframe(package['step_df'], controller_label), agent_df=relabel_rollout_dataframe(package['agent_df'], controller_label), grid_df=relabel_rollout_dataframe(package['grid_df'], controller_label), summary=relabel_rollout_dataframe(package['summary_df'], controller_label), meta=meta)
    return {'manifest': package['manifest'], 'diagnostics': package['diagnostics'], 'rollout': rollout}
