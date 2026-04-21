from __future__ import annotations
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from controllers.action_feasibility import build_safety_local_numpy, compute_action_gap_metrics_numpy, merge_action_info_into_step_info
from controllers.mpc import gurobi_agent_mpc as local_mpc_module
from envs.grid.deployments import resolve_fixed_battery_spec
from predictors.artifacts import get_default_lstm_artifact_dir
from predictors.training import ensure_lstm_artifacts
from scripts.utils.forecast_shared_preset import merge_managed_forecast_controls
from scripts.mainline_compare import build_compare_economic_table, build_compare_safety_table, build_compare_warning_banner, compare_rollout_metrics, summarize_rollout_metrics
from scripts.utils.grid_notebook_plotting import plot_battery_power_and_soc_comparison, plot_global_misocp_validation, plot_net_load_comparison, plot_power_balance_bars, plot_power_balance_comparison, plot_price_prediction_comparison, plot_rollout_comparison_dashboard, plot_rollout_dashboard, plot_shared_forecast_vs_actual, plot_test_rollout, plot_test_voltage_profile, plot_voltage_profile_comparison
from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN, IMPORT_PRICE_MARKUP_KEY, IMPORT_PRICE_PRED_COLUMN, IMPORT_PRICE_SEQ_FIELD, WHOLESALE_PRICE_PRED_COLUMN, WHOLESALE_PRICE_SEQ_FIELD, derive_import_price, derive_import_price_seq, get_import_price_markup
PERFECT_PREDICTION_MODE = 'perfect'
NORMAL_PREDICTION_MODE = 'normal'
ORACLE_EVAL_MODE = 'oracle_eval'
FORECAST_EVAL_MODE = 'forecast_eval'

def normalize_prediction_mode(prediction_mode: str) -> str:
    normalized = str(prediction_mode).strip().lower()
    if normalized in {'perfect', 'perfect_prediction', 'perfect prediction'}:
        return PERFECT_PREDICTION_MODE
    if normalized in {'normal', 'normal_prediction', 'normal prediction'}:
        return NORMAL_PREDICTION_MODE
    raise ValueError(f"prediction_mode must be one of {{'perfect', 'normal'}}, got '{prediction_mode}'.")

def normalize_date_input(value: str | int | None) -> str | None:
    if value in (None, ''):
        return None
    text_value = str(value).strip()
    if len(text_value) == 8 and text_value.isdigit():
        return f'{text_value[:4]}-{text_value[4:6]}-{text_value[6:]}'
    return str(pd.Timestamp(text_value).date())

def normalize_agent_scale(scale: float | list[float] | tuple[float, ...], *, n_agents: int, name: str) -> list[float]:
    values = np.asarray(scale if isinstance(scale, (list, tuple, np.ndarray)) else [scale], dtype=np.float32)
    if values.size == 1:
        values = np.repeat(values, n_agents)
    if values.size != n_agents:
        raise ValueError(f'{name} should provide {n_agents} value(s), got {values.size}.')
    if np.any(values < 0.0):
        raise ValueError(f'{name} must be non-negative, got {values.tolist()}.')
    return values.astype(np.float32).tolist()

def resolve_battery_controls(cfg, battery_controls: Mapping[str, object] | None=None) -> dict[str, object]:
    controls = dict(battery_controls or {})
    capacity_kwh, c_rate, p_max_kw = resolve_fixed_battery_spec(controls.get('battery_capacity', cfg.env.battery_capacity), controls.get('max_charge_rate', cfg.env.max_charge_rate), n_agents=int(cfg.env.num_agents))
    resolved = {'mode': 'fixed', 'battery_capacity': list(capacity_kwh), 'max_charge_rate': float(c_rate), 'p_max_kw': list(p_max_kw), 'efficiency': float(controls.get('efficiency', cfg.env.efficiency)), 'init_soc': float(controls.get('init_soc', cfg.env.init_soc)), 'soc_min': float(controls.get('soc_min', cfg.env.soc_min)), 'soc_max': float(controls.get('soc_max', cfg.env.soc_max)), 'soc_target': float(controls.get('soc_target', cfg.env.soc_target))}
    if resolved['efficiency'] <= 0.0 or resolved['efficiency'] > 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {resolved['efficiency']}.")
    if not 0.0 <= resolved['soc_min'] <= resolved['soc_max'] <= 1.0:
        raise ValueError(f"Invalid SoC range: soc_min={resolved['soc_min']}, soc_max={resolved['soc_max']}.")
    if not 0.0 <= resolved['init_soc'] <= 1.0:
        raise ValueError(f"init_soc must be in [0, 1], got {resolved['init_soc']}.")
    if not 0.0 <= resolved['soc_target'] <= 1.0:
        raise ValueError(f"soc_target must be in [0, 1], got {resolved['soc_target']}.")
    return resolved

def _normalize_signal_training_overrides(overrides: Mapping[str, Mapping[str, object]] | None) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for signal_name, signal_overrides in dict(overrides or {}).items():
        if not isinstance(signal_overrides, Mapping):
            raise ValueError(f"signal_training_overrides['{signal_name}'] must be a mapping, got {signal_overrides!r}.")
        normalized[str(signal_name).strip().lower()] = {str(field_name): value for field_name, value in dict(signal_overrides).items()}
    return normalized

def _canonical_forecast_controls_from_cfg(cfg) -> dict[str, object]:
    artifact_root = getattr(cfg.forecast, 'lstm_artifact_root', None)
    resolved_artifact_root = None if artifact_root in (None, '') else str(Path(artifact_root).resolve())
    return {'artifact_root': resolved_artifact_root, 'target_signals': [str(value) for value in cfg.forecast.target_signals], 'future_horizon': int(cfg.env.future_horizon), 'history_window': int(cfg.forecast.history_window), 'load_model_mode': str(cfg.forecast.load_model_mode), 'load_component_split': bool(cfg.forecast.load_component_split), 'load_scaler_type': str(cfg.forecast.load_scaler_type), 'load_time_feature_mode': str(cfg.forecast.load_time_feature_mode), 'pv_time_feature_mode': str(cfg.forecast.pv_time_feature_mode), 'load_hybrid_mode': str(cfg.forecast.load_hybrid_mode), 'load_baseline_mode': str(cfg.forecast.load_baseline_mode), 'pv_postprocess_mode': str(cfg.forecast.pv_postprocess_mode), 'auto_train_missing': bool(cfg.forecast.auto_train_missing), 'signal_training_overrides': _normalize_signal_training_overrides(getattr(cfg.forecast, 'signal_training_overrides', {}))}

def resolve_forecast_controls(cfg, forecast_controls: Mapping[str, object] | None=None) -> dict[str, object]:
    controls = dict(forecast_controls or {})
    target_signals = [str(value) for value in controls.get('target_signals', cfg.forecast.target_signals)]
    history_window = int(controls.get('history_window', cfg.forecast.history_window))
    configured_future_horizon = controls.get('future_horizon')
    if configured_future_horizon is not None and int(configured_future_horizon) != int(cfg.env.future_horizon):
        raise ValueError(f'forecast_controls.future_horizon must match cfg.env.future_horizon, got {configured_future_horizon} vs {cfg.env.future_horizon}.')
    merged_controls = merge_managed_forecast_controls(_canonical_forecast_controls_from_cfg(cfg), controls)
    artifact_root = merged_controls.get('artifact_root', cfg.forecast.lstm_artifact_root)
    resolved_artifact_root = None if artifact_root in (None, '') else str(Path(artifact_root).resolve())
    signal_training_overrides = _normalize_signal_training_overrides(merged_controls.get('signal_training_overrides', getattr(cfg.forecast, 'signal_training_overrides', {})))
    resolved = {'target_signals': [str(value) for value in merged_controls.get('target_signals', target_signals)], 'history_window': int(merged_controls.get('history_window', history_window)), 'load_model_mode': str(merged_controls.get('load_model_mode', cfg.forecast.load_model_mode)), 'load_component_split': bool(merged_controls.get('load_component_split', cfg.forecast.load_component_split)), 'load_scaler_type': str(merged_controls.get('load_scaler_type', cfg.forecast.load_scaler_type)), 'load_time_feature_mode': str(merged_controls.get('load_time_feature_mode', cfg.forecast.load_time_feature_mode)), 'pv_time_feature_mode': str(merged_controls.get('pv_time_feature_mode', cfg.forecast.pv_time_feature_mode)), 'load_hybrid_mode': str(merged_controls.get('load_hybrid_mode', cfg.forecast.load_hybrid_mode)), 'load_baseline_mode': str(merged_controls.get('load_baseline_mode', cfg.forecast.load_baseline_mode)), 'pv_postprocess_mode': str(merged_controls.get('pv_postprocess_mode', cfg.forecast.pv_postprocess_mode)), 'auto_train_missing': bool(merged_controls.get('auto_train_missing', cfg.forecast.auto_train_missing)), 'artifact_root': resolved_artifact_root, 'signal_training_overrides': signal_training_overrides}
    return resolved

def resolve_forecast_backend(prediction_mode: str, future_horizon: int) -> str:
    mode = normalize_prediction_mode(prediction_mode)
    if mode == PERFECT_PREDICTION_MODE:
        return 'perfect'
    if int(future_horizon) <= 0:
        raise ValueError('Normal prediction mode requires future_horizon > 0 for the LSTM forecaster.')
    return 'lstm'

def resolve_evaluation_mode(prediction_mode: str) -> str:
    mode = normalize_prediction_mode(prediction_mode)
    return ORACLE_EVAL_MODE if mode == PERFECT_PREDICTION_MODE else FORECAST_EVAL_MODE

def resolve_prediction_mode_from_forecast_backend(forecast_backend: str) -> str:
    normalized = str(forecast_backend).strip().lower()
    return PERFECT_PREDICTION_MODE if normalized == 'perfect' else NORMAL_PREDICTION_MODE

def apply_notebook_experiment_settings(cfg, *, prediction_mode: str, test_start_date: str | int | None, test_end_date: str | int | None, agent_profiles: list[str] | None=None, agent_bus_ids: list[int] | None=None, load_scale: float | list[float] | None=None, pv_scale: float | list[float] | None=None, future_horizon: int | None=None, battery_controls: Mapping[str, object] | None=None, forecast_controls: Mapping[str, object] | None=None, train_year: int | None=None, test_year: int | None=None) -> dict[str, object]:
    cfg.obs.local_features = ['calendar_time', 'soc']
    cfg.obs.sequence_features = [WHOLESALE_PRICE_SEQ_FIELD.removesuffix('_seq'), 'load', 'pv']
    cfg.forecast.target_signals = [WHOLESALE_PRICE_SEQ_FIELD.removesuffix('_seq'), 'load', 'pv']
    cfg.data.agent_profiles = list(cfg.data.agent_profiles if agent_profiles is None else agent_profiles)
    cfg.env.num_agents = int(len(cfg.data.agent_profiles))
    cfg.env.future_horizon = int(cfg.env.future_horizon if future_horizon is None else future_horizon)
    resolved_agent_bus_ids = list(cfg.grid.agent_bus_ids if agent_bus_ids is None else agent_bus_ids)
    if len(resolved_agent_bus_ids) < cfg.env.num_agents:
        raise ValueError(f'grid.agent_bus_ids only provides {len(resolved_agent_bus_ids)} buses, but {cfg.env.num_agents} agents were requested.')
    cfg.grid.agent_bus_ids = [int(bus_id) for bus_id in resolved_agent_bus_ids[:cfg.env.num_agents]]
    if train_year is not None:
        cfg.data.train_year = int(train_year)
    if test_year is not None:
        cfg.data.test_year = int(test_year)
    cfg.data.test_start_date = normalize_date_input(test_start_date)
    cfg.data.test_end_date = normalize_date_input(test_end_date)
    resolved_load_scale = cfg.data.resolved_load_scale(cfg.env.num_agents) if load_scale is None else load_scale
    resolved_pv_scale = cfg.data.resolved_pv_scale(cfg.env.num_agents) if pv_scale is None else pv_scale
    cfg.data.load_scale = normalize_agent_scale(resolved_load_scale, n_agents=cfg.env.num_agents, name='load_scale')
    cfg.data.pv_scale = normalize_agent_scale(resolved_pv_scale, n_agents=cfg.env.num_agents, name='pv_scale')
    resolved_battery_controls = resolve_battery_controls(cfg, battery_controls)
    cfg.env.battery_capacity = list(resolved_battery_controls['battery_capacity'])
    cfg.env.max_charge_rate = resolved_battery_controls['max_charge_rate']
    cfg.env.efficiency = resolved_battery_controls['efficiency']
    cfg.env.init_soc = resolved_battery_controls['init_soc']
    cfg.env.soc_min = resolved_battery_controls['soc_min']
    cfg.env.soc_max = resolved_battery_controls['soc_max']
    cfg.env.soc_target = resolved_battery_controls['soc_target']
    if cfg.data.pv_capacity_kw and len(cfg.data.pv_capacity_kw) != cfg.env.num_agents:
        cfg.data.pv_capacity_kw = []
    resolved_prediction_mode = normalize_prediction_mode(prediction_mode)
    cfg.forecast.type = resolve_forecast_backend(resolved_prediction_mode, cfg.env.future_horizon)
    resolved_forecast_controls = resolve_forecast_controls(cfg, forecast_controls)
    cfg.forecast.target_signals = list(resolved_forecast_controls['target_signals'])
    cfg.forecast.history_window = int(resolved_forecast_controls['history_window'])
    cfg.forecast.load_model_mode = str(resolved_forecast_controls['load_model_mode'])
    cfg.forecast.load_component_split = bool(resolved_forecast_controls['load_component_split'])
    cfg.forecast.load_scaler_type = str(resolved_forecast_controls['load_scaler_type'])
    cfg.forecast.load_time_feature_mode = str(resolved_forecast_controls['load_time_feature_mode'])
    cfg.forecast.pv_time_feature_mode = str(resolved_forecast_controls['pv_time_feature_mode'])
    cfg.forecast.load_hybrid_mode = str(resolved_forecast_controls['load_hybrid_mode'])
    cfg.forecast.load_baseline_mode = str(resolved_forecast_controls['load_baseline_mode'])
    cfg.forecast.pv_postprocess_mode = str(resolved_forecast_controls['pv_postprocess_mode'])
    cfg.forecast.auto_train_missing = bool(resolved_forecast_controls['auto_train_missing'])
    cfg.forecast.signal_training_overrides = dict(resolved_forecast_controls['signal_training_overrides'])
    cfg.runtime.observation_normalization_state = None
    if cfg.forecast.type == 'lstm':
        cfg.forecast.lstm_artifact_root = resolved_forecast_controls['artifact_root'] or str(get_default_lstm_artifact_dir())
    else:
        cfg.forecast.lstm_artifact_root = resolved_forecast_controls['artifact_root']
    return {'prediction_mode': resolved_prediction_mode, 'evaluation_mode': resolve_evaluation_mode(resolved_prediction_mode), 'forecast_backend': cfg.forecast.type, 'future_horizon': int(cfg.env.future_horizon), 'test_start_date': cfg.data.test_start_date, 'test_end_date': cfg.data.test_end_date, 'agent_profiles': list(cfg.data.agent_profiles), 'load_scale': list(cfg.data.load_scale), 'pv_scale': list(cfg.data.pv_scale), 'battery': dict(resolved_battery_controls), 'forecast': {'artifact_root': cfg.forecast.lstm_artifact_root, 'target_signals': list(cfg.forecast.target_signals), 'history_window': int(cfg.forecast.history_window), 'auto_train_missing': bool(cfg.forecast.auto_train_missing), 'signal_training_overrides': dict(cfg.forecast.signal_training_overrides), 'load_component_split': bool(cfg.forecast.load_component_split), 'load_scaler_type': str(cfg.forecast.load_scaler_type)}, 'train_year': int(cfg.data.train_year), 'test_year': int(cfg.data.test_year), 'agent_bus_ids': list(cfg.grid.agent_bus_ids)}

def ensure_forecast_ready(cfg) -> dict[str, object] | None:
    if cfg.forecast.type != 'lstm':
        return None
    return ensure_lstm_artifacts(cfg, device=cfg.runtime.device)

def cache_only_forecast_enabled(cfg) -> bool:
    shared_data_dir = getattr(getattr(cfg, 'runtime', None), 'shared_data_dir', None)
    return shared_data_dir not in (None, '')

def build_comparison_cfg(cfg, *, prediction_mode: str):
    comparison_cfg = deepcopy(cfg)
    resolved_type = resolve_forecast_backend(prediction_mode, comparison_cfg.env.future_horizon)
    if resolved_type != str(cfg.forecast.type):
        comparison_cfg.runtime.shared_data_dir = None
        comparison_cfg.runtime.shared_data_signature = None
        comparison_cfg.runtime.forecast_ready = None
    comparison_cfg.forecast.type = resolved_type
    if comparison_cfg.forecast.type == 'lstm' and comparison_cfg.forecast.lstm_artifact_root is None:
        comparison_cfg.forecast.lstm_artifact_root = get_default_lstm_artifact_dir()
    return comparison_cfg

def _aligned_prediction(previous_obs: dict | None, current_obs: dict, field_name: str):
    current_value = np.asarray(current_obs[field_name], dtype=np.float32)
    if previous_obs is None:
        if current_value.ndim == 1:
            return float(current_value[0])
        return current_value[..., 0].astype(np.float32)
    previous_value = np.asarray(previous_obs[field_name], dtype=np.float32)
    forecast_index = 1 if previous_value.shape[-1] > 1 else 0
    if previous_value.ndim == 1:
        return float(previous_value[forecast_index])
    return previous_value[..., forecast_index].astype(np.float32)

def _step_timestamp(reset_info: dict[str, object], step_idx: int) -> pd.Timestamp:
    episode_meta = dict(reset_info.get('episode_meta', {}))
    timestamps = episode_meta.get('timestamps') or []
    if step_idx < len(timestamps):
        return pd.Timestamp(timestamps[step_idx])
    return pd.Timestamp(step_idx, unit='m')

def _purchase_cost_per_agent(info: dict[str, object], dt: float) -> np.ndarray:
    grid_import = np.asarray(info.get('grid_import_kw', np.maximum(np.asarray(info['net_load'], dtype=np.float32), 0.0)), dtype=np.float32)
    return (grid_import * np.float32(dt) * np.float32(info[IMPORT_PRICE_COLUMN])).astype(np.float32)

def _export_subsidy_per_agent(info: dict[str, object], dt: float, subsidy_rate: float) -> np.ndarray:
    grid_export = np.asarray(info.get('grid_export_kw', np.maximum(-np.asarray(info['net_load'], dtype=np.float32), 0.0)), dtype=np.float32)
    return (grid_export * np.float32(dt) * np.float32(subsidy_rate)).astype(np.float32)

def _mean_component_total(info: dict[str, object], component_key: str, n_agents: int) -> float:
    values = np.asarray(info.get(component_key, np.zeros((n_agents,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))

def _approx_trafo_limit_kw(env, *, loading_limit_pct: float | None=None) -> float | None:
    net = getattr(getattr(env, '_grid_core', None), 'net', None)
    trafo_table = getattr(net, 'trafo', None)
    if trafo_table is None or len(trafo_table) == 0:
        return None
    try:
        if 'sn_mva' not in trafo_table:
            return None
        sn_mva = np.asarray(trafo_table['sn_mva'], dtype=np.float64).reshape(-1)
        if sn_mva.size == 0:
            return None
        limit_scale = float(getattr(getattr(env, '_grid_cfg', None), 'line_max_loading_pct', 100.0) if loading_limit_pct is None else loading_limit_pct) / 100.0
        return float(np.sum(sn_mva) * max(limit_scale, 0.0) * 1000.0)
    except Exception:
        return None

def _fixed_feeder_components_kw(env) -> tuple[float, float]:
    net = getattr(getattr(env, '_grid_core', None), 'net', None)
    if net is None:
        return (0.0, 0.0)
    agent_bus_set = set((int(bus_id) for bus_id in getattr(getattr(env, '_grid_core', None), 'agent_bus_ids', [])))
    fixed_load_kw = 0.0
    fixed_generation_kw = 0.0
    try:
        load_table = getattr(net, 'load', None)
        if load_table is not None and (not load_table.empty) and ('bus' in load_table) and ('p_mw' in load_table):
            fixed_load_kw += float(np.asarray(load_table.loc[~load_table['bus'].isin(agent_bus_set), 'p_mw'], dtype=np.float64).sum() * 1000.0)
        sgen_table = getattr(net, 'sgen', None)
        if sgen_table is not None and (not sgen_table.empty) and ('bus' in sgen_table) and ('p_mw' in sgen_table):
            fixed_generation_kw += float(np.asarray(sgen_table.loc[~sgen_table['bus'].isin(agent_bus_set), 'p_mw'], dtype=np.float64).sum() * 1000.0)
    except Exception:
        return (0.0, 0.0)
    return (max(fixed_load_kw, 0.0), max(fixed_generation_kw, 0.0))

def load_training_run_bundle(model_root) -> dict[str, object]:
    resolved_model_root = Path(model_root).resolve()
    meta_dir = resolved_model_root / '_meta'
    required_files = ('train_result.json', 'experiment_controls.json', 'data_controls.json', 'battery_controls.json', 'train_controls.json', 'checkpoint_controls.json')
    missing = [filename for filename in required_files if not (meta_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Training metadata is incomplete for '{resolved_model_root}'. Missing: {missing}")
    bundle = {'model_root': str(resolved_model_root), 'meta_dir': str(meta_dir)}
    for filename in required_files:
        bundle[filename.removesuffix('.json')] = json.loads((meta_dir / filename).read_text(encoding='utf-8'))
    return bundle

def validate_compare_model_bundles(model_roots: Mapping[str, str | Path], *, expected_count: int=3) -> dict[str, dict[str, object]]:
    if len(model_roots) != int(expected_count):
        raise ValueError(f'Compare requires exactly {expected_count} DRL model roots, got {len(model_roots)}.')

    def _canonicalize(value):
        if isinstance(value, Mapping):
            return {str(key): _canonicalize(sub) for key, sub in sorted(dict(value).items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [_canonicalize(item) for item in value]
        return round(float(value), 10) if isinstance(value, float) else value
    bundles: dict[str, dict[str, object]] = {}
    signatures: dict[str, dict[str, object]] = {}
    for label, model_root in dict(model_roots).items():
        bundle = load_training_run_bundle(model_root)
        experiment_controls = dict(bundle['experiment_controls'])
        data_controls = dict(bundle['data_controls'])
        reward_controls = dict(experiment_controls.get('reward_controls', {}))
        removed_reward_keys = sorted(set(reward_controls).intersection({'w_action_pen', 'lambda_throughput'}))
        if removed_reward_keys:
            raise ValueError(f'Stored reward_controls use removed legacy key(s): {removed_reward_keys}. Regenerate the checkpoint bundle with the current mainline schema.')
        bundles[str(label)] = bundle
        train_controls = dict(bundle['train_controls'])
        signatures[str(label)] = {'prediction_mode': str(data_controls.get('prediction_mode', 'perfect')), 'seed': int(experiment_controls.get('seed', 0)), 'export_subsidy_eur_per_kwh': float(reward_controls.get('export_subsidy_eur_per_kwh', 0.079)), 'w_soc_pen': float(reward_controls.get('w_soc_pen', 0.0)), 'agent_profiles': list(data_controls.get('agent_profiles', [])), 'agent_bus_ids': list(data_controls.get('agent_bus_ids', [])), 'load_scale': list(data_controls.get('load_scale', [])), 'pv_scale': list(data_controls.get('pv_scale', [])), 'future_horizon': int(data_controls.get('future_horizon', 0)), 'train_year': data_controls.get('train_year'), 'test_year': data_controls.get('test_year'), 'test_start_date': data_controls.get('test_start_date'), 'test_end_date': data_controls.get('test_end_date'), 'shared_data_signature': bundle.get('train_result', {}).get('shared_data_signature'), 'battery_controls': dict(bundle['battery_controls']), 'forecast_controls': dict(experiment_controls.get('forecast_controls', {})), 'model_controls': dict(experiment_controls.get('model_controls', {})), 'train_controls': {key: train_controls.get(key) for key in ('profile', 'model_family', 'train_episodes', 'max_train_steps', 'num_envs', 'vec_env_type', 'parallel_episode_sampling', 'batch_size', 'buffer_size', 'update_interval', 'updates_per_step', 'policy_update_freq', 'actor_lr', 'critic_lr', 'noise_std_init', 'noise_std_min', 'noise_decay_steps', 'use_noise_decay')}}
    labels = list(signatures)
    reference_label = labels[0]
    reference_signature = signatures[reference_label]
    mismatch_lines: list[str] = []
    for label in labels[1:]:
        current_signature = signatures[label]
        for field_name, reference_value in reference_signature.items():
            current_value = current_signature[field_name]
            if _canonicalize(current_value) != _canonicalize(reference_value):
                mismatch_lines.append(f'{field_name}: {reference_label}={reference_value!r} vs {label}={current_value!r}')
    if mismatch_lines:
        joined = '\n'.join(mismatch_lines)
        raise ValueError(f'Compare model metadata mismatch detected. All DRL runs must share the same subsidy, hyperparameters, forecast controls, battery controls, dates, and seed.\n{joined}')
    return bundles

def _get_local_mpc_solver(env, *, agent_idx: int, import_price_seq: np.ndarray, battery_capacity_kwh: float, p_max_kw: float, dt_hours: float, efficiency: float, soc_min: float, soc_max: float, export_subsidy_eur_per_kwh: float):
    cache = getattr(env, '_local_mpc_solver_cache', None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, '_local_mpc_solver_cache', cache)
    stats = getattr(env, '_local_mpc_stats', None)
    if not isinstance(stats, dict):
        stats = {}
        setattr(env, '_local_mpc_stats', stats)
    for key in ('solver_build_count', 'solver_reuse_count', 'solve_count', 'solve_time_sec_total', 'guarded_fallback_count'):
        stats.setdefault(key, 0.0)
    use_guarded_fallback = False
    horizon = int(np.asarray(import_price_seq, dtype=np.float32).reshape(-1).size)
    cache_key = (int(agent_idx), int(horizon), float(battery_capacity_kwh), float(p_max_kw), float(dt_hours), float(efficiency), float(soc_min), float(soc_max), float(export_subsidy_eur_per_kwh), bool(use_guarded_fallback))
    solver = cache.get(cache_key)
    if solver is None:
        solver = local_mpc_module._ReusableLocalMPCSolver(horizon=horizon, battery_capacity_kwh=float(battery_capacity_kwh), p_max_kw=float(p_max_kw), dt_hours=float(dt_hours), efficiency=float(efficiency), soc_min=float(soc_min), soc_max=float(soc_max), export_subsidy_eur_per_kwh=float(export_subsidy_eur_per_kwh), use_guarded_fallback=bool(use_guarded_fallback))
        cache[cache_key] = solver
        stats['solver_build_count'] = float(stats['solver_build_count']) + 1.0
    else:
        stats['solver_reuse_count'] = float(stats['solver_reuse_count']) + 1.0
    return (solver, stats)

def _mpc_policy(env, obs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    requested_battery_kw = np.zeros((env.n,), dtype=np.float32)
    requested_pv_curtail_kw = np.zeros((env.n,), dtype=np.float32)
    wholesale_price_seq = np.asarray(obs[WHOLESALE_PRICE_SEQ_FIELD], dtype=np.float32)
    import_price_seq = derive_import_price_seq(wholesale_price_seq, markup_eur_per_kwh=float(getattr(env, 'import_price_markup_eur_per_kwh', get_import_price_markup(env))))
    load_seq = np.asarray(obs['load_seq'], dtype=np.float32)
    pv_seq = np.asarray(obs['pv_seq'], dtype=np.float32)
    export_subsidy_eur_per_kwh = float(getattr(env.reward_fn, 'export_subsidy_eur_per_kwh', 0.079))
    for agent_idx in range(env.n):
        solver, solver_stats = _get_local_mpc_solver(env, agent_idx=agent_idx, import_price_seq=import_price_seq, battery_capacity_kwh=float(env.agent_c_bat[agent_idx]), p_max_kw=float(env.agent_p_max[agent_idx]), dt_hours=float(env.dt), efficiency=float(env.eff), soc_min=float(env.soc_min), soc_max=float(env.soc_max), export_subsidy_eur_per_kwh=export_subsidy_eur_per_kwh)
        solve_result = solver.solve_full_horizon(import_price_seq=import_price_seq, load_seq=load_seq[agent_idx], pv_seq=pv_seq[agent_idx], soc=float(env.soc[agent_idx]), pv_curtail_upper_kw=np.maximum(pv_seq[agent_idx], 0.0).astype(np.float32, copy=False))
        if not bool(solve_result.feasible):
            raise RuntimeError(f'Local MPC returned an infeasible or unbounded full-horizon solution for agent {agent_idx}. This usually means the relaxed continuous local MPC formulation is not properly bounded on the current price window.')
        solver_stats['solve_count'] = float(solver_stats['solve_count']) + 1.0
        solver_stats['solve_time_sec_total'] = float(solver_stats['solve_time_sec_total']) + float(solve_result.solve_time_sec)
        if solve_result.used_guarded_fallback:
            solver_stats['guarded_fallback_count'] = float(solver_stats['guarded_fallback_count']) + 1.0
        requested_battery_kw[agent_idx] = np.float32(solve_result.signed_battery_kw[0]) if solve_result.signed_battery_kw.size else np.float32(0.0)
        requested_pv_curtail_kw[agent_idx] = np.float32(solve_result.pv_curtail_kw[0]) if solve_result.pv_curtail_kw.size else np.float32(0.0)
    actions, action_array = _assemble_global_oracle_actions(env, battery_power_kw=requested_battery_kw, pv_curtail_kw=requested_pv_curtail_kw)
    load_raw = np.asarray(env.get_signal_step('load'), dtype=np.float32)
    pv_raw = np.asarray(env.get_signal_step('pv'), dtype=np.float32)
    action_info = compute_action_gap_metrics_numpy(build_safety_local_numpy(soc=np.asarray(env.soc, dtype=np.float32), load_raw=load_raw, pv_raw=pv_raw, battery_capacity_kwh=np.asarray(env.agent_c_bat, dtype=np.float32), p_max_kw=np.asarray(env.agent_p_max, dtype=np.float32)), action_array, action_array)
    return (actions, action_info)

def _clip_global_oracle_battery_power_kw(env, battery_power_kw: np.ndarray) -> np.ndarray:
    battery_power_kw = np.asarray(battery_power_kw, dtype=np.float32)
    e_t = np.asarray(env.soc, dtype=np.float32) * np.asarray(env.agent_c_bat, dtype=np.float32)
    e_min = float(env.soc_min) * np.asarray(env.agent_c_bat, dtype=np.float32)
    e_max = float(env.soc_max) * np.asarray(env.agent_c_bat, dtype=np.float32)
    p_max = np.asarray(env.agent_p_max, dtype=np.float32)
    eff = max(float(env.eff), 1e-06)
    p_max_charge = np.minimum(p_max, np.maximum(0.0, (e_max - e_t) / (eff * float(env.dt))))
    p_max_discharge = np.minimum(p_max, np.maximum(0.0, (e_t - e_min) * eff / float(env.dt)))
    return np.clip(battery_power_kw, -p_max_discharge, p_max_charge).astype(np.float32)

def _assemble_global_oracle_actions(env, battery_power_kw: np.ndarray, pv_curtail_kw: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    clipped_battery_kw = _clip_global_oracle_battery_power_kw(env, battery_power_kw)
    pv_raw_kw = np.maximum(np.asarray(env.get_signal_step('pv'), dtype=np.float32), 0.0)
    clipped_curtail_kw = np.minimum(np.maximum(np.asarray(pv_curtail_kw, dtype=np.float32), 0.0), pv_raw_kw)
    p_max_kw = np.maximum(np.asarray(env.agent_p_max, dtype=np.float32), 1e-06)
    battery_action = np.clip(clipped_battery_kw / p_max_kw, -1.0, 1.0).astype(np.float32)
    pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
    valid_mask = pv_raw_kw > 1e-06
    pv_utilization[valid_mask] = 1.0 - clipped_curtail_kw[valid_mask] / pv_raw_kw[valid_mask]
    pv_action = np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0).astype(np.float32)
    action_array = np.stack([battery_action, pv_action], axis=-1).astype(np.float32)
    return ([action_array[agent_idx].copy() for agent_idx in range(env.n)], action_array)

@dataclass
class RolloutResult:
    step_df: pd.DataFrame
    agent_df: pd.DataFrame
    grid_df: pd.DataFrame
    summary: pd.DataFrame
    meta: dict[str, object]

def collect_controller_rollout(cfg, *, label: str, controller=None, controller_builder=None, action_fn=None, episode_indices: list[int] | None=None) -> RolloutResult:
    provided = int(controller is not None) + int(controller_builder is not None) + int(action_fn is not None)
    if provided != 1:
        raise ValueError('Provide exactly one of controller, controller_builder, or action_fn.')
    from scripts.builder import build_env
    if cache_only_forecast_enabled(cfg):
        cfg.runtime.forecast_ready = None
    else:
        cfg.runtime.forecast_ready = ensure_forecast_ready(cfg)
    env = build_env(cfg, mode='test')
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(env._grid_core, 'agent_bus_ids', cfg.grid.agent_bus_ids)]
    agent_bus_set = set(agent_bus_ids)
    loading_limit_pct = float(cfg.grid.line_max_loading_pct)
    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_kw(env)
    trafo_limit_kw = _approx_trafo_limit_kw(env, loading_limit_pct=loading_limit_pct)
    try:
        active_controller = controller_builder(env) if controller_builder is not None else controller
        effective_episode_indices = [int(index) for index in episode_indices] if episode_indices is not None else [int(index) for index in list(getattr(cfg.runtime, 'selected_episode_indices', []) or [])] or list(range(int(env.num_available_episodes)))
        for episode_idx in effective_episode_indices:
            obs, reset_info = env.reset(episode_idx=episode_idx)
            raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, 'build_raw') else obs
            if active_controller is not None:
                active_controller.reset()
            previous_raw_obs = None
            done = False
            step_in_episode = 0
            while not done:
                wholesale_price_pred = _aligned_prediction(previous_raw_obs, raw_obs, WHOLESALE_PRICE_SEQ_FIELD)
                import_price_pred = float(derive_import_price(wholesale_price_pred, markup_eur_per_kwh=float(getattr(env, 'import_price_markup_eur_per_kwh', get_import_price_markup(env)))))
                load_pred = _aligned_prediction(previous_raw_obs, raw_obs, 'load_seq')
                pv_pred = _aligned_prediction(previous_raw_obs, raw_obs, 'pv_seq')
                timestamp = _step_timestamp(reset_info, step_in_episode)
                if active_controller is not None:
                    actions = active_controller.act(obs, deterministic=True)
                    action_info = getattr(active_controller, 'last_action_info', None)
                else:
                    action_result = action_fn(env, raw_obs)
                    if isinstance(action_result, tuple) and len(action_result) == 2:
                        actions, action_info = action_result
                    else:
                        actions = action_result
                        action_info = None
                next_obs, reward, terminated, truncated, info = env.step(actions)
                reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
                apply_action_penalty = bool(getattr(active_controller, 'apply_action_penalty', False))
                info, action_penalty = merge_action_info_into_step_info(info, action_info, soc_pen_weight=float(cfg.reward.w_soc_pen), apply_action_penalty=apply_action_penalty)
                reward_array = reward_array - np.asarray(action_penalty, dtype=np.float32)
                info['reward'] = reward_array.astype(np.float32)
                del terminated, truncated
                raw_next_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, 'build_raw') and (not bool(info.get('episode_done', False))) else next_obs
                purchase_cost_per_agent = _purchase_cost_per_agent(info, float(env.dt))
                base_net_load = np.asarray(info.get('base_net_load', np.asarray(info['load'], dtype=np.float32) - np.asarray(info['pv'], dtype=np.float32)), dtype=np.float32)
                base_net_load_effective = np.asarray(info.get('base_net_load_effective', base_net_load), dtype=np.float32)
                net_load = np.asarray(info.get('net_load', base_net_load_effective + np.asarray(info['e_bat'], dtype=np.float32)), dtype=np.float32)
                pv_raw = np.asarray(info.get('pv_raw', info['pv']), dtype=np.float32)
                pv_effective = np.asarray(info.get('pv_effective', pv_raw), dtype=np.float32)
                pv_curtail = np.asarray(info.get('pv_curtail', pv_raw - pv_effective), dtype=np.float32)
                pv_utilization = np.asarray(info.get('pv_utilization', np.ones_like(pv_raw, dtype=np.float32)), dtype=np.float32)
                grid_import = np.asarray(info.get('grid_import_kw', np.maximum(net_load, 0.0)), dtype=np.float32)
                grid_export = np.asarray(info.get('grid_export_kw', np.maximum(-net_load, 0.0)), dtype=np.float32)
                controller_action_gap = np.asarray(info.get('controller_action_gap', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                battery_action_req = np.asarray(info.get('battery_action_req', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                battery_action_exec = np.asarray(info.get('battery_action_exec', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                pv_action_req = np.asarray(info.get('pv_action_req', np.ones(env.n, dtype=np.float32)), dtype=np.float32)
                pv_action_exec = np.asarray(info.get('pv_action_exec', info.get('pv_action', np.ones(env.n, dtype=np.float32))), dtype=np.float32)
                soc_penalty_unweighted = np.asarray(info.get('soc_penalty_unweighted', info.get('action_penalty_unweighted', np.zeros(env.n, dtype=np.float32))), dtype=np.float32)
                soc_penalty = np.asarray(info.get('r_soc_pen', info.get('r_action_pen', np.zeros(env.n, dtype=np.float32))), dtype=np.float32)
                battery_power = np.asarray(info['e_bat'], dtype=np.float32)
                battery_power_req = np.asarray(info.get('e_bat_req', battery_power), dtype=np.float32)
                battery_charge = np.clip(battery_power, 0.0, None).astype(np.float32)
                battery_discharge = np.maximum(-battery_power, 0.0).astype(np.float32)
                battery_charge_req = np.clip(battery_power_req, 0.0, None).astype(np.float32)
                battery_discharge_req = np.maximum(-battery_power_req, 0.0).astype(np.float32)
                pv_curtail_req = np.asarray(info.get('pv_curtail_req', pv_curtail), dtype=np.float32)
                battery_request_gap_kw = np.abs(battery_power_req - battery_power).astype(np.float32)
                pv_curtail_request_gap_kw = np.abs(pv_curtail_req - pv_curtail).astype(np.float32)
                line_loading_pct = np.asarray(info.get('line_loading_pct', np.zeros(0, dtype=np.float32)), dtype=np.float32)
                trafo_loading_pct = np.asarray(info.get('trafo_loading_pct', np.zeros(0, dtype=np.float32)), dtype=np.float32)
                misocp_fallback = float(info.get('misocp_fallback', 0.0))
                misocp_time_limit_feasible = float(info.get('misocp_time_limit_feasible', 0.0))
                solve_time_sec = float(info.get('solve_time_sec', np.nan))
                root_import_kw = float(info.get('root_import_kw', np.nan))
                root_export_kw = float(info.get('root_export_kw', np.nan))
                simultaneous_charge_discharge_kw_total = float(info.get('simultaneous_charge_discharge_kw_total', 0.0))
                simultaneous_agent_count = int(float(info.get('simultaneous_agent_count', 0.0)))
                simultaneous_step_flag = float(info.get('simultaneous_step_flag', 0.0))
                export_subsidy_per_agent = _export_subsidy_per_agent(info, float(env.dt), float(getattr(cfg.reward, 'export_subsidy_eur_per_kwh', 0.079)))
                voltage_penalty_per_agent = np.asarray(info.get('r_safe_v', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                line_penalty_per_agent = np.asarray(info.get('r_safe_line', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                trafo_penalty_per_agent = np.asarray(info.get('r_safe_trafo', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                objective_per_agent = (purchase_cost_per_agent - export_subsidy_per_agent + soc_penalty + voltage_penalty_per_agent + line_penalty_per_agent + trafo_penalty_per_agent).astype(np.float32)
                voltage_penalty_total = _mean_component_total(info, 'r_safe_v', env.n)
                line_penalty_total = _mean_component_total(info, 'r_safe_line', env.n)
                trafo_penalty_total = _mean_component_total(info, 'r_safe_trafo', env.n)
                soc_penalty_total = float(np.sum(soc_penalty))
                purchase_cost_total = float(np.sum(purchase_cost_per_agent))
                export_subsidy_total = float(np.sum(export_subsidy_per_agent))
                pp_root_p_kw = float(np.asarray(info.get('trafo_p_signed_kw', np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1).sum())
                agent_raw_net_load_kw = float(np.sum(base_net_load))
                agent_effective_net_load_kw = float(np.sum(base_net_load_effective))
                agent_post_action_net_load_kw = float(np.sum(net_load))
                feeder_raw_net_load_kw = float(agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_effective_net_load_kw = float(agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_post_action_net_load_kw = float(agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw)
                step_rows.append({'controller': label, 'episode_idx': episode_idx, 'step': step_in_episode, 'timestamp': timestamp, 'wholesale_price': float(info['wholesale_price']), IMPORT_PRICE_COLUMN: float(info[IMPORT_PRICE_COLUMN]), WHOLESALE_PRICE_PRED_COLUMN: float(wholesale_price_pred), IMPORT_PRICE_PRED_COLUMN: import_price_pred, 'base_net_load_total': float(np.sum(base_net_load)), 'base_net_load_effective_total': float(np.sum(base_net_load_effective)), 'net_load_total': float(np.sum(net_load)), 'agent_raw_net_load_kw': agent_raw_net_load_kw, 'agent_effective_net_load_kw': agent_effective_net_load_kw, 'agent_post_action_net_load_kw': agent_post_action_net_load_kw, 'fixed_load_kw': fixed_load_kw, 'fixed_generation_kw': fixed_generation_kw, 'feeder_raw_net_load_kw': feeder_raw_net_load_kw, 'feeder_effective_net_load_kw': feeder_effective_net_load_kw, 'feeder_post_action_net_load_kw': feeder_post_action_net_load_kw, 'load_total': float(np.sum(np.asarray(info['load'], dtype=np.float32))), 'pv_raw_total': float(np.sum(pv_raw)), 'pv_effective_total': float(np.sum(pv_effective)), 'pv_curtail_total': float(np.sum(pv_curtail)), 'grid_import_total': float(np.sum(grid_import)), 'grid_export_total': float(np.sum(grid_export)), 'battery_charge_total': float(np.sum(battery_charge)), 'battery_charge_req_total': float(np.sum(battery_charge_req)), 'battery_discharge_total': float(np.sum(battery_discharge)), 'battery_discharge_req_total': float(np.sum(battery_discharge_req)), 'pv_curtail_req_total': float(np.sum(pv_curtail_req)), 'battery_request_gap_kw_total': float(np.sum(battery_request_gap_kw)), 'pv_curtail_request_gap_kw_total': float(np.sum(pv_curtail_request_gap_kw)), 'projector_adjustment_kw_total': float(np.sum(battery_request_gap_kw) + np.sum(pv_curtail_request_gap_kw)), 'purchase_cost_total': purchase_cost_total, 'export_subsidy_total': export_subsidy_total, 'soc_penalty_total': soc_penalty_total, 'voltage_penalty_total': voltage_penalty_total, 'line_penalty_total': line_penalty_total, 'trafo_penalty_total': trafo_penalty_total, 'psi_v_raw': float(info.get('psi_v_raw', 0.0)), 'psi_line_raw': float(info.get('psi_line_raw', 0.0)), 'psi_trafo_raw': float(info.get('psi_trafo_raw', 0.0)), 'line_loading_pct_max': float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0, 'trafo_loading_pct_max': float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0, 'line_violation': float(info.get('line_violation', info.get('l_violation', 0.0))), 'trafo_violation': float(info.get('trafo_violation', 0.0)), 'n_line_violations': int(info.get('n_line_violations', info.get('n_l_violations', int(np.any(line_loading_pct > loading_limit_pct))))), 'n_trafo_violations': int(info.get('n_trafo_violations', info.get('n_t_violations', int(np.any(trafo_loading_pct > loading_limit_pct))))), 'objective_total': purchase_cost_total - export_subsidy_total + soc_penalty_total + voltage_penalty_total + line_penalty_total + trafo_penalty_total, 'voltage_violation_count': int(((np.asarray(info.get('vm_pu', []), dtype=np.float32) < float(cfg.grid.v_min_pu)) | (np.asarray(info.get('vm_pu', []), dtype=np.float32) > float(cfg.grid.v_max_pu))).sum()), 'controller_action_gap_total': float(np.sum(controller_action_gap)), 'soc_penalty_step_total': soc_penalty_total, 'misocp_fallback': misocp_fallback, 'misocp_time_limit_feasible': misocp_time_limit_feasible, 'solve_time_sec': solve_time_sec, 'root_import_kw': root_import_kw, 'root_export_kw': root_export_kw, 'simultaneous_charge_discharge_kw_total': simultaneous_charge_discharge_kw_total, 'simultaneous_agent_count': simultaneous_agent_count, 'simultaneous_step_flag': simultaneous_step_flag, 'trafo_limit_reference_kw': trafo_limit_kw})
                last_diagnostic = getattr(active_controller, 'last_diagnostic', None) if active_controller is not None else None
                if isinstance(last_diagnostic, dict) and last_diagnostic:
                    diagnostic_row = dict(last_diagnostic)
                    diagnostic_row['controller'] = label
                    diagnostic_row['episode_idx'] = episode_idx
                    diagnostic_row['step'] = step_in_episode
                    diagnostic_row['timestamp'] = timestamp
                    diagnostic_row['pp_vm_pu'] = np.asarray(info.get('vm_pu', np.zeros(0, dtype=np.float32)), dtype=np.float32).copy()
                    diagnostic_row['pp_line_loading_pct'] = line_loading_pct.copy()
                    diagnostic_row['pp_trafo_loading_pct'] = trafo_loading_pct.copy()
                    diagnostic_row['pp_root_p_kw'] = float(np.asarray(info.get('trafo_p_signed_kw', np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)[0])
                    diagnostic_rows.append(diagnostic_row)
                for agent_idx, profile in enumerate(cfg.data.agent_profiles):
                    agent_rows.append({'controller': label, 'episode_idx': episode_idx, 'step': step_in_episode, 'timestamp': timestamp, 'agent_id': agent_idx, 'agent_profile': str(profile), 'load': float(np.asarray(info['load'], dtype=np.float32)[agent_idx]), 'load_pred': float(np.asarray(load_pred, dtype=np.float32)[agent_idx]), 'pv': float(np.asarray(info['pv'], dtype=np.float32)[agent_idx]), 'pv_raw': float(pv_raw[agent_idx]), 'pv_effective': float(pv_effective[agent_idx]), 'pv_curtail': float(pv_curtail[agent_idx]), 'pv_curtail_req': float(pv_curtail_req[agent_idx]), 'pv_utilization': float(pv_utilization[agent_idx]), 'pv_pred': float(np.asarray(pv_pred, dtype=np.float32)[agent_idx]), 'base_net_load': float(base_net_load[agent_idx]), 'base_net_load_effective': float(base_net_load_effective[agent_idx]), 'net_load': float(net_load[agent_idx]), 'grid_import_kw': float(grid_import[agent_idx]), 'grid_export_kw': float(grid_export[agent_idx]), 'e_bat': float(battery_power[agent_idx]), 'e_bat_req': float(battery_power_req[agent_idx]), 'battery_action_req': float(battery_action_req[agent_idx]), 'battery_action_exec': float(battery_action_exec[agent_idx]), 'pv_action_req': float(pv_action_req[agent_idx]), 'pv_action_exec': float(pv_action_exec[agent_idx]), 'controller_action_gap': float(controller_action_gap[agent_idx]), 'battery_request_gap_kw': float(battery_request_gap_kw[agent_idx]), 'pv_curtail_request_gap_kw': float(pv_curtail_request_gap_kw[agent_idx]), 'soc_penalty_unweighted': float(soc_penalty_unweighted[agent_idx]), 'r_soc_pen': float(soc_penalty[agent_idx]), 'r_safe_v': float(voltage_penalty_per_agent[agent_idx]), 'r_safe_line': float(line_penalty_per_agent[agent_idx]), 'r_safe_trafo': float(trafo_penalty_per_agent[agent_idx]), 'soc': float(np.asarray(info['soc_next'], dtype=np.float32)[agent_idx]), 'purchase_cost': float(purchase_cost_per_agent[agent_idx]), 'export_subsidy': float(export_subsidy_per_agent[agent_idx]), 'objective_total': float(objective_per_agent[agent_idx])})
                vm_pu = np.asarray(info.get('vm_pu', np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32)
                if vm_pu.shape[0] == len(bus_ids):
                    for bus_id, vm_value in zip(bus_ids, vm_pu, strict=False):
                        grid_rows.append({'controller': label, 'episode_idx': episode_idx, 'step': step_in_episode, 'timestamp': timestamp, 'bus_id': int(bus_id), 'vm_pu': float(vm_value), 'is_agent_bus': bool(int(bus_id) in agent_bus_set)})
                done = bool(info.get('episode_done', False))
                previous_raw_obs = raw_obs
                obs = next_obs
                raw_obs = raw_next_obs
                step_in_episode += 1
        step_df = pd.DataFrame(step_rows).sort_values(['episode_idx', 'step']).reset_index(drop=True)
        agent_df = pd.DataFrame(agent_rows).sort_values(['episode_idx', 'step', 'agent_id']).reset_index(drop=True)
        grid_df = pd.DataFrame(grid_rows).sort_values(['episode_idx', 'step', 'bus_id']).reset_index(drop=True)
        summary = agent_df.groupby(['controller', 'agent_profile'], as_index=False)[['purchase_cost', 'export_subsidy', 'objective_total']].sum() if not agent_df.empty else pd.DataFrame(columns=['controller', 'agent_profile', 'purchase_cost', 'export_subsidy', 'objective_total'])
        return RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=summary, meta={'controller': label, 'n_agents': int(env.n), 'agent_profiles': list(cfg.data.agent_profiles), 'agent_bus_ids': agent_bus_ids, 'bus_ids': bus_ids, 'v_min_pu': float(cfg.grid.v_min_pu), 'v_max_pu': float(cfg.grid.v_max_pu), 'future_horizon': int(cfg.env.future_horizon), 'prediction_mode': resolve_prediction_mode_from_forecast_backend(cfg.forecast.type), 'evaluation_mode': resolve_evaluation_mode(resolve_prediction_mode_from_forecast_backend(cfg.forecast.type)), 'dt_hours': float(cfg.env.dt), 'trafo_limit_kw': trafo_limit_kw, 'trafo_limit_note': 'Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.', 'loading_limit_pct': loading_limit_pct, 'trafo_loading_limit_pct': loading_limit_pct, 'export_subsidy_eur_per_kwh': float(getattr(cfg.reward, 'export_subsidy_eur_per_kwh', 0.079)), IMPORT_PRICE_MARKUP_KEY: float(getattr(getattr(cfg, 'reward', None), IMPORT_PRICE_MARKUP_KEY, 0.0)), 'local_mpc_solver_build_count': int(float(getattr(env, '_local_mpc_stats', {}).get('solver_build_count', 0.0))), 'local_mpc_solver_reuse_count': int(float(getattr(env, '_local_mpc_stats', {}).get('solver_reuse_count', 0.0))), 'local_mpc_solve_count': int(float(getattr(env, '_local_mpc_stats', {}).get('solve_count', 0.0))), 'local_mpc_total_solve_time_sec': float(getattr(env, '_local_mpc_stats', {}).get('solve_time_sec_total', 0.0)), 'local_mpc_avg_solve_time_sec': float(float(getattr(env, '_local_mpc_stats', {}).get('solve_time_sec_total', 0.0)) / max(float(getattr(env, '_local_mpc_stats', {}).get('solve_count', 0.0)), 1.0) if float(getattr(env, '_local_mpc_stats', {}).get('solve_count', 0.0)) > 0.0 else 0.0), 'local_mpc_guarded_fallback_count': int(float(getattr(env, '_local_mpc_stats', {}).get('guarded_fallback_count', 0.0))), 'controller_diagnostic_log': diagnostic_rows, 'soc_mode': 'reset', 'selected_episode_indices': effective_episode_indices})
    finally:
        env.close()

def collect_madrl_rollout(cfg, *, model_root=None, algorithm: str | None=None, episode_tag: int | None=None, experiment_name: str='grid_mainline', checkpoint_root=None, label: str | None=None, episode_indices: list[int] | None=None) -> RolloutResult:
    from scripts.utils.experiment_notebook_utils import load_madrl_controller
    prediction_mode = resolve_prediction_mode_from_forecast_backend(cfg.forecast.type)
    loaded = load_madrl_controller(cfg, model_root, algorithm=algorithm, episode_tag=episode_tag, device=cfg.runtime.device, prediction_mode=prediction_mode, experiment_name=experiment_name, checkpoint_root=checkpoint_root)
    return collect_controller_rollout(loaded['cfg'], label=label or f'DRL ({resolve_evaluation_mode(prediction_mode)})', controller=loaded['controller'], episode_indices=episode_indices)

def collect_local_mpc_rollout(cfg, *, prediction_mode: str, label: str | None=None) -> RolloutResult:
    comparison_cfg = build_comparison_cfg(cfg, prediction_mode=prediction_mode)
    resolved_mode = normalize_prediction_mode(prediction_mode)
    rollout_label = label or f'Local MPC ({resolve_evaluation_mode(resolved_mode)})'
    rollout = collect_controller_rollout(comparison_cfg, label=rollout_label, action_fn=_mpc_policy)
    if not rollout.step_df.empty and {'purchase_cost_total', 'export_subsidy_total'}.issubset(rollout.step_df.columns):
        rollout.step_df['objective_total'] = rollout.step_df['purchase_cost_total'].astype(float) - rollout.step_df['export_subsidy_total'].astype(float)
    if not rollout.agent_df.empty and {'purchase_cost', 'export_subsidy'}.issubset(rollout.agent_df.columns):
        rollout.agent_df['objective_total'] = rollout.agent_df['purchase_cost'].astype(float) - rollout.agent_df['export_subsidy'].astype(float)
    if not rollout.summary.empty and {'purchase_cost', 'export_subsidy'}.issubset(rollout.summary.columns):
        rollout.summary['objective_total'] = rollout.summary['purchase_cost'].astype(float) - rollout.summary['export_subsidy'].astype(float)
    rollout.meta['local_mpc_price_mode'] = 'import_adjusted'
    rollout.meta['local_mpc_objective_mode'] = 'economic_only'
    return rollout

def collect_global_full_horizon_rollout(cfg, *, label: str | None=None, time_limit_sec: float=600.0) -> RolloutResult:
    from controllers.mpc.global_socp_mpc import GurobiSolveConfig, GlobalMISOCPProblem, _FULL_HORIZON_TIME_LIMIT_SEC, _SIMULTANEOUS_THRESHOLD_RATIO, default_primary_solve_config, default_retry_solve_config
    from scripts.builder import build_env
    comparison_cfg = build_comparison_cfg(cfg, prediction_mode=PERFECT_PREDICTION_MODE)
    rollout_label = str(label) if label is not None else ''
    resolved_time_limit = float(time_limit_sec if time_limit_sec is not None else _FULL_HORIZON_TIME_LIMIT_SEC)
    if cache_only_forecast_enabled(comparison_cfg):
        comparison_cfg.runtime.forecast_ready = None
    else:
        comparison_cfg.runtime.forecast_ready = ensure_forecast_ready(comparison_cfg)
    env = build_env(comparison_cfg, mode='test')
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(env._grid_core, 'agent_bus_ids', comparison_cfg.grid.agent_bus_ids)]
    agent_bus_set = set(agent_bus_ids)
    loading_limit_pct = float(comparison_cfg.grid.line_max_loading_pct)
    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_kw(env)
    trafo_limit_kw = _approx_trafo_limit_kw(env, loading_limit_pct=loading_limit_pct)
    try:
        problem = GlobalMISOCPProblem.from_env(env, comparison_cfg)
        selected_episode_indices = list(getattr(comparison_cfg.runtime, 'selected_episode_indices', []) or [])
        full_input = problem.build_full_horizon_input(env, episode_indices=None if not selected_episode_indices else selected_episode_indices)
        export_subsidy = float(getattr(comparison_cfg.reward, 'export_subsidy_eur_per_kwh', problem.export_subsidy_default))
        primary_config = default_primary_solve_config()
        primary_config = GurobiSolveConfig(time_limit_sec=resolved_time_limit, mip_gap=float(primary_config.mip_gap), threads=primary_config.threads, presolve=primary_config.presolve, cuts=primary_config.cuts, heuristics=primary_config.heuristics, mip_focus=primary_config.mip_focus)
        retry_config = default_retry_solve_config(primary_config)
        result = problem.solve_adaptive_full_horizon(full_input, export_subsidy=export_subsidy, verbose=False, primary_window_steps=int(full_input.horizon_steps), fallback_window_steps=96, solve_config=primary_config, retry_solve_config=retry_config, export_debug=True, debug_tag='adaptive_global_misocp')
        solve_mode = str(result.solve_mode)
        if not result.has_solution:
            raise RuntimeError(f'Global MISOCP failed to produce a feasible incumbent in both single_window and chunked_window modes. Final status={result.status_label!r}, debug_artifacts={result.debug_artifacts}.')
        if not rollout_label:
            rollout_label = 'Global MISOCP (chunked, near-optimal)' if solve_mode == 'chunked_window' else 'Global MISOCP (single_window)'
        carried_soc = np.asarray(full_input.soc_init, dtype=np.float32).copy()
        threshold_kw = (_SIMULTANEOUS_THRESHOLD_RATIO * np.asarray(env.agent_p_max, dtype=np.float32)).astype(np.float32)
        for episode_list_idx, episode_idx in enumerate(full_input.episode_indices.tolist()):
            obs, reset_info = env.reset(episode_idx=int(episode_idx))
            if episode_list_idx > 0:
                env.soc = carried_soc.copy()
                obs = env.obs_builder.build(env)
            raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, 'build_raw') else obs
            previous_raw_obs = None
            episode_offset = int(full_input.episode_offsets[episode_list_idx])
            episode_length = int(full_input.episode_lengths[episode_list_idx])
            for step_in_episode in range(episode_length):
                global_step = episode_offset + step_in_episode
                timestamp = _step_timestamp(reset_info, step_in_episode)
                wholesale_price_pred = _aligned_prediction(previous_raw_obs, raw_obs, WHOLESALE_PRICE_SEQ_FIELD)
                import_price_pred = float(derive_import_price(wholesale_price_pred, markup_eur_per_kwh=float(getattr(env, 'import_price_markup_eur_per_kwh', get_import_price_markup(env)))))
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
                info, action_penalty = merge_action_info_into_step_info(info, action_info, soc_pen_weight=float(comparison_cfg.reward.w_soc_pen), apply_action_penalty=False)
                reward_array = reward_array - np.asarray(action_penalty, dtype=np.float32)
                info['reward'] = reward_array.astype(np.float32)
                del terminated, truncated
                raw_next_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, 'build_raw') and (not bool(info.get('episode_done', False))) else next_obs
                purchase_cost_per_agent = _purchase_cost_per_agent(info, float(env.dt))
                base_net_load = np.asarray(info.get('base_net_load', np.asarray(info['load'], dtype=np.float32) - np.asarray(info['pv'], dtype=np.float32)), dtype=np.float32)
                base_net_load_effective = np.asarray(info.get('base_net_load_effective', base_net_load), dtype=np.float32)
                net_load = np.asarray(info.get('net_load', base_net_load_effective + np.asarray(info['e_bat'], dtype=np.float32)), dtype=np.float32)
                pv_raw = np.asarray(info.get('pv_raw', info['pv']), dtype=np.float32)
                pv_effective = np.asarray(info.get('pv_effective', pv_raw), dtype=np.float32)
                pv_curtail = np.asarray(info.get('pv_curtail', pv_raw - pv_effective), dtype=np.float32)
                pv_utilization = np.asarray(info.get('pv_utilization', np.ones_like(pv_raw, dtype=np.float32)), dtype=np.float32)
                grid_import = np.asarray(info.get('grid_import_kw', np.maximum(net_load, 0.0)), dtype=np.float32)
                grid_export = np.asarray(info.get('grid_export_kw', np.maximum(-net_load, 0.0)), dtype=np.float32)
                controller_action_gap = np.asarray(info.get('controller_action_gap', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                battery_action_req = np.asarray(info.get('battery_action_req', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                battery_action_exec = np.asarray(info.get('battery_action_exec', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                pv_action_req = np.asarray(info.get('pv_action_req', np.ones(env.n, dtype=np.float32)), dtype=np.float32)
                pv_action_exec = np.asarray(info.get('pv_action_exec', info.get('pv_action', np.ones(env.n, dtype=np.float32))), dtype=np.float32)
                soc_penalty_unweighted = np.asarray(info.get('soc_penalty_unweighted', info.get('action_penalty_unweighted', np.zeros(env.n, dtype=np.float32))), dtype=np.float32)
                soc_penalty = np.asarray(info.get('r_soc_pen', info.get('r_action_pen', np.zeros(env.n, dtype=np.float32))), dtype=np.float32)
                battery_power = np.asarray(info['e_bat'], dtype=np.float32)
                battery_power_req = np.asarray(info.get('e_bat_req', battery_power), dtype=np.float32)
                battery_charge = np.clip(battery_power, 0.0, None).astype(np.float32)
                battery_discharge = np.maximum(-battery_power, 0.0).astype(np.float32)
                battery_charge_req = np.clip(battery_power_req, 0.0, None).astype(np.float32)
                battery_discharge_req = np.maximum(-battery_power_req, 0.0).astype(np.float32)
                pv_curtail_req = np.asarray(info.get('pv_curtail_req', pv_curtail), dtype=np.float32)
                battery_request_gap_kw = np.abs(battery_power_req - battery_power).astype(np.float32)
                pv_curtail_request_gap_kw = np.abs(pv_curtail_req - pv_curtail).astype(np.float32)
                line_loading_pct = np.asarray(info.get('line_loading_pct', np.zeros(0, dtype=np.float32)), dtype=np.float32)
                trafo_loading_pct = np.asarray(info.get('trafo_loading_pct', np.zeros(0, dtype=np.float32)), dtype=np.float32)
                export_subsidy_per_agent = _export_subsidy_per_agent(info, float(env.dt), export_subsidy)
                voltage_penalty_per_agent = np.asarray(info.get('r_safe_v', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                line_penalty_per_agent = np.asarray(info.get('r_safe_line', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                trafo_penalty_per_agent = np.asarray(info.get('r_safe_trafo', np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                objective_per_agent = (purchase_cost_per_agent - export_subsidy_per_agent + soc_penalty + voltage_penalty_per_agent + line_penalty_per_agent + trafo_penalty_per_agent).astype(np.float32)
                voltage_penalty_total = _mean_component_total(info, 'r_safe_v', env.n)
                line_penalty_total = _mean_component_total(info, 'r_safe_line', env.n)
                trafo_penalty_total = _mean_component_total(info, 'r_safe_trafo', env.n)
                soc_penalty_total = float(np.sum(soc_penalty))
                purchase_cost_total = float(np.sum(purchase_cost_per_agent))
                export_subsidy_total = float(np.sum(export_subsidy_per_agent))
                pp_root_p_kw = float(np.asarray(info.get('trafo_p_signed_kw', np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1).sum())
                agent_raw_net_load_kw = float(np.sum(base_net_load))
                agent_effective_net_load_kw = float(np.sum(base_net_load_effective))
                agent_post_action_net_load_kw = float(np.sum(net_load))
                feeder_raw_net_load_kw = float(agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_effective_net_load_kw = float(agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_post_action_net_load_kw = float(agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw)
                step_rows.append({'controller': rollout_label, 'episode_idx': int(episode_idx), 'step': step_in_episode, 'timestamp': timestamp, 'wholesale_price': float(info['wholesale_price']), IMPORT_PRICE_COLUMN: float(info[IMPORT_PRICE_COLUMN]), WHOLESALE_PRICE_PRED_COLUMN: float(wholesale_price_pred), IMPORT_PRICE_PRED_COLUMN: import_price_pred, 'base_net_load_total': agent_raw_net_load_kw, 'base_net_load_effective_total': agent_effective_net_load_kw, 'net_load_total': agent_post_action_net_load_kw, 'agent_raw_net_load_kw': agent_raw_net_load_kw, 'agent_effective_net_load_kw': agent_effective_net_load_kw, 'agent_post_action_net_load_kw': agent_post_action_net_load_kw, 'fixed_load_kw': fixed_load_kw, 'fixed_generation_kw': fixed_generation_kw, 'feeder_raw_net_load_kw': feeder_raw_net_load_kw, 'feeder_effective_net_load_kw': feeder_effective_net_load_kw, 'feeder_post_action_net_load_kw': feeder_post_action_net_load_kw, 'root_net_exchange_kw': pp_root_p_kw, 'pp_root_p_kw': pp_root_p_kw, 'load_total': float(np.sum(np.asarray(info['load'], dtype=np.float32))), 'pv_raw_total': float(np.sum(pv_raw)), 'pv_effective_total': float(np.sum(pv_effective)), 'pv_curtail_total': float(np.sum(pv_curtail)), 'grid_import_total': float(np.sum(grid_import)), 'grid_export_total': float(np.sum(grid_export)), 'battery_charge_total': float(np.sum(battery_charge)), 'battery_charge_req_total': float(np.sum(battery_charge_req)), 'battery_discharge_total': float(np.sum(battery_discharge)), 'battery_discharge_req_total': float(np.sum(battery_discharge_req)), 'pv_curtail_req_total': float(np.sum(pv_curtail_req)), 'battery_request_gap_kw_total': float(np.sum(battery_request_gap_kw)), 'pv_curtail_request_gap_kw_total': float(np.sum(pv_curtail_request_gap_kw)), 'projector_adjustment_kw_total': float(np.sum(battery_request_gap_kw) + np.sum(pv_curtail_request_gap_kw)), 'purchase_cost_total': purchase_cost_total, 'export_subsidy_total': export_subsidy_total, 'soc_penalty_total': soc_penalty_total, 'voltage_penalty_total': voltage_penalty_total, 'line_penalty_total': line_penalty_total, 'trafo_penalty_total': trafo_penalty_total, 'psi_v_raw': float(info.get('psi_v_raw', 0.0)), 'psi_line_raw': float(info.get('psi_line_raw', 0.0)), 'psi_trafo_raw': float(info.get('psi_trafo_raw', 0.0)), 'line_loading_pct_max': float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0, 'trafo_loading_pct_max': float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0, 'line_violation': float(info.get('line_violation', info.get('l_violation', 0.0))), 'trafo_violation': float(info.get('trafo_violation', 0.0)), 'n_line_violations': int(info.get('n_line_violations', info.get('n_l_violations', int(np.any(line_loading_pct > loading_limit_pct))))), 'n_trafo_violations': int(info.get('n_trafo_violations', info.get('n_t_violations', int(np.any(trafo_loading_pct > loading_limit_pct))))), 'objective_total': purchase_cost_total - export_subsidy_total + soc_penalty_total + voltage_penalty_total + line_penalty_total + trafo_penalty_total, 'voltage_violation_count': int(((np.asarray(info.get('vm_pu', []), dtype=np.float32) < float(comparison_cfg.grid.v_min_pu)) | (np.asarray(info.get('vm_pu', []), dtype=np.float32) > float(comparison_cfg.grid.v_max_pu))).sum()), 'controller_action_gap_total': float(np.sum(controller_action_gap)), 'soc_penalty_step_total': soc_penalty_total, 'misocp_fallback': 0.0, 'misocp_time_limit_feasible': float(result.time_limit_feasible), 'solve_time_sec': float(result.solve_time_sec), 'root_import_kw': float(result.root_import_mw[global_step] * 1000.0), 'root_export_kw': float(result.root_export_mw[global_step] * 1000.0), 'simultaneous_charge_discharge_kw_total': float(np.sum(simultaneous_kw)), 'simultaneous_agent_count': int(np.sum(simultaneous_mask)), 'simultaneous_step_flag': float(np.any(simultaneous_mask)), 'trafo_limit_reference_kw': trafo_limit_kw})
                diagnostic_row = {'solver_type': 'gurobi_misocp', 'status_code': int(result.status_code), 'status_label': str(result.status_label), 'solve_mode': str(solve_mode), 'solve_time_sec': float(result.solve_time_sec), 'misocp_fallback': 0.0, 'misocp_time_limit_feasible': float(result.time_limit_feasible), 'mip_gap': float(result.mip_gap), 'best_bound': float(result.best_bound), 'objective_value': float(result.objective_value), 'agent_purchase_cost_eur': float(result.agent_purchase_cost_eur), 'agent_export_subsidy_eur': float(result.agent_export_subsidy_eur), 'agent_net_cost_eur': float(result.agent_net_cost_eur), 'feeder_purchase_cost_eur': float(result.feeder_purchase_cost_eur), 'feeder_export_subsidy_eur': float(result.feeder_export_subsidy_eur), 'feeder_net_cost_eur': float(result.feeder_net_cost_eur), 'throughput_regularization_eur': float(result.throughput_regularization_eur), 'throughput_regularization_weight': float(result.throughput_regularization_weight), 'root_import_kw': float(result.root_import_mw[global_step] * 1000.0), 'root_export_kw': float(result.root_export_mw[global_step] * 1000.0), 'root_p_kw': float(result.root_p_kw[global_step]), 'root_q_kvar': float(result.root_q_kvar[global_step]), 'misocp_vm_pu': np.asarray(result.bus_vm_pu[:, global_step], dtype=np.float32).copy(), 'misocp_line_loading_pct': np.asarray(result.line_loading_pct[:, global_step], dtype=np.float32).copy(), 'misocp_trafo_loading_pct': np.asarray(result.trafo_loading_pct[:, global_step], dtype=np.float32).copy(), 'model_size_num_vars': float(result.model_size.num_vars), 'model_size_num_binary_vars': float(result.model_size.num_binary_vars), 'model_size_num_linear_constraints': float(result.model_size.num_linear_constraints), 'model_size_num_quadratic_constraints': float(result.model_size.num_quadratic_constraints), 'simultaneous_step_ratio': float(result.simultaneous_step_ratio), 'max_simultaneous_kw': float(result.max_simultaneous_kw), 'simultaneous_agent_steps': float(result.simultaneous_agent_steps), 'debug_artifacts': dict(result.debug_artifacts), 'sanity_warning': str(result.sanity_warning or ''), 'controller': rollout_label, 'episode_idx': int(episode_idx), 'step': step_in_episode, 'timestamp': timestamp, 'pp_vm_pu': np.asarray(info.get('vm_pu', np.zeros(0, dtype=np.float32)), dtype=np.float32).copy(), 'pp_line_loading_pct': line_loading_pct.copy(), 'pp_trafo_loading_pct': trafo_loading_pct.copy(), 'pp_root_p_kw': float(np.asarray(info.get('trafo_p_signed_kw', np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)[0])}
                diagnostic_rows.append(diagnostic_row)
                for agent_idx, profile in enumerate(comparison_cfg.data.agent_profiles):
                    agent_rows.append({'controller': rollout_label, 'episode_idx': int(episode_idx), 'step': step_in_episode, 'timestamp': timestamp, 'agent_id': agent_idx, 'agent_profile': str(profile), 'load': float(np.asarray(info['load'], dtype=np.float32)[agent_idx]), 'load_pred': float(np.asarray(load_pred, dtype=np.float32)[agent_idx]), 'pv': float(np.asarray(info['pv'], dtype=np.float32)[agent_idx]), 'pv_raw': float(pv_raw[agent_idx]), 'pv_effective': float(pv_effective[agent_idx]), 'pv_curtail': float(pv_curtail[agent_idx]), 'pv_curtail_req': float(pv_curtail_req[agent_idx]), 'pv_utilization': float(pv_utilization[agent_idx]), 'pv_pred': float(np.asarray(pv_pred, dtype=np.float32)[agent_idx]), 'base_net_load': float(base_net_load[agent_idx]), 'base_net_load_effective': float(base_net_load_effective[agent_idx]), 'net_load': float(net_load[agent_idx]), 'grid_import_kw': float(grid_import[agent_idx]), 'grid_export_kw': float(grid_export[agent_idx]), 'e_bat': float(battery_power[agent_idx]), 'e_bat_req': float(battery_power_req[agent_idx]), 'battery_action_req': float(battery_action_req[agent_idx]), 'battery_action_exec': float(battery_action_exec[agent_idx]), 'pv_action_req': float(pv_action_req[agent_idx]), 'pv_action_exec': float(pv_action_exec[agent_idx]), 'controller_action_gap': float(controller_action_gap[agent_idx]), 'battery_request_gap_kw': float(battery_request_gap_kw[agent_idx]), 'pv_curtail_request_gap_kw': float(pv_curtail_request_gap_kw[agent_idx]), 'soc_penalty_unweighted': float(soc_penalty_unweighted[agent_idx]), 'r_soc_pen': float(soc_penalty[agent_idx]), 'r_safe_v': float(voltage_penalty_per_agent[agent_idx]), 'r_safe_line': float(line_penalty_per_agent[agent_idx]), 'r_safe_trafo': float(trafo_penalty_per_agent[agent_idx]), 'soc': float(np.asarray(info['soc_next'], dtype=np.float32)[agent_idx]), 'purchase_cost': float(purchase_cost_per_agent[agent_idx]), 'export_subsidy': float(export_subsidy_per_agent[agent_idx]), 'objective_total': float(objective_per_agent[agent_idx])})
                vm_pu = np.asarray(info.get('vm_pu', np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32)
                if vm_pu.shape[0] == len(bus_ids):
                    for bus_id, vm_value in zip(bus_ids, vm_pu, strict=False):
                        grid_rows.append({'controller': rollout_label, 'episode_idx': int(episode_idx), 'step': step_in_episode, 'timestamp': timestamp, 'bus_id': int(bus_id), 'vm_pu': float(vm_value), 'is_agent_bus': bool(int(bus_id) in agent_bus_set)})
                previous_raw_obs = raw_obs
                obs = next_obs
                raw_obs = raw_next_obs
            carried_soc = np.asarray(env.soc, dtype=np.float32).copy()
        step_df = pd.DataFrame(step_rows).sort_values(['episode_idx', 'step']).reset_index(drop=True)
        agent_df = pd.DataFrame(agent_rows).sort_values(['episode_idx', 'step', 'agent_id']).reset_index(drop=True)
        grid_df = pd.DataFrame(grid_rows).sort_values(['episode_idx', 'step', 'bus_id']).reset_index(drop=True)
        summary = agent_df.groupby(['controller', 'agent_profile'], as_index=False)[['purchase_cost', 'export_subsidy', 'objective_total']].sum() if not agent_df.empty else pd.DataFrame(columns=['controller', 'agent_profile', 'purchase_cost', 'export_subsidy', 'objective_total'])
        rollout = RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=summary, meta={'controller': rollout_label, 'n_agents': int(env.n), 'agent_profiles': list(comparison_cfg.data.agent_profiles), 'agent_bus_ids': agent_bus_ids, 'bus_ids': bus_ids, 'v_min_pu': float(comparison_cfg.grid.v_min_pu), 'v_max_pu': float(comparison_cfg.grid.v_max_pu), 'future_horizon': int(comparison_cfg.env.future_horizon), 'prediction_mode': PERFECT_PREDICTION_MODE, 'evaluation_mode': resolve_evaluation_mode(PERFECT_PREDICTION_MODE), 'dt_hours': float(comparison_cfg.env.dt), 'trafo_limit_kw': trafo_limit_kw, 'trafo_limit_note': 'Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.', 'loading_limit_pct': loading_limit_pct, 'trafo_loading_limit_pct': loading_limit_pct, 'export_subsidy_eur_per_kwh': export_subsidy, IMPORT_PRICE_MARKUP_KEY: float(getattr(getattr(comparison_cfg, 'reward', None), IMPORT_PRICE_MARKUP_KEY, 0.0)), 'controller_diagnostic_log': diagnostic_rows, 'soc_mode': 'continuous', 'solve_mode': str(solve_mode), 'global_oracle_time_limit_sec': resolved_time_limit, 'global_oracle_runtime_sec': float(result.solve_time_sec), 'global_oracle_gap': float(result.mip_gap), 'global_misocp_gap': float(result.mip_gap), 'global_oracle_status_label': str(result.status_label), 'global_oracle_debug_artifacts': dict(result.debug_artifacts), 'is_near_optimal': bool(str(solve_mode) == 'chunked_window'), 'economics_scope': 'agent_only', 'agent_purchase_cost_eur': float(result.agent_purchase_cost_eur), 'agent_export_subsidy_eur': float(result.agent_export_subsidy_eur), 'agent_net_cost_eur': float(result.agent_net_cost_eur), 'feeder_purchase_cost_eur': float(result.feeder_purchase_cost_eur), 'feeder_export_subsidy_eur': float(result.feeder_export_subsidy_eur), 'feeder_net_cost_eur': float(result.feeder_net_cost_eur), 'episode_offsets': np.asarray(full_input.episode_offsets, dtype=np.int32).copy(), 'episode_lengths': np.asarray(full_input.episode_lengths, dtype=np.int32).copy()})
        from controllers.mpc.global_socp_mpc import _LINE_LOADING_ERR_TOL_PCT, _ROOT_POWER_ERR_TOL_KW, _TRAFO_LOADING_ERR_TOL_PCT, _VOLTAGE_ERR_TOL_PU
        validation_rows: list[dict[str, object]] = []
        for entry in diagnostic_rows:
            misocp_vm = entry.get('misocp_vm_pu')
            misocp_line = entry.get('misocp_line_loading_pct')
            misocp_trafo = entry.get('misocp_trafo_loading_pct')
            if misocp_vm is None or misocp_line is None or misocp_trafo is None:
                max_vm_abs_err = np.nan
                max_line_abs_err = np.nan
                trafo_abs_err = np.nan
                root_p_abs_err = np.nan
                within_tolerance = False
            else:
                max_vm_abs_err = float(np.max(np.abs(np.asarray(misocp_vm, dtype=np.float32) - np.asarray(entry.get('pp_vm_pu'), dtype=np.float32))))
                max_line_abs_err = float(np.max(np.abs(np.asarray(misocp_line, dtype=np.float32) - np.asarray(entry.get('pp_line_loading_pct'), dtype=np.float32))))
                trafo_abs_err = float(np.max(np.abs(np.asarray(misocp_trafo, dtype=np.float32) - np.asarray(entry.get('pp_trafo_loading_pct'), dtype=np.float32))))
                root_p_abs_err = float(abs(float(entry.get('root_p_kw', np.nan)) - float(entry.get('pp_root_p_kw', np.nan))))
                within_tolerance = bool(max_vm_abs_err < _VOLTAGE_ERR_TOL_PU and max_line_abs_err < _LINE_LOADING_ERR_TOL_PCT and (trafo_abs_err < _TRAFO_LOADING_ERR_TOL_PCT) and (root_p_abs_err < _ROOT_POWER_ERR_TOL_KW))
            validation_rows.append({'controller': entry.get('controller', 'Global SOCP-MPC'), 'episode_idx': int(entry.get('episode_idx', 0)), 'step': int(entry.get('step', 0)), 'timestamp': entry.get('timestamp'), 'misocp_fallback': float(entry.get('misocp_fallback', 0.0)), 'misocp_time_limit_feasible': float(entry.get('misocp_time_limit_feasible', 0.0)), 'solve_time_sec': float(entry.get('solve_time_sec', np.nan)), 'max_vm_abs_err_pu': max_vm_abs_err, 'max_line_loading_abs_err_pct': max_line_abs_err, 'trafo_loading_abs_err_pct': trafo_abs_err, 'root_p_abs_err_kw': root_p_abs_err, 'within_tolerance': within_tolerance})
        validation_df = pd.DataFrame(validation_rows)
        fallback_ratio = float(rollout.step_df['misocp_fallback'].mean()) if not rollout.step_df.empty and 'misocp_fallback' in rollout.step_df.columns else 0.0
        simultaneous_step_ratio = float(rollout.step_df['simultaneous_step_flag'].mean()) if not rollout.step_df.empty and 'simultaneous_step_flag' in rollout.step_df.columns else 0.0
        health_warning = ''
        if fallback_ratio > 0.05:
            health_warning = 'Global SOCP-MPC fallback ratio exceeds 5%; hard-constrained steps were often infeasible or timed out without an incumbent.'
        if simultaneous_step_ratio > 0.05:
            suffix = 'Simultaneous charge/discharge exceeded 5% of steps; check subsidy-driven arbitrage and the tiny throughput regularization weight.'
            health_warning = f'{health_warning} {suffix}'.strip()
        if not validation_df.empty and bool((~validation_df['within_tolerance']).any()):
            health_warning = f'{health_warning} MISOCP-vs-pandapower validation exceeded at least one tolerance.'.strip()
        rollout.meta['misocp_validation_df'] = validation_df
        rollout.meta['misocp_fallback_ratio'] = fallback_ratio
        rollout.meta['misocp_simultaneous_step_ratio'] = simultaneous_step_ratio
        rollout.meta['misocp_health_warning'] = health_warning
        return rollout
    finally:
        env.close()
