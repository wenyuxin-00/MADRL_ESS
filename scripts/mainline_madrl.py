from __future__ import annotations
import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any
for env_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(env_var, '1')
warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*', category=FutureWarning)
import torch
from configs.profiles import compose_experiment_config, print_experiment_summary
from controllers.madrl.safety_projector import is_safe_poc_algorithm
from scripts.builder import build_train_runner
from scripts.checkpoints import build_checkpoint_manifest, build_training_run_paths, resolve_checkpoint_to_load, slugify_checkpoint_token
from scripts.utils.mainline_setup import apply_mainline_experiment_settings, ensure_mainline_forecast_ready
from scripts.utils.project_paths import get_checkpoint_root, get_tensorboard_run_dir, project_root
from scripts.utils.train_mainline_support import default_algorithm as _default_algorithm, format_progress_summary as _format_progress_summary, json_default as _json_default, load_json as _load_json, load_progress_payload as _load_progress_payload, load_train_mainline_result, monitor_process_progress, public_vec_env_name as _public_vec_env_name, resolve_save_dir as _resolve_save_dir, run_external_train_mainline as _run_external_train_mainline, write_json as _write_json
from scripts.utils.torch_runtime import configure_torch_runtime, describe_device

def _apply_train_controls(cfg, train_controls: dict[str, Any]) -> None:
    field_names = ('train_episodes', 'max_train_steps', 'num_envs', 'vec_env_type', 'parallel_episode_sampling', 'batch_size', 'buffer_size', 'update_interval', 'updates_per_step', 'actor_lr', 'critic_lr', 'noise_std_init', 'noise_std_min', 'noise_decay_steps', 'use_noise_decay', 'show_progress', 'progress_postfix_interval', 'progress_episode_interval', 'progress_write_interval_seconds')
    for field_name in field_names:
        if field_name in train_controls:
            setattr(cfg.train, field_name, train_controls[field_name])
    if 'policy_update_freq' in train_controls:
        cfg.algo.policy_update_freq = int(train_controls['policy_update_freq'])

def _apply_model_controls(cfg, model_controls: dict[str, Any] | None) -> None:
    controls = dict(model_controls or {})
    if 'hidden_dim' not in controls:
        return
    hidden_dim = int(controls['hidden_dim'])
    if hidden_dim <= 0:
        raise ValueError(f'model_controls.hidden_dim must be positive, got {hidden_dim}.')
    cfg.model.hidden_dim = hidden_dim

def _apply_runtime_controls(cfg, runtime_controls: dict[str, Any] | None) -> None:
    controls = dict(runtime_controls or {})
    bool_fields = ('pin_memory', 'non_blocking_transfers', 'enable_amp', 'enable_compile', 'compile_fullgraph', 'compile_dynamic')
    str_fields = ('amp_dtype', 'compile_mode', 'matmul_precision')
    deprecated_fields = ('forecast_data_source', 'observation_cache_root', 'observation_cache_batch_size', 'refresh_observation_cache')
    deprecated_hits = [field_name for field_name in deprecated_fields if field_name in controls]
    if deprecated_hits:
        joined = ', '.join(sorted(deprecated_hits))
        raise ValueError(f'runtime_controls no longer supports legacy cache fields: {joined}. Use shared_data_dir/shared_data_signature for precomputed data, or omit them to use the live forecaster path.')
    for field_name in bool_fields:
        if field_name in controls:
            setattr(cfg.runtime, field_name, bool(controls[field_name]))
    for field_name in str_fields:
        if field_name in controls:
            setattr(cfg.runtime, field_name, str(controls[field_name]))
    if 'shared_data_dir' in controls:
        cfg.runtime.shared_data_dir = None if controls['shared_data_dir'] in (None, '') else str(Path(controls['shared_data_dir']).resolve())
    if 'shared_data_signature' in controls:
        cfg.runtime.shared_data_signature = None if controls['shared_data_signature'] in (None, '') else str(controls['shared_data_signature'])

def _apply_reward_controls(cfg, reward_controls: dict[str, Any] | None) -> None:
    controls = dict(reward_controls or {})
    removed_reward_keys = {'lambda_throughput': "Remove 'lambda_throughput'; it is no longer supported.", 'w_action_pen': "Use 'w_soc_pen' instead of 'w_action_pen'."}
    removed_hits = sorted(set(controls).intersection(removed_reward_keys))
    if removed_hits:
        details = ' '.join((removed_reward_keys[key] for key in removed_hits))
        raise ValueError(f'Legacy reward_controls key(s) are no longer supported: {removed_hits}. {details}')
    known_reward_keys = {'export_subsidy_eur_per_kwh', 'import_price_markup_eur_per_kwh', 'w_soc_pen', 'w_voltage_pen', 'w_line_pen', 'w_trafo_pen'}
    unknown_keys = set(controls) - known_reward_keys
    if unknown_keys:
        raise ValueError(f'Unknown reward_controls key(s): {sorted(unknown_keys)}. Supported keys: {sorted(known_reward_keys)}.')
    if 'w_soc_pen' in controls:
        cfg.reward.w_soc_pen = float(controls['w_soc_pen'])
    for field_name in ('export_subsidy_eur_per_kwh', 'import_price_markup_eur_per_kwh', 'w_voltage_pen', 'w_line_pen', 'w_trafo_pen'):
        if field_name in controls:
            setattr(cfg.reward, field_name, float(controls[field_name]))

def _apply_safety_controls(cfg, safety_controls: dict[str, Any] | None) -> None:
    controls = dict(safety_controls or {})
    numeric_fields = ('projection_iters', 'voltage_margin_pu', 'line_margin_pct', 'trafo_margin_pct', 'linearization_delta_kw')
    for field_name in numeric_fields:
        if field_name in controls:
            value = controls[field_name]
            setattr(cfg.safety, field_name, int(value) if field_name == 'projection_iters' else float(value))
    if 'record_diagnostics' in controls:
        cfg.safety.record_diagnostics = bool(controls['record_diagnostics'])
    if 'projector_mode' in controls:
        cfg.safety.projector_mode = str(controls['projector_mode'])
    cfg.safety.enabled = bool(is_safe_poc_algorithm(cfg) and controls.get('enabled', True))

def _serialize_safety_cfg(cfg) -> dict[str, Any]:
    return {'enabled': bool(getattr(cfg.safety, 'enabled', False)), 'projector_mode': str(getattr(cfg.safety, 'projector_mode', 'joint_linearized')), 'projection_iters': int(getattr(cfg.safety, 'projection_iters', 6)), 'voltage_margin_pu': float(getattr(cfg.safety, 'voltage_margin_pu', 0.005)), 'line_margin_pct': float(getattr(cfg.safety, 'line_margin_pct', 5.0)), 'trafo_margin_pct': float(getattr(cfg.safety, 'trafo_margin_pct', 5.0)), 'linearization_delta_kw': float(getattr(cfg.safety, 'linearization_delta_kw', 0.25)), 'record_diagnostics': bool(getattr(cfg.safety, 'record_diagnostics', True))}

def _build_result_payload(*, cfg, experiment_controls: dict[str, Any], data_controls: dict[str, Any], battery_controls: dict[str, Any], train_controls: dict[str, Any], checkpoint_controls: dict[str, Any], runtime_state, forecast_ready, summary: dict[str, Any], runner, episodes_completed: int, save_dir: Path, meta_dir: Path, log_path: Path, reward_summary_path: Path, env_name: str, run_number: int, seed: int, applied_controls: dict[str, Any]) -> dict[str, Any]:
    experiment_name = slugify_checkpoint_token(checkpoint_controls.get('experiment_name', save_dir.parent.name), default='grid_mainline')
    run_metadata = dict(getattr(runner, 'run_metadata', {}))
    return {'algorithm': str(cfg.algo.name), 'prediction_mode': str(applied_controls['prediction_mode']), 'evaluation_mode': str(applied_controls['evaluation_mode']), 'experiment_name': experiment_name, 'run_label': str(save_dir.name), 'episodes_completed': int(episodes_completed), 'saved_episode_tag': int(episodes_completed), 'save_dir': str(save_dir), 'model_root': str(save_dir), 'meta_dir': str(meta_dir), 'log_path': str(log_path), 'reward_summary_path': str(reward_summary_path), 'checkpoint_info': resolve_checkpoint_to_load(save_dir, cfg.algo.name, episode_tag=episodes_completed), 'tensorboard_dir': str(get_tensorboard_run_dir(cfg.algo.name, env_name, run_number=run_number, seed=seed, root=project_root())), 'device': str(cfg.runtime.device), 'vec_env': _public_vec_env_name(runner.env), 'perf_summary': dict(runner.perf_summary), 'shared_data_dir': getattr(cfg.runtime, 'shared_data_dir', None), 'shared_data_signature': getattr(cfg.runtime, 'shared_data_signature', None), 'started_at': run_metadata.get('started_at'), 'finished_at': run_metadata.get('finished_at'), 'elapsed_seconds': run_metadata.get('elapsed_seconds'), 'estimated_end_time': run_metadata.get('estimated_end_time'), 'summary': summary, 'safety_summary': runner.build_safety_summary(), 'device_info': describe_device(runtime_state), 'experiment_controls': dict(experiment_controls), 'data_controls': dict(data_controls), 'battery_controls': dict(battery_controls), 'train_controls': dict(train_controls), 'checkpoint_controls': dict(checkpoint_controls), 'safety_controls': _serialize_safety_cfg(cfg), 'forecast_ready': forecast_ready, 'pid': int(os.getpid())}

def _monitor_process_progress(process: subprocess.Popen[str], *, progress_json_path: Path, summary_interval_s: float, progress_episode_interval: int) -> dict[str, Any] | None:
    return monitor_process_progress(process, progress_json_path=progress_json_path, summary_interval_s=summary_interval_s, progress_episode_interval=progress_episode_interval, load_progress=_load_progress_payload, format_summary=_format_progress_summary)

def run_external_train_mainline(*, project_root, experiment_controls: dict[str, Any], data_controls: dict[str, Any], train_controls: dict[str, Any], battery_controls: dict[str, Any] | None=None, checkpoint_controls: dict[str, Any] | None=None, data_dir=None, save_dir=None, env_name: str='GridTrainMainline', run_number: int=1, python_executable: str | None=None, stream_output: bool=True, summary_interval_s: float=2.0) -> dict[str, Any]:
    return _run_external_train_mainline(project_root=project_root, experiment_controls=experiment_controls, data_controls=data_controls, train_controls=train_controls, battery_controls=battery_controls, checkpoint_controls=checkpoint_controls, data_dir=data_dir, save_dir=save_dir, env_name=env_name, run_number=run_number, python_executable=python_executable, stream_output=stream_output, summary_interval_s=summary_interval_s, write_json_fn=_write_json, load_result_fn=load_train_mainline_result, monitor_progress_fn=_monitor_process_progress, checkpoint_root_resolver=get_checkpoint_root, mainline_module='scripts.mainline_madrl')

def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the Grid MADRL training mainline.')
    parser.add_argument('--experiment-controls', required=True)
    parser.add_argument('--data-controls', required=True)
    parser.add_argument('--battery-controls')
    parser.add_argument('--train-controls', required=True)
    parser.add_argument('--checkpoint-controls')
    parser.add_argument('--data-dir')
    parser.add_argument('--save-dir')
    parser.add_argument('--result-json')
    parser.add_argument('--env-name', default='GridTrainMainline')
    parser.add_argument('--run-number', type=int, default=1)
    return parser.parse_args(argv)

def main(argv: list[str] | None=None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except (AttributeError, RuntimeError):
        pass
    experiment_controls = _load_json(args.experiment_controls)
    data_controls = _load_json(args.data_controls)
    battery_controls = _load_json(args.battery_controls) if args.battery_controls else {}
    train_controls = _load_json(args.train_controls)
    checkpoint_controls = _load_json(args.checkpoint_controls) if args.checkpoint_controls else {}
    seed = int(experiment_controls.get('seed', 0))
    data_dir = Path(args.data_dir).resolve() if args.data_dir is not None else project_root() / 'data'
    save_dir = _resolve_save_dir(args=args, experiment_controls=experiment_controls, data_controls=data_controls, train_controls=train_controls, checkpoint_controls=checkpoint_controls, checkpoint_root_resolver=lambda _root: get_checkpoint_root(project_root()))
    meta_dir = (save_dir / '_meta').resolve()
    result_json = Path(args.result_json).resolve() if args.result_json is not None else meta_dir / 'train_result.json'
    log_path = meta_dir / 'train.log'
    progress_json = meta_dir / 'progress.json'
    reward_summary_path = meta_dir / 'train_reward_summary.json'
    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg = compose_experiment_config(profile=str(train_controls.get('profile', 'base')), algorithm=experiment_controls.get('algorithm'), model_family=str(train_controls.get('model_family', 'mlp')), vec_env_type=train_controls.get('vec_env_type'), data_dir=data_dir, device=experiment_controls.get('device_request'), runtime_mode=str(experiment_controls.get('runtime_mode', 'performance')), seed=seed, require_cuda=experiment_controls.get('require_cuda'))
    _apply_model_controls(cfg, experiment_controls.get('model_controls'))
    _apply_runtime_controls(cfg, experiment_controls.get('runtime_controls'))
    _apply_reward_controls(cfg, experiment_controls.get('reward_controls'))
    _apply_safety_controls(cfg, experiment_controls.get('safety_controls'))
    agent_profiles = list(data_controls.get('agent_profiles', cfg.data.agent_profiles))
    n_requested_agents = len(agent_profiles)
    applied_controls = apply_mainline_experiment_settings(cfg, prediction_mode=str(data_controls.get('prediction_mode', 'perfect')), test_start_date=data_controls.get('test_start_date'), test_end_date=data_controls.get('test_end_date'), agent_profiles=agent_profiles, agent_bus_ids=data_controls.get('agent_bus_ids'), load_scale=data_controls.get('load_scale', cfg.data.load_scale or [1.0] * n_requested_agents), pv_scale=data_controls.get('pv_scale', cfg.data.pv_scale or [1.0] * n_requested_agents), battery_controls=battery_controls, forecast_controls=experiment_controls.get('forecast_controls'), future_horizon=int(data_controls.get('future_horizon', cfg.env.future_horizon)), train_year=data_controls.get('train_year'), test_year=data_controls.get('test_year'))
    _apply_train_controls(cfg, train_controls)
    cfg.train.noise_decay_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.runtime.progress_state_path = str(progress_json)
    runtime_state = configure_torch_runtime(cfg, device=experiment_controls.get('device_request'), seed=seed, require_cuda=experiment_controls.get('require_cuda'))
    forecast_ready = None if getattr(cfg.runtime, 'shared_data_dir', None) else ensure_mainline_forecast_ready(cfg)
    cfg.runtime.forecast_ready = forecast_ready
    summary = print_experiment_summary(cfg)
    summary.update({'applied_controls': applied_controls, 'device_info': describe_device(runtime_state), 'battery_controls': dict(battery_controls), 'checkpoint_controls': dict(checkpoint_controls), 'model_root': str(save_dir), 'meta_dir': str(meta_dir), 'log_path': str(log_path), 'run_label': str(save_dir.name), 'training_backend': 'mainline', 'shared_data_dir': getattr(cfg.runtime, 'shared_data_dir', None), 'shared_data_signature': getattr(cfg.runtime, 'shared_data_signature', None), 'safety': _serialize_safety_cfg(cfg)})
    runner = build_train_runner(cfg, seed=seed, env_name=args.env_name, number=args.run_number)
    summary['shared_data_metadata'] = dict(getattr(runner, 'shared_data_metadata', {}) or {})
    episodes_completed = 0
    try:
        episodes_completed = runner.run()
        save_dir.mkdir(parents=True, exist_ok=True)
        runner.save_model(str(save_dir), episode=episodes_completed)
        _write_json(reward_summary_path, runner.build_reward_summary())
        result_payload = _build_result_payload(cfg=cfg, experiment_controls=experiment_controls, data_controls=data_controls, battery_controls=battery_controls, train_controls=train_controls, checkpoint_controls=checkpoint_controls, runtime_state=runtime_state, forecast_ready=forecast_ready, summary=summary, runner=runner, episodes_completed=episodes_completed, save_dir=save_dir, meta_dir=meta_dir, log_path=log_path, reward_summary_path=reward_summary_path, env_name=args.env_name, run_number=int(args.run_number), seed=seed, applied_controls=applied_controls)
        result_payload['perf_summary'].update({'shared_data_enabled': bool(getattr(cfg.runtime, 'shared_data_dir', None)), 'shared_data_dir': getattr(cfg.runtime, 'shared_data_dir', None), 'shared_data_signature': getattr(cfg.runtime, 'shared_data_signature', None)})
        _write_json(result_json, result_payload)
        print(json.dumps(result_payload, indent=2, default=_json_default))
        return 0
    finally:
        runner.close()
__all__ = ['_apply_reward_controls', '_apply_runtime_controls', '_apply_train_controls', '_format_progress_summary', '_load_progress_payload', '_monitor_process_progress', 'load_train_mainline_result', 'main', 'parse_args', 'run_external_train_mainline']
if __name__ == '__main__':
    raise SystemExit(main())
