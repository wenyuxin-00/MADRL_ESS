from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from scripts.checkpoints import build_training_run_paths, slugify_checkpoint_token
from scripts.utils.project_paths import get_checkpoint_root


def default_algorithm(experiment_controls: dict[str, Any], train_controls: dict[str, Any]) -> str:
    algorithm = experiment_controls.get('algorithm')
    if algorithm:
        return str(algorithm)
    return 'MATD3' if str(train_controls.get('profile', 'base')) == 'gpu_fast' else 'MADDPG'


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding='utf-8')
    return target_path


def resolve_save_dir(*, args, experiment_controls: dict[str, Any], data_controls: dict[str, Any], train_controls: dict[str, Any], checkpoint_controls: dict[str, Any], checkpoint_root_resolver: Callable[[Path], str | Path]) -> Path:
    if args.save_dir is not None:
        return Path(args.save_dir).resolve()
    checkpoint_root = Path(checkpoint_controls.get('checkpoint_root') or checkpoint_root_resolver(Path.cwd())).resolve()
    algorithm = default_algorithm(experiment_controls, train_controls)
    prediction_mode = str(data_controls.get('prediction_mode', 'perfect'))
    run_paths = build_training_run_paths(checkpoint_root, algorithm=algorithm, prediction_mode=prediction_mode, experiment_name=checkpoint_controls.get('experiment_name', 'grid_mainline'), train_episodes=train_controls.get('train_episodes'), max_train_steps=train_controls.get('max_train_steps'))
    return Path(run_paths['model_root']).resolve()


def public_vec_env_name(env: Any) -> str:
    return type(env).__name__


def load_train_mainline_result(result_json_path) -> dict[str, Any]:
    return json.loads(Path(result_json_path).read_text(encoding='utf-8'))


def load_progress_payload(progress_json_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(progress_json_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def format_progress_summary(payload: dict[str, Any]) -> str:
    interaction_step = int(payload.get('interaction_step', 0))
    target_interactions = max(int(payload.get('target_interactions', 0)), 1)
    episodes_completed = int(payload.get('episodes_completed', 0))
    avg_reward = float(payload.get('avg_reward', 0.0))
    steps_per_sec = float(payload.get('steps_per_sec', 0.0))
    percent = 100.0 * interaction_step / target_interactions
    status = str(payload.get('status', 'running'))
    estimated_end_time = payload.get('estimated_end_time')
    remaining_seconds = payload.get('remaining_seconds')
    if estimated_end_time:
        eta_segment = f' | ETA={estimated_end_time}' if remaining_seconds is None else f' | ETA={estimated_end_time} ({float(remaining_seconds):.1f}s)'
    else:
        eta_segment = ' | ETA=--'
    return f'[train:{status}] {interaction_step}/{target_interactions} iters ({percent:5.1f}%) | episodes={episodes_completed} | avg_reward={avg_reward:8.3f} | steps/s={steps_per_sec:7.1f}{eta_segment}'


def monitor_process_progress(process: subprocess.Popen[str], *, progress_json_path: Path, summary_interval_s: float, progress_episode_interval: int, load_progress: Callable[[Path], dict[str, Any] | None]=load_progress_payload, format_summary: Callable[[dict[str, Any]], str]=format_progress_summary) -> dict[str, Any] | None:
    last_mtime_ns = -1
    last_print_time = 0.0
    last_summary = ''
    last_payload: dict[str, Any] | None = None
    next_episode_report = max(int(progress_episode_interval), 1)
    while process.poll() is None:
        payload = load_progress(progress_json_path)
        if payload is not None and progress_json_path.exists():
            last_payload = payload
            stat = progress_json_path.stat()
            summary = format_summary(payload)
            now = time.monotonic()
            episodes_completed = int(payload.get('episodes_completed', 0))
            status = str(payload.get('status', 'running'))
            should_print = status != 'running' or (episodes_completed > 0 and episodes_completed >= next_episode_report)
            if should_print and stat.st_mtime_ns != last_mtime_ns and (now - last_print_time >= summary_interval_s):
                print(summary)
                last_summary = summary
                last_mtime_ns = stat.st_mtime_ns
                last_print_time = now
                while next_episode_report <= episodes_completed:
                    next_episode_report += max(int(progress_episode_interval), 1)
        time.sleep(0.5)
    payload = load_progress(progress_json_path)
    if payload is not None:
        last_payload = payload
        summary = format_summary(payload)
        if summary != last_summary:
            print(summary)
    return last_payload


def run_external_train_mainline(*, project_root, experiment_controls: dict[str, Any], data_controls: dict[str, Any], train_controls: dict[str, Any], battery_controls: dict[str, Any] | None=None, checkpoint_controls: dict[str, Any] | None=None, data_dir=None, save_dir=None, env_name: str='GridTrainMainline', run_number: int=1, python_executable: str | None=None, stream_output: bool=True, summary_interval_s: float=2.0, write_json_fn: Callable[[str | Path, dict[str, Any]], Path]=write_json, load_result_fn: Callable[[str | Path], dict[str, Any]]=load_train_mainline_result, monitor_progress_fn: Callable[..., dict[str, Any] | None]=monitor_process_progress, checkpoint_root_resolver: Callable[[Path], str | Path]=get_checkpoint_root, mainline_module: str='scripts.mainline_madrl') -> dict[str, Any]:
    root = Path(project_root).resolve()
    resolved_data_dir = Path(data_dir).resolve() if data_dir is not None else (root / 'data').resolve()
    checkpoint_controls = dict(checkpoint_controls or {})
    algorithm = default_algorithm(experiment_controls, train_controls)
    prediction_mode = str(data_controls.get('prediction_mode', 'perfect')).strip().lower()
    checkpoint_root = Path(checkpoint_controls.get('checkpoint_root') or checkpoint_root_resolver(root)).resolve()
    experiment_name = slugify_checkpoint_token(checkpoint_controls.get('experiment_name', 'grid_mainline'), default='grid_mainline')
    if save_dir is None:
        run_paths = build_training_run_paths(checkpoint_root, algorithm=algorithm, prediction_mode=prediction_mode, experiment_name=experiment_name, train_episodes=train_controls.get('train_episodes'), max_train_steps=train_controls.get('max_train_steps'))
        model_root = Path(run_paths['model_root']).resolve()
        meta_dir = Path(run_paths['meta_dir']).resolve()
        run_label = str(run_paths['run_label'])
        prediction_mode = str(run_paths['prediction_mode'])
        experiment_name = str(run_paths['experiment_name'])
    else:
        model_root = Path(save_dir).resolve()
        meta_dir = (model_root / '_meta').resolve()
        run_label = model_root.name
    meta_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {'experiment_name': experiment_name, 'checkpoint_root': str(checkpoint_root), 'prediction_mode': prediction_mode, 'run_label': run_label, 'model_root': str(model_root)}
    battery_controls = dict(battery_controls or {})
    launch_info = {'project_root': str(root), 'experiment_controls_path': str(write_json_fn(meta_dir / 'experiment_controls.json', experiment_controls)), 'data_controls_path': str(write_json_fn(meta_dir / 'data_controls.json', data_controls)), 'battery_controls_path': str(write_json_fn(meta_dir / 'battery_controls.json', battery_controls)), 'train_controls_path': str(write_json_fn(meta_dir / 'train_controls.json', train_controls)), 'checkpoint_controls_path': str(write_json_fn(meta_dir / 'checkpoint_controls.json', checkpoint_payload)), 'result_json_path': str(meta_dir / 'train_result.json'), 'progress_json_path': str(meta_dir / 'progress.json'), 'log_path': str(meta_dir / 'train.log'), 'data_dir': str(resolved_data_dir), 'save_dir': str(model_root), 'model_root': str(model_root), 'meta_dir': str(meta_dir), 'checkpoint_root': str(checkpoint_root), 'prediction_mode': str(prediction_mode), 'experiment_name': str(experiment_name), 'run_label': str(run_label), 'env_name': str(env_name), 'run_number': int(run_number)}
    command = [str(python_executable or sys.executable), '-m', mainline_module, '--experiment-controls', str(launch_info['experiment_controls_path']), '--data-controls', str(launch_info['data_controls_path']), '--battery-controls', str(launch_info['battery_controls_path']), '--train-controls', str(launch_info['train_controls_path']), '--checkpoint-controls', str(launch_info['checkpoint_controls_path']), '--data-dir', str(launch_info['data_dir']), '--save-dir', str(launch_info['save_dir']), '--result-json', str(launch_info['result_json_path']), '--env-name', str(launch_info['env_name']), '--run-number', str(launch_info['run_number'])]
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    log_path = Path(launch_info['log_path'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_progress = None
    with log_path.open('w', encoding='utf-8') as log_handle:
        process = subprocess.Popen(command, cwd=str(project_root), env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        if stream_output:
            print(f'Training log: {log_path}')
            last_progress = monitor_progress_fn(process, progress_json_path=Path(launch_info['progress_json_path']), summary_interval_s=float(summary_interval_s), progress_episode_interval=int(train_controls.get('progress_episode_interval', 10)))
        returncode = process.wait()
    if last_progress is None:
        last_progress = load_progress_payload(Path(launch_info['progress_json_path']))
    result = load_result_fn(launch_info['result_json_path']) if Path(launch_info['result_json_path']).exists() else None
    launch_metadata = {'pid': int(process.pid), 'returncode': int(returncode), 'command': command, 'launch_info': launch_info, 'result': result, 'last_progress': last_progress, 'stream_output': bool(stream_output)}
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return launch_metadata


__all__ = ['default_algorithm', 'format_progress_summary', 'json_default', 'load_json', 'load_progress_payload', 'load_train_mainline_result', 'monitor_process_progress', 'public_vec_env_name', 'resolve_save_dir', 'run_external_train_mainline', 'write_json']
