from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path


def iso_timestamp(value: datetime) -> str:
    return value.astimezone().isoformat(timespec='seconds')


def estimate_remaining_seconds(*, interaction_step: int, target_interactions: int, elapsed_seconds: float) -> float | None:
    remaining_interactions = max(int(target_interactions) - int(interaction_step), 0)
    if remaining_interactions == 0:
        return 0.0
    if interaction_step <= 0 or elapsed_seconds <= 0.0:
        return None
    return float(remaining_interactions * (elapsed_seconds / float(interaction_step)))


def build_progress_payload(*, interaction_step: int, target_interactions: int, episodes_completed: int, total_steps: int, avg_reward: float, action_time_total: float, env_step_time_total: float, update_time_total: float, update_calls: int, run_start: float, started_at: datetime, status: str, error_message: str='') -> dict[str, object]:
    now = datetime.now().astimezone()
    elapsed = max(time.perf_counter() - run_start, 0.0)
    safe_elapsed = max(elapsed, 1e-06)
    remaining_seconds = estimate_remaining_seconds(interaction_step=interaction_step, target_interactions=target_interactions, elapsed_seconds=elapsed)
    if status != 'running' and int(interaction_step) >= int(target_interactions):
        remaining_seconds = 0.0
        estimated_end_time = now
    elif remaining_seconds is None:
        estimated_end_time = None
    else:
        estimated_end_time = now + timedelta(seconds=float(remaining_seconds))
    return {'status': str(status), 'error_message': str(error_message), 'interaction_step': int(interaction_step), 'target_interactions': int(target_interactions), 'episodes_completed': int(episodes_completed), 'total_steps': int(total_steps), 'avg_reward': float(avg_reward), 'steps_per_sec': float(total_steps / safe_elapsed), 'avg_action_ms_per_iter': float(1000.0 * action_time_total / max(interaction_step, 1)), 'avg_env_ms_per_iter': float(1000.0 * env_step_time_total / max(interaction_step, 1)), 'avg_update_ms_per_call': float(1000.0 * update_time_total / max(update_calls, 1)), 'started_at': iso_timestamp(started_at), 'updated_at': iso_timestamp(now), 'elapsed_seconds': float(round(elapsed, 3)), 'remaining_seconds': None if remaining_seconds is None else float(round(remaining_seconds, 3)), 'estimated_end_time': None if estimated_end_time is None else iso_timestamp(estimated_end_time)}


def write_progress_snapshot(path: str | os.PathLike[str], payload: dict[str, object]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


__all__ = ['build_progress_payload', 'estimate_remaining_seconds', 'iso_timestamp', 'write_progress_snapshot']
