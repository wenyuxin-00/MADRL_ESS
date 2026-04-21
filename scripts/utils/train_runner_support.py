from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from controllers.action_feasibility import action_info_to_numpy, compute_action_gap_metrics_torch, enforce_local_action_feasibility_torch, merge_action_info_into_step_info
from controllers.madrl.safety_projector import SAFE_POC_ALGO_NAME
from controllers.madrl_controller import _override_soc_penalty_metrics
from scripts.checkpoints import build_checkpoint_manifest, write_checkpoint_manifest
from scripts.utils.nested import to_torch_nested

_PROJECTION_RESIDUAL_TOL = 1e-06


def init_safety_tracking(runner: Any) -> None:
    runner._safety_projection_batches = 0
    runner._safety_projection_samples = 0
    runner._safety_diagnostic_samples = 0
    runner._safety_projection_calls_by_stage = {'rollout': 0, 'target': 0, 'actor': 0}
    runner._safety_projection_samples_by_stage = {'rollout': 0, 'target': 0, 'actor': 0}
    runner._safety_projection_time_s_by_stage = {'rollout': 0.0, 'target': 0.0, 'actor': 0.0}
    runner._safety_projected_fraction_weighted = 0.0
    runner._safety_mean_abs_delta_weighted = 0.0
    runner._safety_mean_abs_delta_kw_weighted = 0.0
    runner._safety_pre_violation_weighted = 0.0
    runner._safety_post_violation_weighted = 0.0
    runner._safety_pre_trafo_import_violation_kw_weighted = 0.0
    runner._safety_pre_trafo_export_violation_kw_weighted = 0.0
    runner._safety_post_trafo_import_violation_kw_weighted = 0.0
    runner._safety_post_trafo_export_violation_kw_weighted = 0.0
    runner._safety_max_abs_delta = 0.0
    runner._safety_env_fallback_steps = 0
    runner._safety_env_observations = 0
    runner._safety_env_action_pen_total = 0.0
    runner._projector_local_infeasible_count = 0


def _safe_projection_enabled(runner: Any) -> bool:
    algo_cfg = getattr(runner.cfg, 'algo', None)
    algo_name = getattr(algo_cfg, 'name', '')
    return str(algo_name) == SAFE_POC_ALGO_NAME and getattr(runner, 'safety_projector', None) is not None


def record_projection_diagnostics(runner: Any, diagnostics: dict[str, Any]) -> None:
    batch_size = max(int(diagnostics.get('batch_size', 0)), 1)
    runner._safety_diagnostic_samples += batch_size
    runner._safety_projected_fraction_weighted += float(diagnostics.get('projected_fraction', 0.0)) * batch_size
    runner._safety_mean_abs_delta_weighted += float(diagnostics.get('mean_abs_delta', 0.0)) * batch_size
    runner._safety_mean_abs_delta_kw_weighted += float(diagnostics.get('mean_abs_delta_kw', 0.0)) * batch_size
    runner._safety_pre_violation_weighted += float(diagnostics.get('pre_violation', 0.0)) * batch_size
    runner._safety_post_violation_weighted += float(diagnostics.get('post_violation', 0.0)) * batch_size
    runner._safety_pre_trafo_import_violation_kw_weighted += float(diagnostics.get('pre_trafo_import_violation_kw', 0.0)) * batch_size
    runner._safety_pre_trafo_export_violation_kw_weighted += float(diagnostics.get('pre_trafo_export_violation_kw', 0.0)) * batch_size
    runner._safety_post_trafo_import_violation_kw_weighted += float(diagnostics.get('post_trafo_import_violation_kw', 0.0)) * batch_size
    runner._safety_post_trafo_export_violation_kw_weighted += float(diagnostics.get('post_trafo_export_violation_kw', 0.0)) * batch_size
    runner._safety_max_abs_delta = max(runner._safety_max_abs_delta, float(diagnostics.get('max_abs_delta', 0.0)))


def record_projection_event(runner: Any, *, stage: str, batch_size: int, elapsed_s: float, diagnostics: dict[str, Any] | None=None) -> None:
    normalized_stage = str(stage) if str(stage) in runner._safety_projection_calls_by_stage else 'rollout'
    safe_batch_size = max(int(batch_size), 1)
    runner._safety_projection_batches += 1
    runner._safety_projection_samples += safe_batch_size
    runner._safety_projection_calls_by_stage[normalized_stage] += 1
    runner._safety_projection_samples_by_stage[normalized_stage] += safe_batch_size
    runner._safety_projection_time_s_by_stage[normalized_stage] += float(elapsed_s)
    if diagnostics is not None:
        record_projection_diagnostics(runner, diagnostics)


def record_projector_local_infeasible(runner: Any, residual_action_info: dict[str, torch.Tensor] | None) -> None:
    if residual_action_info is None:
        return
    residual = residual_action_info.get('soc_penalty_unweighted')
    if residual is None:
        return
    residual_tensor = torch.as_tensor(residual, dtype=torch.float32)
    if residual_tensor.ndim == 1:
        residual_tensor = residual_tensor.unsqueeze(0)
    affected = torch.any(residual_tensor > _PROJECTION_RESIDUAL_TOL, dim=-1)
    runner._projector_local_infeasible_count += int(torch.count_nonzero(affected).item())


def build_safety_summary(runner: Any) -> dict[str, Any]:
    if not _safe_projection_enabled(runner):
        return {'enabled': False, 'algorithm': str(runner.cfg.algo.name)}
    diagnostic_denominator = max(runner._safety_diagnostic_samples, 1)
    env_denominator = max(runner._safety_env_observations, 1)
    projection_time_total = float(sum(runner._safety_projection_time_s_by_stage.values()))
    return {'enabled': True, 'algorithm': str(runner.cfg.algo.name), 'projector_mode': str(getattr(runner.cfg.safety, 'projector_mode', 'joint_linearized')), 'projection_batches': int(runner._safety_projection_batches), 'projection_samples': int(runner._safety_projection_samples), 'projection_diagnostic_samples': int(runner._safety_diagnostic_samples), 'projected_fraction': float(runner._safety_projected_fraction_weighted / diagnostic_denominator), 'mean_abs_action_delta': float(runner._safety_mean_abs_delta_weighted / diagnostic_denominator), 'max_abs_action_delta': float(runner._safety_max_abs_delta), 'mean_abs_action_delta_kw': float(runner._safety_mean_abs_delta_kw_weighted / diagnostic_denominator), 'mean_pre_projection_violation': float(runner._safety_pre_violation_weighted / diagnostic_denominator), 'mean_post_projection_violation': float(runner._safety_post_violation_weighted / diagnostic_denominator), 'mean_pre_trafo_import_violation_kw': float(runner._safety_pre_trafo_import_violation_kw_weighted / diagnostic_denominator), 'mean_pre_trafo_export_violation_kw': float(runner._safety_pre_trafo_export_violation_kw_weighted / diagnostic_denominator), 'mean_post_trafo_import_violation_kw': float(runner._safety_post_trafo_import_violation_kw_weighted / diagnostic_denominator), 'mean_post_trafo_export_violation_kw': float(runner._safety_post_trafo_export_violation_kw_weighted / diagnostic_denominator), 'projection_time_s': projection_time_total, 'rollout_projection_calls': int(runner._safety_projection_calls_by_stage['rollout']), 'target_projection_calls': int(runner._safety_projection_calls_by_stage['target']), 'actor_projection_calls': int(runner._safety_projection_calls_by_stage['actor']), 'rollout_projection_samples': int(runner._safety_projection_samples_by_stage['rollout']), 'target_projection_samples': int(runner._safety_projection_samples_by_stage['target']), 'actor_projection_samples': int(runner._safety_projection_samples_by_stage['actor']), 'rollout_projection_time_s': float(runner._safety_projection_time_s_by_stage['rollout']), 'target_projection_time_s': float(runner._safety_projection_time_s_by_stage['target']), 'actor_projection_time_s': float(runner._safety_projection_time_s_by_stage['actor']), 'env_fallback_action_pen_steps': int(runner._safety_env_fallback_steps), 'env_fallback_action_pen_rate': float(runner._safety_env_fallback_steps / env_denominator), 'env_fallback_action_pen_mean': float(runner._safety_env_action_pen_total / env_denominator), 'projector_local_infeasible_count': int(runner._projector_local_infeasible_count)}


def select_action_batch_with_info(runner: Any, obs_np: dict) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    obs_t = to_torch_nested(obs_np, runner.cfg.runtime.device)
    with torch.inference_mode():
        raw_action_t = torch.stack([agent.act_from_torch_obs(obs_t, noise_std=runner.noise_std) for agent in runner.agent_n], dim=1)
        action_info = None
        if _safe_projection_enabled(runner):
            projection_started = time.perf_counter()
            projected_action_t, diagnostics = runner.safety_projector.project_actions_from_safety_local(obs_t['safety_local'], raw_action_t, return_diagnostics=True)
            record_projection_event(runner, stage='rollout', batch_size=int(raw_action_t.shape[0]), elapsed_s=time.perf_counter() - projection_started, diagnostics=diagnostics)
            action_t, projector_residual_info = enforce_local_action_feasibility_torch(obs_t['safety_local'], projected_action_t, efficiency=float(runner.cfg.env.efficiency), dt_hours=float(runner.cfg.env.dt), soc_min=float(runner.cfg.env.soc_min), soc_max=float(runner.cfg.env.soc_max))
            record_projector_local_infeasible(runner, projector_residual_info)
            action_info = compute_action_gap_metrics_torch(obs_t['safety_local'], raw_action_t, action_t)
            action_info = _override_soc_penalty_metrics(action_info, projector_residual_info)
        elif 'safety_local' in obs_t:
            action_t, action_info = enforce_local_action_feasibility_torch(obs_t['safety_local'], raw_action_t, efficiency=float(runner.cfg.env.efficiency), dt_hours=float(runner.cfg.env.dt), soc_min=float(runner.cfg.env.soc_min), soc_max=float(runner.cfg.env.soc_max))
        else:
            action_t = raw_action_t
    return (action_t.to(dtype=torch.float32).cpu().numpy(), action_info_to_numpy(action_info))


def apply_controller_action_postprocessing(runner: Any, reward: np.ndarray, info_list: list[dict[str, Any]], action_info: dict[str, np.ndarray] | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    reward_array = np.asarray(reward, dtype=np.float32)
    if reward_array.ndim == 2:
        reward_array = reward_array[..., None]
    processed_info_list: list[dict[str, Any]] = []
    for env_idx, info in enumerate(info_list):
        env_action_info = None if action_info is None else {key: np.asarray(value[env_idx], dtype=np.float32) for key, value in action_info.items()}
        merged_info, action_penalty = merge_action_info_into_step_info(info, env_action_info, soc_pen_weight=float(runner.cfg.reward.w_soc_pen), apply_action_penalty=runner.apply_action_penalty)
        reward_array[env_idx, :, 0] -= np.asarray(action_penalty, dtype=np.float32)
        merged_info['reward'] = np.asarray(reward_array[env_idx, :, 0], dtype=np.float32)
        processed_info_list.append(merged_info)
    return (reward_array.astype(np.float32), processed_info_list)


def build_shared_update_ctx(runner: Any, batch: dict[str, Any]) -> dict[str, Any]:
    next_obs = batch['next_obs']
    with torch.no_grad():
        target_actor_actions_clean = torch.stack([agent._actor_target_call(next_obs) for agent in runner.agent_n], dim=1)
    shared_ctx = {'target_actor_actions_clean': target_actor_actions_clean, 'batch_size': int(batch['action'].shape[0]), 'device': batch['action'].device}
    if _safe_projection_enabled(runner):
        shared_ctx['safety_projector'] = runner.safety_projector
        shared_ctx['projection_event_recorder'] = lambda **kwargs: record_projection_event(runner, **kwargs)
        with torch.no_grad():
            noise = (torch.randn_like(target_actor_actions_clean) * float(runner.cfg.algo.policy_noise)).clamp(-float(runner.cfg.algo.noise_clip), float(runner.cfg.algo.noise_clip))
            target_actor_actions = (target_actor_actions_clean + noise).clamp(-float(runner.cfg.model.max_action), float(runner.cfg.model.max_action))
            projection_started = time.perf_counter()
            shared_ctx['projected_target_actions'] = runner.safety_projector.project_actions_from_safety_local(next_obs['safety_local'], target_actor_actions)
            record_projection_event(runner, stage='target', batch_size=int(target_actor_actions.shape[0]), elapsed_s=time.perf_counter() - projection_started)
    return shared_ctx


def save_runner_model(runner: Any, model_dir: str, episode: int) -> None:
    algo_dir = os.path.join(model_dir, runner.cfg.algo.name)
    os.makedirs(algo_dir, exist_ok=True)
    for agent in runner.agent_n:
        agent.save_model(algo_dir, episode)
    manifest = build_checkpoint_manifest(algorithm=runner.cfg.algo.name, saved_episode_tag=episode, episodes_completed=runner.episodes_completed, total_steps=runner.total_steps, num_envs=runner.cfg.train.num_envs, episode_limit=runner.cfg.env.episode_limit, save_dir=algo_dir)
    write_checkpoint_manifest(algo_dir, manifest)


def close_runner(runner: Any) -> None:
    if runner._closed:
        return
    runner.env.close()
    runner.env_evaluate.close()
    runner.writer.close()
    runner._closed = True


def build_reward_summary(runner: Any) -> dict[str, Any]:
    episode_indices = list(range(1, len(runner.episode_rewards) + 1))
    components: dict[str, dict[str, Any]] = {}
    for meta in runner.reward_component_meta:
        key = str(meta.key)
        values = [float(value) for value in runner.episode_reward_components.get(key, [])]
        components[key] = {'label': str(getattr(meta, 'label', meta.key)), 'color': str(getattr(meta, 'color', '#111827')), 'sign': int(getattr(meta, 'sign', 0)), 'values': values}
    aggregates: dict[str, list[float]] = {}
    grid_safety_keys = [str(meta.key) for meta in runner.reward_component_meta if str(meta.key).startswith('r_safe_') and int(getattr(meta, 'sign', 0)) == -1]
    if grid_safety_keys and all((key in components for key in grid_safety_keys)):
        num_episodes = len(episode_indices)
        aggregates['grid_safety_penalty'] = [float(sum((runner.episode_reward_components[key][episode_idx] for key in grid_safety_keys))) for episode_idx in range(num_episodes)]
    return {'episodes': episode_indices, 'episode_total_reward': [float(value) for value in runner.episode_rewards], 'components': components, 'aggregates': aggregates}


__all__ = ['apply_controller_action_postprocessing', 'build_reward_summary', 'build_safety_summary', 'build_shared_update_ctx', 'close_runner', 'init_safety_tracking', 'record_projection_diagnostics', 'record_projection_event', 'record_projector_local_infeasible', 'save_runner_model', 'select_action_batch_with_info']
