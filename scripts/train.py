from __future__ import annotations
import os
import time
from datetime import datetime
from typing import Any
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from controllers.madrl.base_agent import get_agent_cls
from controllers.madrl.safety_projector import action_info_to_numpy, compute_action_gap_metrics_torch, enforce_local_action_feasibility_torch, merge_action_info_into_step_info
from controllers.madrl.safety_projector import SAFE_POC_ALGO_NAME
from controllers.madrl_controller import _override_soc_penalty_metrics
from scripts.checkpoints import build_checkpoint_manifest, write_checkpoint_manifest
from scripts.utils.nested import to_torch_nested
from scripts.utils.project_paths import get_tensorboard_run_dir
from scripts.utils.replay_buffer import ReplayBuffer

_PROJECTION_RESIDUAL_TOL = 1e-06


def _iso_timestamp(value: datetime) -> str:
    return value.astimezone().isoformat(timespec='seconds')


def _estimate_remaining_seconds(*, interaction_step: int, target_interactions: int, elapsed_seconds: float) -> float | None:
    remaining_interactions = max(int(target_interactions) - int(interaction_step), 0)
    if remaining_interactions == 0:
        return 0.0
    if interaction_step <= 0 or elapsed_seconds <= 0.0:
        return None
    return float(remaining_interactions * (elapsed_seconds / float(interaction_step)))


def init_safety_tracking(runner: Any) -> None:
    runner._safety_projection_stats = {stage: {'calls': 0, 'time_s': 0.0} for stage in ('rollout', 'target', 'actor')}
    runner._safety_diag = {'samples': 0, 'projected_fraction': 0.0, 'pre_trafo_import_violation_kw': 0.0, 'pre_trafo_export_violation_kw': 0.0, 'post_trafo_import_violation_kw': 0.0, 'post_trafo_export_violation_kw': 0.0, 'max_abs_delta': 0.0}
    runner._projector_local_infeasible_count = 0


def _safe_projection_enabled(runner: Any) -> bool:
    algo_cfg = getattr(runner.cfg, 'algo', None)
    algo_name = getattr(algo_cfg, 'name', '')
    return str(algo_name) == SAFE_POC_ALGO_NAME and getattr(runner, 'safety_projector', None) is not None


def _record_projection_event(runner: Any, *, stage: str, batch_size: int, elapsed_s: float, diagnostics: dict[str, Any] | None=None) -> None:
    normalized_stage = str(stage) if str(stage) in runner._safety_projection_stats else 'rollout'
    runner._safety_projection_stats[normalized_stage]['calls'] += 1
    runner._safety_projection_stats[normalized_stage]['time_s'] += float(elapsed_s)
    if diagnostics is not None:
        weighted_batch = max(int(diagnostics.get('batch_size', batch_size)), 1)
        runner._safety_diag['samples'] += weighted_batch
        for field_name in ('projected_fraction', 'pre_trafo_import_violation_kw', 'pre_trafo_export_violation_kw', 'post_trafo_import_violation_kw', 'post_trafo_export_violation_kw'):
            runner._safety_diag[field_name] += float(diagnostics.get(field_name, 0.0)) * weighted_batch
        runner._safety_diag['max_abs_delta'] = max(runner._safety_diag['max_abs_delta'], float(diagnostics.get('max_abs_delta', 0.0)))


def _record_projector_local_infeasible(runner: Any, residual_action_info: dict[str, torch.Tensor] | None) -> None:
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
    diagnostic_denominator = max(int(runner._safety_diag['samples']), 1)
    projection_time_total = float(sum((stage_stats['time_s'] for stage_stats in runner._safety_projection_stats.values())))
    return {'enabled': True, 'algorithm': str(runner.cfg.algo.name), 'projector_mode': str(getattr(runner.cfg.safety, 'projector_mode', 'joint_linearized')), 'projection_batches': int(sum((stage_stats['calls'] for stage_stats in runner._safety_projection_stats.values()))), 'projected_fraction': float(runner._safety_diag['projected_fraction'] / diagnostic_denominator), 'max_abs_action_delta': float(runner._safety_diag['max_abs_delta']), 'mean_pre_trafo_import_violation_kw': float(runner._safety_diag['pre_trafo_import_violation_kw'] / diagnostic_denominator), 'mean_pre_trafo_export_violation_kw': float(runner._safety_diag['pre_trafo_export_violation_kw'] / diagnostic_denominator), 'mean_post_trafo_import_violation_kw': float(runner._safety_diag['post_trafo_import_violation_kw'] / diagnostic_denominator), 'mean_post_trafo_export_violation_kw': float(runner._safety_diag['post_trafo_export_violation_kw'] / diagnostic_denominator), 'projection_time_s': projection_time_total, 'rollout_projection_calls': int(runner._safety_projection_stats['rollout']['calls']), 'target_projection_calls': int(runner._safety_projection_stats['target']['calls']), 'actor_projection_calls': int(runner._safety_projection_stats['actor']['calls']), 'rollout_projection_time_s': float(runner._safety_projection_stats['rollout']['time_s']), 'target_projection_time_s': float(runner._safety_projection_stats['target']['time_s']), 'actor_projection_time_s': float(runner._safety_projection_stats['actor']['time_s']), 'projector_local_infeasible_count': int(runner._projector_local_infeasible_count)}


def select_action_batch_with_info(runner: Any, obs_np: dict) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    obs_t = to_torch_nested(obs_np, runner.cfg.runtime.device)
    with torch.inference_mode():
        raw_action_t = torch.stack([agent.act_from_torch_obs(obs_t, noise_std=runner.noise_std) for agent in runner.agent_n], dim=1)
        action_info = None
        if _safe_projection_enabled(runner):
            projection_started = time.perf_counter()
            projected_action_t, diagnostics = runner.safety_projector.project_actions_from_safety_local(obs_t['safety_local'], raw_action_t, return_diagnostics=True)
            _record_projection_event(runner, stage='rollout', batch_size=int(raw_action_t.shape[0]), elapsed_s=time.perf_counter() - projection_started, diagnostics=diagnostics)
            action_t, projector_residual_info = enforce_local_action_feasibility_torch(obs_t['safety_local'], projected_action_t, efficiency=float(runner.cfg.env.efficiency), dt_hours=float(runner.cfg.env.dt), soc_min=float(runner.cfg.env.soc_min), soc_max=float(runner.cfg.env.soc_max))
            _record_projector_local_infeasible(runner, projector_residual_info)
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
        shared_ctx['projection_event_recorder'] = lambda **kwargs: _record_projection_event(runner, **kwargs)
        with torch.no_grad():
            noise = (torch.randn_like(target_actor_actions_clean) * float(runner.cfg.algo.policy_noise)).clamp(-float(runner.cfg.algo.noise_clip), float(runner.cfg.algo.noise_clip))
            target_actor_actions = (target_actor_actions_clean + noise).clamp(-float(runner.cfg.model.max_action), float(runner.cfg.model.max_action))
            projection_started = time.perf_counter()
            shared_ctx['projected_target_actions'] = runner.safety_projector.project_actions_from_safety_local(next_obs['safety_local'], target_actor_actions)
            _record_projection_event(runner, stage='target', batch_size=int(target_actor_actions.shape[0]), elapsed_s=time.perf_counter() - projection_started)
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


class TrainRunner:
    def __init__(self, cfg: Any, train_env: Any, eval_env: Any, env_name: str='GridEnv', number: int=1, seed: int=0) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.env_name = env_name
        self.env = train_env
        self.env_evaluate = eval_env
        agent_cls = get_agent_cls(self.cfg.algo.name)
        self.agent_n = [agent_cls(cfg, agent_id) for agent_id in range(self.cfg.env.num_agents)]
        self.safety_projector = getattr(self.agent_n[0], 'safety_projector', None) if self.agent_n else None
        self.apply_action_penalty = True
        self.replay_buffer = ReplayBuffer(self.cfg)
        log_dir = get_tensorboard_run_dir(algorithm=self.cfg.algo.name, env_name=env_name, run_number=number, seed=seed)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.vec_env_name = type(self.env).__name__
        self.history: list[Any] = []
        self.episode_rewards: list[float] = []
        self.total_steps = 0
        self.episodes_completed = 0
        self.noise_std = float(self.cfg.train.noise_std_init)
        self.perf_summary: dict[str, Any] = {}
        self.run_metadata: dict[str, Any] = {}
        self.reward_component_meta: list[Any] = []
        self.episode_reward_components: dict[str, list[float]] = {}
        init_safety_tracking(self)
        self._closed = False
    def select_action_batch(self, obs_np: dict) -> np.ndarray:
        action_batch, _ = self._select_action_batch_with_info(obs_np)
        return action_batch
    _apply_controller_action_postprocessing = apply_controller_action_postprocessing
    _build_shared_update_ctx = build_shared_update_ctx
    _select_action_batch_with_info = select_action_batch_with_info
    build_reward_summary = build_reward_summary
    build_safety_summary = build_safety_summary
    close = close_runner
    save_model = save_runner_model
    def run(self) -> int:
        target_interactions = self.cfg.train.resolved_max_train_steps(self.cfg.env.episode_limit) // self.cfg.train.num_envs
        interaction_step = 0
        episodes_completed = 0
        noise_decay = float(self.cfg.train.resolved_noise_std_decay())
        started_at = datetime.now().astimezone()
        run_start = time.perf_counter()
        action_time_total = 0.0
        env_step_time_total = 0.0
        update_time_total = 0.0
        update_calls = 0
        reward_metas = list(self.env_evaluate.reward_fn.component_meta)
        self.reward_component_meta = reward_metas
        self.episode_reward_components = {str(meta.key): [] for meta in reward_metas}
        progress_postfix_interval = max(1, int(getattr(self.cfg.train, 'progress_postfix_interval', 10)))
        active_episode_rewards = np.zeros(self.cfg.train.num_envs, dtype=np.float32)
        active_component_totals = {str(meta.key): np.zeros(self.cfg.train.num_envs, dtype=np.float32) for meta in reward_metas}
        progress = tqdm(total=target_interactions, desc='Training', unit='iters', disable=not bool(getattr(self.cfg.train, 'show_progress', True)))
        pending_progress_steps = 0
        last_progress_emit_step = -1
        def emit_progress(*, force: bool=False) -> None:
            nonlocal pending_progress_steps, last_progress_emit_step
            if not (force or interaction_step % progress_postfix_interval == 0 or interaction_step >= target_interactions):
                return
            if force and pending_progress_steps == 0 and (last_progress_emit_step == interaction_step):
                return
            avg_reward = float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0.0
            elapsed_seconds = max(time.perf_counter() - run_start, 0.0)
            remaining_seconds = _estimate_remaining_seconds(interaction_step=interaction_step, target_interactions=target_interactions, elapsed_seconds=elapsed_seconds)
            if pending_progress_steps > 0:
                progress.update(pending_progress_steps)
                pending_progress_steps = 0
            last_progress_emit_step = interaction_step
            progress.set_postfix({'avg_reward': f'{avg_reward:.2f}', 'steps/s': f'{self.total_steps / max(time.perf_counter() - run_start, 1e-06):.1f}', 'act_ms': f'{1000.0 * action_time_total / max(interaction_step, 1):.2f}', 'env_ms': f'{1000.0 * env_step_time_total / max(interaction_step, 1):.2f}', 'upd_ms': f'{1000.0 * update_time_total / max(update_calls, 1):.2f}', 'eta': '--' if remaining_seconds is None else f'{remaining_seconds:.1f}s'})
        try:
            obs, _ = self.env.reset()
            while interaction_step < target_interactions:
                action_start = time.perf_counter()
                action_batch, action_info = self._select_action_batch_with_info(obs)
                action_time_total += time.perf_counter() - action_start
                env_step_start = time.perf_counter()
                next_obs, reward, terminated, truncated, info_list = self.env.step([action_batch[:, agent_id].copy() for agent_id in range(self.cfg.env.num_agents)])
                reward, info_list = self._apply_controller_action_postprocessing(reward, info_list, action_info)
                done = np.logical_or(terminated, truncated).astype(np.float32)
                env_step_time_total += time.perf_counter() - env_step_start
                for env_idx, info in enumerate(info_list):
                    step_total = float(np.sum(reward[env_idx]))
                    active_episode_rewards[env_idx] += step_total
                    for meta in reward_metas:
                        component_key = str(meta.key)
                        component_value = float(np.sum(np.asarray(info[meta.key], dtype=np.float32)))
                        active_component_totals[component_key][env_idx] += float(meta.sign) * component_value
                self.replay_buffer.store_transitions_batched(obs, action_batch, reward, next_obs, done)
                obs = next_obs
                interaction_step += 1
                pending_progress_steps += 1
                self.total_steps += self.cfg.train.num_envs
                for env_idx, info in enumerate(info_list):
                    if not bool(info.get('episode_done', False)):
                        continue
                    episode_reward = float(active_episode_rewards[env_idx])
                    self.episode_rewards.append(episode_reward)
                    for meta in reward_metas:
                        component_key = str(meta.key)
                        self.episode_reward_components[component_key].append(float(active_component_totals[component_key][env_idx]))
                    self.writer.add_scalar('train_episode_total_reward', episode_reward, global_step=self.total_steps)
                    active_episode_rewards[env_idx] = 0.0
                    for meta in reward_metas:
                        active_component_totals[str(meta.key)][env_idx] = 0.0
                    episodes_completed += 1
                    self.episodes_completed = episodes_completed
                if self.cfg.train.use_noise_decay:
                    self.noise_std = max(self.noise_std - noise_decay, float(self.cfg.train.noise_std_min))
                if self.replay_buffer.current_size >= self.cfg.train.batch_size and interaction_step % self.cfg.train.update_interval == 0:
                    update_start = time.perf_counter()
                    for _ in range(self.cfg.train.updates_per_step):
                        batch_torch = self.replay_buffer.sample_torch(self.cfg.runtime.device, pin_memory=bool(getattr(self.cfg.runtime, 'pin_memory', False)), non_blocking=bool(getattr(self.cfg.runtime, 'non_blocking_transfers', False)))
                        shared_update_ctx = self._build_shared_update_ctx(batch_torch)
                        for agent in self.agent_n:
                            agent.train_on_batch(batch_torch, self.agent_n, shared_ctx=shared_update_ctx)
                        update_calls += 1
                    update_time_total += time.perf_counter() - update_start
                emit_progress()
        finally:
            emit_progress(force=True)
            progress.close()
        finished_at = datetime.now().astimezone()
        total_elapsed = max(time.perf_counter() - run_start, 1e-06)
        self.run_metadata = {'started_at': _iso_timestamp(started_at), 'finished_at': _iso_timestamp(finished_at), 'elapsed_seconds': float(round(total_elapsed, 3)), 'estimated_end_time': _iso_timestamp(finished_at)}
        self.perf_summary = {'seed': self.seed, 'runtime_mode': str(self.cfg.runtime.execution_mode), 'device': str(self.cfg.runtime.device), 'vec_env': self.vec_env_name, 'total_wall_time_s': total_elapsed, 'action_time_s': action_time_total, 'env_step_time_s': env_step_time_total, 'update_time_s': update_time_total, 'sample_time_s': 0.0, 'history_time_s': 0.0, 'agent_update_time_s': 0.0, 'update_calls': update_calls, 'steps_per_sec': self.total_steps / total_elapsed, 'avg_action_ms_per_iter': 1000.0 * action_time_total / max(interaction_step, 1), 'avg_env_ms_per_iter': 1000.0 * env_step_time_total / max(interaction_step, 1), 'avg_update_ms_per_call': 1000.0 * update_time_total / max(update_calls, 1), 'projection_time_s': float(sum((stage_stats['time_s'] for stage_stats in self._safety_projection_stats.values()))), 'rollout_projection_time_s': float(self._safety_projection_stats['rollout']['time_s']), 'target_projection_time_s': float(self._safety_projection_stats['target']['time_s']), 'actor_projection_time_s': float(self._safety_projection_stats['actor']['time_s']), 'rollout_projection_calls': int(self._safety_projection_stats['rollout']['calls']), 'target_projection_calls': int(self._safety_projection_stats['target']['calls']), 'actor_projection_calls': int(self._safety_projection_stats['actor']['calls'])}
        self.episodes_completed = episodes_completed
        return episodes_completed
