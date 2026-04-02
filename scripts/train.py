"""Training loop for MADRL experiments."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from controllers.action_feasibility import (
    action_info_to_numpy,
    compute_action_gap_metrics_torch,
    enforce_local_action_feasibility_torch,
    merge_action_info_into_step_info,
)
from controllers.madrl.registry import get_agent_cls
from controllers.madrl.safety_projector import SAFE_POC_ALGO_NAME
from scripts.checkpoints import build_checkpoint_manifest, write_checkpoint_manifest
from scripts.utils.nested import to_torch_nested
from scripts.utils.project_paths import get_tensorboard_run_dir
from scripts.utils.replay_buffer import ReplayBuffer

_PROJECTION_RESIDUAL_TOL = 1e-6


def _iso_timestamp(value: datetime) -> str:
    return value.astimezone().isoformat(timespec="seconds")


def _estimate_remaining_seconds(
    *,
    interaction_step: int,
    target_interactions: int,
    elapsed_seconds: float,
) -> float | None:
    remaining_interactions = max(int(target_interactions) - int(interaction_step), 0)
    if remaining_interactions == 0:
        return 0.0
    if interaction_step <= 0 or elapsed_seconds <= 0.0:
        return None
    return float(remaining_interactions * (elapsed_seconds / float(interaction_step)))


def _build_progress_payload(
    *,
    interaction_step: int,
    target_interactions: int,
    episodes_completed: int,
    total_steps: int,
    avg_reward: float,
    action_time_total: float,
    env_step_time_total: float,
    update_time_total: float,
    update_calls: int,
    run_start: float,
    started_at: datetime,
    status: str,
    error_message: str = "",
) -> dict[str, object]:
    now = datetime.now().astimezone()
    elapsed = max(time.perf_counter() - run_start, 0.0)
    safe_elapsed = max(elapsed, 1e-6)
    remaining_seconds = _estimate_remaining_seconds(
        interaction_step=interaction_step,
        target_interactions=target_interactions,
        elapsed_seconds=elapsed,
    )
    if status != "running" and int(interaction_step) >= int(target_interactions):
        remaining_seconds = 0.0
        estimated_end_time = now
    elif remaining_seconds is None:
        estimated_end_time = None
    else:
        estimated_end_time = now + timedelta(seconds=float(remaining_seconds))
    return {
        "status": str(status),
        "error_message": str(error_message),
        "interaction_step": int(interaction_step),
        "target_interactions": int(target_interactions),
        "episodes_completed": int(episodes_completed),
        "total_steps": int(total_steps),
        "avg_reward": float(avg_reward),
        "steps_per_sec": float(total_steps / safe_elapsed),
        "avg_action_ms_per_iter": float(1000.0 * action_time_total / max(interaction_step, 1)),
        "avg_env_ms_per_iter": float(1000.0 * env_step_time_total / max(interaction_step, 1)),
        "avg_update_ms_per_call": float(1000.0 * update_time_total / max(update_calls, 1)),
        "started_at": _iso_timestamp(started_at),
        "updated_at": _iso_timestamp(now),
        "elapsed_seconds": float(round(elapsed, 3)),
        "remaining_seconds": None if remaining_seconds is None else float(round(remaining_seconds, 3)),
        "estimated_end_time": None if estimated_end_time is None else _iso_timestamp(estimated_end_time),
    }


def _write_progress_snapshot(path: str | os.PathLike[str], payload: dict[str, object]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _override_soc_penalty_metrics(
    action_info: dict[str, torch.Tensor] | None,
    penalty_source_info: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor] | None:
    if action_info is None:
        return penalty_source_info
    if penalty_source_info is None:
        return action_info
    merged = dict(action_info)
    for key in ("soc_penalty_unweighted", "action_penalty_unweighted"):
        if key in penalty_source_info:
            merged[key] = penalty_source_info[key]
    return merged


class TrainRunner:
    """Lightweight explicit training runner."""

    def __init__(
        self,
        cfg: Any,
        train_env: Any,
        eval_env: Any,
        env_name: str = "GridEnv",
        number: int = 1,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.env_name = env_name
        self.env = train_env
        self.env_evaluate = eval_env

        agent_cls = get_agent_cls(self.cfg.algo.name)
        self.agent_n = [agent_cls(cfg, agent_id) for agent_id in range(self.cfg.env.num_agents)]
        self.safety_projector = getattr(self.agent_n[0], "safety_projector", None) if self.agent_n else None
        self.apply_action_penalty = True

        self.replay_buffer = ReplayBuffer(self.cfg)
        log_dir = get_tensorboard_run_dir(
            algorithm=self.cfg.algo.name,
            env_name=env_name,
            run_number=number,
            seed=seed,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = str(log_dir)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.vec_env_name = type(self.env).__name__
        self.agent_performance = dict(getattr(self.agent_n[0], "performance_summary", {}))
        self.progress_write_interval_seconds = max(
            0.0,
            float(getattr(self.cfg.train, "progress_write_interval_seconds", 5.0)),
        )
        self.history: list[dict[str, Any]] = []
        self.episode_rewards: list[float] = []
        self.total_steps = 0
        self.episodes_completed = 0
        self.noise_std = float(self.cfg.train.noise_std_init)
        self.perf_summary: dict[str, Any] = {}
        self.run_metadata: dict[str, Any] = {}
        self.reward_component_meta: list[Any] = []
        self.episode_reward_components: dict[str, list[float]] = {}
        self._safety_projection_batches = 0
        self._safety_projection_samples = 0
        self._safety_diagnostic_samples = 0
        self._safety_projection_calls_by_stage = {
            "rollout": 0,
            "target": 0,
            "actor": 0,
        }
        self._safety_projection_samples_by_stage = {
            "rollout": 0,
            "target": 0,
            "actor": 0,
        }
        self._safety_projection_time_s_by_stage = {
            "rollout": 0.0,
            "target": 0.0,
            "actor": 0.0,
        }
        self._safety_projected_fraction_weighted = 0.0
        self._safety_mean_abs_delta_weighted = 0.0
        self._safety_mean_abs_delta_kw_weighted = 0.0
        self._safety_pre_violation_weighted = 0.0
        self._safety_post_violation_weighted = 0.0
        self._safety_max_abs_delta = 0.0
        self._safety_env_fallback_steps = 0
        self._safety_env_observations = 0
        self._safety_env_action_pen_total = 0.0
        self._projector_local_infeasible_count = 0
        self._closed = False

    def _safe_projection_enabled(self) -> bool:
        algo_cfg = getattr(self.cfg, "algo", None)
        algo_name = getattr(algo_cfg, "name", "")
        return str(algo_name) == SAFE_POC_ALGO_NAME and getattr(self, "safety_projector", None) is not None

    def _record_projection_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        batch_size = max(int(diagnostics.get("batch_size", 0)), 1)
        self._safety_diagnostic_samples += batch_size
        self._safety_projected_fraction_weighted += float(diagnostics.get("projected_fraction", 0.0)) * batch_size
        self._safety_mean_abs_delta_weighted += float(diagnostics.get("mean_abs_delta", 0.0)) * batch_size
        self._safety_mean_abs_delta_kw_weighted += float(diagnostics.get("mean_abs_delta_kw", 0.0)) * batch_size
        self._safety_pre_violation_weighted += float(diagnostics.get("pre_violation", 0.0)) * batch_size
        self._safety_post_violation_weighted += float(diagnostics.get("post_violation", 0.0)) * batch_size
        self._safety_max_abs_delta = max(
            self._safety_max_abs_delta,
            float(diagnostics.get("max_abs_delta", 0.0)),
        )

    def _record_projection_event(
        self,
        *,
        stage: str,
        batch_size: int,
        elapsed_s: float,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        normalized_stage = str(stage)
        if normalized_stage not in self._safety_projection_calls_by_stage:
            normalized_stage = "rollout"
        safe_batch_size = max(int(batch_size), 1)
        self._safety_projection_batches += 1
        self._safety_projection_samples += safe_batch_size
        self._safety_projection_calls_by_stage[normalized_stage] += 1
        self._safety_projection_samples_by_stage[normalized_stage] += safe_batch_size
        self._safety_projection_time_s_by_stage[normalized_stage] += float(elapsed_s)
        if diagnostics is not None:
            self._record_projection_diagnostics(diagnostics)

    def _record_env_fallback(self, info: dict[str, Any]) -> None:
        del info
        return

    def _record_projector_local_infeasible(
        self,
        residual_action_info: dict[str, torch.Tensor] | None,
    ) -> None:
        if residual_action_info is None:
            return
        residual = residual_action_info.get("soc_penalty_unweighted")
        if residual is None:
            return
        residual_tensor = torch.as_tensor(residual, dtype=torch.float32)
        if residual_tensor.ndim == 1:
            residual_tensor = residual_tensor.unsqueeze(0)
        affected = torch.any(residual_tensor > _PROJECTION_RESIDUAL_TOL, dim=-1)
        self._projector_local_infeasible_count += int(torch.count_nonzero(affected).item())

    def build_safety_summary(self) -> dict[str, Any]:
        if not self._safe_projection_enabled():
            return {
                "enabled": False,
                "algorithm": str(self.cfg.algo.name),
            }

        diagnostic_denominator = max(self._safety_diagnostic_samples, 1)
        env_denominator = max(self._safety_env_observations, 1)
        projection_time_total = float(sum(self._safety_projection_time_s_by_stage.values()))
        return {
            "enabled": True,
            "algorithm": str(self.cfg.algo.name),
            "projector_mode": str(getattr(self.cfg.safety, "projector_mode", "joint_linearized")),
            "projection_batches": int(self._safety_projection_batches),
            "projection_samples": int(self._safety_projection_samples),
            "projection_diagnostic_samples": int(self._safety_diagnostic_samples),
            "projected_fraction": float(self._safety_projected_fraction_weighted / diagnostic_denominator),
            "mean_abs_action_delta": float(self._safety_mean_abs_delta_weighted / diagnostic_denominator),
            "max_abs_action_delta": float(self._safety_max_abs_delta),
            "mean_abs_action_delta_kw": float(self._safety_mean_abs_delta_kw_weighted / diagnostic_denominator),
            "mean_pre_projection_violation": float(self._safety_pre_violation_weighted / diagnostic_denominator),
            "mean_post_projection_violation": float(self._safety_post_violation_weighted / diagnostic_denominator),
            "projection_time_s": projection_time_total,
            "rollout_projection_calls": int(self._safety_projection_calls_by_stage["rollout"]),
            "target_projection_calls": int(self._safety_projection_calls_by_stage["target"]),
            "actor_projection_calls": int(self._safety_projection_calls_by_stage["actor"]),
            "rollout_projection_samples": int(self._safety_projection_samples_by_stage["rollout"]),
            "target_projection_samples": int(self._safety_projection_samples_by_stage["target"]),
            "actor_projection_samples": int(self._safety_projection_samples_by_stage["actor"]),
            "rollout_projection_time_s": float(self._safety_projection_time_s_by_stage["rollout"]),
            "target_projection_time_s": float(self._safety_projection_time_s_by_stage["target"]),
            "actor_projection_time_s": float(self._safety_projection_time_s_by_stage["actor"]),
            "env_fallback_action_pen_steps": int(self._safety_env_fallback_steps),
            "env_fallback_action_pen_rate": float(self._safety_env_fallback_steps / env_denominator),
            "env_fallback_action_pen_mean": float(self._safety_env_action_pen_total / env_denominator),
            "projector_local_infeasible_count": int(self._projector_local_infeasible_count),
        }

    def format_env_actions(self, action_batch: np.ndarray) -> list[np.ndarray]:
        return [action_batch[:, agent_id].copy() for agent_id in range(self.cfg.env.num_agents)]

    def _select_action_batch_with_info(
        self,
        obs_np: dict,
    ) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        obs_t = to_torch_nested(obs_np, self.cfg.runtime.device)
        with torch.inference_mode():
            raw_action_t = torch.stack(
                [agent.act_from_torch_obs(obs_t, noise_std=self.noise_std) for agent in self.agent_n],
                dim=1,
            )
            action_info = None
            if self._safe_projection_enabled():
                projection_started = time.perf_counter()
                projected_action_t, diagnostics = self.safety_projector.project_actions_from_safety_local(
                    obs_t["safety_local"],
                    raw_action_t,
                    return_diagnostics=True,
                )
                self._record_projection_event(
                    stage="rollout",
                    batch_size=int(raw_action_t.shape[0]),
                    elapsed_s=time.perf_counter() - projection_started,
                    diagnostics=diagnostics,
                )
                action_t, projector_residual_info = enforce_local_action_feasibility_torch(
                    obs_t["safety_local"],
                    projected_action_t,
                    efficiency=float(self.cfg.env.efficiency),
                    dt_hours=float(self.cfg.env.dt),
                    soc_min=float(self.cfg.env.soc_min),
                    soc_max=float(self.cfg.env.soc_max),
                )
                self._record_projector_local_infeasible(projector_residual_info)
                action_info = compute_action_gap_metrics_torch(obs_t["safety_local"], raw_action_t, action_t)
                action_info = _override_soc_penalty_metrics(action_info, projector_residual_info)
            elif "safety_local" in obs_t:
                action_t, action_info = enforce_local_action_feasibility_torch(
                    obs_t["safety_local"],
                    raw_action_t,
                    efficiency=float(self.cfg.env.efficiency),
                    dt_hours=float(self.cfg.env.dt),
                    soc_min=float(self.cfg.env.soc_min),
                    soc_max=float(self.cfg.env.soc_max),
                )
            else:
                action_t = raw_action_t
        return (
            action_t.to(dtype=torch.float32).cpu().numpy(),
            action_info_to_numpy(action_info),
        )

    def select_action_batch(self, obs_np: dict) -> np.ndarray:
        action_batch, _ = self._select_action_batch_with_info(obs_np)
        return action_batch

    def _apply_controller_action_postprocessing(
        self,
        reward: np.ndarray,
        info_list: list[dict[str, Any]],
        action_info: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        reward_array = np.asarray(reward, dtype=np.float32)
        if reward_array.ndim == 2:
            reward_array = reward_array[..., None]

        processed_info_list: list[dict[str, Any]] = []
        for env_idx, info in enumerate(info_list):
            env_action_info = None
            if action_info is not None:
                env_action_info = {
                    key: np.asarray(value[env_idx], dtype=np.float32)
                    for key, value in action_info.items()
                }
            merged_info, action_penalty = merge_action_info_into_step_info(
                info,
                env_action_info,
                soc_pen_weight=float(self.cfg.reward.w_soc_pen),
                apply_action_penalty=self.apply_action_penalty,
            )
            reward_array[env_idx, :, 0] -= np.asarray(action_penalty, dtype=np.float32)
            merged_info["reward"] = np.asarray(reward_array[env_idx, :, 0], dtype=np.float32)
            processed_info_list.append(merged_info)
        return reward_array.astype(np.float32), processed_info_list

    def rollout_once(self, obs_np: dict | None = None) -> dict:
        if obs_np is None:
            obs_np, reset_info = self.env.reset()
        else:
            reset_info = None
        action_batch, action_info = self._select_action_batch_with_info(obs_np)
        next_obs, reward, terminated, truncated, info_list = self.env.step(
            self.format_env_actions(action_batch)
        )
        reward, info_list = self._apply_controller_action_postprocessing(reward, info_list, action_info)
        done = np.logical_or(terminated, truncated).astype(np.float32)
        return {
            "obs": obs_np,
            "reset_info": reset_info,
            "action_batch": action_batch,
            "next_obs": next_obs,
            "reward": reward,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "info_list": info_list,
        }

    def _build_shared_update_ctx(self, batch: dict[str, Any]) -> dict[str, Any]:
        next_obs = batch["next_obs"]
        with torch.no_grad():
            target_actor_actions_clean = torch.stack(
                [agent._actor_target_call(next_obs) for agent in self.agent_n],
                dim=1,
            )
        shared_ctx = {
            "target_actor_actions_clean": target_actor_actions_clean,
            "batch_size": int(batch["action"].shape[0]),
            "device": batch["action"].device,
        }
        if self._safe_projection_enabled():
            shared_ctx["safety_projector"] = self.safety_projector
            shared_ctx["projection_event_recorder"] = self._record_projection_event
            with torch.no_grad():
                noise = (torch.randn_like(target_actor_actions_clean) * float(self.cfg.algo.policy_noise)).clamp(
                    -float(self.cfg.algo.noise_clip),
                    float(self.cfg.algo.noise_clip),
                )
                target_actor_actions = (target_actor_actions_clean + noise).clamp(
                    -float(self.cfg.model.max_action),
                    float(self.cfg.model.max_action),
                )
                projection_started = time.perf_counter()
                shared_ctx["projected_target_actions"] = self.safety_projector.project_actions_from_safety_local(
                    next_obs["safety_local"],
                    target_actor_actions,
                )
                self._record_projection_event(
                    stage="target",
                    batch_size=int(target_actor_actions.shape[0]),
                    elapsed_s=time.perf_counter() - projection_started,
                )
        return shared_ctx

    def save_model(self, model_dir: str, episode: int) -> None:
        algo_dir = os.path.join(model_dir, self.cfg.algo.name)
        os.makedirs(algo_dir, exist_ok=True)
        for agent in self.agent_n:
            agent.save_model(algo_dir, episode)

        manifest = build_checkpoint_manifest(
            algorithm=self.cfg.algo.name,
            saved_episode_tag=episode,
            episodes_completed=self.episodes_completed,
            total_steps=self.total_steps,
            num_envs=self.cfg.train.num_envs,
            episode_limit=self.cfg.env.episode_limit,
            save_dir=algo_dir,
        )
        write_checkpoint_manifest(algo_dir, manifest)

    def load_model(self, model_dir: str, episode: int) -> None:
        algo_dir = os.path.join(model_dir, self.cfg.algo.name)
        for agent in self.agent_n:
            agent.load_model(algo_dir, episode)

    def close(self) -> None:
        if self._closed:
            return
        self.env.close()
        self.env_evaluate.close()
        self.writer.close()
        self._closed = True

    def build_reward_summary(self) -> dict[str, Any]:
        episode_indices = list(range(1, len(self.episode_rewards) + 1))
        components: dict[str, dict[str, Any]] = {}
        for meta in self.reward_component_meta:
            key = str(meta.key)
            values = [float(value) for value in self.episode_reward_components.get(key, [])]
            components[key] = {
                "label": str(getattr(meta, "label", meta.key)),
                "color": str(getattr(meta, "color", "#111827")),
                "sign": int(getattr(meta, "sign", 0)),
                "values": values,
            }

        aggregates: dict[str, list[float]] = {}
        grid_safety_keys = [
            str(meta.key)
            for meta in self.reward_component_meta
            if str(meta.key).startswith("r_safe_") and int(getattr(meta, "sign", 0)) == -1
        ]
        if grid_safety_keys and all(key in components for key in grid_safety_keys):
            num_episodes = len(episode_indices)
            aggregates["grid_safety_penalty"] = [
                float(sum(self.episode_reward_components[key][episode_idx] for key in grid_safety_keys))
                for episode_idx in range(num_episodes)
            ]

        return {
            "episodes": episode_indices,
            "episode_total_reward": [float(value) for value in self.episode_rewards],
            "components": components,
            "aggregates": aggregates,
        }

    def run(self) -> int:
        target_interactions = (
            self.cfg.train.resolved_max_train_steps(self.cfg.env.episode_limit)
            // self.cfg.train.num_envs
        )
        interaction_step = 0
        episodes_completed = 0
        noise_decay = float(self.cfg.train.resolved_noise_std_decay())
        started_at = datetime.now().astimezone()
        run_start = time.perf_counter()
        action_time_total = 0.0
        env_step_time_total = 0.0
        update_time_total = 0.0
        sample_time_total = 0.0
        history_time_total = 0.0
        progress_io_time_total = 0.0
        agent_update_time_total = 0.0
        update_calls = 0
        error_message = ""
        run_status = "completed"

        reward_metas = list(self.env_evaluate.reward_fn.component_meta)
        self.reward_component_meta = reward_metas
        self.episode_reward_components = {str(meta.key): [] for meta in reward_metas}
        progress_postfix_interval = max(1, int(getattr(self.cfg.train, "progress_postfix_interval", 10)))
        progress_state_path = getattr(self.cfg.runtime, "progress_state_path", None)
        active_episode_rewards = np.zeros(self.cfg.train.num_envs, dtype=np.float32)
        active_component_totals = {
            str(meta.key): np.zeros(self.cfg.train.num_envs, dtype=np.float32)
            for meta in reward_metas
        }

        progress = tqdm(
            total=target_interactions,
            desc="Training",
            unit="iters",
            disable=not bool(getattr(self.cfg.train, "show_progress", True)),
        )
        pending_progress_steps = 0
        last_progress_emit_step = -1
        last_progress_write_at = run_start

        def write_progress_snapshot(payload: dict[str, object]) -> None:
            nonlocal progress_io_time_total, last_progress_write_at
            if progress_state_path is None:
                return
            write_started = time.perf_counter()
            _write_progress_snapshot(progress_state_path, payload)
            progress_io_time_total += time.perf_counter() - write_started
            last_progress_write_at = time.perf_counter()

        def emit_progress(*, force: bool = False) -> None:
            nonlocal pending_progress_steps, last_progress_emit_step
            should_refresh = (
                force
                or interaction_step % progress_postfix_interval == 0
                or interaction_step >= target_interactions
            )
            avg_reward = float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0.0
            should_write_progress = (
                progress_state_path is not None
                and (
                    force
                    or self.progress_write_interval_seconds <= 0.0
                    or (time.perf_counter() - last_progress_write_at) >= self.progress_write_interval_seconds
                )
            )
            if not should_refresh and not should_write_progress:
                return
            if force and pending_progress_steps == 0 and last_progress_emit_step == interaction_step:
                if not should_write_progress:
                    return
            elapsed_seconds = max(time.perf_counter() - run_start, 0.0)
            remaining_seconds = _estimate_remaining_seconds(
                interaction_step=interaction_step,
                target_interactions=target_interactions,
                elapsed_seconds=elapsed_seconds,
            )
            if should_refresh:
                if pending_progress_steps > 0:
                    progress.update(pending_progress_steps)
                    pending_progress_steps = 0
                last_progress_emit_step = interaction_step
                progress.set_postfix(
                    {
                        "avg_reward": f"{avg_reward:.2f}",
                        "steps/s": f"{self.total_steps / max(time.perf_counter() - run_start, 1e-6):.1f}",
                        "act_ms": f"{1000.0 * action_time_total / max(interaction_step, 1):.2f}",
                        "env_ms": f"{1000.0 * env_step_time_total / max(interaction_step, 1):.2f}",
                        "upd_ms": f"{1000.0 * update_time_total / max(update_calls, 1):.2f}",
                        "eta": "--" if remaining_seconds is None else f"{remaining_seconds:.1f}s",
                    }
                )
            if should_write_progress:
                write_progress_snapshot(
                    _build_progress_payload(
                        interaction_step=interaction_step,
                        target_interactions=target_interactions,
                        episodes_completed=episodes_completed,
                        total_steps=self.total_steps,
                        avg_reward=avg_reward,
                        action_time_total=action_time_total,
                        env_step_time_total=env_step_time_total,
                        update_time_total=update_time_total,
                        update_calls=update_calls,
                        run_start=run_start,
                        started_at=started_at,
                        status="running",
                    )
                )

        if progress_state_path is not None:
            write_progress_snapshot(
                _build_progress_payload(
                    interaction_step=0,
                    target_interactions=target_interactions,
                    episodes_completed=0,
                    total_steps=0,
                    avg_reward=0.0,
                    action_time_total=0.0,
                    env_step_time_total=0.0,
                    update_time_total=0.0,
                    update_calls=0,
                    run_start=run_start,
                    started_at=started_at,
                    status="running",
                ),
            )

        try:
            obs, _ = self.env.reset()
            while interaction_step < target_interactions:
                action_start = time.perf_counter()
                action_batch, action_info = self._select_action_batch_with_info(obs)
                action_time_total += time.perf_counter() - action_start

                env_step_start = time.perf_counter()
                next_obs, reward, terminated, truncated, info_list = self.env.step(
                    self.format_env_actions(action_batch)
                )
                reward, info_list = self._apply_controller_action_postprocessing(reward, info_list, action_info)
                done = np.logical_or(terminated, truncated).astype(np.float32)
                env_step_time_total += time.perf_counter() - env_step_start

                history_started = time.perf_counter()
                for env_idx, info in enumerate(info_list):
                    step_total = float(np.sum(reward[env_idx]))
                    active_episode_rewards[env_idx] += step_total
                    for meta in reward_metas:
                        component_key = str(meta.key)
                        component_value = float(np.sum(np.asarray(info[meta.key], dtype=np.float32)))
                        active_component_totals[component_key][env_idx] += float(meta.sign) * component_value
                history_time_total += time.perf_counter() - history_started

                self.replay_buffer.store_transitions_batched(
                    obs,
                    action_batch,
                    reward,
                    next_obs,
                    done,
                )

                obs = next_obs
                interaction_step += 1
                pending_progress_steps += 1
                self.total_steps += self.cfg.train.num_envs

                history_started = time.perf_counter()
                for env_idx, info in enumerate(info_list):
                    if not bool(info.get("episode_done", False)):
                        continue

                    episode_reward = float(active_episode_rewards[env_idx])
                    self.episode_rewards.append(episode_reward)
                    for meta in reward_metas:
                        component_key = str(meta.key)
                        self.episode_reward_components[component_key].append(
                            float(active_component_totals[component_key][env_idx])
                        )
                    self.writer.add_scalar(
                        "train_episode_total_reward",
                        episode_reward,
                        global_step=self.total_steps,
                    )

                    active_episode_rewards[env_idx] = 0.0
                    for meta in reward_metas:
                        active_component_totals[str(meta.key)][env_idx] = 0.0
                    episodes_completed += 1
                    self.episodes_completed = episodes_completed
                history_time_total += time.perf_counter() - history_started

                if self.cfg.train.use_noise_decay:
                    self.noise_std = max(
                        self.noise_std - noise_decay,
                        float(self.cfg.train.noise_std_min),
                    )

                if (
                    self.replay_buffer.current_size >= self.cfg.train.batch_size
                    and interaction_step % self.cfg.train.update_interval == 0
                ):
                    update_start = time.perf_counter()
                    for _ in range(self.cfg.train.updates_per_step):
                        sample_start = time.perf_counter()
                        batch_torch = self.replay_buffer.sample_torch(
                            self.cfg.runtime.device,
                            pin_memory=bool(getattr(self.cfg.runtime, "pin_memory", False)),
                            non_blocking=bool(getattr(self.cfg.runtime, "non_blocking_transfers", False)),
                        )
                        sample_time_total += time.perf_counter() - sample_start

                        agent_update_start = time.perf_counter()
                        shared_update_ctx = self._build_shared_update_ctx(batch_torch)
                        for agent in self.agent_n:
                            agent.train_on_batch(batch_torch, self.agent_n, shared_ctx=shared_update_ctx)
                        agent_update_time_total += time.perf_counter() - agent_update_start
                        update_calls += 1
                    update_time_total += time.perf_counter() - update_start

                emit_progress()
        except Exception as exc:
            run_status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            emit_progress(force=True)
            if progress_state_path is not None:
                avg_reward = float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0.0
                write_progress_snapshot(
                    _build_progress_payload(
                        interaction_step=interaction_step,
                        target_interactions=target_interactions,
                        episodes_completed=episodes_completed,
                        total_steps=self.total_steps,
                        avg_reward=avg_reward,
                        action_time_total=action_time_total,
                        env_step_time_total=env_step_time_total,
                        update_time_total=update_time_total,
                        update_calls=update_calls,
                        run_start=run_start,
                        started_at=started_at,
                        status=run_status,
                        error_message=error_message,
                    ),
                )
            progress.close()

        finished_at = datetime.now().astimezone()
        total_elapsed = max(time.perf_counter() - run_start, 1e-6)
        self.run_metadata = {
            "started_at": _iso_timestamp(started_at),
            "finished_at": _iso_timestamp(finished_at),
            "elapsed_seconds": float(round(total_elapsed, 3)),
            "estimated_end_time": _iso_timestamp(finished_at),
        }
        self.perf_summary = {
            "seed": self.seed,
            "runtime_mode": str(self.cfg.runtime.execution_mode),
            "device": str(self.cfg.runtime.device),
            "vec_env": self.vec_env_name,
            "tensorboard_dir": self.tensorboard_dir,
            "total_wall_time_s": total_elapsed,
            "action_time_s": action_time_total,
            "env_step_time_s": env_step_time_total,
            "update_time_s": update_time_total,
            "sample_time_s": sample_time_total,
            "history_time_s": history_time_total,
            "progress_io_time_s": progress_io_time_total,
            "agent_update_time_s": agent_update_time_total,
            "update_calls": update_calls,
            "steps_per_sec": self.total_steps / total_elapsed,
            "avg_action_ms_per_iter": 1000.0 * action_time_total / max(interaction_step, 1),
            "avg_env_ms_per_iter": 1000.0 * env_step_time_total / max(interaction_step, 1),
            "avg_update_ms_per_call": 1000.0 * update_time_total / max(update_calls, 1),
            "projection_time_s": float(sum(self._safety_projection_time_s_by_stage.values())),
            "rollout_projection_time_s": float(self._safety_projection_time_s_by_stage["rollout"]),
            "target_projection_time_s": float(self._safety_projection_time_s_by_stage["target"]),
            "actor_projection_time_s": float(self._safety_projection_time_s_by_stage["actor"]),
            "rollout_projection_calls": int(self._safety_projection_calls_by_stage["rollout"]),
            "target_projection_calls": int(self._safety_projection_calls_by_stage["target"]),
            "actor_projection_calls": int(self._safety_projection_calls_by_stage["actor"]),
        }
        self.perf_summary.update(self.agent_performance)
        self.episodes_completed = episodes_completed
        return episodes_completed
