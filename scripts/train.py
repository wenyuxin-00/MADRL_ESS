"""Training loop for MADRL experiments."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from controllers.madrl.registry import get_agent_cls
from scripts.checkpoints import build_checkpoint_manifest, write_checkpoint_manifest
from scripts.recorders.episode_recorder import append_step_record, init_episode_record
from scripts.utils.nested import to_torch_nested
from scripts.utils.project_paths import get_tensorboard_run_dir
from scripts.utils.replay_buffer import ReplayBuffer, to_torch_batch


class TrainRunner:
    """Lightweight explicit training runner."""

    def __init__(
        self,
        cfg: Any,
        train_env: Any,
        eval_env: Any,
        env_name: str = "EnergyStorageEnv",
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

        self.replay_buffer = ReplayBuffer(self.cfg)
        log_dir = get_tensorboard_run_dir(
            algorithm=self.cfg.algo.name,
            env_name=env_name,
            run_number=number,
            seed=seed,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        self.history: deque = deque(maxlen=2000)
        self.episode_rewards = []
        self.total_steps = 0
        self.episodes_completed = 0
        self.noise_std = float(self.cfg.train.noise_std_init)
        self.perf_summary = {}
        self._closed = False

    def format_env_actions(self, action_batch: np.ndarray) -> list[np.ndarray]:
        """Convert `(num_envs, n_agents, action_dim)` to vec-env action layout."""
        return [action_batch[:, agent_id].copy() for agent_id in range(self.cfg.env.num_agents)]

    def select_action_batch(self, obs_np: dict) -> np.ndarray:
        """Run all actors on one batched observation."""
        obs_t = to_torch_nested(obs_np, self.cfg.runtime.device)
        with torch.no_grad():
            action_t = torch.stack(
                [agent.act_from_torch_obs(obs_t, noise_std=self.noise_std) for agent in self.agent_n],
                dim=1,
            )
        return action_t.cpu().numpy().astype(np.float32)

    def rollout_once(self, obs_np: dict | None = None) -> dict:
        """Execute one rollout step for notebooks and smoke checks."""
        if obs_np is None:
            obs_np, reset_info = self.env.reset()
        else:
            reset_info = None
        action_batch = self.select_action_batch(obs_np)
        next_obs, reward, terminated, truncated, info_list = self.env.step(
            self.format_env_actions(action_batch)
        )
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

    def save_model(self, model_dir: str, episode: int) -> None:
        """Save all agent checkpoints and refresh the latest manifest."""
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
        """Load all agent checkpoints from disk."""
        algo_dir = os.path.join(model_dir, self.cfg.algo.name)
        for agent in self.agent_n:
            agent.load_model(algo_dir, episode)

    def close(self) -> None:
        """Close environments and the TensorBoard writer."""
        if self._closed:
            return
        self.env.close()
        self.env_evaluate.close()
        self.writer.close()
        self._closed = True

    def run(self) -> int:
        """Run the training loop and return the number of finished episodes."""
        target_interactions = (
            self.cfg.train.resolved_max_train_steps(self.cfg.env.episode_limit)
            // self.cfg.train.num_envs
        )
        interaction_step = 0
        episodes_completed = 0
        noise_decay = float(self.cfg.train.resolved_noise_std_decay())
        run_start = time.perf_counter()
        action_time_total = 0.0
        env_step_time_total = 0.0
        update_time_total = 0.0
        update_calls = 0

        reward_metas = self.env_evaluate.reward_fn.component_meta
        active_histories = [
            init_episode_record(
                n_agents=self.cfg.env.num_agents,
                init_soc=float(self.cfg.env.init_soc),
                reward_metas=reward_metas,
            )
            for _ in range(self.cfg.train.num_envs)
        ]
        active_episode_rewards = np.zeros(self.cfg.train.num_envs, dtype=np.float32)

        progress = tqdm(total=target_interactions, desc="Training", unit="iters")

        try:
            obs, _ = self.env.reset()
            while interaction_step < target_interactions:
                action_start = time.perf_counter()
                action_batch = self.select_action_batch(obs)
                action_time_total += time.perf_counter() - action_start

                env_step_start = time.perf_counter()
                next_obs, reward, terminated, truncated, info_list = self.env.step(
                    self.format_env_actions(action_batch)
                )
                done = np.logical_or(terminated, truncated).astype(np.float32)
                env_step_time_total += time.perf_counter() - env_step_start

                for env_idx, info in enumerate(info_list):
                    step_total = float(np.sum(reward[env_idx]))
                    active_episode_rewards[env_idx] += step_total
                    append_step_record(
                        active_histories[env_idx],
                        info,
                        step_total=step_total,
                        reward_metas=reward_metas,
                    )

                self.replay_buffer.store_transitions_batched(
                    obs,
                    action_batch,
                    reward,
                    next_obs,
                    done,
                )

                obs = next_obs
                interaction_step += 1
                self.total_steps += self.cfg.train.num_envs

                for env_idx, info in enumerate(info_list):
                    if not bool(info.get("episode_done", False)):
                        continue

                    self.history.append(active_histories[env_idx])

                    episode_reward = float(active_episode_rewards[env_idx])
                    self.episode_rewards.append(episode_reward)
                    self.writer.add_scalar(
                        "train_episode_total_reward",
                        episode_reward,
                        global_step=self.total_steps,
                    )

                    active_histories[env_idx] = init_episode_record(
                        n_agents=self.cfg.env.num_agents,
                        init_soc=float(self.cfg.env.init_soc),
                        reward_metas=reward_metas,
                    )
                    active_episode_rewards[env_idx] = 0.0
                    episodes_completed += 1
                    self.episodes_completed = episodes_completed

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
                        batch_np = self.replay_buffer.sample()
                        batch_torch = to_torch_batch(batch_np, self.cfg.runtime.device)
                        for agent in self.agent_n:
                            agent.train_on_batch(batch_torch, self.agent_n)
                        update_calls += 1
                    update_time_total += time.perf_counter() - update_start

                progress.update(1)
                if interaction_step % 10 == 0 or interaction_step == target_interactions:
                    avg_reward = float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0.0
                    elapsed = max(time.perf_counter() - run_start, 1e-6)
                    progress.set_postfix(
                        {
                            "avg_reward": f"{avg_reward:.2f}",
                            "steps/s": f"{self.total_steps / elapsed:.1f}",
                            "act_ms": f"{1000.0 * action_time_total / max(interaction_step, 1):.2f}",
                            "env_ms": f"{1000.0 * env_step_time_total / max(interaction_step, 1):.2f}",
                            "upd_ms": f"{1000.0 * update_time_total / max(update_calls, 1):.2f}",
                        }
                    )
        finally:
            progress.close()

        total_elapsed = max(time.perf_counter() - run_start, 1e-6)
        self.perf_summary = {
            "seed": self.seed,
            "runtime_mode": str(self.cfg.runtime.execution_mode),
            "device": str(self.cfg.runtime.device),
            "total_wall_time_s": total_elapsed,
            "action_time_s": action_time_total,
            "env_step_time_s": env_step_time_total,
            "update_time_s": update_time_total,
            "update_calls": update_calls,
            "steps_per_sec": self.total_steps / total_elapsed,
            "avg_action_ms_per_iter": 1000.0 * action_time_total / max(interaction_step, 1),
            "avg_env_ms_per_iter": 1000.0 * env_step_time_total / max(interaction_step, 1),
            "avg_update_ms_per_call": 1000.0 * update_time_total / max(update_calls, 1),
        }
        self.episodes_completed = episodes_completed
        return episodes_completed
