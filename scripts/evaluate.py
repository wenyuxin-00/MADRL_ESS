"""Evaluation helpers for controllers."""

from __future__ import annotations

import numpy as np

from scripts.recorders.episode_recorder import append_step_record, init_episode_record
from scripts.recorders.grid_recorder import append_grid_step_record, init_grid_record


def evaluate_controller(
    env,
    controller,
    n_episodes: int = 1,
    deterministic: bool = True,
    episode_indices: list[int] | None = None,
    record_history: bool = True,
):
    """Evaluate an arbitrary controller on a single environment instance."""
    if episode_indices is not None:
        n_episodes = len(episode_indices)

    reward_metas = getattr(env.reward_fn, "component_meta", [])
    histories = []
    grid_histories = []
    episode_rewards = []

    for episode_offset in range(int(n_episodes)):
        episode_idx = None if episode_indices is None else episode_indices[episode_offset]
        obs_n, reset_info = env.reset(episode_idx=episode_idx)
        del reset_info
        controller.reset()

        history = None
        grid_history = None
        if record_history:
            history = init_episode_record(
                n_agents=env.n,
                init_soc=float(env.init_soc),
                reward_metas=reward_metas,
            )
            grid_history = init_grid_record(env.n)

        done = False
        episode_reward = 0.0
        while not done:
            action_n = controller.act(obs_n, deterministic=deterministic)
            obs_n, r_n, terminated_n, truncated_n, info = env.step(action_n)

            step_total = float(np.sum(np.asarray(r_n, dtype=np.float32)))
            episode_reward += step_total

            if history is not None:
                append_step_record(history, info, step_total=step_total, reward_metas=reward_metas)
            if grid_history is not None:
                append_grid_step_record(grid_history, info)

            done = bool(
                info.get("episode_done", False)
                or np.all(np.logical_or(np.asarray(terminated_n), np.asarray(truncated_n)))
            )

        episode_rewards.append(float(episode_reward))
        if history is not None:
            histories.append(history)
        if grid_history is not None:
            grid_histories.append(grid_history)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    results = {
        "episode_rewards": episode_rewards,
        "mean_episode_reward": mean_reward,
        "n_episodes": int(n_episodes),
    }
    if record_history:
        results["histories"] = histories
        results["grid_histories"] = grid_histories
    return results
