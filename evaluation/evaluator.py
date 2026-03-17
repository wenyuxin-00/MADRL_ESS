"""统一 controller 评估入口。"""

from __future__ import annotations

import numpy as np

from evaluation.episode_recorder import append_step_record, init_episode_record


def evaluate_controller(
    env,
    controller,
    n_episodes: int = 1,
    deterministic: bool = True,
    episode_indices: list[int] | None = None,
    record_history: bool = True,
):
    """在单个环境实例上评估任意 controller。"""
    if episode_indices is not None:
        n_episodes = len(episode_indices)

    reward_metas = getattr(env.reward_fn, "component_meta", [])
    histories = []
    episode_rewards = []

    for episode_offset in range(int(n_episodes)):
        episode_idx = None if episode_indices is None else episode_indices[episode_offset]
        obs_n = env.reset(episode_idx=episode_idx)
        controller.reset()

        history = None
        if record_history:
            history = init_episode_record(
                n_agents=env.n,
                init_soc=float(env.init_soc),
                reward_metas=reward_metas,
            )

        done = False
        episode_reward = 0.0
        while not done:
            action_n = controller.act(obs_n, deterministic=deterministic)
            obs_n, r_n, done_n, info = env.step(action_n)

            step_total = float(np.sum(np.asarray(r_n, dtype=np.float32)))
            episode_reward += step_total

            if history is not None:
                append_step_record(history, info, step_total=step_total, reward_metas=reward_metas)

            done = bool(info.get("episode_done", False) or all(done_n))

        episode_rewards.append(float(episode_reward))
        if history is not None:
            histories.append(history)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    results = {
        "episode_rewards": episode_rewards,
        "mean_episode_reward": mean_reward,
        "n_episodes": int(n_episodes),
    }
    if record_history:
        results["histories"] = histories
    return results
