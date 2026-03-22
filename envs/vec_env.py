"""Single-process vectorized environment wrapper."""

from __future__ import annotations

import numpy as np

from scripts.utils.nested import stack_nested


def split_batched_actions(actions_n_batched, env_idx: int, num_agents: int):
    """Extract one environment's per-agent actions from a batched action list."""
    return [
        np.asarray(actions_n_batched[agent_id][env_idx], dtype=np.float32)
        for agent_id in range(int(num_agents))
    ]


def stack_step_outputs(
    obs_list,
    reward_list,
    terminated_list,
    truncated_list,
    info_list,
):
    """Stack vector-environment step outputs into batched arrays."""
    batched_obs = stack_nested(obs_list)
    batched_reward = np.stack(reward_list, axis=0)
    batched_terminated = np.stack(terminated_list, axis=0)
    batched_truncated = np.stack(truncated_list, axis=0)
    return batched_obs, batched_reward, batched_terminated, batched_truncated, info_list


class DummyVecEnv:
    """Run multiple env copies sequentially in one process."""

    def __init__(self, num_envs, env_fn_or_cls, cfg=None, mode: str = "train"):
        self.num_envs = int(num_envs)

        if cfg is None:
            self.envs = [env_fn_or_cls() for _ in range(self.num_envs)]
            self.num_agents = self.envs[0].n
        else:
            self.envs = [env_fn_or_cls(cfg, mode=mode) for _ in range(self.num_envs)]
            self.num_agents = cfg.env.num_agents

    def reset(self):
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return stack_nested(obs_list), info_list

    def step(self, actions_n_batched):
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []

        for env_idx, env in enumerate(self.envs):
            action_n = split_batched_actions(actions_n_batched, env_idx, self.num_agents)
            obs, reward, terminated, truncated, info = env.step(action_n)

            episode_done = bool(
                info.get("episode_done", False)
                or np.all(np.logical_or(np.asarray(terminated), np.asarray(truncated)))
            )
            if episode_done:
                reset_obs, reset_info = env.reset()
                info = dict(info)
                info["episode_done"] = True
                info["reset_info"] = reset_info
                obs = reset_obs
            else:
                info = dict(info)
                info["episode_done"] = False

            obs_list.append(obs)
            reward_list.append(np.asarray(reward, dtype=np.float32).reshape(self.num_agents, 1))
            terminated_list.append(
                np.asarray(terminated, dtype=np.float32).reshape(self.num_agents, 1)
            )
            truncated_list.append(
                np.asarray(truncated, dtype=np.float32).reshape(self.num_agents, 1)
            )
            info_list.append(info)

        return stack_step_outputs(
            obs_list,
            reward_list,
            terminated_list,
            truncated_list,
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()
