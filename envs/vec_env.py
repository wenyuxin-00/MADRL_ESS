"""向量化环境封装（DummyVecEnv）。

在单进程中串行运行多个环境副本，提供统一的批量接口。

主要类:
    DummyVecEnv -- 单进程向量化环境
"""

from __future__ import annotations

import numpy as np

from scripts.utils.nested import stack_nested


def split_batched_actions(actions_n_batched, env_idx: int, num_agents: int):
    """Extract one env's per-agent action list from an agent-major batch."""
    return [
        np.asarray(actions_n_batched[agent_id][env_idx], dtype=np.float32)
        for agent_id in range(int(num_agents))
    ]


def stack_step_outputs(obs_list, reward_list, done_list, info_list):
    """Pack per-env step outputs back into canonical batched arrays."""
    batched_obs = stack_nested(obs_list)
    batched_reward = np.stack(reward_list, axis=0)
    batched_done = np.stack(done_list, axis=0)
    return batched_obs, batched_reward, batched_done, info_list


class DummyVecEnv:
    """Single-process vectorized env for canonical structured observations."""

    def __init__(self, num_envs, env_fn_or_cls, cfg=None, mode: str = "train"):
        self.num_envs = int(num_envs)

        if cfg is None:
            self.envs = [env_fn_or_cls() for _ in range(self.num_envs)]
            self.num_agents = self.envs[0].n
        else:
            self.envs = [env_fn_or_cls(cfg, mode=mode) for _ in range(self.num_envs)]
            self.num_agents = cfg.env.num_agents

    def reset(self):
        obs_list = [env.reset() for env in self.envs]
        return stack_nested(obs_list)

    def step(self, actions_n_batched):
        obs_list, reward_list, done_list, info_list = [], [], [], []

        for env_idx, env in enumerate(self.envs):
            action_n = split_batched_actions(actions_n_batched, env_idx, self.num_agents)
            obs, reward, done, info = env.step(action_n)

            if all(done):
                obs = env.reset()
                info["episode_done"] = True
            else:
                info["episode_done"] = False

            obs_list.append(obs)
            reward_list.append(np.asarray(reward, dtype=np.float32).reshape(self.num_agents, 1))
            done_list.append(np.asarray(done, dtype=np.float32).reshape(self.num_agents, 1))
            info_list.append(info)

        return stack_step_outputs(obs_list, reward_list, done_list, info_list)

    def close(self):
        for env in self.envs:
            env.close()
