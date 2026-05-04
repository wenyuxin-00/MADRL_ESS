from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import numpy as np


def _stack_obs(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([np.asarray(obs[key], dtype=np.float32) for obs in items]).astype(np.float32) for key in items[0]}


def validate_parallel_episode_sampling_mode(mode: str) -> str:
    value = str(mode).strip()
    if value != "unique_active":
        raise ValueError(f"parallel_episode_sampling expected 'unique_active', got {mode!r}.")
    return value


@dataclass
class ParallelEpisodeSampler:
    num_available_episodes: int
    base_seed: int
    num_envs: int

    def __post_init__(self) -> None:
        self.num_available_episodes = int(self.num_available_episodes); self.num_envs = int(self.num_envs); self._rng = np.random.default_rng(int(self.base_seed))
        if self.num_available_episodes <= 0 or self.num_envs <= 0:
            raise ValueError("ParallelEpisodeSampler expected positive episode and env counts.")

    def next_wave(self) -> list[int]:
        indices: list[int] = []
        while len(indices) < self.num_envs:
            remaining = self.num_envs - len(indices)
            indices.extend(int(value) for value in self._rng.permutation(self.num_available_episodes)[:remaining])
        return indices


class SyncVecEnv:
    def __init__(self, num_envs: int, env_factory: Callable[[], Any], *, seed: int, parallel_episode_sampling: str = "unique_active") -> None:
        validate_parallel_episode_sampling_mode(parallel_episode_sampling)
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError(f"SyncVecEnv expected num_envs > 0, got {num_envs}.")
        self.envs = [env_factory() for _ in range(self.num_envs)]
        self.num_available_episodes = int(self.envs[0].num_available_episodes); self.episode_length = int(self.envs[0].episode_length)
        for idx, env in enumerate(self.envs):
            if int(env.num_available_episodes) != self.num_available_episodes or int(env.episode_length) != self.episode_length:
                raise RuntimeError(f"SyncVecEnv env {idx} has inconsistent episode contract.")
        self.sampler = ParallelEpisodeSampler(self.num_available_episodes, int(seed), self.num_envs)

    def reset(self) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        obs, info = [], []
        for env, episode_idx in zip(self.envs, self.sampler.next_wave(), strict=True):
            env_obs, env_info = env.reset(episode_index=int(episode_idx)); obs.append(env_obs); info.append(env_info)
        return _stack_obs(obs), info

    def step(self, action_batch: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(action_batch, dtype=np.float32)
        if actions.shape[:2] != (self.num_envs, int(self.envs[0].n)):
            raise ValueError(f"SyncVecEnv action batch expected first dims {(self.num_envs, int(self.envs[0].n))}, got {actions.shape}.")
        obs, rewards, done_flags, infos = [], [], [], []
        for env, action in zip(self.envs, actions, strict=True):
            next_obs, reward, done, truncated, info = env.step(action)
            if bool(truncated):
                raise RuntimeError("SyncVecEnv does not support truncated episodes.")
            obs.append(next_obs); rewards.append(np.asarray(reward, dtype=np.float32)); done_flags.append(bool(done)); infos.append(dict(info))
        done_count = sum(done_flags)
        if done_count not in {0, self.num_envs}:
            raise RuntimeError(f"SyncVecEnv requires synchronized episode boundaries, got {done_count}/{self.num_envs} done.")
        if done_count == self.num_envs:
            reset_obs, reset_info = [], []
            for env, episode_idx, info in zip(self.envs, self.sampler.next_wave(), infos, strict=True):
                info["terminal_observation"] = obs[len(reset_obs)]
                env_obs, env_info = env.reset(episode_index=int(episode_idx)); reset_obs.append(env_obs); reset_info.append(env_info)
            obs = reset_obs
        terminated = np.asarray(done_flags, dtype=bool)
        return _stack_obs(obs), np.stack(rewards).astype(np.float32), terminated, np.zeros((self.num_envs,), dtype=bool), infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
