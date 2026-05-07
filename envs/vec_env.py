from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from typing import Any, Callable

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


def _subproc_worker(conn: Any, env_factory: Callable[[], Any]) -> None:
    env = None
    try:
        env = env_factory()
        conn.send(("ready", {"num_available_episodes": int(env.num_available_episodes), "episode_length": int(env.episode_length), "num_agents": int(env.n)}))
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send(("ok", env.reset(episode_index=int(payload))))
            elif command == "step":
                conn.send(("ok", env.step(payload)))
            elif command == "close":
                conn.send(("ok", None)); break
            else:
                raise RuntimeError(f"SubprocVecEnv worker received unknown command {command!r}.")
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class SubprocVecEnv:
    def __init__(self, num_envs: int, env_factory: Callable[[], Any], *, seed: int, parallel_episode_sampling: str = "unique_active") -> None:
        validate_parallel_episode_sampling_mode(parallel_episode_sampling)
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError(f"SubprocVecEnv expected num_envs > 0, got {num_envs}.")
        self._closed = False; self._ctx = mp.get_context("spawn")
        self.parents, self.processes = [], []
        for _ in range(self.num_envs):
            parent, child = self._ctx.Pipe()
            proc = self._ctx.Process(target=_subproc_worker, args=(child, env_factory), daemon=True)
            proc.start(); child.close(); self.parents.append(parent); self.processes.append(proc)
        try:
            specs = [self._recv(parent) for parent in self.parents]
        except BaseException:
            self.close()
            raise
        self.num_available_episodes = int(specs[0]["num_available_episodes"]); self.episode_length = int(specs[0]["episode_length"]); self.n = int(specs[0]["num_agents"])
        for idx, spec in enumerate(specs):
            if int(spec["num_available_episodes"]) != self.num_available_episodes or int(spec["episode_length"]) != self.episode_length or int(spec["num_agents"]) != self.n:
                raise RuntimeError(f"SubprocVecEnv worker {idx} has inconsistent episode contract.")
        self.sampler = ParallelEpisodeSampler(self.num_available_episodes, int(seed), self.num_envs)

    def _recv(self, parent: Any) -> Any:
        status, payload = parent.recv()
        if status == "error":
            raise RuntimeError(f"SubprocVecEnv worker failed:\n{payload}")
        return payload

    def reset(self) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        episodes = self.sampler.next_wave()
        for parent, episode_idx in zip(self.parents, episodes, strict=True):
            parent.send(("reset", int(episode_idx)))
        rows = [self._recv(parent) for parent in self.parents]
        obs, info = zip(*rows, strict=True)
        return _stack_obs(list(obs)), [dict(item) for item in info]

    def step(self, action_batch: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(action_batch, dtype=np.float32)
        if actions.shape[:2] != (self.num_envs, self.n):
            raise ValueError(f"SubprocVecEnv action batch expected first dims {(self.num_envs, self.n)}, got {actions.shape}.")
        for parent, action in zip(self.parents, actions, strict=True):
            parent.send(("step", action))
        rows = [self._recv(parent) for parent in self.parents]
        obs, rewards, done_flags, truncated_flags, infos = zip(*rows, strict=True)
        if any(bool(flag) for flag in truncated_flags):
            raise RuntimeError("SubprocVecEnv does not support truncated episodes.")
        done_flags = [bool(flag) for flag in done_flags]
        done_count = sum(done_flags)
        if done_count not in {0, self.num_envs}:
            raise RuntimeError(f"SubprocVecEnv requires synchronized episode boundaries, got {done_count}/{self.num_envs} done.")
        obs_list, infos_list = list(obs), [dict(info) for info in infos]
        if done_count == self.num_envs:
            terminal_obs = obs_list
            episodes = self.sampler.next_wave()
            for parent, episode_idx in zip(self.parents, episodes, strict=True):
                parent.send(("reset", int(episode_idx)))
            reset_rows = [self._recv(parent) for parent in self.parents]
            obs_list = []
            for idx, (env_obs, _) in enumerate(reset_rows):
                infos_list[idx]["terminal_observation"] = terminal_obs[idx]
                obs_list.append(env_obs)
        terminated = np.asarray(done_flags, dtype=bool)
        return _stack_obs(obs_list), np.stack(rewards).astype(np.float32), terminated, np.zeros((self.num_envs,), dtype=bool), infos_list

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for parent in self.parents:
            try:
                parent.send(("close", None)); self._recv(parent)
            except (BrokenPipeError, EOFError, RuntimeError):
                pass
            finally:
                parent.close()
        for proc in self.processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate(); proc.join(timeout=2.0)
