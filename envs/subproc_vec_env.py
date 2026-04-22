from __future__ import annotations

import multiprocessing as mp
import traceback
import warnings
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener

import numpy as np
import torch

from scripts.utils.nested import stack_nested
from scripts.utils.torch_runtime import configure_torch_runtime, derive_worker_seed

_WORKER_READY = "worker_ready"
_WORKER_INIT_ERROR = "worker_init_error"
_WORKER_SET_NEXT_EPISODE = "set_next_episode_idx"


def validate_parallel_episode_sampling_mode(mode: str) -> str:
    resolved = str(mode).strip().lower()
    if resolved != "unique_active":
        raise ValueError(f"train.parallel_episode_sampling only supports 'unique_active', got {mode!r}.")
    return resolved


def validate_wave_done_flags(done_flags: list[bool], *, env_name: str) -> None:
    done_count = int(sum(bool(flag) for flag in done_flags))
    if done_count not in {0, len(done_flags)}:
        raise RuntimeError(f"{env_name} with parallel_episode_sampling='unique_active' requires synchronized episode boundaries, but got {done_count}/{len(done_flags)} envs done in one step.")


@dataclass
class ParallelEpisodeSampler:
    num_available_episodes: int
    base_seed: int | None
    num_envs: int

    def __post_init__(self) -> None:
        self.num_available_episodes, self.num_envs = int(self.num_available_episodes), int(self.num_envs)
        if self.num_available_episodes <= 0 or self.num_envs <= 0:
            raise ValueError("ParallelEpisodeSampler requires num_available_episodes > 0 and num_envs > 0.")
        self._rng = np.random.default_rng(None if self.base_seed is None else int(self.base_seed))

    def next_wave(self) -> list[int]:
        indices: list[int] = []
        while len(indices) < self.num_envs:
            remaining = self.num_envs - len(indices)
            indices.extend(int(value) for value in self._rng.permutation(self.num_available_episodes)[:remaining])
        return indices


def split_batched_actions(actions_n_batched, env_idx: int, num_agents: int):
    return [np.asarray(actions_n_batched[agent_id][env_idx], dtype=np.float32) for agent_id in range(int(num_agents))]


def stack_step_outputs(obs_list, reward_list, terminated_list, truncated_list, info_list):
    return (
        stack_nested(obs_list),
        np.stack(reward_list, axis=0),
        np.stack(terminated_list, axis=0),
        np.stack(truncated_list, axis=0),
        info_list,
    )


def _episode_done(info, terminated, truncated) -> bool:
    return bool(info.get("episode_done", False) or np.all(np.logical_or(np.asarray(terminated), np.asarray(truncated))))


def _reset_env(env, *, episode_idx: int | None = None, seed: int | None = None):
    kwargs = {name: int(value) for name, value in {"episode_idx": episode_idx, "seed": seed}.items() if value is not None}
    return env.reset(**kwargs)


def _episode_indices(sampler: ParallelEpisodeSampler | None, *, num_envs: int, env_name: str) -> list[int]:
    if sampler is None:
        raise RuntimeError(f"{env_name} episode sampler is not initialized.")
    wave = sampler.next_wave()
    if len(wave) != int(num_envs):
        raise RuntimeError(f"{env_name} episode sampler returned {len(wave)} indices for num_envs={num_envs}.")
    return [int(index) for index in wave]


def _subproc_worker(address, authkey: bytes, cfg, mode: str, worker_rank: int, seed: int | None, dataset) -> None:
    from copy import deepcopy
    from scripts.builder import build_env

    remote = Client(address, family="AF_INET", authkey=authkey)
    env = None
    next_episode_idx: int | None = None
    validate_parallel_episode_sampling_mode(getattr(getattr(cfg, "train", None), "parallel_episode_sampling", "unique_active"))
    try:
        warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*", category=FutureWarning)
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except (AttributeError, RuntimeError):
            pass
        worker_cfg = deepcopy(cfg)
        worker_cfg.runtime.device = torch.device("cpu")
        worker_cfg.runtime.require_cuda = False
        runtime_state = configure_torch_runtime(worker_cfg, device="cpu", require_cuda=False, seed=seed, worker_rank=worker_rank)
        if runtime_state.seed is not None:
            worker_cfg.runtime.seed = int(runtime_state.seed)
        env = build_env(worker_cfg, mode=mode, dataset=dataset)
        remote.send((_WORKER_READY, {"num_agents": int(env.n), "num_available_episodes": int(getattr(env, "num_available_episodes", 0)), "episode_length": int(getattr(env, "episode_length", 0))}))
    except Exception:
        try:
            remote.send((_WORKER_INIT_ERROR, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        remote.close()
        return
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "reset":
                remote.send(_reset_env(env, episode_idx=payload))
                continue
            if cmd == _WORKER_SET_NEXT_EPISODE:
                next_episode_idx = None if payload is None else int(payload)
                remote.send(True)
                continue
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(payload)
                info = dict(info)
                if _episode_done(info, terminated, truncated):
                    if next_episode_idx is None:
                        raise RuntimeError("SubprocVecEnv worker reached episode end without a primed next_episode_idx in parallel_episode_sampling='unique_active' mode.")
                    obs, info["reset_info"] = _reset_env(env, episode_idx=next_episode_idx)
                    next_episode_idx = None
                    info["episode_done"] = True
                else:
                    info["episode_done"] = False
                remote.send((obs, np.asarray(reward, dtype=np.float32).reshape(env.n, 1), np.asarray(terminated, dtype=np.float32).reshape(env.n, 1), np.asarray(truncated, dtype=np.float32).reshape(env.n, 1), info))
                continue
            if cmd == "close":
                remote.close()
                break
            raise ValueError(f"Unknown worker command '{cmd}'.")
    except EOFError:
        pass
    finally:
        if env is not None:
            env.close()


class DummyVecEnv:
    def __init__(self, num_envs, env_fn_or_cls, cfg=None, mode: str = "train", seed: int | None = None, parallel_episode_sampling: str = "unique_active"):
        self.num_envs = int(num_envs)
        self.parallel_episode_sampling = validate_parallel_episode_sampling_mode(parallel_episode_sampling)
        self.envs = [env_fn_or_cls() if cfg is None else env_fn_or_cls(cfg, mode=mode) for _ in range(self.num_envs)]
        self.num_agents = int(self.envs[0].n if cfg is None else cfg.env.num_agents)
        runtime_seed = getattr(getattr(self.envs[0], "cfg", None), "runtime", None)
        runtime_seed = getattr(runtime_seed, "seed", None)
        self.base_seed = int(seed) if seed is not None else None if runtime_seed is None else int(runtime_seed)
        self._seeded_envs = [False] * self.num_envs
        self._next_wave_indices: list[int] = []
        self._episode_sampler = ParallelEpisodeSampler(
            num_available_episodes=int(getattr(self.envs[0], "num_available_episodes", 0)),
            base_seed=self.base_seed,
            num_envs=self.num_envs,
        )
        first = self.envs[0]
        for env in self.envs[1:]:
            if int(getattr(env, "num_available_episodes", 0)) != int(getattr(first, "num_available_episodes", 0)):
                raise RuntimeError("DummyVecEnv requires matching num_available_episodes across envs in parallel_episode_sampling='unique_active' mode.")
            if int(getattr(env, "episode_length", 0)) != int(getattr(first, "episode_length", 0)):
                raise RuntimeError("DummyVecEnv requires matching episode_length across envs in parallel_episode_sampling='unique_active' mode.")

    def _seed_once(self, env_idx: int) -> int | None:
        if self._seeded_envs[env_idx]:
            return None
        self._seeded_envs[env_idx] = True
        return None if self.base_seed is None else derive_worker_seed(self.base_seed, env_idx)

    def _prime_next_wave(self) -> None:
        self._next_wave_indices = [int(index) for index in _episode_indices(self._episode_sampler, num_envs=self.num_envs, env_name="DummyVecEnv")]

    def reset(self):
        episode_indices = _episode_indices(self._episode_sampler, num_envs=self.num_envs, env_name="DummyVecEnv")
        results = [_reset_env(env, episode_idx=episode_indices[idx], seed=self._seed_once(idx)) for idx, env in enumerate(self.envs)]
        obs_list, info_list = zip(*results)
        self._prime_next_wave()
        return stack_nested(list(obs_list)), list(info_list)

    def step(self, actions_n_batched):
        outputs = [env.step(split_batched_actions(actions_n_batched, env_idx, self.num_agents)) for env_idx, env in enumerate(self.envs)]
        obs_list, reward_list, terminated_list, truncated_list, info_list = [], [], [], [], []
        done_flags = []
        for obs, reward, terminated, truncated, info in outputs:
            obs_list.append(obs)
            reward_list.append(np.asarray(reward, dtype=np.float32).reshape(self.num_agents, 1))
            terminated_list.append(np.asarray(terminated, dtype=np.float32).reshape(self.num_agents, 1))
            truncated_list.append(np.asarray(truncated, dtype=np.float32).reshape(self.num_agents, 1))
            info = dict(info)
            info_list.append(info)
            done_flags.append(_episode_done(info, terminated, truncated))
        validate_wave_done_flags(done_flags, env_name="DummyVecEnv")
        if any(done_flags):
            for env_idx, env in enumerate(self.envs):
                obs_list[env_idx], info_list[env_idx]["reset_info"] = _reset_env(env, episode_idx=self._next_wave_indices[env_idx])
                info_list[env_idx]["episode_done"] = True
            self._prime_next_wave()
        else:
            for info in info_list:
                info["episode_done"] = False
        return stack_step_outputs(obs_list, reward_list, terminated_list, truncated_list, info_list)

    def close(self):
        for env in self.envs:
            env.close()


class SubprocVecEnv:
    def __init__(self, num_envs: int, cfg, mode: str = "train", seed: int | None = None):
        from data.loaders.registry import build_dataset

        self.num_envs = int(num_envs)
        self.parallel_episode_sampling = validate_parallel_episode_sampling_mode(getattr(getattr(cfg, "train", None), "parallel_episode_sampling", "unique_active"))
        runtime_seed = getattr(getattr(cfg, "runtime", None), "seed", None)
        self.base_seed = int(seed) if seed is not None else None if runtime_seed is None else int(runtime_seed)
        self.closed = False
        self.authkey = b"madrl_subproc_vec_env"
        self._episode_sampler: ParallelEpisodeSampler | None = None
        self._next_wave_indices: list[int] = []
        self.remotes, self.processes = [], []
        self.num_agents = self.num_available_episodes = self.episode_length = 0
        try:
            dataset = build_dataset(cfg, mode=mode)
        except FileNotFoundError as exc:
            raise RuntimeError(f"SubprocVecEnv worker 0 failed during initialization:\n{exc}") from exc
        ctx = mp.get_context("spawn")
        try:
            for worker_rank in range(self.num_envs):
                listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=self.authkey)
                try:
                    process = ctx.Process(target=_subproc_worker, args=(listener.address, self.authkey, cfg, mode, worker_rank, seed, dataset), daemon=True)
                    process.start()
                    remote = listener.accept()
                finally:
                    listener.close()
                self.remotes.append(remote)
                self.processes.append(process)
            for worker_rank, (remote, process) in enumerate(zip(self.remotes, self.processes)):
                try:
                    status, payload = remote.recv()
                except (ConnectionAbortedError, BrokenPipeError, EOFError, OSError) as exc:
                    raise RuntimeError(f"SubprocVecEnv worker {worker_rank} exited before initialization completed (exitcode={process.exitcode}).") from exc
                if status == _WORKER_INIT_ERROR:
                    raise RuntimeError(f"SubprocVecEnv worker {worker_rank} failed during initialization:\n{payload}")
                if status != _WORKER_READY:
                    raise RuntimeError(f"SubprocVecEnv worker {worker_rank} sent unexpected init status {status!r}.")
                meta = {key: int(payload[key]) for key in ("num_agents", "num_available_episodes", "episode_length")}
                if worker_rank == 0:
                    self.num_agents, self.num_available_episodes, self.episode_length = (meta["num_agents"], meta["num_available_episodes"], meta["episode_length"])
                    continue
                for key, expected in {"num_agents": self.num_agents, "num_available_episodes": self.num_available_episodes, "episode_length": self.episode_length}.items():
                    if int(meta[key]) != int(expected):
                        raise RuntimeError(f"SubprocVecEnv workers reported inconsistent {key}: worker 0 -> {expected}, worker {worker_rank} -> {int(meta[key])}.")
            self._episode_sampler = ParallelEpisodeSampler(num_available_episodes=self.num_available_episodes, base_seed=self.base_seed, num_envs=self.num_envs)
        except Exception:
            self.close()
            raise

    def _prime_next_wave(self) -> None:
        self._next_wave_indices = [int(index) for index in _episode_indices(self._episode_sampler, num_envs=self.num_envs, env_name="SubprocVecEnv")]
        for env_idx, remote in enumerate(self.remotes):
            remote.send((_WORKER_SET_NEXT_EPISODE, self._next_wave_indices[env_idx]))
        for remote in self.remotes:
            remote.recv()

    def reset(self):
        episode_indices = _episode_indices(self._episode_sampler, num_envs=self.num_envs, env_name="SubprocVecEnv")
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", episode_indices[env_idx]))
        obs_list, info_list = zip(*(remote.recv() for remote in self.remotes))
        self._prime_next_wave()
        return stack_nested(list(obs_list)), list(info_list)

    def step(self, actions_n_batched):
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("step", split_batched_actions(actions_n_batched, env_idx, self.num_agents)))
        obs_list, reward_list, terminated_list, truncated_list, info_list = zip(*(remote.recv() for remote in self.remotes))
        done_flags = [_episode_done(info, terminated, truncated) for info, terminated, truncated in zip(info_list, terminated_list, truncated_list)]
        validate_wave_done_flags(done_flags, env_name="SubprocVecEnv")
        if any(done_flags):
            self._prime_next_wave()
        return stack_step_outputs(list(obs_list), list(reward_list), list(terminated_list), list(truncated_list), list(info_list))

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self.closed = True
