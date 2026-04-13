"""Multi-process vectorized environment wrapper."""

from __future__ import annotations

import multiprocessing as mp
import traceback
import warnings
from multiprocessing.connection import Client, Listener

import numpy as np
import torch

from envs.parallel_episode_sampling import (
    ParallelEpisodeSampler,
    validate_parallel_episode_sampling_mode,
    validate_wave_done_flags,
)
from envs.vec_env import split_batched_actions, stack_step_outputs
from scripts.utils.nested import stack_nested
from scripts.utils.torch_runtime import configure_torch_runtime


_WORKER_READY = "worker_ready"
_WORKER_INIT_ERROR = "worker_init_error"
_WORKER_SET_NEXT_EPISODE = "set_next_episode_idx"


def _make_worker_env(cfg, mode: str, *, worker_rank: int, seed: int | None):
    """Build one worker env with isolated config/runtime state."""
    from copy import deepcopy

    from scripts.builder import build_env

    worker_cfg = deepcopy(cfg)
    worker_cfg.runtime.device = torch.device("cpu")
    worker_cfg.runtime.require_cuda = False
    runtime_state = configure_torch_runtime(
        worker_cfg,
        device="cpu",
        require_cuda=False,
        seed=seed,
        worker_rank=worker_rank,
    )
    if runtime_state.seed is not None:
        worker_cfg.runtime.seed = int(runtime_state.seed)
    return build_env(worker_cfg, mode=mode)


def _subproc_worker(address, authkey: bytes, cfg, mode: str, worker_rank: int, seed: int | None) -> None:
    """Child-process event loop for one environment worker."""
    remote = Client(address, family="AF_INET", authkey=authkey)
    env = None
    next_episode_idx: int | None = None
    parallel_episode_sampling = validate_parallel_episode_sampling_mode(
        getattr(getattr(cfg, "train", None), "parallel_episode_sampling", "unique_active")
    )

    try:
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except (AttributeError, RuntimeError):
            pass
        env = _make_worker_env(cfg, mode=mode, worker_rank=worker_rank, seed=seed)
        remote.send(
            (
                _WORKER_READY,
                {
                    "num_agents": int(env.n),
                    "num_available_episodes": int(getattr(env, "num_available_episodes", 0)),
                    "episode_length": int(getattr(env, "episode_length", 0)),
                },
            )
        )
    except Exception:
        try:
            remote.send((_WORKER_INIT_ERROR, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            remote.close()
        return

    try:
        while True:
            cmd, payload = remote.recv()

            if cmd == "reset":
                obs, info = env.reset(episode_idx=payload)
                remote.send((obs, info))
                continue

            if cmd == _WORKER_SET_NEXT_EPISODE:
                next_episode_idx = None if payload is None else int(payload)
                remote.send(True)
                continue

            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(payload)
                episode_done = bool(
                    info.get("episode_done", False)
                    or np.all(np.logical_or(np.asarray(terminated), np.asarray(truncated)))
                )
                if episode_done:
                    if parallel_episode_sampling == "unique_active" and next_episode_idx is None:
                        raise RuntimeError(
                            "SubprocVecEnv worker reached episode end without a primed next_episode_idx "
                            "in parallel_episode_sampling='unique_active' mode."
                        )
                    reset_obs, reset_info = (
                        env.reset(episode_idx=next_episode_idx)
                        if next_episode_idx is not None
                        else env.reset()
                    )
                    next_episode_idx = None
                    info = dict(info)
                    info["episode_done"] = True
                    info["reset_info"] = reset_info
                    obs = reset_obs
                else:
                    info = dict(info)
                    info["episode_done"] = False

                remote.send(
                    (
                        obs,
                        np.asarray(reward, dtype=np.float32).reshape(env.n, 1),
                        np.asarray(terminated, dtype=np.float32).reshape(env.n, 1),
                        np.asarray(truncated, dtype=np.float32).reshape(env.n, 1),
                        info,
                    )
                )
                continue

            if cmd == "get_num_agents":
                remote.send(int(env.n))
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


def _recv_worker_ready(remote, process, worker_rank: int) -> dict[str, int]:
    try:
        status, payload = remote.recv()
    except (ConnectionAbortedError, BrokenPipeError, EOFError, OSError) as exc:
        raise RuntimeError(
            "SubprocVecEnv worker "
            f"{worker_rank} exited before initialization completed (exitcode={process.exitcode})."
        ) from exc

    if status == _WORKER_READY:
        return {
            "num_agents": int(payload["num_agents"]),
            "num_available_episodes": int(payload["num_available_episodes"]),
            "episode_length": int(payload["episode_length"]),
        }

    if status == _WORKER_INIT_ERROR:
        raise RuntimeError(
            f"SubprocVecEnv worker {worker_rank} failed during initialization:\n{payload}"
        )

    raise RuntimeError(
        f"SubprocVecEnv worker {worker_rank} sent unexpected init status {status!r}."
    )


class SubprocVecEnv:
    """Run multiple env copies in parallel subprocesses."""

    def __init__(self, num_envs: int, cfg, mode: str = "train", seed: int | None = None):
        self.num_envs = int(num_envs)
        self.parallel_episode_sampling = validate_parallel_episode_sampling_mode(
            getattr(getattr(cfg, "train", None), "parallel_episode_sampling", "unique_active")
        )
        runtime = getattr(getattr(cfg, "runtime", None), "seed", None)
        self.base_seed = int(seed) if seed is not None else (None if runtime is None else int(runtime))
        self.closed = False
        self.authkey = b"madrl_subproc_vec_env"
        self._episode_sampler: ParallelEpisodeSampler | None = None
        self._next_wave_indices: list[int] | None = None

        ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        self.num_agents = 0
        self.num_available_episodes = 0
        self.episode_length = 0

        try:
            for worker_rank in range(self.num_envs):
                listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=self.authkey)
                try:
                    process = ctx.Process(
                        target=_subproc_worker,
                        args=(listener.address, self.authkey, cfg, mode, worker_rank, seed),
                        daemon=True,
                    )
                    process.start()
                    remote = listener.accept()
                finally:
                    listener.close()

                self.remotes.append(remote)
                self.processes.append(process)

            for worker_rank, (remote, process) in enumerate(zip(self.remotes, self.processes)):
                worker_meta = _recv_worker_ready(remote, process, worker_rank)
                worker_num_agents = worker_meta["num_agents"]
                if worker_rank == 0:
                    self.num_agents = worker_num_agents
                    self.num_available_episodes = int(worker_meta["num_available_episodes"])
                    self.episode_length = int(worker_meta["episode_length"])
                elif worker_num_agents != self.num_agents:
                    raise RuntimeError(
                        "SubprocVecEnv workers reported inconsistent agent counts: "
                        f"worker 0 -> {self.num_agents}, worker {worker_rank} -> {worker_num_agents}."
                    )
                elif int(worker_meta["num_available_episodes"]) != self.num_available_episodes:
                    raise RuntimeError(
                        "SubprocVecEnv workers reported inconsistent num_available_episodes: "
                        f"worker 0 -> {self.num_available_episodes}, "
                        f"worker {worker_rank} -> {int(worker_meta['num_available_episodes'])}."
                    )
                elif int(worker_meta["episode_length"]) != self.episode_length:
                    raise RuntimeError(
                        "SubprocVecEnv workers reported inconsistent episode_length: "
                        f"worker 0 -> {self.episode_length}, "
                        f"worker {worker_rank} -> {int(worker_meta['episode_length'])}."
                    )
            if self.parallel_episode_sampling == "unique_active":
                self._episode_sampler = ParallelEpisodeSampler(
                    num_available_episodes=self.num_available_episodes,
                    base_seed=self.base_seed,
                    num_envs=self.num_envs,
                )
        except Exception:
            self.close()
            raise

    def _set_next_wave_indices(self) -> None:
        if self.parallel_episode_sampling != "unique_active":
            self._next_wave_indices = None
            return
        if self._episode_sampler is None:
            raise RuntimeError("SubprocVecEnv episode sampler is not initialized.")
        self._next_wave_indices = self._episode_sampler.next_wave()
        for env_idx, remote in enumerate(self.remotes):
            remote.send((_WORKER_SET_NEXT_EPISODE, int(self._next_wave_indices[env_idx])))
        for remote in self.remotes:
            remote.recv()

    def reset(self):
        if self.parallel_episode_sampling == "unique_active":
            if self._episode_sampler is None:
                raise RuntimeError("SubprocVecEnv episode sampler is not initialized.")
            episode_indices = self._episode_sampler.next_wave()
        else:
            episode_indices = [None] * self.num_envs
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", episode_indices[env_idx]))
        results = [remote.recv() for remote in self.remotes]
        obs_list, info_list = zip(*results)
        if self.parallel_episode_sampling == "unique_active":
            self._set_next_wave_indices()
        else:
            self._next_wave_indices = None
        return stack_nested(list(obs_list)), list(info_list)

    def step(self, actions_n_batched):
        for env_idx, remote in enumerate(self.remotes):
            action_n = split_batched_actions(actions_n_batched, env_idx, self.num_agents)
            remote.send(("step", action_n))

        results = [remote.recv() for remote in self.remotes]
        obs_list, reward_list, terminated_list, truncated_list, info_list = zip(*results)
        done_flags = [
            bool(
                info.get("episode_done", False)
                or np.all(
                    np.logical_or(np.asarray(terminated), np.asarray(truncated))
                )
            )
            for info, terminated, truncated in zip(info_list, terminated_list, truncated_list)
        ]
        if self.parallel_episode_sampling == "unique_active":
            validate_wave_done_flags(done_flags, env_name="SubprocVecEnv")
            if any(done_flags):
                self._set_next_wave_indices()
        return stack_step_outputs(
            list(obs_list),
            list(reward_list),
            list(terminated_list),
            list(truncated_list),
            list(info_list),
        )

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
