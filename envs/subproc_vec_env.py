"""Multi-process vectorized environment wrapper."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Client, Listener

import numpy as np
import torch

from envs.vec_env import split_batched_actions, stack_step_outputs
from scripts.utils.nested import stack_nested
from scripts.utils.torch_runtime import configure_torch_runtime


def _make_worker_env(cfg, mode: str, *, worker_rank: int, seed: int | None):
    """Build one worker env with isolated config/runtime state."""
    from copy import deepcopy

    from scripts.builder import build_env

    worker_cfg = deepcopy(cfg)
    worker_cfg.runtime.device = torch.device("cpu")
    worker_cfg.runtime.require_cuda = False
    configure_torch_runtime(
        worker_cfg,
        device="cpu",
        require_cuda=False,
        seed=seed,
        worker_rank=worker_rank,
    )
    return build_env(worker_cfg, mode=mode)


def _subproc_worker(address, authkey: bytes, cfg, mode: str, worker_rank: int, seed: int | None) -> None:
    """Child-process event loop for one environment worker."""
    remote = Client(address, family="AF_INET", authkey=authkey)
    env = _make_worker_env(cfg, mode=mode, worker_rank=worker_rank, seed=seed)

    try:
        while True:
            cmd, payload = remote.recv()

            if cmd == "reset":
                obs, info = env.reset(episode_idx=payload)
                remote.send((obs, info))
                continue

            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(payload)
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
        env.close()


class SubprocVecEnv:
    """Run multiple env copies in parallel subprocesses."""

    def __init__(self, num_envs: int, cfg, mode: str = "train", seed: int | None = None):
        self.num_envs = int(num_envs)
        self.closed = False
        self.authkey = b"madrl_subproc_vec_env"

        ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []

        for worker_rank in range(self.num_envs):
            listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=self.authkey)
            process = ctx.Process(
                target=_subproc_worker,
                args=(listener.address, self.authkey, cfg, mode, worker_rank, seed),
                daemon=True,
            )
            process.start()
            remote = listener.accept()
            listener.close()
            self.remotes.append(remote)
            self.processes.append(process)

        self.remotes[0].send(("get_num_agents", None))
        self.num_agents = int(self.remotes[0].recv())

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs_list, info_list = zip(*results)
        return stack_nested(list(obs_list)), list(info_list)

    def step(self, actions_n_batched):
        for env_idx, remote in enumerate(self.remotes):
            action_n = split_batched_actions(actions_n_batched, env_idx, self.num_agents)
            remote.send(("step", action_n))

        results = [remote.recv() for remote in self.remotes]
        obs_list, reward_list, terminated_list, truncated_list, info_list = zip(*results)
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
