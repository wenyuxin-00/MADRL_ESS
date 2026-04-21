from __future__ import annotations

import numpy as np
import pytest

from envs.parallel_episode_sampling import ParallelEpisodeSampler, validate_wave_done_flags
from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_env import DummyVecEnv
from scripts.builder import _build_train_vec_env
from scripts.mainline_madrl import _apply_train_controls
from tests.support.helpers import make_case_dir, make_smoke_config


class _PartialDoneEnv:
    def __init__(
        self,
        *,
        done_after: int,
        episode_length: int = 2,
        num_available_episodes: int = 8,
    ) -> None:
        self.n = 1
        self.done_after = int(done_after)
        self.num_available_episodes = int(num_available_episodes)
        self.episode_length = int(episode_length)
        self.step_count = 0

    def reset(self, *, episode_idx: int | None = None, seed: int | None = None):
        _ = seed
        self.step_count = 0
        obs = {"local": np.zeros((1, 1), dtype=np.float32)}
        info = {"episode_idx": 0 if episode_idx is None else int(episode_idx)}
        return obs, info

    def step(self, _action_n):
        self.step_count += 1
        done = self.step_count >= self.done_after
        obs = {"local": np.full((1, 1), self.step_count, dtype=np.float32)}
        reward = [0.0]
        terminated = [False]
        truncated = [done]
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        return None


def test_apply_train_controls_sets_parallel_episode_sampling(tmp_path):
    cfg = make_smoke_config(tmp_path, algorithm="MADDPG")

    _apply_train_controls(cfg, {"parallel_episode_sampling": "per_env_rng"})

    assert cfg.train.parallel_episode_sampling == "per_env_rng"


def test_parallel_episode_sampler_prefers_unique_indices():
    sampler = ParallelEpisodeSampler(num_available_episodes=5, base_seed=7, num_envs=4)

    wave = sampler.next_wave()

    assert len(wave) == 4
    assert len(set(wave)) == 4


def test_parallel_episode_sampler_allows_minimal_duplicates_when_needed():
    sampler = ParallelEpisodeSampler(num_available_episodes=2, base_seed=11, num_envs=5)

    wave = sampler.next_wave()

    assert len(wave) == 5
    assert set(wave) == {0, 1}


def test_validate_wave_done_flags_rejects_partial_done():
    with pytest.raises(RuntimeError, match="requires synchronized episode boundaries"):
        validate_wave_done_flags([True, False], env_name="DummyVecEnv")


def test_dummy_vec_env_unique_active_reset_assigns_distinct_episode_indices(tmp_path):
    case_dir = make_case_dir(tmp_path, "dummy_unique_active")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.train.num_envs = 4
    cfg.train.parallel_episode_sampling = "unique_active"

    vec_env = _build_train_vec_env(cfg, seed=7)
    try:
        _, reset_infos = vec_env.reset()
        episode_indices = [int(info["episode_idx"]) for info in reset_infos]

        assert len(episode_indices) == 4
        assert len(set(episode_indices)) == 4
        assert vec_env._next_wave_indices is not None
        assert len(vec_env._next_wave_indices) == 4
        assert len(set(vec_env._next_wave_indices)) == 4
    finally:
        vec_env.close()


def test_dummy_vec_env_per_env_rng_sequences_are_not_fully_synchronized(tmp_path):
    case_dir = make_case_dir(tmp_path, "dummy_per_env_rng")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.train.num_envs = 4
    cfg.train.parallel_episode_sampling = "per_env_rng"

    vec_env = _build_train_vec_env(cfg, seed=13)
    try:
        sequences = [[] for _ in range(vec_env.num_envs)]
        for _ in range(4):
            _, reset_infos = vec_env.reset()
            for env_idx, info in enumerate(reset_infos):
                sequences[env_idx].append(int(info["episode_idx"]))

        assert len({tuple(sequence) for sequence in sequences}) > 1
    finally:
        vec_env.close()


def test_dummy_vec_env_unique_active_rejects_partial_done():
    done_after_values = iter([1, 2])
    vec_env = DummyVecEnv(
        2,
        lambda: _PartialDoneEnv(done_after=next(done_after_values), episode_length=2),
        parallel_episode_sampling="unique_active",
    )

    try:
        vec_env.reset()
        action_list = [np.zeros((2, 1), dtype=np.float32)]
        with pytest.raises(RuntimeError, match="requires synchronized episode boundaries"):
            vec_env.step(action_list)
    finally:
        vec_env.close()


def test_subproc_vec_env_unique_active_reset_assigns_distinct_episode_indices(tmp_path):
    case_dir = make_case_dir(tmp_path, "subproc_unique_active")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.train.num_envs = 4
    cfg.train.vec_env_type = "subproc"
    cfg.train.parallel_episode_sampling = "unique_active"

    vec_env = SubprocVecEnv(4, cfg, mode="train", seed=17)
    try:
        _, reset_infos = vec_env.reset()
        episode_indices = [int(info["episode_idx"]) for info in reset_infos]

        assert len(set(episode_indices)) == 4
        assert vec_env.num_available_episodes == 4
        assert vec_env.episode_length == cfg.env.episode_limit
        assert vec_env._next_wave_indices is not None
        assert len(set(vec_env._next_wave_indices)) == 4
    finally:
        vec_env.close()


def test_subproc_vec_env_per_env_rng_sequences_are_not_fully_synchronized(tmp_path):
    case_dir = make_case_dir(tmp_path, "subproc_per_env_rng")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.train.num_envs = 2
    cfg.train.vec_env_type = "subproc"
    cfg.train.parallel_episode_sampling = "per_env_rng"

    vec_env = SubprocVecEnv(2, cfg, mode="train", seed=19)
    try:
        sequences = [[] for _ in range(vec_env.num_envs)]
        for _ in range(4):
            _, reset_infos = vec_env.reset()
            for env_idx, info in enumerate(reset_infos):
                sequences[env_idx].append(int(info["episode_idx"]))

        assert len({tuple(sequence) for sequence in sequences}) > 1
    finally:
        vec_env.close()
