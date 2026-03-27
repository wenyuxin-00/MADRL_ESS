from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from scripts.builder import build_train_runner
from scripts.builder_fastlab import build_train_runner_fastlab
from scripts.utils.madrl_observation_cache_lab import build_or_load_observation_cache
from scripts.utils.torch_runtime import STRICT_REPRO_RUNTIME_MODE
from tests.support.helpers import make_smoke_config


def _make_train_equivalence_cfg(tmp_path):
    cfg = make_smoke_config(tmp_path, algorithm="MATD3")
    cfg.forecast.type = "perfect"
    cfg.train.num_envs = 1
    cfg.train.vec_env_type = "dummy"
    cfg.train.train_episodes = 2
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.train.batch_size = 2
    cfg.train.buffer_size = 32
    cfg.train.update_interval = 1
    cfg.train.updates_per_step = 1
    cfg.train.show_progress = False
    cfg.runtime.device = torch.device("cpu")
    cfg.runtime.execution_mode = STRICT_REPRO_RUNTIME_MODE
    cfg.runtime.enable_amp = False
    cfg.runtime.enable_compile = False
    return cfg


def _run_base_runner(cfg, *, seed: int):
    runner = build_train_runner(cfg, seed=seed, env_name="GridTrainMainlineExact", number=1)
    try:
        episodes = runner.run()
        return {
            "episodes": episodes,
            "total_steps": runner.total_steps,
            "episode_rewards": list(runner.episode_rewards),
            "reward_summary": runner.build_reward_summary(),
        }
    finally:
        runner.close()


def _run_cached_mainline_runner(cfg, *, seed: int, tmp_path):
    cache_result = build_or_load_observation_cache(
        cfg,
        split="train",
        forecast_ready=None,
        refresh=True,
        root=tmp_path / "cache",
    )
    cfg.runtime.fastlab_observation_cache_dir = str(cache_result.cache_dir)
    cfg.runtime.fastlab_train_info_mode = "minimal"
    cfg.runtime.fastlab_fast_grid_core = True

    runner = build_train_runner_fastlab(cfg, seed=seed, env_name="GridTrainMainlineCachedExact", number=1)
    try:
        episodes = runner.run()
        return {
            "episodes": episodes,
            "total_steps": runner.total_steps,
            "episode_rewards": list(runner.episode_rewards),
            "reward_summary": runner.build_reward_summary(),
        }
    finally:
        runner.close()


def test_cached_mainline_train_runner_matches_base_runner_with_fixed_seed(tmp_path) -> None:
    seed = 11
    base_cfg = _make_train_equivalence_cfg(tmp_path / "base")
    fast_cfg = deepcopy(base_cfg)

    base_result = _run_base_runner(base_cfg, seed=seed)
    cached_result = _run_cached_mainline_runner(fast_cfg, seed=seed, tmp_path=tmp_path)

    assert base_result["episodes"] == cached_result["episodes"]
    assert base_result["total_steps"] == cached_result["total_steps"]
    assert np.allclose(base_result["episode_rewards"], cached_result["episode_rewards"], atol=1e-6)

    base_summary = base_result["reward_summary"]
    fast_summary = cached_result["reward_summary"]
    assert base_summary["episodes"] == fast_summary["episodes"]
    assert np.allclose(base_summary["episode_total_reward"], fast_summary["episode_total_reward"], atol=1e-6)
    assert set(base_summary["components"]) == set(fast_summary["components"])
    for key in base_summary["components"]:
        assert np.allclose(
            base_summary["components"][key]["values"],
            fast_summary["components"][key]["values"],
            atol=1e-6,
        )
