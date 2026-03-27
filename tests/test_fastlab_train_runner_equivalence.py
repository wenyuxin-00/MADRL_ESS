from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

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


def _run_cached_mainline_runner(cfg, *, seed: int, tmp_path, refresh_cache: bool):
    cache_result = build_or_load_observation_cache(
        cfg,
        split="train",
        forecast_ready=None,
        refresh=refresh_cache,
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
            "cache_hit": cache_result.cache_hit,
            "cache_dir": str(cache_result.cache_dir),
        }
    finally:
        runner.close()


def test_cached_mainline_train_runner_is_deterministic_with_reused_cache(tmp_path) -> None:
    seed = 11
    base_cfg = _make_train_equivalence_cfg(tmp_path / "deterministic")

    first_result = _run_cached_mainline_runner(
        deepcopy(base_cfg),
        seed=seed,
        tmp_path=tmp_path,
        refresh_cache=True,
    )
    second_result = _run_cached_mainline_runner(
        deepcopy(base_cfg),
        seed=seed,
        tmp_path=tmp_path,
        refresh_cache=False,
    )

    assert first_result["cache_hit"] is False
    assert second_result["cache_hit"] is True
    assert first_result["cache_dir"] == second_result["cache_dir"]
    assert first_result["episodes"] == second_result["episodes"]
    assert first_result["total_steps"] == second_result["total_steps"]
    assert np.allclose(first_result["episode_rewards"], second_result["episode_rewards"], atol=1e-6)

    first_summary = first_result["reward_summary"]
    second_summary = second_result["reward_summary"]
    assert first_summary["episodes"] == second_summary["episodes"]
    assert np.allclose(first_summary["episode_total_reward"], second_summary["episode_total_reward"], atol=1e-6)
    assert set(first_summary["components"]) == set(second_summary["components"])
    for key in first_summary["components"]:
        assert np.allclose(
            first_summary["components"][key]["values"],
            second_summary["components"][key]["values"],
            atol=1e-6,
        )
