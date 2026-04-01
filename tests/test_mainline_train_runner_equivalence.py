from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from scripts.builder import build_train_runner
from scripts.utils.madrl_shared_data import ensure_madrl_shared_data
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


def _run_mainline_runner_with_shared_data(cfg, *, seed: int, tmp_path):
    shared_data = ensure_madrl_shared_data(cfg, root=tmp_path / "shared_data_root")
    cfg.runtime.shared_data_dir = str(shared_data.shared_data_dir)
    cfg.runtime.shared_data_signature = str(shared_data.signature_hash)

    runner = build_train_runner(cfg, seed=seed, env_name="GridTrainMainlineSharedData", number=1)
    try:
        episodes = runner.run()
        shared_data_meta = dict(getattr(runner, "shared_data_metadata", {}) or {})
        return {
            "episodes": episodes,
            "total_steps": runner.total_steps,
            "episode_rewards": list(runner.episode_rewards),
            "reward_summary": runner.build_reward_summary(),
            "shared_data_dir": str(shared_data_meta.get("shared_data_dir", "")),
            "shared_data_signature": str(shared_data_meta.get("shared_data_signature", "")),
            "history_length": len(runner.history),
            "perf_summary": dict(runner.perf_summary),
        }
    finally:
        runner.close()


def test_mainline_train_runner_is_deterministic_with_reused_shared_data(tmp_path) -> None:
    seed = 11
    base_cfg = _make_train_equivalence_cfg(tmp_path / "deterministic")

    first_result = _run_mainline_runner_with_shared_data(
        deepcopy(base_cfg),
        seed=seed,
        tmp_path=tmp_path,
    )
    second_result = _run_mainline_runner_with_shared_data(
        deepcopy(base_cfg),
        seed=seed,
        tmp_path=tmp_path,
    )

    assert first_result["shared_data_dir"] == second_result["shared_data_dir"]
    assert first_result["shared_data_signature"] == second_result["shared_data_signature"]
    assert first_result["episodes"] == second_result["episodes"]
    assert first_result["total_steps"] == second_result["total_steps"]
    assert first_result["history_length"] == 0
    assert second_result["history_length"] == 0
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
