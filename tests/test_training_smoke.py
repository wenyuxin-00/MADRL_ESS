import os
from pathlib import Path

import pytest

from scripts.builder import build_train_runner
from scripts.checkpoints import LATEST_CHECKPOINT_MANIFEST, resolve_checkpoint_to_load
from tests.support.helpers import make_case_dir, make_smoke_config


@pytest.mark.parametrize(
    ("algorithm", "expected_critic_head"),
    [("MADDPG", "single_q"), ("MATD3", "twin_q")],
)
def test_training_smoke_save_and_load(tmp_path, algorithm, expected_critic_head):
    case_dir = make_case_dir(tmp_path, f"train_{algorithm.lower()}")
    original_cwd = Path.cwd()
    os.chdir(case_dir)
    cfg = make_smoke_config(case_dir, algorithm=algorithm)
    try:
        runner = build_train_runner(cfg, seed=0, env_name="SmokeEnv", number=1)
        try:
            episodes_completed = runner.run()
            assert episodes_completed == cfg.train.train_episodes
            assert runner.cfg.model.critic_head_type == expected_critic_head

            model_dir = case_dir / "saved_models"
            runner.save_model(str(model_dir), episode=episodes_completed)
            runner.load_model(str(model_dir), episode=episodes_completed)

            algo_dir = model_dir / algorithm
            assert (algo_dir / f"actor_agent_0_ep_{episodes_completed}.pth").exists()
            assert (algo_dir / f"critic_agent_0_ep_{episodes_completed}.pth").exists()
            assert (algo_dir / LATEST_CHECKPOINT_MANIFEST).exists()

            checkpoint_info = resolve_checkpoint_to_load(model_dir, algorithm)
            assert checkpoint_info["saved_episode_tag"] == episodes_completed
            assert checkpoint_info["episodes_completed"] == episodes_completed
        finally:
            runner.close()
    finally:
        os.chdir(original_cwd)
