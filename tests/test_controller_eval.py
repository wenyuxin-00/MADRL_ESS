import os
from pathlib import Path

import pytest

from controllers import ClassicDRLController, MADRLController, MPCController, ZeroController
from scripts.builder import build_env, build_train_runner
from scripts import evaluate_controller
from tests.support.helpers import make_case_dir, make_smoke_config


def test_zero_controller_can_be_evaluated(tmp_path):
    case_dir = make_case_dir(tmp_path, "eval_zero")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    env = build_env(cfg, mode="test")

    results = evaluate_controller(
        env,
        ZeroController(action_dim_n=[1] * cfg.env.num_agents),
        n_episodes=1,
        deterministic=True,
    )

    assert results["n_episodes"] == 1
    assert len(results["episode_rewards"]) == 1
    assert len(results["histories"]) == 1

    history = results["histories"][0]
    expected_keys = {
        "price",
        "e_bat_req",
        "e_bat_exec",
        "soc",
        "r_total_sum",
        "r_inc_sum",
        "r_pen_sum",
        "r_pbrs_sum",
        "r_soc_sum",
        "r_bonus_sum",
    }
    assert expected_keys.issubset(history.keys())
    env.close()


@pytest.mark.parametrize("algorithm", ["MADDPG", "MATD3"])
def test_runner_controller_can_be_evaluated(tmp_path, algorithm):
    case_dir = make_case_dir(tmp_path, f"eval_{algorithm.lower()}")
    original_cwd = Path.cwd()
    os.chdir(case_dir)
    cfg = make_smoke_config(case_dir, algorithm=algorithm)
    try:
        runner = build_train_runner(cfg, seed=0, env_name="ControllerEval", number=1)
        try:
            episodes_completed = runner.run()
            controller = MADRLController(runner.agent_n, noise_std=runner.noise_std)
            eval_env = build_env(cfg, mode="test")
            try:
                results = evaluate_controller(eval_env, controller, n_episodes=1, deterministic=True)
            finally:
                eval_env.close()

            assert episodes_completed == cfg.train.train_episodes
            assert results["n_episodes"] == 1
            assert len(results["episode_rewards"]) == 1
            assert len(results["histories"]) == 1
        finally:
            runner.close()
    finally:
        os.chdir(original_cwd)


def test_placeholder_controllers_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        MPCController().act({"local": [[[0.0]]]}, deterministic=True)

    with pytest.raises(NotImplementedError):
        ClassicDRLController().act({"local": [[[0.0]]]}, deterministic=True)
