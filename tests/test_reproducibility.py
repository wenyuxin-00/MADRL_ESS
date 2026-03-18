import numpy as np
import torch

from common.torch_runtime import STRICT_REPRO_RUNTIME_MODE
from core.builder import build_train_runner
from tests.support.helpers import make_case_dir, make_smoke_config


def _run_strict_training(case_dir, seed: int):
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.runtime.execution_mode = STRICT_REPRO_RUNTIME_MODE
    cfg.runtime.seed = seed

    runner = build_train_runner(cfg, seed=seed, env_name="StrictRepro", number=1)
    try:
        episodes_completed = runner.run()
        actor_state = {
            key: value.detach().cpu().clone()
            for key, value in runner.agent_n[0].actor.state_dict().items()
        }
        episode_rewards = np.asarray(runner.episode_rewards, dtype=np.float32)
        return episodes_completed, episode_rewards, actor_state
    finally:
        runner.close()


def test_strict_reproducibility_repeats_minimal_training_chain(tmp_path):
    first_case = make_case_dir(tmp_path / "run_a", "strict_a")
    second_case = make_case_dir(tmp_path / "run_b", "strict_b")

    first_episodes, first_rewards, first_state = _run_strict_training(first_case, seed=17)
    second_episodes, second_rewards, second_state = _run_strict_training(second_case, seed=17)

    assert first_episodes == second_episodes
    assert np.allclose(first_rewards, second_rewards)
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        assert torch.equal(first_state[key], second_state[key])
