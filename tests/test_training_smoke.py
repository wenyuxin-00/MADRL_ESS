import csv

import torch
import pytest

from configs.default_config import Config
from core.builder import build_train_runner
from runners.checkpoints import LATEST_CHECKPOINT_MANIFEST, resolve_checkpoint_to_load


def _write_dataset(path, num_agents: int, total_steps: int) -> None:
    fieldnames = ["price"] + [f"load{i + 1}" for i in range(num_agents)]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(total_steps):
            row = {"price": 10.0 + step}
            for agent_id in range(num_agents):
                row[f"load{agent_id + 1}"] = 1.0 + 0.1 * agent_id + 0.01 * step
            writer.writerow(row)


def _make_smoke_config(tmp_path, algorithm: str) -> Config:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    episode_limit = 3
    total_steps = episode_limit * 4
    _write_dataset(data_dir / "train_prices.csv", num_agents=2, total_steps=total_steps)
    _write_dataset(data_dir / "test_prices.csv", num_agents=2, total_steps=total_steps)

    args = Config()
    args.algorithm = algorithm
    args.critic_type = None
    args.data_dir = data_dir
    args.num_agents = 2
    args.N = 2
    args.episode_limit = episode_limit
    args.future_horizon = 1
    args.train_episodes = 2
    args.max_train_steps = args.train_episodes * args.episode_limit
    args.num_envs = 1
    args.batch_size = 2
    args.buffer_size = 32
    args.hidden_dim = 16
    args.use_orthogonal_init = False
    args.use_noise_decay = False
    args.device = torch.device("cpu")
    return args


@pytest.mark.parametrize(
    ("algorithm", "expected_critic"),
    [("MADDPG", "maddpg_mlp"), ("MATD3", "matd3_mlp")],
)
def test_training_smoke_save_and_load(tmp_path, monkeypatch, algorithm, expected_critic):
    monkeypatch.chdir(tmp_path)

    args = _make_smoke_config(tmp_path, algorithm)
    runner = build_train_runner(args, seed=0, env_name="SmokeEnv", number=1)

    episodes_completed = runner.run()
    assert episodes_completed == args.train_episodes
    assert runner.args.critic_type == expected_critic

    model_dir = tmp_path / "saved_models"
    runner.save_model(str(model_dir), episode=episodes_completed)
    runner.load_model(str(model_dir), episode=episodes_completed)

    algo_dir = model_dir / algorithm
    assert (algo_dir / f"actor_agent_0_ep_{episodes_completed}.pth").exists()
    assert (algo_dir / f"critic_agent_0_ep_{episodes_completed}.pth").exists()
    assert (algo_dir / LATEST_CHECKPOINT_MANIFEST).exists()

    checkpoint_info = resolve_checkpoint_to_load(model_dir, algorithm)
    assert checkpoint_info["saved_episode_tag"] == episodes_completed
    assert checkpoint_info["episodes_completed"] == episodes_completed
