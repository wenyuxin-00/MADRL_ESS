"""Factories and tiny datasets shared across tests."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from configs.experiment_config import ExperimentConfig


def write_dataset(path: str | Path, num_agents: int, total_steps: int) -> None:
    """Write a deterministic toy price/load dataset."""
    path = Path(path)
    fieldnames = ["price"] + [f"load{i + 1}" for i in range(num_agents)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(total_steps):
            row = {"price": 10.0 + step}
            for agent_id in range(num_agents):
                row[f"load{agent_id + 1}"] = 1.0 + 0.1 * agent_id + 0.01 * step
            writer.writerow(row)


def make_smoke_config(tmp_path: str | Path, algorithm: str = "MADDPG") -> ExperimentConfig:
    """Create a minimal but trainable configuration for tests."""
    tmp_path = Path(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    episode_limit = 3
    total_steps = episode_limit * 4
    write_dataset(data_dir / "train_prices.csv", num_agents=2, total_steps=total_steps)
    write_dataset(data_dir / "test_prices.csv", num_agents=2, total_steps=total_steps)

    cfg = ExperimentConfig()
    cfg.algo.name = algorithm
    cfg.data.data_dir = data_dir
    cfg.env.env_type = "energy_storage"
    cfg.data.dataset_type = "csv_price_load"
    cfg.obs.builder_type = "default"
    cfg.env.num_agents = 2
    cfg.env.episode_limit = episode_limit
    cfg.env.future_horizon = 1
    cfg.train.train_episodes = 2
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.train.num_envs = 1
    cfg.train.batch_size = 2
    cfg.train.buffer_size = 32
    cfg.train.use_noise_decay = False
    cfg.model.hidden_dim = 16
    cfg.model.use_orthogonal_init = False
    cfg.runtime.device = torch.device("cpu")
    return cfg


def make_case_dir(tmp_path: str | Path, label: str) -> Path:
    """Return the test-specific temporary directory for one case."""
    _ = label
    return Path(tmp_path)
