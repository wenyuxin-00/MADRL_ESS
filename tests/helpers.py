"""Shared test helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from configs.experiment_config import ExperimentConfig

DEFAULT_TEST_BUSES = [10, 6, 12, 7, 8]


def write_dataset(path: str | Path, num_agents: int, total_steps: int) -> None:
    path = Path(path)
    fieldnames = ["segment_id", "price"]
    fieldnames += [f"load{i + 1}" for i in range(num_agents)]
    fieldnames += [f"pv{i + 1}" for i in range(num_agents)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(total_steps):
            row = {
                "segment_id": 0,
                "price": 10.0 + 0.1 * step,
            }
            solar_phase = max(0.0, 1.0 - abs((step % 24) - 12) / 12.0)
            for agent_id in range(num_agents):
                row[f"load{agent_id + 1}"] = 1.0 + 0.1 * agent_id + 0.01 * step
                row[f"pv{agent_id + 1}"] = 0.2 + 0.15 * agent_id + 0.5 * solar_phase
            writer.writerow(row)


def make_smoke_config(tmp_path: str | Path, algorithm: str = "MADDPG") -> ExperimentConfig:
    tmp_path = Path(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    episode_limit = 3
    total_steps = episode_limit * 4
    write_dataset(data_dir / "simbench_2016_train.csv", num_agents=2, total_steps=total_steps)
    write_dataset(data_dir / "simbench_2016_test.csv", num_agents=2, total_steps=total_steps)

    cfg = ExperimentConfig()
    cfg.algo.name = algorithm
    cfg.data.data_dir = data_dir
    cfg.env.env_type = "grid_pf"
    cfg.data.dataset_type = "csv_prosumer"
    cfg.reward.type = "grid_composite"
    cfg.obs.builder_type = "default"
    cfg.obs.local_features = ["time", "price", "load", "pv", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.forecast.target_signals = ["price", "load", "pv"]
    cfg.env.num_agents = 2
    cfg.env.episode_limit = episode_limit
    cfg.env.future_horizon = 1
    cfg.grid.agent_bus_ids = DEFAULT_TEST_BUSES[: cfg.env.num_agents]
    cfg.train.train_episodes = 2
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.train.num_envs = 1
    cfg.train.vec_env_type = "dummy"
    cfg.train.batch_size = 2
    cfg.train.buffer_size = 32
    cfg.train.use_noise_decay = False
    cfg.model.hidden_dim = 16
    cfg.model.use_orthogonal_init = False
    cfg.runtime.device = torch.device("cpu")
    return cfg


def make_case_dir(tmp_path: Path, label: str) -> Path:
    _ = label
    return Path(tmp_path)
