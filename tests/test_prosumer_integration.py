from __future__ import annotations

import numpy as np

from configs.experiment_config import ExperimentConfig
from data.loaders.prosumer import ProsumerDataset
from data.loaders.registry import build_dataset
from scripts.builder import build_env
from tests.support.helpers import make_smoke_config, write_prosumer_processed_dataset


def _make_prosumer_cfg(tmp_path) -> ExperimentConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    agent_profiles = ["SFH12", "SFH14", "SFH16"]
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=agent_profiles,
        train_steps=12,
        test_steps=12,
    )

    cfg = ExperimentConfig()
    cfg.data.data_dir = data_dir
    cfg.data.agent_profiles = agent_profiles
    cfg.data.load_components = ["household", "heatpump"]
    cfg.data.pv_reference = "south"
    cfg.env.num_agents = len(agent_profiles)
    cfg.env.episode_limit = 4
    cfg.env.train_window_days = 1
    cfg.env.window_stride_days = 1
    cfg.env.future_horizon = 2
    cfg.obs.local_features = ["time", "soc"]
    cfg.obs.sequence_features = ["wholesale_price", "load", "pv"]
    cfg.forecast.target_signals = ["wholesale_price", "load", "pv"]
    cfg.data.load_scale = [1.0] * len(agent_profiles)
    cfg.data.pv_scale = [1.0] * len(agent_profiles)
    cfg.grid.agent_bus_ids = [10, 6, 12]
    return cfg


def test_build_dataset_builds_prosumer_dataset(tmp_path):
    cfg = _make_prosumer_cfg(tmp_path)

    dataset = build_dataset(cfg, mode="train")
    episode = dataset.get_episode(0)

    assert isinstance(dataset, ProsumerDataset)
    assert set(episode["signals"].keys()) == {
        "wholesale_price",
        "load",
        "pv",
        "load_household",
        "load_heatpump",
    }
    assert episode["signals"]["load"].shape == (cfg.env.episode_limit, cfg.env.num_agents)
    assert episode["signals"]["load_household"].shape == (cfg.env.episode_limit, cfg.env.num_agents)
    assert episode["signals"]["load_heatpump"].shape == (cfg.env.episode_limit, cfg.env.num_agents)
    assert episode["signals"]["pv"].shape == (cfg.env.episode_limit, cfg.env.num_agents)
    assert np.allclose(episode["meta"]["node_ids"], np.array([10, 6, 12], dtype=np.int32))
    assert episode["meta"]["agent_profiles"] == ["SFH12", "SFH14", "SFH16"]


def test_build_env_smoke_uses_processed_prosumer_dataset(tmp_path):
    cfg = make_smoke_config(tmp_path, algorithm="MADDPG")
    env = build_env(cfg, mode="test")

    try:
        obs, reset_info = env.reset(episode_idx=0)
        assert {"local", "wholesale_price_relative_seq", "wholesale_price_spread_seq", "load_seq", "pv_seq"} <= set(obs.keys())
        assert reset_info["episode_meta"]["agent_profiles"] == ["SFH12", "SFH14"]
        assert reset_info["episode_meta"]["year"] == cfg.data.test_year
    finally:
        env.close()
