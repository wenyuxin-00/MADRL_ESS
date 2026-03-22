import json
from pathlib import Path

import numpy as np

from configs.experiment_config import ExperimentConfig
from data.loaders.csv_prosumer import CsvProsumerDataset
from data.loaders.registry import build_dataset, get_dataset_cls
from scripts.builder import build_env


def _write_prosumer_csv(path: Path, total_steps: int, n_agents: int) -> None:
    header = ["timestamp", "price"]
    header += [f"load{i + 1}" for i in range(n_agents)]
    header += [f"pv{i + 1}" for i in range(n_agents)]
    rows = [",".join(header)]
    for step in range(total_steps):
        row = [f"2016-01-01 00:{step:02d}:00", f"{0.05 + 0.001 * step:.6f}"]
        row += [f"{4.0 + agent + 0.1 * step:.6f}" for agent in range(n_agents)]
        row += [f"{1.0 + 0.5 * agent + 0.05 * step:.6f}" for agent in range(n_agents)]
        rows.append(",".join(row))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _make_prosumer_cfg(tmp_path: Path) -> ExperimentConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 12
    n_agents = 3
    _write_prosumer_csv(data_dir / "simbench_2016_train.csv", total_steps=total_steps, n_agents=n_agents)
    _write_prosumer_csv(data_dir / "simbench_2016_test.csv", total_steps=total_steps, n_agents=n_agents)
    (data_dir / "simbench_2016_metadata.json").write_text(
        json.dumps(
            {
                "selected_buses": [10, 6, 12],
                "pv_peak_kw": [40.0, 20.0, 10.0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg = ExperimentConfig()
    cfg.data.data_dir = data_dir
    cfg.data.dataset_type = "csv_prosumer"
    cfg.env.num_agents = n_agents
    cfg.env.episode_limit = 4
    cfg.env.future_horizon = 2
    cfg.obs.local_features = ["time", "price", "load", "pv", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.reward.type = "sparse"
    return cfg


def test_dataset_registry_builds_prosumer_dataset(tmp_path):
    cfg = _make_prosumer_cfg(tmp_path)

    dataset = build_dataset(cfg, mode="train")
    episode = dataset.get_episode(0)

    assert isinstance(dataset, CsvProsumerDataset)
    assert get_dataset_cls("csv_prosumer") is CsvProsumerDataset
    assert set(episode["signals"].keys()) == {"price", "load", "pv"}
    assert episode["signals"]["load"].shape == (cfg.env.episode_limit, cfg.env.num_agents)
    assert episode["signals"]["pv"].shape == (cfg.env.episode_limit, cfg.env.num_agents)
    assert np.allclose(episode["meta"]["ess_power_kw"], np.array([20.0, 10.0, 5.0], dtype=np.float32))
    assert np.allclose(episode["meta"]["ess_capacity_kwh"], np.array([50.0, 25.0, 12.5], dtype=np.float32))


def test_env_uses_pv_net_load_and_agent_storage_sizing(tmp_path):
    cfg = _make_prosumer_cfg(tmp_path)
    env = build_env(cfg, mode="test")

    try:
        obs, reset_info = env.reset(episode_idx=0)
        assert set(obs.keys()) == {"local", "price_seq", "load_seq", "pv_seq", "adjacency"}
        assert "p_max" in reset_info

        _, reward, _, _, info = env.step(
            [np.array([0.0], dtype=np.float32) for _ in range(cfg.env.num_agents)]
        )

        expected_load = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected_pv = np.array([1.0, 1.5, 2.0], dtype=np.float32)
        expected_base_net = expected_load - expected_pv

        assert np.allclose(info["load"], expected_load)
        assert np.allclose(info["pv"], expected_pv)
        assert np.allclose(info["base_net_load"], expected_base_net)
        assert np.allclose(info["net_load"], expected_base_net)
        assert np.allclose(info["p_max"], np.array([20.0, 10.0, 5.0], dtype=np.float32))
        assert np.allclose(info["battery_capacity_kwh"], np.array([50.0, 25.0, 12.5], dtype=np.float32))
        assert np.allclose(np.asarray(reward, dtype=np.float32), np.zeros((cfg.env.num_agents,), dtype=np.float32))
    finally:
        env.close()
