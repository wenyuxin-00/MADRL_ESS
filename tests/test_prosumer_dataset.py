from __future__ import annotations

import numpy as np
import pytest

from configs.experiment_config import ExperimentConfig
from data.loaders.registry import build_dataset
from data.loaders.prosumer import ProsumerDataset
from envs.grid_env import GridEnv
from envs.observation.registry import build_obs_builder
from envs.rewards import NormalReward
from predictors.registry import build_forecaster
from predictors.training import _load_signal_matrix_from_source, resolve_signal_csv_source
from tests.support.helpers import (
    make_case_dir,
    make_smoke_config,
    synthetic_prosumer_heatpump_kw,
    synthetic_prosumer_household_kw,
    synthetic_prosumer_price_eur_per_kwh,
    synthetic_prosumer_ref_pv_kw,
    write_prosumer_processed_dataset,
)


class TinyGridCore:
    def __init__(self, n_agents: int) -> None:
        self.n_buses = 4
        self.n_lines = 3
        self.n_trafos = 1
        self.last_pf_error = ""
        self._n_agents = n_agents

    def reset(self, base_load_kw, base_pv_kw) -> None:
        del base_load_kw, base_pv_kw

    def step(self, p_batt_kw, base_load_kw):
        del base_load_kw
        from envs.grid.core.grid_types import GridStepResult

        n = len(p_batt_kw)
        return GridStepResult(
            converged=True,
            vm_pu=np.ones(self.n_buses, dtype=np.float32),
            va_degree=np.zeros(self.n_buses, dtype=np.float32),
            line_loading_pct=np.zeros(self.n_lines, dtype=np.float32),
            trafo_loading_pct=np.zeros(self.n_trafos, dtype=np.float32),
            p_mw_from=np.zeros(self.n_lines, dtype=np.float32),
            agent_vm_pu=np.ones(n, dtype=np.float32),
            v_violation=np.zeros(n, dtype=np.float32),
            line_violation=0.0,
            trafo_violation=0.0,
            l_violation=0.0,
            n_buses=self.n_buses,
            n_lines=self.n_lines,
            n_trafos=self.n_trafos,
            bus_v_excess=np.zeros(self.n_buses, dtype=np.float32),
            line_excess=np.zeros(self.n_lines, dtype=np.float32),
            trafo_excess=np.zeros(self.n_trafos, dtype=np.float32),
            psi_v_raw=0.0,
            psi_line_raw=0.0,
            psi_trafo_raw=0.0,
        )


def test_prosumer_dataset_filters_year_and_profiles(tmp_path):
    case_dir = make_case_dir(tmp_path, "prosumer_loader_year")
    data_dir = case_dir / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14", "SFH16"],
        train_steps=8,
        test_steps=8,
    )

    dataset = ProsumerDataset(
        data_dir=data_dir,
        episode_length=4,
        n_agents=2,
        agent_profiles=["SFH12", "SFH14"],
        year=2019,
        load_components=["household"],
        pv_reference="east",
        node_ids=[10, 6],
    )

    assert dataset.num_episodes() == 2

    episode = dataset.get_episode(0)

    assert np.allclose(
        episode["signals"]["load"][0],
        np.array(
            [
                synthetic_prosumer_household_kw(0, 0, year_offset=0),
                synthetic_prosumer_household_kw(1, 0, year_offset=0),
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(
        episode["signals"]["pv"][0],
        np.array(
            [
                synthetic_prosumer_ref_pv_kw("east", 0, year_offset=0),
                synthetic_prosumer_ref_pv_kw("east", 0, year_offset=0),
            ],
            dtype=np.float32,
        ),
    )
    assert np.isclose(
        episode["signals"]["wholesale_price"][0],
        synthetic_prosumer_price_eur_per_kwh(0, year_offset=0),
    )
    assert episode["meta"]["node_ids"] == [10, 6]
    assert episode["meta"]["agent_profiles"] == ["SFH12", "SFH14"]
    assert episode["meta"]["year"] == 2019
    assert episode["meta"]["load_components"] == ["household"]
    assert episode["meta"]["pv_reference"] == "east"
    assert episode["meta"]["signal_names"] == ["wholesale_price", "load", "pv"]
    assert episode["meta"]["timestamps"][0].startswith("2019-01-01 00:00:00")


def test_prosumer_dataset_combines_heatpump_and_scales_pv_and_ess(tmp_path):
    case_dir = make_case_dir(tmp_path, "prosumer_loader_scale")
    data_dir = case_dir / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14", "SFH16"],
        train_steps=8,
        test_steps=8,
    )

    pv_capacity_kw = [8.0, 4.0]
    load_scale = [1.5, 0.5]
    pv_scale = [0.5, 1.25]
    dataset = ProsumerDataset(
        data_dir=data_dir,
        episode_length=4,
        n_agents=2,
        agent_profiles=["SFH12", "SFH14"],
        year=2020,
        load_components=["household", "heatpump"],
        pv_reference="south",
        pv_capacity_kw=pv_capacity_kw,
        load_scale=load_scale,
        pv_scale=pv_scale,
        node_ids=[10, 6],
    )

    episode = dataset.get_episode(0)
    ref_peak_kw = max(synthetic_prosumer_ref_pv_kw("south", step, year_offset=1) for step in range(8))
    expected_pv = np.array(
        [
            synthetic_prosumer_ref_pv_kw("south", 0, year_offset=1) * pv_capacity_kw[0] / ref_peak_kw * pv_scale[0],
            synthetic_prosumer_ref_pv_kw("south", 0, year_offset=1) * pv_capacity_kw[1] / ref_peak_kw * pv_scale[1],
        ],
        dtype=np.float32,
    )
    expected_load = np.array(
        [
            synthetic_prosumer_household_kw(0, 0, year_offset=1)
            + synthetic_prosumer_heatpump_kw(0, 0, year_offset=1),
            synthetic_prosumer_household_kw(1, 0, year_offset=1)
            + synthetic_prosumer_heatpump_kw(1, 0, year_offset=1),
        ],
        dtype=np.float32,
    ) * np.array(load_scale, dtype=np.float32)

    assert np.allclose(episode["signals"]["load"][0], expected_load)
    assert np.allclose(episode["signals"]["pv"][0], expected_pv)
    assert np.allclose(
        episode["meta"]["pv_peak_kw"],
        np.array([8.0 * pv_scale[0], 4.0 * pv_scale[1]], dtype=np.float32),
    )
    assert np.allclose(episode["meta"]["load_scale"], np.array(load_scale, dtype=np.float32))
    assert np.allclose(episode["meta"]["pv_scale"], np.array(pv_scale, dtype=np.float32))
    assert "ess_power_kw" not in episode["meta"]
    assert "ess_capacity_kwh" not in episode["meta"]
    assert "storage_scale" not in episode["meta"]


def test_prosumer_dataset_filters_explicit_date_range_and_build_dataset_uses_it(tmp_path):
    case_dir = make_case_dir(tmp_path, "prosumer_loader_dates")
    data_dir = case_dir / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14", "SFH16"],
        train_steps=96 * 40,
        test_steps=96 * 40,
    )

    dataset = ProsumerDataset(
        data_dir=data_dir,
        episode_length=96,
        n_agents=2,
        agent_profiles=["SFH12", "SFH14"],
        year=2019,
        start_date="2019-01-05",
        end_date="2019-01-06",
        node_ids=[10, 6],
    )

    assert dataset.num_episodes() == 2
    first_episode = dataset.get_episode(0)
    second_episode = dataset.get_episode(1)
    assert first_episode["meta"]["timestamps"][0].startswith("2019-01-05 00:00:00")
    assert second_episode["meta"]["timestamps"][0].startswith("2019-01-06 00:00:00")
    assert first_episode["meta"]["date_range"] == {
        "start_date": "2019-01-05",
        "end_date": "2019-01-06",
    }

    cfg = ExperimentConfig()
    cfg.data.data_dir = data_dir
    cfg.data.agent_profiles = ["SFH12", "SFH14"]
    cfg.data.load_scale = [1.0, 1.0]
    cfg.data.pv_scale = [1.0, 1.0]
    cfg.data.train_year = 2019
    cfg.data.train_start_date = "2019-01-05"
    cfg.data.train_end_date = "2019-01-06"
    cfg.env.num_agents = 2
    cfg.env.episode_limit = 96
    cfg.grid.agent_bus_ids = [10, 6]

    built = build_dataset(cfg, mode="train")
    built_episode = built.get_episode(0)
    assert built_episode["meta"]["timestamps"][0].startswith("2019-01-05 00:00:00")


def test_prosumer_dataset_validates_inputs_and_build_dataset_lengths(tmp_path):
    case_dir = make_case_dir(tmp_path, "prosumer_loader_validate")
    data_dir = case_dir / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14"],
        train_steps=8,
        test_steps=8,
    )

    with pytest.raises(ValueError, match="Unknown agent profiles"):
        ProsumerDataset(
            data_dir=data_dir,
            episode_length=4,
            n_agents=1,
            agent_profiles=["BAD_PROFILE"],
            year=2019,
            node_ids=[10],
        )

    with pytest.raises(ValueError, match="pv_capacity_kw length should equal n_agents"):
        ProsumerDataset(
            data_dir=data_dir,
            episode_length=4,
            n_agents=2,
            agent_profiles=["SFH12", "SFH14"],
            year=2019,
            pv_capacity_kw=[4.0],
            node_ids=[10, 6],
        )

    with pytest.raises(ValueError, match="node_ids length should equal n_agents"):
        ProsumerDataset(
            data_dir=data_dir,
            episode_length=4,
            n_agents=2,
            agent_profiles=["SFH12", "SFH14"],
            year=2019,
            node_ids=[10],
        )

    cfg = ExperimentConfig()
    cfg.data.data_dir = data_dir
    cfg.data.agent_profiles = ["SFH12"]
    cfg.data.load_scale = [1.0, 1.0]
    cfg.data.pv_scale = [1.0, 1.0]
    cfg.env.num_agents = 2
    cfg.env.episode_limit = 4
    cfg.grid.agent_bus_ids = [10, 6]

    with pytest.raises(ValueError, match="agent_profiles length should equal n_agents"):
        build_dataset(cfg, mode="train")


def test_prosumer_build_dataset_and_grid_env_smoke(tmp_path):
    case_dir = make_case_dir(tmp_path, "prosumer_env_smoke")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.grid.train_compact_info = False

    dataset = build_dataset(cfg, mode="train")
    assert isinstance(dataset, ProsumerDataset)

    env = GridEnv(
        cfg,
        mode="train",
        dataset=dataset,
        reward_fn=NormalReward(cfg),
        forecaster=build_forecaster(cfg),
        obs_builder=build_obs_builder(cfg),
        grid_core=TinyGridCore(cfg.env.num_agents),
    )
    try:
        obs, reset_info = env.reset(episode_idx=0)
        assert {"local", "wholesale_price_seq", "load_seq", "pv_seq", "adjacency"} <= set(obs.keys())
        assert reset_info["episode_meta"]["node_ids"] == cfg.grid.agent_bus_ids

        next_obs, reward, terminated, truncated, info = env.step(
            [np.zeros((2,), dtype=np.float32) for _ in range(cfg.env.num_agents)]
        )
        assert next_obs["pv_seq"].shape == (cfg.env.num_agents, cfg.env.future_horizon + 1)
        assert len(reward) == cfg.env.num_agents
        assert len(terminated) == cfg.env.num_agents
        assert len(truncated) == cfg.env.num_agents
        assert set(info) == {"episode_done", *[str(meta.key) for meta in env.reward_fn.component_meta]}
    finally:
        env.close()

    fallback_env = GridEnv(
        cfg,
        mode="test",
        dataset=None,
        reward_fn=NormalReward(cfg),
        forecaster=build_forecaster(cfg),
        obs_builder=build_obs_builder(cfg),
        grid_core=TinyGridCore(cfg.env.num_agents),
    )
    try:
        obs, reset_info = fallback_env.reset(episode_idx=0)
        assert {"local", "wholesale_price_seq", "load_seq", "pv_seq", "adjacency"} <= set(obs.keys())
        assert reset_info["episode_meta"]["year"] == cfg.data.test_year

        _, reward, terminated, truncated, info = fallback_env.step(
            [np.zeros((2,), dtype=np.float32) for _ in range(cfg.env.num_agents)]
        )
        assert len(reward) == cfg.env.num_agents
        assert len(terminated) == cfg.env.num_agents
        assert len(truncated) == cfg.env.num_agents
        assert sorted(info["available_signals"]) == [
            "load",
            "load_heatpump",
            "load_household",
            "pv",
            "wholesale_price",
        ]
    finally:
        fallback_env.close()


def test_prosumer_forecast_source_respects_profiles_scales_and_date_range(tmp_path):
    case_dir = make_case_dir(tmp_path, "prosumer_forecast_source")
    data_dir = case_dir / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14", "SFH16"],
        train_steps=96 * 12,
        test_steps=96 * 12,
    )

    cfg = ExperimentConfig()
    cfg.data.data_dir = data_dir
    cfg.data.agent_profiles = ["SFH12", "SFH14"]
    cfg.data.train_year = 2019
    cfg.data.test_year = 2020
    cfg.data.train_start_date = "2019-01-03"
    cfg.data.train_end_date = "2019-01-04"
    cfg.data.load_scale = [2.0, 0.5]
    cfg.data.pv_scale = [0.75, 1.25]
    cfg.data.load_components = ["household", "heatpump"]
    cfg.data.pv_reference = "south"
    cfg.env.num_agents = 2

    source = resolve_signal_csv_source(data_dir, "load", cfg=cfg)
    assert source is not None
    assert source.source_kind == "prosumer"

    frame, values, value_columns = _load_signal_matrix_from_source(source, "load", split="train")
    assert value_columns == ("load_SFH12", "load_SFH14")
    assert frame["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S") == "2019-01-03 00:00:00"
    assert values.shape == (96 * 2, 2)
    assert np.allclose(
        values[0],
        np.array(
            [
                (
                    synthetic_prosumer_household_kw(0, 96 * 2, year_offset=0)
                    + synthetic_prosumer_heatpump_kw(0, 96 * 2, year_offset=0)
                )
                * 2.0,
                (
                    synthetic_prosumer_household_kw(1, 96 * 2, year_offset=0)
                    + synthetic_prosumer_heatpump_kw(1, 96 * 2, year_offset=0)
                )
                * 0.5,
            ],
            dtype=np.float32,
        ),
    )


