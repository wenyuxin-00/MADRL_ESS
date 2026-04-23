from __future__ import annotations

from configs.experiment_config import ExperimentConfig
from scripts.utils.grid_notebook_workflow import apply_notebook_experiment_settings


def test_experiment_config_defaults_align_with_madrl_notebooks() -> None:
    cfg = ExperimentConfig()

    assert list(cfg.data.agent_profiles) == ["SFH12", "SFH18", "SFH20"]
    assert list(cfg.grid.agent_bus_ids) == [12, 4, 2]
    assert int(cfg.data.train_year) == 2019
    assert int(cfg.data.test_year) == 2020
    assert cfg.data.test_start_date == "2020-06-01"
    assert cfg.data.test_end_date == "2020-06-07"
    assert int(cfg.env.num_agents) == 3
    assert int(cfg.env.episode_limit) == 96
    assert int(cfg.env.future_horizon) == 24


def test_data_config_resolved_scale_defaults_follow_global_values() -> None:
    cfg = ExperimentConfig()
    n_agents = len(cfg.data.agent_profiles)

    assert cfg.data.resolved_load_scale(n_agents) == [float(value) for value in cfg.data.load_scale]
    assert cfg.data.resolved_pv_scale(n_agents) == [float(value) for value in cfg.data.pv_scale]


def test_apply_notebook_experiment_settings_uses_cfg_defaults_when_optional_values_are_none() -> None:
    cfg = ExperimentConfig()

    controls = apply_notebook_experiment_settings(
        cfg,
        prediction_mode="normal",
        test_start_date=None,
        test_end_date=None,
        agent_profiles=None,
        agent_bus_ids=None,
        load_scale=None,
        pv_scale=None,
        future_horizon=None,
        battery_controls=None,
        forecast_controls=None,
        train_year=None,
        test_year=None,
    )

    assert list(cfg.data.agent_profiles) == ["SFH12", "SFH18", "SFH20"]
    assert list(cfg.grid.agent_bus_ids) == [12, 4, 2]
    assert cfg.data.test_start_date == "2020-06-01"
    assert cfg.data.test_end_date == "2020-06-07"
    assert list(cfg.data.load_scale) == [float(value) for value in ExperimentConfig().data.load_scale]
    assert list(cfg.data.pv_scale) == [float(value) for value in ExperimentConfig().data.pv_scale]
    assert controls["agent_bus_ids"] == [12, 4, 2]
    assert controls["load_scale"] == [float(value) for value in ExperimentConfig().data.load_scale]
    assert controls["pv_scale"] == [float(value) for value in ExperimentConfig().data.pv_scale]
    assert controls["test_start_date"] == "2020-06-01"
    assert controls["test_end_date"] == "2020-06-07"
