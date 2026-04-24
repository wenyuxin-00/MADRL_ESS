from controllers.madrl.base_agent import get_agent_cls
from configs.experiment_config import ExperimentConfig
from data.loaders.prosumer import ProsumerDataset
from data.loaders.registry import build_dataset, get_dataset_cls
from envs.grid_env import GridEnv
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.normalization import build_observation_normalizer
from tests.support.helpers import make_case_dir, make_smoke_config


def test_grid_env_direct_import_exposes_mainline_env():
    assert GridEnv.__name__ == "GridEnv"


def test_dataset_registry_builds_default_dataset(tmp_path):
    case_dir = make_case_dir(tmp_path, "dataset_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    dataset = build_dataset(cfg, mode="train")

    assert isinstance(dataset, ProsumerDataset)
    assert get_dataset_cls("prosumer") is ProsumerDataset


def test_default_observation_builder_builds_from_config(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    builder = DefaultObservationBuilder(
        local_features=cfg.obs.local_features,
        sequence_features=cfg.obs.sequence_features,
        future_horizon=cfg.env.future_horizon,
        adjacency_type=cfg.obs.adjacency_type,
        normalizer=build_observation_normalizer(cfg),
    )

    assert isinstance(builder, DefaultObservationBuilder)
    assert builder.normalizer is not None


def test_default_compose_config_targets_grid_training_mainline():
    from configs.profiles import compose_experiment_config

    cfg = compose_experiment_config()

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.env.num_agents == 3
    assert cfg.data.agent_profiles == ["SFH12", "SFH18", "SFH20"]
    assert cfg.data.train_year == 2019
    assert cfg.data.test_year == 2020
    assert cfg.data.test_start_date == "2020-04-01"
    assert cfg.data.test_end_date == "2020-04-15"
    assert cfg.env.episode_limit == 96
    assert cfg.env.train_window_days == 7
    assert cfg.env.window_stride_days == 1
    assert cfg.env.future_horizon == 48
    assert cfg.reward.action_boundary_penalty_weight >= 0.0
    assert cfg.reward.soc_boundary_regularization_weight >= 0.0
    assert cfg.reward.throughput_bonus_eur_per_kwh_max >= 0.0
    assert cfg.reward.w_voltage_pen >= 0.0
    assert cfg.reward.w_line_pen >= 0.0
    assert cfg.reward.w_trafo_pen == 10.0
    assert cfg.reward.import_price_markup_eur_per_kwh == 0.0
    assert cfg.grid.sb_code == "1-LV-rural1--0-sw"
    assert cfg.grid.agent_bus_ids == [12, 4, 2]
    assert cfg.obs.local_features == ["calendar_time", "soc"]
    assert cfg.obs.sequence_features == ["wholesale_price_relative", "wholesale_price_spread", "load", "pv"]
    assert cfg.safety.enabled is False
    assert get_agent_cls("MATD3_SAFE_POC").__name__ == "MATD3SafePOC"
