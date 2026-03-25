from configs.experiment_config import ExperimentConfig
from data.loaders.prosumer import ProsumerDataset
from data.loaders.registry import build_dataset, get_dataset_cls
from envs import DEFAULT_ENV_NAME, SUPPORTED_ENV_NAMES, GridEnv, get_env_cls
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.registry import build_obs_builder, get_obs_builder_cls
from tests.support.helpers import make_case_dir, make_smoke_config


def test_env_helper_only_exposes_grid_mainline():
    assert DEFAULT_ENV_NAME == "grid"
    assert SUPPORTED_ENV_NAMES == ("grid",)
    assert get_env_cls() is GridEnv
    assert get_env_cls("grid") is GridEnv


def test_dataset_registry_builds_default_dataset(tmp_path):
    case_dir = make_case_dir(tmp_path, "dataset_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    dataset = build_dataset(cfg, mode="train")

    assert isinstance(dataset, ProsumerDataset)
    assert get_dataset_cls("prosumer") is ProsumerDataset


def test_observation_builder_helper_builds_default_builder(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    builder = build_obs_builder(cfg)

    assert isinstance(builder, DefaultObservationBuilder)
    assert builder.normalizer is not None
    assert get_obs_builder_cls() is DefaultObservationBuilder
    assert get_obs_builder_cls("default") is DefaultObservationBuilder


def test_default_compose_config_targets_grid_training_mainline():
    from configs import compose_experiment_config

    cfg = compose_experiment_config()

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.data.agent_profiles == ["SFH12", "SFH14", "SFH16"]
    assert cfg.data.train_year == 2019
    assert cfg.data.test_year == 2020
    assert cfg.reward.w_action_pen == 6.0
    assert cfg.reward.lambda_throughput == 0.001
    assert cfg.reward.w_voltage_pen == 10.0
    assert cfg.reward.w_trafo_pen == 10.0
    assert cfg.grid.sb_code == "1-LV-rural1--0-sw"
    assert cfg.grid.agent_bus_ids == [10, 6, 12]
    assert cfg.obs.local_features == ["calendar_time", "soc"]
    assert cfg.obs.sequence_features == ["price", "load", "pv"]
