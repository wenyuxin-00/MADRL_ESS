from data.loaders.csv_prosumer import CsvProsumerDataset
from data.loaders.registry import build_dataset, get_dataset_cls
from envs.grid_env import GridEnv
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.registry import build_obs_builder, get_obs_builder_cls
from envs.registry import ENV_REGISTRY, get_env_cls
from tests.support.helpers import make_case_dir, make_smoke_config


def test_env_registry_only_exposes_grid_environment():
    assert ENV_REGISTRY == {"grid_pf": GridEnv}
    assert get_env_cls("grid_pf") is GridEnv


def test_dataset_registry_builds_default_dataset(tmp_path):
    case_dir = make_case_dir(tmp_path, "dataset_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    dataset = build_dataset(cfg, mode="train")

    assert isinstance(dataset, CsvProsumerDataset)
    assert get_dataset_cls("csv_prosumer") is CsvProsumerDataset


def test_observation_builder_registry_builds_default_builder(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    builder = build_obs_builder(cfg)

    assert isinstance(builder, DefaultObservationBuilder)
    assert get_obs_builder_cls("default") is DefaultObservationBuilder


def test_default_compose_config_targets_grid_training_mainline():
    from configs import compose_experiment_config

    cfg = compose_experiment_config()

    assert cfg.env.env_type == "grid_pf"
    assert cfg.data.dataset_type == "csv_prosumer"
    assert cfg.reward.type == "grid_composite"
    assert cfg.grid.sb_code == "1-LV-rural1--0-sw"
    assert cfg.grid.agent_bus_ids == [10, 6, 12]
    assert cfg.obs.local_features == ["time", "price", "load", "pv", "soc"]
    assert cfg.obs.sequence_features == ["price", "load", "pv"]
