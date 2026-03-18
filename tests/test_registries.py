from data.loaders.csv_price_load import CsvPriceLoadDataset
from data.loaders.registry import build_dataset, get_dataset_cls
from envs.hems_env import EnergyStorageEnv
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.registry import build_obs_builder, get_obs_builder_cls
from envs.registry import get_env_cls
from tests.support.helpers import make_case_dir, make_smoke_config


def test_env_registry_returns_default_environment_class():
    assert get_env_cls("energy_storage") is EnergyStorageEnv


def test_dataset_registry_builds_default_dataset(tmp_path):
    case_dir = make_case_dir(tmp_path, "dataset_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    dataset = build_dataset(cfg, mode="train")

    assert isinstance(dataset, CsvPriceLoadDataset)
    assert get_dataset_cls("csv_price_load") is CsvPriceLoadDataset


def test_observation_builder_registry_builds_default_builder(tmp_path):
    case_dir = make_case_dir(tmp_path, "obs_registry")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    builder = build_obs_builder(cfg)

    assert isinstance(builder, DefaultObservationBuilder)
    assert get_obs_builder_cls("default") is DefaultObservationBuilder
