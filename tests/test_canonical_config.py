import pytest

from configs import compose_experiment_config
from configs.experiment_config import ExperimentConfig
from forecast.artifacts import get_default_lstm_artifact_paths
from models import validate_and_finalize_model_config


def test_canonical_config_defaults_are_structured():
    cfg = ExperimentConfig()

    assert cfg.algo.name == "MADDPG"
    assert cfg.env.env_type == "energy_storage"
    assert cfg.obs.local_features == ["time", "price", "load", "soc"]
    assert cfg.obs.sequence_features == ["price", "load"]
    assert cfg.obs.builder_type == "default"
    assert cfg.data.dataset_type == "csv_price_load"
    assert not hasattr(cfg.reward, "w_pv_rolling")
    assert not hasattr(cfg.env, "soc_eps")
    assert cfg.train.resolved_max_train_steps(cfg.env.episode_limit) == (
        cfg.train.train_episodes * cfg.env.episode_limit
    )


def test_compose_experiment_config_applies_profiles():
    cfg = compose_experiment_config(
        profile="debug",
        algorithm="MATD3",
        model_family="graph",
        reward_type="sparse",
        observation_profile="default",
        forecast_type="naive",
    )

    assert cfg.algo.name == "MATD3"
    assert cfg.model.family == "graph"
    assert cfg.reward.type == "sparse"
    assert cfg.forecast.type == "naive"
    assert cfg.env.env_type == "energy_storage"
    assert cfg.data.dataset_type == "csv_price_load"
    assert cfg.obs.builder_type == "default"
    assert cfg.train.num_envs == 1
    assert cfg.train.vec_env_type == "dummy"
    assert cfg.obs.sequence_features == ["price", "load"]


def test_compose_experiment_config_sets_default_lstm_artifact_path():
    cfg = compose_experiment_config(forecast_type="lstm")

    assert cfg.forecast.type == "lstm"
    assert cfg.forecast.lstm_model_path == get_default_lstm_artifact_paths()["model_path"]


@pytest.mark.parametrize(
    ("algorithm", "expected_head"),
    [("MADDPG", "single_q"), ("MATD3", "twin_q")],
)
def test_model_validation_sets_algorithm_specific_critic_head(algorithm, expected_head):
    cfg = ExperimentConfig()
    cfg.algo.name = algorithm
    cfg.model.critic_head_type = None

    validate_and_finalize_model_config(cfg)

    assert cfg.model.critic_head_type == expected_head


def test_model_validation_rejects_unknown_family():
    cfg = ExperimentConfig()
    cfg.model.family = "unknown"

    with pytest.raises(ValueError, match="model.family"):
        validate_and_finalize_model_config(cfg)