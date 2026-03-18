import pytest

from configs import compose_experiment_config
from configs.profiles import summarize_experiment
from configs.experiment_config import ExperimentConfig
from common.torch_runtime import STRICT_REPRO_RUNTIME_MODE
from forecast.artifacts import get_default_lstm_artifact_dir
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


def test_compose_experiment_config_sets_default_lstm_artifact_root():
    cfg = compose_experiment_config(forecast_type="lstm")

    assert cfg.forecast.type == "lstm"
    assert cfg.forecast.lstm_artifact_root == get_default_lstm_artifact_dir()
    assert cfg.forecast.lstm_model_path is None


def test_compose_experiment_config_supports_runtime_mode_and_seed():
    cfg = compose_experiment_config(runtime_mode=STRICT_REPRO_RUNTIME_MODE, seed=7)

    assert cfg.runtime.execution_mode == STRICT_REPRO_RUNTIME_MODE
    assert cfg.runtime.seed == 7


def test_summarize_experiment_reports_parallel_training_budget():
    cfg = compose_experiment_config(profile="fast_train")
    cfg.train.train_episodes = 200
    cfg.train.max_train_steps = None
    cfg.train.num_envs = 12
    cfg.env.episode_limit = 96 * 2

    summary = summarize_experiment(cfg)

    assert summary["train_budget_source"] == "train_episodes * episode_limit"
    assert summary["resolved_train_steps"] == 200 * (96 * 2)
    assert summary["parallel_rollout_iterations"] == (200 * (96 * 2)) // 12
    assert summary["expected_completed_episodes_floor"] == 192
    assert summary["partial_steps_per_env_at_stop"] == 128


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
