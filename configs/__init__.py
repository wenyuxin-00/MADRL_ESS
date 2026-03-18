"""Experiment configuration system.
实验配置系统。

Architecture / 架构:
    - experiment_config.py  -- Dataclass hierarchy (EnvConfig, AlgoConfig, ModelConfig, ...)
                               composed into a single ExperimentConfig
    - profiles.py           -- Notebook-friendly composition functions that apply
                               predefined profiles (debug, fast_train, mlp, transformer, ...)

Typical usage / 典型用法::

    from configs import compose_experiment_config, print_experiment_summary

    cfg = compose_experiment_config(
        train_profile="debug",
        model_family="mlp",
        algo_name="MADDPG",
    )
    print_experiment_summary(cfg)
"""

from configs.experiment_config import (
    AlgoConfig,
    DataConfig,
    EnvConfig,
    ExperimentConfig,
    ForecastConfig,
    ModelConfig,
    ObsConfig,
    RewardConfig,
    RuntimeConfig,
    TrainConfig,
)
from configs.profiles import (
    apply_grid_profile,
    apply_forecast_profile,
    apply_model_profile,
    apply_observation_profile,
    apply_reward_profile,
    apply_runtime_profile,
    apply_train_profile,
    compose_experiment_config,
    default_data_dir,
    make_base_config,
    print_experiment_summary,
    project_root,
    summarize_experiment,
)

__all__ = [
    "AlgoConfig",
    "DataConfig",
    "EnvConfig",
    "ExperimentConfig",
    "ForecastConfig",
    "ModelConfig",
    "ObsConfig",
    "RewardConfig",
    "RuntimeConfig",
    "TrainConfig",
    "apply_forecast_profile",
    "apply_grid_profile",
    "apply_model_profile",
    "apply_observation_profile",
    "apply_reward_profile",
    "apply_runtime_profile",
    "apply_train_profile",
    "compose_experiment_config",
    "default_data_dir",
    "make_base_config",
    "print_experiment_summary",
    "project_root",
    "summarize_experiment",
]
