"""Notebook-friendly configuration composition utilities."""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

from configs.experiment_config import ExperimentConfig
from envs.grid.config import DEFAULT_GRID_PROFILE, apply_grid_profile
from predictors.artifacts import get_default_lstm_artifact_dir
from scripts.utils.project_paths import get_data_root, project_root as resolve_project_root
from scripts.utils.torch_runtime import (
    PERFORMANCE_RUNTIME_MODE,
    STRICT_REPRO_RUNTIME_MODE,
    resolve_device,
    resolve_runtime_mode,
)


def project_root() -> Path:
    return resolve_project_root()


def default_data_dir() -> Path:
    return get_data_root()


def make_base_config(data_dir: str | Path | None = None, device=None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    cfg.runtime.device = resolve_device(device)
    return cfg


def apply_train_profile(cfg: ExperimentConfig, profile_name: str) -> ExperimentConfig:
    if profile_name == "base":
        return cfg
    if profile_name == "debug":
        cfg.train.train_episodes = 8
        cfg.train.max_train_steps = None
        cfg.train.num_envs = 1
        cfg.train.vec_env_type = "dummy"
        cfg.train.batch_size = 128
        cfg.train.buffer_size = 4096
        cfg.train.update_interval = 1
        cfg.train.updates_per_step = 1
        cfg.train.use_noise_decay = False
        return cfg
    if profile_name == "fast_train":
        cpu_workers = max(4, min(12, max(2, (os.cpu_count() or 8) - 2)))
        cfg.train.train_episodes = 300
        cfg.train.max_train_steps = None
        cfg.train.num_envs = cpu_workers
        cfg.train.vec_env_type = "subproc" if cpu_workers > 1 else "dummy"
        cfg.train.batch_size = 4096 if resolve_device(cfg.runtime.device).type == "cuda" else 1024
        cfg.train.buffer_size = 100000
        cfg.train.update_interval = 1
        cfg.train.updates_per_step = 1
        cfg.train.use_noise_decay = True
        return cfg
    raise ValueError(f"Unknown train profile: '{profile_name}'")


def apply_model_profile(cfg: ExperimentConfig, family: str) -> ExperimentConfig:
    cfg.model.family = family
    if family == "mlp":
        return cfg
    if family == "transformer":
        cfg.model.hidden_dim = 128
        cfg.model.transformer_num_heads = 4
        cfg.model.transformer_num_layers = 1
        return cfg
    if family == "graph":
        cfg.obs.adjacency_type = "fully_connected_no_self"
        cfg.model.graph_num_layers = 2
        return cfg
    raise ValueError(f"Unknown model family: '{family}'")


def apply_observation_profile(
    cfg: ExperimentConfig,
    profile_name: str = "default",
    *,
    local_features: list[str] | None = None,
    sequence_features: list[str] | None = None,
) -> ExperimentConfig:
    if profile_name == "default":
        cfg.obs.local_features = ["time", "price", "load", "soc"]
        cfg.obs.sequence_features = ["price", "load"]
    elif profile_name == "simbench":
        cfg.obs.local_features = ["time", "price", "load", "pv", "soc"]
        cfg.obs.sequence_features = ["price", "load", "pv"]
    elif profile_name == "minimal":
        cfg.obs.local_features = ["price", "load", "soc"]
        cfg.obs.sequence_features = ["price", "load"]
    elif profile_name == "local_only":
        cfg.obs.local_features = ["time", "price", "load", "soc"]
        cfg.obs.sequence_features = []
    else:
        raise ValueError(f"Unknown observation profile: '{profile_name}'")

    if local_features is not None:
        cfg.obs.local_features = list(local_features)
    if sequence_features is not None:
        cfg.obs.sequence_features = list(sequence_features)
    return cfg


def apply_reward_profile(cfg: ExperimentConfig, reward_type: str) -> ExperimentConfig:
    cfg.reward.type = reward_type
    return cfg


def apply_forecast_profile(cfg: ExperimentConfig, forecast_type: str) -> ExperimentConfig:
    cfg.forecast.type = forecast_type
    if forecast_type == "lstm" and cfg.forecast.lstm_artifact_root is None:
        cfg.forecast.lstm_artifact_root = get_default_lstm_artifact_dir()
    return cfg


def apply_runtime_profile(cfg: ExperimentConfig, runtime_mode: str) -> ExperimentConfig:
    cfg.runtime.execution_mode = resolve_runtime_mode(runtime_mode)
    cfg.runtime.allow_tf32 = None
    cfg.runtime.cudnn_benchmark = None
    cfg.runtime.cudnn_deterministic = None
    cfg.runtime.use_deterministic_algorithms = None
    cfg.runtime.pin_memory = None
    cfg.runtime.non_blocking_transfers = None
    return cfg


def compose_experiment_config(
    *,
    profile: str = "base",
    algorithm: str = "MADDPG",
    model_family: str = "mlp",
    reward_type: str | None = None,
    observation_profile: str | None = None,
    forecast_type: str | None = None,
    grid_profile: str | None = DEFAULT_GRID_PROFILE,
    vec_env_type: str | None = None,
    env_type: str | None = None,
    dataset_type: str | None = None,
    obs_builder_type: str | None = None,
    local_features: list[str] | None = None,
    sequence_features: list[str] | None = None,
    data_dir: str | Path | None = None,
    device=None,
    runtime_mode: str = PERFORMANCE_RUNTIME_MODE,
    seed: int = 0,
    require_cuda: bool | None = None,
) -> ExperimentConfig:
    cfg = make_base_config(data_dir=data_dir, device=device)
    apply_train_profile(cfg, profile)
    apply_runtime_profile(cfg, runtime_mode)
    cfg.algo.name = algorithm
    apply_model_profile(cfg, model_family)

    if grid_profile is not None:
        apply_grid_profile(cfg, grid_profile)

    if reward_type is not None:
        apply_reward_profile(cfg, reward_type)
    if forecast_type is not None:
        apply_forecast_profile(cfg, forecast_type)
    if observation_profile is not None:
        apply_observation_profile(
            cfg,
            observation_profile,
            local_features=local_features,
            sequence_features=sequence_features,
        )
    else:
        if local_features is not None:
            cfg.obs.local_features = list(local_features)
        if sequence_features is not None:
            cfg.obs.sequence_features = list(sequence_features)

    if vec_env_type is not None:
        cfg.train.vec_env_type = vec_env_type
    if env_type is not None:
        cfg.env.env_type = env_type
    if dataset_type is not None:
        cfg.data.dataset_type = dataset_type
    if obs_builder_type is not None:
        cfg.obs.builder_type = obs_builder_type

    cfg.runtime.seed = int(seed)
    if require_cuda is not None:
        cfg.runtime.require_cuda = bool(require_cuda)
    return cfg


def _derive_training_budget(cfg: ExperimentConfig) -> dict[str, int | str | None]:
    episode_limit = max(1, int(cfg.env.episode_limit))
    num_envs = max(1, int(cfg.train.num_envs))
    resolved_steps = int(cfg.train.resolved_max_train_steps(episode_limit))
    target_parallel_iters = resolved_steps // num_envs
    completed_rounds = target_parallel_iters // episode_limit
    expected_completed_episodes = completed_rounds * num_envs
    partial_steps_per_env = target_parallel_iters % episode_limit
    budget_source = (
        "max_train_steps"
        if cfg.train.max_train_steps is not None
        else "train_episodes * episode_limit"
    )
    return {
        "episode_limit": episode_limit,
        "future_horizon": int(cfg.env.future_horizon),
        "max_train_steps": int(cfg.train.max_train_steps) if cfg.train.max_train_steps is not None else None,
        "resolved_train_steps": resolved_steps,
        "parallel_rollout_iterations": int(target_parallel_iters),
        "expected_completed_episodes_floor": int(expected_completed_episodes),
        "partial_steps_per_env_at_stop": int(partial_steps_per_env),
        "budget_source": budget_source,
    }


def summarize_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    budget = _derive_training_budget(cfg)
    summary: dict[str, object] = {
        "algo": cfg.algo.name,
        "env_type": cfg.env.env_type,
        "dataset_type": cfg.data.dataset_type,
        "obs_builder_type": cfg.obs.builder_type,
        "model_family": cfg.model.family,
        "reward": cfg.reward.type,
        "forecast": cfg.forecast.type,
        "local_features": list(cfg.obs.local_features),
        "sequence_features": list(cfg.obs.sequence_features),
        "vec_env": cfg.train.vec_env_type,
        "num_envs": cfg.train.num_envs,
        "batch_size": cfg.train.batch_size,
        "train_episodes": cfg.train.train_episodes,
        "episode_limit": budget["episode_limit"],
        "future_horizon": budget["future_horizon"],
        "max_train_steps": budget["max_train_steps"],
        "resolved_train_steps": budget["resolved_train_steps"],
        "parallel_rollout_iterations": budget["parallel_rollout_iterations"],
        "expected_completed_episodes_floor": budget["expected_completed_episodes_floor"],
        "partial_steps_per_env_at_stop": budget["partial_steps_per_env_at_stop"],
        "train_budget_source": budget["budget_source"],
        "device": str(cfg.runtime.device),
        "runtime_mode": cfg.runtime.execution_mode,
        "seed": int(cfg.runtime.seed),
        "data_dir": str(cfg.data.data_dir),
        "grid_profile": DEFAULT_GRID_PROFILE,
        "grid_sb_code": cfg.grid.sb_code,
        "grid_agent_bus_ids": list(cfg.grid.agent_bus_ids),
    }
    if cfg.forecast.type == "lstm":
        summary["forecast_signals"] = list(cfg.forecast.target_signals)
        summary["history_window"] = int(cfg.forecast.history_window)
        summary["artifact_root"] = str(cfg.forecast.lstm_artifact_root or get_default_lstm_artifact_dir())
        if cfg.forecast.lstm_model_path is not None:
            summary["legacy_lstm_model_path"] = str(cfg.forecast.lstm_model_path)
    if cfg.runtime.execution_mode == STRICT_REPRO_RUNTIME_MODE:
        summary["strict_reproducibility"] = True
    return summary


def print_experiment_summary(cfg: ExperimentConfig) -> dict[str, object]:
    summary = summarize_experiment(cfg)
    pprint(summary)
    return summary
