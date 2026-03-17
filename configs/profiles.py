"""notebook 友好的实验 profile 与组合入口。

目标是把实验切换项收敛成几组简单开关：
- 训练 profile：base / debug / fast_train
- 模型 family：mlp / transformer / graph
- 观测 profile：default / minimal / local_only
- 奖励与预测器 profile

notebook 只负责选择这些开关，不再手写大段样板配置。
"""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

import torch

from configs.experiment_config import ExperimentConfig
from forecast.artifacts import get_default_lstm_artifact_paths


def project_root() -> Path:
    """返回仓库根目录，便于 notebook 与脚本共用。"""
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    """返回默认数据目录。"""
    return project_root() / "data"


def make_base_config(data_dir: str | Path | None = None, device=None) -> ExperimentConfig:
    """创建适合 notebook 的基础配置。"""
    cfg = ExperimentConfig()
    cfg.data.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    cfg.runtime.device = torch.device(device) if device is not None else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg.train.vec_env_type = "dummy"
    return cfg


def apply_train_profile(cfg: ExperimentConfig, profile_name: str) -> ExperimentConfig:
    """应用训练规模 profile。"""
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
        cpu_workers = max(2, min(8, (os.cpu_count() or 8) // 2))
        cfg.train.train_episodes = 80
        cfg.train.max_train_steps = None
        cfg.train.num_envs = cpu_workers
        cfg.train.vec_env_type = "subproc"
        cfg.train.batch_size = 1024 if str(cfg.runtime.device).startswith("cuda") else 512
        cfg.train.buffer_size = 50000
        cfg.train.update_interval = 1
        cfg.train.updates_per_step = 1
        cfg.train.use_noise_decay = True
        return cfg

    raise ValueError(f"未知训练 profile: '{profile_name}'")


def apply_model_profile(cfg: ExperimentConfig, family: str) -> ExperimentConfig:
    """应用模型 family profile。"""
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

    raise ValueError(f"未知模型 family: '{family}'")


def apply_observation_profile(
    cfg: ExperimentConfig,
    profile_name: str = "default",
    *,
    local_features: list[str] | None = None,
    sequence_features: list[str] | None = None,
) -> ExperimentConfig:
    """应用观测 profile，必要时允许 notebook 直接覆写 feature 列表。"""
    if profile_name == "default":
        cfg.obs.local_features = ["time", "price", "load", "soc"]
        cfg.obs.sequence_features = ["price", "load"]
    elif profile_name == "minimal":
        cfg.obs.local_features = ["price", "load", "soc"]
        cfg.obs.sequence_features = ["price", "load"]
    elif profile_name == "local_only":
        cfg.obs.local_features = ["time", "price", "load", "soc"]
        cfg.obs.sequence_features = []
    else:
        raise ValueError(f"未知观测 profile: '{profile_name}'")

    if local_features is not None:
        cfg.obs.local_features = list(local_features)
    if sequence_features is not None:
        cfg.obs.sequence_features = list(sequence_features)
    return cfg


def apply_reward_profile(cfg: ExperimentConfig, reward_type: str) -> ExperimentConfig:
    """应用奖励函数选择。"""
    cfg.reward.type = reward_type
    return cfg


def apply_forecast_profile(cfg: ExperimentConfig, forecast_type: str) -> ExperimentConfig:
    """应用预测器选择。"""
    cfg.forecast.type = forecast_type
    if forecast_type == "lstm" and cfg.forecast.lstm_model_path is None:
        cfg.forecast.lstm_model_path = get_default_lstm_artifact_paths()["model_path"]
    return cfg


def compose_experiment_config(
    *,
    profile: str = "base",
    algorithm: str = "MADDPG",
    model_family: str = "mlp",
    reward_type: str = "composite",
    observation_profile: str = "default",
    forecast_type: str = "perfect",
    vec_env_type: str | None = None,
    env_type: str | None = None,
    dataset_type: str | None = None,
    obs_builder_type: str | None = None,
    local_features: list[str] | None = None,
    sequence_features: list[str] | None = None,
    data_dir: str | Path | None = None,
    device=None,
) -> ExperimentConfig:
    """按 notebook 常用开关组合出完整实验配置。"""
    cfg = make_base_config(data_dir=data_dir, device=device)
    apply_train_profile(cfg, profile)
    cfg.algo.name = algorithm
    apply_model_profile(cfg, model_family)
    apply_reward_profile(cfg, reward_type)
    apply_forecast_profile(cfg, forecast_type)
    apply_observation_profile(
        cfg,
        observation_profile,
        local_features=local_features,
        sequence_features=sequence_features,
    )

    if vec_env_type is not None:
        cfg.train.vec_env_type = vec_env_type
    if env_type is not None:
        cfg.env.env_type = env_type
    if dataset_type is not None:
        cfg.data.dataset_type = dataset_type
    if obs_builder_type is not None:
        cfg.obs.builder_type = obs_builder_type
    return cfg


def summarize_experiment(cfg: ExperimentConfig) -> dict:
    """返回 notebook 最常关心的实验摘要。"""
    summary = {
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
        "device": str(cfg.runtime.device),
        "data_dir": str(cfg.data.data_dir),
    }
    if cfg.forecast.type == "lstm":
        summary["lstm_model_path"] = str(cfg.forecast.lstm_model_path)
    return summary


def print_experiment_summary(cfg: ExperimentConfig) -> dict:
    """打印实验摘要，便于 notebook 中一眼检查当前设置。"""
    summary = summarize_experiment(cfg)
    pprint(summary)
    return summary
