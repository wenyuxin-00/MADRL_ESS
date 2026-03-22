"""实验 profile 组合入口（含电网 grid profile）。

提供 notebook 友好的函数，把实验切换项收敛成几组简单开关：
- 训练 profile：base / debug / fast_train
- 模型 family：mlp / transformer / graph
- 观测 profile：default / minimal / local_only
- 奖励与预测器 profile
- 电网 profile：rural1_phase1 等（原 grid_profiles.py）

主要函数:
    compose_experiment_config -- 按 notebook 常用开关组合出完整实验配置
    apply_grid_profile       -- 在已有配置上叠加电网拓扑方案
    summarize_experiment     -- 返回 notebook 常关心的实验摘要
    print_experiment_summary -- 打印摘要

典型用法::
    cfg = compose_experiment_config(profile="debug", algorithm="MADDPG")
    cfg = apply_grid_profile(cfg, "rural1_phase1")
"""

from __future__ import annotations

import os
from typing import Any
from pathlib import Path
from pprint import pprint

from scripts.utils.torch_runtime import (
    PERFORMANCE_RUNTIME_MODE,
    STRICT_REPRO_RUNTIME_MODE,
    resolve_device,
    resolve_runtime_mode,
)
from scripts.utils.project_paths import get_data_root, project_root as resolve_project_root
from configs.experiment_config import ExperimentConfig
from predictors.artifacts import get_default_lstm_artifact_dir


def project_root() -> Path:
    """Return the repository root."""
    return resolve_project_root()


def default_data_dir() -> Path:
    """Return the default dataset directory."""
    return get_data_root()


def make_base_config(data_dir: str | Path | None = None, device=None) -> ExperimentConfig:
    """创建适合 notebook 的基础配置。"""
    cfg = ExperimentConfig()
    cfg.data.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    cfg.runtime.device = resolve_device(device)
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
    if forecast_type == "lstm":
        if cfg.forecast.lstm_artifact_root is None:
            cfg.forecast.lstm_artifact_root = get_default_lstm_artifact_dir()
    return cfg


def apply_runtime_profile(cfg: ExperimentConfig, runtime_mode: str) -> ExperimentConfig:
    """设置 performance / strict reproducibility 模式。"""
    cfg.runtime.execution_mode = resolve_runtime_mode(runtime_mode)
    # 这些字段默认交给统一 runtime 入口根据模式推导，避免局部配置漂移。
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
    runtime_mode: str = PERFORMANCE_RUNTIME_MODE,
    seed: int = 0,
    require_cuda: bool | None = None,
) -> ExperimentConfig:
    """按 notebook 常用开关组合出完整实验配置。"""
    cfg = make_base_config(data_dir=data_dir, device=device)
    apply_train_profile(cfg, profile)
    apply_runtime_profile(cfg, runtime_mode)
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
    cfg.runtime.seed = int(seed)
    if require_cuda is not None:
        cfg.runtime.require_cuda = bool(require_cuda)
    return cfg


def _derive_training_budget(cfg: ExperimentConfig) -> dict[str, int | str | None]:
    """返回训练预算的关键派生量，方便 notebook 解释 step/episode 的关系。"""
    episode_limit = max(1, int(cfg.env.episode_limit))
    num_envs = max(1, int(cfg.train.num_envs))
    resolved_steps = int(cfg.train.resolved_max_train_steps(episode_limit))
    target_parallel_iters = resolved_steps // num_envs
    completed_rounds = target_parallel_iters // episode_limit
    expected_completed_episodes = completed_rounds * num_envs
    partial_steps_per_env = target_parallel_iters % episode_limit
    budget_source = "max_train_steps" if cfg.train.max_train_steps is not None else "train_episodes * episode_limit"
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


def summarize_experiment(cfg: ExperimentConfig) -> dict:
    """返回 notebook 最常关心的实验摘要。"""
    budget = _derive_training_budget(cfg)
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
    }
    if cfg.forecast.type == "lstm":
        summary["forecast_signals"] = list(cfg.forecast.target_signals)
        summary["history_window"] = int(cfg.forecast.history_window)
        summary["artifact_root"] = str(
            cfg.forecast.lstm_artifact_root or get_default_lstm_artifact_dir()
        )
        if cfg.forecast.lstm_model_path is not None:
            summary["legacy_lstm_model_path"] = str(cfg.forecast.lstm_model_path)
    if cfg.runtime.execution_mode == STRICT_REPRO_RUNTIME_MODE:
        summary["strict_reproducibility"] = True
    return summary


def print_experiment_summary(cfg: ExperimentConfig) -> dict:
    """打印实验摘要，便于 notebook 中一眼检查当前设置。"""
    summary = summarize_experiment(cfg)
    pprint(summary)
    return summary


# ---------------------------------------------------------------------------
# Grid profiles（原 grid_profiles.py 迁入）
# ---------------------------------------------------------------------------

def apply_grid_profile(cfg: Any, profile: str = "rural1_phase1") -> Any:
    """Apply a named grid profile to an existing ``ExperimentConfig``.

    Parameters
    ----------
    cfg:
        Any ``ExperimentConfig`` instance, typically built with
        ``compose_experiment_config()`` first.
    profile:
        Name of the grid profile to apply.

    Returns
    -------
    ExperimentConfig
        The same object, mutated in-place and returned for chaining.

    Raises
    ------
    ValueError
        If *profile* is not recognised.
    """
    if profile == "rural1_phase1":
        _apply_rural1_phase1(cfg)
    else:
        raise ValueError(
            f"Unknown grid profile '{profile}'. Available: ['rural1_phase1']"
        )
    return cfg


def _apply_rural1_phase1(cfg: Any) -> None:
    """Phase 1: SimBench 1-LV-rural1--0-sw, 3 prosumer agents, p_batt only."""
    # Environment.
    cfg.env.env_type = "grid_pf"
    cfg.env.num_agents = 3
    cfg.env.episode_limit = 96 * 2          # 2 days × 96 steps/day
    cfg.env.future_horizon = 24             # 6-hour look-ahead (24 × 15 min)
    cfg.env.battery_capacity = 5.0          # kWh
    cfg.env.max_charge_rate = 2.5           # kW
    cfg.env.efficiency = 0.95
    cfg.env.init_soc = 0.5
    cfg.env.soc_min = 0.05
    cfg.env.soc_max = 0.95
    cfg.env.soc_target = 0.5
    cfg.env.dt = 0.25                       # 15-minute intervals
    cfg.env.storage_power_scale = 12.0
    cfg.env.storage_capacity_scale = 12.0

    # Grid.
    cfg.grid.sb_code = "1-LV-rural1--0-sw"
    cfg.grid.pf_solver = "nr"
    cfg.grid.agent_bus_ids = [10, 6, 12]
    cfg.grid.v_min_pu = 0.95
    cfg.grid.v_max_pu = 1.05
    cfg.grid.line_max_loading_pct = 100.0
    cfg.grid.w_v_pen = 10.0
    cfg.grid.w_l_pen = 10.0
    cfg.grid.w_line_pen = 10.0
    cfg.grid.w_trafo_pen = 10.0

    # Dataset.
    cfg.data.dataset_type = "csv_prosumer"

    # Reward.
    cfg.reward.type = "grid_composite"
    cfg.reward.w_pen = 6.0
    cfg.reward.w_soc = 0.30
    cfg.reward.lambda_bonus = 0.001

    # Observation — same as simbench profile.
    cfg.obs.local_features = ["time", "price", "load", "pv", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.obs.adjacency_type = "identity"

    # Forecast — oracle (perfect knowledge) for Phase 1.
    cfg.forecast.type = "perfect"
    cfg.forecast.target_signals = ["price", "load", "pv"]

    # Training — Phase 1 uses a single environment to avoid multi-process
    # pandapower serialisation issues.
    cfg.train.num_envs = 1
    cfg.train.vec_env_type = "dummy"
