"""实验组件构建入口。

根据 ExperimentConfig 构建环境、控制器、预测器、数据集等
训练所需的各个组件，是训练流程的核心装配逻辑。

主要函数:
    build_env          -- 构建环境
    build_train_runner -- 构建完整训练运行器
"""

from __future__ import annotations

from typing import Any

from envs.rewards import get_reward_fn
from envs.subproc_vec_env import SubprocVecEnv
from scripts.utils.torch_runtime import configure_torch_runtime
from envs.vec_env import DummyVecEnv
from data.loaders.registry import build_dataset
from envs.observation.registry import build_obs_builder
from envs.registry import get_env_cls
from predictors.registry import build_forecaster
from predictors.training import ensure_lstm_artifacts
from models import validate_and_finalize_model_config
from scripts.train import TrainRunner


def build_env(
    cfg: Any,
    mode: str,
    dataset: Any | None = None,
    reward_fn: Any | None = None,
    forecaster: Any | None = None,
    obs_builder: Any | None = None,
) -> Any:
    """创建一个标准环境实例。"""
    if dataset is None:
        dataset = build_dataset(cfg, mode=mode)
    if reward_fn is None:
        reward_fn = get_reward_fn(cfg.reward.type, cfg)
    if forecaster is None:
        forecaster = build_forecaster(cfg)
    if obs_builder is None:
        obs_builder = build_obs_builder(cfg)

    env_cls = get_env_cls(cfg.env.env_type)

    if cfg.env.env_type == "grid_pf":
        from envs.grid.core.grid_core import GridCore
        from envs.grid.config.grid_config import build_agent_deployments

        grid_core = GridCore(build_agent_deployments(cfg), cfg.grid)
        return env_cls(
            cfg,
            mode=mode,
            dataset=dataset,
            reward_fn=reward_fn,
            forecaster=forecaster,
            obs_builder=obs_builder,
            grid_core=grid_core,
        )

    return env_cls(
        cfg,
        mode=mode,
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=forecaster,
        obs_builder=obs_builder,
    )


def _build_train_vec_env(cfg: Any, *, seed: int) -> Any:
    """按配置创建训练侧向量环境。"""
    if cfg.train.vec_env_type == "dummy":
        train_dataset = build_dataset(cfg, mode="train")

        def make_train_env():
            return build_env(cfg, mode="train", dataset=train_dataset)

        return DummyVecEnv(cfg.train.num_envs, make_train_env)

    if cfg.train.vec_env_type == "subproc":
        return SubprocVecEnv(cfg.train.num_envs, cfg, mode="train", seed=seed)

    raise ValueError(
        f"Unknown train.vec_env_type '{cfg.train.vec_env_type}', expected 'dummy' or 'subproc'."
    )


def _finalize_runtime_from_env(cfg: Any, env: Any) -> None:
    """用评估环境回填运行时所需的派生信息。"""
    cfg.runtime.observation_schema = dict(env.observation_schema)
    cfg.runtime.observation_layout = dict(env.observation_layout)
    cfg.runtime.action_dim = int(env.action_space[0].shape[0])


def build_train_runner(
    cfg: Any,
    seed: int = 0,
    env_name: str = "EnergyStorageEnv",
    number: int = 1,
) -> TrainRunner:
    """按统一配置创建训练 runner。"""
    cfg.runtime.seed = int(seed)
    configure_torch_runtime(cfg, seed=seed)
    validate_and_finalize_model_config(cfg)

    if cfg.forecast.type == "lstm" and cfg.forecast.lstm_model_path is None:
        ensure_lstm_artifacts(cfg, device=cfg.runtime.device)

    train_env = _build_train_vec_env(cfg, seed=seed)
    eval_dataset = build_dataset(cfg, mode="test")
    eval_env = build_env(cfg, mode="test", dataset=eval_dataset)
    _finalize_runtime_from_env(cfg, eval_env)
    validate_and_finalize_model_config(cfg)

    return TrainRunner(
        cfg,
        train_env=train_env,
        eval_env=eval_env,
        env_name=env_name,
        number=number,
        seed=seed,
    )
