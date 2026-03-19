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
    """创建一个标准环境实例。

    根据配置构建数据集、奖励函数、预测器、观测构建器等组件，
    并实例化对应类型的环境对象。

    参数:
        cfg: 实验配置对象（ExperimentConfig）。
        mode: 运行模式，"train" 或 "test"。
        dataset: 数据集实例，为 None 时自动构建。
        reward_fn: 奖励函数实例，为 None 时自动构建。
        forecaster: 预测器实例，为 None 时自动构建。
        obs_builder: 观测构建器实例，为 None 时自动构建。

    返回:
        构建好的环境实例。
    """
    # 未提供组件时自动从配置构建
    if dataset is None:
        dataset = build_dataset(cfg, mode=mode)
    if reward_fn is None:
        reward_fn = get_reward_fn(cfg.reward.type, cfg)
    if forecaster is None:
        forecaster = build_forecaster(cfg)
    if obs_builder is None:
        obs_builder = build_obs_builder(cfg)

    # 根据环境类型获取对应的环境类
    env_cls = get_env_cls(cfg.env.env_type)

    # 电网潮流环境需要额外的 GridCore 组件
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
    """按配置创建训练侧向量化环境。

    根据 cfg.train.vec_env_type 选择 DummyVecEnv（单进程）或
    SubprocVecEnv（多进程）来并行运行多个训练环境。

    参数:
        cfg: 实验配置对象。
        seed: 随机种子，用于子进程环境的种子派生。

    返回:
        向量化环境实例（DummyVecEnv 或 SubprocVecEnv）。

    异常:
        ValueError: 当 vec_env_type 不是 'dummy' 或 'subproc' 时抛出。
    """
    # 单进程模式：所有环境共享同一进程
    if cfg.train.vec_env_type == "dummy":
        train_dataset = build_dataset(cfg, mode="train")

        def make_train_env():
            return build_env(cfg, mode="train", dataset=train_dataset)

        return DummyVecEnv(cfg.train.num_envs, make_train_env)

    # 多进程模式：每个环境运行在独立子进程中
    if cfg.train.vec_env_type == "subproc":
        return SubprocVecEnv(cfg.train.num_envs, cfg, mode="train", seed=seed)

    raise ValueError(
        f"Unknown train.vec_env_type '{cfg.train.vec_env_type}', expected 'dummy' or 'subproc'."
    )


def _finalize_runtime_from_env(cfg: Any, env: Any) -> None:
    """用评估环境回填运行时所需的派生信息。

    从已构建的环境中提取观测 schema、观测布局和动作维度，
    写入 cfg.runtime，供后续模型构建使用。

    参数:
        cfg: 实验配置对象，runtime 子配置将被原地修改。
        env: 已构建的环境实例，用于读取观测和动作空间信息。
    """
    cfg.runtime.observation_schema = dict(env.observation_schema)
    cfg.runtime.observation_layout = dict(env.observation_layout)
    cfg.runtime.action_dim = int(env.action_space[0].shape[0])


def build_train_runner(
    cfg: Any,
    seed: int = 0,
    env_name: str = "EnergyStorageEnv",
    number: int = 1,
) -> TrainRunner:
    """按统一配置创建完整的训练运行器（TrainRunner）。

    完整流程包括：配置运行时环境、校验模型配置、构建训练/评估环境，
    最终组装为 TrainRunner 实例。

    参数:
        cfg: 实验配置对象。
        seed: 全局随机种子。
        env_name: 环境名称标识，用于日志和 checkpoint 命名。
        number: 实验编号，用于区分同一配置的不同运行。

    返回:
        组装完成的 TrainRunner 实例。
    """
    cfg.runtime.seed = int(seed)
    # 配置 PyTorch 运行时（设备、种子、精度等）
    configure_torch_runtime(cfg, seed=seed)
    # 校验并补全模型配置（初次，部分字段可能尚缺）
    validate_and_finalize_model_config(cfg)

    # 如果使用 LSTM 预测器且未指定模型路径，自动训练/下载 LSTM 模型
    if cfg.forecast.type == "lstm" and cfg.forecast.lstm_model_path is None:
        ensure_lstm_artifacts(cfg, device=cfg.runtime.device)

    # 构建训练和评估环境
    train_env = _build_train_vec_env(cfg, seed=seed)
    eval_dataset = build_dataset(cfg, mode="test")
    eval_env = build_env(cfg, mode="test", dataset=eval_dataset)

    # 用评估环境回填运行时派生信息，再次校验模型配置
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
