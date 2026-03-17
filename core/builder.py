"""训练与评估对象的统一构建入口。

这里把 dataset、env、forecast、reward、runner 串起来，
让 notebook、脚本和测试只需要关心“我要跑什么实验配置”。
"""

from __future__ import annotations

from common.rewards import get_reward_fn
from common.subproc_vec_env import SubprocVecEnv
from common.vec_env import DummyVecEnv
from datasets.registry import build_dataset
from envs.observation.registry import build_obs_builder
from envs.registry import get_env_cls
from forecast.registry import build_forecaster
from models import validate_and_finalize_model_config
from runners.train_runner import TrainRunner


def build_env(cfg, mode: str, dataset=None, reward_fn=None, forecaster=None, obs_builder=None):
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
    return env_cls(
        cfg,
        mode=mode,
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=forecaster,
        obs_builder=obs_builder,
    )


def _build_train_vec_env(cfg):
    """按配置创建训练侧向量环境。"""
    if cfg.train.vec_env_type == "dummy":
        train_dataset = build_dataset(cfg, mode="train")

        def make_train_env():
            return build_env(cfg, mode="train", dataset=train_dataset)

        return DummyVecEnv(cfg.train.num_envs, make_train_env)

    if cfg.train.vec_env_type == "subproc":
        return SubprocVecEnv(cfg.train.num_envs, cfg, mode="train")

    raise ValueError(
        f"Unknown train.vec_env_type '{cfg.train.vec_env_type}', expected 'dummy' or 'subproc'."
    )


def _finalize_runtime_from_env(cfg, env) -> None:
    """用评估环境回填运行时所需的派生信息。"""
    cfg.runtime.observation_schema = dict(env.observation_schema)
    cfg.runtime.observation_layout = dict(env.observation_layout)
    cfg.runtime.action_dim = int(env.action_space[0].shape[0])


def build_train_runner(
    cfg,
    seed: int = 0,
    env_name: str = "EnergyStorageEnv",
    number: int = 1,
) -> TrainRunner:
    """按统一配置创建训练 runner。"""
    validate_and_finalize_model_config(cfg)

    train_env = _build_train_vec_env(cfg)
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
