"""Factory helpers for training environments and runners."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

from data.loaders.registry import build_dataset
from envs.grid.core.grid_core import GridCore
from envs.grid.deployments import build_agent_deployments
from envs.grid_env import GridEnv
from envs.observation.default_builder import DefaultObservationBuilder
from envs.rewards import NormalReward
from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_env import DummyVecEnv
from models import validate_and_finalize_model_config
from predictors.registry import build_forecaster
from predictors.training import ensure_lstm_artifacts
from scripts.train import TrainRunner
from scripts.utils.torch_runtime import configure_torch_runtime


def _build_dummy_train_vec_env(cfg: Any) -> Any:
    train_dataset = build_dataset(cfg, mode="train")

    def make_train_env():
        return build_env(cfg, mode="train", dataset=train_dataset)

    return DummyVecEnv(cfg.train.num_envs, make_train_env)


def _subproc_vec_env_is_supported_in_current_process() -> tuple[bool, str | None]:
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)

    if "ipykernel" in sys.modules or os.environ.get("JPY_PARENT_PID"):
        return (
            False,
            "Jupyter/IPython kernels do not reliably support spawn-based vector environments.",
        )

    if main_file is None:
        return False, "the current __main__ module has no importable file path."

    if str(main_file).startswith("<"):
        return False, f"the current __main__ entrypoint is {main_file!r}."

    return True, None


def build_env(
    cfg: Any,
    mode: str,
    dataset: Any | None = None,
    reward_fn: Any | None = None,
    forecaster: Any | None = None,
    obs_builder: Any | None = None,
) -> Any:
    if dataset is None:
        dataset = build_dataset(cfg, mode=mode)
    if reward_fn is None:
        reward_fn = NormalReward(cfg)
    if forecaster is None:
        forecaster = build_forecaster(cfg)
    if obs_builder is None:
        obs_builder = DefaultObservationBuilder(
            local_features=cfg.obs.local_features,
            sequence_features=cfg.obs.sequence_features,
            future_horizon=cfg.env.future_horizon,
            adjacency_type=cfg.obs.adjacency_type,
        )

    grid_core = GridCore(build_agent_deployments(cfg), cfg.grid)
    return GridEnv(
        cfg,
        mode=mode,
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=forecaster,
        obs_builder=obs_builder,
        grid_core=grid_core,
    )


def _build_train_vec_env(cfg: Any, *, seed: int) -> Any:
    if cfg.train.vec_env_type == "dummy":
        return _build_dummy_train_vec_env(cfg)

    if cfg.train.vec_env_type == "subproc":
        supported, reason = _subproc_vec_env_is_supported_in_current_process()
        if not supported:
            warnings.warn(
                "Falling back to DummyVecEnv because "
                f"`train.vec_env_type='subproc'` is unsupported in this session: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _build_dummy_train_vec_env(cfg)
        return SubprocVecEnv(cfg.train.num_envs, cfg, mode="train", seed=seed)

    raise ValueError(
        f"Unknown train.vec_env_type '{cfg.train.vec_env_type}', expected 'dummy' or 'subproc'."
    )


def _finalize_runtime_from_env(cfg: Any, env: Any) -> None:
    cfg.runtime.observation_schema = dict(env.observation_schema)
    cfg.runtime.observation_layout = dict(env.observation_layout)
    cfg.runtime.action_dim = int(env.action_space[0].shape[0])


def build_train_runner(
    cfg: Any,
    seed: int = 0,
    env_name: str = "GridEnv",
    number: int = 1,
) -> TrainRunner:
    cfg.runtime.seed = int(seed)
    configure_torch_runtime(cfg, seed=seed)
    validate_and_finalize_model_config(cfg)

    if cfg.forecast.type == "lstm":
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
