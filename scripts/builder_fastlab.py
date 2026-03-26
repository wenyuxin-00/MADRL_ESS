"""Builder helpers for the exact-equivalence MADRL fast-lab path."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

from data.loaders.registry import build_dataset
from envs.fastlab import (
    CachedObservationBuilderFastLab,
    GridCoreFastLab,
    GridEnvFastLab,
    SubprocVecEnvFastLab,
)
from envs.grid.core.grid_core import GridCore
from envs.grid.deployments import build_agent_deployments
from envs.observation.normalization import build_observation_normalizer
from envs.rewards import NormalReward
from envs.vec_env import DummyVecEnv
from models import validate_and_finalize_model_config
from predictors.oracle import PerfectForecaster
from scripts.builder import _finalize_runtime_from_env, build_env
from scripts.train import TrainRunner
from scripts.utils.torch_runtime import configure_torch_runtime


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


def build_fastlab_env(
    cfg: Any,
    mode: str,
    dataset: Any | None = None,
    reward_fn: Any | None = None,
    obs_builder: Any | None = None,
    grid_core: Any | None = None,
) -> Any:
    if dataset is None:
        dataset = build_dataset(cfg, mode=mode)
    if reward_fn is None:
        reward_fn = NormalReward(cfg)
    if obs_builder is None:
        obs_builder = CachedObservationBuilderFastLab(
            local_features=cfg.obs.local_features,
            sequence_features=cfg.obs.sequence_features,
            future_horizon=cfg.env.future_horizon,
            adjacency_type=cfg.obs.adjacency_type,
            normalizer=build_observation_normalizer(cfg),
        )
    if grid_core is None:
        grid_cls = GridCoreFastLab if bool(getattr(cfg.runtime, "fastlab_fast_grid_core", True)) else GridCore
        grid_core = grid_cls(build_agent_deployments(cfg), cfg.grid)

    return GridEnvFastLab(
        cfg,
        mode=mode,
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=PerfectForecaster(),
        obs_builder=obs_builder,
        grid_core=grid_core,
        observation_cache_dir=getattr(cfg.runtime, "fastlab_observation_cache_dir", None),
        train_info_mode=getattr(cfg.runtime, "fastlab_train_info_mode", "minimal"),
    )


def _build_dummy_train_vec_env_fastlab(cfg: Any) -> Any:
    train_dataset = build_dataset(cfg, mode="train")

    def make_train_env():
        return build_fastlab_env(cfg, mode="train", dataset=train_dataset)

    return DummyVecEnv(cfg.train.num_envs, make_train_env)


def _build_train_vec_env_fastlab(cfg: Any, *, seed: int) -> Any:
    if cfg.train.vec_env_type == "dummy":
        return _build_dummy_train_vec_env_fastlab(cfg)

    if cfg.train.vec_env_type == "subproc":
        supported, reason = _subproc_vec_env_is_supported_in_current_process()
        if not supported:
            warnings.warn(
                "Falling back to DummyVecEnv because "
                f"`train.vec_env_type='subproc'` is unsupported in this session: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _build_dummy_train_vec_env_fastlab(cfg)
        return SubprocVecEnvFastLab(cfg.train.num_envs, cfg, mode="train", seed=seed)

    raise ValueError(
        f"Unknown train.vec_env_type '{cfg.train.vec_env_type}', expected 'dummy' or 'subproc'."
    )


def build_train_runner_fastlab(
    cfg: Any,
    seed: int = 0,
    env_name: str = "GridEnv",
    number: int = 1,
) -> TrainRunner:
    if not getattr(cfg.runtime, "fastlab_observation_cache_dir", None):
        raise ValueError("Fast-lab training requires cfg.runtime.fastlab_observation_cache_dir to be set.")

    cfg.runtime.seed = int(seed)
    configure_torch_runtime(cfg, seed=seed)
    validate_and_finalize_model_config(cfg)

    train_env = _build_train_vec_env_fastlab(cfg, seed=seed)
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
