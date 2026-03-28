"""Factory helpers for the single-stack cached Grid MADRL mainline."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

from data.loaders.registry import build_dataset
from envs.grid.core.grid_core import GridCore
from envs.grid.deployments import build_agent_deployments
from envs.grid_env import GridEnv
from envs.observation.cached_builder import CachedObservationBuilder
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.normalization import build_observation_normalizer
from envs.rewards import NormalReward
from envs.subproc_vec_env import SubprocVecEnv
from envs.vec_env import DummyVecEnv
from models import validate_and_finalize_model_config
from predictors.registry import build_forecaster
from scripts.train import TrainRunner
from scripts.utils.madrl_observation_cache_lab import build_or_load_observation_cache
from scripts.utils.torch_runtime import configure_torch_runtime


def _runtime_cache_results(cfg: Any) -> dict[str, Any]:
    cached = getattr(cfg.runtime, "_observation_cache_results", None)
    if not isinstance(cached, dict):
        cached = {}
        setattr(cfg.runtime, "_observation_cache_results", cached)
    return cached


def _runtime_cache_dirs(cfg: Any) -> dict[str, str]:
    cached = getattr(cfg.runtime, "_observation_cache_dirs", None)
    if not isinstance(cached, dict):
        cached = {}
        setattr(cfg.runtime, "_observation_cache_dirs", cached)
    return cached


def _resolve_cache_root(cfg: Any) -> str | Path | None:
    return getattr(cfg.runtime, "observation_cache_root", None)


def _resolve_cache_batch_size(cfg: Any) -> int:
    return int(getattr(cfg.runtime, "observation_cache_batch_size", 8192))


def _resolve_cache_refresh(cfg: Any) -> bool:
    return bool(getattr(cfg.runtime, "refresh_observation_cache", False))


def _prepare_observation_cache(
    cfg: Any,
    *,
    split: str,
    refresh: bool | None = None,
) -> Any:
    split_name = str(split)
    cached_results = _runtime_cache_results(cfg)
    if split_name in cached_results:
        return cached_results[split_name]

    cache_result = build_or_load_observation_cache(
        cfg,
        split=split_name,
        forecast_ready=getattr(cfg.runtime, "forecast_ready", None),
        refresh=_resolve_cache_refresh(cfg) if refresh is None else bool(refresh),
        root=_resolve_cache_root(cfg),
        batch_size=_resolve_cache_batch_size(cfg),
    )
    cached_results[split_name] = cache_result
    _runtime_cache_dirs(cfg)[split_name] = str(cache_result.cache_dir)
    return cache_result


def _cache_dir_for_split(cfg: Any, split: str) -> Path | None:
    split_name = str(split)
    cached_dirs = _runtime_cache_dirs(cfg)
    if split_name not in cached_dirs:
        cached_dirs[split_name] = str(_prepare_observation_cache(cfg, split=split_name).cache_dir)
    cache_dir = cached_dirs.get(split_name)
    return None if not cache_dir else Path(cache_dir).resolve()


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
    cache_dir = _cache_dir_for_split(cfg, mode) if str(mode) in {"train", "test"} else None
    if dataset is None:
        dataset = build_dataset(cfg, mode=mode)
    if reward_fn is None:
        reward_fn = NormalReward(cfg)
    if forecaster is None:
        forecaster = build_forecaster(cfg)
    if obs_builder is None:
        normalizer = build_observation_normalizer(cfg)
        builder_cls = CachedObservationBuilder if cache_dir is not None else DefaultObservationBuilder
        obs_builder = builder_cls(
            local_features=cfg.obs.local_features,
            sequence_features=cfg.obs.sequence_features,
            future_horizon=cfg.env.future_horizon,
            adjacency_type=cfg.obs.adjacency_type,
            normalizer=normalizer,
        )

    grid_core = GridCore(build_agent_deployments(cfg), cfg.grid)
    env = GridEnv(
        cfg,
        mode=mode,
        dataset=dataset,
        reward_fn=reward_fn,
        forecaster=forecaster,
        obs_builder=obs_builder,
        grid_core=grid_core,
        observation_cache_dir=cache_dir,
    )
    if cache_dir is not None:
        env.cache_dir = str(cache_dir)
    return env


def _build_dummy_train_vec_env(cfg: Any) -> Any:
    train_dataset = build_dataset(cfg, mode="train")

    def make_train_env():
        return build_env(cfg, mode="train", dataset=train_dataset)

    return DummyVecEnv(cfg.train.num_envs, make_train_env)


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

    train_cache_result = _prepare_observation_cache(cfg, split="train")
    test_cache_result = _prepare_observation_cache(cfg, split="test", refresh=False)

    train_env = _build_train_vec_env(cfg, seed=seed)
    eval_dataset = build_dataset(cfg, mode="test")
    eval_env = build_env(cfg, mode="test", dataset=eval_dataset)

    _finalize_runtime_from_env(cfg, eval_env)
    validate_and_finalize_model_config(cfg)

    runner = TrainRunner(
        cfg,
        train_env=train_env,
        eval_env=eval_env,
        env_name=env_name,
        number=number,
        seed=seed,
    )
    runner.cache_metadata = {
        "train": {
            "cache_dir": str(train_cache_result.cache_dir),
            "cache_hit": bool(train_cache_result.cache_hit),
            "cache_build_time_s": float(train_cache_result.cache_build_time_s),
        },
        "test": {
            "cache_dir": str(test_cache_result.cache_dir),
            "cache_hit": bool(test_cache_result.cache_hit),
            "cache_build_time_s": float(test_cache_result.cache_build_time_s),
        },
        "cache_root": None if _resolve_cache_root(cfg) is None else str(Path(_resolve_cache_root(cfg)).resolve()),
        "observation_cache_batch_size": _resolve_cache_batch_size(cfg),
    }
    return runner


__all__ = [
    "_build_dummy_train_vec_env",
    "_build_train_vec_env",
    "_finalize_runtime_from_env",
    "_subproc_vec_env_is_supported_in_current_process",
    "build_env",
    "build_train_runner",
]
