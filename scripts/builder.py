"""Factory helpers for shared grid environment construction."""

from __future__ import annotations

from typing import Any

from data.loaders.registry import build_dataset
from envs.grid.core.grid_core import GridCore
from envs.grid.deployments import build_agent_deployments
from envs.grid_env import GridEnv
from envs.observation.default_builder import DefaultObservationBuilder
from envs.rewards import NormalReward
from predictors.registry import build_forecaster


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
