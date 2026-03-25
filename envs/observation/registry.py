"""Helpers for the default observation builder used by the grid mainline."""

from __future__ import annotations

from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.normalization import build_observation_normalizer

DEFAULT_OBS_BUILDER_NAME = "default"
SUPPORTED_OBS_BUILDERS = (DEFAULT_OBS_BUILDER_NAME,)


def get_obs_builder_cls(name: str | None = None) -> type[DefaultObservationBuilder]:
    normalized = DEFAULT_OBS_BUILDER_NAME if name is None else str(name).strip().lower()
    if normalized not in SUPPORTED_OBS_BUILDERS:
        raise ValueError(
            f"Unsupported observation builder '{name}'. Only the default builder is available."
        )
    return DefaultObservationBuilder


def build_obs_builder(cfg) -> DefaultObservationBuilder:
    return DefaultObservationBuilder(
        local_features=cfg.obs.local_features,
        sequence_features=cfg.obs.sequence_features,
        future_horizon=cfg.env.future_horizon,
        adjacency_type=cfg.obs.adjacency_type,
        normalizer=build_observation_normalizer(cfg),
    )
