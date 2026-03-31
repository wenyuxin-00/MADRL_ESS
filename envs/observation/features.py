"""Observation feature registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from envs.observation.feature_blocks import (
    broadcast_scalar_feature,
    build_adjacency,
    build_calendar_time_features,
    build_time_features,
    pad_sequence_1d,
    pad_sequence_2d,
    reshape_agent_scalar_feature,
)


@dataclass(frozen=True)
class ObservationFeatureSpec:
    name: str
    group: str
    dim: int
    scope: str
    builder: Callable
    description: str = ""

    def field_name(self) -> str:
        return self.name if self.group == "local" else f"{self.name}_seq"

    def schema(self, n_agents: int, sequence_length: int) -> tuple[int, ...]:
        if self.group == "local":
            return (int(n_agents), int(self.dim))
        if self.scope == "shared":
            return (sequence_length,) if self.dim == 1 else (sequence_length, int(self.dim))
        if self.scope == "per_agent":
            return (
                (int(n_agents), sequence_length)
                if self.dim == 1
                else (int(n_agents), sequence_length, int(self.dim))
            )
        raise ValueError(f"Unknown feature scope '{self.scope}'.")

    def layout(self) -> dict[str, str | int]:
        return {
            "feature_name": self.name,
            "group": self.group,
            "scope": self.scope,
            "dim": int(self.dim),
            "description": self.description,
        }


LOCAL_FEATURES: dict[str, ObservationFeatureSpec] = {}
SEQUENCE_FEATURES: dict[str, ObservationFeatureSpec] = {}


def register_local_feature(spec: ObservationFeatureSpec) -> None:
    if spec.group != "local":
        raise ValueError("register_local_feature only accepts group='local' features.")
    LOCAL_FEATURES[spec.name] = spec


def register_sequence_feature(spec: ObservationFeatureSpec) -> None:
    if spec.group != "sequence":
        raise ValueError("register_sequence_feature only accepts group='sequence' features.")
    SEQUENCE_FEATURES[spec.name] = spec


def get_local_feature_spec(name: str) -> ObservationFeatureSpec:
    if name not in LOCAL_FEATURES:
        raise ValueError(f"Unknown local feature '{name}', available: {list(LOCAL_FEATURES)}")
    return LOCAL_FEATURES[name]


def get_sequence_feature_spec(name: str) -> ObservationFeatureSpec:
    if name not in SEQUENCE_FEATURES:
        raise ValueError(f"Unknown sequence feature '{name}', available: {list(SEQUENCE_FEATURES)}")
    return SEQUENCE_FEATURES[name]


def resolve_local_features(names: list[str]) -> list[ObservationFeatureSpec]:
    return [get_local_feature_spec(name) for name in names]


def resolve_sequence_features(names: list[str]) -> list[ObservationFeatureSpec]:
    return [get_sequence_feature_spec(name) for name in names]


def _current_timestamp_value(env) -> str:
    timestamps = list(dict(getattr(env, "episode_meta", {})).get("timestamps") or [])
    if 0 <= int(env.cur_step) < len(timestamps):
        return str(timestamps[int(env.cur_step)])
    fallback = pd.Timestamp("2000-01-01 00:00:00+00:00") + pd.Timedelta(minutes=15 * int(env.cur_step))
    return str(fallback)


def _history_timestamps(env) -> list[str]:
    if hasattr(env, "get_signal_history_timestamps"):
        return [str(timestamp) for timestamp in env.get_signal_history_timestamps()]
    timestamps = list(dict(getattr(env, "episode_meta", {})).get("timestamps") or [])
    if not timestamps:
        return []
    end_idx = max(0, int(env.cur_step) + 1)
    return [str(timestamp) for timestamp in timestamps[:end_idx]]


def _current_shared_signal_feature(signal_name: str, description: str) -> ObservationFeatureSpec:
    return ObservationFeatureSpec(
        name=signal_name,
        group="local",
        dim=1,
        scope="shared",
        builder=lambda env, _: broadcast_scalar_feature(env.get_signal_step(signal_name), env.n),
        description=description,
    )


def _current_per_agent_signal_feature(signal_name: str, description: str) -> ObservationFeatureSpec:
    return ObservationFeatureSpec(
        name=signal_name,
        group="local",
        dim=1,
        scope="per_agent",
        builder=lambda env, _: reshape_agent_scalar_feature(env.get_signal_step(signal_name)),
        description=description,
    )


def _shared_signal_sequence_feature(
    signal_name: str,
    description: str,
    *,
    use_forecaster: bool = False,
) -> ObservationFeatureSpec:
    def builder(env, sequence_length: int) -> np.ndarray:
        if use_forecaster:
            return env.forecaster.predict(
                env.get_signal_history(signal_name),
                sequence_length,
                signal_name=signal_name,
                history_timestamps=_history_timestamps(env),
            ).astype(np.float32)
        return pad_sequence_1d(env.get_signal(signal_name), env.cur_step, sequence_length).astype(np.float32)

    return ObservationFeatureSpec(
        name=signal_name,
        group="sequence",
        dim=1,
        scope="shared",
        builder=builder,
        description=description,
    )


def _per_agent_signal_sequence_feature(
    signal_name: str,
    description: str,
    *,
    use_forecaster: bool = False,
) -> ObservationFeatureSpec:
    def builder(env, sequence_length: int) -> np.ndarray:
        if use_forecaster:
            return env.forecaster.predict(
                env.get_signal_history(signal_name),
                sequence_length,
                signal_name=signal_name,
                history_timestamps=_history_timestamps(env),
            ).astype(np.float32)
        return pad_sequence_2d(env.get_signal(signal_name), env.cur_step, sequence_length).T.astype(np.float32)

    return ObservationFeatureSpec(
        name=signal_name,
        group="sequence",
        dim=1,
        scope="per_agent",
        builder=builder,
        description=description,
    )


register_local_feature(
    ObservationFeatureSpec(
        name="time",
        group="local",
        dim=2,
        scope="shared",
        builder=lambda env, _: build_time_features(env.cur_step, env.episode_length, env.n),
        description="Episode progress encoded as one sin/cos pair.",
    )
)
register_local_feature(
    ObservationFeatureSpec(
        name="calendar_time",
        group="local",
        dim=4,
        scope="shared",
        builder=lambda env, _: build_calendar_time_features(_current_timestamp_value(env), env.n),
        description="Absolute hour-of-day and day-of-year sin/cos encodings.",
    )
)
register_local_feature(_current_shared_signal_feature("price", "Current electricity price."))
register_local_feature(_current_per_agent_signal_feature("load", "Current per-agent load."))
register_local_feature(_current_per_agent_signal_feature("pv", "Current per-agent PV output."))
register_local_feature(
    ObservationFeatureSpec(
        name="soc",
        group="local",
        dim=1,
        scope="per_agent",
        builder=lambda env, _: reshape_agent_scalar_feature(env.soc),
        description="Current battery state of charge per agent.",
    )
)

register_sequence_feature(
    _shared_signal_sequence_feature("price", "Future shared price window.", use_forecaster=True)
)
register_sequence_feature(
    _per_agent_signal_sequence_feature("load", "Future per-agent load window.", use_forecaster=True)
)
register_sequence_feature(
    _per_agent_signal_sequence_feature("pv", "Future per-agent PV window.", use_forecaster=True)
)


def build_adjacency_field(n_agents: int, adjacency_type: str) -> np.ndarray:
    return build_adjacency(n_agents, adjacency_type)
