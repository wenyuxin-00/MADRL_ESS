"""Default structured observation builder."""

from __future__ import annotations

import numpy as np

from envs.observation.base import ObservationBuilder
from envs.observation.features import (
    build_adjacency_field,
    resolve_local_features,
    resolve_sequence_features,
)
from envs.observation.normalization import ObservationNormalizer


class DefaultObservationBuilder(ObservationBuilder):
    """Build local, sequence, and adjacency observation fields."""

    def __init__(
        self,
        local_features: list[str],
        sequence_features: list[str],
        future_horizon: int,
        adjacency_type: str = "identity",
        normalizer: ObservationNormalizer | None = None,
    ):
        self.local_feature_names = list(local_features)
        self.sequence_feature_names = list(sequence_features)
        self.future_horizon = int(future_horizon)
        self.adjacency_type = adjacency_type
        self.normalizer = normalizer
        self.sequence_length = self.future_horizon + 1
        self.local_specs = resolve_local_features(self.local_feature_names)
        self.sequence_specs = resolve_sequence_features(self.sequence_feature_names)
        self.local_dim = sum(spec.dim for spec in self.local_specs)

    def get_schema(self, n_agents: int) -> dict[str, tuple[int, ...]]:
        schema = {
            "local": (int(n_agents), self.local_dim),
            "adjacency": (int(n_agents), int(n_agents)),
        }
        for spec in self.sequence_specs:
            schema[spec.field_name()] = spec.schema(int(n_agents), self.sequence_length)
        return schema

    def get_layout(self, n_agents: int) -> dict[str, dict]:
        layout = {
            "local": {
                "group": "local",
                "scope": "per_agent",
                "dim": int(self.local_dim),
                "fields": list(self.local_feature_names),
            },
            "adjacency": {
                "group": "graph",
                "scope": "shared",
                "dim": int(n_agents),
                "fields": ["adjacency"],
            },
        }
        schema = self.get_schema(n_agents)
        for spec in self.sequence_specs:
            field_name = spec.field_name()
            layout[field_name] = {
                **spec.layout(),
                "shape": schema[field_name],
            }
        return layout

    def build_raw(self, env) -> dict[str, np.ndarray]:
        return self._build(env, apply_normalization=False)

    def build(self, env) -> dict[str, np.ndarray]:
        return self._build(env, apply_normalization=True)

    def _build(self, env, *, apply_normalization: bool) -> dict[str, np.ndarray]:
        local_parts: list[np.ndarray] = []
        for spec in self.local_specs:
            values = np.asarray(spec.builder(env, self.sequence_length), dtype=np.float32)
            if apply_normalization and self.normalizer is not None:
                values = self.normalizer.transform_local(spec.name, values)
            local_parts.append(values.astype(np.float32))

        if local_parts:
            local = np.concatenate(local_parts, axis=1).astype(np.float32)
        else:
            local = np.zeros((env.n, 0), dtype=np.float32)
        obs = {
            "local": local,
            "adjacency": build_adjacency_field(env.n, self.adjacency_type),
        }
        for spec in self.sequence_specs:
            values = np.asarray(spec.builder(env, self.sequence_length), dtype=np.float32)
            if apply_normalization and self.normalizer is not None:
                values = self.normalizer.transform_sequence(spec.name, values)
            obs[spec.field_name()] = values.astype(np.float32)
        return obs
