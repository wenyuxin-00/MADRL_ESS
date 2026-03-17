"""结构化观测模块。"""

from envs.observation.base import ObservationBuilder
from envs.observation.default_builder import DefaultObservationBuilder
from envs.observation.features import (
    LOCAL_FEATURES,
    SEQUENCE_FEATURES,
    ObservationFeatureSpec,
    get_local_feature_spec,
    get_sequence_feature_spec,
)
from envs.observation.registry import (
    OBS_BUILDER_REGISTRY,
    build_obs_builder,
    get_obs_builder_cls,
    register_obs_builder,
)

__all__ = [
    "LOCAL_FEATURES",
    "OBS_BUILDER_REGISTRY",
    "SEQUENCE_FEATURES",
    "DefaultObservationBuilder",
    "ObservationBuilder",
    "ObservationFeatureSpec",
    "build_obs_builder",
    "get_local_feature_spec",
    "get_obs_builder_cls",
    "get_sequence_feature_spec",
    "register_obs_builder",
]