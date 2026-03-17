"""Structured observation building framework.
结构化观测构建框架。

The observation system decouples *what* features are observed from *how* they
are assembled, enabling flexible experimentation with different observation
designs without modifying the environment.

Key components / 核心组件:
    - ObservationBuilder (base class)     -- defines the build/schema/layout contract
    - DefaultObservationBuilder           -- configurable builder with local + sequence features
    - ObservationFeatureSpec              -- declarative feature descriptors
    - feature_blocks                      -- low-level building blocks (time encoding, padding, etc.)

How to add a new observation feature / 如何添加新的观测特征:
    1. Define an ``ObservationFeatureSpec`` in ``features.py``
    2. Implement its builder function
    3. Add it to ``LOCAL_FEATURES`` or ``SEQUENCE_FEATURES``
    4. Include it in ``cfg.obs.local_features`` or ``cfg.obs.sequence_features``
"""

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