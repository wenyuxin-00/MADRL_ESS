"""观测构造器 registry。

当前只注册一个默认观测构造器，但保留最小 registry 骨架，
方便未来按 `builder_type` 扩展新的观测拼装方式。
"""

from __future__ import annotations

from envs.observation.default_builder import DefaultObservationBuilder

OBS_BUILDER_REGISTRY: dict[str, type] = {
    "default": DefaultObservationBuilder,
}


def register_obs_builder(name: str, builder_cls: type) -> None:
    """注册一个观测构造器类。"""
    OBS_BUILDER_REGISTRY[name] = builder_cls


def get_obs_builder_cls(name: str) -> type:
    """按名称读取观测构造器类。"""
    if name not in OBS_BUILDER_REGISTRY:
        raise ValueError(f"Unknown obs.builder_type '{name}', available: {list(OBS_BUILDER_REGISTRY)}")
    return OBS_BUILDER_REGISTRY[name]


def build_obs_builder(cfg):
    """按实验配置创建观测构造器。"""
    builder_cls = get_obs_builder_cls(cfg.obs.builder_type)
    return builder_cls(
        local_features=cfg.obs.local_features,
        sequence_features=cfg.obs.sequence_features,
        future_horizon=cfg.env.future_horizon,
        adjacency_type=cfg.obs.adjacency_type,
    )
