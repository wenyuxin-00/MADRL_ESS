"""
envs/observation/registry.py
职责：观测构造器注册表与工厂函数。

用法：
    from envs.observation.registry import build_obs_builder
    obs_builder = build_obs_builder(args)

后续扩展：注册新 builder 只需在 OBS_BUILDER_REGISTRY 中添加条目。
"""

from envs.observation.default_builder import DefaultObservationBuilder

OBS_BUILDER_REGISTRY: dict = {
    "default": DefaultObservationBuilder,
}


def build_obs_builder(args):
    """根据 args.observation_builder_type 构建观测构造器实例。

    Parameters
    ----------
    args : Config
        须包含 future_horizon, obs_config。
        可选：observation_builder_type（默认 "default"）。

    Returns
    -------
    ObservationBuilder
    """
    builder_type = getattr(args, "observation_builder_type", "default")
    if builder_type not in OBS_BUILDER_REGISTRY:
        raise ValueError(
            f"未知 observation_builder_type '{builder_type}'，可选: {list(OBS_BUILDER_REGISTRY)}"
        )
    obs_config = getattr(args, "obs_config", ["time", "price", "load", "soc"])
    future_horizon = int(args.future_horizon)
    return OBS_BUILDER_REGISTRY[builder_type](obs_config, future_horizon)
