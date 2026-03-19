"""观测构造器注册表。

根据构造器类型名称（如 "default"）返回对应的 ObservationBuilder 类，
并提供按实验配置创建构造器实例的工厂函数。

如何添加新观测构造器:
    1. 创建 ``envs/observation/your_builder.py``，实现 ``ObservationBuilder`` 接口
       -- 必须提供 ``get_schema()``、``get_layout()``、``build(env)``、``zeros()``
    2. 在此处注册::

           register_obs_builder("your_builder", YourObsBuilder)

    3. 在配置中使用: ``cfg.obs.builder_type = "your_builder"``

主要函数:
    register_obs_builder -- 注册观测构造器类
    get_obs_builder_cls  -- 按名称获取构造器类
    build_obs_builder    -- 按实验配置创建构造器实例
"""

from __future__ import annotations

from envs.observation.default_builder import DefaultObservationBuilder

# 全局观测构造器注册表: 构造器类型名称 -> 构造器类
OBS_BUILDER_REGISTRY: dict[str, type] = {
    "default": DefaultObservationBuilder,
}


def register_obs_builder(name: str, builder_cls: type) -> None:
    """注册一个观测构造器类到全局注册表。

    参数:
        name: 构造器类型名称（如 "default"）
        builder_cls: 对应的观测构造器类
    """
    OBS_BUILDER_REGISTRY[name] = builder_cls


def get_obs_builder_cls(name: str) -> type:
    """按名称从注册表中读取观测构造器类。

    参数:
        name: 构造器类型名称

    返回:
        type: 对应的观测构造器类

    异常:
        ValueError: 当名称不存在于注册表中时抛出
    """
    if name not in OBS_BUILDER_REGISTRY:
        raise ValueError(f"Unknown obs.builder_type '{name}', available: {list(OBS_BUILDER_REGISTRY)}")
    return OBS_BUILDER_REGISTRY[name]


def build_obs_builder(cfg):
    """按实验配置创建观测构造器实例。

    从配置中读取构造器类型、局部特征列表、序列特征列表、
    未来预测窗口和邻接矩阵类型，构造并返回对应的构造器实例。

    参数:
        cfg: 实验配置对象，需包含 cfg.obs 和 cfg.env 子配置

    返回:
        ObservationBuilder: 构造好的观测构造器实例
    """
    builder_cls = get_obs_builder_cls(cfg.obs.builder_type)
    return builder_cls(
        local_features=cfg.obs.local_features,
        sequence_features=cfg.obs.sequence_features,
        future_horizon=cfg.env.future_horizon,
        adjacency_type=cfg.obs.adjacency_type,
    )
