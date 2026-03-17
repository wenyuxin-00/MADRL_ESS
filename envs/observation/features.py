"""观测 feature 注册表。

这里把观测拆成两类：
  - local：每个 agent 当前时刻的局部特征，最终会拼成 `obs["local"]`
  - sequence：窗口类特征，单独保留成 `xxx_seq`

未来新增 PV 时，通常只需要：
  1. 数据集提供 `signals["pv"]`
  2. 在这里注册 `pv` 对应 feature
  3. notebook 把 `pv` 加入 feature 列表
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.observation.feature_blocks import (
    broadcast_scalar_feature,
    build_adjacency,
    build_time_features,
    pad_sequence_1d,
    pad_sequence_2d,
    reshape_agent_scalar_feature,
)


@dataclass(frozen=True)
class ObservationFeatureSpec:
    """描述一个可选观测 feature。"""

    name: str
    group: str
    dim: int
    scope: str
    builder: callable
    description: str = ""

    def field_name(self) -> str:
        """返回该 feature 在观测字典中的字段名。"""
        return self.name if self.group == "local" else f"{self.name}_seq"

    def schema(self, n_agents: int, sequence_length: int) -> tuple[int, ...]:
        """返回该 feature 对应字段的 shape。"""
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
        raise ValueError(f"未知 feature scope: '{self.scope}'")

    def layout(self) -> dict[str, str | int]:
        """返回 notebook / 模型装配可直接读取的字段布局。"""
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
    """注册局部特征。"""
    if spec.group != "local":
        raise ValueError("register_local_feature 只能注册 group='local' 的 feature")
    LOCAL_FEATURES[spec.name] = spec


def register_sequence_feature(spec: ObservationFeatureSpec) -> None:
    """注册序列特征。"""
    if spec.group != "sequence":
        raise ValueError("register_sequence_feature 只能注册 group='sequence' 的 feature")
    SEQUENCE_FEATURES[spec.name] = spec


def get_local_feature_spec(name: str) -> ObservationFeatureSpec:
    """按名称读取局部特征定义。"""
    if name not in LOCAL_FEATURES:
        raise ValueError(f"未知 local feature '{name}'，可选项：{list(LOCAL_FEATURES)}")
    return LOCAL_FEATURES[name]


def get_sequence_feature_spec(name: str) -> ObservationFeatureSpec:
    """按名称读取序列特征定义。"""
    if name not in SEQUENCE_FEATURES:
        raise ValueError(f"未知 sequence feature '{name}'，可选项：{list(SEQUENCE_FEATURES)}")
    return SEQUENCE_FEATURES[name]


def resolve_local_features(names: list[str]) -> list[ObservationFeatureSpec]:
    """解析 local feature 列表。"""
    return [get_local_feature_spec(name) for name in names]


def resolve_sequence_features(names: list[str]) -> list[ObservationFeatureSpec]:
    """解析 sequence feature 列表。"""
    return [get_sequence_feature_spec(name) for name in names]


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
            ).astype(np.float32)
        return pad_sequence_2d(
            env.get_signal(signal_name),
            env.cur_step,
            sequence_length,
        ).T.astype(np.float32)

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
        description="当前时刻的周期时间编码",
    )
)
register_local_feature(_current_shared_signal_feature("price", "当前真实电价"))
register_local_feature(_current_per_agent_signal_feature("load", "当前每个 agent 的负荷"))
register_local_feature(_current_per_agent_signal_feature("pv", "当前每个 agent 的光伏出力"))
register_local_feature(
    ObservationFeatureSpec(
        name="soc",
        group="local",
        dim=1,
        scope="per_agent",
        builder=lambda env, _: reshape_agent_scalar_feature(env.soc),
        description="当前每个 agent 的电池 SoC",
    )
)

register_sequence_feature(_shared_signal_sequence_feature("price", "价格窗口，支持预测器替换", use_forecaster=True))
register_sequence_feature(_per_agent_signal_sequence_feature("load", "每个 agent 的负荷窗口", use_forecaster=True))
register_sequence_feature(_per_agent_signal_sequence_feature("pv", "每个 agent 的光伏窗口", use_forecaster=True))


def build_adjacency_field(n_agents: int, adjacency_type: str) -> np.ndarray:
    """统一导出 adjacency 构造。"""
    return build_adjacency(n_agents, adjacency_type)
