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
    """观测特征规格定义，描述一个可选的观测特征。

    属性:
        name: 特征名称（如 "price"、"load"、"soc"）
        group: 特征组别，"local" 表示局部特征，"sequence" 表示序列特征
        dim: 特征维度（如时间编码为 2，标量特征为 1）
        scope: 作用域，"shared" 表示所有智能体共享，"per_agent" 表示逐智能体
        builder: 构造函数，签名为 (env, sequence_length) -> np.ndarray
        description: 特征的可读描述
    """

    name: str
    group: str
    dim: int
    scope: str
    builder: callable
    description: str = ""

    def field_name(self) -> str:
        """返回该特征在观测字典中的字段名。

        局部特征直接使用 name，序列特征追加 "_seq" 后缀。

        返回:
            str: 字段名，如 "price" 或 "price_seq"
        """
        return self.name if self.group == "local" else f"{self.name}_seq"

    def schema(self, n_agents: int, sequence_length: int) -> tuple[int, ...]:
        """返回该特征对应字段的 shape。

        根据 group 和 scope 组合确定维度:
            - local: (n_agents, dim)
            - sequence + shared: (seq_len,) 或 (seq_len, dim)
            - sequence + per_agent: (n_agents, seq_len) 或 (n_agents, seq_len, dim)

        参数:
            n_agents: 智能体数量
            sequence_length: 序列长度

        返回:
            tuple[int, ...]: shape 元组
        """
        if self.group == "local":
            return (int(n_agents), int(self.dim))
        if self.scope == "shared":
            # 共享序列：所有智能体共用一条序列
            return (sequence_length,) if self.dim == 1 else (sequence_length, int(self.dim))
        if self.scope == "per_agent":
            # 逐智能体序列：每个智能体有独立的序列
            return (
                (int(n_agents), sequence_length)
                if self.dim == 1
                else (int(n_agents), sequence_length, int(self.dim))
            )
        raise ValueError(f"未知 feature scope: '{self.scope}'")

    def layout(self) -> dict[str, str | int]:
        """返回供 notebook / 模型装配读取的字段布局元数据。

        返回:
            dict: 包含 feature_name、group、scope、dim、description 的字典
        """
        return {
            "feature_name": self.name,
            "group": self.group,
            "scope": self.scope,
            "dim": int(self.dim),
            "description": self.description,
        }


# 全局特征注册表：名称 → 特征规格
LOCAL_FEATURES: dict[str, ObservationFeatureSpec] = {}       # 局部特征注册表
SEQUENCE_FEATURES: dict[str, ObservationFeatureSpec] = {}    # 序列特征注册表


def register_local_feature(spec: ObservationFeatureSpec) -> None:
    """注册局部特征到全局注册表。

    参数:
        spec: 特征规格，group 必须为 "local"

    异常:
        ValueError: 当 spec.group != "local" 时抛出
    """
    if spec.group != "local":
        raise ValueError("register_local_feature 只能注册 group='local' 的 feature")
    LOCAL_FEATURES[spec.name] = spec


def register_sequence_feature(spec: ObservationFeatureSpec) -> None:
    """注册序列特征到全局注册表。

    参数:
        spec: 特征规格，group 必须为 "sequence"

    异常:
        ValueError: 当 spec.group != "sequence" 时抛出
    """
    if spec.group != "sequence":
        raise ValueError("register_sequence_feature 只能注册 group='sequence' 的 feature")
    SEQUENCE_FEATURES[spec.name] = spec


def get_local_feature_spec(name: str) -> ObservationFeatureSpec:
    """按名称从注册表中读取局部特征定义。

    参数:
        name: 特征名称

    返回:
        ObservationFeatureSpec: 对应的特征规格

    异常:
        ValueError: 当名称不存在于注册表中时抛出
    """
    if name not in LOCAL_FEATURES:
        raise ValueError(f"未知 local feature '{name}'，可选项：{list(LOCAL_FEATURES)}")
    return LOCAL_FEATURES[name]


def get_sequence_feature_spec(name: str) -> ObservationFeatureSpec:
    """按名称从注册表中读取序列特征定义。

    参数:
        name: 特征名称

    返回:
        ObservationFeatureSpec: 对应的特征规格

    异常:
        ValueError: 当名称不存在于注册表中时抛出
    """
    if name not in SEQUENCE_FEATURES:
        raise ValueError(f"未知 sequence feature '{name}'，可选项：{list(SEQUENCE_FEATURES)}")
    return SEQUENCE_FEATURES[name]


def resolve_local_features(names: list[str]) -> list[ObservationFeatureSpec]:
    """批量解析局部特征名称列表为特征规格列表。

    参数:
        names: 特征名称列表

    返回:
        list[ObservationFeatureSpec]: 对应的特征规格列表
    """
    return [get_local_feature_spec(name) for name in names]


def resolve_sequence_features(names: list[str]) -> list[ObservationFeatureSpec]:
    """批量解析序列特征名称列表为特征规格列表。

    参数:
        names: 特征名称列表

    返回:
        list[ObservationFeatureSpec]: 对应的特征规格列表
    """
    return [get_sequence_feature_spec(name) for name in names]


def _current_shared_signal_feature(signal_name: str, description: str) -> ObservationFeatureSpec:
    """创建共享标量信号的局部特征规格（当前时刻值广播到所有智能体）。

    参数:
        signal_name: 信号名称（如 "price"），对应 env.get_signal_step() 的键
        description: 特征描述

    返回:
        ObservationFeatureSpec: 局部特征规格，builder 输出 shape (n_agents, 1)
    """
    return ObservationFeatureSpec(
        name=signal_name,
        group="local",
        dim=1,
        scope="shared",
        # 取当前步的标量值，广播为 (n_agents, 1)
        builder=lambda env, _: broadcast_scalar_feature(env.get_signal_step(signal_name), env.n),
        description=description,
    )


def _current_per_agent_signal_feature(signal_name: str, description: str) -> ObservationFeatureSpec:
    """创建逐智能体标量信号的局部特征规格（每个智能体有独立值）。

    参数:
        signal_name: 信号名称（如 "load"），对应 env.get_signal_step() 的键
        description: 特征描述

    返回:
        ObservationFeatureSpec: 局部特征规格，builder 输出 shape (n_agents, 1)
    """
    return ObservationFeatureSpec(
        name=signal_name,
        group="local",
        dim=1,
        scope="per_agent",
        # 取当前步各智能体的标量值，reshape 为 (n_agents, 1)
        builder=lambda env, _: reshape_agent_scalar_feature(env.get_signal_step(signal_name)),
        description=description,
    )


def _shared_signal_sequence_feature(
    signal_name: str,
    description: str,
    *,
    use_forecaster: bool = False,
) -> ObservationFeatureSpec:
    """创建共享信号的序列特征规格（所有智能体共用一条时间序列）。

    参数:
        signal_name: 信号名称（如 "price"）
        description: 特征描述
        use_forecaster: 是否使用预测器生成序列（而非直接切片真实数据）

    返回:
        ObservationFeatureSpec: 序列特征规格，builder 输出 shape (seq_len,)
    """
    def builder(env, sequence_length: int) -> np.ndarray:
        if use_forecaster:
            # 使用预测器：输入历史数据，输出预测序列
            return env.forecaster.predict(
                env.get_signal_history(signal_name),
                sequence_length,
                signal_name=signal_name,
            ).astype(np.float32)
        # 直接从真实数据中切片，尾部不足时补零
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
    """创建逐智能体信号的序列特征规格（每个智能体有独立的时间序列）。

    参数:
        signal_name: 信号名称（如 "load"）
        description: 特征描述
        use_forecaster: 是否使用预测器生成序列

    返回:
        ObservationFeatureSpec: 序列特征规格，builder 输出 shape (n_agents, seq_len)
    """
    def builder(env, sequence_length: int) -> np.ndarray:
        if use_forecaster:
            # 使用预测器：输入历史数据，输出预测序列
            return env.forecaster.predict(
                env.get_signal_history(signal_name),
                sequence_length,
                signal_name=signal_name,
            ).astype(np.float32)
        # 从 2D 真实数据切片，转置使 agent 维度在前
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


# =====================================================================
# 注册内置局部特征
# =====================================================================

# time: 周期时间编码（sin/cos），dim=2，所有智能体共享
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
# price: 当前电价标量，广播到所有智能体
register_local_feature(_current_shared_signal_feature("price", "当前真实电价"))
# load: 当前各智能体的负荷值
register_local_feature(_current_per_agent_signal_feature("load", "当前每个 agent 的负荷"))
# pv: 当前各智能体的光伏出力
register_local_feature(_current_per_agent_signal_feature("pv", "当前每个 agent 的光伏出力"))
# soc: 当前各智能体的电池荷电状态
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

# =====================================================================
# 注册内置序列特征（均支持预测器替换真实未来数据）
# =====================================================================

# price_seq: 价格时间序列窗口
register_sequence_feature(_shared_signal_sequence_feature("price", "价格窗口，支持预测器替换", use_forecaster=True))
# load_seq: 各智能体负荷时间序列窗口
register_sequence_feature(_per_agent_signal_sequence_feature("load", "每个 agent 的负荷窗口", use_forecaster=True))
# pv_seq: 各智能体光伏时间序列窗口
register_sequence_feature(_per_agent_signal_sequence_feature("pv", "每个 agent 的光伏窗口", use_forecaster=True))


def build_adjacency_field(n_agents: int, adjacency_type: str) -> np.ndarray:
    """统一导出邻接矩阵构造，封装底层 build_adjacency 调用。

    参数:
        n_agents: 智能体数量
        adjacency_type: 邻接矩阵类型

    返回:
        np.ndarray: shape (n_agents, n_agents) 的邻接矩阵
    """
    return build_adjacency(n_agents, adjacency_type)
