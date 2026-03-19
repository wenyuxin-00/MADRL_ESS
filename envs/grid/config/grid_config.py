"""电网拓扑与约束配置。

定义电网 SimBench 编码、求解器类型、电压约束等参数，
以及智能体在电网中的部署位置和设备参数。

主要类:
    AgentDeployment -- 单个智能体的物理部署参数

主要函数:
    build_agent_deployments -- 从实验配置解析智能体部署列表
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class AgentDeployment:
    """单个强化学习智能体的物理部署位置和设备参数。

    属性:
        bus_id: 该智能体电池连接的 pandapower 母线索引
        battery_capacity_kwh: 可用电池能量容量（kWh）
        battery_power_kw: 最大充放电功率（kW，对称式）
        init_soc: 每个 episode 开始时的初始荷电状态（0--1 之间的比例值）
        soc_min: 允许的最低 SoC（比例值）
        soc_max: 允许的最高 SoC（比例值）
        efficiency: 充放电效率（按半程计算，即非 sqrt 约定，
                    与 EnergyStorageEnv 保持一致）
    """

    bus_id: int                         # 母线索引
    battery_capacity_kwh: float         # 电池容量（kWh）
    battery_power_kw: float             # 最大充放电功率（kW）
    init_soc: float = 0.5              # 初始 SoC
    soc_min: float = 0.05             # 最低 SoC
    soc_max: float = 0.95             # 最高 SoC
    efficiency: float = 0.95          # 充放电效率


def build_agent_deployments(cfg: Any) -> list[AgentDeployment]:
    """从实验配置解析智能体部署列表。

    若 cfg.grid.agent_bus_ids 已设置，则使用指定的母线 ID，
    设备参数取自数据集元数据（如可用）或 cfg.env 中的默认值。
    若未配置自定义 ID，则回退到 "1-LV-rural1--0-sw" 的
    Phase-1 固定拓扑预设。

    参数:
        cfg: 完整的 ExperimentConfig 实验配置实例

    返回:
        list[AgentDeployment]: 每个智能体一个条目，
                               总数为 cfg.env.num_agents
    """
    from envs.grid.topology.rural1_fixed import RURAL1_AGENT_DEPLOYMENTS

    n = int(cfg.env.num_agents)
    bus_ids: list[int] = list(cfg.grid.agent_bus_ids)

    if not bus_ids:
        # 配置中未指定母线 ID -- 使用 Phase-1 默认部署
        return RURAL1_AGENT_DEPLOYMENTS[:n]

    # 从配置和环境默认值构建部署列表
    c_bat = float(cfg.env.battery_capacity)
    p_max = float(cfg.env.max_charge_rate)
    init_soc = float(cfg.env.init_soc)
    soc_min = float(cfg.env.soc_min)
    soc_max = float(cfg.env.soc_max)
    eff = float(cfg.env.efficiency)

    deployments = [
        AgentDeployment(
            bus_id=int(bus_ids[i]),
            battery_capacity_kwh=c_bat,
            battery_power_kw=p_max,
            init_soc=init_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            efficiency=eff,
        )
        for i in range(n)
    ]
    return deployments
