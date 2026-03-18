"""Rural1 固定拓扑定义。

为 SimBench "1-LV-rural1--0-sw" 拓扑提供预定义的
智能体安装位置和网络参数。
"""

from __future__ import annotations

from envs.grid.config.grid_config import AgentDeployment

# Bus IDs in the SimBench 1-LV-rural1--0-sw network where agents are placed.
# These are validated by the slow tests in test_grid_core.py.
RURAL1_PROSUMER_BUS_IDS: list[int] = [10, 6, 12]

RURAL1_AGENT_DEPLOYMENTS: list[AgentDeployment] = [
    AgentDeployment(
        bus_id=10,
        battery_capacity_kwh=5.0,
        battery_power_kw=2.5,
        init_soc=0.5,
        soc_min=0.05,
        soc_max=0.95,
        efficiency=0.95,
    ),
    AgentDeployment(
        bus_id=6,
        battery_capacity_kwh=5.0,
        battery_power_kw=2.5,
        init_soc=0.5,
        soc_min=0.05,
        soc_max=0.95,
        efficiency=0.95,
    ),
    AgentDeployment(
        bus_id=12,
        battery_capacity_kwh=5.0,
        battery_power_kw=2.5,
        init_soc=0.5,
        soc_min=0.05,
        soc_max=0.95,
        efficiency=0.95,
    ),
]
