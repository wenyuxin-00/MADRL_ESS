"""Rural1 固定拓扑定义。

为 SimBench "1-LV-rural1--0-sw" 低压农村配电网拓扑提供预定义的
智能体安装位置（母线 ID）和电池设备参数。

主要常量:
    RURAL1_PROSUMER_BUS_IDS     -- 产消者（prosumer）所在的母线 ID 列表
    RURAL1_AGENT_DEPLOYMENTS    -- 各智能体的完整部署参数列表
"""

from __future__ import annotations

from envs.grid.config.grid_config import AgentDeployment

# SimBench 1-LV-rural1--0-sw 网络中智能体部署的母线 ID
# 这些 ID 由 test_grid_core.py 中的慢速测试验证
RURAL1_PROSUMER_BUS_IDS: list[int] = [10, 6, 12]

# 各智能体的完整部署参数（母线位置、电池容量、功率限制、SoC 约束、效率）
RURAL1_AGENT_DEPLOYMENTS: list[AgentDeployment] = [
    AgentDeployment(
        bus_id=10,                      # 第 1 个智能体部署在母线 10
        battery_capacity_kwh=5.0,       # 电池容量 5 kWh
        battery_power_kw=2.5,           # 最大充放电功率 2.5 kW
        init_soc=0.5,                   # 初始 SoC 50%
        soc_min=0.05,                   # 最低 SoC 5%
        soc_max=0.95,                   # 最高 SoC 95%
        efficiency=0.95,                # 充放电效率 95%
    ),
    AgentDeployment(
        bus_id=6,                       # 第 2 个智能体部署在母线 6
        battery_capacity_kwh=5.0,
        battery_power_kw=2.5,
        init_soc=0.5,
        soc_min=0.05,
        soc_max=0.95,
        efficiency=0.95,
    ),
    AgentDeployment(
        bus_id=12,                      # 第 3 个智能体部署在母线 12
        battery_capacity_kwh=5.0,
        battery_power_kw=2.5,
        init_soc=0.5,
        soc_min=0.05,
        soc_max=0.95,
        efficiency=0.95,
    ),
]
