"""Fixed deployment defaults for the rural1 SimBench feeder."""

from __future__ import annotations

from envs.grid.deployments import AgentDeployment

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

__all__ = [
    "RURAL1_AGENT_DEPLOYMENTS",
    "RURAL1_PROSUMER_BUS_IDS",
]
