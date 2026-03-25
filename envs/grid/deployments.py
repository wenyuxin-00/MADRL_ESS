"""Battery deployment helpers for the single grid-training mainline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentDeployment:
    """Physical deployment details for one storage agent in the grid."""

    bus_id: int
    battery_capacity_kwh: float
    battery_power_kw: float
    init_soc: float = 0.5
    soc_min: float = 0.05
    soc_max: float = 0.95
    efficiency: float = 0.95


def build_agent_deployments(cfg: Any) -> list[AgentDeployment]:
    """Build per-agent storage deployments from the experiment config."""
    from envs.grid.topology.rural1_fixed import RURAL1_AGENT_DEPLOYMENTS

    n_agents = int(cfg.env.num_agents)
    bus_ids = list(cfg.grid.agent_bus_ids)
    if not bus_ids:
        return list(RURAL1_AGENT_DEPLOYMENTS[:n_agents])

    battery_capacity = float(cfg.env.battery_capacity)
    battery_power = float(cfg.env.max_charge_rate)
    init_soc = float(cfg.env.init_soc)
    soc_min = float(cfg.env.soc_min)
    soc_max = float(cfg.env.soc_max)
    efficiency = float(cfg.env.efficiency)

    return [
        AgentDeployment(
            bus_id=int(bus_ids[idx]),
            battery_capacity_kwh=battery_capacity,
            battery_power_kw=battery_power,
            init_soc=init_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            efficiency=efficiency,
        )
        for idx in range(n_agents)
    ]


__all__ = [
    "AgentDeployment",
    "build_agent_deployments",
]
