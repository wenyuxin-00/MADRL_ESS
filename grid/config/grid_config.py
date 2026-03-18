"""Agent deployment configuration for grid environments.

``AgentDeployment`` describes a single RL agent's physical placement
(which bus in the pandapower network) and the device parameters of the
storage system at that bus.

``build_agent_deployments()`` resolves the deployments from the experiment
config, falling back to the Phase-1 fixed topology when the config does not
specify custom deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class AgentDeployment:
    """Physical placement and device parameters for one RL agent.

    Attributes
    ----------
    bus_id:
        Pandapower bus index where this agent's battery is connected.
    battery_capacity_kwh:
        Usable battery energy capacity in kWh.
    battery_power_kw:
        Maximum charge/discharge power in kW (symmetric).
    init_soc:
        Initial state-of-charge at the start of each episode (fraction, 0–1).
    soc_min:
        Minimum allowed SoC (fraction).
    soc_max:
        Maximum allowed SoC (fraction).
    efficiency:
        Round-trip efficiency (applied per half-trip, i.e. sqrt convention is
        *not* used — same as ``EnergyStorageEnv``).
    """

    bus_id: int
    battery_capacity_kwh: float
    battery_power_kw: float
    init_soc: float = 0.5
    soc_min: float = 0.05
    soc_max: float = 0.95
    efficiency: float = 0.95


def build_agent_deployments(cfg: Any) -> list[AgentDeployment]:
    """Resolve ``AgentDeployment`` list from the experiment config.

    If ``cfg.grid.agent_bus_ids`` is set, those bus IDs are used with
    per-agent device parameters derived from the dataset metadata (if
    available) or from ``cfg.env`` defaults.

    Falls back to the Phase-1 fixed topology for
    ``1-LV-rural1--0-sw`` when no custom IDs are configured.

    Parameters
    ----------
    cfg:
        Full ``ExperimentConfig`` instance.

    Returns
    -------
    list[AgentDeployment]
        One entry per ``cfg.env.num_agents``.
    """
    from grid.topology.rural1_fixed import RURAL1_AGENT_DEPLOYMENTS

    n = int(cfg.env.num_agents)
    bus_ids: list[int] = list(cfg.grid.agent_bus_ids)

    if not bus_ids:
        # No IDs in config — use Phase-1 defaults.
        return RURAL1_AGENT_DEPLOYMENTS[:n]

    # Build deployments from config + env defaults.
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
