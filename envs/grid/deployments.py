from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_AGENT_BUS_IDS: tuple[int, ...] = (10, 6, 12, 4, 2)


@dataclass(frozen=True)
class AgentDeployment:
    bus_id: int
    battery_capacity_kwh: float
    battery_power_kw: float
    init_soc: float = 0.5
    soc_min: float = 0.05
    soc_max: float = 0.95
    efficiency: float = 0.95

def _resolve_positive_scalar(value: object, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive scalar, got {value!r}.") from exc
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive, got {scalar}.")
    return scalar

def _resolve_positive_vector(value: float | list[float] | tuple[float, ...], *, n_agents: int, name: str) -> list[float]:
    values = [float(item) for item in value] if isinstance(value, (list, tuple, np.ndarray)) else [float(_resolve_positive_scalar(value, name=name))]
    if len(values) == 1:
        values *= n_agents
    elif len(values) != n_agents:
        if not values or any(abs(item - values[0]) > 1e-9 for item in values):
            raise ValueError(f"{name} should provide {n_agents} value(s) for fixed mode, got {len(values)}.")
        values = [values[0]] * n_agents
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{name} must be positive for all agents, got {values}.")
    return values

def resolve_fixed_battery_spec(
    battery_capacity: float | list[float] | tuple[float, ...],
    max_charge_rate: float,
    *,
    n_agents: int,
) -> tuple[list[float], float, list[float]]:
    if isinstance(max_charge_rate, (list, tuple)):
        raise ValueError(f"fixed battery mode expects max_charge_rate to be a positive scalar C-rate, got {max_charge_rate!r}.")
    capacity_kwh = _resolve_positive_vector(battery_capacity, n_agents=n_agents, name="battery_capacity")
    c_rate = _resolve_positive_scalar(max_charge_rate, name="max_charge_rate")
    p_max_kw = [float(capacity) * c_rate for capacity in capacity_kwh]
    return capacity_kwh, c_rate, p_max_kw

def build_agent_deployments(cfg: Any) -> list[AgentDeployment]:
    n_agents = int(cfg.env.num_agents)
    bus_ids = list(cfg.grid.agent_bus_ids) or list(DEFAULT_AGENT_BUS_IDS[:n_agents])
    capacity_kwh, _, power_kw = resolve_fixed_battery_spec(cfg.env.battery_capacity, cfg.env.max_charge_rate, n_agents=n_agents)
    return [
        AgentDeployment(
            bus_id=int(bus_ids[idx]),
            battery_capacity_kwh=float(capacity_kwh[idx]),
            battery_power_kw=float(power_kw[idx]),
            init_soc=float(cfg.env.init_soc),
            soc_min=float(cfg.env.soc_min),
            soc_max=float(cfg.env.soc_max),
            efficiency=float(cfg.env.efficiency),
        )
        for idx in range(n_agents)
    ]
