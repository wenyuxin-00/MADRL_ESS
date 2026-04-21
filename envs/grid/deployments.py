from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from typing import Any
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

def _coerce_vector_like(value: object) -> list[object] | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Iterable):
        return list(value)
    return None

def _resolve_capacity_vector(
    battery_capacity: float | list[float] | tuple[float, ...],
    *,
    n_agents: int,
) -> list[float]:
    vector_like = _coerce_vector_like(battery_capacity)
    if vector_like is not None:
        values = [float(value) for value in vector_like]
        if len(values) == 1:
            values = values * n_agents
        elif len(values) != n_agents:
            if values and all(abs(value - values[0]) <= 1e-9 for value in values):
                values = [values[0]] * n_agents
            else:
                raise ValueError(
                    "battery_capacity should provide "
                    f"{n_agents} value(s) for fixed battery mode, got {len(values)}."
                )
    else:
        values = [_resolve_positive_scalar(battery_capacity, name="battery_capacity")] * n_agents

    if any(value <= 0.0 for value in values):
        raise ValueError(f"battery_capacity must be positive for all agents, got {values}.")
    return values

def resolve_fixed_battery_spec(
    battery_capacity: float | list[float] | tuple[float, ...],
    max_charge_rate: float,
    *,
    n_agents: int,
) -> tuple[list[float], float, list[float]]:
    if _coerce_vector_like(max_charge_rate) is not None:
        raise ValueError(
            "fixed battery mode expects max_charge_rate to be a positive scalar C-rate, "
            f"got {max_charge_rate!r}."
        )

    capacity_kwh = _resolve_capacity_vector(battery_capacity, n_agents=n_agents)
    c_rate = _resolve_positive_scalar(max_charge_rate, name="max_charge_rate")
    p_max_kw = [float(capacity * c_rate) for capacity in capacity_kwh]
    return capacity_kwh, c_rate, p_max_kw

def build_agent_deployments(cfg: Any) -> list[AgentDeployment]:
    from envs.grid.topology.rural1_fixed import RURAL1_AGENT_DEPLOYMENTS
    n_agents = int(cfg.env.num_agents)
    bus_ids = list(cfg.grid.agent_bus_ids)
    if not bus_ids:
        bus_ids = [deployment.bus_id for deployment in RURAL1_AGENT_DEPLOYMENTS[:n_agents]]

    capacity_kwh, _, power_kw = resolve_fixed_battery_spec(
        cfg.env.battery_capacity,
        cfg.env.max_charge_rate,
        n_agents=n_agents,
    )
    init_soc = float(cfg.env.init_soc)
    soc_min = float(cfg.env.soc_min)
    soc_max = float(cfg.env.soc_max)
    efficiency = float(cfg.env.efficiency)
    return [
        AgentDeployment(
            bus_id=int(bus_ids[idx]),
            battery_capacity_kwh=float(capacity_kwh[idx]),
            battery_power_kw=float(power_kw[idx]),
            init_soc=init_soc,
            soc_min=soc_min,
            soc_max=soc_max,
            efficiency=efficiency,
        )
        for idx in range(n_agents)
    ]
