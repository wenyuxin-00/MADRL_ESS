from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_AGENT_BUS_IDS: tuple[int, ...] = (10, 6, 12, 4, 2)


# 作用：描述一个 agent 在电网中的接入位置和固定电池参数。
@dataclass(frozen=True)
class AgentDeployment:
    bus_id: int
    battery_capacity_kwh: float
    battery_power_kw: float
    init_soc: float = 0.5
    soc_min: float = 0.05
    soc_max: float = 0.95
    efficiency: float = 0.95


# 作用：把配置值解析为正标量，避免非法电池或功率参数进入环境。
def _resolve_positive_scalar(value: object, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive scalar, got {value!r}.") from exc
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive, got {scalar}.")
    return scalar


# 作用：把标量或向量配置统一解析成逐 agent 的正数列表，，避免非法电池容量参数进入环境。
def _resolve_positive_vector(
    value: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    n_agents: int,
    name: str,
) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(item) for item in value]
    else:
        values = [float(_resolve_positive_scalar(value, name=name))]

    if len(values) == 1:
        values *= n_agents
    elif len(values) != n_agents:
        if not values or any(abs(item - values[0]) > 1e-09 for item in values):
            raise ValueError(f"{name} should provide {n_agents} value(s) for fixed mode, got {len(values)}.")
        values = [values[0]] * n_agents

    if any(value <= 0.0 for value in values):
        raise ValueError(f"{name} must be positive for all agents, got {values}.")
    return values


# 作用：解析固定电池配置，输出容量、C-rate 和每个 agent 的功率上限。
def resolve_fixed_battery_spec(
    battery_capacity: float | list[float] | tuple[float, ...] | np.ndarray,max_charge_rate: float,*,n_agents: int,) -> tuple[list[float], float, list[float]]:
    if isinstance(max_charge_rate, (list, tuple)):
        raise ValueError(
            f"fixed battery mode expects max_charge_rate to be a positive scalar C-rate, got {max_charge_rate!r}."
        )
    capacity_kwh = _resolve_positive_vector(
        battery_capacity,
        n_agents=n_agents,
        name="battery_capacity",
    )
    c_rate = _resolve_positive_scalar(max_charge_rate, name="max_charge_rate")
    p_max_kw = [float(capacity) * c_rate for capacity in capacity_kwh]
    return capacity_kwh, c_rate, p_max_kw


# 作用：根据配置生成 GridCore 使用的每个 agent 电网部署合同。
def build_agent_deployments(cfg: Any) -> list[AgentDeployment]:
    n_agents = int(cfg.env.num_agents)
    bus_ids = list(cfg.grid.agent_bus_ids) or list(DEFAULT_AGENT_BUS_IDS[:n_agents])
    capacity_kwh, _, power_kw = resolve_fixed_battery_spec(cfg.env.battery_capacity,cfg.env.max_charge_rate,n_agents=n_agents,)
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
