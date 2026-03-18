"""Phase 1 fixed topology: SimBench ``1-LV-rural1--0-sw``.

The three prosumer buses (10, 6, 12) were selected by running::

    import simbench as sb
    net = sb.get_simbench_net("1-LV-rural1--0-sw")
    # Buses that have both a load and a static generator (PV):
    load_buses = set(net.load["bus"])
    sgen_buses = set(net.sgen["bus"])
    prosumer_buses = sorted(load_buses & sgen_buses)

These are the first three prosumer buses in the network (by index).
Validate them by running ``tests/test_grid_core.py::test_simbench_net_loads``.

Device parameters are representative residential-scale PV+battery systems:
- 5 kWh capacity, 2.5 kW max power → C-rate 0.5.
"""

from __future__ import annotations

from grid.config.grid_config import AgentDeployment

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
