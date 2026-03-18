"""pandapower network builder utilities.

Responsibilities
----------------
- Load a SimBench network by code (with module-level caching to avoid
  re-parsing on every episode reset — loading takes ~3-5 s the first time).
- Apply per-bus active-power injections so that ``pp.runpp()`` can be called.

Only pandapower / simbench are imported here; no RL or gym dependencies.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandapower as pp

# Module-level cache: sb_code → the *template* net loaded from SimBench.
# Each env gets an independent deep-copy so they can modify it safely.
_NET_CACHE: dict[str, "pp.pandapowerNet"] = {}


def build_simbench_net(sb_code: str) -> "pp.pandapowerNet":
    """Return a fresh deep-copy of the SimBench network identified by *sb_code*.

    The template is loaded once and cached; subsequent calls are O(copy) fast.

    Parameters
    ----------
    sb_code:
        SimBench network identifier, e.g. ``"1-LV-rural1--0-sw"``.

    Returns
    -------
    pandapowerNet
        An independent pandapower network object safe to mutate.
    """
    if sb_code not in _NET_CACHE:
        import simbench as sb

        _NET_CACHE[sb_code] = sb.get_simbench_net(sb_code)
    return copy.deepcopy(_NET_CACHE[sb_code])


def apply_bus_injections(
    net: "pp.pandapowerNet",
    bus_id_to_p_kw: dict[int, float],
    bus_id_to_q_kvar: dict[int, float] | None = None,
) -> None:
    """Set the net-injection power for a set of buses.

    The function modifies *net* **in-place**.  For each bus in
    *bus_id_to_p_kw*, it finds the first static generator (``sgen``) or
    load element at that bus and adjusts its active power.  Positive
    ``p_kw`` means *generation* (net export to the grid); negative means
    *consumption* (net import from the grid).

    If no existing ``sgen`` exists at a bus, a new one is created.

    Parameters
    ----------
    net:
        The pandapower network to modify.
    bus_id_to_p_kw:
        Mapping from pandapower bus index → net injection in **kW**.
        Positive = generation (e.g. PV surplus or discharging battery).
        Negative = consumption (e.g. net load or charging battery).
    bus_id_to_q_kvar:
        Optional reactive-power injection in kVAr.  Defaults to zero for
        all buses when omitted.
    """
    import pandapower as pp

    if bus_id_to_q_kvar is None:
        bus_id_to_q_kvar = {}

    for bus_id, p_kw in bus_id_to_p_kw.items():
        q_kvar = bus_id_to_q_kvar.get(bus_id, 0.0)
        p_mw = p_kw / 1000.0
        q_mvar = q_kvar / 1000.0

        # Look for an existing sgen at this bus.
        sgen_mask = net.sgen["bus"] == bus_id
        if sgen_mask.any():
            idx = net.sgen.index[sgen_mask][0]
            net.sgen.at[idx, "p_mw"] = max(0.0, p_mw)   # sgen cannot be negative
            net.sgen.at[idx, "q_mvar"] = q_mvar
            # Residual negative part becomes additional load.
            load_mask = net.load["bus"] == bus_id
            extra_load_mw = max(0.0, -p_mw)
            if load_mask.any():
                base_load = net.load.at[net.load.index[load_mask][0], "p_mw"]
                net.load.at[net.load.index[load_mask][0], "p_mw"] = base_load + extra_load_mw
            elif extra_load_mw > 0.0:
                pp.create_load(net, bus=bus_id, p_mw=extra_load_mw, q_mvar=0.0)
        else:
            # No sgen — use a load with inverted sign convention.
            load_mask = net.load["bus"] == bus_id
            if load_mask.any():
                idx = net.load.index[load_mask][0]
                net.load.at[idx, "p_mw"] = -p_mw   # positive load = consumption
                net.load.at[idx, "q_mvar"] = -q_mvar
            else:
                # Create a new load element (negative p_mw = generation).
                pp.create_load(net, bus=bus_id, p_mw=-p_mw, q_mvar=-q_mvar)


def set_timeseries_profiles(
    net: "pp.pandapowerNet",
    bus_load_kw: dict[int, float],
    bus_pv_kw: dict[int, float],
) -> None:
    """Apply background load and PV profiles to the net for one timestep.

    This is a convenience wrapper that sets ``load.p_mw`` and ``sgen.p_mw``
    for all buses from time-series data before calling ``runpp``.

    Parameters
    ----------
    net:
        Pandapower network (modified in-place).
    bus_load_kw:
        Bus index → background demand in kW (positive = consumption).
    bus_pv_kw:
        Bus index → PV generation in kW (positive = generation).
    """
    for bus_id, p_kw in bus_load_kw.items():
        mask = net.load["bus"] == bus_id
        if mask.any():
            net.load.at[net.load.index[mask][0], "p_mw"] = p_kw / 1000.0

    for bus_id, p_kw in bus_pv_kw.items():
        mask = net.sgen["bus"] == bus_id
        if mask.any():
            net.sgen.at[net.sgen.index[mask][0], "p_mw"] = p_kw / 1000.0
