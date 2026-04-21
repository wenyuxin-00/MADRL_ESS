from __future__ import annotations
import copy
import os
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    import pandapower as pp

_NET_CACHE: dict[str, "pp.pandapowerNet"] = {}
def _ensure_safe_pandapower_imports() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

def build_simbench_net(sb_code: str) -> "pp.pandapowerNet":
    if sb_code not in _NET_CACHE:
        _ensure_safe_pandapower_imports()
        import simbench as sb
        _NET_CACHE[sb_code] = sb.get_simbench_net(sb_code)
    return copy.deepcopy(_NET_CACHE[sb_code])

def apply_bus_injections(
    net: "pp.pandapowerNet",
    bus_id_to_p_kw: dict[int, float],
    bus_id_to_q_kvar: dict[int, float] | None = None,
) -> None:
    _ensure_safe_pandapower_imports()
    import pandapower as pp
    if bus_id_to_q_kvar is None:
        bus_id_to_q_kvar = {}

    for bus_id, p_kw in bus_id_to_p_kw.items():
        q_kvar = bus_id_to_q_kvar.get(bus_id, 0.0)
        p_mw = p_kw / 1000.0
        q_mvar = q_kvar / 1000.0
        sgen_mask = net.sgen["bus"] == bus_id
        if sgen_mask.any():
            idx = net.sgen.index[sgen_mask][0]
            net.sgen.at[idx, "p_mw"] = max(0.0, p_mw)
            net.sgen.at[idx, "q_mvar"] = q_mvar
            load_mask = net.load["bus"] == bus_id
            extra_load_mw = max(0.0, -p_mw)
            if load_mask.any():
                net.load.at[net.load.index[load_mask][0], "p_mw"] = extra_load_mw
            elif extra_load_mw > 0.0:
                pp.create_load(net, bus=bus_id, p_mw=extra_load_mw, q_mvar=0.0)
        else:
            load_mask = net.load["bus"] == bus_id
            if load_mask.any():
                idx = net.load.index[load_mask][0]
                net.load.at[idx, "p_mw"] = -p_mw     # 正负荷 = 用电
                net.load.at[idx, "q_mvar"] = -q_mvar
            else:
                pp.create_load(net, bus=bus_id, p_mw=-p_mw, q_mvar=-q_mvar)
