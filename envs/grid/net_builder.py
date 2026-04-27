from __future__ import annotations
import copy
import os
from typing import TYPE_CHECKING
# 负责：创建一个干净的 pandapower 网络

if TYPE_CHECKING:
    import pandapower as pp

_NET_CACHE: dict[str, "pp.pandapowerNet"] = {}


# 作用：禁用 numba JIT，避免 pandapower/simbench 在当前环境中触发不稳定导入。
def _ensure_safe_pandapower_imports() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# 作用：清零网络中原有静态负荷和发电，让 prosumer 数据成为唯一注入来源。
def zero_static_power_elements(net: "pp.pandapowerNet") -> "pp.pandapowerNet":
    for element_name in ("load", "sgen"):
        table = getattr(net, element_name, None)
        if table is None or table.empty:
            continue
        zero_columns = [column for column in ("p_mw", "q_mvar") if column in table.columns]
        if zero_columns:
            table.loc[:, zero_columns] = 0.0
    return net


# 作用：按 simbench code 构建电网，并用缓存避免重复加载同一个原始网络。
def build_simbench_net(sb_code: str) -> "pp.pandapowerNet":
    if sb_code not in _NET_CACHE:
        _ensure_safe_pandapower_imports()
        import simbench as sb

        _NET_CACHE[sb_code] = sb.get_simbench_net(sb_code)
    return copy.deepcopy(_NET_CACHE[sb_code])
