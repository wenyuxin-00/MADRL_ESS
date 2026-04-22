from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

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
