"""Fast-lab GridCore with exact-equivalence index caching."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from envs.grid.core.grid_core import GridCore


class GridCoreFastLab(GridCore):
    """GridCore variant that caches pandapower row indices for agent buses."""

    def __init__(self, deployments, grid_cfg: Any) -> None:
        super().__init__(deployments, grid_cfg)
        self._load_row_index_by_bus: dict[int, int | None] = {}
        self._sgen_row_index_by_bus: dict[int, int | None] = {}
        self._fast_path_supported = True

        for bus_id in self.agent_bus_ids:
            load_mask = self.net.load["bus"] == bus_id
            sgen_mask = self.net.sgen["bus"] == bus_id
            load_index = int(self.net.load.index[load_mask][0]) if load_mask.any() else None
            sgen_index = int(self.net.sgen.index[sgen_mask][0]) if sgen_mask.any() else None
            self._load_row_index_by_bus[bus_id] = load_index
            self._sgen_row_index_by_bus[bus_id] = sgen_index
            if load_index is None and sgen_index is None:
                self._fast_path_supported = False
            if load_index is None and sgen_index is not None:
                # Base path may create a load row on demand. Fall back to preserve semantics.
                self._fast_path_supported = False

    def _restore_agent_buses(self) -> None:
        if not self._fast_path_supported:
            super()._restore_agent_buses()
            return

        for bus_id in self.agent_bus_ids:
            load_index = self._load_row_index_by_bus.get(bus_id)
            if load_index is not None and bus_id in self._original_load_p_mw:
                self.net.load.at[load_index, "p_mw"] = self._original_load_p_mw[bus_id]
            sgen_index = self._sgen_row_index_by_bus.get(bus_id)
            if sgen_index is not None and bus_id in self._original_sgen_p_mw:
                self.net.sgen.at[sgen_index, "p_mw"] = self._original_sgen_p_mw[bus_id]

    def _apply_bus_injections_cached(self, bus_id_to_p_kw: dict[int, float]) -> None:
        for bus_id, p_kw in bus_id_to_p_kw.items():
            p_mw = float(p_kw) / 1000.0
            q_mvar = 0.0
            sgen_index = self._sgen_row_index_by_bus.get(bus_id)
            load_index = self._load_row_index_by_bus.get(bus_id)

            if sgen_index is not None:
                self.net.sgen.at[sgen_index, "p_mw"] = max(0.0, p_mw)
                self.net.sgen.at[sgen_index, "q_mvar"] = q_mvar
                if load_index is not None:
                    self.net.load.at[load_index, "p_mw"] = max(0.0, -p_mw)
                continue

            if load_index is not None:
                self.net.load.at[load_index, "p_mw"] = -p_mw
                self.net.load.at[load_index, "q_mvar"] = -q_mvar
                continue

            # Should never happen on the fast path, but preserve behavior if it does.
            self._fast_path_supported = False
            from envs.grid.core.net_builder import apply_bus_injections

            apply_bus_injections(self.net, {bus_id: float(p_kw)})

    def step(
        self,
        p_batt_kw: np.ndarray,
        base_load_kw: np.ndarray,
    ):
        if not self._fast_path_supported:
            return super().step(p_batt_kw=p_batt_kw, base_load_kw=base_load_kw)

        self._restore_agent_buses()
        p_inject_kw = -(base_load_kw + p_batt_kw)
        bus_id_to_p_kw = {
            self.agent_bus_ids[i]: float(p_inject_kw[i]) for i in range(self.n_agents)
        }
        self._apply_bus_injections_cached(bus_id_to_p_kw)

        try:
            self._runpp_with_fallback()
            result = self._extract_result(converged=True)
            self.last_pf_error = ""
            self._last_valid = result
            self._prev_converged = True
            return result
        except Exception as exc:
            self.last_pf_error = f"{type(exc).__name__}: {exc}"
            self._prev_converged = False
            return dataclasses.replace(self._last_valid, converged=False)
