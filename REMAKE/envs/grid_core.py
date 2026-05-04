from __future__ import annotations

from dataclasses import dataclass
import copy
import os
import warnings
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridStep:
    vm_pu: np.ndarray
    line_loading_pct: np.ndarray
    trafo_loading_pct: np.ndarray
    v_violation: np.ndarray
    psi_v_raw: float
    psi_line_raw: float
    psi_trafo_raw: float


_NET_CACHE: dict[str, Any] = {}


def _simbench_net(sb_code: str) -> Any:
    if sb_code not in _NET_CACHE:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        import simbench as sb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _NET_CACHE[sb_code] = sb.get_simbench_net(sb_code)
    return copy.deepcopy(_NET_CACHE[sb_code])


def _zero_power_tables(net: Any) -> Any:
    for table_name in ("load", "sgen"):
        table = getattr(net, table_name, None)
        if table is None or table.empty:
            continue
        for column in ("p_mw", "q_mvar"):
            if column in table.columns:
                table.loc[:, column] = 0.0
    return net


def transformer_limit_kw(net: Any) -> float:
    trafo = getattr(net, "trafo", None)
    if trafo is None or trafo.empty:
        return 0.0
    loading = trafo["max_loading_percent"].to_numpy(dtype=np.float32) if "max_loading_percent" in trafo.columns else np.full((len(trafo),), 100.0, dtype=np.float32)
    ratings = trafo["sn_mva"].to_numpy(dtype=np.float32) * np.float32(1000.0) * loading / np.float32(100.0)
    ratings = ratings[np.isfinite(ratings) & (ratings > 0.0)]
    if ratings.size == 0:
        raise ValueError("SimBench transformer ratings must contain positive sn_mva and loading values.")
    return float(np.min(ratings))


class PowerFlowGridCore:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.agent_bus_ids = tuple(int(x) for x in cfg.grid.agent_bus_ids)
        self.n = len(self.agent_bus_ids)
        self.net = _zero_power_tables(_simbench_net(str(cfg.grid.sb_code)))
        self.n_lines = int(len(self.net.line))
        self.n_trafos = int(len(getattr(self.net, "trafo", [])))
        self.trafo_limit_kw = transformer_limit_kw(self.net)
        bus_positions = {int(bus_id): pos for pos, bus_id in enumerate(self.net.bus.index.tolist())}
        missing = [bus_id for bus_id in self.agent_bus_ids if bus_id not in bus_positions]
        if missing:
            raise ValueError(f"agent_bus_ids are not present in the simbench net: {missing}.")
        self.agent_bus_pos = np.asarray([bus_positions[bus_id] for bus_id in self.agent_bus_ids], dtype=np.int64)
        self.load_rows, self.sgen_rows = self._agent_power_rows()

    def _agent_power_rows(self) -> tuple[list[int], list[int]]:
        import pandapower as pp

        load_rows: list[int] = []
        sgen_rows: list[int] = []
        for bus_id in self.agent_bus_ids:
            load_mask = self.net.load["bus"] == bus_id
            sgen_mask = self.net.sgen["bus"] == bus_id
            load_rows.append(int(self.net.load.index[load_mask][0]) if load_mask.any() else int(pp.create_load(self.net, bus=bus_id, p_mw=0.0, q_mvar=0.0)))
            sgen_rows.append(int(self.net.sgen.index[sgen_mask][0]) if sgen_mask.any() else int(pp.create_sgen(self.net, bus=bus_id, p_mw=0.0, q_mvar=0.0)))
        return load_rows, sgen_rows

    def _write_net_load(self, net_load_kw: np.ndarray) -> None:
        values = np.asarray(net_load_kw, dtype=np.float32).reshape(self.n)
        for load_row, sgen_row, value in zip(self.load_rows, self.sgen_rows, values, strict=True):
            p_mw = float(value) / 1e3
            self.net.load.at[load_row, "p_mw"] = max(0.0, p_mw)
            self.net.load.at[load_row, "q_mvar"] = 0.0
            self.net.sgen.at[sgen_row, "p_mw"] = max(0.0, -p_mw)
            self.net.sgen.at[sgen_row, "q_mvar"] = 0.0

    def _run_power_flow(self) -> None:
        import pandapower as pp

        pp.runpp(self.net, algorithm=str(self.cfg.grid.pf_solver), init="auto", verbose=False, numba=False, lightsim2grid=False, calculate_voltage_angles=False, voltage_depend_loads=False)

    def step(self, net_load_kw: np.ndarray) -> GridStep:
        self._write_net_load(net_load_kw)
        self._run_power_flow()
        vm_all = self.net.res_bus["vm_pu"].to_numpy(dtype=np.float32)
        line = self.net.res_line["loading_percent"].to_numpy(dtype=np.float32)
        trafo = self.net.res_trafo["loading_percent"].to_numpy(dtype=np.float32) if self.n_trafos else np.zeros(0, dtype=np.float32)
        agent_vm = np.take(vm_all, self.agent_bus_pos).astype(np.float32)
        v_min, v_max, loading_limit = float(self.cfg.grid.v_min_pu), float(self.cfg.grid.v_max_pu), float(self.cfg.grid.line_max_loading_pct)
        v_violation = (np.maximum(0.0, v_min - agent_vm) + np.maximum(0.0, agent_vm - v_max)).astype(np.float32)
        bus_excess = (np.maximum(0.0, v_min - vm_all) + np.maximum(0.0, vm_all - v_max)).astype(np.float32)
        line_excess = (np.maximum(0.0, line - loading_limit) / np.float32(100.0)).astype(np.float32)
        trafo_excess = (np.maximum(0.0, trafo - loading_limit) / np.float32(100.0)).astype(np.float32)
        return GridStep(vm_pu=agent_vm, line_loading_pct=line, trafo_loading_pct=trafo, v_violation=v_violation, psi_v_raw=float(np.sum(bus_excess * bus_excess)), psi_line_raw=float(np.sum(line_excess * line_excess)), psi_trafo_raw=float(np.sum(trafo_excess * trafo_excess)))
