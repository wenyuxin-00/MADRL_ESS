from __future__ import annotations

from typing import Any
import numpy as np

from configs.cfg import Cfg
from controllers.local_mpc_solver import ReusableLocalMPCSolver
from controllers.mpc_common import action_array_from_power
from utils.price import derive_import_price_seq, get_import_price_markup


class LocalMPCController:
    name = "LOCAL_MPC"

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg; self.cache: dict[tuple[Any, ...], ReusableLocalMPCSolver] = {}

    def reset(self, env_state: dict) -> None:
        self.env_state = dict(env_state)

    def __call__(self, env, window: dict[str, np.ndarray], *, progress_desc: str | None = None) -> tuple[list[np.ndarray], dict[str, np.ndarray], dict[str, object]]:
        del progress_desc
        import_price_seq = derive_import_price_seq(window["price_seq"], markup_eur_per_kwh=float(get_import_price_markup(env)))
        load_seq, pv_seq = np.asarray(window["load_seq"], dtype=np.float32), np.asarray(window["pv_seq"], dtype=np.float32)
        battery_power_kw = np.zeros((int(env.n),), dtype=np.float32); pv_curtail_kw = np.zeros((int(env.n),), dtype=np.float32); solve_time_sec = 0.0
        for agent_idx in range(int(env.n)):
            key = (agent_idx, int(import_price_seq.size), float(env.cap[agent_idx]), float(env.pmax[agent_idx]), float(env.cfg.env.dt_hours), float(env.cfg.env.efficiency), float(env.cfg.env.soc_min), float(env.cfg.env.soc_max))
            solver = self.cache.get(key)
            if solver is None:
                solver = ReusableLocalMPCSolver(horizon=int(import_price_seq.size), battery_capacity_kwh=float(env.cap[agent_idx]), p_max_kw=float(env.pmax[agent_idx]), dt_hours=float(env.cfg.env.dt_hours), efficiency=float(env.cfg.env.efficiency), soc_min=float(env.cfg.env.soc_min), soc_max=float(env.cfg.env.soc_max))
                self.cache[key] = solver
            result = solver.solve_full_horizon(import_price_seq=import_price_seq, load_seq=load_seq[agent_idx], pv_seq=pv_seq[agent_idx], soc=float(env.soc[agent_idx]))
            battery_power_kw[agent_idx] = np.float32(result.signed_battery_kw[0]); pv_curtail_kw[agent_idx] = np.float32(result.pv_curtail_kw[0]); solve_time_sec += float(result.solve_time_sec)
        actions, _, action_info = action_array_from_power(env, battery_power_kw, pv_curtail_kw)
        action_info["solve_time_sec"] = np.asarray(solve_time_sec, dtype=np.float32)
        return actions, action_info, {"solve_time_sec": solve_time_sec}

    def close(self) -> None:
        for solver in self.cache.values():
            solver.dispose()
        self.cache.clear()
