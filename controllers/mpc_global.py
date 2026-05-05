from __future__ import annotations

import numpy as np

from configs.cfg import Cfg
from controllers.misocp_solver import GlobalMisocpSolver
from controllers.mpc_common import action_array_from_power
from utils.price import derive_import_price_seq, get_import_price_markup


class MisocpController:
    name = "MISOCP"

    def __init__(self, cfg: Cfg, env=None) -> None:
        self.cfg = cfg
        self.solver: GlobalMisocpSolver | None = None
        self.plan_key: tuple[int, int] | None = None
        self.plan_start_step = 0
        self.battery_power_kw: np.ndarray | None = None
        self.pv_curtail_kw: np.ndarray | None = None
        self.solve_time_sec = 0.0
        if env is not None:
            self.solver = GlobalMisocpSolver(cfg)

    def reset(self, env_state: dict) -> None:
        self.env_state = dict(env_state)
        self.plan_key = None

    def __call__(self, env, window: dict[str, np.ndarray], *, progress_desc: str | None = None):
        del window, progress_desc
        if self.solver is None:
            self.solver = GlobalMisocpSolver(self.cfg)
        key = (int(env.episode), int(env.cur_step))
        if self.plan_key is None or int(env.cur_step) == 0:
            episode, step = env._cursor()
            horizon = int(env.episode_length - env.cur_step)
            price = derive_import_price_seq(env.data["price"][episode, step:step + horizon], markup_eur_per_kwh=float(get_import_price_markup(env.cfg)))
            load = np.asarray(env.data["load"][episode, step:step + horizon], dtype=np.float32).T
            pv_shared = np.asarray(env.data["pv"][episode, step:step + horizon], dtype=np.float32).reshape(1, -1)
            pv = np.repeat(pv_shared, int(env.n), axis=0).astype(np.float32)
            result = self.solver.solve(import_price_seq=price, load_seq=load, pv_seq=pv, soc_init=np.asarray(env.soc, dtype=np.float32))
            self.plan_key = key; self.plan_start_step = int(env.cur_step); self.battery_power_kw = result.battery_power_kw; self.pv_curtail_kw = result.pv_curtail_kw; self.solve_time_sec = float(result.solve_time_sec)
        offset = int(env.cur_step - self.plan_start_step)
        battery = np.asarray(self.battery_power_kw[:, offset], dtype=np.float32)
        curtail = np.asarray(self.pv_curtail_kw[:, offset], dtype=np.float32)
        actions, action_array, info = action_array_from_power(env, battery, curtail)
        solve_time = self.solve_time_sec if offset == 0 else 0.0
        info["solve_time_sec"] = np.asarray(solve_time, dtype=np.float32)
        return actions, info, {"solve_time_sec": solve_time}

__all__ = ["MisocpController"]
