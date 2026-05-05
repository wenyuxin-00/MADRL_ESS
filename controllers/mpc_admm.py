from __future__ import annotations

import numpy as np

from configs.cfg import Cfg
from controllers.admm_mpc_solver import AdmmMpcWindowData, build_admm_mpc_coordination_cache, compute_default_rho, run_admm_mpc_step
from utils.price import derive_import_price_seq, get_import_price_markup


class AdmmMpcController:
    name = "ADMM_MPC"

    def __init__(self, cfg: Cfg, env=None, *, horizon: int | None = None) -> None:
        self.cfg = cfg
        default_horizon = int(cfg.obs.sequence_length)
        self.coordination_cache = None if env is None else build_admm_mpc_coordination_cache(cfg, env, horizon_steps=int(horizon if horizon is not None else default_horizon))

    def reset(self, env_state: dict) -> None:
        self.env_state = dict(env_state)

    def __call__(self, env, window: dict[str, np.ndarray], *, progress_desc: str | None = None) -> tuple[list[np.ndarray], dict[str, np.ndarray], dict[str, object]]:
        if self.coordination_cache is None:
            self.coordination_cache = build_admm_mpc_coordination_cache(self.cfg, env, horizon_steps=int(np.asarray(window["price_seq"]).size))
        capacity = np.asarray(env.cap, dtype=np.float32)
        data = AdmmMpcWindowData(import_price_eur_per_kwh=derive_import_price_seq(window["price_seq"], markup_eur_per_kwh=get_import_price_markup(env.cfg)), load_seq=np.asarray(window["load_seq"], dtype=np.float32), pv_seq=np.asarray(window["pv_seq"], dtype=np.float32), battery_capacity_kwh=capacity, p_max_kw=np.asarray(env.pmax, dtype=np.float32), efficiency=float(env.cfg.env.efficiency), energy_init_kwh=(env.soc * capacity).astype(np.float32), energy_min_kwh=(float(env.cfg.env.soc_min) * capacity).astype(np.float32), energy_max_kwh=(float(env.cfg.env.soc_max) * capacity).astype(np.float32), dt_hours=float(env.cfg.env.dt_hours))
        rho = compute_default_rho(data, self.coordination_cache)
        result = run_admm_mpc_step(env, data, coordination_cache=self.coordination_cache, rho_init=rho, rho_min=1e-3, rho_max=1e3, rho_adaptation="residual_balancing", max_iters=int(self.cfg.mpc.admm_max_iter), max_iters_first_step=max(int(self.cfg.mpc.admm_max_iter), 2), primal_tol=1e-3, dual_tol=1e-3, progress_desc=progress_desc)
        info = {"solve_time_sec": np.asarray(result.solve_time_sec, dtype=np.float32)}
        return [result.executed_action_array[idx].copy() for idx in range(int(env.n))], info, {"solve_time_sec": float(result.solve_time_sec), "admm_converged": bool(result.converged), "admm_iterations": int(result.iterations)}
