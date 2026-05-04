from __future__ import annotations

import numpy as np

from REMAKE.envs.grid_env import project_action_to_soc


def action_array_from_power(env, battery_power_kw: np.ndarray, pv_curtail_kw: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    cfg = env.cfg
    pmax = np.asarray(env.pmax, dtype=np.float32)
    battery_action = project_action_to_soc(cfg, env.soc, np.asarray(battery_power_kw, dtype=np.float32).reshape(-1) / np.maximum(pmax, 1e-6))
    episode, step = env._cursor()
    pv_raw_kw = np.maximum(np.repeat(np.float32(env.data["pv"][episode, step]), int(env.n)), 0.0)
    pv_effective_kw = np.maximum(pv_raw_kw - np.asarray(pv_curtail_kw, dtype=np.float32).reshape(-1), 0.0)
    pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
    mask = pv_raw_kw > 1e-6
    pv_utilization[mask] = pv_effective_kw[mask] / pv_raw_kw[mask]
    action_array = np.stack([battery_action, np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0)], axis=-1).astype(np.float32)
    zeros = np.zeros((int(env.n),), dtype=np.float32)
    return [action_array[idx].copy() for idx in range(int(env.n))], action_array, {"action_gap_l1": zeros, "action_gap_linf": zeros}
