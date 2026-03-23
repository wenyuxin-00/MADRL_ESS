"""Offline finite-difference sensitivity diagnostics for GridEnv."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from scripts.builder import build_env


def estimate_static_grid_sensitivity(
    cfg: Any,
    *,
    episode_idx: int = 0,
    step_idx: int = 0,
    delta_kw: float = 1.0,
    mode: str = "test",
) -> dict[str, np.ndarray | float | int]:
    """Estimate voltage and thermal sensitivities by central finite differences."""
    if delta_kw <= 0.0:
        raise ValueError(f"delta_kw must be positive, got {delta_kw}.")

    diag_cfg = deepcopy(cfg)
    env = build_env(diag_cfg, mode=mode)
    try:
        env.reset(episode_idx=episode_idx)
        if step_idx < 0 or step_idx >= int(env.episode_length):
            raise IndexError(
                f"step_idx={step_idx} is out of range [0, {int(env.episode_length) - 1}]."
            )

        base_load_kw = np.asarray(env.ep_load[step_idx], dtype=np.float32)
        base_pv_kw = np.asarray(env.ep_pv[step_idx], dtype=np.float32)
        base_net_load_kw = (base_load_kw - base_pv_kw).astype(np.float32)
        base_action = np.zeros(env.n, dtype=np.float32)
        grid_core = env._grid_core

        def _run(p_batt_kw: np.ndarray):
            grid_core.reset(base_load_kw, base_pv_kw)
            return grid_core.step(
                p_batt_kw=np.asarray(p_batt_kw, dtype=np.float32),
                base_load_kw=base_net_load_kw,
            )

        base_result = _run(base_action)
        n_buses = grid_core.n_buses
        n_lines = grid_core.n_lines
        n_trafos = grid_core.n_trafos
        voltage_sensitivity = np.zeros((n_buses, env.n), dtype=np.float32)
        line_loading_sensitivity = np.zeros((n_lines, env.n), dtype=np.float32)
        trafo_loading_sensitivity = np.zeros((n_trafos, env.n), dtype=np.float32)

        for agent_id in range(env.n):
            plus = np.zeros(env.n, dtype=np.float32)
            minus = np.zeros(env.n, dtype=np.float32)
            plus[agent_id] = float(delta_kw)
            minus[agent_id] = -float(delta_kw)

            plus_result = _run(plus)
            minus_result = _run(minus)
            inv_2delta = 1.0 / (2.0 * float(delta_kw))

            voltage_sensitivity[:, agent_id] = (plus_result.vm_pu - minus_result.vm_pu) * inv_2delta
            line_loading_sensitivity[:, agent_id] = (
                plus_result.line_loading_pct - minus_result.line_loading_pct
            ) * inv_2delta
            trafo_loading_sensitivity[:, agent_id] = (
                plus_result.trafo_loading_pct - minus_result.trafo_loading_pct
            ) * inv_2delta

        agent_bus_positions = [grid_core._bus_id_to_pos[bus_id] for bus_id in grid_core.agent_bus_ids]
        agent_voltage_sensitivity = voltage_sensitivity[agent_bus_positions, :]

        return {
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "delta_kw": float(delta_kw),
            "agent_bus_ids": np.asarray(getattr(cfg.grid, "agent_bus_ids", []), dtype=np.int32),
            "base_agent_vm_pu": np.asarray(base_result.agent_vm_pu, dtype=np.float32),
            "voltage_sensitivity_pu_per_kw": agent_voltage_sensitivity,
            "line_loading_sensitivity_pct_per_kw": np.array(
                [
                    float(np.max(np.abs(line_loading_sensitivity[:, i])))
                    if line_loading_sensitivity[:, i].size
                    else 0.0
                    for i in range(env.n)
                ],
                dtype=np.float32,
            ),
            "trafo_loading_sensitivity_pct_per_kw": np.array(
                [
                    float(np.max(np.abs(trafo_loading_sensitivity[:, i])))
                    if trafo_loading_sensitivity[:, i].size
                    else 0.0
                    for i in range(env.n)
                ],
                dtype=np.float32,
            ),
            "full_bus_voltage_sensitivity": voltage_sensitivity,
            "full_line_loading_sensitivity": line_loading_sensitivity,
            "full_trafo_loading_sensitivity": trafo_loading_sensitivity,
        }
    finally:
        env.close()
