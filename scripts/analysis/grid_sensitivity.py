"""Offline grid-sensitivity diagnostics.

These helpers are intentionally excluded from the training loop. They provide a
finite-difference view of which agent injections most strongly influence local
voltages and worst-case thermal loadings at a selected operating point.
"""

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
        if getattr(env, "_grid_core", None) is None:
            raise ValueError("Static grid sensitivity requires GridEnv with an attached GridCore.")

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
        voltage_sensitivity = np.zeros((env.n, env.n), dtype=np.float32)
        line_loading_sensitivity = np.zeros(env.n, dtype=np.float32)
        trafo_loading_sensitivity = np.zeros(env.n, dtype=np.float32)

        for agent_id in range(env.n):
            plus = np.zeros(env.n, dtype=np.float32)
            minus = np.zeros(env.n, dtype=np.float32)
            plus[agent_id] = float(delta_kw)
            minus[agent_id] = -float(delta_kw)

            plus_result = _run(plus)
            minus_result = _run(minus)

            voltage_sensitivity[:, agent_id] = (
                plus_result.agent_vm_pu - minus_result.agent_vm_pu
            ) / (2.0 * float(delta_kw))

            plus_line = (
                float(np.max(plus_result.line_loading_pct))
                if plus_result.line_loading_pct.size
                else 0.0
            )
            minus_line = (
                float(np.max(minus_result.line_loading_pct))
                if minus_result.line_loading_pct.size
                else 0.0
            )
            line_loading_sensitivity[agent_id] = (plus_line - minus_line) / (2.0 * float(delta_kw))

            plus_trafo = (
                float(np.max(plus_result.trafo_loading_pct))
                if plus_result.trafo_loading_pct.size
                else 0.0
            )
            minus_trafo = (
                float(np.max(minus_result.trafo_loading_pct))
                if minus_result.trafo_loading_pct.size
                else 0.0
            )
            trafo_loading_sensitivity[agent_id] = (plus_trafo - minus_trafo) / (
                2.0 * float(delta_kw)
            )

        return {
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "delta_kw": float(delta_kw),
            "agent_bus_ids": np.asarray(getattr(cfg.grid, "agent_bus_ids", []), dtype=np.int32),
            "base_agent_vm_pu": np.asarray(base_result.agent_vm_pu, dtype=np.float32),
            "voltage_sensitivity_pu_per_kw": voltage_sensitivity,
            "line_loading_sensitivity_pct_per_kw": line_loading_sensitivity,
            "trafo_loading_sensitivity_pct_per_kw": trafo_loading_sensitivity,
        }
    finally:
        env.close()
