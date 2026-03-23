"""Helpers for recording grid-related episode traces."""

from __future__ import annotations

import numpy as np


def init_grid_record(n_agents: int) -> dict:
    """Create an empty grid history dict for one episode."""
    return {
        "agent_vm_pu": [[] for _ in range(int(n_agents))],
        "line_loading_pct": [],
        "trafo_loading_pct": [],
        "line_violation": [],
        "trafo_violation": [],
        "l_violation": [],
        "n_v_violations": [],
        "n_l_violations": [],
        "n_line_violations": [],
        "n_t_violations": [],
        "n_trafo_violations": [],
        "pf_converged": [],
        "psi_v_raw": [],
        "psi_line_raw": [],
        "psi_trafo_raw": [],
    }


def append_grid_step_record(grid_history: dict, info: dict) -> None:
    """Append one GridEnv step to a grid history dict."""
    if "agent_vm_pu" not in info:
        return

    vm = np.asarray(info["agent_vm_pu"], dtype=np.float32)
    n_agents = len(grid_history["agent_vm_pu"])
    for i in range(n_agents):
        grid_history["agent_vm_pu"][i].append(float(vm[i]))

    line_loading = info.get("line_loading_pct")
    if line_loading is not None:
        grid_history["line_loading_pct"].append(np.asarray(line_loading, dtype=np.float32).copy())
    else:
        grid_history["line_loading_pct"].append(np.zeros(0, dtype=np.float32))

    trafo_loading = info.get("trafo_loading_pct")
    if trafo_loading is not None:
        grid_history["trafo_loading_pct"].append(np.asarray(trafo_loading, dtype=np.float32).copy())
    else:
        grid_history["trafo_loading_pct"].append(np.zeros(0, dtype=np.float32))

    grid_history["line_violation"].append(float(info.get("line_violation", info.get("l_violation", 0.0))))
    grid_history["trafo_violation"].append(float(info.get("trafo_violation", 0.0)))
    grid_history["l_violation"].append(float(info.get("l_violation", 0.0)))
    grid_history["n_v_violations"].append(int(info.get("n_v_violations", 0)))
    grid_history["n_l_violations"].append(int(info.get("n_l_violations", 0)))
    grid_history["n_line_violations"].append(int(info.get("n_line_violations", 0)))
    grid_history["n_t_violations"].append(int(info.get("n_t_violations", 0)))
    grid_history["n_trafo_violations"].append(int(info.get("n_trafo_violations", info.get("n_t_violations", 0))))
    grid_history["pf_converged"].append(bool(info.get("pf_converged", True)))
    grid_history["psi_v_raw"].append(float(info.get("psi_v_raw", 0.0)))
    grid_history["psi_line_raw"].append(float(info.get("psi_line_raw", 0.0)))
    grid_history["psi_trafo_raw"].append(float(info.get("psi_trafo_raw", 0.0)))
