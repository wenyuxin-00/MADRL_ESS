"""Result dataclasses for one pandapower power-flow step.

These are pure data holders — no logic, no pandapower import required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridStepResult:
    """Raw physics output from one pandapower power-flow solve.

    Attributes
    ----------
    converged:
        Whether the power flow converged. If False, all arrays carry the
        last successfully converged values (or zeros for the first step).
    vm_pu:
        Per-bus voltage magnitude in per-unit. Shape ``(n_buses,)``.
    va_degree:
        Per-bus voltage angle in degrees. Shape ``(n_buses,)``.
    line_loading_pct:
        Line utilisation as a percentage of the thermal rating (0–100+).
        Shape ``(n_lines,)``.
    p_mw_from:
        Active power flow at the *from* end of each line in MW.
        Shape ``(n_lines,)``.
    agent_vm_pu:
        Voltage magnitude at the buses where agents are deployed.
        Shape ``(n_agents,)``.
    v_violation:
        Per-agent voltage violation magnitude in pu (>= 0).
        ``v_violation[i] = max(0, v_min - vm[i]) + max(0, vm[i] - v_max)``.
        Shape ``(n_agents,)``.
    l_violation:
        Worst-case line overload beyond the thermal limit (>= 0).
        ``max(0, max(line_loading_pct) - line_max_loading_pct) / 100``.
    n_buses:
        Total number of buses in the network.
    n_lines:
        Total number of lines in the network.
    """

    converged: bool
    vm_pu: np.ndarray
    va_degree: np.ndarray
    line_loading_pct: np.ndarray
    p_mw_from: np.ndarray
    agent_vm_pu: np.ndarray
    v_violation: np.ndarray
    l_violation: float
    n_buses: int
    n_lines: int
