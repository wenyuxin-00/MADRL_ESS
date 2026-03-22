"""Structured result types for grid power-flow steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridStepResult:
    """Container for one pandapower step result."""

    converged: bool
    vm_pu: np.ndarray
    va_degree: np.ndarray
    line_loading_pct: np.ndarray
    trafo_loading_pct: np.ndarray
    p_mw_from: np.ndarray
    agent_vm_pu: np.ndarray
    v_violation: np.ndarray
    line_violation: float
    trafo_violation: float
    l_violation: float
    n_buses: int
    n_lines: int
    n_trafos: int
