"""Structured result types for grid power-flow steps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _empty_f32() -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


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

    # Full-grid safety diagnostics used by the global reward.
    bus_v_excess: np.ndarray = field(default_factory=_empty_f32)
    bus_v_signed_indicator: np.ndarray = field(default_factory=_empty_f32)
    line_excess: np.ndarray = field(default_factory=_empty_f32)
    trafo_excess: np.ndarray = field(default_factory=_empty_f32)
    psi_v_raw: float = 0.0
    psi_line_raw: float = 0.0
    psi_trafo_raw: float = 0.0
