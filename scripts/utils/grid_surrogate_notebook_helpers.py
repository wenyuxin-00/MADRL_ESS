"""Shared transformer-surrogate helpers for notebook workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from controllers.madrl.safety_projector import JointGridSafetyProjector


@dataclass(frozen=True)
class GridTrafoSurrogate:
    trafo_limit_kw: float
    alpha_netload_kw: np.ndarray
    alpha_netload_window_kw: np.ndarray
    baseline_root_p_kw: np.ndarray
    export_overload_mask: np.ndarray
    import_overload_mask: np.ndarray


def _projector_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _get_grid_projector(cfg, env) -> JointGridSafetyProjector:
    projector = getattr(env, "_grid_notebook_projector", None)
    if isinstance(projector, JointGridSafetyProjector):
        return projector
    projector = JointGridSafetyProjector.from_cfg(cfg, device="cpu")
    setattr(env, "_grid_notebook_projector", projector)
    return projector


def _battery_to_netload_sensitivity(projector: JointGridSafetyProjector) -> np.ndarray:
    """Map battery signed-power sensitivity into net-load sensitivity."""

    trafo_sensitivity = _projector_to_numpy(projector.trafo_power_sensitivity)
    if trafo_sensitivity.size == 0:
        return np.zeros((int(projector.n_agents),), dtype=np.float32)
    if trafo_sensitivity.shape[0] != 1:
        raise ValueError(
            "Notebook surrogate helpers expect exactly one transformer sensitivity row, "
            f"got shape {trafo_sensitivity.shape}."
        )
    return np.asarray(trafo_sensitivity[0, : int(projector.n_agents)], dtype=np.float32)


__all__ = [
    "GridTrafoSurrogate",
    "_battery_to_netload_sensitivity",
    "_get_grid_projector",
    "_projector_to_numpy",
]
