from __future__ import annotations
from typing import Any
import numpy as np
import torch
_ACTION_EPS = 1e-06
_ACTION_TOL = 1e-05

def build_safety_local_numpy(*, soc: np.ndarray, load_raw: np.ndarray, pv_raw: np.ndarray, battery_capacity_kwh: np.ndarray, p_max_kw: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(soc, dtype=np.float32), np.asarray(load_raw, dtype=np.float32), np.asarray(pv_raw, dtype=np.float32), np.asarray(battery_capacity_kwh, dtype=np.float32), np.asarray(p_max_kw, dtype=np.float32)]).astype(np.float32)

def _canonicalize_actions_numpy(actions: np.ndarray | list[np.ndarray], *, default_pv_action: float=1.0, require_full_action_dim: bool=False) -> tuple[np.ndarray, bool]:
    action_array = np.asarray(actions, dtype=np.float32)
    squeezed = False
    if action_array.ndim == 1:
        action_array = action_array.reshape(1, -1)
        squeezed = True
    if action_array.ndim != 2:
        raise ValueError(f'Expected 2D action array, got shape {action_array.shape}.')
    if action_array.shape[-1] == 1 and (not require_full_action_dim):
        pv_column = np.full((action_array.shape[0], 1), float(default_pv_action), dtype=np.float32)
        action_array = np.concatenate([action_array, pv_column], axis=-1)
    elif action_array.shape[-1] != 2:
        raise ValueError(f'Expected action_dim=2, got shape {action_array.shape}.')
    return (action_array.astype(np.float32, copy=False), squeezed)

def _canonicalize_actions_torch(actions: torch.Tensor, *, default_pv_action: float=1.0, require_full_action_dim: bool=False) -> tuple[torch.Tensor, bool]:
    squeezed = False
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
        squeezed = True
    if actions.ndim != 3:
        raise ValueError(f'Expected action tensor rank 3, got shape {tuple(actions.shape)}.')
    if actions.shape[-1] == 1 and (not require_full_action_dim):
        pv_column = torch.full((*actions.shape[:-1], 1), float(default_pv_action), dtype=actions.dtype, device=actions.device)
        actions = torch.cat([actions, pv_column], dim=-1)
    elif actions.shape[-1] != 2:
        raise ValueError(f'Expected action_dim=2, got shape {tuple(actions.shape)}.')
    return (actions, squeezed)

def _local_bounds_numpy(safety_local: np.ndarray, *, efficiency: float, dt_hours: float, soc_min: float, soc_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    soc = np.clip(np.asarray(safety_local[..., 0], dtype=np.float32), soc_min, soc_max)
    capacity_kwh = np.maximum(np.asarray(safety_local[..., 3], dtype=np.float32), _ACTION_EPS)
    p_max_kw = np.maximum(np.asarray(safety_local[..., 4], dtype=np.float32), _ACTION_EPS)
    pv_raw_kw = np.maximum(np.asarray(safety_local[..., 2], dtype=np.float32), 0.0)
    energy_now = soc * capacity_kwh
    eff = max(float(efficiency), _ACTION_EPS)
    dt = max(float(dt_hours), _ACTION_EPS)
    energy_min = float(soc_min) * capacity_kwh
    energy_max = float(soc_max) * capacity_kwh
    charge_limit_kw = np.minimum(p_max_kw, np.maximum(0.0, (energy_max - energy_now) / (eff * dt)))
    discharge_limit_kw = np.minimum(p_max_kw, np.maximum(0.0, (energy_now - energy_min) * eff / dt))
    return (-discharge_limit_kw, charge_limit_kw, p_max_kw, pv_raw_kw)

def _local_bounds_torch(safety_local: torch.Tensor, *, efficiency: float, dt_hours: float, soc_min: float, soc_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    soc = torch.clamp(safety_local[..., 0], min=float(soc_min), max=float(soc_max))
    capacity_kwh = torch.clamp(safety_local[..., 3], min=_ACTION_EPS)
    p_max_kw = torch.clamp(safety_local[..., 4], min=_ACTION_EPS)
    pv_raw_kw = torch.clamp(safety_local[..., 2], min=0.0)
    energy_now = soc * capacity_kwh
    eff = max(float(efficiency), _ACTION_EPS)
    dt = max(float(dt_hours), _ACTION_EPS)
    energy_min = float(soc_min) * capacity_kwh
    energy_max = float(soc_max) * capacity_kwh
    charge_limit_kw = torch.minimum(p_max_kw, torch.clamp((energy_max - energy_now) / (eff * dt), min=0.0))
    discharge_limit_kw = torch.minimum(p_max_kw, torch.clamp((energy_now - energy_min) * eff / dt, min=0.0))
    return (-discharge_limit_kw, charge_limit_kw, p_max_kw, pv_raw_kw)

def compute_action_gap_metrics_numpy(safety_local: np.ndarray, requested_actions: np.ndarray | list[np.ndarray], executed_actions: np.ndarray | list[np.ndarray]) -> dict[str, np.ndarray]:
    requested, squeezed = _canonicalize_actions_numpy(requested_actions)
    executed, _ = _canonicalize_actions_numpy(executed_actions)
    safety_local = np.asarray(safety_local, dtype=np.float32)
    if safety_local.shape[0] != requested.shape[0]:
        raise ValueError(f'safety_local and action arrays should share the same agent dimension, got {safety_local.shape} vs {requested.shape}.')
    p_max_kw = np.maximum(safety_local[:, 4], _ACTION_EPS)
    pv_raw_kw = np.maximum(safety_local[:, 2], 0.0)
    battery_action_req = requested[:, 0].astype(np.float32)
    battery_action_exec = executed[:, 0].astype(np.float32)
    pv_action_req = requested[:, 1].astype(np.float32)
    pv_action_exec = executed[:, 1].astype(np.float32)
    battery_power_req_kw = (battery_action_req * p_max_kw).astype(np.float32)
    battery_power_exec_kw = (battery_action_exec * p_max_kw).astype(np.float32)
    pv_util_req = (0.5 * (pv_action_req + 1.0)).astype(np.float32)
    pv_util_exec = (0.5 * (pv_action_exec + 1.0)).astype(np.float32)
    pv_effective_req_kw = (pv_util_req * pv_raw_kw).astype(np.float32)
    pv_effective_exec_kw = (pv_util_exec * pv_raw_kw).astype(np.float32)
    pv_curtail_req_kw = (pv_raw_kw - pv_effective_req_kw).astype(np.float32)
    pv_curtail_exec_kw = (pv_raw_kw - pv_effective_exec_kw).astype(np.float32)
    battery_gap = np.abs(battery_power_req_kw - battery_power_exec_kw) / (p_max_kw + _ACTION_EPS)
    pv_gap = np.abs(pv_effective_req_kw - pv_effective_exec_kw) / np.maximum(pv_raw_kw, _ACTION_EPS)
    pv_gap = np.where(pv_raw_kw > _ACTION_EPS, pv_gap, 0.0).astype(np.float32)
    soc_penalty_unweighted = battery_gap.astype(np.float32)
    metrics = {'requested_action': requested.astype(np.float32), 'executed_action': executed.astype(np.float32), 'action_gap': np.abs(executed - requested).astype(np.float32), 'controller_action_gap': np.sum(np.abs(executed - requested), axis=-1).astype(np.float32), 'battery_action_req': battery_action_req, 'battery_action_exec': battery_action_exec, 'pv_action_req': pv_action_req, 'pv_action_exec': pv_action_exec, 'battery_power_req_kw': battery_power_req_kw, 'battery_power_exec_kw': battery_power_exec_kw, 'pv_effective_req_kw': pv_effective_req_kw, 'pv_effective_exec_kw': pv_effective_exec_kw, 'pv_curtail_req_kw': pv_curtail_req_kw, 'pv_curtail_exec_kw': pv_curtail_exec_kw, 'soc_penalty_unweighted': soc_penalty_unweighted, 'action_penalty_unweighted': soc_penalty_unweighted}
    if squeezed:
        return {key: np.asarray(value[0], dtype=np.float32) for key, value in metrics.items()}
    return metrics

def compute_action_gap_metrics_torch(safety_local: torch.Tensor, requested_actions: torch.Tensor, executed_actions: torch.Tensor) -> dict[str, torch.Tensor]:
    requested, squeezed = _canonicalize_actions_torch(requested_actions)
    executed, _ = _canonicalize_actions_torch(executed_actions)
    if int(safety_local.shape[0]) != int(requested.shape[0]):
        raise ValueError(f'safety_local batch dimension should match requested_actions batch dimension, got {tuple(safety_local.shape)} vs {tuple(requested.shape)}.')
    p_max_kw = torch.clamp(safety_local[..., 4], min=_ACTION_EPS)
    pv_raw_kw = torch.clamp(safety_local[..., 2], min=0.0)
    battery_action_req = requested[..., 0]
    battery_action_exec = executed[..., 0]
    pv_action_req = requested[..., 1]
    pv_action_exec = executed[..., 1]
    battery_power_req_kw = battery_action_req * p_max_kw
    battery_power_exec_kw = battery_action_exec * p_max_kw
    pv_util_req = 0.5 * (pv_action_req + 1.0)
    pv_util_exec = 0.5 * (pv_action_exec + 1.0)
    pv_effective_req_kw = pv_util_req * pv_raw_kw
    pv_effective_exec_kw = pv_util_exec * pv_raw_kw
    pv_curtail_req_kw = pv_raw_kw - pv_effective_req_kw
    pv_curtail_exec_kw = pv_raw_kw - pv_effective_exec_kw
    battery_gap = (battery_power_req_kw - battery_power_exec_kw).abs() / torch.clamp(p_max_kw, min=_ACTION_EPS)
    pv_gap = (pv_effective_req_kw - pv_effective_exec_kw).abs() / torch.clamp(pv_raw_kw, min=_ACTION_EPS)
    pv_gap = torch.where(pv_raw_kw > _ACTION_EPS, pv_gap, torch.zeros_like(pv_gap))
    soc_penalty_unweighted = battery_gap
    metrics = {'requested_action': requested, 'executed_action': executed, 'action_gap': (executed - requested).abs(), 'controller_action_gap': (executed - requested).abs().sum(dim=-1), 'battery_action_req': battery_action_req, 'battery_action_exec': battery_action_exec, 'pv_action_req': pv_action_req, 'pv_action_exec': pv_action_exec, 'battery_power_req_kw': battery_power_req_kw, 'battery_power_exec_kw': battery_power_exec_kw, 'pv_effective_req_kw': pv_effective_req_kw, 'pv_effective_exec_kw': pv_effective_exec_kw, 'pv_curtail_req_kw': pv_curtail_req_kw, 'pv_curtail_exec_kw': pv_curtail_exec_kw, 'soc_penalty_unweighted': soc_penalty_unweighted, 'action_penalty_unweighted': soc_penalty_unweighted}
    if squeezed:
        return {key: value.squeeze(0) for key, value in metrics.items()}
    return metrics

def enforce_local_action_feasibility_torch(safety_local: torch.Tensor, raw_actions: torch.Tensor, *, efficiency: float, dt_hours: float, soc_min: float, soc_max: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    requested_actions, squeezed = _canonicalize_actions_torch(raw_actions)
    battery_lower_kw, battery_upper_kw, p_max_kw, _ = _local_bounds_torch(safety_local, efficiency=efficiency, dt_hours=dt_hours, soc_min=soc_min, soc_max=soc_max)
    requested_battery_kw = requested_actions[..., 0] * p_max_kw
    executed_battery_kw = torch.clamp(requested_battery_kw, min=battery_lower_kw, max=battery_upper_kw)
    executed_battery_action = executed_battery_kw / torch.clamp(p_max_kw, min=_ACTION_EPS)
    executed_pv_action = torch.clamp(requested_actions[..., 1], min=-1.0, max=1.0)
    executed_actions = torch.stack([executed_battery_action, executed_pv_action], dim=-1)
    metrics = compute_action_gap_metrics_torch(safety_local, requested_actions, executed_actions)
    if squeezed:
        return (executed_actions.squeeze(0), metrics)
    return (executed_actions, metrics)

def action_info_to_numpy(action_info: dict[str, torch.Tensor] | None) -> dict[str, np.ndarray] | None:
    if action_info is None:
        return None
    converted: dict[str, np.ndarray] = {}
    for key, value in action_info.items():
        if isinstance(value, torch.Tensor):
            converted[key] = value.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            converted[key] = np.asarray(value, dtype=np.float32)
    return converted

def merge_action_info_into_step_info(info: dict[str, Any], action_info: dict[str, np.ndarray] | None, *, soc_pen_weight: float=0.0, action_pen_weight: float | None=None, apply_action_penalty: bool=False) -> tuple[dict[str, Any], np.ndarray]:
    if action_pen_weight is not None and soc_pen_weight == 0.0:
        soc_pen_weight = float(action_pen_weight)
    updated = dict(info)
    if action_info is None:
        existing_penalty = np.asarray(updated.get('r_soc_pen', updated.get('r_action_pen', np.zeros_like(np.asarray(updated.get('e_bat', []), dtype=np.float32)))), dtype=np.float32)
        return (updated, existing_penalty)
    for key, value in action_info.items():
        updated[key] = np.asarray(value, dtype=np.float32)
    if 'battery_power_req_kw' in action_info:
        updated['e_bat_req'] = np.asarray(action_info['battery_power_req_kw'], dtype=np.float32)
    if 'battery_power_exec_kw' in action_info:
        updated['e_bat'] = np.asarray(action_info['battery_power_exec_kw'], dtype=np.float32)
    if 'pv_effective_req_kw' in action_info:
        updated['pv_effective_req'] = np.asarray(action_info['pv_effective_req_kw'], dtype=np.float32)
    if 'pv_effective_exec_kw' in action_info:
        updated['pv_effective'] = np.asarray(action_info['pv_effective_exec_kw'], dtype=np.float32)
    if 'pv_curtail_req_kw' in action_info:
        updated['pv_curtail_req'] = np.asarray(action_info['pv_curtail_req_kw'], dtype=np.float32)
    if 'pv_curtail_exec_kw' in action_info:
        updated['pv_curtail'] = np.asarray(action_info['pv_curtail_exec_kw'], dtype=np.float32)
    if 'pv_action_exec' in action_info:
        updated['pv_action'] = np.asarray(action_info['pv_action_exec'], dtype=np.float32)
    n_agents = int(np.asarray(updated.get('e_bat', np.zeros(0, dtype=np.float32))).reshape(-1).shape[0])
    if apply_action_penalty:
        penalty = (float(soc_pen_weight) * np.asarray(action_info['soc_penalty_unweighted'], dtype=np.float32)).astype(np.float32)
    else:
        penalty = np.zeros((n_agents,), dtype=np.float32)
    updated['r_soc_pen'] = penalty
    updated['r_action_pen'] = penalty
    return (updated, penalty)

def validate_executed_actions_numpy(safety_local: np.ndarray, actions: np.ndarray | list[np.ndarray], *, efficiency: float, dt_hours: float, soc_min: float, soc_max: float) -> None:
    action_array, _ = _canonicalize_actions_numpy(actions, require_full_action_dim=True)
    if np.any(~np.isfinite(action_array)):
        raise ValueError('Environment received non-finite action values.')
    if np.any(action_array < -1.0 - _ACTION_TOL) or np.any(action_array > 1.0 + _ACTION_TOL):
        raise ValueError('Environment received action values outside [-1, 1]. Controllers must output already-feasible normalized actions.')
    battery_lower_kw, battery_upper_kw, p_max_kw, pv_raw_kw = _local_bounds_numpy(safety_local, efficiency=efficiency, dt_hours=dt_hours, soc_min=soc_min, soc_max=soc_max)
    battery_power_kw = action_array[:, 0] * p_max_kw
    invalid_battery = np.logical_or(battery_power_kw < battery_lower_kw - _ACTION_TOL, battery_power_kw > battery_upper_kw + _ACTION_TOL)
    if np.any(invalid_battery):
        bad_agent = int(np.nonzero(invalid_battery)[0][0])
        raise ValueError(f'Environment received a locally infeasible battery action for agent {bad_agent}: requested {battery_power_kw[bad_agent]:.4f} kW, allowed range [{battery_lower_kw[bad_agent]:.4f}, {battery_upper_kw[bad_agent]:.4f}] kW.')
    pv_utilization = 0.5 * (action_array[:, 1] + 1.0)
    pv_effective_kw = pv_utilization * pv_raw_kw
    invalid_pv = np.logical_or(pv_effective_kw < -_ACTION_TOL, pv_effective_kw > pv_raw_kw + _ACTION_TOL)
    if np.any(invalid_pv):
        bad_agent = int(np.nonzero(invalid_pv)[0][0])
        raise ValueError(f'Environment received a locally infeasible PV action for agent {bad_agent}: effective PV {pv_effective_kw[bad_agent]:.4f} kW, valid range [0.0000, {pv_raw_kw[bad_agent]:.4f}] kW.')
