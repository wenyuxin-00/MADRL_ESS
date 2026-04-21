from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch import nn
from envs.grid.core.grid_core import GridCore
from envs.grid.deployments import build_agent_deployments
SAFE_POC_ALGO_NAME = 'MATD3_SAFE_POC'
SAFETY_LOCAL_FIELD_NAMES = ('soc_raw', 'load_raw', 'pv_raw', 'battery_capacity_kwh', 'p_max_kw')
SAFETY_LOCAL_DIM = len(SAFETY_LOCAL_FIELD_NAMES)
_SAFETY_EPS = 1e-06
def is_safe_poc_algorithm(cfg_or_name: Any) -> bool:
    if isinstance(cfg_or_name, str):
        name = cfg_or_name
    else:
        algo_cfg = getattr(cfg_or_name, 'algo', None)
        name = getattr(algo_cfg, 'name', '')
    return str(name) == SAFE_POC_ALGO_NAME
def _row_norm_sq(rows: torch.Tensor) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.zeros((int(rows.shape[0]),), dtype=rows.dtype, device=rows.device)
    return torch.clamp(torch.sum(rows * rows, dim=-1), min=_SAFETY_EPS)
@dataclass(frozen=True)
class ProjectionDiagnostics:
    enabled: bool
    batch_size: int
    projected_fraction: float
    mean_abs_delta: float
    max_abs_delta: float
    mean_abs_delta_kw: float
    pre_violation: float
    post_violation: float
    pre_trafo_import_violation_kw: float = 0.0
    pre_trafo_export_violation_kw: float = 0.0
    post_trafo_import_violation_kw: float = 0.0
    post_trafo_export_violation_kw: float = 0.0
    def to_dict(self) -> dict[str, float | int | bool]:
        return {'enabled': bool(self.enabled), 'batch_size': int(self.batch_size), 'projected_fraction': float(self.projected_fraction), 'mean_abs_delta': float(self.mean_abs_delta), 'max_abs_delta': float(self.max_abs_delta), 'mean_abs_delta_kw': float(self.mean_abs_delta_kw), 'pre_violation': float(self.pre_violation), 'post_violation': float(self.post_violation), 'pre_trafo_import_violation_kw': float(self.pre_trafo_import_violation_kw), 'pre_trafo_export_violation_kw': float(self.pre_trafo_export_violation_kw), 'post_trafo_import_violation_kw': float(self.post_trafo_import_violation_kw), 'post_trafo_export_violation_kw': float(self.post_trafo_export_violation_kw)}
class JointGridSafetyProjector(nn.Module):
    def __init__(self, *, n_agents: int, voltage_sensitivity: np.ndarray, line_loading_sensitivity: np.ndarray, trafo_power_sensitivity: np.ndarray, voltage_base: np.ndarray, line_loading_base: np.ndarray, trafo_power_base_kw: np.ndarray, voltage_min_pu: float, voltage_max_pu: float, line_limit_pct: float, trafo_limit_pct: float, trafo_rating_kw: np.ndarray, efficiency: float, dt_hours: float, soc_min: float, soc_max: float, projector_mode: str, projection_iters: int, voltage_margin_pu: float, line_margin_pct: float, trafo_margin_pct: float, linearization_delta_kw: float) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.action_dim = 2
        self.projector_mode = str(projector_mode)
        self.projection_iters = max(int(projection_iters), 1)
        self.voltage_min_pu = float(voltage_min_pu)
        self.voltage_max_pu = float(voltage_max_pu)
        self.line_limit_pct = float(line_limit_pct)
        self.trafo_limit_pct = float(trafo_limit_pct)
        self.efficiency = float(efficiency)
        self.dt_hours = float(dt_hours)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.voltage_margin_pu = float(voltage_margin_pu)
        self.line_margin_pct = float(line_margin_pct)
        self.trafo_margin_pct = float(trafo_margin_pct)
        self.linearization_delta_kw = float(linearization_delta_kw)
        voltage_sensitivity_t = torch.as_tensor(voltage_sensitivity, dtype=torch.float32)
        line_loading_sensitivity_t = torch.as_tensor(line_loading_sensitivity, dtype=torch.float32)
        trafo_power_sensitivity_t = torch.as_tensor(trafo_power_sensitivity, dtype=torch.float32)
        self.register_buffer('voltage_sensitivity', voltage_sensitivity_t)
        self.register_buffer('line_loading_sensitivity', line_loading_sensitivity_t)
        self.register_buffer('trafo_power_sensitivity', trafo_power_sensitivity_t)
        self.register_buffer('voltage_base', torch.as_tensor(voltage_base, dtype=torch.float32))
        self.register_buffer('line_loading_base', torch.as_tensor(line_loading_base, dtype=torch.float32))
        self.register_buffer('trafo_power_base_kw', torch.as_tensor(trafo_power_base_kw, dtype=torch.float32))
        trafo_rating_kw_t = torch.as_tensor(trafo_rating_kw, dtype=torch.float32)
        trafo_limit_kw_t = trafo_rating_kw_t * (self.trafo_limit_pct / 100.0)
        trafo_usable_limit_kw_t = trafo_limit_kw_t * max(0.0, 1.0 - self.trafo_margin_pct / 100.0)
        self.register_buffer('trafo_import_limit_kw', trafo_usable_limit_kw_t.clone())
        self.register_buffer('trafo_export_limit_kw', trafo_usable_limit_kw_t.clone())
        self.register_buffer('voltage_rows_upper', voltage_sensitivity_t.clone())
        self.register_buffer('voltage_rows_lower', -voltage_sensitivity_t.clone())
        self.register_buffer('line_rows', line_loading_sensitivity_t.clone())
        self.register_buffer('trafo_rows_import', trafo_power_sensitivity_t.clone())
        self.register_buffer('trafo_rows_export', -trafo_power_sensitivity_t.clone())
        self.register_buffer('voltage_row_norm_sq', _row_norm_sq(voltage_sensitivity_t))
        self.register_buffer('line_row_norm_sq', _row_norm_sq(line_loading_sensitivity_t))
        self.register_buffer('trafo_row_norm_sq', _row_norm_sq(trafo_power_sensitivity_t))
        combined_rows = [voltage_sensitivity_t, -voltage_sensitivity_t]
        if line_loading_sensitivity_t.numel() > 0:
            combined_rows.append(line_loading_sensitivity_t)
        if trafo_power_sensitivity_t.numel() > 0:
            combined_rows.append(trafo_power_sensitivity_t)
            combined_rows.append(-trafo_power_sensitivity_t)
        combined_constraint_rows = torch.cat(combined_rows, dim=0)
        self.register_buffer('combined_constraint_rows', combined_constraint_rows)
        self.register_buffer('combined_constraint_row_norm_sq', _row_norm_sq(combined_constraint_rows))
    @classmethod
    def from_cfg(cls, cfg: Any, *, device: torch.device | str | None=None) -> 'JointGridSafetyProjector':
        deployments = build_agent_deployments(cfg)
        grid_core = GridCore(deployments, cfg.grid)
        n_agents = int(len(deployments))
        delta_kw = max(float(getattr(cfg.safety, 'linearization_delta_kw', 0.25)), 0.05)
        zero = np.zeros(n_agents, dtype=np.float32)
        grid_core.reset(zero, zero)
        baseline = grid_core.step(zero, zero)
        net_voltage_sensitivity = np.zeros((grid_core.n_buses, n_agents), dtype=np.float32)
        net_line_loading_sensitivity = np.zeros((grid_core.n_lines, n_agents), dtype=np.float32)
        trafo_power_available = bool(grid_core.n_trafos > 0 and hasattr(grid_core.net, 'res_trafo') and ('p_hv_mw' in getattr(grid_core.net, 'res_trafo').columns))
        if trafo_power_available:
            net_trafo_power_sensitivity = np.zeros((grid_core.n_trafos, n_agents), dtype=np.float32)
            trafo_power_base_kw = np.asarray(baseline.trafo_p_signed_kw, dtype=np.float32)
            trafo_rating_kw = np.asarray(grid_core.net.trafo['sn_mva'].to_numpy(dtype=np.float32), dtype=np.float32) * 1000.0
        else:
            net_trafo_power_sensitivity = np.zeros((0, n_agents), dtype=np.float32)
            trafo_power_base_kw = np.zeros(0, dtype=np.float32)
            trafo_rating_kw = np.zeros(0, dtype=np.float32)
        for agent_id in range(n_agents):
            perturb = zero.copy()
            perturb[agent_id] = np.float32(delta_kw)
            grid_core.reset(zero, zero)
            result_plus = grid_core.step(perturb, zero)
            grid_core.reset(zero, zero)
            result_minus = grid_core.step(-perturb, zero)
            net_voltage_sensitivity[:, agent_id] = (np.asarray(result_plus.vm_pu, dtype=np.float32) - np.asarray(baseline.vm_pu, dtype=np.float32)) / np.float32(delta_kw)
            if grid_core.n_lines > 0:
                net_line_loading_sensitivity[:, agent_id] = (np.asarray(result_plus.line_loading_pct, dtype=np.float32) - np.asarray(baseline.line_loading_pct, dtype=np.float32)) / np.float32(delta_kw)
            if trafo_power_available:
                net_trafo_power_sensitivity[:, agent_id] = (np.asarray(result_plus.trafo_p_signed_kw, dtype=np.float32) - np.asarray(result_minus.trafo_p_signed_kw, dtype=np.float32)) / np.float32(2.0 * delta_kw)
        voltage_sensitivity = np.concatenate([net_voltage_sensitivity, net_voltage_sensitivity], axis=1)
        line_loading_sensitivity = np.concatenate([net_line_loading_sensitivity, net_line_loading_sensitivity], axis=1)
        trafo_power_sensitivity = np.concatenate([net_trafo_power_sensitivity, net_trafo_power_sensitivity], axis=1)
        projector = cls(n_agents=n_agents, voltage_sensitivity=voltage_sensitivity, line_loading_sensitivity=line_loading_sensitivity, trafo_power_sensitivity=trafo_power_sensitivity, voltage_base=np.asarray(baseline.vm_pu, dtype=np.float32), line_loading_base=np.asarray(baseline.line_loading_pct, dtype=np.float32), trafo_power_base_kw=trafo_power_base_kw, voltage_min_pu=float(cfg.grid.v_min_pu), voltage_max_pu=float(cfg.grid.v_max_pu), line_limit_pct=float(cfg.grid.line_max_loading_pct), trafo_limit_pct=float(cfg.grid.line_max_loading_pct), trafo_rating_kw=trafo_rating_kw, efficiency=float(cfg.env.efficiency), dt_hours=float(cfg.env.dt), soc_min=float(cfg.env.soc_min), soc_max=float(cfg.env.soc_max), projector_mode=str(getattr(cfg.safety, 'projector_mode', 'joint_linearized')), projection_iters=int(getattr(cfg.safety, 'projection_iters', 6)), voltage_margin_pu=float(getattr(cfg.safety, 'voltage_margin_pu', 0.005)), line_margin_pct=float(getattr(cfg.safety, 'line_margin_pct', 5.0)), trafo_margin_pct=float(getattr(cfg.safety, 'trafo_margin_pct', 5.0)), linearization_delta_kw=delta_kw)
        if device is not None:
            projector = projector.to(device=torch.device(device))
        return projector
    def _extract_safety_local(self, obs_t: dict[str, torch.Tensor], *, dtype: torch.dtype) -> tuple[torch.Tensor, bool]:
        if 'safety_local' not in obs_t:
            raise KeyError("Observation does not contain 'safety_local'. Enable the safety-aware observation schema before using MATD3_SAFE_POC.")
        safety_local = obs_t['safety_local']
        squeezed = safety_local.ndim == 2
        if squeezed:
            safety_local = safety_local.unsqueeze(0)
        return (safety_local.to(dtype=dtype), squeezed)
    @staticmethod
    def _as_work_dtype(buffer: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if buffer.dtype == dtype and buffer.device == device:
            return buffer
        return buffer.to(device=device, dtype=dtype)
    def _local_bounds_kw(self, safety_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        soc = torch.clamp(safety_local[..., 0], min=self.soc_min, max=self.soc_max)
        capacity_kwh = torch.clamp(safety_local[..., 3], min=_SAFETY_EPS)
        p_max_kw = torch.clamp(safety_local[..., 4], min=_SAFETY_EPS)
        pv_raw_kw = torch.clamp(safety_local[..., 2], min=0.0)
        energy_now = soc * capacity_kwh
        eff = max(self.efficiency, _SAFETY_EPS)
        dt = max(self.dt_hours, _SAFETY_EPS)
        energy_min = self.soc_min * capacity_kwh
        energy_max = self.soc_max * capacity_kwh
        charge_limit_kw = torch.minimum(p_max_kw, torch.clamp((energy_max - energy_now) / (eff * dt), min=0.0))
        discharge_limit_kw = torch.minimum(p_max_kw, torch.clamp((energy_now - energy_min) * eff / dt, min=0.0))
        return (-discharge_limit_kw, charge_limit_kw, p_max_kw, pv_raw_kw)
    def _affine_terms(self, safety_local: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_net_load_kw = (safety_local[..., 1] - safety_local[..., 2]).to(dtype=dtype)
        voltage_sensitivity = self._as_work_dtype(self.voltage_sensitivity, dtype=dtype, device=device)[:, :self.n_agents]
        line_sensitivity = self._as_work_dtype(self.line_loading_sensitivity, dtype=dtype, device=device)[:, :self.n_agents]
        trafo_sensitivity = self._as_work_dtype(self.trafo_power_sensitivity, dtype=dtype, device=device)[:, :self.n_agents]
        voltage_base = self._as_work_dtype(self.voltage_base, dtype=dtype, device=device)
        line_base = self._as_work_dtype(self.line_loading_base, dtype=dtype, device=device)
        trafo_base = self._as_work_dtype(self.trafo_power_base_kw, dtype=dtype, device=device)
        voltage_affine = voltage_base.unsqueeze(0) + base_net_load_kw @ voltage_sensitivity.transpose(0, 1)
        line_affine = line_base.unsqueeze(0)
        if line_sensitivity.numel() > 0:
            line_affine = line_affine + base_net_load_kw @ line_sensitivity.transpose(0, 1)
        trafo_affine = trafo_base.unsqueeze(0)
        if trafo_sensitivity.numel() > 0:
            trafo_affine = trafo_affine + base_net_load_kw @ trafo_sensitivity.transpose(0, 1)
        return (voltage_affine, line_affine, trafo_affine)
    def _trafo_power_violation_terms(self, *, x_kw: torch.Tensor, trafo_affine: torch.Tensor, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(x_kw.shape[0])
        zero = torch.zeros((batch_size,), dtype=dtype, device=device)
        trafo_sensitivity = self._as_work_dtype(self.trafo_power_sensitivity, dtype=dtype, device=device)
        if trafo_sensitivity.numel() == 0 or trafo_affine.shape[-1] == 0:
            return (zero, zero)
        trafo_import_limit_kw = self._as_work_dtype(self.trafo_import_limit_kw, dtype=dtype, device=device)
        trafo_export_limit_kw = self._as_work_dtype(self.trafo_export_limit_kw, dtype=dtype, device=device)
        trafo_power = trafo_affine + x_kw @ trafo_sensitivity.transpose(0, 1)
        import_violation = torch.sum(torch.clamp(trafo_power - trafo_import_limit_kw.unsqueeze(0), min=0.0), dim=-1)
        export_violation = torch.sum(torch.clamp(-trafo_power - trafo_export_limit_kw.unsqueeze(0), min=0.0), dim=-1)
        return (import_violation, export_violation)
    def _approximate_violation(self, *, x_kw: torch.Tensor, voltage_affine: torch.Tensor, line_affine: torch.Tensor, trafo_affine: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        voltage_sensitivity = self._as_work_dtype(self.voltage_sensitivity, dtype=dtype, device=device)
        line_sensitivity = self._as_work_dtype(self.line_loading_sensitivity, dtype=dtype, device=device)
        voltage = voltage_affine + x_kw @ voltage_sensitivity.transpose(0, 1)
        voltage_high = torch.clamp(voltage - (self.voltage_max_pu - self.voltage_margin_pu), min=0.0)
        voltage_low = torch.clamp(self.voltage_min_pu + self.voltage_margin_pu - voltage, min=0.0)
        line_violation = torch.zeros((x_kw.shape[0],), dtype=dtype, device=device)
        if line_sensitivity.numel() > 0:
            line_loading = line_affine + x_kw @ line_sensitivity.transpose(0, 1)
            line_violation = torch.sum(torch.clamp(line_loading - (self.line_limit_pct - self.line_margin_pct), min=0.0), dim=-1)
        trafo_import_violation, trafo_export_violation = self._trafo_power_violation_terms(x_kw=x_kw, trafo_affine=trafo_affine, dtype=dtype, device=device)
        trafo_violation = trafo_import_violation + trafo_export_violation
        return torch.sum(voltage_high + voltage_low, dim=-1) + line_violation + trafo_violation
    def _project_rows_sequential(self, x_kw: torch.Tensor, rows: torch.Tensor, bounds: torch.Tensor, row_norm_sq: torch.Tensor) -> torch.Tensor:
        if rows.numel() == 0 or bounds.numel() == 0:
            return x_kw
        valid_indices = torch.nonzero(row_norm_sq > _SAFETY_EPS, as_tuple=False).squeeze(-1)
        for idx in valid_indices.tolist():
            row = rows[idx]
            bound = bounds[:, idx]
            violation = torch.sum(x_kw * row.unsqueeze(0), dim=-1) - bound
            correction = torch.clamp(violation, min=0.0).unsqueeze(-1) * row.unsqueeze(0) / row_norm_sq[idx]
            x_kw = x_kw - correction
        return x_kw
    def _project_rows_batched(self, x_kw: torch.Tensor, rows: torch.Tensor, bounds: torch.Tensor, row_norm_sq: torch.Tensor) -> torch.Tensor:
        if rows.numel() == 0 or bounds.numel() == 0:
            return x_kw
        valid_rows = row_norm_sq > _SAFETY_EPS
        if not torch.any(valid_rows):
            return x_kw
        rows = rows[valid_rows]
        bounds = bounds[:, valid_rows]
        row_norm_sq = row_norm_sq[valid_rows]
        violations = x_kw @ rows.transpose(0, 1) - bounds
        correction_weights = torch.clamp(violations, min=0.0) / row_norm_sq.unsqueeze(0)
        return x_kw - correction_weights @ rows
    def _combined_bounds(self, *, voltage_affine: torch.Tensor, line_affine: torch.Tensor, trafo_affine: torch.Tensor, voltage_low_limit: float, voltage_high_limit: float, line_limit: float, trafo_import_limit_kw: torch.Tensor, trafo_export_limit_kw: torch.Tensor) -> torch.Tensor:
        bounds = [voltage_high_limit - voltage_affine, voltage_affine - voltage_low_limit]
        if line_affine.shape[-1] > 0:
            bounds.append(line_limit - line_affine)
        if trafo_affine.shape[-1] > 0:
            bounds.append(trafo_import_limit_kw.unsqueeze(0) - trafo_affine)
            bounds.append(trafo_export_limit_kw.unsqueeze(0) + trafo_affine)
        return torch.cat(bounds, dim=-1)
    @staticmethod
    def _clamp_joint_action_kw(x_kw: torch.Tensor, *, battery_lower_kw: torch.Tensor, battery_upper_kw: torch.Tensor, pv_curtail_upper_kw: torch.Tensor, n_agents: int) -> torch.Tensor:
        battery_kw = torch.clamp(x_kw[..., :n_agents], min=battery_lower_kw, max=battery_upper_kw)
        pv_curtail_kw = torch.maximum(x_kw[..., n_agents:], torch.zeros_like(pv_curtail_upper_kw))
        pv_curtail_kw = torch.minimum(pv_curtail_kw, pv_curtail_upper_kw)
        return torch.cat([battery_kw, pv_curtail_kw], dim=-1)
    def project_actions_from_safety_local(self, safety_local: torch.Tensor, raw_actions: torch.Tensor, *, return_diagnostics: bool=False) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int | bool]]:
        squeezed = raw_actions.ndim == 2
        if squeezed:
            raw_actions = raw_actions.unsqueeze(0)
        if safety_local.ndim == 2:
            safety_local = safety_local.unsqueeze(0)
        if int(safety_local.shape[0]) != int(raw_actions.shape[0]):
            raise ValueError('safety_local batch dimension must match raw_actions batch dimension for safety projection.')
        original_dtype = raw_actions.dtype
        device = raw_actions.device
        work_dtype = torch.float32 if raw_actions.dtype in {torch.float16, torch.bfloat16} else raw_actions.dtype
        safety_local = safety_local.to(device=device, dtype=work_dtype)
        battery_action = raw_actions[..., 0].to(dtype=work_dtype)
        if raw_actions.shape[-1] >= 2:
            pv_action = raw_actions[..., 1].to(dtype=work_dtype)
        else:
            pv_action = torch.ones_like(battery_action, dtype=work_dtype)
        battery_lower_kw, battery_upper_kw, p_max_kw, pv_raw_kw = self._local_bounds_kw(safety_local)
        raw_battery_kw = battery_action * p_max_kw
        raw_pv_utilization = torch.clamp(0.5 * (pv_action + 1.0), min=0.0, max=1.0)
        raw_pv_curtail_kw = pv_raw_kw * (1.0 - raw_pv_utilization)
        raw_kw = torch.cat([raw_battery_kw, raw_pv_curtail_kw], dim=-1)
        x_kw = self._clamp_joint_action_kw(raw_kw, battery_lower_kw=battery_lower_kw, battery_upper_kw=battery_upper_kw, pv_curtail_upper_kw=pv_raw_kw, n_agents=self.n_agents)
        voltage_affine, line_affine, trafo_affine = self._affine_terms(safety_local, dtype=work_dtype, device=device)
        pre_violation = None
        if return_diagnostics:
            pre_violation = self._approximate_violation(x_kw=raw_kw, voltage_affine=voltage_affine, line_affine=line_affine, trafo_affine=trafo_affine, dtype=work_dtype, device=device)
        voltage_rows_upper = self._as_work_dtype(self.voltage_rows_upper, dtype=work_dtype, device=device)
        voltage_rows_lower = self._as_work_dtype(self.voltage_rows_lower, dtype=work_dtype, device=device)
        line_rows = self._as_work_dtype(self.line_rows, dtype=work_dtype, device=device)
        trafo_rows_import = self._as_work_dtype(self.trafo_rows_import, dtype=work_dtype, device=device)
        trafo_rows_export = self._as_work_dtype(self.trafo_rows_export, dtype=work_dtype, device=device)
        voltage_row_norm_sq = self._as_work_dtype(self.voltage_row_norm_sq, dtype=work_dtype, device=device)
        line_row_norm_sq = self._as_work_dtype(self.line_row_norm_sq, dtype=work_dtype, device=device)
        trafo_row_norm_sq = self._as_work_dtype(self.trafo_row_norm_sq, dtype=work_dtype, device=device)
        combined_constraint_rows = self._as_work_dtype(self.combined_constraint_rows, dtype=work_dtype, device=device)
        combined_constraint_row_norm_sq = self._as_work_dtype(self.combined_constraint_row_norm_sq, dtype=work_dtype, device=device)
        voltage_low_limit = self.voltage_min_pu + self.voltage_margin_pu
        voltage_high_limit = self.voltage_max_pu - self.voltage_margin_pu
        line_limit = self.line_limit_pct - self.line_margin_pct
        trafo_import_limit_kw = self._as_work_dtype(self.trafo_import_limit_kw, dtype=work_dtype, device=device)
        trafo_export_limit_kw = self._as_work_dtype(self.trafo_export_limit_kw, dtype=work_dtype, device=device)
        for _ in range(self.projection_iters):
            x_kw = self._clamp_joint_action_kw(x_kw, battery_lower_kw=battery_lower_kw, battery_upper_kw=battery_upper_kw, pv_curtail_upper_kw=pv_raw_kw, n_agents=self.n_agents)
            if self.projector_mode == 'joint_linearized_fast':
                combined_bounds = self._combined_bounds(voltage_affine=voltage_affine, line_affine=line_affine, trafo_affine=trafo_affine, voltage_low_limit=voltage_low_limit, voltage_high_limit=voltage_high_limit, line_limit=line_limit, trafo_import_limit_kw=trafo_import_limit_kw, trafo_export_limit_kw=trafo_export_limit_kw)
                x_kw = self._project_rows_batched(x_kw, combined_constraint_rows, combined_bounds, combined_constraint_row_norm_sq)
                continue
            x_kw = self._project_rows_sequential(x_kw, voltage_rows_upper, voltage_high_limit - voltage_affine, voltage_row_norm_sq)
            x_kw = self._project_rows_sequential(x_kw, voltage_rows_lower, voltage_affine - voltage_low_limit, voltage_row_norm_sq)
            x_kw = self._project_rows_sequential(x_kw, line_rows, line_limit - line_affine, line_row_norm_sq)
            x_kw = self._project_rows_sequential(x_kw, trafo_rows_import, trafo_import_limit_kw.unsqueeze(0) - trafo_affine, trafo_row_norm_sq)
            x_kw = self._project_rows_sequential(x_kw, trafo_rows_export, trafo_export_limit_kw.unsqueeze(0) + trafo_affine, trafo_row_norm_sq)
        x_kw = self._clamp_joint_action_kw(x_kw, battery_lower_kw=battery_lower_kw, battery_upper_kw=battery_upper_kw, pv_curtail_upper_kw=pv_raw_kw, n_agents=self.n_agents)
        projected_battery_kw = x_kw[..., :self.n_agents]
        projected_pv_curtail_kw = x_kw[..., self.n_agents:]
        projected_battery_action = torch.clamp(projected_battery_kw / torch.clamp(p_max_kw, min=_SAFETY_EPS), min=-1.0, max=1.0)
        projected_pv_utilization = torch.ones_like(projected_pv_curtail_kw, dtype=work_dtype)
        valid_pv_mask = pv_raw_kw > _SAFETY_EPS
        projected_pv_utilization[valid_pv_mask] = 1.0 - projected_pv_curtail_kw[valid_pv_mask] / torch.clamp(pv_raw_kw[valid_pv_mask], min=_SAFETY_EPS)
        projected_pv_utilization = torch.clamp(projected_pv_utilization, min=0.0, max=1.0)
        projected_pv_action = torch.clamp(2.0 * projected_pv_utilization - 1.0, min=-1.0, max=1.0)
        projected_actions = torch.stack([projected_battery_action, projected_pv_action], dim=-1)
        if raw_actions.shape[-1] > 2:
            projected_actions = torch.cat([projected_actions, raw_actions[..., 2:]], dim=-1)
        projected_actions = projected_actions.to(dtype=original_dtype)
        if squeezed:
            projected_actions = projected_actions.squeeze(0)
        if not return_diagnostics:
            return projected_actions
        post_violation = self._approximate_violation(x_kw=x_kw, voltage_affine=voltage_affine, line_affine=line_affine, trafo_affine=trafo_affine, dtype=work_dtype, device=device)
        pre_trafo_import_violation = torch.zeros_like(post_violation)
        pre_trafo_export_violation = torch.zeros_like(post_violation)
        if pre_violation is not None:
            pre_trafo_import_violation, pre_trafo_export_violation = self._trafo_power_violation_terms(x_kw=raw_kw, trafo_affine=trafo_affine, dtype=work_dtype, device=device)
        post_trafo_import_violation, post_trafo_export_violation = self._trafo_power_violation_terms(x_kw=x_kw, trafo_affine=trafo_affine, dtype=work_dtype, device=device)
        raw_action_slice = raw_actions[..., :projected_actions.shape[-1]].to(dtype=work_dtype)
        projected_action_slice = projected_actions.unsqueeze(0).to(dtype=work_dtype) if squeezed else projected_actions.to(dtype=work_dtype)
        delta = (projected_action_slice - raw_action_slice).abs()
        delta_kw = (x_kw - raw_kw).abs()
        projected_any = torch.any(delta.reshape(delta.shape[0], -1) > 1e-05, dim=-1)
        diagnostics = ProjectionDiagnostics(enabled=True, batch_size=int(battery_action.shape[0]), projected_fraction=float(projected_any.float().mean().item()), mean_abs_delta=float(delta.mean().item()), max_abs_delta=float(delta.max().item()), mean_abs_delta_kw=float(delta_kw.mean().item()), pre_violation=float(pre_violation.mean().item()) if pre_violation is not None else 0.0, post_violation=float(post_violation.mean().item()), pre_trafo_import_violation_kw=float(pre_trafo_import_violation.mean().item()), pre_trafo_export_violation_kw=float(pre_trafo_export_violation.mean().item()), post_trafo_import_violation_kw=float(post_trafo_import_violation.mean().item()), post_trafo_export_violation_kw=float(post_trafo_export_violation.mean().item()))
        return (projected_actions, diagnostics.to_dict())
    def project_actions(self, obs_t: dict[str, torch.Tensor], raw_actions: torch.Tensor, *, return_diagnostics: bool=False) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int | bool]]:
        safety_local, _ = self._extract_safety_local(obs_t, dtype=torch.float32)
        return self.project_actions_from_safety_local(safety_local, raw_actions, return_diagnostics=return_diagnostics)
