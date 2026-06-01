from __future__ import annotations

import numpy as np
import torch
from torch import nn

from configs.cfg import Cfg
from envs.grid_core import PowerFlowGridCore

EPS = 1e-6


def _safety_columns(safety_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(safety_local.shape[-1]) >= 9:
        return safety_local[..., 0], safety_local[..., 3], safety_local[..., 4], safety_local[..., 5], safety_local[..., 6], safety_local[..., 2], safety_local[..., 8]
    zeros = torch.zeros_like(safety_local[..., 0])
    return safety_local[..., 0], safety_local[..., 1], safety_local[..., 2], safety_local[..., 3], safety_local[..., 4], zeros, zeros


def _ev_charge_kw_from_action(cfg: Cfg, safety_local: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if int(actions.shape[-1]) < 3 or not bool(cfg.env.ev_enabled):
        return torch.zeros_like(actions[..., 0])
    _, _, _, _, _, ev_available, ev_pmax = _safety_columns(safety_local)
    return torch.clamp(ev_available, 0.0, 1.0) * torch.clamp(0.5 * (actions[..., 2] + 1.0), 0.0, 1.0) * torch.clamp(ev_pmax, min=0.0)


def _stack_action(cfg: Cfg, battery_action: torch.Tensor, pv_action: torch.Tensor, ev_action: torch.Tensor | None = None) -> torch.Tensor:
    if int(cfg.model.action_dim) >= 3:
        ev = torch.full_like(battery_action, -1.0) if ev_action is None else torch.clamp(ev_action, -1.0, 1.0)
        return torch.stack([torch.clamp(battery_action, -1.0, 1.0), torch.clamp(pv_action, -1.0, 1.0), ev], dim=-1)
    return torch.stack([torch.clamp(battery_action, -1.0, 1.0), torch.clamp(pv_action, -1.0, 1.0)], dim=-1)


def local_bounds_torch(cfg: Cfg, safety_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    soc_raw, _, pv_raw, cap_raw, pmax_raw, _, _ = _safety_columns(safety_local)
    soc = torch.clamp(soc_raw, float(cfg.env.soc_min), float(cfg.env.soc_max))
    cap = torch.clamp(cap_raw, min=EPS); pmax = torch.clamp(pmax_raw, min=EPS); pv = torch.clamp(pv_raw, min=0.0)
    dt, eff = max(float(cfg.env.dt_hours), EPS), max(float(cfg.env.efficiency), EPS)
    lower = -torch.minimum(pmax, torch.clamp((soc * cap - float(cfg.env.soc_min) * cap) * eff / dt, min=0.0))
    upper = torch.minimum(pmax, torch.clamp((float(cfg.env.soc_max) * cap - soc * cap) / (eff * dt), min=0.0))
    return lower, upper, pmax, pv


def map_actor_output_to_soc_feasible_action(cfg: Cfg, safety_local: torch.Tensor, raw_actions: torch.Tensor) -> torch.Tensor:
    raw = torch.clamp(raw_actions, -float(cfg.model.max_action), float(cfg.model.max_action))
    lower, upper, pmax, _ = local_bounds_torch(cfg, safety_local)
    battery_kw = lower + 0.5 * (raw[..., 0] + 1.0) * torch.clamp(upper - lower, min=0.0)
    ev_action = raw[..., 2] if int(raw.shape[-1]) >= 3 and bool(cfg.env.ev_enabled) else None
    return _stack_action(cfg, battery_kw / torch.clamp(pmax, min=EPS), raw[..., 1], ev_action)


def enforce_local_action_feasibility(cfg: Cfg, safety_local: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    lower, upper, pmax, _ = local_bounds_torch(cfg, safety_local)
    battery = torch.clamp(actions[..., 0] * pmax, min=lower, max=upper) / torch.clamp(pmax, min=EPS)
    ev_action = actions[..., 2] if int(actions.shape[-1]) >= 3 and bool(cfg.env.ev_enabled) else None
    return _stack_action(cfg, battery, actions[..., 1], ev_action)


class JointGridSafetyProjector(nn.Module):
    def __init__(self, cfg: Cfg) -> None:
        super().__init__(); self.cfg = cfg; core = PowerFlowGridCore(cfg)
        n, delta = int(cfg.env.num_agents), np.float32(0.25)
        zero = np.zeros(n, dtype=np.float32); base = core.step(zero)
        vm_sens = np.zeros((n, n), dtype=np.float32); line_sens = np.zeros((len(base.line_loading_pct), n), dtype=np.float32)
        for agent_id in range(n):
            perturb = zero.copy(); perturb[agent_id] = delta
            plus = core.step(perturb); minus = core.step(-perturb)
            vm_sens[:, agent_id] = (plus.vm_pu - minus.vm_pu) / (2.0 * delta)
            if line_sens.size:
                line_sens[:, agent_id] = (plus.line_loading_pct - minus.line_loading_pct) / (2.0 * delta)
        self.register_buffer("vm_base", torch.as_tensor(base.vm_pu, dtype=torch.float32))
        self.register_buffer("line_base", torch.as_tensor(base.line_loading_pct, dtype=torch.float32))
        self.register_buffer("vm_sens", torch.as_tensor(vm_sens, dtype=torch.float32))
        self.register_buffer("line_sens", torch.as_tensor(line_sens, dtype=torch.float32))
        self.trafo_limit_kw = float(core.trafo_limit_kw)

    def _rows(self, base_net: torch.Tensor, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(self.cfg.env.num_agents)
        vm_sens = self.vm_sens.to(device=device, dtype=dtype); line_sens = self.line_sens.to(device=device, dtype=dtype)
        rows, bounds = [], []
        vm_affine = self.vm_base.to(device=device, dtype=dtype).unsqueeze(0) + base_net @ vm_sens.T
        for sign, limit in ((1.0, float(self.cfg.grid.v_max_pu) - float(self.cfg.safety.voltage_margin_pu)), (-1.0, -float(self.cfg.grid.v_min_pu) - float(self.cfg.safety.voltage_margin_pu))):
            rows.append(torch.cat([sign * vm_sens, sign * vm_sens], dim=1)); bounds.append(torch.as_tensor(limit, dtype=dtype, device=device) - sign * vm_affine)
        if line_sens.numel():
            line_affine = self.line_base.to(device=device, dtype=dtype).unsqueeze(0) + base_net @ line_sens.T
            rows.append(torch.cat([line_sens, line_sens], dim=1)); bounds.append(torch.as_tensor(float(self.cfg.grid.line_max_loading_pct) - float(self.cfg.safety.line_margin_pct), dtype=dtype, device=device) - line_affine)
        net_row = torch.ones((1, 2 * n), dtype=dtype, device=device); base_sum = torch.sum(base_net, dim=1, keepdim=True); limit = torch.as_tensor(self.trafo_limit_kw * (1.0 - float(self.cfg.safety.trafo_margin_pct) / 100.0), dtype=dtype, device=device)
        rows.extend([net_row, -net_row]); bounds.extend([limit - base_sum, limit + base_sum])
        return torch.cat(rows, dim=0), torch.cat(bounds, dim=1)

    @staticmethod
    def _project_rows(x: torch.Tensor, rows: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        norms = torch.clamp(torch.sum(rows * rows, dim=1), min=EPS)
        for idx in range(int(rows.shape[0])):
            row = rows[idx]; violation = torch.sum(x * row.unsqueeze(0), dim=1) - bounds[:, idx]
            x = x - torch.clamp(violation, min=0.0).unsqueeze(1) * row.unsqueeze(0) / norms[idx]
        return x

    def project_actions_from_safety_local(self, safety_local: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        squeezed = actions.ndim == 2
        if squeezed:
            actions = actions.unsqueeze(0); safety_local = safety_local.unsqueeze(0) if safety_local.ndim == 2 else safety_local
        dtype, device, n = actions.dtype, actions.device, int(self.cfg.env.num_agents)
        safety = safety_local.to(device=device, dtype=dtype)
        lower, upper, pmax, pv = local_bounds_torch(self.cfg, safety)
        battery_kw = torch.clamp(actions[..., 0] * pmax, min=lower, max=upper)
        pv_curtail_kw = torch.minimum(torch.clamp(pv * (1.0 - torch.clamp(0.5 * (actions[..., 1] + 1.0), 0.0, 1.0)), min=0.0), pv)
        x = torch.cat([battery_kw, pv_curtail_kw], dim=1)
        _, load, pv_raw, _, _, _, _ = _safety_columns(safety)
        rows, bounds = self._rows(load - pv_raw + _ev_charge_kw_from_action(self.cfg, safety, actions), dtype, device)
        for _ in range(max(int(self.cfg.safety.projection_iters), 1)):
            x = self._project_rows(x, rows, bounds)
            x = torch.cat([torch.clamp(x[:, :n], min=lower, max=upper), torch.minimum(torch.clamp(x[:, n:], min=0.0), pv)], dim=1)
        battery_action = torch.clamp(x[:, :n] / torch.clamp(pmax, min=EPS), -1.0, 1.0)
        pv_util = torch.where(pv > EPS, 1.0 - x[:, n:] / torch.clamp(pv, min=EPS), torch.ones_like(pv))
        ev_action = actions[..., 2] if int(actions.shape[-1]) >= 3 and bool(self.cfg.env.ev_enabled) else None
        projected = _stack_action(self.cfg, battery_action, 2.0 * pv_util - 1.0, ev_action)
        return projected.squeeze(0) if squeezed else projected
