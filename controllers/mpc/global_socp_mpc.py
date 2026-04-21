from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import math
import os
from typing import Any, Sequence
import numpy as np
try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    gp = None
    GRB = None
from controllers.action_feasibility import build_safety_local_numpy, compute_action_gap_metrics_numpy
from controllers.base import BaseController
from envs.grid.core.net_builder import build_simbench_net
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY, WHOLESALE_PRICE_SEQ_FIELD, derive_import_price_seq, get_import_price_markup
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DEBUG_DIR = _PROJECT_ROOT / 'artifacts' / 'misocp_debug'
_GUROBI_ERROR_PREFIX = 'Global SOCP-MPC requires a working Gurobi installation/license'
_ROOT_VM_EPS = 1e-06
_PV_EPS_KW = 1e-06
_VOLTAGE_ERR_TOL_PU = 0.001
_LINE_LOADING_ERR_TOL_PCT = 1.0
_TRAFO_LOADING_ERR_TOL_PCT = 1.0
_ROOT_POWER_ERR_TOL_KW = 0.1
_TRANSFORMER_SANITY_TOL_PU = 0.005
_TIME_LIMIT_SEC = 30.0
_FULL_HORIZON_TIME_LIMIT_SEC = 600.0
_MIP_GAP = 0.001
_THROUGHPUT_REGULARIZATION_EUR_PER_KWH = 0.0001
_SIMULTANEOUS_THRESHOLD_RATIO = 0.01
_SIMULTANEOUS_WARNING_RATIO = 0.05
_ADAPTIVE_PRIMARY_WINDOW_STEPS = 384
_ADAPTIVE_FALLBACK_WINDOW_STEPS = 96
_ADAPTIVE_PRIMARY_TIME_LIMIT_SEC = 300.0
_ADAPTIVE_RETRY_TIME_LIMIT_SEC = 600.0
_ADAPTIVE_TARGET_MIP_GAP = 0.005

@dataclass(frozen=True)
class NetworkModel:
    s_base_mva: float
    root_bus_id: int
    trafo_lv_bus_id: int
    root_vm_pu: float
    loading_limit_scale: float
    bus_ids: tuple[int, ...]
    bus_pos: dict[int, int]
    v_base_kv: np.ndarray
    p_base_mw: np.ndarray
    q_base_mvar: np.ndarray
    branch_parent_pos: np.ndarray
    branch_child_pos: np.ndarray
    branch_r_pu: np.ndarray
    branch_x_pu: np.ndarray
    branch_is_trafo: np.ndarray
    branch_l_max_pu: np.ndarray
    incoming_branch_by_bus: np.ndarray
    outgoing_branches_by_bus: tuple[tuple[int, ...], ...]
    root_outgoing_branches: np.ndarray
    trafo_branch_index: int
    line_branch_indices: np.ndarray
    agent_bus_positions: np.ndarray

    @classmethod
    def from_cfg(cls, cfg: Any, *, agent_bus_ids: list[int]) -> 'NetworkModel':
        grid_cfg = cfg.grid
        net = build_simbench_net(str(grid_cfg.sb_code))
        if len(getattr(net, 'ext_grid', [])) != 1:
            raise ValueError('Global SOCP-MPC requires exactly one ext_grid bus.')
        if len(getattr(net, 'trafo', [])) != 1:
            raise ValueError('Global SOCP-MPC currently requires exactly one transformer.')
        root_bus_id = int(net.ext_grid.iloc[0]['bus'])
        root_vm_pu = float(net.ext_grid.iloc[0]['vm_pu']) if 'vm_pu' in net.ext_grid.columns else 1.0
        trafo_row = net.trafo.iloc[0]
        trafo_hv_bus = int(trafo_row['hv_bus'])
        trafo_lv_bus = int(trafo_row['lv_bus'])
        if trafo_hv_bus != root_bus_id:
            raise ValueError(f'Global SOCP-MPC expects the transformer HV bus to match the ext_grid root bus, got ext_grid={root_bus_id} and trafo_hv_bus={trafo_hv_bus}.')
        s_base_mva = float(trafo_row['sn_mva'])
        if s_base_mva <= 0.0:
            raise ValueError(f'Transformer sn_mva must be positive, got {s_base_mva}.')
        agent_bus_ids = [int(bus_id) for bus_id in agent_bus_ids]
        cls._zero_agent_bus_active_injections(net, agent_bus_ids)
        p_base_mw, q_base_mvar = cls._aggregate_bus_demands(net)
        bus_ids = tuple((int(bus_id) for bus_id in net.bus.index.tolist()))
        bus_pos = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        v_base_kv = np.asarray([float(net.bus.at[bus_id, 'vn_kv']) for bus_id in bus_ids], dtype=np.float32)
        raw_edges: list[dict[str, object]] = []
        vk_percent = float(trafo_row.get('vk_percent', 0.0))
        vkr_percent = float(trafo_row.get('vkr_percent', 0.0))
        x_percent = float(np.sqrt(max(vk_percent * vk_percent - vkr_percent * vkr_percent, 0.0)))
        raw_edges.append({'u': root_bus_id, 'v': trafo_lv_bus, 'r_pu': float(vkr_percent / 100.0), 'x_pu': float(x_percent / 100.0), 'l_max_pu': np.nan, 'is_trafo': True})
        for _, row in net.line.iterrows():
            from_bus = int(row['from_bus'])
            to_bus = int(row['to_bus'])
            vn_kv = float(net.bus.at[from_bus, 'vn_kv'])
            if vn_kv <= 0.0:
                raise ValueError(f'Bus {from_bus} has invalid vn_kv={vn_kv}.')
            z_base_ohm = vn_kv * vn_kv / s_base_mva
            parallel = float(row.get('parallel', 1.0) or 1.0)
            df = float(row.get('df', 1.0) or 1.0)
            length_km = float(row.get('length_km', 0.0))
            r_total_ohm = float(row['r_ohm_per_km']) * length_km / max(parallel, 1.0)
            x_total_ohm = float(row['x_ohm_per_km']) * length_km / max(parallel, 1.0)
            i_base_ka = s_base_mva / (np.sqrt(3.0) * vn_kv)
            max_i_ka = float(row.get('max_i_ka', 0.0)) * max(parallel, 1.0) * max(df, 0.0)
            l_max_pu = np.inf if max_i_ka <= 0.0 else float((max_i_ka / max(i_base_ka, _ROOT_VM_EPS)) ** 2)
            raw_edges.append({'u': from_bus, 'v': to_bus, 'r_pu': float(r_total_ohm / z_base_ohm), 'x_pu': float(x_total_ohm / z_base_ohm), 'l_max_pu': l_max_pu, 'is_trafo': False})
        branch_parent_pos, branch_child_pos, branch_r_pu, branch_x_pu, branch_l_max_pu, branch_is_trafo, incoming_branch_by_bus, outgoing_branches_by_bus = cls._build_tree(bus_ids, bus_pos, raw_edges, root_bus_id)
        root_outgoing = np.asarray(outgoing_branches_by_bus[bus_pos[root_bus_id]], dtype=np.int32)
        if root_outgoing.size != 1:
            raise ValueError(f'Global SOCP-MPC expects exactly one outgoing root branch, got {int(root_outgoing.size)}.')
        trafo_branch_index = int(root_outgoing[0])
        if not bool(branch_is_trafo[trafo_branch_index]):
            raise ValueError('The root branch must be the transformer branch.')
        try:
            agent_bus_positions = np.asarray([bus_pos[bus_id] for bus_id in agent_bus_ids], dtype=np.int32)
        except KeyError as exc:
            raise ValueError(f'Unknown agent bus id {exc.args[0]!r} in NetworkModel extraction.') from exc
        line_branch_indices = np.asarray([idx for idx, is_trafo in enumerate(branch_is_trafo.tolist()) if not bool(is_trafo)], dtype=np.int32)
        return cls(s_base_mva=s_base_mva, root_bus_id=root_bus_id, trafo_lv_bus_id=trafo_lv_bus, root_vm_pu=float(root_vm_pu), loading_limit_scale=float(max(float(grid_cfg.line_max_loading_pct), 0.0) / 100.0), bus_ids=bus_ids, bus_pos=bus_pos, v_base_kv=v_base_kv.astype(np.float32, copy=False), p_base_mw=p_base_mw.astype(np.float32, copy=False), q_base_mvar=q_base_mvar.astype(np.float32, copy=False), branch_parent_pos=branch_parent_pos.astype(np.int32, copy=False), branch_child_pos=branch_child_pos.astype(np.int32, copy=False), branch_r_pu=branch_r_pu.astype(np.float32, copy=False), branch_x_pu=branch_x_pu.astype(np.float32, copy=False), branch_is_trafo=branch_is_trafo.astype(bool, copy=False), branch_l_max_pu=branch_l_max_pu.astype(np.float32, copy=False), incoming_branch_by_bus=incoming_branch_by_bus.astype(np.int32, copy=False), outgoing_branches_by_bus=outgoing_branches_by_bus, root_outgoing_branches=root_outgoing.astype(np.int32, copy=False), trafo_branch_index=trafo_branch_index, line_branch_indices=line_branch_indices, agent_bus_positions=agent_bus_positions.astype(np.int32, copy=False))

    @staticmethod
    def _zero_agent_bus_active_injections(net: Any, agent_bus_ids: list[int]) -> None:
        agent_bus_set = {int(bus_id) for bus_id in agent_bus_ids}
        if hasattr(net, 'load') and (not net.load.empty):
            load_mask = net.load['bus'].isin(agent_bus_set)
            if bool(load_mask.any()):
                net.load.loc[load_mask, ['p_mw', 'q_mvar']] = 0.0
        if hasattr(net, 'sgen') and (not net.sgen.empty):
            sgen_mask = net.sgen['bus'].isin(agent_bus_set)
            if bool(sgen_mask.any()):
                net.sgen.loc[sgen_mask, ['p_mw', 'q_mvar']] = 0.0

    @staticmethod
    def _aggregate_bus_demands(net: Any) -> tuple[np.ndarray, np.ndarray]:
        bus_ids = [int(bus_id) for bus_id in net.bus.index.tolist()]
        bus_pos = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        p_base = np.zeros((len(bus_ids),), dtype=np.float32)
        q_base = np.zeros((len(bus_ids),), dtype=np.float32)
        if hasattr(net, 'load') and (not net.load.empty):
            load_group = net.load.groupby('bus', sort=False)[['p_mw', 'q_mvar']].sum()
            for bus_id, row in load_group.iterrows():
                idx = bus_pos[int(bus_id)]
                p_base[idx] += float(row.get('p_mw', 0.0))
                q_base[idx] += float(row.get('q_mvar', 0.0))
        if hasattr(net, 'sgen') and (not net.sgen.empty):
            sgen_group = net.sgen.groupby('bus', sort=False)[['p_mw', 'q_mvar']].sum()
            for bus_id, row in sgen_group.iterrows():
                idx = bus_pos[int(bus_id)]
                p_base[idx] -= float(row.get('p_mw', 0.0))
                q_base[idx] -= float(row.get('q_mvar', 0.0))
        return (p_base, q_base)

    @staticmethod
    def _build_tree(bus_ids: tuple[int, ...], bus_pos: dict[int, int], raw_edges: list[dict[str, object]], root_bus_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
        adjacency: dict[int, list[tuple[int, int]]] = {int(bus_id): [] for bus_id in bus_ids}
        for edge_idx, record in enumerate(raw_edges):
            u = int(record['u'])
            v = int(record['v'])
            adjacency[u].append((v, edge_idx))
            adjacency[v].append((u, edge_idx))
        visited = {int(root_bus_id)}
        queue = [int(root_bus_id)]
        branch_parent_pos: list[int] = []
        branch_child_pos: list[int] = []
        branch_r_pu: list[float] = []
        branch_x_pu: list[float] = []
        branch_l_max_pu: list[float] = []
        branch_is_trafo: list[bool] = []
        incoming_branch_by_bus = np.full((len(bus_ids),), -1, dtype=np.int32)
        outgoing_lists: list[list[int]] = [[] for _ in bus_ids]
        while queue:
            parent_bus = queue.pop(0)
            parent_pos = bus_pos[parent_bus]
            for neighbor_bus, raw_edge_idx in adjacency[parent_bus]:
                if int(neighbor_bus) in visited:
                    continue
                visited.add(int(neighbor_bus))
                queue.append(int(neighbor_bus))
                record = raw_edges[int(raw_edge_idx)]
                child_pos = bus_pos[int(neighbor_bus)]
                branch_idx = len(branch_parent_pos)
                branch_parent_pos.append(parent_pos)
                branch_child_pos.append(child_pos)
                branch_r_pu.append(float(record['r_pu']))
                branch_x_pu.append(float(record['x_pu']))
                branch_l_max_pu.append(float(record['l_max_pu']))
                branch_is_trafo.append(bool(record['is_trafo']))
                incoming_branch_by_bus[child_pos] = int(branch_idx)
                outgoing_lists[parent_pos].append(int(branch_idx))
        if len(visited) != len(bus_ids):
            missing = sorted(set(bus_ids) - visited)
            raise ValueError(f'Global SOCP-MPC requires a single connected tree. Unreachable buses: {missing}.')
        if len(branch_parent_pos) != len(bus_ids) - 1:
            raise ValueError(f'Global SOCP-MPC requires a tree topology; got {len(branch_parent_pos)} branches for {len(bus_ids)} buses.')
        return (np.asarray(branch_parent_pos, dtype=np.int32), np.asarray(branch_child_pos, dtype=np.int32), np.asarray(branch_r_pu, dtype=np.float32), np.asarray(branch_x_pu, dtype=np.float32), np.asarray(branch_l_max_pu, dtype=np.float32), np.asarray(branch_is_trafo, dtype=bool), incoming_branch_by_bus, tuple((tuple((edge_idx for edge_idx in outgoing)) for outgoing in outgoing_lists)))

@dataclass(frozen=True)
class ModelSize:
    num_vars: int
    num_binary_vars: int
    num_linear_constraints: int
    num_quadratic_constraints: int

@dataclass(frozen=True)
class FullHorizonProblemInput:
    wholesale_price_seq: np.ndarray
    import_price_seq: np.ndarray
    load_seq: np.ndarray
    pv_seq: np.ndarray
    soc_init: np.ndarray
    timestamps: tuple[str, ...]
    episode_offsets: np.ndarray
    episode_lengths: np.ndarray
    episode_indices: np.ndarray

    @property
    def horizon_steps(self) -> int:
        return int(self.import_price_seq.shape[0])

@dataclass(frozen=True)
class GurobiSolveConfig:
    time_limit_sec: float | None = None
    mip_gap: float = _MIP_GAP
    threads: int | None = None
    presolve: int | None = None
    cuts: int | None = None
    heuristics: float | None = None
    mip_focus: int | None = None

@dataclass
class MISOCPResult:
    status_code: int
    status_label: str
    has_solution: bool
    time_limit_feasible: bool
    solve_time_sec: float
    mip_gap: float
    best_bound: float
    objective_value: float
    agent_purchase_cost_eur: float
    agent_export_subsidy_eur: float
    agent_net_cost_eur: float
    feeder_purchase_cost_eur: float
    feeder_export_subsidy_eur: float
    feeder_net_cost_eur: float
    throughput_regularization_eur: float
    throughput_regularization_weight: float
    physical_tiebreaker_eur: float
    physical_tiebreaker_weight: float
    model_size: ModelSize
    debug_artifacts: dict[str, str]
    agent_net_grid_mw: np.ndarray | None
    agent_import_mw: np.ndarray | None
    agent_export_mw: np.ndarray | None
    battery_charge_mw: np.ndarray | None
    battery_discharge_mw: np.ndarray | None
    pv_curtail_mw: np.ndarray | None
    energy_mwh: np.ndarray | None
    branch_p_pu: np.ndarray | None
    branch_q_pu: np.ndarray | None
    branch_i2_pu: np.ndarray | None
    bus_v_sq: np.ndarray | None
    root_import_mw: np.ndarray | None
    root_export_mw: np.ndarray | None
    root_p_kw: np.ndarray | None
    root_q_kvar: np.ndarray | None
    bus_vm_pu: np.ndarray | None
    line_loading_pct: np.ndarray | None
    trafo_loading_pct: np.ndarray | None
    simultaneous_charge_discharge_kw: np.ndarray | None
    simultaneous_agent_steps: int
    simultaneous_step_ratio: float
    max_simultaneous_kw: float
    sol_count: int = 0
    node_count: float = float('nan')
    iter_count: float = float('nan')
    bar_iter_count: float = float('nan')
    sanity_warning: str = ''
    horizon_steps: int = 0
    solve_mode: str = 'single_window'
    episode_offsets: np.ndarray | None = None
    episode_lengths: np.ndarray | None = None
    chunk_summaries: list[dict[str, object]] | None = None
    no_retry_or_fallback_used: bool = True
    chunk_retry_count: int = 0
    stage1_primary_objective_eur: float = float('nan')
    stage2_primary_objective_eur: float = float('nan')
    stage2_objective_slack_eur: float = float('nan')
    stage2_branch_l_objective: float = float('nan')
    physics_refinement_mode: str = 'none'
    physics_refinement_status: str = 'not_enabled'
    physics_refinement_runtime_sec: float = 0.0
    floor_p95_soc_slack: float = float('nan')
    floor_mean_abs_solver_feeder_gap_kw: float = float('nan')
    floor_primary_objective_eur: float = float('nan')
    floor_primary_delta_signed_eur: float = float('nan')
    floor_primary_delta_positive_eur: float = float('nan')
    physics_refinement_slack_cap_eur: float = float('nan')
    initial_physics_refinement_slack_cap_eur: float = float('nan')
    returned_primary_objective_eur: float = float('nan')
    returned_primary_delta_abs_eur: float = float('nan')
    returned_primary_delta_pct: float = float('nan')
    floor_accepted_tier: int | None = None
    used_physics_refinement_tier: int | None = None
    total_tiers_configured: int = 0
    physics_refinement_attempt_count: int = 0
    physics_refinement_attempt_caps_eur: list[float] | None = None
    physics_refinement_cap_utilization: float = float('nan')
    branch_l_gap_ratio_to_floor: float = float('nan')
    returned_mean_abs_solver_feeder_gap_kw: float = float('nan')
    returned_max_solver_feeder_gap_kw: float = float('nan')
    returned_mean_abs_export_gap_ratio: float = float('nan')
    high_budget_refinement_warn: bool = False
    returned_solution_source: str = 'stage1'
    formulation_tightening_required: bool = False
    negative_floor_delta_warn: bool = False
    refinement_status_counts: dict[str, int] | None = None

    @property
    def first_step_battery_power_kw(self) -> np.ndarray | None:
        if self.battery_charge_mw is None or self.battery_discharge_mw is None:
            return None
        return ((self.battery_charge_mw[:, 0] - self.battery_discharge_mw[:, 0]) * 1000.0).astype(np.float32)

    @property
    def first_step_battery_charge_kw(self) -> np.ndarray | None:
        if self.battery_charge_mw is None:
            return None
        return (self.battery_charge_mw[:, 0] * 1000.0).astype(np.float32)

    @property
    def first_step_battery_discharge_kw(self) -> np.ndarray | None:
        if self.battery_discharge_mw is None:
            return None
        return (self.battery_discharge_mw[:, 0] * 1000.0).astype(np.float32)

    @property
    def first_step_pv_curtail_kw(self) -> np.ndarray | None:
        if self.pv_curtail_mw is None:
            return None
        return (self.pv_curtail_mw[:, 0] * 1000.0).astype(np.float32)

    @property
    def first_step_vm_pu(self) -> np.ndarray | None:
        if self.bus_vm_pu is None:
            return None
        return np.asarray(self.bus_vm_pu[:, 0], dtype=np.float32)

    @property
    def first_step_line_loading_pct(self) -> np.ndarray | None:
        if self.line_loading_pct is None:
            return None
        return np.asarray(self.line_loading_pct[:, 0], dtype=np.float32)

    @property
    def first_step_trafo_loading_pct(self) -> np.ndarray | None:
        if self.trafo_loading_pct is None:
            return None
        return np.asarray(self.trafo_loading_pct[:, 0], dtype=np.float32)

    @property
    def first_step_root_import_kw(self) -> float:
        if self.root_import_mw is None:
            return float('nan')
        return float(self.root_import_mw[0] * 1000.0)

    @property
    def first_step_root_export_kw(self) -> float:
        if self.root_export_mw is None:
            return float('nan')
        return float(self.root_export_mw[0] * 1000.0)

    @property
    def first_step_root_p_kw(self) -> float:
        if self.root_p_kw is None:
            return float('nan')
        return float(self.root_p_kw[0])

    @property
    def first_step_root_q_kvar(self) -> float:
        if self.root_q_kvar is None:
            return float('nan')
        return float(self.root_q_kvar[0])

    @property
    def first_step_simultaneous_kw_total(self) -> float:
        if self.simultaneous_charge_discharge_kw is None:
            return 0.0
        return float(np.sum(self.simultaneous_charge_discharge_kw[:, 0]))

    @property
    def first_step_simultaneous_agent_count(self) -> int:
        if self.simultaneous_charge_discharge_kw is None:
            return 0
        return int(np.sum(self.simultaneous_charge_discharge_kw[:, 0] > 0.0))

@dataclass
class SequenceModelBundle:
    model: Any
    import_price_seq: np.ndarray
    load_seq: np.ndarray
    pv_seq: np.ndarray
    soc_init: np.ndarray
    subsidy: float
    horizon_steps: int
    p_charge: Any
    p_discharge: Any
    pv_curtail: Any
    agent_abs_grid: Any
    energy: Any
    branch_p: Any
    branch_q: Any
    branch_l: Any
    bus_v: Any
    p_import: Any
    p_export: Any
    u_grid: Any
    primary_objective_expr: Any
    branch_l_objective_expr: Any

    @property
    def first_step_battery_power_kw(self) -> np.ndarray | None:
        if self.battery_charge_mw is None or self.battery_discharge_mw is None:
            return None
        return ((self.battery_charge_mw[:, 0] - self.battery_discharge_mw[:, 0]) * 1000.0).astype(np.float32)

    @property
    def first_step_battery_charge_kw(self) -> np.ndarray | None:
        if self.battery_charge_mw is None:
            return None
        return (self.battery_charge_mw[:, 0] * 1000.0).astype(np.float32)

    @property
    def first_step_battery_discharge_kw(self) -> np.ndarray | None:
        if self.battery_discharge_mw is None:
            return None
        return (self.battery_discharge_mw[:, 0] * 1000.0).astype(np.float32)

    @property
    def first_step_pv_curtail_kw(self) -> np.ndarray | None:
        if self.pv_curtail_mw is None:
            return None
        return (self.pv_curtail_mw[:, 0] * 1000.0).astype(np.float32)

    @property
    def first_step_vm_pu(self) -> np.ndarray | None:
        if self.bus_vm_pu is None:
            return None
        return np.asarray(self.bus_vm_pu[:, 0], dtype=np.float32)

    @property
    def first_step_line_loading_pct(self) -> np.ndarray | None:
        if self.line_loading_pct is None:
            return None
        return np.asarray(self.line_loading_pct[:, 0], dtype=np.float32)

    @property
    def first_step_trafo_loading_pct(self) -> np.ndarray | None:
        if self.trafo_loading_pct is None:
            return None
        return np.asarray(self.trafo_loading_pct[:, 0], dtype=np.float32)

    @property
    def first_step_root_import_kw(self) -> float:
        if self.root_import_mw is None:
            return float('nan')
        return float(self.root_import_mw[0] * 1000.0)

    @property
    def first_step_root_export_kw(self) -> float:
        if self.root_export_mw is None:
            return float('nan')
        return float(self.root_export_mw[0] * 1000.0)

    @property
    def first_step_root_p_kw(self) -> float:
        if self.root_p_kw is None:
            return float('nan')
        return float(self.root_p_kw[0])

    @property
    def first_step_root_q_kvar(self) -> float:
        if self.root_q_kvar is None:
            return float('nan')
        return float(self.root_q_kvar[0])

    @property
    def first_step_simultaneous_kw_total(self) -> float:
        if self.simultaneous_charge_discharge_kw is None:
            return 0.0
        return float(np.sum(self.simultaneous_charge_discharge_kw[:, 0]))

    @property
    def first_step_simultaneous_agent_count(self) -> int:
        if self.simultaneous_charge_discharge_kw is None:
            return 0
        return int(np.sum(self.simultaneous_charge_discharge_kw[:, 0] > 0.0))

def _default_threads() -> int:
    return int(max(1, min(8, os.cpu_count() or 1)))

def default_primary_solve_config() -> GurobiSolveConfig:
    return GurobiSolveConfig(time_limit_sec=_ADAPTIVE_PRIMARY_TIME_LIMIT_SEC, mip_gap=_ADAPTIVE_TARGET_MIP_GAP, threads=_default_threads(), presolve=2, cuts=2, heuristics=0.1, mip_focus=1)

def default_retry_solve_config(primary: GurobiSolveConfig | None=None) -> GurobiSolveConfig:
    base = primary or default_primary_solve_config()
    return GurobiSolveConfig(time_limit_sec=_ADAPTIVE_RETRY_TIME_LIMIT_SEC, mip_gap=float(base.mip_gap), threads=base.threads, presolve=base.presolve, cuts=1, heuristics=0.2, mip_focus=1 if base.mip_focus is None else int(base.mip_focus))

class GlobalMISOCPProblem:
    uses_nonconvex = False

    def __init__(self, *, network: NetworkModel, n_agents: int, horizon: int, dt_hours: float, efficiency: float, capacity_mwh: np.ndarray, p_max_mw: np.ndarray, soc_min: float, soc_max: float, soc_target: float, v_min_sq: float, v_max_sq: float, export_subsidy_default: float, import_price_markup_eur_per_kwh: float=0.0, throughput_regularization_eur_per_kwh: float=_THROUGHPUT_REGULARIZATION_EUR_PER_KWH, branch_current_tiebreaker_eur_per_pu_step: float=0.0, physics_refinement_mode: str='none', physics_refinement_slack_ratio: float=0.02, physics_refinement_slack_abs_floor_eur: float=2.0, physics_refinement_slack_ratio_schedule: Sequence[float] | None=None, physics_refinement_slack_abs_floor_schedule_eur: Sequence[float] | None=None, physics_refinement_enable_aggressive_third_tier: bool=False, physics_refinement_aggressive_third_tier_ratio: float=0.1, physics_refinement_aggressive_third_tier_abs_floor_eur: float=10.0, physics_refinement_cap_utilization_trigger: float=0.95, physics_refinement_branch_l_gap_ratio_trigger: float=0.01, physics_refinement_time_limit_sec: float=20.0, physics_refinement_total_time_limit_sec: float=40.0, physics_refinement_target_mean_solver_gap_kw: float=3.0, physics_refinement_target_max_solver_gap_kw: float=15.0, physics_refinement_target_export_gap_ratio: float=0.05, physics_refinement_use_full_start: bool=True, simultaneous_threshold_ratio: float=_SIMULTANEOUS_THRESHOLD_RATIO, debug_dir: str | Path | None=None) -> None:
        self.network = network
        self.n_agents = int(n_agents)
        self.n_buses = int(len(network.bus_ids))
        self.n_branches = int(network.branch_parent_pos.shape[0])
        self.horizon = int(horizon)
        self.dt_hours = float(dt_hours)
        self.efficiency = float(efficiency)
        self.capacity_mwh = np.asarray(capacity_mwh, dtype=np.float32)
        self.p_max_mw = np.asarray(p_max_mw, dtype=np.float32)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.soc_target = float(soc_target)
        self.root_vm_sq = float(max(network.root_vm_pu * network.root_vm_pu, _ROOT_VM_EPS))
        self.v_min_sq = float(v_min_sq)
        self.v_max_sq = float(v_max_sq)
        self.export_subsidy_default = float(export_subsidy_default)
        self.import_price_markup_eur_per_kwh = float(import_price_markup_eur_per_kwh)
        self.throughput_regularization_eur_per_kwh = float(throughput_regularization_eur_per_kwh)
        self.branch_current_tiebreaker_eur_per_pu_step = float(branch_current_tiebreaker_eur_per_pu_step)
        self.physics_refinement_mode = str(physics_refinement_mode)
        self.physics_refinement_slack_ratio = float(physics_refinement_slack_ratio)
        self.physics_refinement_slack_abs_floor_eur = float(physics_refinement_slack_abs_floor_eur)
        ratio_schedule = [float(value) for value in list(physics_refinement_slack_ratio_schedule)] if physics_refinement_slack_ratio_schedule is not None else [float(physics_refinement_slack_ratio)]
        abs_floor_schedule = [float(value) for value in list(physics_refinement_slack_abs_floor_schedule_eur)] if physics_refinement_slack_abs_floor_schedule_eur is not None else [float(physics_refinement_slack_abs_floor_eur)]
        if len(ratio_schedule) != len(abs_floor_schedule) or len(ratio_schedule) == 0:
            raise ValueError('physics refinement slack schedules must be non-empty and have matching lengths.')
        self.physics_refinement_slack_ratio_schedule = tuple((float(value) for value in ratio_schedule))
        self.physics_refinement_slack_abs_floor_schedule_eur = tuple((float(value) for value in abs_floor_schedule))
        self.physics_refinement_enable_aggressive_third_tier = bool(physics_refinement_enable_aggressive_third_tier)
        self.physics_refinement_aggressive_third_tier_ratio = float(physics_refinement_aggressive_third_tier_ratio)
        self.physics_refinement_aggressive_third_tier_abs_floor_eur = float(physics_refinement_aggressive_third_tier_abs_floor_eur)
        self.physics_refinement_cap_utilization_trigger = float(physics_refinement_cap_utilization_trigger)
        self.physics_refinement_branch_l_gap_ratio_trigger = float(physics_refinement_branch_l_gap_ratio_trigger)
        self.physics_refinement_time_limit_sec = float(physics_refinement_time_limit_sec)
        self.physics_refinement_total_time_limit_sec = float(physics_refinement_total_time_limit_sec)
        self.physics_refinement_target_mean_solver_gap_kw = float(physics_refinement_target_mean_solver_gap_kw)
        self.physics_refinement_target_max_solver_gap_kw = float(physics_refinement_target_max_solver_gap_kw)
        self.physics_refinement_target_export_gap_ratio = float(physics_refinement_target_export_gap_ratio)
        self.physics_refinement_use_full_start = bool(physics_refinement_use_full_start)
        self.simultaneous_threshold_ratio = float(simultaneous_threshold_ratio)
        self.trafo_limit_mva = float(self.network.s_base_mva * self.network.loading_limit_scale)
        self.debug_dir = Path(debug_dir) if debug_dir is not None else _DEFAULT_DEBUG_DIR

    @classmethod
    def from_env(cls, env: Any, cfg: Any, *, throughput_regularization_eur_per_kwh: float=_THROUGHPUT_REGULARIZATION_EUR_PER_KWH, simultaneous_threshold_ratio: float=_SIMULTANEOUS_THRESHOLD_RATIO, debug_dir: str | Path | None=None) -> 'GlobalMISOCPProblem':
        network = NetworkModel.from_cfg(cfg, agent_bus_ids=list(getattr(env._grid_core, 'agent_bus_ids', cfg.grid.agent_bus_ids)))
        return cls(network=network, n_agents=int(env.n), horizon=int(cfg.env.future_horizon) + 1, dt_hours=float(env.dt), efficiency=float(env.eff), capacity_mwh=np.asarray(env.agent_c_bat, dtype=np.float32) / 1000.0, p_max_mw=np.asarray(env.agent_p_max, dtype=np.float32) / 1000.0, soc_min=float(env.soc_min), soc_max=float(env.soc_max), soc_target=float(getattr(cfg.env, 'soc_target', env.soc_max)), v_min_sq=float(cfg.grid.v_min_pu * cfg.grid.v_min_pu), v_max_sq=float(cfg.grid.v_max_pu * cfg.grid.v_max_pu), export_subsidy_default=float(getattr(cfg.reward, 'export_subsidy_eur_per_kwh', 0.079)), import_price_markup_eur_per_kwh=float(get_import_price_markup(cfg)), throughput_regularization_eur_per_kwh=throughput_regularization_eur_per_kwh, branch_current_tiebreaker_eur_per_pu_step=float(getattr(getattr(cfg, 'mpc', None), 'branch_current_tiebreaker_eur_per_pu_step', 0.0)), physics_refinement_mode=str(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_mode', 'none')), physics_refinement_slack_ratio=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_slack_ratio', 0.02)), physics_refinement_slack_abs_floor_eur=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_slack_abs_floor_eur', 2.0)), physics_refinement_slack_ratio_schedule=list(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_slack_ratio_schedule', [0.02])), physics_refinement_slack_abs_floor_schedule_eur=list(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_slack_abs_floor_schedule_eur', [2.0])), physics_refinement_enable_aggressive_third_tier=bool(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_enable_aggressive_third_tier', False)), physics_refinement_aggressive_third_tier_ratio=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_aggressive_third_tier_ratio', 0.1)), physics_refinement_aggressive_third_tier_abs_floor_eur=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_aggressive_third_tier_abs_floor_eur', 10.0)), physics_refinement_cap_utilization_trigger=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_cap_utilization_trigger', 0.95)), physics_refinement_branch_l_gap_ratio_trigger=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_branch_l_gap_ratio_trigger', 0.01)), physics_refinement_time_limit_sec=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_time_limit_sec', 20.0)), physics_refinement_total_time_limit_sec=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_total_time_limit_sec', 40.0)), physics_refinement_target_mean_solver_gap_kw=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_target_mean_solver_gap_kw', 3.0)), physics_refinement_target_max_solver_gap_kw=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_target_max_solver_gap_kw', 15.0)), physics_refinement_target_export_gap_ratio=float(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_target_export_gap_ratio', 0.05)), physics_refinement_use_full_start=bool(getattr(getattr(cfg, 'mpc', None), 'physics_refinement_use_full_start', True)), simultaneous_threshold_ratio=simultaneous_threshold_ratio, debug_dir=debug_dir)

    def apply_import_price_markup(self, wholesale_price_seq: np.ndarray) -> np.ndarray:
        wholesale_price_seq = np.asarray(wholesale_price_seq, dtype=np.float32).reshape(-1)
        if abs(float(self.import_price_markup_eur_per_kwh)) <= _ROOT_VM_EPS:
            return wholesale_price_seq.astype(np.float32, copy=False)
        return derive_import_price_seq(wholesale_price_seq, markup_eur_per_kwh=float(self.import_price_markup_eur_per_kwh))

    def _resolve_solve_config(self, solve_config: GurobiSolveConfig | None, *, time_limit_sec: float | None=None) -> GurobiSolveConfig:
        if solve_config is None:
            return GurobiSolveConfig(time_limit_sec=time_limit_sec)
        if time_limit_sec is not None and solve_config.time_limit_sec is None:
            return replace(solve_config, time_limit_sec=float(time_limit_sec))
        return solve_config

    def _build_physics_refinement_solve_config(self, base: GurobiSolveConfig) -> GurobiSolveConfig:
        return GurobiSolveConfig(time_limit_sec=float(self.physics_refinement_time_limit_sec), mip_gap=float(base.mip_gap), threads=base.threads, presolve=base.presolve, cuts=base.cuts, heuristics=base.heuristics, mip_focus=base.mip_focus)

    def _primary_objective_from_result(self, result: MISOCPResult) -> float:
        return float(result.agent_net_cost_eur + result.throughput_regularization_eur)

    def _branch_l_objective_from_result(self, result: MISOCPResult) -> float:
        if result.branch_i2_pu is None:
            return float('nan')
        return float(np.sum(np.asarray(result.branch_i2_pu, dtype=np.float64)))

    def _configured_refinement_tiers(self) -> list[tuple[float, float]]:
        tiers = list(zip([float(value) for value in self.physics_refinement_slack_ratio_schedule], [float(value) for value in self.physics_refinement_slack_abs_floor_schedule_eur]))
        if self.physics_refinement_enable_aggressive_third_tier:
            tiers.append((float(self.physics_refinement_aggressive_third_tier_ratio), float(self.physics_refinement_aggressive_third_tier_abs_floor_eur)))
        return tiers

    def _refinement_slack_cap_eur(self, stage1_primary_objective_eur: float) -> float:
        return float(max(self.physics_refinement_slack_ratio * max(abs(float(stage1_primary_objective_eur)), 1.0), self.physics_refinement_slack_abs_floor_eur))

    def _refinement_slack_cap_schedule_eur(self, stage1_primary_objective_eur: float) -> list[float]:
        return [float(max(float(ratio) * max(abs(float(stage1_primary_objective_eur)), 1.0), float(abs_floor))) for ratio, abs_floor in self._configured_refinement_tiers()]

    def _negative_floor_delta_warn(self, *, stage1_primary_objective_eur: float, floor_primary_delta_signed_eur: float) -> bool:
        threshold = max(0.001 * max(abs(float(stage1_primary_objective_eur)), 1.0), 0.1)
        return bool(float(floor_primary_delta_signed_eur) < -float(threshold))

    @staticmethod
    def _single_refinement_status_counts(status: str) -> dict[str, int]:
        return {str(status): 1}

    def _high_budget_refinement_warn(self, *, returned_primary_delta_eur: float, returned_primary_delta_pct: float) -> bool:
        return bool(float(returned_primary_delta_pct) > 5.0 and float(returned_primary_delta_eur) > 2.0 * float(self.physics_refinement_slack_abs_floor_schedule_eur[0]))

    def _soc_relaxation_summary_from_result(self, result: MISOCPResult) -> tuple[float, float]:
        if result.branch_p_pu is None or result.branch_q_pu is None or result.branch_i2_pu is None or (result.bus_v_sq is None):
            return (float('nan'), float('nan'))
        branch_p_pu = np.asarray(result.branch_p_pu, dtype=np.float64)
        branch_q_pu = np.asarray(result.branch_q_pu, dtype=np.float64)
        branch_i2_pu = np.asarray(result.branch_i2_pu, dtype=np.float64)
        bus_v_sq = np.asarray(result.bus_v_sq, dtype=np.float64)
        parent_pos = np.asarray(self.network.branch_parent_pos, dtype=np.int32).reshape(-1)
        v_parent_sq = np.asarray(bus_v_sq[parent_pos, :], dtype=np.float64)
        soc_slack = branch_i2_pu - (branch_p_pu * branch_p_pu + branch_q_pu * branch_q_pu) / np.maximum(v_parent_sq, 1e-06)
        finite_slack = soc_slack[np.isfinite(soc_slack)]
        if finite_slack.size == 0:
            return (float('nan'), float('nan'))
        return (float(np.percentile(finite_slack, 95.0)), float(np.mean(finite_slack)))

    def _solver_feeder_gap_summary_from_result(self, result: MISOCPResult, *, load_seq: np.ndarray, pv_seq: np.ndarray) -> tuple[float, float]:
        if result.root_p_kw is None or result.battery_charge_mw is None or result.battery_discharge_mw is None or (result.pv_curtail_mw is None):
            return (float('nan'), float('nan'))
        load_kw = np.asarray(load_seq, dtype=np.float64)
        pv_kw = np.asarray(pv_seq, dtype=np.float64)
        battery_charge_kw = np.asarray(result.battery_charge_mw, dtype=np.float64) * 1000.0
        battery_discharge_kw = np.asarray(result.battery_discharge_mw, dtype=np.float64) * 1000.0
        pv_curtail_kw = np.asarray(result.pv_curtail_mw, dtype=np.float64) * 1000.0
        agent_post_action_net_load_kw = np.sum(load_kw - pv_kw + pv_curtail_kw + battery_charge_kw - battery_discharge_kw, axis=0)
        fixed_active_kw = np.asarray(self.network.p_base_mw, dtype=np.float64).reshape(-1) * 1000.0
        fixed_load_kw = float(np.clip(fixed_active_kw, 0.0, None).sum())
        fixed_generation_kw = float(np.clip(-fixed_active_kw, 0.0, None).sum())
        feeder_post_action_net_load_kw = agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw
        solver_feeder_gap_kw = np.asarray(result.root_p_kw, dtype=np.float64).reshape(-1) - feeder_post_action_net_load_kw
        finite_gap = solver_feeder_gap_kw[np.isfinite(solver_feeder_gap_kw)]
        if finite_gap.size == 0:
            return (float('nan'), float('nan'))
        return (float(np.mean(np.abs(finite_gap))), float(np.max(np.abs(finite_gap))))

    def _export_gap_ratio_summary_from_result(self, result: MISOCPResult, *, load_seq: np.ndarray, pv_seq: np.ndarray) -> tuple[float, int]:
        if result.root_p_kw is None or result.battery_charge_mw is None or result.battery_discharge_mw is None or (result.pv_curtail_mw is None):
            return (float('nan'), 0)
        load_kw = np.asarray(load_seq, dtype=np.float64)
        pv_kw = np.asarray(pv_seq, dtype=np.float64)
        battery_charge_kw = np.asarray(result.battery_charge_mw, dtype=np.float64) * 1000.0
        battery_discharge_kw = np.asarray(result.battery_discharge_mw, dtype=np.float64) * 1000.0
        pv_curtail_kw = np.asarray(result.pv_curtail_mw, dtype=np.float64) * 1000.0
        agent_post_action_net_load_kw = np.sum(load_kw - pv_kw + pv_curtail_kw + battery_charge_kw - battery_discharge_kw, axis=0)
        fixed_active_kw = np.asarray(self.network.p_base_mw, dtype=np.float64).reshape(-1) * 1000.0
        feeder_post_action_net_load_kw = agent_post_action_net_load_kw + float(np.clip(fixed_active_kw, 0.0, None).sum()) - float(np.clip(-fixed_active_kw, 0.0, None).sum())
        root_net_exchange_kw = np.asarray(result.root_p_kw, dtype=np.float64).reshape(-1)
        export_mask = np.isfinite(root_net_exchange_kw) & (root_net_exchange_kw < -1.0)
        if not np.any(export_mask):
            return (0.0, 0)
        solver_gap_kw = root_net_exchange_kw - feeder_post_action_net_load_kw
        export_ratio = np.abs(solver_gap_kw[export_mask]) / np.maximum(np.abs(root_net_exchange_kw[export_mask]), 1.0)
        finite_ratio = export_ratio[np.isfinite(export_ratio)]
        if finite_ratio.size == 0:
            return (float('nan'), 0)
        return (float(np.mean(finite_ratio)), int(finite_ratio.size))

    def _refinement_metrics_from_result(self, result: MISOCPResult, *, load_seq: np.ndarray, pv_seq: np.ndarray, floor_branch_l_objective: float) -> dict[str, float]:
        mean_gap, max_gap = self._solver_feeder_gap_summary_from_result(result, load_seq=load_seq, pv_seq=pv_seq)
        export_gap_ratio, export_gap_steps = self._export_gap_ratio_summary_from_result(result, load_seq=load_seq, pv_seq=pv_seq)
        branch_l_objective = self._branch_l_objective_from_result(result)
        branch_l_gap_ratio = float('nan')
        if np.isfinite(branch_l_objective) and np.isfinite(floor_branch_l_objective):
            branch_l_gap_ratio = float((branch_l_objective - float(floor_branch_l_objective)) / max(abs(float(floor_branch_l_objective)), 1e-06))
        return {'mean_gap_kw': float(mean_gap), 'max_gap_kw': float(max_gap), 'mean_export_gap_ratio': float(export_gap_ratio), 'export_gap_step_count': int(export_gap_steps), 'branch_l_gap_ratio': float(branch_l_gap_ratio), 'branch_l_objective': float(branch_l_objective)}

    def _refinement_meets_targets(self, metrics: dict[str, float]) -> bool:
        mean_gap = float(metrics.get('mean_gap_kw', float('nan')))
        max_gap = float(metrics.get('max_gap_kw', float('nan')))
        export_gap_ratio = float(metrics.get('mean_export_gap_ratio', float('nan')))
        branch_l_gap_ratio = float(metrics.get('branch_l_gap_ratio', float('nan')))
        export_gap_ok = True if not np.isfinite(export_gap_ratio) else bool(export_gap_ratio < float(self.physics_refinement_target_export_gap_ratio))
        return bool(np.isfinite(mean_gap) and np.isfinite(max_gap) and np.isfinite(branch_l_gap_ratio) and (mean_gap < float(self.physics_refinement_target_mean_solver_gap_kw)) and (max_gap < float(self.physics_refinement_target_max_solver_gap_kw)) and export_gap_ok and (branch_l_gap_ratio < 0.02))

    def _window_slices(self, full_input: FullHorizonProblemInput, *, chunk_steps: int) -> list[tuple[int, int]]:
        resolved_chunk_steps = int(chunk_steps)
        if resolved_chunk_steps <= 0:
            raise ValueError('chunk_steps must be positive.')
        horizon_steps = int(full_input.horizon_steps)
        windows: list[tuple[int, int]] = []
        cursor = 0
        while cursor < horizon_steps:
            end = min(cursor + resolved_chunk_steps, horizon_steps)
            windows.append((int(cursor), int(end)))
            cursor = int(end)
        return windows

    def _make_chunk_summary(self, result: MISOCPResult, *, chunk_idx: int, start_step: int, end_step: int, attempt_used: str) -> dict[str, object]:
        return {'chunk_idx': int(chunk_idx), 'start_step': int(start_step), 'end_step': int(end_step), 'window_steps': int(end_step - start_step), 'status_label': str(result.status_label), 'has_solution': bool(result.has_solution), 'solve_time_sec': float(result.solve_time_sec), 'mip_gap': float(result.mip_gap), 'objective_value': float(result.objective_value), 'attempt_used': str(attempt_used), 'physics_refinement_status': str(getattr(result, 'physics_refinement_status', 'not_enabled')), 'stage2_runtime_sec': float(getattr(result, 'physics_refinement_runtime_sec', 0.0)), 'stage2_branch_l_objective': float(getattr(result, 'stage2_branch_l_objective', float('nan')))}

    def _build_sequence_model(self, *, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc_init: np.ndarray, export_subsidy: float | None, enforce_terminal_soc: bool, solve_mode: str, primary_objective_upper_bound: float | None=None, include_branch_current_tiebreaker_in_primary: bool=True) -> SequenceModelBundle:
        if gp is None or GRB is None:
            raise RuntimeError(f'{_GUROBI_ERROR_PREFIX}: gurobipy import failed')
        import_price_seq = np.asarray(import_price_seq, dtype=np.float32).reshape(-1)
        load_seq = np.asarray(load_seq, dtype=np.float32)
        pv_seq = np.asarray(pv_seq, dtype=np.float32)
        soc_init = np.asarray(soc_init, dtype=np.float32).reshape(-1)
        horizon_steps = int(import_price_seq.size)
        if load_seq.shape != (self.n_agents, horizon_steps):
            raise ValueError(f'Expected load_seq shape {(self.n_agents, horizon_steps)}, got {load_seq.shape}.')
        if pv_seq.shape != (self.n_agents, horizon_steps):
            raise ValueError(f'Expected pv_seq shape {(self.n_agents, horizon_steps)}, got {pv_seq.shape}.')
        if soc_init.shape != (self.n_agents,):
            raise ValueError(f'Expected soc_init shape {(self.n_agents,)}, got {soc_init.shape}.')
        subsidy = float(self.export_subsidy_default if export_subsidy is None else export_subsidy)
        if float(np.min(import_price_seq)) <= subsidy:
            raise ValueError(f'Agent-only MISOCP economics requires import prices to stay strictly above the export subsidy for the current absolute-value reformulation, got min(import_price_seq)={float(np.min(import_price_seq)):.6f} and subsidy={subsidy:.6f}.')
        model = gp.Model(f'global_misocp_{solve_mode}')
        energy_min_mwh = (self.soc_min * self.capacity_mwh).astype(np.float64)
        energy_max_mwh = (self.soc_max * self.capacity_mwh).astype(np.float64)
        energy_target_mwh = (self.soc_target * self.capacity_mwh).astype(np.float64)
        soc0_mwh = (soc_init * self.capacity_mwh).astype(np.float64)
        root_pos = int(self.network.bus_pos[self.network.root_bus_id])
        scale_sq = float(self.network.loading_limit_scale * self.network.loading_limit_scale)
        p_charge = model.addVars(self.n_agents, horizon_steps, lb=0.0, name='p_charge_mw')
        p_discharge = model.addVars(self.n_agents, horizon_steps, lb=0.0, name='p_discharge_mw')
        pv_curtail = model.addVars(self.n_agents, horizon_steps, lb=0.0, name='pv_curtail_mw')
        agent_abs_grid = model.addVars(self.n_agents, horizon_steps, lb=0.0, name='agent_abs_grid_mw')
        energy = model.addVars(self.n_agents, horizon_steps + 1, lb=0.0, name='energy_mwh')
        branch_p = model.addVars(self.n_branches, horizon_steps, lb=-GRB.INFINITY, name='branch_p_pu')
        branch_q = model.addVars(self.n_branches, horizon_steps, lb=-GRB.INFINITY, name='branch_q_pu')
        branch_l = model.addVars(self.n_branches, horizon_steps, lb=0.0, name='branch_i2_pu')
        bus_v = model.addVars(self.n_buses, horizon_steps, lb=self.v_min_sq, ub=self.v_max_sq, name='bus_v_sq')
        p_import = model.addVars(horizon_steps, lb=0.0, ub=self.trafo_limit_mva, name='p_import_mw')
        p_export = model.addVars(horizon_steps, lb=0.0, ub=self.trafo_limit_mva, name='p_export_mw')
        u_grid = model.addVars(horizon_steps, vtype=GRB.BINARY, name='u_grid')
        objective_terms = []
        for step_idx in range(horizon_steps):
            price_value = float(import_price_seq[step_idx])
            objective_terms.append(1000.0 * self.dt_hours * self.throughput_regularization_eur_per_kwh * gp.quicksum((p_charge[agent_idx, step_idx] + p_discharge[agent_idx, step_idx] for agent_idx in range(self.n_agents))))
            for agent_idx in range(self.n_agents):
                agent_net_grid_expr = float(load_seq[agent_idx, step_idx] - pv_seq[agent_idx, step_idx]) / 1000.0 + p_charge[agent_idx, step_idx] - p_discharge[agent_idx, step_idx] + pv_curtail[agent_idx, step_idx]
                model.addConstr(agent_abs_grid[agent_idx, step_idx] >= agent_net_grid_expr, name=f'agent_abs_grid_lb_pos[{agent_idx},{step_idx}]')
                model.addConstr(agent_abs_grid[agent_idx, step_idx] >= -agent_net_grid_expr, name=f'agent_abs_grid_lb_neg[{agent_idx},{step_idx}]')
                objective_terms.append(1000.0 * self.dt_hours * 0.5 * (price_value - subsidy) * agent_abs_grid[agent_idx, step_idx])
                objective_terms.append(1000.0 * self.dt_hours * 0.5 * (price_value + subsidy) * agent_net_grid_expr)
            model.addConstr(p_import[step_idx] <= self.trafo_limit_mva * u_grid[step_idx], name=f'grid_import_gate[{step_idx}]')
            model.addConstr(p_export[step_idx] <= self.trafo_limit_mva * (1.0 - u_grid[step_idx]), name=f'grid_export_gate[{step_idx}]')
            model.addConstr(bus_v[root_pos, step_idx] == self.root_vm_sq, name=f'root_v[{step_idx}]')
        primary_objective_expr = gp.quicksum(objective_terms)
        branch_l_objective_expr = gp.quicksum((branch_l[branch_idx, step_idx] for branch_idx in range(self.n_branches) for step_idx in range(horizon_steps)))
        if include_branch_current_tiebreaker_in_primary and abs(float(self.branch_current_tiebreaker_eur_per_pu_step)) > _ROOT_VM_EPS:
            primary_objective_expr += self.dt_hours * self.branch_current_tiebreaker_eur_per_pu_step * branch_l_objective_expr
        for agent_idx in range(self.n_agents):
            model.addConstr(energy[agent_idx, 0] == float(soc0_mwh[agent_idx]), name=f'energy_init[{agent_idx}]')
            for step_idx in range(horizon_steps + 1):
                model.addConstr(energy[agent_idx, step_idx] >= float(energy_min_mwh[agent_idx]), name=f'energy_lb[{agent_idx},{step_idx}]')
                model.addConstr(energy[agent_idx, step_idx] <= float(energy_max_mwh[agent_idx]), name=f'energy_ub[{agent_idx},{step_idx}]')
            if enforce_terminal_soc:
                model.addConstr(energy[agent_idx, horizon_steps] >= float(energy_target_mwh[agent_idx]), name=f'terminal_soc[{agent_idx}]')
            for step_idx in range(horizon_steps):
                model.addConstr(p_charge[agent_idx, step_idx] <= float(self.p_max_mw[agent_idx]), name=f'charge_ub[{agent_idx},{step_idx}]')
                model.addConstr(p_discharge[agent_idx, step_idx] <= float(self.p_max_mw[agent_idx]), name=f'discharge_ub[{agent_idx},{step_idx}]')
                model.addConstr(pv_curtail[agent_idx, step_idx] <= float(max(pv_seq[agent_idx, step_idx], 0.0) / 1000.0), name=f'pv_curtail_ub[{agent_idx},{step_idx}]')
                model.addConstr(energy[agent_idx, step_idx + 1] == energy[agent_idx, step_idx] + self.efficiency * p_charge[agent_idx, step_idx] * self.dt_hours - p_discharge[agent_idx, step_idx] / max(self.efficiency, _ROOT_VM_EPS) * self.dt_hours, name=f'energy_balance[{agent_idx},{step_idx}]')
        for step_idx in range(horizon_steps):
            for branch_idx in range(self.n_branches):
                parent_pos = int(self.network.branch_parent_pos[branch_idx])
                child_pos = int(self.network.branch_child_pos[branch_idx])
                outgoing = tuple((int(value) for value in self.network.outgoing_branches_by_bus[child_pos]))
                downstream_p = gp.quicksum((branch_p[out_idx, step_idx] for out_idx in outgoing))
                downstream_q = gp.quicksum((branch_q[out_idx, step_idx] for out_idx in outgoing))
                p_const_pu = float(self.network.p_base_mw[child_pos] / self.network.s_base_mva)
                for agent_idx in range(self.n_agents):
                    if int(self.network.agent_bus_positions[agent_idx]) != child_pos:
                        continue
                    p_const_pu += float(load_seq[agent_idx, step_idx] - pv_seq[agent_idx, step_idx]) / (1000.0 * self.network.s_base_mva)
                q_const_pu = float(self.network.q_base_mvar[child_pos] / self.network.s_base_mva)
                agent_expr = gp.quicksum(((p_charge[agent_idx, step_idx] - p_discharge[agent_idx, step_idx] + pv_curtail[agent_idx, step_idx]) / self.network.s_base_mva for agent_idx in range(self.n_agents) if int(self.network.agent_bus_positions[agent_idx]) == child_pos))
                r_pu = float(self.network.branch_r_pu[branch_idx])
                x_pu = float(self.network.branch_x_pu[branch_idx])
                model.addConstr(branch_p[branch_idx, step_idx] == downstream_p + branch_l[branch_idx, step_idx] * r_pu + agent_expr + p_const_pu, name=f'p_balance[{branch_idx},{step_idx}]')
                model.addConstr(branch_q[branch_idx, step_idx] == downstream_q + branch_l[branch_idx, step_idx] * x_pu + q_const_pu, name=f'q_balance[{branch_idx},{step_idx}]')
                model.addConstr(bus_v[child_pos, step_idx] == bus_v[parent_pos, step_idx] - 2.0 * (r_pu * branch_p[branch_idx, step_idx] + x_pu * branch_q[branch_idx, step_idx]) + (r_pu * r_pu + x_pu * x_pu) * branch_l[branch_idx, step_idx], name=f'voltage_drop[{branch_idx},{step_idx}]')
                cone_axis = bus_v[parent_pos, step_idx] + branch_l[branch_idx, step_idx]
                cone_residual = bus_v[parent_pos, step_idx] - branch_l[branch_idx, step_idx]
                model.addQConstr(cone_axis * cone_axis >= 4.0 * branch_p[branch_idx, step_idx] * branch_p[branch_idx, step_idx] + 4.0 * branch_q[branch_idx, step_idx] * branch_q[branch_idx, step_idx] + cone_residual * cone_residual, name=f'flow_soc[{branch_idx},{step_idx}]')
                if bool(self.network.branch_is_trafo[branch_idx]):
                    model.addConstr(branch_l[branch_idx, step_idx] <= scale_sq / max(self.v_min_sq, _ROOT_VM_EPS), name=f'trafo_i2_limit[{branch_idx},{step_idx}]')
                else:
                    l_max_pu = float(self.network.branch_l_max_pu[branch_idx])
                    model.addConstr(branch_l[branch_idx, step_idx] <= l_max_pu * scale_sq, name=f'line_limit[{branch_idx},{step_idx}]')
            root_p_pu = gp.quicksum((branch_p[int(idx), step_idx] for idx in self.network.root_outgoing_branches.tolist()))
            root_q_pu = gp.quicksum((branch_q[int(idx), step_idx] for idx in self.network.root_outgoing_branches.tolist()))
            model.addConstr(self.network.s_base_mva * root_p_pu == p_import[step_idx] - p_export[step_idx], name=f'root_active_split[{step_idx}]')
            model.addQConstr(root_p_pu * root_p_pu + root_q_pu * root_q_pu <= scale_sq, name=f'trafo_limit[{step_idx}]')
        if primary_objective_upper_bound is not None:
            model.addConstr(primary_objective_expr <= float(primary_objective_upper_bound), name='primary_objective_upper_bound')
        return SequenceModelBundle(model=model, import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, subsidy=subsidy, horizon_steps=horizon_steps, p_charge=p_charge, p_discharge=p_discharge, pv_curtail=pv_curtail, agent_abs_grid=agent_abs_grid, energy=energy, branch_p=branch_p, branch_q=branch_q, branch_l=branch_l, bus_v=bus_v, p_import=p_import, p_export=p_export, u_grid=u_grid, primary_objective_expr=primary_objective_expr, branch_l_objective_expr=branch_l_objective_expr)

    def _run_window_with_retry(self, *, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc_init: np.ndarray, export_subsidy: float | None, verbose: bool, export_debug: bool, debug_tag: str | None, enforce_terminal_soc: bool, solve_mode: str, primary_solve_config: GurobiSolveConfig, retry_solve_config: GurobiSolveConfig | None, partial_mip_start: dict[str, np.ndarray] | None=None) -> tuple[MISOCPResult, MISOCPResult, MISOCPResult | None, str]:
        primary_result = self._solve_sequences(import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, solve_config=primary_solve_config, export_debug=export_debug, debug_tag=debug_tag, expected_horizon=None, enforce_terminal_soc=enforce_terminal_soc, solve_mode=solve_mode, partial_mip_start=partial_mip_start, include_branch_current_tiebreaker_in_primary=bool(self.physics_refinement_mode == 'none'))
        if primary_result.has_solution or retry_solve_config is None:
            refined_primary = self._maybe_apply_physics_refinement(base_result=primary_result, import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, export_debug=export_debug, debug_tag=debug_tag, enforce_terminal_soc=enforce_terminal_soc, solve_mode=solve_mode, primary_solve_config=primary_solve_config)
            return (refined_primary, refined_primary, None, 'primary')
        retry_result = self._solve_sequences(import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, solve_config=retry_solve_config, export_debug=export_debug, debug_tag=None if debug_tag is None else f'{debug_tag}_retry', expected_horizon=None, enforce_terminal_soc=enforce_terminal_soc, solve_mode=solve_mode, partial_mip_start=partial_mip_start, include_branch_current_tiebreaker_in_primary=bool(self.physics_refinement_mode == 'none'))
        refined_retry = self._maybe_apply_physics_refinement(base_result=retry_result, import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, export_debug=export_debug, debug_tag=None if debug_tag is None else f'{debug_tag}_retry', enforce_terminal_soc=enforce_terminal_soc, solve_mode=solve_mode, primary_solve_config=retry_solve_config)
        return (refined_retry, primary_result, refined_retry, 'retry')

    def _maybe_apply_physics_refinement(self, *, base_result: MISOCPResult, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc_init: np.ndarray, export_subsidy: float | None, verbose: bool, export_debug: bool, debug_tag: str | None, enforce_terminal_soc: bool, solve_mode: str, primary_solve_config: GurobiSolveConfig) -> MISOCPResult:
        stage1_primary = self._primary_objective_from_result(base_result) if base_result.has_solution else float('nan')
        returned_primary_stage1 = float(stage1_primary) if np.isfinite(float(stage1_primary)) else float('nan')
        tier_caps = self._refinement_slack_cap_schedule_eur(stage1_primary) if base_result.has_solution else []
        total_tiers_configured = int(len(tier_caps))
        initial_slack_cap_eur = float(tier_caps[0]) if tier_caps else float('nan')
        result_with_defaults = replace(base_result, stage1_primary_objective_eur=stage1_primary, stage2_primary_objective_eur=float('nan'), stage2_objective_slack_eur=float('nan'), stage2_branch_l_objective=float('nan'), physics_refinement_mode=str(self.physics_refinement_mode), physics_refinement_status='stage1_no_solution' if not base_result.has_solution else 'not_enabled' if str(self.physics_refinement_mode) == 'none' else 'floor_pending', physics_refinement_runtime_sec=0.0, floor_p95_soc_slack=float('nan'), floor_mean_abs_solver_feeder_gap_kw=float('nan'), floor_primary_objective_eur=float('nan'), floor_primary_delta_signed_eur=float('nan'), floor_primary_delta_positive_eur=float('nan'), physics_refinement_slack_cap_eur=initial_slack_cap_eur, initial_physics_refinement_slack_cap_eur=initial_slack_cap_eur, returned_primary_objective_eur=returned_primary_stage1, returned_primary_delta_abs_eur=0.0 if np.isfinite(returned_primary_stage1) else float('nan'), returned_primary_delta_pct=0.0 if np.isfinite(returned_primary_stage1) else float('nan'), floor_accepted_tier=None, used_physics_refinement_tier=None, total_tiers_configured=total_tiers_configured, physics_refinement_attempt_count=0, physics_refinement_attempt_caps_eur=[], physics_refinement_cap_utilization=float('nan'), branch_l_gap_ratio_to_floor=float('nan'), returned_mean_abs_solver_feeder_gap_kw=float('nan'), returned_max_solver_feeder_gap_kw=float('nan'), returned_mean_abs_export_gap_ratio=float('nan'), high_budget_refinement_warn=False, returned_solution_source='stage1' if base_result.has_solution else 'none', formulation_tightening_required=False, negative_floor_delta_warn=False, refinement_status_counts=self._single_refinement_status_counts('stage1_no_solution' if not base_result.has_solution else 'not_enabled' if str(self.physics_refinement_mode) == 'none' else 'floor_pending'))
        if not base_result.has_solution or str(self.physics_refinement_mode) != 'two_stage_min_branch_l':
            return result_with_defaults
        refinement_config = self._build_physics_refinement_solve_config(primary_solve_config)
        floor_result = self._solve_sequences(import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, solve_config=refinement_config, export_debug=export_debug, debug_tag=None if debug_tag is None else f'{debug_tag}_floor', expected_horizon=None, enforce_terminal_soc=enforce_terminal_soc, solve_mode=solve_mode, objective_mode='min_branch_l', include_branch_current_tiebreaker_in_primary=False)
        floor_p95_soc_slack = float('nan')
        floor_mean_abs_solver_feeder_gap_kw = float('nan')
        floor_max_abs_solver_feeder_gap_kw = float('nan')
        floor_primary_objective_eur = float('nan')
        floor_primary_delta_signed_eur = float('nan')
        floor_primary_delta_positive_eur = float('nan')
        floor_branch_l_objective = float('nan')
        negative_floor_delta_warn = False
        floor_metrics = {'mean_gap_kw': float('nan'), 'max_gap_kw': float('nan'), 'mean_export_gap_ratio': float('nan'), 'export_gap_step_count': 0, 'branch_l_gap_ratio': float('nan'), 'branch_l_objective': float('nan')}
        if floor_result.has_solution:
            floor_primary_objective_eur = float(self._primary_objective_from_result(floor_result))
            floor_primary_delta_signed_eur = float(floor_primary_objective_eur - stage1_primary)
            floor_primary_delta_positive_eur = float(max(0.0, floor_primary_delta_signed_eur))
            negative_floor_delta_warn = self._negative_floor_delta_warn(stage1_primary_objective_eur=stage1_primary, floor_primary_delta_signed_eur=floor_primary_delta_signed_eur)
            floor_p95_soc_slack, _ = self._soc_relaxation_summary_from_result(floor_result)
            floor_mean_abs_solver_feeder_gap_kw, floor_max_abs_solver_feeder_gap_kw = self._solver_feeder_gap_summary_from_result(floor_result, load_seq=load_seq, pv_seq=pv_seq)
            floor_branch_l_objective = float(self._branch_l_objective_from_result(floor_result))
            floor_metrics = self._refinement_metrics_from_result(floor_result, load_seq=load_seq, pv_seq=pv_seq, floor_branch_l_objective=floor_branch_l_objective)
        stage1_metrics = self._refinement_metrics_from_result(base_result, load_seq=load_seq, pv_seq=pv_seq, floor_branch_l_objective=floor_branch_l_objective) if base_result.has_solution else {'mean_gap_kw': float('nan'), 'max_gap_kw': float('nan'), 'mean_export_gap_ratio': float('nan'), 'export_gap_step_count': 0, 'branch_l_gap_ratio': float('nan'), 'branch_l_objective': float('nan')}
        base_with_floor = replace(result_with_defaults, floor_p95_soc_slack=floor_p95_soc_slack, floor_mean_abs_solver_feeder_gap_kw=floor_mean_abs_solver_feeder_gap_kw, floor_primary_objective_eur=floor_primary_objective_eur, floor_primary_delta_signed_eur=floor_primary_delta_signed_eur, floor_primary_delta_positive_eur=floor_primary_delta_positive_eur, physics_refinement_slack_cap_eur=initial_slack_cap_eur, negative_floor_delta_warn=negative_floor_delta_warn)
        if not floor_result.has_solution:
            return replace(base_with_floor, physics_refinement_status='floor_no_solution', physics_refinement_runtime_sec=float(floor_result.solve_time_sec), refinement_status_counts=self._single_refinement_status_counts('floor_no_solution'))
        if not (np.isfinite(floor_p95_soc_slack) and np.isfinite(floor_mean_abs_solver_feeder_gap_kw) and (floor_p95_soc_slack < 0.0001) and (floor_mean_abs_solver_feeder_gap_kw < 5.0)):
            return replace(base_with_floor, physics_refinement_status='floor_failed_gate', physics_refinement_runtime_sec=float(floor_result.solve_time_sec), branch_l_gap_ratio_to_floor=float(stage1_metrics['branch_l_gap_ratio']), returned_mean_abs_solver_feeder_gap_kw=float(stage1_metrics['mean_gap_kw']), returned_max_solver_feeder_gap_kw=float(stage1_metrics['max_gap_kw']), returned_mean_abs_export_gap_ratio=float(stage1_metrics['mean_export_gap_ratio']), formulation_tightening_required=True, refinement_status_counts=self._single_refinement_status_counts('floor_failed_gate'))

        def _finalize_with_candidate(candidate_result: MISOCPResult, *, status: str, used_cap_eur: float, used_tier: int, metrics: dict[str, float], candidate_primary_objective_eur: float, runtime_sec: float, attempt_count: int, attempt_caps_eur: list[float], source: str, floor_tier: int | None=None) -> MISOCPResult:
            delta_signed = float(candidate_primary_objective_eur - stage1_primary)
            delta_abs = float(abs(delta_signed))
            delta_pct = 100.0 * delta_signed / max(abs(float(stage1_primary)), 1.0)
            return replace(candidate_result, stage1_primary_objective_eur=float(stage1_primary), stage2_primary_objective_eur=float('nan') if source == 'floor' else float(candidate_primary_objective_eur), stage2_objective_slack_eur=float('nan') if source == 'floor' else float(used_cap_eur), stage2_branch_l_objective=float('nan') if source == 'floor' else float(metrics['branch_l_objective']), physics_refinement_mode=str(self.physics_refinement_mode), physics_refinement_status=str(status), physics_refinement_runtime_sec=float(runtime_sec), floor_p95_soc_slack=floor_p95_soc_slack, floor_mean_abs_solver_feeder_gap_kw=floor_mean_abs_solver_feeder_gap_kw, floor_primary_objective_eur=floor_primary_objective_eur, floor_primary_delta_signed_eur=floor_primary_delta_signed_eur, floor_primary_delta_positive_eur=floor_primary_delta_positive_eur, physics_refinement_slack_cap_eur=float(used_cap_eur), initial_physics_refinement_slack_cap_eur=float(initial_slack_cap_eur), returned_primary_objective_eur=float(candidate_primary_objective_eur), returned_primary_delta_abs_eur=float(delta_abs), returned_primary_delta_pct=float(delta_pct), floor_accepted_tier=floor_tier, used_physics_refinement_tier=int(used_tier), total_tiers_configured=int(total_tiers_configured), physics_refinement_attempt_count=int(attempt_count), physics_refinement_attempt_caps_eur=[float(value) for value in attempt_caps_eur], physics_refinement_cap_utilization=float(max(0.0, delta_signed) / max(float(used_cap_eur), _ROOT_VM_EPS)) if np.isfinite(delta_signed) else float('nan'), branch_l_gap_ratio_to_floor=float(metrics['branch_l_gap_ratio']), returned_mean_abs_solver_feeder_gap_kw=float(metrics['mean_gap_kw']), returned_max_solver_feeder_gap_kw=float(metrics['max_gap_kw']), returned_mean_abs_export_gap_ratio=float(metrics['mean_export_gap_ratio']), high_budget_refinement_warn=self._high_budget_refinement_warn(returned_primary_delta_eur=max(0.0, delta_signed), returned_primary_delta_pct=max(0.0, float(delta_pct))), returned_solution_source=str(source), formulation_tightening_required=False, negative_floor_delta_warn=negative_floor_delta_warn, refinement_status_counts=self._single_refinement_status_counts(str(status)))
        floor_accept_tier = next((tier_idx for tier_idx, cap_eur in enumerate(tier_caps) if float(floor_primary_delta_positive_eur) <= float(cap_eur)), None)
        if floor_accept_tier is not None:
            used_cap_eur = float(tier_caps[floor_accept_tier])
            return _finalize_with_candidate(floor_result, status='floor_accepted', used_cap_eur=used_cap_eur, used_tier=int(floor_accept_tier), metrics=floor_metrics, candidate_primary_objective_eur=float(floor_primary_objective_eur), runtime_sec=float(floor_result.solve_time_sec), attempt_count=0, attempt_caps_eur=[], source='floor', floor_tier=int(floor_accept_tier))
        full_mip_start = self.build_full_mip_start(base_result) if self.physics_refinement_use_full_start else None
        previous_successful_start = full_mip_start
        stage2_attempt_caps_eur: list[float] = []
        stage2_runtime_total_sec = 0.0
        best_stage2_payload: tuple[MISOCPResult, int, float, dict[str, float]] | None = None
        for tier_idx, slack_cap_eur in enumerate(tier_caps):
            stage2_result = self._solve_sequences(import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, solve_config=refinement_config, export_debug=export_debug, debug_tag=None if debug_tag is None else f'{debug_tag}_stage2_tier{tier_idx + 1}', expected_horizon=None, enforce_terminal_soc=enforce_terminal_soc, solve_mode=solve_mode, partial_mip_start=previous_successful_start, objective_mode='min_branch_l', primary_objective_upper_bound=float(stage1_primary + slack_cap_eur), include_branch_current_tiebreaker_in_primary=False, strict_mip_start=bool(self.physics_refinement_use_full_start))
            stage2_attempt_caps_eur.append(float(slack_cap_eur))
            stage2_runtime_total_sec += float(stage2_result.solve_time_sec)
            if stage2_result.has_solution:
                stage2_primary_objective_eur = float(self._primary_objective_from_result(stage2_result))
                stage2_metrics = self._refinement_metrics_from_result(stage2_result, load_seq=load_seq, pv_seq=pv_seq, floor_branch_l_objective=floor_branch_l_objective)
                stage2_cap_utilization = float(max(0.0, stage2_primary_objective_eur - stage1_primary) / max(float(slack_cap_eur), _ROOT_VM_EPS)) if np.isfinite(stage2_primary_objective_eur) else float('nan')
                best_stage2_payload = (stage2_result, int(tier_idx), float(slack_cap_eur), dict(stage2_metrics))
                if self._refinement_meets_targets(stage2_metrics):
                    return _finalize_with_candidate(stage2_result, status='refined', used_cap_eur=float(slack_cap_eur), used_tier=int(tier_idx), metrics=stage2_metrics, candidate_primary_objective_eur=stage2_primary_objective_eur, runtime_sec=float(stage2_runtime_total_sec + float(floor_result.solve_time_sec)), attempt_count=len(stage2_attempt_caps_eur), attempt_caps_eur=stage2_attempt_caps_eur, source='stage2')
                if float(stage2_runtime_total_sec) >= float(self.physics_refinement_total_time_limit_sec):
                    return _finalize_with_candidate(stage2_result, status='refined_time_budget_limited', used_cap_eur=float(slack_cap_eur), used_tier=int(tier_idx), metrics=stage2_metrics, candidate_primary_objective_eur=stage2_primary_objective_eur, runtime_sec=float(stage2_runtime_total_sec + float(floor_result.solve_time_sec)), attempt_count=len(stage2_attempt_caps_eur), attempt_caps_eur=stage2_attempt_caps_eur, source='stage2')
                should_escalate = bool(tier_idx + 1 < len(tier_caps) and np.isfinite(stage2_cap_utilization) and (float(stage2_cap_utilization) >= float(self.physics_refinement_cap_utilization_trigger)) and np.isfinite(float(stage2_metrics['branch_l_gap_ratio'])) and (float(stage2_metrics['branch_l_gap_ratio']) >= float(self.physics_refinement_branch_l_gap_ratio_trigger)))
                if should_escalate:
                    previous_successful_start = self.build_full_mip_start(stage2_result) if self.physics_refinement_use_full_start else None
                    continue
                return _finalize_with_candidate(stage2_result, status='refined_cap_limited' if tier_idx + 1 >= len(tier_caps) else 'refined_nonbudget_limited', used_cap_eur=float(slack_cap_eur), used_tier=int(tier_idx), metrics=stage2_metrics, candidate_primary_objective_eur=stage2_primary_objective_eur, runtime_sec=float(stage2_runtime_total_sec + float(floor_result.solve_time_sec)), attempt_count=len(stage2_attempt_caps_eur), attempt_caps_eur=stage2_attempt_caps_eur, source='stage2')
            if float(stage2_runtime_total_sec) >= float(self.physics_refinement_total_time_limit_sec):
                if best_stage2_payload is not None:
                    best_result, best_tier, best_cap_eur, best_metrics = best_stage2_payload
                    best_primary = float(self._primary_objective_from_result(best_result))
                    return _finalize_with_candidate(best_result, status='refined_time_budget_limited', used_cap_eur=float(best_cap_eur), used_tier=int(best_tier), metrics=best_metrics, candidate_primary_objective_eur=best_primary, runtime_sec=float(stage2_runtime_total_sec + float(floor_result.solve_time_sec)), attempt_count=len(stage2_attempt_caps_eur), attempt_caps_eur=stage2_attempt_caps_eur, source='stage2')
                return replace(base_with_floor, stage2_objective_slack_eur=float(slack_cap_eur), physics_refinement_status='fallback_stage1', physics_refinement_runtime_sec=float(stage2_runtime_total_sec + float(floor_result.solve_time_sec)), branch_l_gap_ratio_to_floor=float(stage1_metrics['branch_l_gap_ratio']), returned_mean_abs_solver_feeder_gap_kw=float(stage1_metrics['mean_gap_kw']), returned_max_solver_feeder_gap_kw=float(stage1_metrics['max_gap_kw']), returned_mean_abs_export_gap_ratio=float(stage1_metrics['mean_export_gap_ratio']), physics_refinement_attempt_count=len(stage2_attempt_caps_eur), physics_refinement_attempt_caps_eur=[float(value) for value in stage2_attempt_caps_eur], refinement_status_counts=self._single_refinement_status_counts('fallback_stage1'))
            previous_successful_start = full_mip_start if self.physics_refinement_use_full_start else None
        if best_stage2_payload is not None:
            best_result, best_tier, best_cap_eur, best_metrics = best_stage2_payload
            best_primary = float(self._primary_objective_from_result(best_result))
            return _finalize_with_candidate(best_result, status='refined_cap_limited', used_cap_eur=float(best_cap_eur), used_tier=int(best_tier), metrics=best_metrics, candidate_primary_objective_eur=best_primary, runtime_sec=float(stage2_runtime_total_sec + float(floor_result.solve_time_sec)), attempt_count=len(stage2_attempt_caps_eur), attempt_caps_eur=stage2_attempt_caps_eur, source='stage2')
        return replace(base_with_floor, stage2_objective_slack_eur=float(initial_slack_cap_eur), physics_refinement_status='fallback_stage1', physics_refinement_runtime_sec=float(stage2_runtime_total_sec + float(floor_result.solve_time_sec)), branch_l_gap_ratio_to_floor=float(stage1_metrics['branch_l_gap_ratio']), returned_mean_abs_solver_feeder_gap_kw=float(stage1_metrics['mean_gap_kw']), returned_max_solver_feeder_gap_kw=float(stage1_metrics['max_gap_kw']), returned_mean_abs_export_gap_ratio=float(stage1_metrics['mean_export_gap_ratio']), physics_refinement_attempt_count=len(stage2_attempt_caps_eur), physics_refinement_attempt_caps_eur=[float(value) for value in stage2_attempt_caps_eur], refinement_status_counts=self._single_refinement_status_counts('fallback_stage1'))

    def solve(self, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc_init: np.ndarray, export_subsidy: float | None=None, *, verbose: bool=False, time_limit_sec: float | None=None, solve_config: GurobiSolveConfig | None=None, export_debug: bool=False, debug_tag: str | None=None) -> MISOCPResult:
        resolved_config = self._resolve_solve_config(solve_config, time_limit_sec=time_limit_sec)
        stage1_result = self._solve_sequences(import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, solve_config=resolved_config, export_debug=export_debug, debug_tag=debug_tag, expected_horizon=self.horizon, enforce_terminal_soc=True, solve_mode='single_window', include_branch_current_tiebreaker_in_primary=bool(self.physics_refinement_mode == 'none'))
        return self._maybe_apply_physics_refinement(base_result=stage1_result, import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, export_debug=export_debug, debug_tag=debug_tag, enforce_terminal_soc=True, solve_mode='single_window', primary_solve_config=resolved_config)

    def solve_full_horizon(self, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc_init: np.ndarray, export_subsidy: float | None=None, *, timestamps: tuple[str, ...] | list[str] | None=None, episode_offsets: np.ndarray | list[int] | tuple[int, ...] | None=None, episode_lengths: np.ndarray | list[int] | tuple[int, ...] | None=None, verbose: bool=False, time_limit_sec: float | None=None, solve_config: GurobiSolveConfig | None=None, export_debug: bool=False, debug_tag: str | None=None, partial_mip_start: dict[str, np.ndarray] | None=None) -> MISOCPResult:
        horizon_steps = int(np.asarray(import_price_seq).reshape(-1).shape[0])
        resolved_offsets = np.asarray([0] if episode_offsets is None else episode_offsets, dtype=np.int32).reshape(-1)
        if resolved_offsets.size == 0 or int(resolved_offsets[0]) != 0:
            raise ValueError('episode_offsets must start at 0 for full-horizon solves.')
        if np.any(resolved_offsets < 0) or np.any(np.diff(resolved_offsets) < 0):
            raise ValueError('episode_offsets must be non-decreasing and non-negative.')
        if timestamps is not None and len(tuple((str(value) for value in timestamps))) != horizon_steps:
            raise ValueError('timestamps length must match full-horizon import_price_seq length.')
        if episode_lengths is None:
            resolved_lengths = np.diff(np.append(resolved_offsets, horizon_steps)).astype(np.int32)
        else:
            resolved_lengths = np.asarray(episode_lengths, dtype=np.int32).reshape(-1)
            if resolved_lengths.size != resolved_offsets.size:
                raise ValueError('episode_lengths must align with episode_offsets.')
            if int(np.sum(resolved_lengths)) != horizon_steps:
                raise ValueError('episode_lengths must sum to the full-horizon length.')
        resolved_config = self._resolve_solve_config(solve_config, time_limit_sec=time_limit_sec)
        stage1_result = self._solve_sequences(import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, solve_config=resolved_config, export_debug=export_debug, debug_tag=debug_tag, expected_horizon=None, enforce_terminal_soc=True, solve_mode='single_window', partial_mip_start=partial_mip_start, include_branch_current_tiebreaker_in_primary=bool(self.physics_refinement_mode == 'none'))
        result = self._maybe_apply_physics_refinement(base_result=stage1_result, import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, verbose=verbose, export_debug=export_debug, debug_tag=debug_tag, enforce_terminal_soc=True, solve_mode='single_window', primary_solve_config=resolved_config)
        return replace(result, horizon_steps=horizon_steps, solve_mode='single_window', episode_offsets=resolved_offsets.astype(np.int32, copy=False), episode_lengths=resolved_lengths.astype(np.int32, copy=False))

    def solve_chunked_windows(self, full_input: FullHorizonProblemInput, export_subsidy: float | None=None, *, chunk_steps: int, verbose: bool=False, time_limit_sec: float | None=None, solve_config: GurobiSolveConfig | None=None, retry_solve_config: GurobiSolveConfig | None=None, export_debug: bool=False, debug_tag: str | None=None) -> MISOCPResult:
        primary_solve_config = self._resolve_solve_config(solve_config, time_limit_sec=time_limit_sec)
        return self._solve_over_windows(full_input, windows=self._window_slices(full_input, chunk_steps=int(chunk_steps)), export_subsidy=export_subsidy, verbose=verbose, primary_solve_config=primary_solve_config, retry_solve_config=retry_solve_config, export_debug=export_debug, debug_tag=debug_tag, solve_mode='chunked_window')

    def solve_episode_chunks(self, full_input: FullHorizonProblemInput, export_subsidy: float | None=None, *, verbose: bool=False, time_limit_sec: float | None=None, solve_config: GurobiSolveConfig | None=None, retry_solve_config: GurobiSolveConfig | None=None, export_debug: bool=False, debug_tag: str | None=None) -> MISOCPResult:
        primary_solve_config = self._resolve_solve_config(solve_config, time_limit_sec=time_limit_sec)
        episode_windows = [(int(start), int(start) + int(length)) for start, length in zip(np.asarray(full_input.episode_offsets, dtype=np.int32).tolist(), np.asarray(full_input.episode_lengths, dtype=np.int32).tolist(), strict=False)]
        return self._solve_over_windows(full_input, windows=episode_windows, export_subsidy=export_subsidy, verbose=verbose, primary_solve_config=primary_solve_config, retry_solve_config=retry_solve_config, export_debug=export_debug, debug_tag=debug_tag, solve_mode='chunked_episode')

    def solve_adaptive_full_horizon(self, full_input: FullHorizonProblemInput, export_subsidy: float | None=None, *, verbose: bool=False, primary_window_steps: int=_ADAPTIVE_PRIMARY_WINDOW_STEPS, fallback_window_steps: int=_ADAPTIVE_FALLBACK_WINDOW_STEPS, solve_config: GurobiSolveConfig | None=None, retry_solve_config: GurobiSolveConfig | None=None, export_debug: bool=False, debug_tag: str | None=None) -> MISOCPResult:
        primary_config = solve_config or default_primary_solve_config()
        retry_config = retry_solve_config or default_retry_solve_config(primary_config)
        horizon_steps = int(full_input.horizon_steps)
        single_result: MISOCPResult | None = None
        if horizon_steps <= int(primary_window_steps):
            single_result, primary_result, retry_result, _ = self._run_window_with_retry(import_price_seq=full_input.import_price_seq, load_seq=full_input.load_seq, pv_seq=full_input.pv_seq, soc_init=full_input.soc_init, export_subsidy=export_subsidy, verbose=verbose, export_debug=export_debug, debug_tag=None if debug_tag is None else f'{debug_tag}_single_window', enforce_terminal_soc=True, solve_mode='single_window', primary_solve_config=primary_config, retry_solve_config=retry_config)
            single_debug_artifacts: dict[str, str] = {}
            for key, value in primary_result.debug_artifacts.items():
                single_debug_artifacts[f'single_window_primary_{key}'] = value
            if retry_result is not None:
                for key, value in retry_result.debug_artifacts.items():
                    single_debug_artifacts[f'single_window_retry_{key}'] = value
            single_result = replace(single_result, debug_artifacts={**single_debug_artifacts, **dict(single_result.debug_artifacts)}, horizon_steps=horizon_steps, episode_offsets=np.asarray(full_input.episode_offsets, dtype=np.int32), episode_lengths=np.asarray(full_input.episode_lengths, dtype=np.int32), no_retry_or_fallback_used=bool(primary_result.has_solution), chunk_retry_count=0)
            if single_result.has_solution:
                return single_result
        chunked_result = self.solve_chunked_windows(full_input, export_subsidy=export_subsidy, chunk_steps=int(fallback_window_steps), verbose=verbose, solve_config=primary_config, retry_solve_config=retry_config, export_debug=export_debug, debug_tag=None if debug_tag is None else f'{debug_tag}_chunked_window')
        if single_result is None:
            return chunked_result
        combined_debug = dict(chunked_result.debug_artifacts)
        for key, value in single_result.debug_artifacts.items():
            combined_debug[f'adaptive_{key}'] = value
        return replace(chunked_result, debug_artifacts=combined_debug, no_retry_or_fallback_used=False)

    def _solve_over_windows(self, full_input: FullHorizonProblemInput, *, windows: list[tuple[int, int]], export_subsidy: float | None, verbose: bool, primary_solve_config: GurobiSolveConfig, retry_solve_config: GurobiSolveConfig | None, export_debug: bool, debug_tag: str | None, solve_mode: str) -> MISOCPResult:
        chunk_results: list[MISOCPResult] = []
        chunk_summaries: list[dict[str, object]] = []
        current_soc = np.asarray(full_input.soc_init, dtype=np.float32).copy()
        debug_artifacts: dict[str, str] = {}
        n_chunks = int(len(windows))
        for chunk_idx, (start, end) in enumerate(windows):
            chosen_result, primary_result, retry_result, attempt_used = self._run_window_with_retry(import_price_seq=full_input.import_price_seq[int(start):int(end)], load_seq=full_input.load_seq[:, int(start):int(end)], pv_seq=full_input.pv_seq[:, int(start):int(end)], soc_init=current_soc, export_subsidy=export_subsidy, verbose=verbose, export_debug=export_debug, debug_tag=None if debug_tag is None else f'{debug_tag}_chunk{chunk_idx:03d}', enforce_terminal_soc=bool(chunk_idx == n_chunks - 1), solve_mode=solve_mode, primary_solve_config=primary_solve_config, retry_solve_config=retry_solve_config)
            for key, value in primary_result.debug_artifacts.items():
                debug_artifacts[f'chunk_{chunk_idx}_primary_{key}'] = value
            if retry_result is not None:
                for key, value in retry_result.debug_artifacts.items():
                    debug_artifacts[f'chunk_{chunk_idx}_retry_{key}'] = value
            chunk_summary = self._make_chunk_summary(chosen_result, chunk_idx=chunk_idx, start_step=int(start), end_step=int(end), attempt_used=attempt_used)
            chunk_summaries.append(chunk_summary)
            if not chosen_result.has_solution:
                chunk_retry_count = int(sum((1 for item in chunk_summaries if str(item.get('attempt_used', '')) == 'retry')))
                return replace(chosen_result, solve_mode=solve_mode, horizon_steps=int(full_input.horizon_steps), episode_offsets=full_input.episode_offsets.astype(np.int32, copy=False), episode_lengths=full_input.episode_lengths.astype(np.int32, copy=False), debug_artifacts=debug_artifacts, chunk_summaries=chunk_summaries, no_retry_or_fallback_used=bool(chunk_retry_count == 0), chunk_retry_count=chunk_retry_count)
            chunk_results.append(chosen_result)
            assert chosen_result.energy_mwh is not None
            current_soc = (np.asarray(chosen_result.energy_mwh[:, -1], dtype=np.float32) / np.maximum(self.capacity_mwh, _ROOT_VM_EPS)).astype(np.float32)
        return self._concatenate_chunk_results(chunk_results, episode_offsets=full_input.episode_offsets, episode_lengths=full_input.episode_lengths, debug_artifacts=debug_artifacts, solve_mode=solve_mode, chunk_summaries=chunk_summaries)

    def build_full_horizon_input(self, env: Any, episode_indices: Sequence[int] | None=None) -> FullHorizonProblemInput:
        if episode_indices is None:
            requested_episode_indices = list(range(int(env.num_available_episodes)))
        else:
            requested_episode_indices = [int(index) for index in episode_indices]
        if not requested_episode_indices:
            raise ValueError('build_full_horizon_input requires at least one episode index.')
        if requested_episode_indices != sorted(requested_episode_indices):
            raise ValueError('Global MISOCP full-horizon stitching requires episode_indices to be sorted in ascending order.')
        for episode_idx in requested_episode_indices:
            if episode_idx < 0 or episode_idx >= int(env.num_available_episodes):
                raise IndexError(f'episode_idx={episode_idx} is out of range [0, {int(env.num_available_episodes) - 1}]')
        if any((requested_episode_indices[offset + 1] != requested_episode_indices[offset] + 1 for offset in range(len(requested_episode_indices) - 1))):
            raise ValueError('Global MISOCP full-horizon stitching only supports contiguous episode index ranges.')
        wholesale_price_chunks: list[np.ndarray] = []
        price_chunks: list[np.ndarray] = []
        load_chunks: list[np.ndarray] = []
        pv_chunks: list[np.ndarray] = []
        timestamps: list[str] = []
        episode_offsets: list[int] = []
        episode_lengths: list[int] = []
        stitched_episode_indices: list[int] = []
        cursor = 0
        for episode_idx in requested_episode_indices:
            episode = env._dataset.get_episode(int(episode_idx))
            signals = dict(episode.get('signals', {}))
            if 'wholesale_price' not in signals or 'load' not in signals:
                raise KeyError("Full-horizon MISOCP requires episode signals['wholesale_price'] and signals['load'].")
            wholesale_price = np.asarray(signals['wholesale_price'], dtype=np.float32).reshape(-1)
            import_price = self.apply_import_price_markup(wholesale_price)
            load = np.asarray(signals['load'], dtype=np.float32)
            pv = np.asarray(signals.get('pv', np.zeros_like(load, dtype=np.float32)), dtype=np.float32)
            if load.shape != (import_price.size, self.n_agents):
                raise ValueError(f'Episode {episode_idx} load signal should have shape {(import_price.size, self.n_agents)}, got {load.shape}.')
            if pv.shape != load.shape:
                raise ValueError(f'Episode {episode_idx} pv signal should match load shape {load.shape}, got {pv.shape}.')
            episode_meta = dict(episode.get('meta', {}))
            episode_timestamps = [str(value) for value in list(episode_meta.get('timestamps') or [])]
            if episode_timestamps and len(episode_timestamps) != import_price.size:
                raise ValueError(f'Episode {episode_idx} timestamps length {len(episode_timestamps)} does not match import price length {import_price.size}.')
            if not episode_timestamps:
                episode_timestamps = [f'episode{episode_idx:03d}_step{step_idx:04d}' for step_idx in range(import_price.size)]
            episode_offsets.append(cursor)
            episode_lengths.append(int(import_price.size))
            stitched_episode_indices.append(int(episode_idx))
            wholesale_price_chunks.append(wholesale_price.astype(np.float32, copy=False))
            price_chunks.append(import_price.astype(np.float32, copy=False))
            load_chunks.append(load.T.astype(np.float32, copy=False))
            pv_chunks.append(pv.T.astype(np.float32, copy=False))
            timestamps.extend(episode_timestamps)
            cursor += int(import_price.size)
        return FullHorizonProblemInput(wholesale_price_seq=np.concatenate(wholesale_price_chunks, axis=0).astype(np.float32, copy=False), import_price_seq=np.concatenate(price_chunks, axis=0).astype(np.float32, copy=False), load_seq=np.concatenate(load_chunks, axis=1).astype(np.float32, copy=False), pv_seq=np.concatenate(pv_chunks, axis=1).astype(np.float32, copy=False), soc_init=np.asarray(getattr(env, 'soc', np.full((self.n_agents,), getattr(env, 'init_soc', 0.5))), dtype=np.float32).reshape(self.n_agents), timestamps=tuple(timestamps), episode_offsets=np.asarray(episode_offsets, dtype=np.int32), episode_lengths=np.asarray(episode_lengths, dtype=np.int32), episode_indices=np.asarray(stitched_episode_indices, dtype=np.int32))

    def build_full_mip_start(self, result: MISOCPResult) -> dict[str, np.ndarray]:
        if not bool(result.has_solution):
            raise ValueError('Cannot build a MIP start from a result without a feasible incumbent.')
        if result.battery_charge_mw is None or result.battery_discharge_mw is None or result.pv_curtail_mw is None or (result.energy_mwh is None) or (result.agent_net_grid_mw is None) or (result.agent_import_mw is None) or (result.agent_export_mw is None) or (result.branch_p_pu is None) or (result.branch_q_pu is None) or (result.branch_i2_pu is None) or (result.bus_v_sq is None):
            raise ValueError('Result is missing one or more full-state trajectories required for a full MIP start.')
        if result.root_import_mw is None and result.root_export_mw is None:
            raise ValueError('Result is missing root import/export trajectories required for u_grid warm start.')
        if result.root_import_mw is None:
            u_grid = np.zeros((int(result.horizon_steps),), dtype=np.float32)
        else:
            u_grid = (np.asarray(result.root_import_mw, dtype=np.float32).reshape(-1) > 1e-06).astype(np.float32)
        return {'u_grid': u_grid.astype(np.float32, copy=False), 'p_charge': np.asarray(result.battery_charge_mw, dtype=np.float32), 'p_discharge': np.asarray(result.battery_discharge_mw, dtype=np.float32), 'pv_curtail': np.asarray(result.pv_curtail_mw, dtype=np.float32), 'agent_abs_grid': np.asarray(np.maximum(result.agent_import_mw, result.agent_export_mw), dtype=np.float32), 'energy': np.asarray(result.energy_mwh, dtype=np.float32), 'branch_p': np.asarray(result.branch_p_pu, dtype=np.float32), 'branch_q': np.asarray(result.branch_q_pu, dtype=np.float32), 'branch_l': np.asarray(result.branch_i2_pu, dtype=np.float32), 'bus_v': np.asarray(result.bus_v_sq, dtype=np.float32), 'p_import': np.asarray(result.root_import_mw, dtype=np.float32), 'p_export': np.asarray(result.root_export_mw, dtype=np.float32)}

    def build_partial_mip_start(self, result: MISOCPResult) -> dict[str, np.ndarray]:
        return self.build_full_mip_start(result)

    def _apply_gurobi_solve_config(self, model: Any, solve_config: GurobiSolveConfig) -> None:
        model.Params.TimeLimit = float(_TIME_LIMIT_SEC if solve_config.time_limit_sec is None else solve_config.time_limit_sec)
        model.Params.MIPGap = float(solve_config.mip_gap)
        if solve_config.threads is not None:
            model.Params.Threads = int(solve_config.threads)
        if solve_config.presolve is not None:
            model.Params.Presolve = int(solve_config.presolve)
        if solve_config.cuts is not None:
            model.Params.Cuts = int(solve_config.cuts)
        if solve_config.heuristics is not None:
            model.Params.Heuristics = float(solve_config.heuristics)
        if solve_config.mip_focus is not None:
            model.Params.MIPFocus = int(solve_config.mip_focus)

    def _apply_partial_mip_start(self, *, u_grid: Any, p_charge: Any, p_discharge: Any, pv_curtail: Any, agent_abs_grid: Any, energy: Any, branch_p: Any, branch_q: Any, branch_l: Any, bus_v: Any, p_import: Any, p_export: Any, partial_mip_start: dict[str, np.ndarray] | None, horizon_steps: int, strict: bool=False) -> str:
        if not partial_mip_start:
            return ''
        try:
            if 'u_grid' in partial_mip_start:
                u_values = np.asarray(partial_mip_start['u_grid'], dtype=np.float32).reshape(-1)
                if u_values.shape[0] != horizon_steps:
                    raise ValueError(f'u_grid warm start length {u_values.shape[0]} != horizon_steps {horizon_steps}.')
                for step_idx in range(horizon_steps):
                    u_grid[step_idx].Start = float(1.0 if u_values[step_idx] >= 0.5 else 0.0)
            matrix_specs = (('p_charge', p_charge, (self.n_agents, horizon_steps)), ('p_discharge', p_discharge, (self.n_agents, horizon_steps)), ('pv_curtail', pv_curtail, (self.n_agents, horizon_steps)), ('agent_abs_grid', agent_abs_grid, (self.n_agents, horizon_steps)), ('energy', energy, (self.n_agents, horizon_steps + 1)), ('branch_p', branch_p, (self.n_branches, horizon_steps)), ('branch_q', branch_q, (self.n_branches, horizon_steps)), ('branch_l', branch_l, (self.n_branches, horizon_steps)), ('bus_v', bus_v, (self.n_buses, horizon_steps)))
            for key, variable_grid, expected_shape in matrix_specs:
                if key not in partial_mip_start:
                    continue
                values = np.asarray(partial_mip_start[key], dtype=np.float32)
                if values.shape != expected_shape:
                    raise ValueError(f'{key} warm start shape {values.shape} does not match expected shape {expected_shape}.')
                for row_idx in range(expected_shape[0]):
                    for col_idx in range(expected_shape[1]):
                        variable_grid[row_idx, col_idx].Start = float(values[row_idx, col_idx])
            vector_specs = (('p_import', p_import, horizon_steps), ('p_export', p_export, horizon_steps))
            for key, variable_grid, expected_len in vector_specs:
                if key not in partial_mip_start:
                    continue
                values = np.asarray(partial_mip_start[key], dtype=np.float32).reshape(-1)
                if values.shape[0] != expected_len:
                    raise ValueError(f'{key} warm start length {values.shape[0]} != expected {expected_len}.')
                for idx in range(expected_len):
                    variable_grid[idx].Start = float(values[idx])
        except Exception as exc:
            if strict:
                raise
            return f'Partial MIP start ignored: {exc}'
        return ''

    def _solve_sequences(self, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc_init: np.ndarray, export_subsidy: float | None, *, verbose: bool, solve_config: GurobiSolveConfig, export_debug: bool, debug_tag: str | None, expected_horizon: int | None, enforce_terminal_soc: bool, solve_mode: str, partial_mip_start: dict[str, np.ndarray] | None=None, objective_mode: str='primary', primary_objective_upper_bound: float | None=None, include_branch_current_tiebreaker_in_primary: bool=True, strict_mip_start: bool=False) -> MISOCPResult:
        import_price_seq = np.asarray(import_price_seq, dtype=np.float32).reshape(-1)
        horizon_steps = int(import_price_seq.size)
        if expected_horizon is not None and horizon_steps != int(expected_horizon):
            raise ValueError(f'Expected import price horizon {expected_horizon}, got {horizon_steps}.')
        bundle = self._build_sequence_model(import_price_seq=import_price_seq, load_seq=load_seq, pv_seq=pv_seq, soc_init=soc_init, export_subsidy=export_subsidy, enforce_terminal_soc=enforce_terminal_soc, solve_mode=solve_mode, primary_objective_upper_bound=primary_objective_upper_bound, include_branch_current_tiebreaker_in_primary=include_branch_current_tiebreaker_in_primary)
        model = bundle.model
        model.Params.OutputFlag = 1 if verbose else 0
        self._apply_gurobi_solve_config(model, solve_config)
        if str(objective_mode) == 'min_branch_l':
            model.setObjective(bundle.branch_l_objective_expr, GRB.MINIMIZE)
        elif str(objective_mode) == 'primary':
            model.setObjective(bundle.primary_objective_expr, GRB.MINIMIZE)
        else:
            raise ValueError(f'Unsupported objective_mode {objective_mode!r}.')
        model.update()
        mip_start_warning = self._apply_partial_mip_start(u_grid=bundle.u_grid, p_charge=bundle.p_charge, p_discharge=bundle.p_discharge, pv_curtail=bundle.pv_curtail, agent_abs_grid=bundle.agent_abs_grid, energy=bundle.energy, branch_p=bundle.branch_p, branch_q=bundle.branch_q, branch_l=bundle.branch_l, bus_v=bundle.bus_v, p_import=bundle.p_import, p_export=bundle.p_export, partial_mip_start=partial_mip_start, horizon_steps=horizon_steps, strict=strict_mip_start)
        model_size = ModelSize(num_vars=int(model.NumVars), num_binary_vars=int(model.NumBinVars), num_linear_constraints=int(model.NumConstrs), num_quadratic_constraints=int(model.NumQConstrs))
        try:
            model.optimize()
        except Exception as exc:
            raise RuntimeError(f'{_GUROBI_ERROR_PREFIX}: Gurobi optimize failed: {exc}') from exc
        status_code = int(model.Status)
        status_label = _status_label(status_code)
        sol_count = int(model.SolCount)
        node_count = float(getattr(model, 'NodeCount', float('nan')))
        iter_count = float(getattr(model, 'IterCount', float('nan')))
        bar_iter_count = float(getattr(model, 'BarIterCount', float('nan')))
        has_solution = status_code == int(GRB.OPTIMAL) or (status_code == int(GRB.TIME_LIMIT) and sol_count > 0)
        time_limit_feasible = status_code == int(GRB.TIME_LIMIT) and sol_count > 0
        debug_artifacts = self._maybe_export_debug_artifacts(model, status_code=status_code, export_debug=export_debug, debug_tag=debug_tag)
        if not has_solution:
            return MISOCPResult(status_code=status_code, status_label=status_label, has_solution=False, time_limit_feasible=False, solve_time_sec=float(model.Runtime), mip_gap=float('nan'), best_bound=float(model.ObjBound) if hasattr(model, 'ObjBound') else float('nan'), objective_value=float(model.ObjVal) if sol_count > 0 else float('nan'), agent_purchase_cost_eur=float('nan'), agent_export_subsidy_eur=float('nan'), agent_net_cost_eur=float('nan'), feeder_purchase_cost_eur=float('nan'), feeder_export_subsidy_eur=float('nan'), feeder_net_cost_eur=float('nan'), throughput_regularization_eur=float('nan'), throughput_regularization_weight=float(self.throughput_regularization_eur_per_kwh), physical_tiebreaker_eur=float('nan'), physical_tiebreaker_weight=float(self.branch_current_tiebreaker_eur_per_pu_step), model_size=model_size, debug_artifacts=debug_artifacts, agent_net_grid_mw=None, agent_import_mw=None, agent_export_mw=None, battery_charge_mw=None, battery_discharge_mw=None, pv_curtail_mw=None, energy_mwh=None, branch_p_pu=None, branch_q_pu=None, branch_i2_pu=None, bus_v_sq=None, root_import_mw=None, root_export_mw=None, root_p_kw=None, root_q_kvar=None, bus_vm_pu=None, line_loading_pct=None, trafo_loading_pct=None, simultaneous_charge_discharge_kw=None, simultaneous_agent_steps=0, simultaneous_step_ratio=0.0, max_simultaneous_kw=0.0, sol_count=sol_count, node_count=node_count, iter_count=iter_count, bar_iter_count=bar_iter_count, sanity_warning=str(mip_start_warning or ''), horizon_steps=horizon_steps, solve_mode=solve_mode)
        battery_charge_mw = _extract_grid_values(bundle.p_charge, self.n_agents, horizon_steps)
        battery_discharge_mw = _extract_grid_values(bundle.p_discharge, self.n_agents, horizon_steps)
        pv_curtail_mw = _extract_grid_values(bundle.pv_curtail, self.n_agents, horizon_steps)
        energy_mwh = _extract_grid_values(bundle.energy, self.n_agents, horizon_steps + 1)
        branch_p_pu = _extract_grid_values(bundle.branch_p, self.n_branches, horizon_steps)
        branch_q_pu = _extract_grid_values(bundle.branch_q, self.n_branches, horizon_steps)
        branch_i2_pu = _extract_grid_values(bundle.branch_l, self.n_branches, horizon_steps)
        bus_v_sq = _extract_grid_values(bundle.bus_v, self.n_buses, horizon_steps)
        root_import_mw = _extract_vector_values(bundle.p_import, horizon_steps)
        root_export_mw = _extract_vector_values(bundle.p_export, horizon_steps)
        root_p_kw = (np.sum(branch_p_pu[self.network.root_outgoing_branches.tolist(), :], axis=0) * self.network.s_base_mva * 1000.0).astype(np.float32)
        root_q_kvar = (np.sum(branch_q_pu[self.network.root_outgoing_branches.tolist(), :], axis=0) * self.network.s_base_mva * 1000.0).astype(np.float32)
        bus_vm_pu = np.sqrt(np.maximum(bus_v_sq, 0.0)).astype(np.float32)
        line_loading_pct = self._line_loading_pct_from_solution(branch_i2_pu)
        trafo_loading_pct = (100.0 * np.sqrt(np.maximum((root_p_kw / (self.network.s_base_mva * 1000.0)) ** 2 + (root_q_kvar / (self.network.s_base_mva * 1000.0)) ** 2, 0.0))).reshape(1, -1).astype(np.float32)
        agent_net_grid_mw = _compute_agent_net_grid_mw(bundle.load_seq, bundle.pv_seq, battery_charge_mw, battery_discharge_mw, pv_curtail_mw)
        agent_import_mw = np.clip(agent_net_grid_mw, 0.0, None).astype(np.float32)
        agent_export_mw = np.clip(-agent_net_grid_mw, 0.0, None).astype(np.float32)
        agent_purchase_cost_eur = float(np.sum(1000.0 * self.dt_hours * bundle.import_price_seq.reshape(1, -1) * agent_import_mw))
        agent_export_subsidy_eur = float(np.sum(1000.0 * self.dt_hours * bundle.subsidy * agent_export_mw))
        agent_net_cost_eur = float(agent_purchase_cost_eur - agent_export_subsidy_eur)
        feeder_purchase_cost_eur = float(np.sum(1000.0 * self.dt_hours * bundle.import_price_seq * root_import_mw))
        feeder_export_subsidy_eur = float(np.sum(1000.0 * self.dt_hours * bundle.subsidy * root_export_mw))
        feeder_net_cost_eur = float(feeder_purchase_cost_eur - feeder_export_subsidy_eur)
        throughput_regularization_eur = float(np.sum(1000.0 * self.dt_hours * self.throughput_regularization_eur_per_kwh * (battery_charge_mw + battery_discharge_mw)))
        physical_tiebreaker_eur = float(self.dt_hours * self.branch_current_tiebreaker_eur_per_pu_step * np.sum(np.asarray(branch_i2_pu, dtype=np.float64)))
        simultaneous_charge_discharge_kw, simultaneous_agent_steps, simultaneous_step_ratio, max_simultaneous_kw = self._simultaneous_metrics(battery_charge_mw, battery_discharge_mw)
        return MISOCPResult(status_code=status_code, status_label=status_label, has_solution=True, time_limit_feasible=time_limit_feasible, solve_time_sec=float(model.Runtime), mip_gap=float(model.MIPGap) if hasattr(model, 'MIPGap') else float('nan'), best_bound=float(model.ObjBound) if hasattr(model, 'ObjBound') else float('nan'), objective_value=float(model.ObjVal), agent_purchase_cost_eur=agent_purchase_cost_eur, agent_export_subsidy_eur=agent_export_subsidy_eur, agent_net_cost_eur=agent_net_cost_eur, feeder_purchase_cost_eur=feeder_purchase_cost_eur, feeder_export_subsidy_eur=feeder_export_subsidy_eur, feeder_net_cost_eur=feeder_net_cost_eur, throughput_regularization_eur=throughput_regularization_eur, throughput_regularization_weight=float(self.throughput_regularization_eur_per_kwh), physical_tiebreaker_eur=physical_tiebreaker_eur, physical_tiebreaker_weight=float(self.branch_current_tiebreaker_eur_per_pu_step), model_size=model_size, debug_artifacts=debug_artifacts, agent_net_grid_mw=agent_net_grid_mw, agent_import_mw=agent_import_mw, agent_export_mw=agent_export_mw, battery_charge_mw=battery_charge_mw, battery_discharge_mw=battery_discharge_mw, pv_curtail_mw=pv_curtail_mw, energy_mwh=energy_mwh, branch_p_pu=branch_p_pu, branch_q_pu=branch_q_pu, branch_i2_pu=branch_i2_pu, bus_v_sq=bus_v_sq, root_import_mw=root_import_mw.astype(np.float32), root_export_mw=root_export_mw.astype(np.float32), root_p_kw=root_p_kw, root_q_kvar=root_q_kvar, bus_vm_pu=bus_vm_pu, line_loading_pct=line_loading_pct, trafo_loading_pct=trafo_loading_pct, simultaneous_charge_discharge_kw=simultaneous_charge_discharge_kw, simultaneous_agent_steps=simultaneous_agent_steps, simultaneous_step_ratio=simultaneous_step_ratio, max_simultaneous_kw=max_simultaneous_kw, sol_count=sol_count, node_count=node_count, iter_count=iter_count, bar_iter_count=bar_iter_count, sanity_warning=str(mip_start_warning or ''), horizon_steps=horizon_steps, solve_mode=solve_mode)

    def _concatenate_chunk_results(self, chunk_results: list[MISOCPResult], *, episode_offsets: np.ndarray, episode_lengths: np.ndarray, debug_artifacts: dict[str, str], solve_mode: str, chunk_summaries: list[dict[str, object]] | None=None) -> MISOCPResult:
        if not chunk_results:
            raise ValueError('chunk_results must not be empty.')
        reference = chunk_results[0]
        optimal_status_code = int(GRB.OPTIMAL) if GRB is not None else 2
        time_limit_status_code = int(GRB.TIME_LIMIT) if GRB is not None else 9
        all_optimal = all((int(result.status_code) == optimal_status_code for result in chunk_results))
        status_code = int(optimal_status_code if all_optimal else time_limit_status_code)
        status_label = 'optimal' if all_optimal else 'time_limit'
        has_solution = True
        time_limit_feasible = not all_optimal
        finite_gaps = [float(result.mip_gap) for result in chunk_results if np.isfinite(float(result.mip_gap))]
        model_sizes = [result.model_size for result in chunk_results]
        node_counts = np.asarray([float(result.node_count) for result in chunk_results], dtype=np.float64)
        iter_counts = np.asarray([float(result.iter_count) for result in chunk_results], dtype=np.float64)
        bar_iter_counts = np.asarray([float(result.bar_iter_count) for result in chunk_results], dtype=np.float64)

        def _finite_sum(values: np.ndarray) -> float:
            return float(np.sum(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float('nan')
        chunk_retry_count = int(sum((1 for item in list(chunk_summaries or []) if str(item.get('attempt_used', '')) == 'retry')))
        refinement_status_counter: Counter[str] = Counter()
        formulation_tightening_required = False
        for result in chunk_results:
            refinement_status_counter.update({str(key): int(value) for key, value in dict(getattr(result, 'refinement_status_counts', None) or {}).items()} or self._single_refinement_status_counts(str(result.physics_refinement_status)))
            formulation_tightening_required = formulation_tightening_required or bool(getattr(result, 'formulation_tightening_required', False))
        if all((str(result.physics_refinement_status) == 'floor_failed_gate' for result in chunk_results)):
            aggregate_refinement_status = 'floor_failed_gate'
        elif any((str(result.physics_refinement_status) == 'fallback_stage1' for result in chunk_results)):
            aggregate_refinement_status = 'partial_fallback_stage1'
        elif any((str(result.physics_refinement_status) == 'refined_time_budget_limited' for result in chunk_results)):
            if all((str(result.physics_refinement_status) == 'refined_time_budget_limited' for result in chunk_results)):
                aggregate_refinement_status = 'refined_time_budget_limited'
            else:
                aggregate_refinement_status = 'partial_refined_time_budget_limited'
        elif any((str(result.physics_refinement_status) == 'refined_cap_limited' for result in chunk_results)):
            if all((str(result.physics_refinement_status) == 'refined_cap_limited' for result in chunk_results)):
                aggregate_refinement_status = 'refined_cap_limited'
            else:
                aggregate_refinement_status = 'partial_refined_cap_limited'
        elif all((str(result.physics_refinement_status) == 'floor_accepted' for result in chunk_results)):
            aggregate_refinement_status = 'floor_accepted'
        elif all((str(result.physics_refinement_status) == 'refined' for result in chunk_results)):
            aggregate_refinement_status = 'refined'
        else:
            aggregate_refinement_status = 'refined'

        def _weighted_mean(attr_name: str) -> float:
            weighted_values: list[float] = []
            weights: list[float] = []
            for result in chunk_results:
                value = float(getattr(result, attr_name, float('nan')))
                if np.isfinite(value):
                    weighted_values.append(value * max(int(result.horizon_steps), 1))
                    weights.append(float(max(int(result.horizon_steps), 1)))
            if not weights:
                return float('nan')
            return float(sum(weighted_values) / sum(weights))

        def _max_finite(attr_name: str) -> float:
            values = [float(getattr(result, attr_name, float('nan'))) for result in chunk_results]
            finite = [value for value in values if np.isfinite(value)]
            return float(max(finite)) if finite else float('nan')
        used_tiers = [int(value) for value in (getattr(result, 'used_physics_refinement_tier', None) for result in chunk_results) if value is not None]
        floor_accepted_tiers = [int(value) for value in (getattr(result, 'floor_accepted_tier', None) for result in chunk_results) if value is not None]
        attempt_caps_union = sorted({float(value) for result in chunk_results for value in list(getattr(result, 'physics_refinement_attempt_caps_eur', None) or [])})
        returned_primary_objective_eur = float(sum((float(result.returned_primary_objective_eur) for result in chunk_results if np.isfinite(float(result.returned_primary_objective_eur))))) if any((np.isfinite(float(result.returned_primary_objective_eur)) for result in chunk_results)) else float('nan')
        stage1_primary_objective_eur = float(sum((float(result.stage1_primary_objective_eur) for result in chunk_results if np.isfinite(float(result.stage1_primary_objective_eur))))) if any((np.isfinite(float(result.stage1_primary_objective_eur)) for result in chunk_results)) else float('nan')
        returned_primary_delta_abs_eur = float(abs(returned_primary_objective_eur - stage1_primary_objective_eur)) if np.isfinite(returned_primary_objective_eur) and np.isfinite(stage1_primary_objective_eur) else float('nan')
        returned_primary_delta_pct = 100.0 * float(returned_primary_objective_eur - stage1_primary_objective_eur) / max(abs(float(stage1_primary_objective_eur)), 1.0) if np.isfinite(returned_primary_objective_eur) and np.isfinite(stage1_primary_objective_eur) else float('nan')
        return MISOCPResult(status_code=status_code, status_label=status_label, has_solution=has_solution, time_limit_feasible=time_limit_feasible, solve_time_sec=float(sum((float(result.solve_time_sec) for result in chunk_results))), mip_gap=float(max(finite_gaps)) if finite_gaps else float('nan'), best_bound=float('nan'), objective_value=float(sum((float(result.objective_value) for result in chunk_results))), agent_purchase_cost_eur=float(sum((float(result.agent_purchase_cost_eur) for result in chunk_results))), agent_export_subsidy_eur=float(sum((float(result.agent_export_subsidy_eur) for result in chunk_results))), agent_net_cost_eur=float(sum((float(result.agent_net_cost_eur) for result in chunk_results))), feeder_purchase_cost_eur=float(sum((float(result.feeder_purchase_cost_eur) for result in chunk_results))), feeder_export_subsidy_eur=float(sum((float(result.feeder_export_subsidy_eur) for result in chunk_results))), feeder_net_cost_eur=float(sum((float(result.feeder_net_cost_eur) for result in chunk_results))), throughput_regularization_eur=float(sum((float(result.throughput_regularization_eur) for result in chunk_results))), throughput_regularization_weight=float(reference.throughput_regularization_weight), physical_tiebreaker_eur=float(sum((float(result.physical_tiebreaker_eur) for result in chunk_results))), physical_tiebreaker_weight=float(reference.physical_tiebreaker_weight), model_size=ModelSize(num_vars=max((int(size.num_vars) for size in model_sizes)), num_binary_vars=max((int(size.num_binary_vars) for size in model_sizes)), num_linear_constraints=max((int(size.num_linear_constraints) for size in model_sizes)), num_quadratic_constraints=max((int(size.num_quadratic_constraints) for size in model_sizes))), debug_artifacts=debug_artifacts, agent_net_grid_mw=np.concatenate([result.agent_net_grid_mw for result in chunk_results], axis=1), agent_import_mw=np.concatenate([result.agent_import_mw for result in chunk_results], axis=1), agent_export_mw=np.concatenate([result.agent_export_mw for result in chunk_results], axis=1), battery_charge_mw=np.concatenate([result.battery_charge_mw for result in chunk_results], axis=1), battery_discharge_mw=np.concatenate([result.battery_discharge_mw for result in chunk_results], axis=1), pv_curtail_mw=np.concatenate([result.pv_curtail_mw for result in chunk_results], axis=1), energy_mwh=np.concatenate([chunk_results[0].energy_mwh, *[result.energy_mwh[:, 1:] for result in chunk_results[1:]]], axis=1), branch_p_pu=np.concatenate([result.branch_p_pu for result in chunk_results], axis=1), branch_q_pu=np.concatenate([result.branch_q_pu for result in chunk_results], axis=1), branch_i2_pu=np.concatenate([result.branch_i2_pu for result in chunk_results], axis=1), bus_v_sq=np.concatenate([result.bus_v_sq for result in chunk_results], axis=1), root_import_mw=np.concatenate([result.root_import_mw for result in chunk_results], axis=0), root_export_mw=np.concatenate([result.root_export_mw for result in chunk_results], axis=0), root_p_kw=np.concatenate([result.root_p_kw for result in chunk_results], axis=0), root_q_kvar=np.concatenate([result.root_q_kvar for result in chunk_results], axis=0), bus_vm_pu=np.concatenate([result.bus_vm_pu for result in chunk_results], axis=1), line_loading_pct=np.concatenate([result.line_loading_pct for result in chunk_results], axis=1), trafo_loading_pct=np.concatenate([result.trafo_loading_pct for result in chunk_results], axis=1), simultaneous_charge_discharge_kw=np.concatenate([result.simultaneous_charge_discharge_kw for result in chunk_results], axis=1), simultaneous_agent_steps=int(sum((int(result.simultaneous_agent_steps) for result in chunk_results))), simultaneous_step_ratio=float(np.mean(np.any(np.concatenate([result.simultaneous_charge_discharge_kw for result in chunk_results], axis=1) > 0.0, axis=0))), max_simultaneous_kw=float(max((float(result.max_simultaneous_kw) for result in chunk_results))), sol_count=int(sum((int(result.sol_count) for result in chunk_results))), node_count=_finite_sum(node_counts), iter_count=_finite_sum(iter_counts), bar_iter_count=_finite_sum(bar_iter_counts), sanity_warning=str(reference.sanity_warning), horizon_steps=int(np.sum(episode_lengths)), solve_mode=str(solve_mode), episode_offsets=np.asarray(episode_offsets, dtype=np.int32), episode_lengths=np.asarray(episode_lengths, dtype=np.int32), chunk_summaries=None if chunk_summaries is None else [dict(item) for item in chunk_summaries], no_retry_or_fallback_used=bool(chunk_retry_count == 0), chunk_retry_count=chunk_retry_count, stage1_primary_objective_eur=stage1_primary_objective_eur, stage2_primary_objective_eur=float(sum((float(result.stage2_primary_objective_eur) for result in chunk_results if np.isfinite(float(result.stage2_primary_objective_eur))))) if any((np.isfinite(float(result.stage2_primary_objective_eur)) for result in chunk_results)) else float('nan'), stage2_objective_slack_eur=float(sum((float(result.stage2_objective_slack_eur) for result in chunk_results if np.isfinite(float(result.stage2_objective_slack_eur))))) if any((np.isfinite(float(result.stage2_objective_slack_eur)) for result in chunk_results)) else float('nan'), stage2_branch_l_objective=float(sum((float(result.stage2_branch_l_objective) for result in chunk_results if np.isfinite(float(result.stage2_branch_l_objective))))) if any((np.isfinite(float(result.stage2_branch_l_objective)) for result in chunk_results)) else float('nan'), physics_refinement_mode=str(reference.physics_refinement_mode), physics_refinement_status=str(aggregate_refinement_status), physics_refinement_runtime_sec=float(sum((float(result.physics_refinement_runtime_sec) for result in chunk_results if np.isfinite(float(result.physics_refinement_runtime_sec))))), floor_p95_soc_slack=float(max((float(result.floor_p95_soc_slack) for result in chunk_results if np.isfinite(float(result.floor_p95_soc_slack))))) if any((np.isfinite(float(result.floor_p95_soc_slack)) for result in chunk_results)) else float('nan'), floor_mean_abs_solver_feeder_gap_kw=float(np.mean([float(result.floor_mean_abs_solver_feeder_gap_kw) for result in chunk_results if np.isfinite(float(result.floor_mean_abs_solver_feeder_gap_kw))])) if any((np.isfinite(float(result.floor_mean_abs_solver_feeder_gap_kw)) for result in chunk_results)) else float('nan'), floor_primary_objective_eur=float(sum((float(result.floor_primary_objective_eur) for result in chunk_results if np.isfinite(float(result.floor_primary_objective_eur))))) if any((np.isfinite(float(result.floor_primary_objective_eur)) for result in chunk_results)) else float('nan'), floor_primary_delta_signed_eur=float(sum((float(result.floor_primary_delta_signed_eur) for result in chunk_results if np.isfinite(float(result.floor_primary_delta_signed_eur))))) if any((np.isfinite(float(result.floor_primary_delta_signed_eur)) for result in chunk_results)) else float('nan'), floor_primary_delta_positive_eur=float(sum((float(result.floor_primary_delta_positive_eur) for result in chunk_results if np.isfinite(float(result.floor_primary_delta_positive_eur))))) if any((np.isfinite(float(result.floor_primary_delta_positive_eur)) for result in chunk_results)) else float('nan'), physics_refinement_slack_cap_eur=float(sum((float(result.physics_refinement_slack_cap_eur) for result in chunk_results if np.isfinite(float(result.physics_refinement_slack_cap_eur))))) if any((np.isfinite(float(result.physics_refinement_slack_cap_eur)) for result in chunk_results)) else float('nan'), initial_physics_refinement_slack_cap_eur=float(max((float(result.initial_physics_refinement_slack_cap_eur) for result in chunk_results if np.isfinite(float(result.initial_physics_refinement_slack_cap_eur))), default=float('nan'))), returned_primary_objective_eur=returned_primary_objective_eur, returned_primary_delta_abs_eur=returned_primary_delta_abs_eur, returned_primary_delta_pct=float(returned_primary_delta_pct), floor_accepted_tier=max(floor_accepted_tiers) if floor_accepted_tiers else None, used_physics_refinement_tier=max(used_tiers) if used_tiers else None, total_tiers_configured=max((int(getattr(result, 'total_tiers_configured', 0)) for result in chunk_results)), physics_refinement_attempt_count=int(sum((int(getattr(result, 'physics_refinement_attempt_count', 0)) for result in chunk_results))), physics_refinement_attempt_caps_eur=attempt_caps_union, physics_refinement_cap_utilization=_max_finite('physics_refinement_cap_utilization'), branch_l_gap_ratio_to_floor=_max_finite('branch_l_gap_ratio_to_floor'), returned_mean_abs_solver_feeder_gap_kw=_weighted_mean('returned_mean_abs_solver_feeder_gap_kw'), returned_max_solver_feeder_gap_kw=_max_finite('returned_max_solver_feeder_gap_kw'), returned_mean_abs_export_gap_ratio=_weighted_mean('returned_mean_abs_export_gap_ratio'), high_budget_refinement_warn=bool(any((bool(getattr(result, 'high_budget_refinement_warn', False)) for result in chunk_results))), returned_solution_source='floor' if all((str(result.returned_solution_source) == 'floor' for result in chunk_results)) else 'stage2' if all((str(result.returned_solution_source) == 'stage2' for result in chunk_results)) else 'mixed', formulation_tightening_required=bool(formulation_tightening_required), negative_floor_delta_warn=bool(any((bool(getattr(result, 'negative_floor_delta_warn', False)) for result in chunk_results))), refinement_status_counts=dict(refinement_status_counter))

    def run_transformer_sanity_check(self) -> dict[str, object]:
        zero_price = np.zeros((self.horizon,), dtype=np.float32)
        zero_load = np.zeros((self.n_agents, self.horizon), dtype=np.float32)
        zero_pv = np.zeros((self.n_agents, self.horizon), dtype=np.float32)
        zero_soc = np.zeros((self.n_agents,), dtype=np.float32)
        original_target = self.soc_target
        original_p_max = self.p_max_mw.copy()
        original_p_base = self.network.p_base_mw.copy()
        original_q_base = self.network.q_base_mvar.copy()
        try:
            self.soc_target = 0.0
            self.p_max_mw[:] = 0.0
            self.network.p_base_mw[:] = 0.0
            self.network.q_base_mvar[:] = 0.0
            result = self.solve(zero_price, zero_load, zero_pv, zero_soc, export_subsidy=0.0, verbose=False)
        finally:
            self.soc_target = original_target
            self.p_max_mw[:] = original_p_max
            self.network.p_base_mw[:] = original_p_base
            self.network.q_base_mvar[:] = original_q_base
        return {'status': str(result.status_label), 'within_tolerance': False, 'abs_error_pu': np.nan, 'misocp_vm_pu': float('nan') if result.first_step_vm_pu is None else float(result.first_step_vm_pu[int(self.network.bus_pos[self.network.trafo_lv_bus_id])]), 'pandapower_vm_pu': float('nan'), 'warning': 'Use notebook or rollout validation for detailed pandapower comparison.'}

    def _line_loading_pct_from_solution(self, branch_i2_pu: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for branch_idx in self.network.line_branch_indices.tolist():
            l_series = np.maximum(branch_i2_pu[int(branch_idx), :], 0.0)
            l_max = float(self.network.branch_l_max_pu[int(branch_idx)])
            if not np.isfinite(l_max) or l_max <= 0.0:
                values.append(np.zeros_like(l_series, dtype=np.float32))
                continue
            values.append((100.0 * np.sqrt(np.maximum(l_series / l_max, 0.0))).astype(np.float32))
        if not values:
            return np.zeros((0, branch_i2_pu.shape[1]), dtype=np.float32)
        return np.stack(values, axis=0).astype(np.float32)

    def _simultaneous_metrics(self, battery_charge_mw: np.ndarray, battery_discharge_mw: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        simultaneous_kw = np.minimum(battery_charge_mw, battery_discharge_mw) * 1000.0
        threshold_kw = (self.simultaneous_threshold_ratio * self.p_max_mw * 1000.0).reshape(-1, 1)
        simultaneous_mask = simultaneous_kw > threshold_kw
        simultaneous_agent_steps = int(np.sum(simultaneous_mask))
        simultaneous_step_ratio = float(np.mean(np.any(simultaneous_mask, axis=0))) if simultaneous_mask.shape[1] else 0.0
        max_simultaneous_kw = float(np.max(simultaneous_kw)) if simultaneous_kw.size else 0.0
        return (simultaneous_kw.astype(np.float32), simultaneous_agent_steps, simultaneous_step_ratio, max_simultaneous_kw)

    def _maybe_export_debug_artifacts(self, model: Any, *, status_code: int, export_debug: bool, debug_tag: str | None) -> dict[str, str]:
        if not export_debug:
            return {}
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        tag = _sanitize_debug_tag(debug_tag or _status_label(status_code))
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = self.debug_dir / f'{stamp}_{tag}'
        output: dict[str, str] = {}
        lp_path = stem.with_suffix('.lp')
        try:
            model.write(str(lp_path))
            output['lp'] = str(lp_path)
        except Exception:
            pass
        if gp is not None and GRB is not None and (int(status_code) == int(GRB.INFEASIBLE)):
            ilp_path = stem.with_suffix('.ilp')
            try:
                model.computeIIS()
                model.write(str(ilp_path))
                output['iis'] = str(ilp_path)
            except Exception:
                pass
        return output

class GlobalSOCPMPCController(BaseController):

    def __init__(self, env: Any, cfg: Any, *, problem: GlobalMISOCPProblem | None=None, solve_verbose: bool=False, export_debug: bool=False, debug_tag_prefix: str='global_misocp') -> None:
        if gp is None or GRB is None:
            raise RuntimeError(f'{_GUROBI_ERROR_PREFIX}: gurobipy import failed')
        self.env = env
        self.cfg = cfg
        self.apply_action_penalty = False
        self.last_action_info: dict[str, np.ndarray] | None = None
        self.last_diagnostic: dict[str, object] | None = None
        self.diagnostic_log: list[dict[str, object]] = []
        self.problem = problem or GlobalMISOCPProblem.from_env(env, cfg)
        self.network = self.problem.network
        self.n_agents = self.problem.n_agents
        self.n_buses = self.problem.n_buses
        self.n_branches = self.problem.n_branches
        self.horizon = self.problem.horizon
        self.soc_min = float(self.problem.soc_min)
        self.soc_max = float(self.problem.soc_max)
        self.solve_verbose = bool(solve_verbose)
        self.export_debug = bool(export_debug)
        self.debug_tag_prefix = str(debug_tag_prefix)
        self._solve_counter = 0
        self.sanity_warning = ''

    def reset(self) -> None:
        self.last_action_info = None
        self.last_diagnostic = None
        self.diagnostic_log = []
        self._solve_counter = 0

    def act(self, obs: dict, deterministic: bool=True) -> list[np.ndarray]:
        del deterministic
        raw_obs = self.env.obs_builder.build_raw(self.env) if hasattr(self.env.obs_builder, 'build_raw') else obs
        result = self._solve_step(raw_obs)
        if result.has_solution and result.first_step_battery_power_kw is not None and (result.first_step_pv_curtail_kw is not None):
            battery_power_kw = self._clip_local_battery_power_kw(result.first_step_battery_power_kw)
            pv_curtail_kw = np.minimum(np.maximum(result.first_step_pv_curtail_kw, 0.0), np.maximum(np.asarray(self.env.get_signal_step('pv'), dtype=np.float32), 0.0)).astype(np.float32)
        else:
            battery_power_kw = np.zeros((self.n_agents,), dtype=np.float32)
            pv_curtail_kw = np.zeros((self.n_agents,), dtype=np.float32)
        actions = self._assemble_actions(battery_power_kw, pv_curtail_kw)
        action_array = np.asarray(actions, dtype=np.float32)
        safety_local = self.env._current_safety_local() if hasattr(self.env, '_current_safety_local') else build_safety_local_numpy(soc=np.asarray(self.env.soc, dtype=np.float32), load_raw=np.asarray(self.env.get_signal_step('load'), dtype=np.float32), pv_raw=np.asarray(self.env.get_signal_step('pv'), dtype=np.float32), battery_capacity_kwh=np.asarray(self.env.agent_c_bat, dtype=np.float32), p_max_kw=np.asarray(self.env.agent_p_max, dtype=np.float32))
        action_info = compute_action_gap_metrics_numpy(safety_local, action_array, action_array)
        action_info.update({'misocp_fallback': np.asarray(float(not result.has_solution), dtype=np.float32), 'misocp_time_limit_feasible': np.asarray(float(result.time_limit_feasible), dtype=np.float32), 'solve_time_sec': np.asarray(result.solve_time_sec, dtype=np.float32), 'root_import_kw': np.asarray(result.first_step_root_import_kw, dtype=np.float32), 'root_export_kw': np.asarray(result.first_step_root_export_kw, dtype=np.float32), 'simultaneous_charge_discharge_kw_total': np.asarray(result.first_step_simultaneous_kw_total, dtype=np.float32), 'simultaneous_agent_count': np.asarray(float(result.first_step_simultaneous_agent_count), dtype=np.float32), 'simultaneous_step_flag': np.asarray(float(result.first_step_simultaneous_agent_count > 0), dtype=np.float32)})
        self.last_action_info = action_info
        diagnostic = {'solver_type': 'gurobi_misocp', 'status_code': int(result.status_code), 'status_label': str(result.status_label), 'solve_time_sec': float(result.solve_time_sec), 'misocp_fallback': float(not result.has_solution), 'misocp_time_limit_feasible': float(result.time_limit_feasible), 'mip_gap': float(result.mip_gap), 'best_bound': float(result.best_bound), 'objective_value': float(result.objective_value), 'agent_purchase_cost_eur': float(result.agent_purchase_cost_eur), 'agent_export_subsidy_eur': float(result.agent_export_subsidy_eur), 'agent_net_cost_eur': float(result.agent_net_cost_eur), 'feeder_purchase_cost_eur': float(result.feeder_purchase_cost_eur), 'feeder_export_subsidy_eur': float(result.feeder_export_subsidy_eur), 'feeder_net_cost_eur': float(result.feeder_net_cost_eur), 'throughput_regularization_eur': float(result.throughput_regularization_eur), 'throughput_regularization_weight': float(result.throughput_regularization_weight), 'root_import_kw': float(result.first_step_root_import_kw), 'root_export_kw': float(result.first_step_root_export_kw), 'root_p_kw': float(result.first_step_root_p_kw), 'root_q_kvar': float(result.first_step_root_q_kvar), 'sol_count': int(result.sol_count), 'node_count': float(result.node_count), 'iter_count': float(result.iter_count), 'bar_iter_count': float(result.bar_iter_count), 'battery_charge_kw': None if result.first_step_battery_charge_kw is None else result.first_step_battery_charge_kw.copy(), 'battery_discharge_kw': None if result.first_step_battery_discharge_kw is None else result.first_step_battery_discharge_kw.copy(), 'misocp_vm_pu': None if result.first_step_vm_pu is None else result.first_step_vm_pu.copy(), 'misocp_line_loading_pct': None if result.first_step_line_loading_pct is None else result.first_step_line_loading_pct.copy(), 'misocp_trafo_loading_pct': None if result.first_step_trafo_loading_pct is None else result.first_step_trafo_loading_pct.copy(), 'model_size_num_vars': float(result.model_size.num_vars), 'model_size_num_binary_vars': float(result.model_size.num_binary_vars), 'model_size_num_linear_constraints': float(result.model_size.num_linear_constraints), 'model_size_num_quadratic_constraints': float(result.model_size.num_quadratic_constraints), 'simultaneous_step_ratio': float(result.simultaneous_step_ratio), 'max_simultaneous_kw': float(result.max_simultaneous_kw), 'simultaneous_agent_steps': float(result.simultaneous_agent_steps), 'debug_artifacts': dict(result.debug_artifacts), 'sanity_warning': str(result.sanity_warning or self.sanity_warning or '')}
        self.last_diagnostic = diagnostic
        self.diagnostic_log.append(diagnostic)
        return actions

    def _solve_step(self, raw_obs: dict[str, np.ndarray]) -> MISOCPResult:
        debug_tag = f'{self.debug_tag_prefix}_step{self._solve_counter:04d}' if self.export_debug else None
        self._solve_counter += 1
        return self.problem.solve(import_price_seq=self.problem.apply_import_price_markup(np.asarray(raw_obs[WHOLESALE_PRICE_SEQ_FIELD], dtype=np.float32)), load_seq=np.asarray(raw_obs['load_seq'], dtype=np.float32), pv_seq=np.asarray(raw_obs['pv_seq'], dtype=np.float32), soc_init=np.asarray(self.env.soc, dtype=np.float32), export_subsidy=float(getattr(self.cfg.reward, 'export_subsidy_eur_per_kwh', self.problem.export_subsidy_default)), verbose=self.solve_verbose, export_debug=self.export_debug, debug_tag=debug_tag)

    def _clip_local_battery_power_kw(self, battery_power_kw: np.ndarray) -> np.ndarray:
        battery_power_kw = np.asarray(battery_power_kw, dtype=np.float32)
        e_t = np.asarray(self.env.soc, dtype=np.float32) * np.asarray(self.env.agent_c_bat, dtype=np.float32)
        e_min = self.soc_min * np.asarray(self.env.agent_c_bat, dtype=np.float32)
        e_max = self.soc_max * np.asarray(self.env.agent_c_bat, dtype=np.float32)
        p_max = np.asarray(self.env.agent_p_max, dtype=np.float32)
        eff = max(float(self.env.eff), _ROOT_VM_EPS)
        p_max_charge = np.minimum(p_max, np.maximum(0.0, (e_max - e_t) / (eff * float(self.env.dt))))
        p_max_discharge = np.minimum(p_max, np.maximum(0.0, (e_t - e_min) * eff / float(self.env.dt)))
        return np.clip(battery_power_kw, -p_max_discharge, p_max_charge).astype(np.float32)

    def _assemble_actions(self, battery_power_kw: np.ndarray, pv_curtail_kw: np.ndarray) -> list[np.ndarray]:
        battery_power_kw = np.asarray(battery_power_kw, dtype=np.float32)
        pv_curtail_kw = np.asarray(pv_curtail_kw, dtype=np.float32)
        p_max_kw = np.maximum(np.asarray(self.env.agent_p_max, dtype=np.float32), _PV_EPS_KW)
        pv_raw_kw = np.maximum(np.asarray(self.env.get_signal_step('pv'), dtype=np.float32), 0.0)
        battery_action = np.clip(battery_power_kw / p_max_kw, -1.0, 1.0).astype(np.float32)
        pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
        valid_mask = pv_raw_kw > _PV_EPS_KW
        pv_utilization[valid_mask] = 1.0 - pv_curtail_kw[valid_mask] / pv_raw_kw[valid_mask]
        pv_action = np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0).astype(np.float32)
        action_array = np.stack([battery_action, pv_action], axis=-1).astype(np.float32)
        return [action_array[agent_idx].copy() for agent_idx in range(self.n_agents)]

def _status_label(status_code: int) -> str:
    if GRB is None:
        mapping = {2: 'optimal', 3: 'infeasible', 4: 'inf_or_unbd', 9: 'time_limit'}
        return mapping.get(int(status_code), f'status_{int(status_code)}')
    mapping = {int(GRB.OPTIMAL): 'optimal', int(GRB.TIME_LIMIT): 'time_limit', int(GRB.INFEASIBLE): 'infeasible', int(GRB.INF_OR_UNBD): 'inf_or_unbd'}
    return mapping.get(int(status_code), f'status_{int(status_code)}')

def _extract_grid_values(var_grid: Any, n_rows: int, n_cols: int) -> np.ndarray:
    return np.asarray([[float(var_grid[row_idx, col_idx].X) for col_idx in range(n_cols)] for row_idx in range(n_rows)], dtype=np.float32)

def _extract_vector_values(var_vector: Any, n_items: int) -> np.ndarray:
    return np.asarray([float(var_vector[idx].X) for idx in range(n_items)], dtype=np.float32)

def _compute_agent_net_grid_mw(load_seq_kw: np.ndarray, pv_raw_seq_kw: np.ndarray, battery_charge_mw: np.ndarray, battery_discharge_mw: np.ndarray, pv_curtail_mw: np.ndarray) -> np.ndarray:
    load_seq_kw = np.asarray(load_seq_kw, dtype=np.float32)
    pv_raw_seq_kw = np.asarray(pv_raw_seq_kw, dtype=np.float32)
    battery_charge_mw = np.asarray(battery_charge_mw, dtype=np.float32)
    battery_discharge_mw = np.asarray(battery_discharge_mw, dtype=np.float32)
    pv_curtail_mw = np.asarray(pv_curtail_mw, dtype=np.float32)
    return ((load_seq_kw - pv_raw_seq_kw) / 1000.0 + battery_charge_mw - battery_discharge_mw + pv_curtail_mw).astype(np.float32)

def _sanitize_debug_tag(tag: str) -> str:
    sanitized = ''.join((ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(tag)))
    return sanitized.strip('_') or 'misocp_debug'
