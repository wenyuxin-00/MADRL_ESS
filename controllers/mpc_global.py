from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from configs.cfg import Cfg
from envs.grid_core import PowerFlowGridCore
from envs.grid_env import action_array_from_power
from utils.price import derive_import_price_seq, get_import_price_markup


@dataclass(frozen=True)
class MisocpNetwork:
    s_base_mva: float
    root_bus_id: int
    root_vm_sq: float
    loading_scale: float
    bus_ids: tuple[int, ...]
    bus_pos: dict[int, int]
    branch_parent: np.ndarray
    branch_child: np.ndarray
    branch_r_pu: np.ndarray
    branch_x_pu: np.ndarray
    branch_l_max_pu: np.ndarray
    branch_is_trafo: np.ndarray
    outgoing: tuple[tuple[int, ...], ...]
    root_branches: np.ndarray
    agent_bus_pos: np.ndarray


@dataclass(frozen=True)
class MisocpSolveResult:
    battery_power_kw: np.ndarray
    pv_curtail_kw: np.ndarray
    ev_charge_kw: np.ndarray
    solve_time_sec: float
    objective_eur: float
    status: int


def _build_tree(bus_ids: tuple[int, ...], root_bus_id: int, edges: list[tuple[int, int, float, float, float, bool]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    bus_pos = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    adjacency: dict[int, list[tuple[int, int]]] = {bus_id: [] for bus_id in bus_ids}
    for edge_idx, (u, v, *_rest) in enumerate(edges):
        adjacency[int(u)].append((int(v), edge_idx)); adjacency[int(v)].append((int(u), edge_idx))
    visited, queue = {int(root_bus_id)}, [int(root_bus_id)]
    parent: list[int] = []; child: list[int] = []; r: list[float] = []; x: list[float] = []; lmax: list[float] = []; is_trafo: list[bool] = []; outgoing = [[] for _ in bus_ids]
    while queue:
        bus = queue.pop(0)
        for next_bus, edge_idx in adjacency[bus]:
            if next_bus in visited:
                continue
            visited.add(next_bus); queue.append(next_bus)
            u, v, r_pu, x_pu, limit_pu, trafo_flag = edges[edge_idx]
            branch_idx = len(parent); parent.append(bus_pos[bus]); child.append(bus_pos[next_bus])
            r.append(float(r_pu)); x.append(float(x_pu)); lmax.append(float(limit_pu)); is_trafo.append(bool(trafo_flag)); outgoing[bus_pos[bus]].append(branch_idx)
    return (np.asarray(parent, dtype=np.int32), np.asarray(child, dtype=np.int32), np.asarray(r, dtype=np.float32), np.asarray(x, dtype=np.float32), np.asarray(lmax, dtype=np.float32), np.asarray(is_trafo, dtype=bool), tuple(tuple(v) for v in outgoing))


def build_misocp_network(cfg: Cfg) -> MisocpNetwork:
    core = PowerFlowGridCore(cfg); net = core.net
    root_bus = int(net.ext_grid.iloc[0]["bus"]); root_vm = float(net.ext_grid.iloc[0]["vm_pu"])
    trafo = net.trafo.iloc[0]; s_base = float(trafo["sn_mva"])
    scale = float(trafo["max_loading_percent"]) / 100.0
    hv, lv = int(trafo["hv_bus"]), int(trafo["lv_bus"])
    vk, vkr = float(trafo["vk_percent"]), float(trafo["vkr_percent"])
    edges: list[tuple[int, int, float, float, float, bool]] = [(hv, lv, vkr / 100.0, float(np.sqrt(max(vk * vk - vkr * vkr, 0.0))) / 100.0, np.inf, True)]
    for _, row in net.line.iterrows():
        u, v = int(row["from_bus"]), int(row["to_bus"])
        vn = float(net.bus.at[u, "vn_kv"]); z_base = vn * vn / s_base
        length = float(row["length_km"]); parallel = max(float(row["parallel"] or 1.0), 1.0); df = max(float(row["df"] or 1.0), 0.0)
        r_ohm = float(row["r_ohm_per_km"]) * length / parallel; x_ohm = float(row["x_ohm_per_km"]) * length / parallel
        i_base = s_base / (np.sqrt(3.0) * vn); max_i = float(row["max_i_ka"]) * parallel * df
        edges.append((u, v, r_ohm / z_base, x_ohm / z_base, np.inf if max_i <= 0.0 else (max_i / i_base) ** 2, False))
    bus_ids = tuple(int(x) for x in net.bus.index.tolist()); bus_pos = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    parent, child, r_pu, x_pu, lmax_pu, is_trafo, outgoing = _build_tree(bus_ids, root_bus, edges)
    root_branches = np.asarray(outgoing[bus_pos[root_bus]], dtype=np.int32)
    return MisocpNetwork(s_base, root_bus, root_vm * root_vm, scale, bus_ids, bus_pos, parent, child, r_pu, x_pu, lmax_pu, is_trafo, outgoing, root_branches, np.asarray([bus_pos[int(x)] for x in cfg.grid.agent_bus_ids], dtype=np.int32))


class GlobalMisocpSolver:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg; self.network = build_misocp_network(cfg)

    def solve(self, *, import_price_seq: np.ndarray, load_seq: np.ndarray, pv_seq: np.ndarray, soc_init: np.ndarray, ev_soc_init: np.ndarray | None = None) -> MisocpSolveResult:
        import gurobipy as gp

        grb = gp.GRB
        price = np.asarray(import_price_seq, dtype=np.float32).reshape(-1)
        load = np.asarray(load_seq, dtype=np.float32); pv = np.asarray(pv_seq, dtype=np.float32)
        n, h = int(self.cfg.env.num_agents), int(price.size)
        cap = np.asarray(self.cfg.env.battery_capacity_kwh, dtype=np.float32).reshape(n)
        pmax = cap * np.float32(self.cfg.env.max_charge_rate)
        ev_enabled = bool(self.cfg.env.ev_enabled) and int(self.cfg.model.action_dim) >= 3
        ev_cap = np.asarray(self.cfg.env.ev_capacity_kwh, dtype=np.float32).reshape(n)
        ev_pmax = np.asarray(self.cfg.env.ev_max_charge_kw, dtype=np.float32).reshape(n)
        ev_init = np.full((n,), float(self.cfg.env.ev_arrival_soc), dtype=np.float32) if ev_soc_init is None else np.asarray(ev_soc_init, dtype=np.float32).reshape(n)
        model = gp.Model("remake_global_misocp"); model.Params.OutputFlag = 0; model.Params.MIPGap = 1e-3
        charge = model.addVars(n, h, lb=0.0, name="charge_kw"); discharge = model.addVars(n, h, lb=0.0, name="discharge_kw")
        curtail = model.addVars(n, h, lb=0.0, name="pv_curtail_kw"); energy = model.addVars(n, h + 1, lb=0.0, name="energy_kwh")
        ev_charge = model.addVars(n, h, lb=0.0, name="ev_charge_kw")
        ev_energy = model.addVars(n, h + 1, lb=0.0, name="ev_energy_kwh")
        ev_dep_gap = model.addVars(n, lb=0.0, name="ev_departure_gap_soc")
        branch_count, bus_count = int(self.network.branch_parent.size), int(len(self.network.bus_ids))
        branch_p = model.addVars(branch_count, h, lb=-grb.INFINITY, name="branch_p_pu"); branch_q = model.addVars(branch_count, h, lb=-grb.INFINITY, name="branch_q_pu")
        branch_l = model.addVars(branch_count, h, lb=0.0, name="branch_i2_pu"); bus_v = model.addVars(bus_count, h, lb=float(self.cfg.grid.v_min_pu) ** 2, ub=float(self.cfg.grid.v_max_pu) ** 2, name="bus_v_sq")
        root_import = model.addVars(h, lb=0.0, ub=self.network.s_base_mva * self.network.loading_scale, name="root_import_mw")
        root_export = model.addVars(h, lb=0.0, ub=self.network.s_base_mva * self.network.loading_scale, name="root_export_mw")
        root_mode = model.addVars(h, vtype=grb.BINARY, name="root_mode")
        objective = gp.QuadExpr(); dt, eff = float(self.cfg.env.dt_hours), float(self.cfg.env.efficiency)
        ev_eff = float(self.cfg.env.ev_efficiency)
        episode_steps = int(self.cfg.env.episode_steps)
        arr, dep = int(self.cfg.env.ev_arrival_step), int(self.cfg.env.ev_departure_step)
        ev_mode = str(getattr(self.cfg.env, "ev_departure_constraint_mode", "soft")).lower()
        def ev_available(step_idx: int) -> bool:
            daily = int(step_idx) % episode_steps
            return bool(daily >= arr or daily < dep) if arr > dep else bool(arr <= daily < dep)
        for agent in range(n):
            model.addConstr(energy[agent, 0] == float(soc_init[agent] * cap[agent]), name=f"energy_init_{agent}")
            model.addConstr(ev_energy[agent, 0] == float(ev_init[agent] * ev_cap[agent]), name=f"ev_energy_init_{agent}")
            for step in range(h + 1):
                model.addConstr(energy[agent, step] >= float(self.cfg.env.soc_min * cap[agent]), name=f"energy_lb_{agent}_{step}")
                model.addConstr(energy[agent, step] <= float(self.cfg.env.soc_max * cap[agent]), name=f"energy_ub_{agent}_{step}")
                model.addConstr(ev_energy[agent, step] >= float(self.cfg.env.ev_soc_min * ev_cap[agent]), name=f"ev_energy_lb_{agent}_{step}")
                model.addConstr(ev_energy[agent, step] <= float(self.cfg.env.ev_soc_max * ev_cap[agent]), name=f"ev_energy_ub_{agent}_{step}")
            for step in range(h):
                model.addConstr(charge[agent, step] <= float(pmax[agent]), name=f"charge_ub_{agent}_{step}")
                model.addConstr(discharge[agent, step] <= float(pmax[agent]), name=f"discharge_ub_{agent}_{step}")
                model.addConstr(curtail[agent, step] <= float(max(pv[agent, step], 0.0)), name=f"curtail_ub_{agent}_{step}")
                model.addConstr(energy[agent, step + 1] == energy[agent, step] + eff * dt * charge[agent, step] - dt / eff * discharge[agent, step], name=f"energy_balance_{agent}_{step}")
                if ev_enabled and ev_available(step):
                    model.addConstr(ev_charge[agent, step] <= float(ev_pmax[agent]), name=f"ev_charge_ub_{agent}_{step}")
                else:
                    model.addConstr(ev_charge[agent, step] == 0.0, name=f"ev_charge_off_{agent}_{step}")
                if ((step + 1) % episode_steps) == arr:
                    model.addConstr(ev_energy[agent, step + 1] == float(self.cfg.env.ev_arrival_soc * ev_cap[agent]), name=f"ev_arrival_reset_{agent}_{step}")
                else:
                    model.addConstr(ev_energy[agent, step + 1] == ev_energy[agent, step] + ev_eff * dt * ev_charge[agent, step], name=f"ev_energy_balance_{agent}_{step}")
                objective += dt * float(price[step]) * (charge[agent, step] - discharge[agent, step])
                objective += dt * float(price[step]) * ev_charge[agent, step]
                objective += dt * 1e-4 * (charge[agent, step] + discharge[agent, step])
            dep_indices = [idx for idx in range(h + 1) if idx % episode_steps == dep]
            if ev_enabled and dep_indices:
                dep_idx = int(dep_indices[0])
                if ev_mode == "soft":
                    model.addConstr(ev_dep_gap[agent] >= float(self.cfg.env.ev_departure_soc_req) - ev_energy[agent, dep_idx] / float(ev_cap[agent]), name=f"ev_departure_gap_{agent}")
                    objective += float(self.cfg.reward.ev_departure_penalty_weight) * ev_dep_gap[agent] * ev_dep_gap[agent]
                elif ev_mode in {"hard", "emergency"}:
                    # Emergency MISOCP is a global baseline: it enforces the same departure requirement directly instead of relying on myopic emergency correction.
                    model.addConstr(ev_energy[agent, dep_idx] >= float(self.cfg.env.ev_departure_soc_req * ev_cap[agent]), name=f"ev_departure_hard_{agent}")
        scale_sq = self.network.loading_scale * self.network.loading_scale
        for step in range(h):
            model.addConstr(root_import[step] <= self.network.s_base_mva * self.network.loading_scale * root_mode[step], name=f"root_import_gate_{step}")
            model.addConstr(root_export[step] <= self.network.s_base_mva * self.network.loading_scale * (1.0 - root_mode[step]), name=f"root_export_gate_{step}")
            model.addConstr(bus_v[self.network.bus_pos[self.network.root_bus_id], step] == self.network.root_vm_sq, name=f"root_v_{step}")
            for branch in range(branch_count):
                parent, child = int(self.network.branch_parent[branch]), int(self.network.branch_child[branch])
                downstream_p = gp.quicksum(branch_p[idx, step] for idx in self.network.outgoing[child])
                downstream_q = gp.quicksum(branch_q[idx, step] for idx in self.network.outgoing[child])
                agent_expr = gp.LinExpr()
                for agent in range(n):
                    if int(self.network.agent_bus_pos[agent]) == child:
                        agent_expr += (float(load[agent, step] - pv[agent, step]) + curtail[agent, step] + charge[agent, step] - discharge[agent, step] + ev_charge[agent, step]) / (1000.0 * self.network.s_base_mva)
                r_pu, x_pu = float(self.network.branch_r_pu[branch]), float(self.network.branch_x_pu[branch])
                model.addConstr(branch_p[branch, step] == downstream_p + r_pu * branch_l[branch, step] + agent_expr, name=f"p_balance_{branch}_{step}")
                model.addConstr(branch_q[branch, step] == downstream_q + x_pu * branch_l[branch, step], name=f"q_balance_{branch}_{step}")
                model.addConstr(bus_v[child, step] == bus_v[parent, step] - 2.0 * (r_pu * branch_p[branch, step] + x_pu * branch_q[branch, step]) + (r_pu * r_pu + x_pu * x_pu) * branch_l[branch, step], name=f"v_drop_{branch}_{step}")
                model.addQConstr((bus_v[parent, step] + branch_l[branch, step]) * (bus_v[parent, step] + branch_l[branch, step]) >= 4.0 * branch_p[branch, step] * branch_p[branch, step] + 4.0 * branch_q[branch, step] * branch_q[branch, step] + (bus_v[parent, step] - branch_l[branch, step]) * (bus_v[parent, step] - branch_l[branch, step]), name=f"soc_{branch}_{step}")
                if bool(self.network.branch_is_trafo[branch]):
                    model.addConstr(branch_l[branch, step] <= scale_sq / max(float(self.cfg.grid.v_min_pu) ** 2, 1e-6), name=f"trafo_i2_{branch}_{step}")
                elif np.isfinite(float(self.network.branch_l_max_pu[branch])):
                    model.addConstr(branch_l[branch, step] <= float(self.network.branch_l_max_pu[branch]) * scale_sq, name=f"line_i2_{branch}_{step}")
            root_p = gp.quicksum(branch_p[int(idx), step] for idx in self.network.root_branches)
            root_q = gp.quicksum(branch_q[int(idx), step] for idx in self.network.root_branches)
            model.addConstr(self.network.s_base_mva * root_p == root_import[step] - root_export[step], name=f"root_split_{step}")
            model.addQConstr(root_p * root_p + root_q * root_q <= scale_sq, name=f"trafo_s_{step}")
        model.setObjective(objective, grb.MINIMIZE); started = perf_counter(); model.optimize()
        battery = np.asarray([[charge[a, t].X - discharge[a, t].X for t in range(h)] for a in range(n)], dtype=np.float32)
        curtailed = np.asarray([[curtail[a, t].X for t in range(h)] for a in range(n)], dtype=np.float32)
        ev_charging = np.asarray([[ev_charge[a, t].X for t in range(h)] for a in range(n)], dtype=np.float32)
        result = MisocpSolveResult(battery, curtailed, ev_charging, float(perf_counter() - started), float(model.ObjVal), int(model.Status))
        model.dispose()
        return result


class MisocpController:
    name = "MISOCP"

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.solver = GlobalMisocpSolver(cfg)
        self.plan_start_step = 0

    def reset(self, env_state: dict) -> None:
        pass

    def __call__(self, env, window: dict[str, np.ndarray]):
        if int(env.cur_step) == 0:
            episode, step = env._cursor()
            horizon = int(env.episode_length - env.cur_step)
            price = derive_import_price_seq(env.data["price"][episode, step:step + horizon], markup_eur_per_kwh=float(get_import_price_markup(env.cfg)))
            load = np.asarray(env.data["load"][episode, step:step + horizon], dtype=np.float32).T
            pv_shared = np.asarray(env.data["pv"][episode, step:step + horizon], dtype=np.float32).reshape(1, -1)
            pv = np.repeat(pv_shared, int(env.n), axis=0).astype(np.float32)
            result = self.solver.solve(import_price_seq=price, load_seq=load, pv_seq=pv, soc_init=np.asarray(env.soc, dtype=np.float32), ev_soc_init=np.asarray(env.ev_soc, dtype=np.float32))
            self.plan_start_step = int(env.cur_step); self.battery_power_kw = result.battery_power_kw; self.pv_curtail_kw = result.pv_curtail_kw; self.ev_charge_kw = result.ev_charge_kw; self.solve_time_sec = float(result.solve_time_sec)
        offset = int(env.cur_step - self.plan_start_step)
        battery = np.asarray(self.battery_power_kw[:, offset], dtype=np.float32)
        curtail = np.asarray(self.pv_curtail_kw[:, offset], dtype=np.float32)
        ev_charge = np.asarray(self.ev_charge_kw[:, offset], dtype=np.float32) if hasattr(self, "ev_charge_kw") else None
        actions, action_array, info = action_array_from_power(env, battery, curtail, ev_charge)
        solve_time = self.solve_time_sec if offset == 0 else 0.0
        info["solve_time_sec"] = np.asarray(solve_time, dtype=np.float32)
        return actions, info, {"solve_time_sec": solve_time}


__all__ = ["MisocpController"]
