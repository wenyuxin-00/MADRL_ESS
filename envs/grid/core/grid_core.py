"""电网核心模型。

封装 pandapower 电网的创建、潮流计算和结果提取，
供 GridEnv 调用。
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from envs.grid.core.grid_types import GridStepResult
from envs.grid.core.net_builder import apply_bus_injections, build_simbench_net

if TYPE_CHECKING:
    from envs.grid.config.grid_config import AgentDeployment
    from configs.experiment_config import GridConfig


class GridCore:
    """单个环境实例对应的 pandapower 电网封装。"""

    def __init__(
        self,
        deployments: list[AgentDeployment],
        grid_cfg: Any,  # GridConfig，避免模块级循环导入
    ) -> None:
        self.deployments = deployments
        self.grid_cfg = grid_cfg
        self.n_agents = len(deployments)
        self.agent_bus_ids: list[int] = [d.bus_id for d in deployments]

        # 创建 pandapower 电网。
        self.net = build_simbench_net(grid_cfg.sb_code)
        self.n_buses: int = len(self.net.bus)
        self.n_lines: int = len(self.net.line)

        # 预构建 bus_id -> 位置索引映射，方便快速取每个 agent 所在母线的结果。
        bus_index_list = list(self.net.bus.index)
        self._bus_id_to_pos: dict[int, int] = {
            bid: pos for pos, bid in enumerate(bus_index_list)
        }
        for bid in self.agent_bus_ids:
            if bid not in self._bus_id_to_pos:
                raise ValueError(
                    f"agent_bus_id={bid} 不存在于电网 bus 索引中。"
                    f"可用的 bus_id: {bus_index_list}"
                )

        # 保存 agent 节点的原始负荷 / 分布式电源值，避免跨步累积。
        self._original_load_p_mw: dict[int, float] = {}
        self._original_sgen_p_mw: dict[int, float] = {}
        for bid in self.agent_bus_ids:
            load_mask = self.net.load["bus"] == bid
            if load_mask.any():
                self._original_load_p_mw[bid] = float(
                    self.net.load.at[self.net.load.index[load_mask][0], "p_mw"]
                )
            sgen_mask = self.net.sgen["bus"] == bid
            if sgen_mask.any():
                self._original_sgen_p_mw[bid] = float(
                    self.net.sgen.at[self.net.sgen.index[sgen_mask][0], "p_mw"]
                )

        # 潮流失败时用上一次有效结果回退。
        self._last_valid: GridStepResult = self._make_zero_result()

    def reset(self, base_load_kw: np.ndarray, base_pv_kw: np.ndarray) -> None:
        """重置 episode 的电网状态。"""
        self._restore_agent_buses()
        self._last_valid = self._make_zero_result()

    def step(
        self,
        p_batt_kw: np.ndarray,
        base_load_kw: np.ndarray,
    ) -> GridStepResult:
        """执行一步潮流计算并返回结果。

        约定：
            `p_batt_kw > 0` 表示电池充电。
            `p_batt_kw < 0` 表示电池放电。
        """
        # 先还原原始工况，再施加本步注入量。
        self._restore_agent_buses()

        p_inject_kw = -(base_load_kw + p_batt_kw)
        bus_id_to_p_kw = {
            self.agent_bus_ids[i]: float(p_inject_kw[i])
            for i in range(self.n_agents)
        }
        apply_bus_injections(self.net, bus_id_to_p_kw)

        try:
            import pandapower as pp

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runpp(
                    self.net,
                    algorithm=self.grid_cfg.pf_solver,
                    numba=True,
                    verbose=False,
                )
            result = self._extract_result(converged=True)
            self._last_valid = result
            return result

        except Exception:
            # 潮流失败时返回上一次有效结果，只把 converged 标为 False。
            import dataclasses

            return dataclasses.replace(self._last_valid, converged=False)

    def _restore_agent_buses(self) -> None:
        """将 agent 节点的 load/sgen 还原到初始值。"""
        for bid in self.agent_bus_ids:
            load_mask = self.net.load["bus"] == bid
            if load_mask.any() and bid in self._original_load_p_mw:
                self.net.load.at[
                    self.net.load.index[load_mask][0], "p_mw"
                ] = self._original_load_p_mw[bid]
            sgen_mask = self.net.sgen["bus"] == bid
            if sgen_mask.any() and bid in self._original_sgen_p_mw:
                self.net.sgen.at[
                    self.net.sgen.index[sgen_mask][0], "p_mw"
                ] = self._original_sgen_p_mw[bid]

    def _extract_result(self, *, converged: bool) -> GridStepResult:
        """从 pandapower 结果表中提取训练会用到的物理量。"""
        vm_pu = self.net.res_bus["vm_pu"].to_numpy(dtype=np.float32)
        va_degree = self.net.res_bus["va_degree"].to_numpy(dtype=np.float32)
        line_loading_pct = self.net.res_line["loading_percent"].to_numpy(dtype=np.float32)
        p_mw_from = self.net.res_line["p_from_mw"].to_numpy(dtype=np.float32)

        # 快速取各 agent 所在母线的电压。
        agent_vm_pu = np.array(
            [float(vm_pu[self._bus_id_to_pos[bid]]) for bid in self.agent_bus_ids],
            dtype=np.float32,
        )

        v_min = float(self.grid_cfg.v_min_pu)
        v_max = float(self.grid_cfg.v_max_pu)
        v_violation = np.maximum(0.0, v_min - agent_vm_pu) + np.maximum(
            0.0, agent_vm_pu - v_max
        )
        v_violation = v_violation.astype(np.float32)

        limit = float(self.grid_cfg.line_max_loading_pct)
        l_violation = float(max(0.0, float(np.max(line_loading_pct)) - limit) / 100.0)

        return GridStepResult(
            converged=converged,
            vm_pu=vm_pu,
            va_degree=va_degree,
            line_loading_pct=line_loading_pct,
            p_mw_from=p_mw_from,
            agent_vm_pu=agent_vm_pu,
            v_violation=v_violation,
            l_violation=l_violation,
            n_buses=self.n_buses,
            n_lines=self.n_lines,
        )

    def _make_zero_result(self) -> GridStepResult:
        """构造潮流未成功时的默认结果。"""
        return GridStepResult(
            converged=False,
            vm_pu=np.ones(self.n_buses, dtype=np.float32),
            va_degree=np.zeros(self.n_buses, dtype=np.float32),
            line_loading_pct=np.zeros(self.n_lines, dtype=np.float32),
            p_mw_from=np.zeros(self.n_lines, dtype=np.float32),
            agent_vm_pu=np.ones(self.n_agents, dtype=np.float32),
            v_violation=np.zeros(self.n_agents, dtype=np.float32),
            l_violation=0.0,
            n_buses=self.n_buses,
            n_lines=self.n_lines,
        )
