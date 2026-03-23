"""Core pandapower wrapper used by GridEnv."""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from envs.grid.core.grid_types import GridStepResult
from envs.grid.core.net_builder import apply_bus_injections, build_simbench_net

if TYPE_CHECKING:
    from envs.grid.config.grid_config import AgentDeployment


class GridCore:
    """One pandapower network instance for one environment."""

    def __init__(
        self,
        deployments: list["AgentDeployment"],
        grid_cfg: Any,
    ) -> None:
        self.deployments = deployments
        self.grid_cfg = grid_cfg
        self.n_agents = len(deployments)
        self.agent_bus_ids = [deployment.bus_id for deployment in deployments]

        self.net = build_simbench_net(grid_cfg.sb_code)
        self.n_buses = int(len(self.net.bus))
        self.n_lines = int(len(self.net.line))
        self.n_trafos = int(len(getattr(self.net, "trafo", [])))

        bus_index_list = list(self.net.bus.index)
        self._bus_id_to_pos = {bus_id: pos for pos, bus_id in enumerate(bus_index_list)}
        for bus_id in self.agent_bus_ids:
            if bus_id not in self._bus_id_to_pos:
                raise ValueError(
                    f"agent_bus_id={bus_id} is not present in network bus index. "
                    f"Available bus ids: {bus_index_list}"
                )

        self._original_load_p_mw: dict[int, float] = {}
        self._original_sgen_p_mw: dict[int, float] = {}
        for bus_id in self.agent_bus_ids:
            load_mask = self.net.load["bus"] == bus_id
            if load_mask.any():
                self._original_load_p_mw[bus_id] = float(
                    self.net.load.at[self.net.load.index[load_mask][0], "p_mw"]
                )
            sgen_mask = self.net.sgen["bus"] == bus_id
            if sgen_mask.any():
                self._original_sgen_p_mw[bus_id] = float(
                    self.net.sgen.at[self.net.sgen.index[sgen_mask][0], "p_mw"]
                )

        self._last_valid = self._make_zero_result()
        self.last_pf_error = ""

        # warm-start 缓存：记录上一次成功的求解器参数
        self._last_success_kwargs: dict | None = None
        self._prev_converged: bool = False

    def reset(self, base_load_kw: np.ndarray, base_pv_kw: np.ndarray) -> None:
        """Reset network state for a new episode."""
        del base_load_kw, base_pv_kw
        self._restore_agent_buses()
        self._last_valid = self._make_zero_result()
        self.last_pf_error = ""
        self._last_success_kwargs = None
        self._prev_converged = False

    def step(
        self,
        p_batt_kw: np.ndarray,
        base_load_kw: np.ndarray,
    ) -> GridStepResult:
        """Apply the current injections and run one power-flow step."""
        self._restore_agent_buses()

        p_inject_kw = -(base_load_kw + p_batt_kw)
        bus_id_to_p_kw = {
            self.agent_bus_ids[i]: float(p_inject_kw[i]) for i in range(self.n_agents)
        }
        apply_bus_injections(self.net, bus_id_to_p_kw)

        try:
            self._runpp_with_fallback()
            result = self._extract_result(converged=True)
            self.last_pf_error = ""
            self._last_valid = result
            self._prev_converged = True
            return result
        except Exception as exc:
            self.last_pf_error = f"{type(exc).__name__}: {exc}"
            self._prev_converged = False
            return dataclasses.replace(self._last_valid, converged=False)

    # ------------------------------------------------------------------
    # 灵敏度快照：中心差分法计算 dvm/dp, dline_loading/dp, dtrafo_loading/dp
    # ------------------------------------------------------------------

    def compute_sensitivity_snapshot(
        self,
        p_batt_kw: np.ndarray,
        base_load_kw: np.ndarray,
        delta_kw: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """中心差分法计算灵敏度快照。

        不修改 _last_valid，使用临时潮流计算。

        返回:
            dict:
                "dvm_dp":            (n_buses, n_agents)  pu/kW
                "dline_loading_dp":  (n_lines, n_agents)  pct/kW
                "dtrafo_loading_dp": (n_trafos, n_agents) pct/kW
        """
        dvm_dp = np.zeros((self.n_buses, self.n_agents), dtype=np.float32)
        dline_dp = np.zeros((self.n_lines, self.n_agents), dtype=np.float32)
        dtrafo_dp = np.zeros((self.n_trafos, self.n_agents), dtype=np.float32)

        for i in range(self.n_agents):
            p_plus = np.array(p_batt_kw, dtype=np.float32)
            p_plus[i] += delta_kw
            p_minus = np.array(p_batt_kw, dtype=np.float32)
            p_minus[i] -= delta_kw

            plus_result = self._run_pf_snapshot(p_plus, base_load_kw)
            minus_result = self._run_pf_snapshot(p_minus, base_load_kw)

            if plus_result is None or minus_result is None:
                # 某个摄动方向潮流失败，该 agent 列保持零
                continue

            inv_2delta = 1.0 / (2.0 * delta_kw)
            dvm_dp[:, i] = (plus_result.vm_pu - minus_result.vm_pu) * inv_2delta
            dline_dp[:, i] = (
                plus_result.line_loading_pct - minus_result.line_loading_pct
            ) * inv_2delta
            dtrafo_dp[:, i] = (
                plus_result.trafo_loading_pct - minus_result.trafo_loading_pct
            ) * inv_2delta

        return {
            "dvm_dp": dvm_dp,
            "dline_loading_dp": dline_dp,
            "dtrafo_loading_dp": dtrafo_dp,
        }

    def _run_pf_snapshot(
        self,
        p_batt_kw: np.ndarray,
        base_load_kw: np.ndarray,
    ) -> GridStepResult | None:
        """运行一次临时潮流，不修改 _last_valid / _prev_converged。"""
        self._restore_agent_buses()
        p_inject_kw = -(base_load_kw + p_batt_kw)
        bus_id_to_p_kw = {
            self.agent_bus_ids[i]: float(p_inject_kw[i]) for i in range(self.n_agents)
        }
        apply_bus_injections(self.net, bus_id_to_p_kw)
        try:
            self._runpp_with_fallback()
            return self._extract_result(converged=True)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # pandapower 求解器（含 warm-start）
    # ------------------------------------------------------------------

    def _runpp_with_fallback(self) -> None:
        """Run pandapower with warm-start cache + stability-first fallback configurations."""
        import pandapower as pp

        base_kwargs = {"verbose": False}

        # warm-start：优先尝试上一次成功的配置
        if self._last_success_kwargs is not None:
            warm_kwargs = dict(self._last_success_kwargs)
            if self._prev_converged:
                warm_kwargs["init"] = "results"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pp.runpp(self.net, **base_kwargs, **warm_kwargs)
                return
            except Exception:
                pass  # warm-start 失败，回退到标准序列

        # 标准 fallback 序列
        attempts = (
            {
                "algorithm": self.grid_cfg.pf_solver,
                "numba": False,
                "lightsim2grid": False,
                "calculate_voltage_angles": False,
                "voltage_depend_loads": False,
                "init": "flat",
            },
            {
                "algorithm": "bfsw",
                "numba": False,
                "lightsim2grid": False,
                "calculate_voltage_angles": False,
                "voltage_depend_loads": False,
                "init": "flat",
            },
            {
                "algorithm": self.grid_cfg.pf_solver,
                "numba": False,
                "lightsim2grid": False,
                "calculate_voltage_angles": False,
                "voltage_depend_loads": False,
                "init": "auto",
            },
            {
                "algorithm": self.grid_cfg.pf_solver,
                "numba": True,
                "lightsim2grid": False,
                "calculate_voltage_angles": False,
                "voltage_depend_loads": False,
                "init": "auto",
            },
        )
        last_exc: Exception | None = None
        error_messages: list[str] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for extra_kwargs in attempts:
                try:
                    pp.runpp(self.net, **base_kwargs, **extra_kwargs)
                    # 缓存成功的配置
                    self._last_success_kwargs = dict(extra_kwargs)
                    return
                except TypeError as exc:
                    last_exc = exc
                    error_messages.append(
                        f"{extra_kwargs.get('algorithm', self.grid_cfg.pf_solver)} / "
                        f"numba={extra_kwargs.get('numba')} / "
                        f"lightsim2grid={extra_kwargs.get('lightsim2grid', 'default')} -> "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if "lightsim2grid" not in str(exc):
                        continue
                except Exception as exc:
                    last_exc = exc
                    error_messages.append(
                        f"{extra_kwargs.get('algorithm', self.grid_cfg.pf_solver)} / "
                        f"numba={extra_kwargs.get('numba')} / "
                        f"lightsim2grid={extra_kwargs.get('lightsim2grid', 'default')} -> "
                        f"{type(exc).__name__}: {exc}"
                    )

        if last_exc is not None:
            raise RuntimeError(" | ".join(error_messages)) from last_exc
        raise RuntimeError("pandapower.runpp failed without exposing an exception.")

    def _restore_agent_buses(self) -> None:
        """Restore tracked load/sgen values before writing the next step."""
        for bus_id in self.agent_bus_ids:
            load_mask = self.net.load["bus"] == bus_id
            if load_mask.any() and bus_id in self._original_load_p_mw:
                self.net.load.at[
                    self.net.load.index[load_mask][0], "p_mw"
                ] = self._original_load_p_mw[bus_id]
            sgen_mask = self.net.sgen["bus"] == bus_id
            if sgen_mask.any() and bus_id in self._original_sgen_p_mw:
                self.net.sgen.at[
                    self.net.sgen.index[sgen_mask][0], "p_mw"
                ] = self._original_sgen_p_mw[bus_id]

    def _extract_result(self, *, converged: bool) -> GridStepResult:
        """Extract the subset of pandapower results used by training and plots."""
        vm_pu = self.net.res_bus["vm_pu"].to_numpy(dtype=np.float32)
        va_degree = self.net.res_bus["va_degree"].to_numpy(dtype=np.float32)
        line_loading_pct = self.net.res_line["loading_percent"].to_numpy(dtype=np.float32)
        p_mw_from = self.net.res_line["p_from_mw"].to_numpy(dtype=np.float32)

        if self.n_trafos > 0 and hasattr(self.net, "res_trafo") and not self.net.res_trafo.empty:
            trafo_loading_pct = self.net.res_trafo["loading_percent"].to_numpy(dtype=np.float32)
        else:
            trafo_loading_pct = np.zeros(self.n_trafos, dtype=np.float32)

        agent_vm_pu = np.array(
            [float(vm_pu[self._bus_id_to_pos[bus_id]]) for bus_id in self.agent_bus_ids],
            dtype=np.float32,
        )

        v_min = float(self.grid_cfg.v_min_pu)
        v_max = float(self.grid_cfg.v_max_pu)

        # 保留旧的 per-agent 电压越限（兼容可视化）
        v_violation = np.maximum(0.0, v_min - agent_vm_pu) + np.maximum(
            0.0, agent_vm_pu - v_max
        )
        v_violation = v_violation.astype(np.float32)

        # 旧的 max-violation 标量（兼容）
        limit = float(self.grid_cfg.line_max_loading_pct)
        max_line_loading = float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0
        max_trafo_loading = float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0
        line_violation = float(max(0.0, max_line_loading - limit) / 100.0)
        trafo_violation = float(max(0.0, max_trafo_loading - limit) / 100.0)
        l_violation = float(max(line_violation, trafo_violation))

        # --- 全网违规详情（全局安全势函数） ---
        bus_v_excess = (
            np.maximum(0.0, v_min - vm_pu) + np.maximum(0.0, vm_pu - v_max)
        ).astype(np.float32)
        bus_v_signed_indicator = np.where(
            vm_pu < v_min, -1.0,
            np.where(vm_pu > v_max, 1.0, 0.0),
        ).astype(np.float32)
        line_excess_arr = (np.maximum(0.0, line_loading_pct - limit) / 100.0).astype(np.float32)
        trafo_excess_arr = (np.maximum(0.0, trafo_loading_pct - limit) / 100.0).astype(np.float32)
        psi_v_raw = float(np.sum(bus_v_excess ** 2))
        psi_line_raw = float(np.sum(line_excess_arr ** 2))
        psi_trafo_raw = float(np.sum(trafo_excess_arr ** 2))

        return GridStepResult(
            converged=converged,
            vm_pu=vm_pu,
            va_degree=va_degree,
            line_loading_pct=line_loading_pct,
            trafo_loading_pct=trafo_loading_pct,
            p_mw_from=p_mw_from,
            agent_vm_pu=agent_vm_pu,
            v_violation=v_violation,
            line_violation=line_violation,
            trafo_violation=trafo_violation,
            l_violation=l_violation,
            n_buses=self.n_buses,
            n_lines=self.n_lines,
            n_trafos=self.n_trafos,
            bus_v_excess=bus_v_excess,
            bus_v_signed_indicator=bus_v_signed_indicator,
            line_excess=line_excess_arr,
            trafo_excess=trafo_excess_arr,
            psi_v_raw=psi_v_raw,
            psi_line_raw=psi_line_raw,
            psi_trafo_raw=psi_trafo_raw,
        )

    def _make_zero_result(self) -> GridStepResult:
        """Fallback result used before the first converged power flow."""
        return GridStepResult(
            converged=False,
            vm_pu=np.ones(self.n_buses, dtype=np.float32),
            va_degree=np.zeros(self.n_buses, dtype=np.float32),
            line_loading_pct=np.zeros(self.n_lines, dtype=np.float32),
            trafo_loading_pct=np.zeros(self.n_trafos, dtype=np.float32),
            p_mw_from=np.zeros(self.n_lines, dtype=np.float32),
            agent_vm_pu=np.ones(self.n_agents, dtype=np.float32),
            v_violation=np.zeros(self.n_agents, dtype=np.float32),
            line_violation=0.0,
            trafo_violation=0.0,
            l_violation=0.0,
            n_buses=self.n_buses,
            n_lines=self.n_lines,
            n_trafos=self.n_trafos,
            bus_v_excess=np.zeros(self.n_buses, dtype=np.float32),
            bus_v_signed_indicator=np.zeros(self.n_buses, dtype=np.float32),
            line_excess=np.zeros(self.n_lines, dtype=np.float32),
            trafo_excess=np.zeros(self.n_trafos, dtype=np.float32),
            psi_v_raw=0.0,
            psi_line_raw=0.0,
            psi_trafo_raw=0.0,
        )
