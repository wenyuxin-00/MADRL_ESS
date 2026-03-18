"""GridCore — the physics simulation layer.

Responsibilities
----------------
- Own one pandapower network object per environment instance.
- Accept per-agent battery injections and background load/PV profiles.
- Run AC power flow and return a ``GridStepResult``.
- Handle non-convergence gracefully (return last valid result + flag).

No RL / gym concepts live here.  SOC tracking stays in ``GridEnv``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from grid.core.grid_types import GridStepResult
from grid.core.net_builder import apply_bus_injections, build_simbench_net

if TYPE_CHECKING:
    from grid.config.grid_config import AgentDeployment
    from configs.experiment_config import GridConfig


class GridCore:
    """Thin wrapper around a pandapower net for one RL environment instance.

    Parameters
    ----------
    deployments:
        One ``AgentDeployment`` per RL agent, containing the bus index and
        device parameters.
    grid_cfg:
        ``GridConfig`` dataclass with network code, solver choice, and
        constraint thresholds.
    """

    def __init__(
        self,
        deployments: list[AgentDeployment],
        grid_cfg: Any,  # GridConfig — avoid circular import at module level
    ) -> None:
        self.deployments = deployments
        self.grid_cfg = grid_cfg
        self.n_agents = len(deployments)
        self.agent_bus_ids: list[int] = [d.bus_id for d in deployments]

        self.net = build_simbench_net(grid_cfg.sb_code)
        self.n_buses: int = len(self.net.bus)
        self.n_lines: int = len(self.net.line)

        # Fallback result used when power flow does not converge.
        self._last_valid: GridStepResult = self._make_zero_result()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, base_load_kw: np.ndarray, base_pv_kw: np.ndarray) -> None:
        """Initialise episode state.

        Parameters
        ----------
        base_load_kw:
            Background demand at each agent bus for timestep 0 in kW.
            Shape ``(n_agents,)``.
        base_pv_kw:
            PV generation at each agent bus for timestep 0 in kW.
            Shape ``(n_agents,)``.
        """
        self._last_valid = self._make_zero_result()

    def step(
        self,
        p_batt_kw: np.ndarray,
        base_load_kw: np.ndarray,
    ) -> GridStepResult:
        """Run one power-flow step and return the result.

        The net injection at agent bus *i* is::

            p_inject_kw[i] = -(base_load_kw[i] + p_batt_kw[i])

        Positive ``p_inject_kw`` means the bus is *exporting* to the grid
        (battery discharging faster than local load).

        Parameters
        ----------
        p_batt_kw:
            Battery power per agent in kW.  Positive = charging (import),
            negative = discharging (export).  Shape ``(n_agents,)``.
        base_load_kw:
            Net background demand (load minus PV) per agent in kW.
            Shape ``(n_agents,)``.

        Returns
        -------
        GridStepResult
            If power flow converges, fresh result.  If not, last valid
            result with ``converged=False``.
        """
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
            # Return last known-good values with converged=False.
            import dataclasses

            return dataclasses.replace(self._last_valid, converged=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_result(self, *, converged: bool) -> GridStepResult:
        """Read pandapower result tables and compute violation metrics."""
        vm_pu = self.net.res_bus["vm_pu"].to_numpy(dtype=np.float32)
        va_degree = self.net.res_bus["va_degree"].to_numpy(dtype=np.float32)
        line_loading_pct = self.net.res_line["loading_percent"].to_numpy(dtype=np.float32)
        p_mw_from = self.net.res_line["p_from_mw"].to_numpy(dtype=np.float32)

        # Voltage at agent buses — match by position in net.bus.index.
        bus_index = list(self.net.bus.index)
        agent_vm_pu = np.array(
            [float(vm_pu[bus_index.index(bid)]) for bid in self.agent_bus_ids],
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
        """Fallback result used before the first successful power flow."""
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
