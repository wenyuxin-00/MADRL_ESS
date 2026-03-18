"""Grid-aware composite reward function.

Extends ``CompositeReward`` with two additional penalty terms:

* **r_v_pen** — voltage violation penalty: sum of per-agent bus voltage
  deviations beyond the [v_min, v_max] band.
* **r_l_pen** — line overload penalty: the worst single line loading
  percentage beyond the thermal limit, broadcast uniformly to all agents.

``env_state`` must contain all fields that ``CompositeReward`` expects,
plus these grid-specific keys (added by ``GridEnv.step()``):

* ``"v_violation"`` — np.ndarray, shape (n_agents,), pu deviation >= 0
* ``"l_violation"`` — float, worst line overload beyond limit >= 0
* ``"w_v_pen"``     — float, voltage penalty weight (from GridConfig)
* ``"w_l_pen"``     — float, line penalty weight (from GridConfig)
"""

from __future__ import annotations

import numpy as np

from common.rewards.base import ComponentMeta, RewardFn
from common.rewards.composite import CompositeReward


class GridCompositeReward(RewardFn):
    """Composite reward with grid constraint penalty terms.

    Inherits the five base reward components (r_inc, r_pen, r_pbrs, r_soc,
    r_bonus) from ``CompositeReward`` and appends r_v_pen and r_l_pen.
    """

    def __init__(self, cfg: object) -> None:
        self._base = CompositeReward(cfg)
        self.w_v_pen = float(cfg.grid.w_v_pen)
        self.w_l_pen = float(cfg.grid.w_l_pen)

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            *self._base.component_meta,
            ComponentMeta("r_v_pen", "- r_v_pen (voltage violation)", "red",   -1),
            ComponentMeta("r_l_pen", "- r_l_pen (line overload)",     "brown", -1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Base five components (same logic as CompositeReward).
        base_total, components = self._base.compute(env_state)

        n_agents = base_total.shape[0]

        # Voltage violation penalty — per-agent.
        v_violation = np.asarray(
            env_state.get("v_violation", np.zeros(n_agents, dtype=np.float32)),
            dtype=np.float32,
        )
        w_v = float(env_state.get("w_v_pen", self.w_v_pen))
        r_v_pen = (w_v * v_violation).astype(np.float32)

        # Line overload penalty — scalar, broadcast to all agents.
        l_violation = float(env_state.get("l_violation", 0.0))
        w_l = float(env_state.get("w_l_pen", self.w_l_pen))
        r_l_pen = np.full(n_agents, w_l * l_violation, dtype=np.float32)

        total = (base_total - r_v_pen - r_l_pen).astype(np.float32)
        components["r_v_pen"] = r_v_pen
        components["r_l_pen"] = r_l_pen
        return total, components
