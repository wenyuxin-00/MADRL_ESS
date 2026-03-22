"""Grid-aware composite reward with local voltage and shared overload penalties."""

from __future__ import annotations

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn
from envs.rewards.composite import CompositeReward


class GridCompositeReward(RewardFn):
    """Extend the base composite reward with grid-security penalty terms.

    Reward allocation policy:
    - Base reward terms remain per-agent.
    - Voltage penalty remains local to each agent's connected bus.
    - Line and transformer overload penalties are shared across agents.
    """

    def __init__(self, cfg: object) -> None:
        self._base = CompositeReward(cfg)
        self.w_v_pen = float(cfg.grid.w_v_pen)
        legacy_overload = float(getattr(cfg.grid, "w_l_pen", 0.0))
        self.w_line_pen = float(getattr(cfg.grid, "w_line_pen", legacy_overload))
        self.w_trafo_pen = float(getattr(cfg.grid, "w_trafo_pen", legacy_overload))

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            *self._base.component_meta,
            ComponentMeta("r_v_pen", "- r_v_pen (voltage violation)", "red", -1),
            ComponentMeta("r_line_pen", "- r_line_pen (line overload)", "brown", -1),
            ComponentMeta(
                "r_trafo_pen",
                "- r_trafo_pen (transformer overload)",
                "maroon",
                -1,
            ),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        base_total, components = self._base.compute(env_state)
        n_agents = int(base_total.shape[0])

        v_violation = np.asarray(
            env_state.get("v_violation", np.zeros(n_agents, dtype=np.float32)),
            dtype=np.float32,
        )
        w_v = float(env_state.get("w_v_pen", self.w_v_pen))
        r_v_pen = (w_v * v_violation).astype(np.float32)

        legacy_overload = float(env_state.get("l_violation", 0.0))
        line_violation = float(env_state.get("line_violation", legacy_overload))
        trafo_violation = float(env_state.get("trafo_violation", 0.0))
        w_line = float(env_state.get("w_line_pen", self.w_line_pen))
        w_trafo = float(env_state.get("w_trafo_pen", self.w_trafo_pen))
        r_line_pen = np.full(n_agents, w_line * line_violation, dtype=np.float32)
        r_trafo_pen = np.full(n_agents, w_trafo * trafo_violation, dtype=np.float32)

        total = (base_total - r_v_pen - r_line_pen - r_trafo_pen).astype(np.float32)
        components["r_v_pen"] = r_v_pen
        components["r_line_pen"] = r_line_pen
        components["r_trafo_pen"] = r_trafo_pen
        return total, components
