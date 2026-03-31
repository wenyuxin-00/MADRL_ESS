"""Default dense reward for the grid training mainline."""

from __future__ import annotations

import numpy as np

from envs.rewards.base import ComponentMeta, RewardFn


class NormalReward(RewardFn):
    """Cost-driven reward with storage and grid-safety penalties."""

    def __init__(self, cfg: object) -> None:
        self.w_action_pen = float(cfg.reward.w_action_pen)
        self.lambda_throughput = float(cfg.reward.lambda_throughput)
        self.w_voltage_pen = float(cfg.reward.w_voltage_pen)
        self.w_line_pen = float(getattr(cfg.reward, "w_line_pen", 0.0))
        self.w_trafo_pen = float(cfg.reward.w_trafo_pen)

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            ComponentMeta("r_cost", "+ r_cost (incremental cost)", "green", +1),
            ComponentMeta("r_throughput", "+ r_throughput (throughput bonus)", "teal", +1),
            ComponentMeta("r_action_pen", "- r_action_pen (action penalty)", "orange", -1),
            ComponentMeta("r_safe_v", "- r_safe_v (voltage penalty)", "red", -1),
            ComponentMeta("r_safe_line", "- r_safe_line (line penalty)", "purple", -1),
            ComponentMeta("r_safe_trafo", "- r_safe_trafo (trafo penalty)", "maroon", -1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        e_bat = np.asarray(env_state["e_bat"], dtype=np.float32)
        price_t = float(env_state["price_t"])
        net_load_t = np.asarray(env_state["net_load_t"], dtype=np.float32)
        actual_grid_power_t = np.asarray(env_state.get("actual_grid_power_t", net_load_t + e_bat), dtype=np.float32)
        dt = float(env_state.get("dt", 1.0))
        v_violation = np.asarray(
            env_state.get("v_violation", np.zeros_like(e_bat)),
            dtype=np.float32,
        )
        psi_v_raw = float(env_state.get("psi_v_raw", 0.0))
        psi_line_raw = float(env_state.get("psi_line_raw", 0.0))
        psi_trafo_raw = float(env_state.get("psi_trafo_raw", 0.0))

        baseline_cost = net_load_t * dt * price_t
        actual_cost = actual_grid_power_t * dt * price_t
        r_cost = (baseline_cost - actual_cost).astype(np.float32)

        r_throughput = (self.lambda_throughput * np.abs(e_bat) * dt).astype(np.float32)
        r_action_pen = np.zeros_like(e_bat, dtype=np.float32)

        n_agents = int(e_bat.shape[0])
        voltage_total = self.w_voltage_pen * psi_v_raw
        v_sum = float(np.sum(v_violation))
        if v_sum > 0.0:
            v_weights = (v_violation / v_sum).astype(np.float32)
        elif psi_v_raw > 0.0:
            v_weights = np.full((n_agents,), 1.0 / max(n_agents, 1), dtype=np.float32)
        else:
            v_weights = np.zeros((n_agents,), dtype=np.float32)
        r_safe_v = (n_agents * voltage_total * v_weights).astype(np.float32)

        line_total = self.w_line_pen * psi_line_raw
        r_safe_line = np.full((n_agents,), line_total, dtype=np.float32)

        trafo_total = self.w_trafo_pen * psi_trafo_raw
        r_safe_trafo = np.full((n_agents,), trafo_total, dtype=np.float32)

        total = (r_cost + r_throughput - r_action_pen - r_safe_v - r_safe_line - r_safe_trafo).astype(
            np.float32
        )
        components = {
            "r_cost": r_cost,
            "r_throughput": r_throughput,
            "r_action_pen": r_action_pen,
            "r_safe_v": r_safe_v,
            "r_safe_line": r_safe_line,
            "r_safe_trafo": r_safe_trafo,
        }
        return total, components
