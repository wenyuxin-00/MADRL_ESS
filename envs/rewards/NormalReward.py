from __future__ import annotations
import numpy as np
from envs.rewards.base import ComponentMeta, RewardFn
class NormalReward(RewardFn):

    def __init__(self, cfg: object) -> None:
        if hasattr(cfg.reward, "w_action_pen") and not hasattr(cfg.reward, "w_soc_pen"):
            raise AttributeError("cfg.reward.w_action_pen is no longer supported; use cfg.reward.w_soc_pen instead.")
        self.w_soc_pen = float(cfg.reward.w_soc_pen)
        self.export_subsidy_eur_per_kwh = float(getattr(cfg.reward, "export_subsidy_eur_per_kwh", 0.079))
        self.w_voltage_pen = float(cfg.reward.w_voltage_pen)
        self.w_line_pen = float(getattr(cfg.reward, "w_line_pen", 0.0))
        self.w_trafo_pen = float(cfg.reward.w_trafo_pen)

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            ComponentMeta("r_purchase_cost", "- r_purchase_cost (grid purchase cost)", "green", -1),
            ComponentMeta("r_export_subsidy", "+ r_export_subsidy (grid export subsidy)", "teal", +1),
            ComponentMeta("r_soc_pen", "- r_soc_pen (SoC feasibility penalty)", "orange", -1),
            ComponentMeta("r_safe_v", "- r_safe_v (voltage penalty)", "red", -1),
            ComponentMeta("r_safe_line", "- r_safe_line (line penalty)", "purple", -1),
            ComponentMeta("r_safe_trafo", "- r_safe_trafo (trafo penalty)", "maroon", -1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        import_price_t = float(env_state["import_price_t"])
        actual_grid_power_t = np.asarray(
            env_state.get("actual_grid_power_t", env_state.get("net_load_t")),
            dtype=np.float32,
        )
        dt = float(env_state.get("dt", 1.0))
        v_violation = np.asarray(
            env_state.get("v_violation", np.zeros_like(actual_grid_power_t)),
            dtype=np.float32,
        )
        psi_v_raw = float(env_state.get("psi_v_raw", 0.0))
        psi_line_raw = float(env_state.get("psi_line_raw", 0.0))
        psi_trafo_raw = float(env_state.get("psi_trafo_raw", 0.0))
        grid_import = np.maximum(actual_grid_power_t, 0.0).astype(np.float32)
        grid_export = np.maximum(-actual_grid_power_t, 0.0).astype(np.float32)
        r_purchase_cost = (grid_import * np.float32(dt) * np.float32(import_price_t)).astype(np.float32)
        r_export_subsidy = (
            grid_export * np.float32(dt) * np.float32(self.export_subsidy_eur_per_kwh)
        ).astype(np.float32)
        n_agents = int(actual_grid_power_t.shape[0])
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
        r_soc_pen = np.zeros((n_agents,), dtype=np.float32)
        total = (-r_purchase_cost + r_export_subsidy - r_soc_pen - r_safe_v - r_safe_line - r_safe_trafo).astype(np.float32)
        components = {
            "r_purchase_cost": r_purchase_cost,
            "r_export_subsidy": r_export_subsidy,
            "r_soc_pen": r_soc_pen,
            "r_safe_v": r_safe_v,
            "r_safe_line": r_safe_line,
            "r_safe_trafo": r_safe_trafo,
        }
        return total, components
