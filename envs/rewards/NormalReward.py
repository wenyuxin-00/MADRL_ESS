from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.utils.storage_profit import compute_storage_profit_components


@dataclass(frozen=True)
class ComponentMeta:
    key: str
    label: str
    color: str
    sign: int


class NormalReward:
    def __init__(self, cfg: object) -> None:
        if hasattr(cfg.reward, "w_action_pen") and not hasattr(cfg.reward, "w_soc_pen"):
            raise AttributeError("cfg.reward.w_action_pen is no longer supported; use cfg.reward.w_soc_pen instead.")
        self.storage_objective_mode = str(getattr(cfg.reward, "storage_objective_mode", "max_storage_profit"))
        if self.storage_objective_mode != "max_storage_profit":
            raise ValueError(
                "NormalReward received old reward objective "
                f"storage_objective_mode={self.storage_objective_mode!r}. Expected the new contract "
                "storage_objective_mode='max_storage_profit'. Re-run notebooks/madrl/train_base.ipynb "
                "after updating configs/experiment_config.py."
            )
        self.storage_price_mode = str(getattr(cfg.reward, "storage_price_mode", "real_time_price"))
        if self.storage_price_mode != "real_time_price":
            raise ValueError(
                "NormalReward received old reward price contract "
                f"storage_price_mode={self.storage_price_mode!r}. Expected storage_price_mode='real_time_price'. "
                "Re-run notebooks/madrl/train_base.ipynb after updating configs/experiment_config.py."
            )
        self.storage_profit_weight = float(getattr(cfg.reward, "storage_profit_weight", 1.0))
        self.export_subsidy_eur_per_kwh = float(getattr(cfg.reward, "export_subsidy_eur_per_kwh", 0.0))
        self.w_soc_pen = float(cfg.reward.w_soc_pen)
        self.w_voltage_pen = float(cfg.reward.w_voltage_pen)
        self.w_line_pen = float(getattr(cfg.reward, "w_line_pen", 0.0))
        self.w_trafo_pen = float(cfg.reward.w_trafo_pen)

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            ComponentMeta("r_storage_discharge_revenue", "+ r_storage_discharge_revenue", "teal", 1),
            ComponentMeta("r_storage_charge_cost", "- r_storage_charge_cost", "green", -1),
            ComponentMeta("r_storage_profit", "+ r_storage_profit", "blue", 1),
            ComponentMeta("r_soc_pen", "- r_soc_pen (SoC feasibility penalty)", "orange", -1),
            ComponentMeta("r_safe_v", "- r_safe_v (voltage penalty)", "red", -1),
            ComponentMeta("r_safe_line", "- r_safe_line (line penalty)", "purple", -1),
            ComponentMeta("r_safe_trafo", "- r_safe_trafo (trafo penalty)", "maroon", -1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if "battery_power_t" not in env_state:
            raise KeyError(
                "NormalReward requires env_state['battery_power_t'] under the max-storage-profit contract. "
                "The old actual_grid_power_t reward input is no longer accepted. Re-run the caller through "
                "GridEnv.step or update notebooks/madrl/train_base.ipynb."
            )
        if "storage_price_t" not in env_state:
            raise KeyError(
                "NormalReward requires env_state['storage_price_t'] under storage_price_mode='real_time_price'. "
                "The old export_subsidy sale-price objective is no longer accepted. Re-run GridEnv.step or "
                "notebooks/madrl/train_base.ipynb."
            )
        battery_power_t = np.asarray(env_state["battery_power_t"], dtype=np.float32)
        price_t = float(env_state["storage_price_t"])
        dt = float(env_state.get("dt", 1.0))
        v_violation = np.asarray(env_state.get("v_violation", np.zeros_like(battery_power_t)), dtype=np.float32)
        psi_v_raw = float(env_state.get("psi_v_raw", 0.0))
        psi_line_raw = float(env_state.get("psi_line_raw", 0.0))
        psi_trafo_raw = float(env_state.get("psi_trafo_raw", 0.0))

        storage_components = compute_storage_profit_components(
            battery_power_kw=battery_power_t,
            price_eur_per_kwh=price_t,
            dt_hours=dt,
        )
        r_storage_charge_cost = storage_components["storage_charge_cost_eur"]
        r_storage_discharge_revenue = storage_components["storage_discharge_revenue_eur"]
        r_storage_profit = storage_components["storage_profit_eur"]
        n_agents = int(battery_power_t.shape[0])

        voltage_total = self.w_voltage_pen * psi_v_raw
        v_sum = float(np.sum(v_violation))
        if v_sum > 0.0:
            v_weights = (v_violation / v_sum).astype(np.float32)
        elif psi_v_raw > 0.0:
            v_weights = np.full((n_agents,), 1.0 / max(n_agents, 1), dtype=np.float32)
        else:
            v_weights = np.zeros((n_agents,), dtype=np.float32)
        r_safe_v = (n_agents * voltage_total * v_weights).astype(np.float32)
        r_safe_line = np.full((n_agents,), self.w_line_pen * psi_line_raw, dtype=np.float32)
        r_safe_trafo = np.full((n_agents,), self.w_trafo_pen * psi_trafo_raw, dtype=np.float32)
        r_soc_pen = np.zeros((n_agents,), dtype=np.float32)
        total = (
            self.storage_profit_weight * r_storage_profit
            - r_soc_pen
            - r_safe_v
            - r_safe_line
            - r_safe_trafo
        ).astype(np.float32)
        components = {
            "r_storage_discharge_revenue": r_storage_discharge_revenue,
            "r_storage_charge_cost": r_storage_charge_cost,
            "r_storage_profit": r_storage_profit,
            "r_soc_pen": r_soc_pen,
            "r_safe_v": r_safe_v,
            "r_safe_line": r_safe_line,
            "r_safe_trafo": r_safe_trafo,
        }
        return total, components
