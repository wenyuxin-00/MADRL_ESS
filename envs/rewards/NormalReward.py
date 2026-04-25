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
        if hasattr(cfg.reward, "w_action_pen"):
            raise AttributeError(
                "cfg.reward.w_action_pen is no longer supported. NormalReward now uses "
                "action_boundary_penalty_weight. Re-run notebooks/madrl/train_base.ipynb "
                "after updating configs/experiment_config.py."
            )
        if hasattr(cfg.reward, "w_soc_pen"):
            raise AttributeError(
                "cfg.reward.w_soc_pen is no longer supported. New NormalReward contract splits "
                "the old SoC shaping into action_boundary_penalty_weight and "
                "soc_boundary_regularization_weight. Re-run notebooks/madrl/train_base.ipynb "
                "after updating configs/experiment_config.py."
            )
        if hasattr(cfg.reward, "action_feasibility_regularization_weight"):
            raise AttributeError(
                "cfg.reward.action_feasibility_regularization_weight belongs to the reverted "
                "SoC-aware mapping experiment. Expected cfg.reward.action_boundary_penalty_weight "
                "for the restored MADRL baseline contract. Re-run notebooks/madrl/train_base.ipynb after updating "
                "configs/experiment_config.py."
            )
        if hasattr(cfg.reward, "terminal_soc_value_weight"):
            raise AttributeError(
                "cfg.reward.terminal_soc_value_weight is no longer supported. New NormalReward "
                "contract removes terminal SoC shaping entirely. Re-run notebooks/madrl/"
                "train_base.ipynb after updating configs/experiment_config.py."
            )
        if hasattr(cfg.reward, "local_action_penalty_mode") or hasattr(cfg.reward, "local_action_penalty_weight"):
            raise AttributeError(
                "local_action_penalty_mode / local_action_penalty_weight were removed in "
                "fixMADRL section 5.3. Local action feasibility is now projected in rollout, "
                "target-Q, and actor-loss; reward-level local action regularization uses "
                "action_boundary_penalty_weight. "
                "Re-run notebooks/madrl/train_base.ipynb after updating configs/experiment_config.py."
            )

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

        missing_fields = [
            field_name
            for field_name in (
                "action_boundary_penalty_weight",
                "soc_boundary_regularization_weight",
                "throughput_bonus_eur_per_kwh_max",
                "soc_boundary_epsilon",
                "soc_boundary_margin",
            )
            if not hasattr(cfg.reward, field_name)
        ]
        if missing_fields:
            raise AttributeError(
                "NormalReward received an old reward config without the new MADRL shaping fields "
                f"{missing_fields}. Expected action_boundary_penalty_weight, "
                "soc_boundary_regularization_weight, throughput_bonus_eur_per_kwh_max, "
                "soc_boundary_epsilon, and soc_boundary_margin. Re-run notebooks/madrl/train_base.ipynb "
                "after updating configs/experiment_config.py."
            )

        self.storage_profit_weight = float(getattr(cfg.reward, "storage_profit_weight", 1.0))
        if not np.isclose(self.storage_profit_weight, 1.0):
            raise ValueError(
                "NormalReward no longer supports storage_profit_weight != 1.0. The new MADRL reward "
                "contract uses unscaled madrl_r_inc. Re-run notebooks/madrl/train_base.ipynb after "
                "resetting storage_profit_weight to 1.0."
            )
        self.action_boundary_penalty_weight = float(cfg.reward.action_boundary_penalty_weight)
        self.soc_boundary_regularization_weight = float(cfg.reward.soc_boundary_regularization_weight)
        self.throughput_bonus_eur_per_kwh_max = float(cfg.reward.throughput_bonus_eur_per_kwh_max)
        self.soc_boundary_epsilon = float(cfg.reward.soc_boundary_epsilon)
        self.soc_boundary_margin = float(cfg.reward.soc_boundary_margin)
        self.w_voltage_pen = float(cfg.reward.w_voltage_pen)
        self.w_line_pen = float(getattr(cfg.reward, "w_line_pen", 0.0))
        self.w_trafo_pen = float(cfg.reward.w_trafo_pen)
        if self.action_boundary_penalty_weight < 0.0:
            raise ValueError("action_boundary_penalty_weight must be non-negative.")
        if self.soc_boundary_regularization_weight < 0.0:
            raise ValueError("soc_boundary_regularization_weight must be non-negative.")
        if self.throughput_bonus_eur_per_kwh_max < 0.0:
            raise ValueError("throughput_bonus_eur_per_kwh_max must be non-negative.")
        if self.soc_boundary_epsilon < 0.0:
            raise ValueError("soc_boundary_epsilon must be non-negative.")
        if self.soc_boundary_margin < 0.0:
            raise ValueError("soc_boundary_margin must be non-negative.")

    @property
    def component_meta(self) -> list[ComponentMeta]:
        return [
            ComponentMeta("madrl_r_inc", "+ madrl_r_inc", "blue", 1),
            ComponentMeta("madrl_r_action_penalty", "- madrl_r_action_penalty", "amber", -1),
            ComponentMeta("madrl_r_soc_regularization", "- madrl_r_soc_regularization", "green", -1),
            ComponentMeta("madrl_r_throughput_bonus", "+ madrl_r_throughput_bonus", "teal", 1),
            ComponentMeta("madrl_r_safe_v", "- madrl_r_safe_v", "red", -1),
            ComponentMeta("madrl_r_safe_line", "- madrl_r_safe_line", "purple", -1),
            ComponentMeta("madrl_r_safe_trafo", "- madrl_r_safe_trafo", "maroon", -1),
            ComponentMeta("madrl_r_safe_total", "- madrl_r_safe_total", "rose", -1),
            ComponentMeta("madrl_r_total_internal", "+ madrl_r_total_internal", "slate", 1),
        ]

    def _throughput_bonus_weight(self, env_state: dict) -> float:
        if "throughput_bonus_weight_t" in env_state:
            return float(env_state["throughput_bonus_weight_t"])
        progress = float(env_state.get("training_progress", 1.0))
        progress = float(np.clip(progress, 0.0, 1.0))
        if progress < 0.20:
            return self.throughput_bonus_eur_per_kwh_max
        if progress < 0.80:
            decay_ratio = (progress - 0.20) / 0.60
            return self.throughput_bonus_eur_per_kwh_max * max(0.0, 1.0 - decay_ratio)
        return 0.0

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        required_keys = ("battery_power_t", "storage_price_t", "soc_t", "soc_min", "soc_max")
        missing_keys = [key for key in required_keys if key not in env_state]
        if missing_keys:
            raise KeyError(
                "NormalReward requires env_state keys "
                f"{missing_keys} under the Step 3 MADRL reward contract. Re-run the caller through "
                "GridEnv.step or update notebooks/madrl/train_base.ipynb."
            )

        battery_power_t = np.asarray(env_state["battery_power_t"], dtype=np.float32)
        price_t = float(env_state["storage_price_t"])
        dt = float(env_state.get("dt", 1.0))
        soc_t = np.asarray(env_state["soc_t"], dtype=np.float32)
        soc_min = float(env_state["soc_min"])
        soc_max = float(env_state["soc_max"])
        v_violation = np.asarray(env_state.get("v_violation", np.zeros_like(battery_power_t)), dtype=np.float32)
        psi_v_raw = float(env_state.get("psi_v_raw", 0.0))
        psi_line_raw = float(env_state.get("psi_line_raw", 0.0))
        psi_trafo_raw = float(env_state.get("psi_trafo_raw", 0.0))
        throughput_bonus_weight_t = self._throughput_bonus_weight(env_state)

        if soc_t.shape != battery_power_t.shape:
            raise ValueError(
                f"NormalReward expects soc_t and battery_power_t to share shape, got {soc_t.shape} "
                f"vs {battery_power_t.shape}."
            )
        if not 0.0 <= soc_min <= soc_max <= 1.0:
            raise ValueError(f"NormalReward received invalid SoC bounds [{soc_min}, {soc_max}].")
        soc_lower_soft = soc_min + self.soc_boundary_margin
        soc_upper_soft = soc_max - self.soc_boundary_margin
        if soc_lower_soft > soc_upper_soft:
            raise ValueError(
                "soc_boundary_margin is too large for the configured SoC interval: "
                f"soft bounds [{soc_lower_soft}, {soc_upper_soft}] are invalid."
            )

        storage_components = compute_storage_profit_components(
            battery_power_kw=battery_power_t,
            price_eur_per_kwh=price_t,
            dt_hours=dt,
        )
        madrl_r_inc = storage_components["storage_profit_eur"].astype(np.float32)

        pushes_lower = (battery_power_t < 0.0) & (soc_t <= soc_min + self.soc_boundary_epsilon)
        pushes_upper = (battery_power_t > 0.0) & (soc_t >= soc_max - self.soc_boundary_epsilon)
        boundary_push = np.logical_or(pushes_lower, pushes_upper)
        madrl_r_action_penalty = (
            self.action_boundary_penalty_weight
            * np.abs(battery_power_t)
            * boundary_push.astype(np.float32)
        ).astype(np.float32)

        below = np.maximum(0.0, soc_lower_soft - soc_t).astype(np.float32)
        above = np.maximum(0.0, soc_t - soc_upper_soft).astype(np.float32)
        madrl_r_soc_regularization = (
            self.soc_boundary_regularization_weight * (below * below + above * above)
        ).astype(np.float32)

        throughput_kwh = (np.abs(battery_power_t) * np.float32(dt)).astype(np.float32)
        madrl_r_throughput_bonus = (throughput_bonus_weight_t * throughput_kwh).astype(np.float32)

        n_agents = int(battery_power_t.shape[0])
        voltage_total = self.w_voltage_pen * psi_v_raw
        v_sum = float(np.sum(v_violation))
        if v_sum > 0.0:
            v_weights = (v_violation / v_sum).astype(np.float32)
        elif psi_v_raw > 0.0:
            v_weights = np.full((n_agents,), 1.0 / max(n_agents, 1), dtype=np.float32)
        else:
            v_weights = np.zeros((n_agents,), dtype=np.float32)
        madrl_r_safe_v = (n_agents * voltage_total * v_weights).astype(np.float32)
        madrl_r_safe_line = np.full((n_agents,), self.w_line_pen * psi_line_raw, dtype=np.float32)
        madrl_r_safe_trafo = np.full((n_agents,), self.w_trafo_pen * psi_trafo_raw, dtype=np.float32)
        madrl_r_safe_total = (madrl_r_safe_v + madrl_r_safe_line + madrl_r_safe_trafo).astype(np.float32)
        madrl_r_total_internal = (
            madrl_r_inc
            - madrl_r_action_penalty
            - madrl_r_soc_regularization
            + madrl_r_throughput_bonus
            - madrl_r_safe_total
        ).astype(np.float32)

        components = {
            "madrl_r_inc": madrl_r_inc,
            "madrl_r_action_penalty": madrl_r_action_penalty,
            "madrl_r_soc_regularization": madrl_r_soc_regularization,
            "madrl_r_throughput_bonus": madrl_r_throughput_bonus,
            "madrl_r_safe_v": madrl_r_safe_v,
            "madrl_r_safe_line": madrl_r_safe_line,
            "madrl_r_safe_trafo": madrl_r_safe_trafo,
            "madrl_r_safe_total": madrl_r_safe_total,
            "madrl_r_total_internal": madrl_r_total_internal,
        }
        return madrl_r_total_internal, components
