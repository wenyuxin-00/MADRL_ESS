from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time
import numpy as np

from configs.cfg import Cfg
from envs.grid_env import _ev_available, _is_ev_arrival_step, action_array_from_power
from utils.price import get_import_price_markup


def _price_thresholds(cfg: Cfg, train_prices: np.ndarray) -> tuple[float, float, float]:
    prices = np.asarray(train_prices, dtype=np.float32).reshape(-1) + np.float32(get_import_price_markup(cfg))
    if prices.size == 0:
        raise ValueError("rule-based battery thresholds require non-empty training prices.")
    low_q = float(getattr(cfg.mpc, "rule_based_low_price_quantile", getattr(cfg.mpc, "low_price_quantile", 0.30)))
    high_q = float(getattr(cfg.mpc, "rule_based_high_price_quantile", getattr(cfg.mpc, "high_price_quantile", 0.70)))
    if not 0.0 <= low_q <= high_q <= 1.0:
        raise ValueError(f"rule-based price quantiles expected 0 <= low <= high <= 1, got {low_q} and {high_q}.")
    return float(np.quantile(prices, low_q)), float(np.quantile(prices, high_q)), float(np.quantile(prices, 0.50))


@dataclass
class RuleBasedEVNoBatteryController:
    cfg: Cfg
    name: str = "rule_based_ev_no_battery"
    display_name: str = "Rule-Based EV without Battery"
    controller_type: str = "rule_based"
    battery_rule: str = "disabled"
    low_price_threshold: float | None = None
    high_price_threshold: float | None = None

    def reset(self, env_state: dict[str, Any]) -> None:
        return None

    def __call__(self, env: Any, window: dict[str, np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
        start = time.perf_counter()
        n = int(self.cfg.env.num_agents)
        battery_power_kw = np.zeros((n,), dtype=np.float32)
        pv_curtail_kw = np.zeros((n,), dtype=np.float32)
        ev_charge_kw = self._ev_charge_request(env)
        actions, action_array, info = action_array_from_power(env, battery_power_kw, pv_curtail_kw, ev_charge_kw)
        info.update({"solve_time_sec": time.perf_counter() - start})
        return actions, action_array, info

    def _ev_charge_request(self, env: Any) -> np.ndarray:
        n = int(self.cfg.env.num_agents)
        episode, step = env._cursor()
        if not bool(getattr(self.cfg.env, "ev_enabled", False)) or not _ev_available(self.cfg, step):
            return np.zeros((n,), dtype=np.float32)
        ev_soc = np.asarray(env.ev_soc, dtype=np.float32).reshape(n)
        if _is_ev_arrival_step(self.cfg, step):
            ev_soc = np.full((n,), float(self.cfg.env.ev_arrival_soc), dtype=np.float32)
        ev_pmax = np.asarray(env.ev_pmax, dtype=np.float32).reshape(n)
        return np.where(ev_soc < np.float32(self.cfg.env.ev_soc_max) - np.float32(1e-7), ev_pmax, 0.0).astype(np.float32)


@dataclass
class RuleBasedEVBatteryController(RuleBasedEVNoBatteryController):
    name: str = "rule_based_ev_with_battery"
    display_name: str = "Rule-Based EV with Battery"
    battery_rule: str = "price_threshold_30_70"
    low_price_threshold: float = 0.0
    high_price_threshold: float = 0.0
    median_price: float = 0.0
    charge_fraction: float = 1.0
    discharge_fraction: float = 1.0
    idle_enabled: bool = True

    @classmethod
    def from_training_prices(cls, cfg: Cfg, train_prices: np.ndarray) -> "RuleBasedEVBatteryController":
        low, high, median = _price_thresholds(cfg, train_prices)
        charge_fraction = float(getattr(cfg.mpc, "rule_based_battery_charge_fraction", 1.0))
        discharge_fraction = float(getattr(cfg.mpc, "rule_based_battery_discharge_fraction", 1.0))
        idle_enabled = bool(getattr(cfg.mpc, "rule_based_battery_idle_enabled", True))
        rule = "price_threshold_30_70" if idle_enabled else "median_always_act"
        return cls(cfg=cfg, low_price_threshold=low, high_price_threshold=high, median_price=median, charge_fraction=charge_fraction, discharge_fraction=discharge_fraction, idle_enabled=idle_enabled, battery_rule=rule)

    def __call__(self, env: Any, window: dict[str, np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
        start = time.perf_counter()
        n = int(self.cfg.env.num_agents)
        battery_power_kw = self._battery_power_request(env)
        pv_curtail_kw = np.zeros((n,), dtype=np.float32)
        ev_charge_kw = self._ev_charge_request(env)
        actions, action_array, info = action_array_from_power(env, battery_power_kw, pv_curtail_kw, ev_charge_kw)
        info.update({"solve_time_sec": time.perf_counter() - start})
        return actions, action_array, info

    def _battery_power_request(self, env: Any) -> np.ndarray:
        episode, step = env._cursor()
        current_price = float(env.data["price"][episode, step]) + float(get_import_price_markup(self.cfg))
        pmax = np.asarray(env.pmax, dtype=np.float32).reshape(int(self.cfg.env.num_agents))
        if self.idle_enabled:
            if current_price <= float(self.low_price_threshold):
                return (pmax * np.float32(np.clip(self.charge_fraction, 0.0, 1.0))).astype(np.float32)
            if current_price >= float(self.high_price_threshold):
                return (-pmax * np.float32(np.clip(self.discharge_fraction, 0.0, 1.0))).astype(np.float32)
            return np.zeros_like(pmax, dtype=np.float32)
        sign = np.float32(1.0 if current_price <= float(self.median_price) else -1.0)
        fraction = self.charge_fraction if sign > 0.0 else self.discharge_fraction
        return (sign * pmax * np.float32(np.clip(fraction, 0.0, 1.0))).astype(np.float32)
