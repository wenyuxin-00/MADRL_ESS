from __future__ import annotations

from dataclasses import replace
from typing import Any
import numpy as np
import pandas as pd

from configs.cfg import Cfg
from data.share_data import ShareData
from envs.grid_core import PowerFlowGridCore


def _vector(value: tuple[float, ...], n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != int(n):
        raise ValueError(f"{name} expected {n} values, got {arr.size}.")
    return arr


def project_action_to_soc(cfg: Cfg, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
    n = int(cfg.env.num_agents); soc = np.asarray(soc, dtype=np.float32).reshape(n); raw = np.asarray(action, dtype=np.float32).reshape(n)
    cap = _vector(cfg.env.battery_capacity_kwh, n, "battery_capacity_kwh")
    pmax = cap * np.float32(cfg.env.max_charge_rate)
    dt, eff = np.float32(cfg.env.dt_hours), np.float32(cfg.env.efficiency)
    upper = (np.float32(cfg.env.soc_max) - soc) * cap / (pmax * dt * eff)
    lower = (np.float32(cfg.env.soc_min) - soc) * eff * cap / (pmax * dt)
    return np.clip(raw, np.maximum(lower, -1.0), np.minimum(upper, 1.0)).astype(np.float32)


def _ev_available(cfg: Cfg, step: int) -> bool:
    daily_step = int(step) % int(cfg.env.episode_steps)
    arr = int(cfg.env.ev_arrival_step); dep = int(cfg.env.ev_departure_step)
    return bool(daily_step >= arr or daily_step < dep) if arr > dep else bool(arr <= daily_step < dep)


def _is_ev_departure_step(cfg: Cfg, step: int) -> bool:
    return int(step) % int(cfg.env.episode_steps) == int(cfg.env.ev_departure_step)


def _is_ev_arrival_step(cfg: Cfg, step: int) -> bool:
    return int(step) % int(cfg.env.episode_steps) == int(cfg.env.ev_arrival_step)


def count_future_connected_steps_until_departure(cfg: Cfg, step: int) -> int:
    base = int(cfg.env.episode_steps)
    daily_step = int(step) % base
    if not _ev_available(cfg, daily_step):
        return 0
    count = 0
    for offset in range(1, base + 1):
        future_step = (daily_step + offset) % base
        if future_step == int(cfg.env.ev_departure_step):
            break
        if _ev_available(cfg, future_step):
            count += 1
    return int(count)


def _ev_constraint_mode(cfg: Cfg) -> str:
    mode = str(getattr(cfg.env, "ev_departure_constraint_mode", "soft")).lower()
    if mode not in {"soft", "hard", "emergency"}:
        raise ValueError(f"ev_departure_constraint_mode expected 'soft', 'hard', or 'emergency', got {mode!r}.")
    return mode


def _is_ev_emergency_step(cfg: Cfg, step: int) -> bool:
    base = int(cfg.env.episode_steps)
    daily_step = int(step) % base
    emergency_steps = int(np.ceil(float(cfg.env.ev_emergency_window_hours) / max(float(cfg.env.dt_hours), 1e-6)))
    emergency_steps = max(0, min(emergency_steps, base))
    dep = int(cfg.env.ev_departure_step) % base
    for offset in range(emergency_steps, 0, -1):
        if daily_step == (dep - offset) % base:
            return True
    return False


def _remaining_emergency_steps_including_current(cfg: Cfg, step: int) -> int:
    if not _is_ev_emergency_step(cfg, step):
        return 0
    base = int(cfg.env.episode_steps)
    daily_step = int(step) % base
    dep = int(cfg.env.ev_departure_step) % base
    count = 0
    for offset in range(base):
        candidate = (daily_step + offset) % base
        if candidate == dep:
            break
        if _is_ev_emergency_step(cfg, candidate) and _ev_available(cfg, candidate):
            count += 1
    return int(count)


def project_ev_action_to_departure_soc(
    cfg: Cfg,
    step: int,
    ev_soc: np.ndarray,
    ev_charge_kw_rl: np.ndarray,
    ev_cap: np.ndarray,
    ev_pmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project EV charging power so that the departure SOC requirement remains feasible.

    Returns projected charging power, added charging power, required minimum charging
    power at this step, and an infeasibility flag per agent.
    """
    n = int(cfg.env.num_agents)
    zeros = np.zeros((n,), dtype=np.float32)
    if not bool(cfg.env.ev_enabled) or not _ev_available(cfg, step) or _is_ev_departure_step(cfg, step):
        return zeros.copy(), zeros.copy(), zeros.copy(), np.zeros((n,), dtype=bool)
    ev_soc_arr = np.asarray(ev_soc, dtype=np.float32).reshape(n)
    rl = np.asarray(ev_charge_kw_rl, dtype=np.float32).reshape(n)
    cap = np.asarray(ev_cap, dtype=np.float32).reshape(n)
    pmax = np.asarray(ev_pmax, dtype=np.float32).reshape(n)
    dt, eff = np.float32(cfg.env.dt_hours), np.float32(cfg.env.ev_efficiency)
    future_steps = np.float32(count_future_connected_steps_until_departure(cfg, step))
    max_future_soc_gain = future_steps * pmax * dt * eff / np.maximum(cap, np.float32(1e-6))
    required_min_charge_kw = np.maximum(
        0.0,
        (np.float32(cfg.env.ev_departure_soc_req) - max_future_soc_gain - ev_soc_arr) * cap / np.maximum(dt * eff, np.float32(1e-6)),
    ).astype(np.float32)
    ev_hard_infeasible = required_min_charge_kw > pmax + np.float32(1e-6)
    projected = np.clip(np.maximum(rl, required_min_charge_kw), 0.0, pmax).astype(np.float32)
    projection_gap = np.maximum(projected - rl, 0.0).astype(np.float32)
    return projected, projection_gap, required_min_charge_kw, ev_hard_infeasible.astype(bool)


def apply_ev_emergency_charging(
    cfg: Cfg,
    step: int,
    ev_soc: np.ndarray,
    ev_charge_kw_rl: np.ndarray,
    ev_cap: np.ndarray,
    ev_pmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(cfg.env.num_agents)
    zeros = np.zeros((n,), dtype=np.float32)
    if not bool(cfg.env.ev_enabled) or not _ev_available(cfg, step):
        return zeros.copy(), zeros.copy(), zeros.copy()
    rl = np.asarray(ev_charge_kw_rl, dtype=np.float32).reshape(n)
    if not _is_ev_emergency_step(cfg, step):
        return np.clip(rl, 0.0, np.asarray(ev_pmax, dtype=np.float32).reshape(n)).astype(np.float32), zeros.copy(), zeros.copy()
    ev_soc_arr = np.asarray(ev_soc, dtype=np.float32).reshape(n)
    cap = np.asarray(ev_cap, dtype=np.float32).reshape(n)
    pmax = np.asarray(ev_pmax, dtype=np.float32).reshape(n)
    below_target = ev_soc_arr < np.float32(cfg.env.ev_departure_soc_req)
    if not bool(np.any(below_target)):
        return np.clip(rl, 0.0, pmax).astype(np.float32), zeros.copy(), zeros.copy()
    strategy = str(cfg.env.ev_emergency_strategy).lower()
    if strategy == "max_power":
        required = pmax.copy()
    elif strategy == "required_power":
        remaining_steps = max(_remaining_emergency_steps_including_current(cfg, step), 1)
        remaining_time_h = np.float32(remaining_steps) * np.float32(cfg.env.dt_hours)
        energy_needed_kwh = np.maximum(0.0, np.float32(cfg.env.ev_departure_soc_req) - ev_soc_arr) * cap
        required = energy_needed_kwh / np.maximum(np.float32(cfg.env.ev_efficiency) * remaining_time_h, np.float32(1e-6))
        required = np.clip(required, 0.0, pmax).astype(np.float32)
    else:
        raise ValueError(f"ev_emergency_strategy expected 'max_power' or 'required_power', got {strategy!r}.")
    required = np.where(below_target, required, 0.0).astype(np.float32)
    projected = np.clip(np.maximum(rl, required), 0.0, pmax).astype(np.float32)
    added = np.maximum(projected - rl, 0.0).astype(np.float32)
    return projected, added, required


def action_array_from_power(env: Any, battery_power_kw: np.ndarray, pv_curtail_kw: np.ndarray, ev_charge_kw: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    cfg = env.cfg
    pmax = np.asarray(env.pmax, dtype=np.float32)
    battery_action = project_action_to_soc(cfg, env.soc, np.asarray(battery_power_kw, dtype=np.float32).reshape(-1) / np.maximum(pmax, 1e-6))
    episode, step = env._cursor()
    pv_raw_kw = np.maximum(np.repeat(np.float32(env.data["pv"][episode, step]), int(env.n)), 0.0)
    pv_effective_kw = np.maximum(pv_raw_kw - np.asarray(pv_curtail_kw, dtype=np.float32).reshape(-1), 0.0)
    pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
    mask = pv_raw_kw > 1e-6
    pv_utilization[mask] = pv_effective_kw[mask] / pv_raw_kw[mask]
    if ev_charge_kw is None:
        ev_action = np.full_like(battery_action, -1.0, dtype=np.float32)
    else:
        ev_pmax = np.asarray(env.ev_pmax, dtype=np.float32)
        ev_utilization = np.clip(np.asarray(ev_charge_kw, dtype=np.float32).reshape(-1) / np.maximum(ev_pmax, np.float32(1e-6)), 0.0, 1.0)
        ev_action = np.clip(2.0 * ev_utilization - 1.0, -1.0, 1.0).astype(np.float32)
    components = [battery_action, np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0)]
    if int(cfg.model.action_dim) >= 3:
        components.append(np.clip(ev_action, -1.0, 1.0))
    action_array = np.stack(components, axis=-1).astype(np.float32)
    zeros = np.zeros((int(env.n),), dtype=np.float32)
    return [action_array[idx].copy() for idx in range(int(env.n))], action_array, {"action_gap_l1": zeros, "action_gap_linf": zeros}


def _robust_stats(values: np.ndarray, low: float, high: float) -> dict[str, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    axis = None if arr.ndim == 1 else 0
    q_low, q_high, med, q25, q75 = np.quantile(arr, [low, high, 0.5, 0.25, 0.75], axis=axis)
    return {"q_low": np.asarray(q_low, dtype=np.float32), "q_high": np.asarray(q_high, dtype=np.float32), "median": np.asarray(med, dtype=np.float32), "iqr": np.maximum(np.asarray(q75 - q25, dtype=np.float32), np.float32(1e-6))}


def _reshape_agent_stats(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return vector.reshape((target.shape[0],) + (1,) * max(0, target.ndim - 1))


def _robust_tanh(values: np.ndarray, stats: dict[str, np.ndarray], scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if np.asarray(stats["median"]).ndim > 0 and arr.shape[0] == np.asarray(stats["median"]).reshape(-1).shape[0]:
        q_low, q_high, med, iqr = (_reshape_agent_stats(stats[key], arr) for key in ("q_low", "q_high", "median", "iqr"))
    else:
        q_low, q_high, med, iqr = (stats[key] for key in ("q_low", "q_high", "median", "iqr"))
    return np.tanh((np.clip(arr, q_low, q_high) - med) / np.maximum(iqr, np.float32(1e-6)) / max(float(scale), 1e-6)).astype(np.float32)


def _calendar_features(timestamp_value: str, n_agents: int) -> np.ndarray:
    ts = pd.Timestamp(str(timestamp_value)); hour = float(ts.hour) + float(ts.minute) / 60.0; doy = float(ts.dayofyear - 1) + hour / 24.0
    features = np.asarray([np.sin(2.0 * np.pi * hour / 24.0), np.cos(2.0 * np.pi * hour / 24.0), np.sin(2.0 * np.pi * doy / 365.25), np.cos(2.0 * np.pi * doy / 365.25)], dtype=np.float32)
    return np.repeat(features.reshape(1, -1), int(n_agents), axis=0)


def _relative_price(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1); spread = float(np.max(arr) - np.min(arr)) if arr.size else 0.0
    return np.zeros_like(arr, dtype=np.float32) if spread <= 1e-6 else (2.0 * (arr - np.float32(np.min(arr))) / np.float32(spread) - 1.0).astype(np.float32)


def _price_spread(values: np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1); spread = float(np.max(arr) - np.min(arr)) if arr.size else 0.0
    return np.full(arr.shape, np.float32(np.clip(spread / max(float(scale), 1e-6), 0.0, 1.0)), dtype=np.float32)


class MadrlObservationStats:
    def __init__(self, cfg: Cfg, train_data: dict[str, np.ndarray]) -> None:
        low, high, n = float(cfg.obs.normalization_clip_low_quantile), float(cfg.obs.normalization_clip_high_quantile), int(cfg.env.num_agents)
        self.cfg = cfg
        self.load = _robust_stats(np.asarray(train_data["load"], dtype=np.float32).reshape(-1, n), low, high)
        pv = np.asarray(train_data["pv"], dtype=np.float32).reshape(-1)
        self.pv_den = np.full((n,), max(float(np.max(pv)), 1e-6), dtype=np.float32)

    def local(self, timestamp: str, soc: np.ndarray, ev_soc: np.ndarray, ev_available: float) -> np.ndarray:
        cfg = self.cfg
        soc_norm = np.clip(2.0 * (np.asarray(soc, dtype=np.float32) - float(cfg.env.soc_min)) / max(float(cfg.env.soc_max - cfg.env.soc_min), 1e-6) - 1.0, -1.0, 1.0).reshape(-1, 1)
        ev_soc_norm = np.clip(2.0 * (np.asarray(ev_soc, dtype=np.float32) - float(cfg.env.ev_soc_min)) / max(float(cfg.env.ev_soc_max - cfg.env.ev_soc_min), 1e-6) - 1.0, -1.0, 1.0).reshape(-1, 1)
        ev_available_col = np.full((int(cfg.env.num_agents), 1), float(ev_available), dtype=np.float32)
        return np.concatenate([_calendar_features(timestamp, int(cfg.env.num_agents)), soc_norm.astype(np.float32), ev_soc_norm.astype(np.float32), ev_available_col], axis=1).astype(np.float32)

    def load_seq(self, values: np.ndarray) -> np.ndarray:
        return _robust_tanh(values, self.load, float(self.cfg.obs.load_tanh_scale)) if bool(self.cfg.obs.normalization_enabled) else np.asarray(values, dtype=np.float32)

    def pv_seq(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return np.clip(arr / _reshape_agent_stats(self.pv_den, arr), 0.0, 1.2).astype(np.float32) if bool(self.cfg.obs.normalization_enabled) else arr


class GridEnv:
    def __init__(self, cfg: Cfg, split: str, forecast_mode: str, share_data: ShareData) -> None:
        self.cfg, self.split, self.forecast_mode, self.data = cfg, split, forecast_mode, share_data.train if split == "train" else share_data.eval
        self.n, self.base_steps = int(cfg.env.num_agents), int(cfg.env.episode_steps)
        self.train_days = int(cfg.env.train_window_days) if split == "train" else 1
        self.steps = self.base_steps * self.train_days
        self.cap = _vector(cfg.env.battery_capacity_kwh, self.n, "battery_capacity_kwh")
        self.pmax = self.cap * np.float32(cfg.env.max_charge_rate)
        self.ev_cap = _vector(cfg.env.ev_capacity_kwh, self.n, "ev_capacity_kwh")
        self.ev_pmax = _vector(cfg.env.ev_max_charge_kw, self.n, "ev_max_charge_kw")
        self.grid_core = PowerFlowGridCore(cfg)
        self.obs_stats = MadrlObservationStats(cfg, share_data.train)
        self.rng = np.random.default_rng(int(cfg.runtime.seed))
        self.cur_step = 0; self.episode = 0; self.soc = np.full((self.n,), float(cfg.env.init_soc), dtype=np.float32)
        self.ev_soc = np.full((self.n,), float(cfg.env.ev_arrival_soc), dtype=np.float32)
        self.train_step_calls = 0; self.target_train_steps = max(1, int(cfg.train.train_episodes) * self.steps)
        self.import_price_markup_eur_per_kwh = float(cfg.reward.import_price_markup_eur_per_kwh)
        self.num_available_episodes = max(1, int(self.data["price"].shape[0]) - self.train_days + 1)
        self.episode_length = self.steps

    def reset(self, episode_index: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.episode = int(self.rng.integers(0, self.num_available_episodes) if episode_index is None else episode_index)
        if not 0 <= self.episode < self.num_available_episodes:
            raise IndexError(f"{self.split} episode_index expected [0, {self.num_available_episodes}), got {self.episode}.")
        self.cur_step = 0
        if self.split == "train":
            low, high = float(self.cfg.env.train_init_soc_low), float(self.cfg.env.train_init_soc_high)
            self.soc = self.rng.uniform(low, high, size=self.n).astype(np.float32)
        else:
            self.soc = np.full((self.n,), float(self.cfg.env.init_soc), dtype=np.float32)
        self.ev_soc = np.full((self.n,), float(self.cfg.env.ev_arrival_soc), dtype=np.float32)
        return self._obs(), {"episode_idx": self.episode, "initial_soc": self.soc.copy(), "initial_ev_soc": self.ev_soc.copy()}

    def _cursor(self) -> tuple[int, int]:
        return self.episode + self.cur_step // self.base_steps, self.cur_step % self.base_steps

    def _seq(self, signal: str) -> np.ndarray:
        episode, step = self._cursor()
        seq = self.data[f"{self.forecast_mode}_{signal}_seq"][episode, step]
        return np.repeat(seq[:, None], self.n, axis=1).astype(np.float32) if signal == "pv" else seq

    def _obs(self) -> dict[str, np.ndarray]:
        episode, step = self._cursor()
        load = self.data["load"][episode, step]; pv = np.repeat(np.float32(self.data["pv"][episode, step]), self.n)
        price = float(self.data["price"][episode, step])
        price_seq = np.repeat(self._seq("price")[:, None], self.n, axis=1).T
        load_seq_raw, pv_seq_raw = self._seq("load").T, self._seq("pv").T
        seq = np.stack([price_seq, load_seq_raw, pv_seq_raw], axis=-1).astype(np.float32)
        ev_available = np.float32(1.0 if _ev_available(self.cfg, step) else 0.0)
        local = np.column_stack([self.soc, self.ev_soc, np.full(self.n, ev_available, dtype=np.float32), load, pv, np.full(self.n, price, dtype=np.float32)]).astype(np.float32)
        safety_local = np.column_stack([self.soc, self.ev_soc, np.full(self.n, ev_available, dtype=np.float32), load, pv, self.cap, self.pmax, self.ev_cap, self.ev_pmax]).astype(np.float32)
        price_window = self._seq("price")
        return {
            "local": local, "sequence": seq, "madrl_local": self.obs_stats.local(str(self.data["timestamps"][episode, step]), self.soc, self.ev_soc, float(ev_available)),
            "safety_local": safety_local, "wholesale_price_relative_seq": _relative_price(price_window),
            "wholesale_price_spread_seq": _price_spread(price_window, float(self.cfg.obs.wholesale_price_spread_scale_eur_per_kwh)),
            "load_seq": self.obs_stats.load_seq(load_seq_raw), "pv_seq": self.obs_stats.pv_seq(pv_seq_raw),
        }
#定义观测空间的终止状态，当智能体完成一个episode时，返回全零的观测，表示环境已经重置或结束。这样可以让智能体在训练过程中更快地识别episode的边界，并且在评估过程中也能清晰地区分不同的episode。
    def _terminal_obs(self) -> dict[str, np.ndarray]:
        s = int(self.cfg.obs.sequence_length)
        return {
            "local": np.zeros((self.n, 6), dtype=np.float32), "sequence": np.zeros((self.n, s, 3), dtype=np.float32),
            "madrl_local": np.zeros((self.n, 7), dtype=np.float32), "safety_local": np.zeros((self.n, 9), dtype=np.float32),
            "wholesale_price_relative_seq": np.zeros((s,), dtype=np.float32), "wholesale_price_spread_seq": np.zeros((s,), dtype=np.float32),
            "load_seq": np.zeros((self.n, s), dtype=np.float32), "pv_seq": np.zeros((self.n, s), dtype=np.float32),
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, bool, bool, dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32).reshape(self.n, int(self.cfg.model.action_dim))
        battery_action, pv_action = act[:, 0], act[:, 1]
        ev_action = act[:, 2] if int(self.cfg.model.action_dim) >= 3 and bool(self.cfg.env.ev_enabled) else np.full((self.n,), -1.0, dtype=np.float32)
        episode, step = self._cursor()
        load = self.data["load"][episode, step]; pv = np.repeat(np.float32(self.data["pv"][episode, step]), self.n)
        price = float(self.data["price"][episode, step]); dt, eff = float(self.cfg.env.dt_hours), float(self.cfg.env.efficiency)
        charge_kw = battery_action * self.pmax
        delta = np.where(charge_kw >= 0.0, charge_kw * dt * eff / self.cap, charge_kw * dt / (eff * self.cap)).astype(np.float32)
        next_soc = self.soc + delta
        if bool(self.cfg.env.ev_enabled) and _is_ev_arrival_step(self.cfg, step):
            self.ev_soc = np.full((self.n,), float(self.cfg.env.ev_arrival_soc), dtype=np.float32)
        ev_available = np.float32(1.0 if _ev_available(self.cfg, step) else 0.0)
        ev_charge_kw_rl = (ev_available * np.clip(0.5 * (ev_action + 1.0), 0.0, 1.0) * self.ev_pmax).astype(np.float32)
        ev_constraint_mode = _ev_constraint_mode(self.cfg)
        ev_emergency_active = np.zeros((self.n,), dtype=bool)
        ev_emergency_required_kw = np.zeros((self.n,), dtype=np.float32)
        ev_emergency_added_kw = np.zeros((self.n,), dtype=np.float32)
        if ev_constraint_mode == "hard" and bool(self.cfg.env.ev_hard_projection_enabled):
            ev_charge_kw, ev_projection_gap_kw, ev_required_min_charge_kw, ev_hard_infeasible = project_ev_action_to_departure_soc(self.cfg, step, self.ev_soc, ev_charge_kw_rl, self.ev_cap, self.ev_pmax)
            ev_control_mode = "hard"
        elif ev_constraint_mode == "emergency" and bool(self.cfg.env.ev_emergency_charging_enabled):
            ev_charge_kw, ev_emergency_added_kw, ev_emergency_required_kw = apply_ev_emergency_charging(self.cfg, step, self.ev_soc, ev_charge_kw_rl, self.ev_cap, self.ev_pmax)
            ev_projection_gap_kw = np.zeros((self.n,), dtype=np.float32)
            ev_required_min_charge_kw = np.zeros((self.n,), dtype=np.float32)
            ev_hard_infeasible = np.zeros((self.n,), dtype=bool)
            ev_emergency_active = (bool(ev_available > 0.0) and _is_ev_emergency_step(self.cfg, step)) & (self.ev_soc < np.float32(self.cfg.env.ev_departure_soc_req))
            ev_control_mode = "emergency"
        else:
            ev_charge_kw = ev_charge_kw_rl.copy()
            ev_projection_gap_kw = np.zeros((self.n,), dtype=np.float32)
            ev_required_min_charge_kw = np.zeros((self.n,), dtype=np.float32)
            ev_hard_infeasible = np.zeros((self.n,), dtype=bool)
            ev_control_mode = "soft"
        ev_delta_soc = (ev_charge_kw * dt * float(self.cfg.env.ev_efficiency) / self.ev_cap).astype(np.float32)
        next_ev_soc = np.clip(self.ev_soc + ev_delta_soc, float(self.cfg.env.ev_soc_min), float(self.cfg.env.ev_soc_max)).astype(np.float32)
        pv_effective = np.clip(pv * (0.5 * (pv_action + 1.0)), 0.0, pv).astype(np.float32)
        net = load - pv_effective + charge_kw + ev_charge_kw
        storage_price = price + float(self.cfg.reward.import_price_markup_eur_per_kwh)
        storage_profit = (np.maximum(-charge_kw, 0.0) - np.maximum(charge_kw, 0.0)) * storage_price * dt
        ev_charging_cost = (ev_charge_kw * storage_price * dt).astype(np.float32)
        ev_reward = -ev_charging_cost
        ev_projection_penalty = (float(self.cfg.reward.ev_projection_penalty_weight) * np.abs(ev_projection_gap_kw) * np.float32(dt)).astype(np.float32)
        ev_emergency_penalty = (float(self.cfg.reward.ev_emergency_penalty_weight) * np.abs(ev_emergency_added_kw) * np.float32(dt)).astype(np.float32)
        pf = self.grid_core.step(net)
        v_viol, vm = pf.v_violation, pf.vm_pu
        line_pct, trafo_pct = pf.line_loading_pct, pf.trafo_loading_pct
        v_sum = float(np.sum(v_viol))
        v_weights = v_viol / np.float32(v_sum) if v_sum > 0.0 else np.zeros((self.n,), dtype=np.float32)
        psi_v, psi_line, psi_trafo = float(pf.psi_v_raw), float(pf.psi_line_raw), float(pf.psi_trafo_raw)
        safe_v = self.n * self.cfg.reward.w_voltage_pen * psi_v * v_weights
        safe_line = np.full((self.n,), float(self.cfg.reward.w_line_pen) * psi_line, dtype=np.float32)
        safe_trafo = np.full((self.n,), float(self.cfg.reward.w_trafo_pen) * psi_trafo, dtype=np.float32)
        boundary_push = ((charge_kw < 0.0) & (self.soc <= self.cfg.env.soc_min + self.cfg.reward.soc_boundary_epsilon)) | ((charge_kw > 0.0) & (self.soc >= self.cfg.env.soc_max - self.cfg.reward.soc_boundary_epsilon))
        action_penalty = (float(self.cfg.reward.action_boundary_penalty_weight) * np.abs(charge_kw) * boundary_push.astype(np.float32)).astype(np.float32)
        soft_low, soft_high = float(self.cfg.env.soc_min + self.cfg.reward.soc_boundary_margin), float(self.cfg.env.soc_max - self.cfg.reward.soc_boundary_margin)
        soc_regularization = (float(self.cfg.reward.soc_boundary_regularization_weight) * (np.maximum(0.0, soft_low - self.soc) ** 2 + np.maximum(0.0, self.soc - soft_high) ** 2)).astype(np.float32)
        ev_soc_regularization = (float(self.cfg.reward.ev_soc_regularization_weight) * (np.maximum(0.0, float(self.cfg.env.ev_soc_min) - self.ev_soc) ** 2 + np.maximum(0.0, self.ev_soc - float(self.cfg.env.ev_soc_max)) ** 2)).astype(np.float32)
        if _is_ev_departure_step(self.cfg, step):
            ev_departure_gap = np.maximum(0.0, float(self.cfg.env.ev_departure_soc_req) - next_ev_soc).astype(np.float32)
            ev_departure_penalty = (float(self.cfg.reward.ev_departure_penalty_weight) * ev_departure_gap ** 2).astype(np.float32)
        else:
            ev_departure_gap = np.zeros((self.n,), dtype=np.float32)
            ev_departure_penalty = np.zeros((self.n,), dtype=np.float32)
        progress = float(np.clip((self.train_step_calls + 1) * int(self.cfg.train.num_envs) / self.target_train_steps, 0.0, 1.0))
        throughput_weight = float(self.cfg.reward.throughput_bonus_eur_per_kwh_max) * float(np.clip((0.80 - progress) / 0.60, 0.0, 1.0))
        throughput_bonus = (throughput_weight * np.abs(charge_kw) * np.float32(dt)).astype(np.float32)
        reward = (storage_profit + ev_reward - action_penalty - soc_regularization - ev_soc_regularization - ev_departure_penalty - ev_projection_penalty - ev_emergency_penalty + throughput_bonus - safe_v - safe_line - safe_trafo).astype(np.float32)
        self.soc = np.clip(next_soc, self.cfg.env.soc_min, self.cfg.env.soc_max).astype(np.float32)
        self.ev_soc = next_ev_soc.copy()
        self.cur_step += 1; self.train_step_calls += int(self.split == "train"); done = self.cur_step >= self.steps
        obs = self._obs() if not done else self._terminal_obs()
        info = {"episode_done": done, "storage_profit_eur": float(np.sum(storage_profit)), "voltage_violation_count": int(np.sum(v_viol > 0.0)), "min_vm_pu": float(np.min(vm)), "max_vm_pu": float(np.max(vm)), "trafo_loading_pct": trafo_pct.astype(np.float32), "line_loading_pct": line_pct.astype(np.float32), "vm_pu": vm, "soc": self.soc.copy(), "ev_soc": self.ev_soc.copy(), "ev_available": float(ev_available), "net_load": net.astype(np.float32), "pv_effective": pv_effective.astype(np.float32), "pv_curtail": (pv - pv_effective).astype(np.float32), "ev_constraint_mode": ev_constraint_mode, "ev_control_mode": ev_control_mode, "ev_charge_kw_rl": ev_charge_kw_rl.astype(np.float32), "ev_charge_kw": ev_charge_kw.astype(np.float32), "ev_required_min_charge_kw": ev_required_min_charge_kw.astype(np.float32), "ev_projection_gap_kw": ev_projection_gap_kw.astype(np.float32), "ev_hard_infeasible": ev_hard_infeasible.astype(bool), "ev_emergency_active": ev_emergency_active.astype(bool), "ev_emergency_required_kw": ev_emergency_required_kw.astype(np.float32), "ev_emergency_added_kw": ev_emergency_added_kw.astype(np.float32), "ev_charging_cost_eur": ev_charging_cost.astype(np.float32), "ev_departure_gap": ev_departure_gap.astype(np.float32), "ev_departure_penalty": ev_departure_penalty.astype(np.float32), "ev_soc_regularization": ev_soc_regularization.astype(np.float32), "madrl_r_inc": storage_profit.astype(np.float32), "madrl_r_action_penalty": action_penalty, "madrl_r_soc_regularization": soc_regularization, "madrl_r_ev_soc_regularization": ev_soc_regularization, "madrl_r_ev_departure_penalty": ev_departure_penalty, "madrl_r_ev_projection_penalty": ev_projection_penalty, "madrl_r_ev_emergency_penalty": ev_emergency_penalty, "madrl_r_throughput_bonus": throughput_bonus, "madrl_r_safe_v": safe_v.astype(np.float32), "madrl_r_safe_line": safe_line, "madrl_r_safe_trafo": safe_trafo, "madrl_r_safe_total": (safe_v + safe_line + safe_trafo).astype(np.float32), "madrl_r_total_internal": reward.astype(np.float32), "madrl_throughput_kwh": (np.abs(charge_kw) * np.float32(dt)).astype(np.float32), "madrl_throughput_bonus_weight": throughput_weight}
        return obs, reward.astype(np.float32), done, False, info

    def close(self) -> None:
        return None


def build_env(cfg: Cfg, mode: str, forecast_mode: str, share_data: ShareData) -> GridEnv:
    return GridEnv(replace(cfg, forecast=replace(cfg.forecast, mode=forecast_mode)), split="train" if mode == "train" else "eval", forecast_mode=forecast_mode, share_data=share_data)
