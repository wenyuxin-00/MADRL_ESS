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


def action_array_from_power(env: Any, battery_power_kw: np.ndarray, pv_curtail_kw: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    cfg = env.cfg
    pmax = np.asarray(env.pmax, dtype=np.float32)
    battery_action = project_action_to_soc(cfg, env.soc, np.asarray(battery_power_kw, dtype=np.float32).reshape(-1) / np.maximum(pmax, 1e-6))
    episode, step = env._cursor()
    pv_raw_kw = np.maximum(np.repeat(np.float32(env.data["pv"][episode, step]), int(env.n)), 0.0)
    pv_effective_kw = np.maximum(pv_raw_kw - np.asarray(pv_curtail_kw, dtype=np.float32).reshape(-1), 0.0)
    pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
    mask = pv_raw_kw > 1e-6
    pv_utilization[mask] = pv_effective_kw[mask] / pv_raw_kw[mask]
    action_array = np.stack([battery_action, np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0)], axis=-1).astype(np.float32)
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

    def local(self, timestamp: str, soc: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        soc_norm = np.clip(2.0 * (np.asarray(soc, dtype=np.float32) - float(cfg.env.soc_min)) / max(float(cfg.env.soc_max - cfg.env.soc_min), 1e-6) - 1.0, -1.0, 1.0).reshape(-1, 1)
        return np.concatenate([_calendar_features(timestamp, int(cfg.env.num_agents)), soc_norm.astype(np.float32)], axis=1).astype(np.float32)

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
        self.grid_core = PowerFlowGridCore(cfg)
        self.obs_stats = MadrlObservationStats(cfg, share_data.train)
        self.rng = np.random.default_rng(int(cfg.runtime.seed))
        self.cur_step = 0; self.episode = 0; self.soc = np.full((self.n,), float(cfg.env.init_soc), dtype=np.float32)
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
        return self._obs(), {"episode_idx": self.episode, "initial_soc": self.soc.copy()}

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
        local = np.column_stack([self.soc, load, pv, np.full(self.n, price, dtype=np.float32)]).astype(np.float32)
        safety_local = np.column_stack([self.soc, load, pv, self.cap, self.pmax]).astype(np.float32)
        price_window = self._seq("price")
        return {
            "local": local, "sequence": seq, "madrl_local": self.obs_stats.local(str(self.data["timestamps"][episode, step]), self.soc),
            "safety_local": safety_local, "wholesale_price_relative_seq": _relative_price(price_window),
            "wholesale_price_spread_seq": _price_spread(price_window, float(self.cfg.obs.wholesale_price_spread_scale_eur_per_kwh)),
            "load_seq": self.obs_stats.load_seq(load_seq_raw), "pv_seq": self.obs_stats.pv_seq(pv_seq_raw),
        }

    def _terminal_obs(self) -> dict[str, np.ndarray]:
        s = int(self.cfg.obs.sequence_length)
        return {
            "local": np.zeros((self.n, 4), dtype=np.float32), "sequence": np.zeros((self.n, s, 3), dtype=np.float32),
            "madrl_local": np.zeros((self.n, 5), dtype=np.float32), "safety_local": np.zeros((self.n, 5), dtype=np.float32),
            "wholesale_price_relative_seq": np.zeros((s,), dtype=np.float32), "wholesale_price_spread_seq": np.zeros((s,), dtype=np.float32),
            "load_seq": np.zeros((self.n, s), dtype=np.float32), "pv_seq": np.zeros((self.n, s), dtype=np.float32),
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, bool, bool, dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32).reshape(self.n, int(self.cfg.model.action_dim))
        battery_action, pv_action = act[:, 0], act[:, 1]
        episode, step = self._cursor()
        load = self.data["load"][episode, step]; pv = np.repeat(np.float32(self.data["pv"][episode, step]), self.n)
        price = float(self.data["price"][episode, step]); dt, eff = float(self.cfg.env.dt_hours), float(self.cfg.env.efficiency)
        charge_kw = battery_action * self.pmax
        delta = np.where(charge_kw >= 0.0, charge_kw * dt * eff / self.cap, charge_kw * dt / (eff * self.cap)).astype(np.float32)
        next_soc = self.soc + delta
        pv_effective = np.clip(pv * (0.5 * (pv_action + 1.0)), 0.0, pv).astype(np.float32)
        net = load - pv_effective + charge_kw
        storage_price = price + float(self.cfg.reward.import_price_markup_eur_per_kwh)
        storage_profit = (np.maximum(-charge_kw, 0.0) - np.maximum(charge_kw, 0.0)) * storage_price * dt
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
        progress = float(np.clip((self.train_step_calls + 1) * int(self.cfg.train.num_envs) / self.target_train_steps, 0.0, 1.0))
        throughput_weight = float(self.cfg.reward.throughput_bonus_eur_per_kwh_max) * float(np.clip((0.80 - progress) / 0.60, 0.0, 1.0))
        throughput_bonus = (throughput_weight * np.abs(charge_kw) * np.float32(dt)).astype(np.float32)
        reward = (storage_profit - action_penalty - soc_regularization + throughput_bonus - safe_v - safe_line - safe_trafo).astype(np.float32)
        self.soc = np.clip(next_soc, self.cfg.env.soc_min, self.cfg.env.soc_max).astype(np.float32)
        self.cur_step += 1; self.train_step_calls += int(self.split == "train"); done = self.cur_step >= self.steps
        obs = self._obs() if not done else self._terminal_obs()
        info = {"episode_done": done, "storage_profit_eur": float(np.sum(storage_profit)), "voltage_violation_count": int(np.sum(v_viol > 0.0)), "min_vm_pu": float(np.min(vm)), "max_vm_pu": float(np.max(vm)), "trafo_loading_pct": trafo_pct.astype(np.float32), "line_loading_pct": line_pct.astype(np.float32), "vm_pu": vm, "soc": self.soc.copy(), "net_load": net.astype(np.float32), "pv_effective": pv_effective.astype(np.float32), "pv_curtail": (pv - pv_effective).astype(np.float32), "madrl_r_inc": storage_profit.astype(np.float32), "madrl_r_action_penalty": action_penalty, "madrl_r_soc_regularization": soc_regularization, "madrl_r_throughput_bonus": throughput_bonus, "madrl_r_safe_v": safe_v.astype(np.float32), "madrl_r_safe_line": safe_line, "madrl_r_safe_trafo": safe_trafo, "madrl_r_safe_total": (safe_v + safe_line + safe_trafo).astype(np.float32), "madrl_r_total_internal": reward.astype(np.float32), "madrl_throughput_kwh": (np.abs(charge_kw) * np.float32(dt)).astype(np.float32), "madrl_throughput_bonus_weight": throughput_weight}
        return obs, reward.astype(np.float32), done, False, info

    def close(self) -> None:
        return None


def build_env(cfg: Cfg, mode: str, forecast_mode: str, share_data: ShareData) -> GridEnv:
    return GridEnv(replace(cfg, forecast=replace(cfg.forecast, mode=forecast_mode)), split="train" if mode == "train" else "eval", forecast_mode=forecast_mode, share_data=share_data)
