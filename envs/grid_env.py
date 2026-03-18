"""Grid-aware multi-agent energy storage environment.

``GridEnv`` is a drop-in sibling of ``EnergyStorageEnv``.  It shares the
same constructor signature and the same downstream contracts (observation
schema, info dict keys, action space), but additionally:

1. Calls ``GridCore.step()`` after resolving battery physics to compute
   bus voltages and line flows.
2. Passes grid fields into ``env_state`` so that ``GridCompositeReward``
   can compute voltage and line penalties.
3. Adds grid-specific keys to the returned ``info`` dict (see docstring in
   ``step()`` for the full list).

Reward strategy (Phase 1)
--------------------------
Shared reward: the per-agent reward vector is set to the mean total reward
broadcast to all agents.  This is appropriate for a cooperative energy
community where grid feasibility is a collective responsibility.

Downstream compatibility
-------------------------
The following attributes are accessed by training/evaluation infrastructure
and must be present on every env instance::

    self.n                  int
    self.soc                np.ndarray (n_agents,)
    self.obs_builder        ObservationBuilder
    self.observation_schema dict
    self.observation_layout dict
    self.action_space       list[spaces.Box]
    self.reward_fn          RewardFn

Methods accessed by observation feature lambdas::

    get_signal(signal_name)
    get_signal_step(signal_name, step=None)
    get_signal_history(signal_name)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces


class GridEnv(gym.Env):
    """Battery-storage environment with distribution-grid power-flow constraints."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        cfg: Any,
        mode: str = "train",
        dataset: Any | None = None,
        reward_fn: Any | None = None,
        forecaster: Any | None = None,
        obs_builder: Any | None = None,
        grid_core: Any | None = None,
        data_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        env_cfg = cfg.env
        reward_cfg = cfg.reward

        self.n = int(env_cfg.num_agents)
        self.episode_length = int(env_cfg.episode_limit)
        self.future_horizon = int(env_cfg.future_horizon)
        self.c_bat = float(env_cfg.battery_capacity)
        self.p_max = float(env_cfg.max_charge_rate)
        self.eff = float(env_cfg.efficiency)
        self.gamma = float(cfg.algo.gamma)
        self.init_soc = float(env_cfg.init_soc)
        self.dt = float(env_cfg.dt)

        self.soc_min = float(env_cfg.soc_min)
        self.soc_max = float(env_cfg.soc_max)
        self.soc_target = float(env_cfg.soc_target)
        if not 0.0 <= self.soc_min <= self.soc_max <= 1.0:
            raise ValueError(
                f"Invalid SoC range: soc_min={self.soc_min}, soc_max={self.soc_max}."
            )
        self.init_soc = float(np.clip(self.init_soc, self.soc_min, self.soc_max))

        # Grid config.
        self._grid_cfg = cfg.grid
        self._w_v_pen = float(cfg.grid.w_v_pen)
        self._w_l_pen = float(cfg.grid.w_l_pen)

        from common.rewards import get_reward_fn

        self.reward_fn = reward_fn if reward_fn is not None else get_reward_fn(cfg.reward.type, cfg)

        if dataset is None:
            from datasets.csv_price_load import CsvPriceLoadDataset

            if data_path is None:
                data_dir = Path(__file__).resolve().parent.parent / "data"
                data_path = str(
                    data_dir / ("train_prices.csv" if mode == "train" else "test_prices.csv")
                )
            dataset = CsvPriceLoadDataset(data_path, self.episode_length, self.n)
        self._dataset = dataset

        if forecaster is None:
            from forecast.oracle import PerfectForecaster

            forecaster = PerfectForecaster()
        self.forecaster = forecaster

        if obs_builder is None:
            from envs.observation.default_builder import DefaultObservationBuilder

            obs_builder = DefaultObservationBuilder(
                local_features=cfg.obs.local_features,
                sequence_features=cfg.obs.sequence_features,
                future_horizon=self.future_horizon,
                adjacency_type=cfg.obs.adjacency_type,
            )
        self.obs_builder = obs_builder
        self.observation_schema = self.obs_builder.get_schema(self.n)
        self.observation_layout = self.obs_builder.get_layout(self.n)

        # GridCore — may be None in tests where it is not needed.
        self._grid_core = grid_core

        self.num_available_episodes = self._dataset.num_episodes()
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        self.signals: dict[str, np.ndarray] = {}
        self.episode_meta: dict[str, Any] = {}
        self.ep_price = np.zeros((self.episode_length,), dtype=np.float32)
        self.ep_load = np.zeros((self.episode_length, self.n), dtype=np.float32)
        self.ep_pv = np.zeros((self.episode_length, self.n), dtype=np.float32)

        self.agent_c_bat = np.full((self.n,), self.c_bat, dtype=np.float32)
        self.agent_p_max = np.full((self.n,), self.p_max, dtype=np.float32)
        self.agent_e_min = (self.soc_min * self.agent_c_bat).astype(np.float32)
        self.agent_e_max = (self.soc_max * self.agent_c_bat).astype(np.float32)
        self.e_min = self.agent_e_min.copy()
        self.e_max = self.agent_e_max.copy()

        # Latest power-flow result — updated by step(), used by obs features.
        self.vm_pu: np.ndarray = np.ones(1, dtype=np.float32)  # placeholder

        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for _ in range(self.n)
        ]
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
                for key, shape in self.observation_schema.items()
            }
        )

    # ------------------------------------------------------------------
    # Signal access (called by ObservationBuilder feature lambdas)
    # ------------------------------------------------------------------

    def _canonicalize_signal(self, name: str, value: np.ndarray) -> np.ndarray:
        signal = np.asarray(value, dtype=np.float32)
        if signal.shape[0] != self.episode_length:
            raise ValueError(
                f"signal '{name}' first dimension should match episode_length="
                f"{self.episode_length}, got {signal.shape[0]}"
            )
        return signal

    def _resolve_meta_vector(self, key: str, default_value: float) -> np.ndarray:
        raw_value = self.episode_meta.get(key)
        if raw_value is None:
            return np.full((self.n,), default_value, dtype=np.float32)
        values = np.asarray(raw_value, dtype=np.float32).reshape(-1)
        if values.size != self.n:
            raise ValueError(
                f"episode meta '{key}' should have {self.n} values, got shape {values.shape}"
            )
        fallback = np.full((self.n,), default_value, dtype=np.float32)
        return np.where(values > 0.0, values, fallback).astype(np.float32)

    def _apply_episode_storage_config(self) -> None:
        self.agent_c_bat = self._resolve_meta_vector("ess_capacity_kwh", self.c_bat)
        self.agent_p_max = self._resolve_meta_vector("ess_power_kw", self.p_max)
        self.agent_e_min = (self.soc_min * self.agent_c_bat).astype(np.float32)
        self.agent_e_max = (self.soc_max * self.agent_c_bat).astype(np.float32)
        self.e_min = self.agent_e_min.copy()
        self.e_max = self.agent_e_max.copy()

    def _load_episode(self, episode_idx: int) -> None:
        episode_data = self._dataset.get_episode(episode_idx)
        raw_signals = episode_data.get("signals", {})
        if "price" not in raw_signals or "load" not in raw_signals:
            raise KeyError("Environment requires signals['price'] and signals['load'].")

        self.signals = {
            name: self._canonicalize_signal(name, signal)
            for name, signal in raw_signals.items()
        }
        self.episode_meta = dict(episode_data.get("meta", {}))

        self.ep_price = self.get_signal("price")
        self.ep_load = self.get_signal("load")
        self.ep_pv = (
            self.get_signal("pv")
            if "pv" in self.signals
            else np.zeros((self.episode_length, self.n), dtype=np.float32)
        )
        if self.ep_price.ndim != 1:
            raise ValueError(
                f"signals['price'] must have shape (T,), got {self.ep_price.shape}"
            )
        if self.ep_load.shape != (self.episode_length, self.n):
            raise ValueError(
                f"signals['load'] must have shape {(self.episode_length, self.n)}, "
                f"got {self.ep_load.shape}"
            )
        if self.ep_pv.shape != (self.episode_length, self.n):
            raise ValueError(
                f"signals['pv'] must have shape {(self.episode_length, self.n)}, "
                f"got {self.ep_pv.shape}"
            )
        self._apply_episode_storage_config()

    def get_signal(self, signal_name: str) -> np.ndarray:
        if signal_name not in self.signals:
            available = sorted(self.signals)
            raise KeyError(
                f"Current episode does not contain signal '{signal_name}'. Available: {available}"
            )
        return self.signals[signal_name]

    def get_signal_step(self, signal_name: str, step: int | None = None) -> Any:
        step = self.cur_step if step is None else int(step)
        signal = self.get_signal(signal_name)
        value = signal[step]
        if signal.ndim == 1:
            return float(value)
        return np.asarray(value, dtype=np.float32)

    def get_signal_history(self, signal_name: str) -> np.ndarray:
        signal = self.get_signal(signal_name)
        return signal[: self.cur_step + 1].copy()

    def _future_mean_price(self, t: int) -> float:
        start = t + 1
        end = min(t + 1 + self.future_horizon, self.episode_length)
        if start >= self.episode_length:
            return float(self.ep_price[min(t, self.episode_length - 1)])
        seg = self.ep_price[start:end]
        if seg.size == 0:
            return float(self.ep_price[min(t, self.episode_length - 1)])
        return float(np.mean(seg))

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, episode_idx: Optional[int] = None) -> dict[str, np.ndarray]:
        if episode_idx is None:
            episode_idx = int(np.random.randint(0, self.num_available_episodes))
        elif episode_idx < 0 or episode_idx >= self.num_available_episodes:
            raise IndexError(
                f"episode_idx={episode_idx} is out of range [0, {self.num_available_episodes - 1}]"
            )

        self._load_episode(int(episode_idx))
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        self.forecaster.reset()
        if hasattr(self.forecaster, "set_episode"):
            self.forecaster.set_episode(self.signals)

        if self._grid_core is not None:
            base_load = np.asarray(self.ep_load[0], dtype=np.float32)
            base_pv = np.asarray(self.ep_pv[0], dtype=np.float32)
            self._grid_core.reset(base_load, base_pv)

        return self.obs_builder.build(self)

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[dict[str, np.ndarray], List[float], List[bool], Dict]:
        t = self.cur_step

        # ----------------------------------------------------------
        # Battery physics (identical to EnergyStorageEnv.step())
        # ----------------------------------------------------------
        action_array = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)[:, 0]
        action_array = np.clip(action_array, -1.0, 1.0)
        e_bat_req = action_array * self.agent_p_max

        soc_t = self.soc.copy().astype(np.float32)
        e_t = soc_t * self.agent_c_bat
        eff = max(self.eff, 1e-6)

        p_max_chg = np.minimum(
            self.agent_p_max,
            np.maximum(0.0, (self.agent_e_max - e_t) / (eff * self.dt)),
        )
        p_max_dis = np.minimum(
            self.agent_p_max,
            np.maximum(0.0, (e_t - self.agent_e_min) * eff / self.dt),
        )

        p_lower = -p_max_dis
        p_upper = p_max_chg
        e_bat = np.clip(e_bat_req, p_lower, p_upper).astype(np.float32)

        delta_e = np.where(e_bat >= 0.0, e_bat * eff, e_bat / eff) * self.dt
        e_next = np.clip(e_t + delta_e, self.agent_e_min, self.agent_e_max).astype(np.float32)
        soc_next = (e_next / self.agent_c_bat).astype(np.float32)

        # ----------------------------------------------------------
        # Background signals for this timestep
        # ----------------------------------------------------------
        price_t = float(self.get_signal_step("price", t))
        load_t = np.asarray(self.get_signal_step("load", t), dtype=np.float32)
        pv_t = np.asarray(self.ep_pv[t], dtype=np.float32)
        net_load_t = (load_t - pv_t).astype(np.float32)

        mu_t = self._future_mean_price(t)
        mu_next = self._future_mean_price(min(t + 1, self.episode_length - 1))

        # ----------------------------------------------------------
        # Power flow step
        # ----------------------------------------------------------
        if self._grid_core is not None:
            pf_result = self._grid_core.step(
                p_batt_kw=e_bat,
                base_load_kw=net_load_t,
            )
            self.vm_pu = pf_result.vm_pu
        else:
            # No grid core (e.g. during unit tests) — zero-violation fallback.
            from grid.core.grid_types import GridStepResult

            pf_result = GridStepResult(
                converged=True,
                vm_pu=np.ones(1, dtype=np.float32),
                va_degree=np.zeros(1, dtype=np.float32),
                line_loading_pct=np.zeros(1, dtype=np.float32),
                p_mw_from=np.zeros(1, dtype=np.float32),
                agent_vm_pu=np.ones(self.n, dtype=np.float32),
                v_violation=np.zeros(self.n, dtype=np.float32),
                l_violation=0.0,
                n_buses=1,
                n_lines=1,
            )
            self.vm_pu = pf_result.vm_pu

        # ----------------------------------------------------------
        # Build env_state and compute reward
        # ----------------------------------------------------------
        env_state = {
            "e_bat_req": e_bat_req,
            "e_bat": e_bat,
            "soc_t": soc_t,
            "soc_next": soc_next,
            "e_t": e_t,
            "e_next": e_next,
            "e_min": self.agent_e_min.copy(),
            "e_max": self.agent_e_max.copy(),
            "p_max": self.agent_p_max.copy(),
            "price_t": price_t,
            "net_load_t": net_load_t,
            "mu_t": mu_t,
            "mu_next": mu_next,
            "gamma": self.gamma,
            "dt": self.dt,
            # Grid-specific fields consumed by GridCompositeReward.
            "vm_pu": pf_result.vm_pu,
            "agent_vm_pu": pf_result.agent_vm_pu,
            "v_violation": pf_result.v_violation,
            "l_violation": pf_result.l_violation,
            "w_v_pen": self._w_v_pen,
            "w_l_pen": self._w_l_pen,
            "pf_converged": pf_result.converged,
        }
        reward_per_agent, components = self.reward_fn.compute(env_state)

        # Phase 1: shared reward — mean broadcast to all agents.
        shared = float(np.mean(reward_per_agent))
        reward = np.full(self.n, shared, dtype=np.float32)

        grid_power = (net_load_t + e_bat).astype(np.float32)

        # ----------------------------------------------------------
        # Advance state
        # ----------------------------------------------------------
        self.soc = soc_next
        self.cur_step += 1
        done = self.cur_step >= self.episode_length
        done_n = [done] * self.n

        obs = self.obs_builder.zeros(self.n) if done else self.obs_builder.build(self)

        # ----------------------------------------------------------
        # Build info dict (superset of EnergyStorageEnv info keys)
        # ----------------------------------------------------------
        n_v_viol = int(np.sum(pf_result.v_violation > 0.0))
        n_l_viol = int(pf_result.l_violation > 0.0)

        info: dict = {
            "episode_done": done,
            "t": int(t),
            "price": float(price_t),
            "load": load_t.astype(np.float32),
            "pv": pv_t.astype(np.float32),
            "base_net_load": net_load_t.astype(np.float32),
            "net_load": grid_power.astype(np.float32),
            "e_bat_req": e_bat_req.astype(np.float32),
            "e_bat": e_bat.astype(np.float32),
            "p_lower": p_lower.astype(np.float32),
            "p_upper": p_upper.astype(np.float32),
            "p_max": self.agent_p_max.astype(np.float32),
            "e_min": self.agent_e_min.astype(np.float32),
            "e_max": self.agent_e_max.astype(np.float32),
            "battery_capacity_kwh": self.agent_c_bat.astype(np.float32),
            "soc_min": float(self.soc_min),
            "soc_max": float(self.soc_max),
            "soc_t": soc_t,
            "soc_next": soc_next,
            "available_signals": sorted(self.signals),
            # Reward components (from ComponentMeta keys).
            **components,
            "reward": reward,
            "mu_t": float(mu_t),
            "mu_next": float(mu_next),
            # Grid-specific fields.
            "pf_converged": pf_result.converged,
            "vm_pu": pf_result.vm_pu,
            "agent_vm_pu": pf_result.agent_vm_pu,
            "line_loading_pct": pf_result.line_loading_pct,
            "v_violation": pf_result.v_violation,
            "l_violation": float(pf_result.l_violation),
            "n_v_violations": n_v_viol,
            "n_l_violations": n_l_viol,
        }
        return obs, reward.tolist(), done_n, info

    def close(self) -> None:
        pass
