"""Grid-aware multi-agent storage environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    try:
        import gym
        from gym import spaces
    except ModuleNotFoundError:
        class _FallbackEnv:
            def __init__(self) -> None:
                self.np_random = np.random.default_rng()

            def reset(self, *, seed: int | None = None):
                self.np_random = np.random.default_rng(seed)
                return None

        class _FallbackBox:
            def __init__(self, low, high, shape, dtype) -> None:
                self.low = low
                self.high = high
                self.shape = tuple(shape)
                self.dtype = dtype

        class _FallbackDict(dict):
            def __init__(self, mapping) -> None:
                super().__init__(mapping)
                self.spaces = dict(mapping)

        class _FallbackSpaces:
            Box = _FallbackBox
            Dict = _FallbackDict

        class _FallbackGym:
            Env = _FallbackEnv

        gym = _FallbackGym()
        spaces = _FallbackSpaces()


class GridEnv(gym.Env):
    """Multi-agent storage environment with power-flow constraints."""

    metadata = {"render_modes": []}
    _COMPACT_INFO_KEYS = (
        "vm_pu",
        "line_loading_pct",
        "trafo_loading_pct",
        "load",
        "pv",
        "p_lower",
        "p_upper",
        "bus_v_excess",
        "line_excess",
        "trafo_excess",
    )

    def __init__(
        self,
        cfg: Any,
        mode: str = "train",
        dataset: Any | None = None,
        reward_fn: Any | None = None,
        forecaster: Any | None = None,
        obs_builder: Any | None = None,
        grid_core: Any | None = None,
        data_path: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        env_cfg = cfg.env
        self.n = int(env_cfg.num_agents)
        self.episode_length = int(env_cfg.episode_limit)
        self.future_horizon = int(env_cfg.future_horizon)
        self.c_bat = float(env_cfg.battery_capacity)
        self.p_max = float(env_cfg.max_charge_rate)
        self.eff = float(env_cfg.efficiency)
        self.gamma = float(cfg.algo.gamma)
        self.init_soc = float(env_cfg.init_soc)
        self.dt = float(env_cfg.dt)
        self.storage_power_scale = float(getattr(env_cfg, "storage_power_scale", 1.0))
        self.storage_capacity_scale = float(getattr(env_cfg, "storage_capacity_scale", 1.0))
        runtime_seed = getattr(getattr(cfg, "runtime", None), "seed", None)
        self._default_seed = None if runtime_seed is None else int(runtime_seed)
        self._seeded_once = False

        self.soc_min = float(env_cfg.soc_min)
        self.soc_max = float(env_cfg.soc_max)
        self.soc_target = float(env_cfg.soc_target)
        if not 0.0 <= self.soc_min <= self.soc_max <= 1.0:
            raise ValueError(
                f"Invalid SoC range: soc_min={self.soc_min}, soc_max={self.soc_max}."
            )
        self.init_soc = float(np.clip(self.init_soc, self.soc_min, self.soc_max))

        self._grid_cfg = cfg.grid
        self._w_v_pen = float(cfg.grid.w_v_pen)
        self._w_line_pen = float(cfg.grid.w_line_pen)
        self._w_trafo_pen = float(cfg.grid.w_trafo_pen)
        self._reward_payload_defaults = {
            "w_global_safe": float(getattr(cfg.reward, "w_global_safe", 1.0)),
            "w_sens_credit": float(getattr(cfg.reward, "w_sens_credit", 0.2)),
            "sens_credit_scale": float(getattr(cfg.reward, "sens_credit_scale", 0.05)),
        }

        self._train_compact_info = bool(getattr(self._grid_cfg, "train_compact_info", True))
        self._sensitivity_delta_kw = float(getattr(self._grid_cfg, "sensitivity_delta_kw", 1.0))
        self._sensitivity_max_staleness = int(
            getattr(self._grid_cfg, "sensitivity_max_staleness_steps", 32)
        )
        self._sensitivity_trigger_action_delta = float(
            getattr(self._grid_cfg, "sensitivity_trigger_action_delta_kw", 1.0)
        )
        self._sensitivity_trigger_load_delta = float(
            getattr(self._grid_cfg, "sensitivity_trigger_load_delta_kw", 2.0)
        )
        self._sensitivity_trigger_psi_delta = float(
            getattr(self._grid_cfg, "sensitivity_trigger_psi_delta", 0.001)
        )
        self._sensitivity_trigger_on_pf_recovery = bool(
            getattr(self._grid_cfg, "sensitivity_trigger_on_pf_recovery", True)
        )
        self._sensitivity_trigger_on_violation_change = bool(
            getattr(self._grid_cfg, "sensitivity_trigger_on_violation_change", True)
        )

        self._sensitivity_cache: dict[str, np.ndarray] | None = None
        self._sensitivity_steps_since_update = 0
        self._prev_e_bat: np.ndarray | None = None
        self._prev_net_load: np.ndarray | None = None
        self._prev_psi_total = 0.0
        self._prev_pf_converged = True
        self._prev_has_violations = False
        self._last_sensitivity_trigger = ""

        from envs.rewards import get_reward_fn

        self.reward_fn = reward_fn if reward_fn is not None else get_reward_fn(cfg.reward.type, cfg)

        if dataset is None:
            from data.loaders.csv_prosumer import CsvProsumerDataset

            data_root = Path(__file__).resolve().parent.parent / "data"
            resolved_data_path = Path(data_path) if data_path is not None else data_root / (
                "simbench_2016_train.csv" if mode == "train" else "simbench_2016_test.csv"
            )
            dataset = CsvProsumerDataset(
                data_path=resolved_data_path,
                episode_length=self.episode_length,
                n_agents=self.n,
                metadata_path=resolved_data_path.with_name("simbench_2016_metadata.json"),
            )
        self._dataset = dataset

        if forecaster is None:
            from predictors.oracle import PerfectForecaster

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

        if grid_core is None:
            raise ValueError("GridEnv requires an attached GridCore instance.")
        self._grid_core = grid_core

        self.num_available_episodes = self._dataset.num_episodes()
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        self.signals: dict[str, np.ndarray] = {}
        self.episode_meta: dict[str, Any] = {}
        self._last_episode_idx: int | None = None
        self.ep_price = np.zeros((self.episode_length,), dtype=np.float32)
        self.ep_load = np.zeros((self.episode_length, self.n), dtype=np.float32)
        self.ep_pv = np.zeros((self.episode_length, self.n), dtype=np.float32)

        self.agent_c_bat = np.full((self.n,), self.c_bat, dtype=np.float32)
        self.agent_p_max = np.full((self.n,), self.p_max, dtype=np.float32)
        self.agent_e_min = (self.soc_min * self.agent_c_bat).astype(np.float32)
        self.agent_e_max = (self.soc_max * self.agent_c_bat).astype(np.float32)
        self.e_min = self.agent_e_min.copy()
        self.e_max = self.agent_e_max.copy()

        self.vm_pu = np.ones(1, dtype=np.float32)

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
        self.agent_c_bat = (self.agent_c_bat * self.storage_capacity_scale).astype(np.float32)
        self.agent_p_max = (self.agent_p_max * self.storage_power_scale).astype(np.float32)
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
            raise ValueError(f"signals['price'] must have shape (T,), got {self.ep_price.shape}")
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

    def _build_reset_info(self, episode_idx: int) -> dict[str, Any]:
        return {
            "episode_idx": int(episode_idx),
            "available_signals": sorted(self.signals),
            "battery_capacity_kwh": self.agent_c_bat.astype(np.float32).copy(),
            "p_max": self.agent_p_max.astype(np.float32).copy(),
            "episode_meta": dict(self.episode_meta),
            "n_buses": int(getattr(self._grid_core, "n_buses", 0)),
            "n_lines": int(getattr(self._grid_core, "n_lines", 0)),
            "n_trafos": int(getattr(self._grid_core, "n_trafos", 0)),
        }

    def _reset_sensitivity_state(self) -> None:
        self._sensitivity_cache = None
        self._sensitivity_steps_since_update = 0
        self._prev_e_bat = None
        self._prev_net_load = None
        self._prev_psi_total = 0.0
        self._prev_pf_converged = True
        self._prev_has_violations = False
        self._last_sensitivity_trigger = ""

    def _try_update_sensitivity(self, p_batt_kw: np.ndarray, base_load_kw: np.ndarray) -> None:
        try:
            self._sensitivity_cache = self._grid_core.compute_sensitivity_snapshot(
                p_batt_kw,
                base_load_kw,
                delta_kw=self._sensitivity_delta_kw,
            )
        except Exception:
            pass

    def _should_update_sensitivity(
        self,
        pf_converged: bool,
        psi_v_raw: float,
        psi_line_raw: float,
        psi_trafo_raw: float,
        e_bat: np.ndarray,
        net_load: np.ndarray,
    ) -> tuple[bool, str]:
        if self._sensitivity_cache is None:
            return True, "no_cache"

        if self._sensitivity_trigger_on_pf_recovery and pf_converged and not self._prev_pf_converged:
            return True, "pf_recovery"

        psi_total = psi_v_raw + psi_line_raw + psi_trafo_raw
        if self._sensitivity_trigger_on_violation_change:
            has_violations = psi_total > 0.0
            if has_violations != self._prev_has_violations:
                return True, "violation_change"

        if abs(psi_total - self._prev_psi_total) > self._sensitivity_trigger_psi_delta:
            return True, "psi_delta"

        if self._prev_e_bat is not None:
            action_delta = float(np.max(np.abs(e_bat - self._prev_e_bat)))
            if action_delta > self._sensitivity_trigger_action_delta:
                return True, "action_delta"

        if self._prev_net_load is not None:
            load_delta = float(np.max(np.abs(net_load - self._prev_net_load)))
            if load_delta > self._sensitivity_trigger_load_delta:
                return True, "load_delta"

        if self._sensitivity_steps_since_update >= self._sensitivity_max_staleness:
            return True, "staleness"

        return False, ""

    def reset(
        self,
        episode_idx: int | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is None and not self._seeded_once:
            seed = self._default_seed
        super().reset(seed=seed)
        if seed is not None:
            self._seeded_once = True
        if options is not None and episode_idx is None:
            episode_idx = options.get("episode_idx")
        if episode_idx is None:
            episode_idx = int(self.np_random.integers(0, self.num_available_episodes))
        elif episode_idx < 0 or episode_idx >= self.num_available_episodes:
            raise IndexError(
                f"episode_idx={episode_idx} is out of range [0, {self.num_available_episodes - 1}]"
            )

        self._load_episode(int(episode_idx))
        self._last_episode_idx = int(episode_idx)
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        self.forecaster.reset()
        if hasattr(self.forecaster, "set_episode"):
            self.forecaster.set_episode(self.signals)

        base_load = np.asarray(self.ep_load[0], dtype=np.float32)
        base_pv = np.asarray(self.ep_pv[0], dtype=np.float32)
        self._grid_core.reset(base_load, base_pv)

        self._reset_sensitivity_state()
        base_net = (base_load - base_pv).astype(np.float32)
        self._try_update_sensitivity(np.zeros(self.n, dtype=np.float32), base_net)

        return self.obs_builder.build(self), self._build_reset_info(int(episode_idx))

    def _apply_storage_dynamics(self, actions: list[np.ndarray]) -> dict[str, np.ndarray]:
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

        return {
            "e_bat_req": e_bat_req.astype(np.float32),
            "e_bat": e_bat,
            "soc_t": soc_t,
            "soc_next": soc_next,
            "e_t": e_t.astype(np.float32),
            "e_next": e_next,
            "p_lower": p_lower.astype(np.float32),
            "p_upper": p_upper.astype(np.float32),
        }

    def _build_signal_state(self, t: int) -> dict[str, Any]:
        price_t = float(self.get_signal_step("price", t))
        load_t = np.asarray(self.get_signal_step("load", t), dtype=np.float32)
        pv_t = np.asarray(self.ep_pv[t], dtype=np.float32)
        base_net_load = (load_t - pv_t).astype(np.float32)
        grid_power = (base_net_load + 0.0).astype(np.float32)
        return {
            "price_t": price_t,
            "load_t": load_t,
            "pv_t": pv_t,
            "base_net_load": base_net_load,
            "net_load": grid_power,
            "mu_t": self._future_mean_price(t),
            "mu_next": self._future_mean_price(min(t + 1, self.episode_length - 1)),
        }

    def _run_power_flow(
        self,
        storage_state: dict[str, np.ndarray],
        signal_state: dict[str, Any],
    ) -> tuple[Any, str]:
        pf_result = self._grid_core.step(
            p_batt_kw=storage_state["e_bat"],
            base_load_kw=signal_state["base_net_load"],
        )
        pf_error = getattr(self._grid_core, "last_pf_error", "")
        self.vm_pu = pf_result.vm_pu
        signal_state["net_load"] = (
            signal_state["base_net_load"] + storage_state["e_bat"]
        ).astype(np.float32)
        return pf_result, pf_error

    def _build_step_payload(
        self,
        t: int,
        storage_state: dict[str, np.ndarray],
        signal_state: dict[str, Any],
        pf_result: Any,
        pf_error: str,
    ) -> dict[str, Any]:
        return {
            "t": int(t),
            **storage_state,
            "p_max": self.agent_p_max.astype(np.float32),
            "e_min": self.agent_e_min.astype(np.float32),
            "e_max": self.agent_e_max.astype(np.float32),
            "battery_capacity_kwh": self.agent_c_bat.astype(np.float32),
            **signal_state,
            "pf_converged": bool(pf_result.converged),
            "pf_error": pf_error,
            "vm_pu": pf_result.vm_pu,
            "agent_vm_pu": pf_result.agent_vm_pu,
            "line_loading_pct": pf_result.line_loading_pct,
            "trafo_loading_pct": pf_result.trafo_loading_pct,
            "v_violation": pf_result.v_violation,
            "line_violation": float(pf_result.line_violation),
            "trafo_violation": float(pf_result.trafo_violation),
            "l_violation": float(pf_result.l_violation),
            "n_buses": int(pf_result.n_buses),
            "n_lines": int(pf_result.n_lines),
            "n_trafos": int(pf_result.n_trafos),
            "psi_v_raw": float(pf_result.psi_v_raw),
            "psi_line_raw": float(pf_result.psi_line_raw),
            "psi_trafo_raw": float(pf_result.psi_trafo_raw),
            "bus_v_excess": pf_result.bus_v_excess,
            "bus_v_signed_indicator": pf_result.bus_v_signed_indicator,
            "line_excess": pf_result.line_excess,
            "trafo_excess": pf_result.trafo_excess,
            "sensitivity_snapshot": self._sensitivity_cache,
        }

    def _build_reward_state(self, step_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "e_bat_req": step_state["e_bat_req"],
            "e_bat": step_state["e_bat"],
            "soc_t": step_state["soc_t"],
            "soc_next": step_state["soc_next"],
            "e_t": step_state["e_t"],
            "e_next": step_state["e_next"],
            "e_min": step_state["e_min"],
            "e_max": step_state["e_max"],
            "p_max": step_state["p_max"],
            "price_t": step_state["price_t"],
            "net_load_t": step_state["base_net_load"],
            "mu_t": step_state["mu_t"],
            "mu_next": step_state["mu_next"],
            "gamma": self.gamma,
            "dt": self.dt,
            "vm_pu": step_state["vm_pu"],
            "agent_vm_pu": step_state["agent_vm_pu"],
            "v_violation": step_state["v_violation"],
            "line_violation": step_state["line_violation"],
            "trafo_violation": step_state["trafo_violation"],
            "l_violation": step_state["l_violation"],
            "w_v_pen": self._w_v_pen,
            "w_line_pen": self._w_line_pen,
            "w_trafo_pen": self._w_trafo_pen,
            "pf_converged": step_state["pf_converged"],
            "pf_error": step_state["pf_error"],
            "psi_v_raw": step_state["psi_v_raw"],
            "psi_line_raw": step_state["psi_line_raw"],
            "psi_trafo_raw": step_state["psi_trafo_raw"],
            "bus_v_excess": step_state["bus_v_excess"],
            "bus_v_signed_indicator": step_state["bus_v_signed_indicator"],
            "line_excess": step_state["line_excess"],
            "trafo_excess": step_state["trafo_excess"],
            "sensitivity_snapshot": step_state["sensitivity_snapshot"],
            **self._reward_payload_defaults,
        }

    def _build_violation_counts(self, step_state: dict[str, Any]) -> tuple[int, int, int]:
        n_v_viol = int(np.sum(step_state["v_violation"] > 0.0))
        line_limit = float(self._grid_cfg.line_max_loading_pct)
        n_line_viol = int(
            np.any(np.asarray(step_state["line_loading_pct"], dtype=np.float32) > line_limit)
        )
        n_trafo_viol = int(
            np.any(np.asarray(step_state["trafo_loading_pct"], dtype=np.float32) > line_limit)
        )
        return n_v_viol, n_line_viol, n_trafo_viol

    def _build_step_info(
        self,
        step_state: dict[str, Any],
        components: dict[str, np.ndarray],
        reward: np.ndarray,
        done: bool,
    ) -> dict[str, Any]:
        n_v_viol, n_line_viol, n_trafo_viol = self._build_violation_counts(step_state)
        info: dict[str, Any] = {
            "episode_done": done,
            "t": step_state["t"],
            "price": float(step_state["price_t"]),
            "load": step_state["load_t"].astype(np.float32),
            "pv": step_state["pv_t"].astype(np.float32),
            "base_net_load": step_state["base_net_load"].astype(np.float32),
            "net_load": step_state["net_load"].astype(np.float32),
            "e_bat_req": step_state["e_bat_req"].astype(np.float32),
            "e_bat": step_state["e_bat"].astype(np.float32),
            "p_lower": step_state["p_lower"].astype(np.float32),
            "p_upper": step_state["p_upper"].astype(np.float32),
            "p_max": step_state["p_max"].astype(np.float32),
            "e_min": step_state["e_min"].astype(np.float32),
            "e_max": step_state["e_max"].astype(np.float32),
            "battery_capacity_kwh": step_state["battery_capacity_kwh"].astype(np.float32),
            "soc_min": float(self.soc_min),
            "soc_max": float(self.soc_max),
            "soc_t": step_state["soc_t"],
            "soc_next": step_state["soc_next"],
            "available_signals": sorted(self.signals),
            **components,
            "reward": reward.astype(np.float32),
            "mu_t": float(step_state["mu_t"]),
            "mu_next": float(step_state["mu_next"]),
            "pf_converged": step_state["pf_converged"],
            "pf_error": step_state["pf_error"],
            "vm_pu": step_state["vm_pu"],
            "agent_vm_pu": step_state["agent_vm_pu"],
            "line_loading_pct": step_state["line_loading_pct"],
            "trafo_loading_pct": step_state["trafo_loading_pct"],
            "v_violation": step_state["v_violation"],
            "line_violation": step_state["line_violation"],
            "trafo_violation": step_state["trafo_violation"],
            "l_violation": step_state["l_violation"],
            "n_v_violations": n_v_viol,
            "n_l_violations": int(max(n_line_viol, n_trafo_viol)),
            "n_line_violations": n_line_viol,
            "n_t_violations": n_trafo_viol,
            "n_trafo_violations": n_trafo_viol,
            "n_buses": step_state["n_buses"],
            "n_lines": step_state["n_lines"],
            "n_trafos": step_state["n_trafos"],
            "psi_v_raw": step_state["psi_v_raw"],
            "psi_line_raw": step_state["psi_line_raw"],
            "psi_trafo_raw": step_state["psi_trafo_raw"],
            "bus_v_excess": step_state["bus_v_excess"],
            "line_excess": step_state["line_excess"],
            "trafo_excess": step_state["trafo_excess"],
        }
        if self.mode == "train" and self._train_compact_info:
            for key in self._COMPACT_INFO_KEYS:
                info.pop(key, None)
        return info

    def _maybe_refresh_sensitivity(self, step_state: dict[str, Any]) -> None:
        self._sensitivity_steps_since_update += 1
        should_update, trigger_reason = self._should_update_sensitivity(
            pf_converged=bool(step_state["pf_converged"]),
            psi_v_raw=float(step_state["psi_v_raw"]),
            psi_line_raw=float(step_state["psi_line_raw"]),
            psi_trafo_raw=float(step_state["psi_trafo_raw"]),
            e_bat=np.asarray(step_state["e_bat"], dtype=np.float32),
            net_load=np.asarray(step_state["base_net_load"], dtype=np.float32),
        )
        if should_update:
            self._try_update_sensitivity(step_state["e_bat"], step_state["base_net_load"])
            self._sensitivity_steps_since_update = 0
            self._last_sensitivity_trigger = trigger_reason

        self._prev_e_bat = np.asarray(step_state["e_bat"], dtype=np.float32).copy()
        self._prev_net_load = np.asarray(step_state["base_net_load"], dtype=np.float32).copy()
        psi_total = (
            float(step_state["psi_v_raw"])
            + float(step_state["psi_line_raw"])
            + float(step_state["psi_trafo_raw"])
        )
        self._prev_psi_total = psi_total
        self._prev_pf_converged = bool(step_state["pf_converged"])
        self._prev_has_violations = psi_total > 0.0

    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[dict[str, np.ndarray], list[float], list[bool], list[bool], dict[str, Any]]:
        t = self.cur_step
        storage_state = self._apply_storage_dynamics(actions)
        signal_state = self._build_signal_state(t)
        pf_result, pf_error = self._run_power_flow(storage_state, signal_state)
        step_state = self._build_step_payload(t, storage_state, signal_state, pf_result, pf_error)

        reward_state = self._build_reward_state(step_state)
        reward_per_agent, components = self.reward_fn.compute(reward_state)
        reward = np.asarray(reward_per_agent, dtype=np.float32)

        self._maybe_refresh_sensitivity(step_state)

        self.soc = storage_state["soc_next"]
        self.cur_step += 1
        done = self.cur_step >= self.episode_length
        terminated_n = [False] * self.n
        truncated_n = [done] * self.n
        obs = self.obs_builder.zeros(self.n) if done else self.obs_builder.build(self)
        info = self._build_step_info(step_state, components, reward, done)
        return obs, reward.tolist(), terminated_n, truncated_n, info

    def close(self) -> None:
        pass


