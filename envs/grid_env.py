from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("GridEnv requires 'gymnasium'. Install gymnasium==0.29.1 in the active environment instead of falling back to legacy Gym.") from exc
from predictors.shared_data import PrecomputedObservationStore
from controllers.action_feasibility import build_safety_local_numpy, validate_executed_actions_numpy
from envs.grid.deployments import resolve_fixed_battery_spec
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY, WHOLESALE_PRICE_SIGNAL, derive_import_price, get_import_price_markup
class GridEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, cfg: Any, mode: str='train', dataset: Any | None=None, reward_fn: Any | None=None, forecaster: Any | None=None, obs_builder: Any | None=None, grid_core: Any | None=None, data_path: str | None=None, precomputed_data_dir: str | Path | None=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        env_cfg = cfg.env
        self.n = int(env_cfg.num_agents)
        self.episode_length = int(env_cfg.episode_limit)
        self.future_horizon = int(env_cfg.future_horizon)
        self.eff = float(env_cfg.efficiency)
        self.gamma = float(cfg.algo.gamma)
        self.init_soc = float(env_cfg.init_soc)
        self.dt = float(env_cfg.dt)
        self._fixed_capacity_kwh: np.ndarray | None = None
        self._fixed_p_max_kw: np.ndarray | None = None
        fixed_capacity_kwh, _, fixed_p_max_kw = resolve_fixed_battery_spec(env_cfg.battery_capacity, env_cfg.max_charge_rate, n_agents=self.n)
        self._fixed_capacity_kwh = np.asarray(fixed_capacity_kwh, dtype=np.float32)
        self._fixed_p_max_kw = np.asarray(fixed_p_max_kw, dtype=np.float32)
        self.c_bat = float(np.max(self._fixed_capacity_kwh))
        self.p_max = float(np.max(self._fixed_p_max_kw))
        runtime_seed = getattr(getattr(cfg, 'runtime', None), 'seed', None)
        self._default_seed = None if runtime_seed is None else int(runtime_seed)
        self._seeded_once = False
        self.soc_min = float(env_cfg.soc_min)
        self.soc_max = float(env_cfg.soc_max)
        self.soc_target = float(env_cfg.soc_target)
        if not 0.0 <= self.soc_min <= self.soc_max <= 1.0:
            raise ValueError(f'Invalid SoC range: soc_min={self.soc_min}, soc_max={self.soc_max}.')
        self.init_soc = float(np.clip(self.init_soc, self.soc_min, self.soc_max))
        self._grid_cfg = cfg.grid
        from envs.rewards import NormalReward
        self.reward_fn = reward_fn if reward_fn is not None else NormalReward(cfg)
        self.import_price_markup_eur_per_kwh = get_import_price_markup(cfg)
        if dataset is None:
            from data.loaders.registry import build_dataset
            if data_path is not None:
                cfg.data.data_dir = str(Path(data_path).parent)
            dataset = build_dataset(cfg, mode=mode)
        self._dataset = dataset
        if forecaster is None and precomputed_data_dir is None:
            from predictors.oracle import PerfectForecaster
            forecaster = PerfectForecaster()
        self.forecaster = forecaster
        if obs_builder is None:
            from envs.observation.default_builder import DefaultObservationBuilder
            from envs.observation.normalization import build_observation_normalizer
            obs_builder = DefaultObservationBuilder(local_features=cfg.obs.local_features, sequence_features=cfg.obs.sequence_features, future_horizon=self.future_horizon, adjacency_type=cfg.obs.adjacency_type, normalizer=build_observation_normalizer(cfg))
        elif getattr(obs_builder, 'normalizer', None) is None and bool(getattr(cfg.obs, 'normalization_enabled', False)):
            from envs.observation.normalization import build_observation_normalizer
            obs_builder.normalizer = build_observation_normalizer(cfg)
        self.obs_builder = obs_builder
        self.observation_schema = self.obs_builder.get_schema(self.n)
        self.observation_layout = self.obs_builder.get_layout(self.n)
        if grid_core is None:
            raise ValueError('GridEnv requires an attached GridCore instance.')
        self._grid_core = grid_core
        self.num_available_episodes = self._dataset.num_episodes()
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)
        self.signals: dict[str, np.ndarray] = {}
        self.history_signals: dict[str, np.ndarray] = {}
        self.history_timestamps: list[str] = []
        self.episode_meta: dict[str, Any] = {}
        self._last_episode_idx: int | None = None
        self._precomputed_data_dir = None if precomputed_data_dir is None else Path(precomputed_data_dir).resolve()
        self._precomputed_store = None if self._precomputed_data_dir is None else PrecomputedObservationStore(self._precomputed_data_dir)
        self._episode_precomputed: dict[str, np.ndarray] = {}
        self.ep_wholesale_price = np.zeros((self.episode_length,), dtype=np.float32)
        self.ep_load = np.zeros((self.episode_length, self.n), dtype=np.float32)
        self.ep_pv = np.zeros((self.episode_length, self.n), dtype=np.float32)
        self._local_mpc_solver_cache: dict[tuple[object, ...], Any] = {}
        self._local_mpc_stats: dict[str, float] = {}
        self.agent_c_bat = self._fixed_capacity_kwh.copy() if self._fixed_capacity_kwh is not None else np.full((self.n,), self.c_bat, dtype=np.float32)
        self.agent_p_max = self._fixed_p_max_kw.copy() if self._fixed_p_max_kw is not None else np.full((self.n,), self.p_max, dtype=np.float32)
        self.agent_e_min = (self.soc_min * self.agent_c_bat).astype(np.float32)
        self.agent_e_max = (self.soc_max * self.agent_c_bat).astype(np.float32)
        self.e_min = self.agent_e_min.copy()
        self.e_max = self.agent_e_max.copy()
        self.vm_pu = np.ones(1, dtype=np.float32)
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for _ in range(self.n)]
        self.observation_space = spaces.Dict({key: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32) for key, shape in self.observation_schema.items()})
    def _canonicalize_signal(self, name: str, value: np.ndarray) -> np.ndarray:
        signal = np.asarray(value, dtype=np.float32)
        if signal.shape[0] != self.episode_length:
            raise ValueError(f"signal '{name}' first dimension should match episode_length={self.episode_length}, got {signal.shape[0]}")
        return signal
    def _canonicalize_history_signal(self, name: str, value: np.ndarray, active_signal: np.ndarray) -> np.ndarray:
        history = np.asarray(value, dtype=np.float32)
        if active_signal.ndim == 1:
            if history.ndim != 1:
                raise ValueError(f"history signal '{name}' should be 1D to match active signal, got {history.shape}")
            return history.astype(np.float32, copy=False)
        if history.ndim != active_signal.ndim:
            raise ValueError(f"history signal '{name}' rank should match active signal rank, got {history.shape} vs {active_signal.shape}")
        if history.shape[1:] != active_signal.shape[1:]:
            raise ValueError(f"history signal '{name}' trailing dimensions should match active signal, got {history.shape} vs {active_signal.shape}")
        return history.astype(np.float32, copy=False)
    def _require_meta_vector(self, key: str) -> np.ndarray:
        raw_value = self.episode_meta.get(key)
        if raw_value is None:
            raise ValueError(f"episode meta '{key}' must be present.")
        values = np.asarray(raw_value, dtype=np.float32).reshape(-1)
        if values.size != self.n:
            raise ValueError(f"episode meta '{key}' should have {self.n} values, got shape {values.shape}")
        if np.any(values <= 0.0):
            raise ValueError(f"episode meta '{key}' must contain positive values for all agents, got {values.tolist()}.")
        return values.astype(np.float32)
    def _apply_episode_storage_config(self) -> None:
        if self._fixed_capacity_kwh is None or self._fixed_p_max_kw is None:
            raise RuntimeError('Fixed battery mode requires precomputed capacity and power vectors.')
        self.agent_c_bat = self._fixed_capacity_kwh.copy()
        self.agent_p_max = self._fixed_p_max_kw.copy()
        self.agent_e_min = (self.soc_min * self.agent_c_bat).astype(np.float32)
        self.agent_e_max = (self.soc_max * self.agent_c_bat).astype(np.float32)
        self.e_min = self.agent_e_min.copy()
        self.e_max = self.agent_e_max.copy()
    def _load_episode(self, episode_idx: int) -> None:
        episode_data = self._dataset.get_episode(episode_idx)
        raw_signals = episode_data.get('signals', {})
        raw_history_signals = dict(episode_data.get('history_signals', {}))
        if WHOLESALE_PRICE_SIGNAL not in raw_signals or 'load' not in raw_signals:
            raise KeyError(f"Environment requires signals['{WHOLESALE_PRICE_SIGNAL}'] and signals['load'].")
        self.signals = {name: self._canonicalize_signal(name, signal) for name, signal in raw_signals.items()}
        self.history_signals = {}
        for name, signal in self.signals.items():
            raw_history = raw_history_signals.get(name)
            if raw_history is None:
                if signal.ndim == 1:
                    self.history_signals[name] = np.zeros((0,), dtype=np.float32)
                else:
                    self.history_signals[name] = np.zeros((0, *signal.shape[1:]), dtype=np.float32)
                continue
            self.history_signals[name] = self._canonicalize_history_signal(name, raw_history, signal)
        self.episode_meta = dict(episode_data.get('meta', {}))
        self.history_timestamps = [str(value) for value in list(episode_data.get('history_timestamps') or [])]
        history_length = int(episode_data.get('history_length', len(self.history_timestamps)))
        if history_length != len(self.history_timestamps):
            raise ValueError(f'history_length={history_length} should match history_timestamps size={len(self.history_timestamps)}')
        if history_length != int(self.history_signals[WHOLESALE_PRICE_SIGNAL].shape[0]):
            raise ValueError(f'history_length should match history_signals length, got {history_length} vs {self.history_signals[WHOLESALE_PRICE_SIGNAL].shape[0]}')
        self.ep_wholesale_price = self.get_signal(WHOLESALE_PRICE_SIGNAL)
        self.ep_load = self.get_signal('load')
        self.ep_pv = self.get_signal('pv') if 'pv' in self.signals else np.zeros((self.episode_length, self.n), dtype=np.float32)
        if self.ep_wholesale_price.ndim != 1:
            raise ValueError(f"signals['{WHOLESALE_PRICE_SIGNAL}'] must have shape (T,), got {self.ep_wholesale_price.shape}")
        if self.ep_load.shape != (self.episode_length, self.n):
            raise ValueError(f"signals['load'] must have shape {(self.episode_length, self.n)}, got {self.ep_load.shape}")
        if self.ep_pv.shape != (self.episode_length, self.n):
            raise ValueError(f"signals['pv'] must have shape {(self.episode_length, self.n)}, got {self.ep_pv.shape}")
        self._apply_episode_storage_config()
    def get_signal(self, signal_name: str) -> np.ndarray:
        if signal_name not in self.signals:
            available = sorted(self.signals)
            raise KeyError(f"Current episode does not contain signal '{signal_name}'. Available: {available}")
        return self.signals[signal_name]
    def get_signal_step(self, signal_name: str, step: int | None=None) -> Any:
        step = self.cur_step if step is None else int(step)
        signal = self.get_signal(signal_name)
        value = signal[step]
        if signal.ndim == 1:
            return float(value)
        return np.asarray(value, dtype=np.float32)
    def get_signal_history(self, signal_name: str) -> np.ndarray:
        signal = self.get_signal(signal_name)
        prefix = np.asarray(self.history_signals.get(signal_name), dtype=np.float32)
        current = signal[:self.cur_step + 1].copy()
        if prefix.size == 0:
            return current
        return np.concatenate([prefix, current], axis=0).astype(np.float32, copy=False)
    def get_signal_history_timestamps(self) -> list[str]:
        timestamps = list(dict(self.episode_meta).get('timestamps') or [])
        end_idx = max(0, int(self.cur_step) + 1)
        return [*self.history_timestamps, *[str(timestamp) for timestamp in timestamps[:end_idx]]]
    def get_combined_episode_signals(self) -> dict[str, np.ndarray]:
        combined: dict[str, np.ndarray] = {}
        for name, signal in self.signals.items():
            prefix = np.asarray(self.history_signals.get(name), dtype=np.float32)
            if prefix.size == 0:
                combined[name] = np.asarray(signal, dtype=np.float32).copy()
            else:
                combined[name] = np.concatenate([prefix, signal], axis=0).astype(np.float32, copy=False)
        return combined
    def has_precomputed_observations(self) -> bool:
        return self._precomputed_store is not None
    def get_precomputed_local_feature(self, feature_name: str) -> np.ndarray:
        if feature_name != 'calendar_time':
            raise KeyError(f"Unknown precomputed local feature '{feature_name}'.")
        if 'calendar_time' not in self._episode_precomputed:
            raise KeyError("Precomputed observation data does not contain 'calendar_time'.")
        return np.asarray(self._episode_precomputed['calendar_time'][self.cur_step], dtype=np.float32)
    def get_precomputed_sequence_feature(self, feature_name: str) -> np.ndarray:
        cache_key = f'{feature_name}_seq'
        if cache_key not in self._episode_precomputed:
            raise KeyError(f"Precomputed observation data does not contain '{cache_key}'.")
        return np.asarray(self._episode_precomputed[cache_key][self.cur_step], dtype=np.float32)
    def _future_mean_price(self, t: int) -> float:
        if 'mu_t' in self._episode_precomputed:
            return float(np.asarray(self._episode_precomputed['mu_t'], dtype=np.float32)[int(t)])
        start = t + 1
        end = min(t + 1 + self.future_horizon, self.episode_length)
        if start >= self.episode_length:
            return float(self.ep_wholesale_price[min(t, self.episode_length - 1)])
        seg = self.ep_wholesale_price[start:end]
        if seg.size == 0:
            return float(self.ep_wholesale_price[min(t, self.episode_length - 1)])
        return float(np.mean(seg))
    def _future_mean_price_next(self, t: int) -> float:
        if 'mu_next' in self._episode_precomputed:
            return float(np.asarray(self._episode_precomputed['mu_next'], dtype=np.float32)[int(t)])
        return self._future_mean_price(min(t + 1, self.episode_length - 1))
    def _build_reset_info(self, episode_idx: int) -> dict[str, Any]:
        return {'episode_idx': int(episode_idx), 'available_signals': sorted(self.signals), 'battery_capacity_kwh': self.agent_c_bat.astype(np.float32).copy(), 'p_max': self.agent_p_max.astype(np.float32).copy(), IMPORT_PRICE_MARKUP_KEY: float(self.import_price_markup_eur_per_kwh), 'episode_meta': dict(self.episode_meta), 'n_buses': int(getattr(self._grid_core, 'n_buses', 0)), 'n_lines': int(getattr(self._grid_core, 'n_lines', 0)), 'n_trafos': int(getattr(self._grid_core, 'n_trafos', 0))}
    def reset(self, episode_idx: int | None=None, *, seed: int | None=None, options: dict[str, Any] | None=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is None and (not self._seeded_once):
            seed = self._default_seed
        super().reset(seed=seed)
        if seed is not None:
            self._seeded_once = True
        if options is not None and episode_idx is None:
            episode_idx = options.get('episode_idx')
        if episode_idx is None:
            episode_idx = int(self.np_random.integers(0, self.num_available_episodes))
        elif episode_idx < 0 or episode_idx >= self.num_available_episodes:
            raise IndexError(f'episode_idx={episode_idx} is out of range [0, {self.num_available_episodes - 1}]')
        self._load_episode(int(episode_idx))
        self._episode_precomputed = {} if self._precomputed_store is None else self._precomputed_store.episode(int(episode_idx))
        self._last_episode_idx = int(episode_idx)
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)
        if self.forecaster is not None:
            self.forecaster.reset()
            self.forecaster.set_episode(self.get_combined_episode_signals(), self.episode_meta)
        base_load = np.asarray(self.ep_load[0], dtype=np.float32)
        base_pv = np.asarray(self.ep_pv[0], dtype=np.float32)
        self._grid_core.reset(base_load, base_pv)
        return (self.obs_builder.build(self), self._build_reset_info(int(episode_idx)))
    def _split_action_components(self, actions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        action_array = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)
        if action_array.shape[1] != 2:
            raise ValueError(f'GridEnv expects exactly two action dimensions per agent: [battery_action, pv_action]. Got action shape {action_array.shape}.')
        if not np.all(np.isfinite(action_array)):
            raise ValueError('GridEnv received non-finite action values.')
        if np.any(action_array < -1.0 - 1e-05) or np.any(action_array > 1.0 + 1e-05):
            raise ValueError('GridEnv received action values outside [-1, 1]. Controllers must output already-feasible normalized actions.')
        battery_action = action_array[:, 0].astype(np.float32)
        pv_action = action_array[:, 1].astype(np.float32)
        return (battery_action, pv_action)
    def _current_safety_local(self) -> np.ndarray:
        return build_safety_local_numpy(soc=np.asarray(self.soc, dtype=np.float32), load_raw=np.asarray(self.get_signal_step('load'), dtype=np.float32), pv_raw=np.asarray(self.get_signal_step('pv'), dtype=np.float32), battery_capacity_kwh=np.asarray(self.agent_c_bat, dtype=np.float32), p_max_kw=np.asarray(self.agent_p_max, dtype=np.float32))
    def _apply_storage_dynamics(self, battery_action: np.ndarray) -> dict[str, np.ndarray]:
        e_bat_req = np.asarray(battery_action, dtype=np.float32) * self.agent_p_max
        soc_t = self.soc.copy().astype(np.float32)
        e_t = soc_t * self.agent_c_bat
        eff = max(self.eff, 1e-06)
        p_max_chg = np.minimum(self.agent_p_max, np.maximum(0.0, (self.agent_e_max - e_t) / (eff * self.dt)))
        p_max_dis = np.minimum(self.agent_p_max, np.maximum(0.0, (e_t - self.agent_e_min) * eff / self.dt))
        p_lower = -p_max_dis
        p_upper = p_max_chg
        invalid_battery = np.logical_or(e_bat_req < p_lower - 1e-05, e_bat_req > p_upper + 1e-05)
        if np.any(invalid_battery):
            bad_agent = int(np.nonzero(invalid_battery)[0][0])
            raise ValueError(f'GridEnv received a locally infeasible battery action from the controller for agent {bad_agent}: requested {e_bat_req[bad_agent]:.4f} kW, allowed range [{p_lower[bad_agent]:.4f}, {p_upper[bad_agent]:.4f}] kW.')
        e_bat = e_bat_req.astype(np.float32)
        delta_e = np.where(e_bat >= 0.0, e_bat * eff, e_bat / eff) * self.dt
        e_next = (e_t + delta_e).astype(np.float32)
        invalid_energy = np.logical_or(e_next < self.agent_e_min - 1e-05, e_next > self.agent_e_max + 1e-05)
        if np.any(invalid_energy):
            bad_agent = int(np.nonzero(invalid_energy)[0][0])
            raise ValueError(f'GridEnv received a battery action that would leave the local energy bounds for agent {bad_agent}: next energy {e_next[bad_agent]:.4f} kWh, allowed range [{self.agent_e_min[bad_agent]:.4f}, {self.agent_e_max[bad_agent]:.4f}] kWh.')
        e_next = np.clip(e_next, self.agent_e_min, self.agent_e_max).astype(np.float32)
        soc_next = (e_next / self.agent_c_bat).astype(np.float32)
        return {'e_bat_req': e_bat_req.astype(np.float32), 'e_bat': e_bat, 'soc_t': soc_t, 'soc_next': soc_next, 'e_t': e_t.astype(np.float32), 'e_next': e_next, 'p_lower': p_lower.astype(np.float32), 'p_upper': p_upper.astype(np.float32)}
    @staticmethod
    def _pv_action_to_utilization(pv_action: np.ndarray) -> np.ndarray:
        return (0.5 * (np.asarray(pv_action, dtype=np.float32) + 1.0)).astype(np.float32)
    @staticmethod
    def _utilization_to_pv_action(utilization: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * np.asarray(utilization, dtype=np.float32) - 1.0, -1.0, 1.0).astype(np.float32)
    def _apply_effective_pv_to_signal_state(self, signal_state: dict[str, Any], pv_effective: np.ndarray) -> None:
        pv_raw = np.asarray(signal_state['pv_raw'], dtype=np.float32)
        pv_effective_candidate = np.asarray(pv_effective, dtype=np.float32)
        invalid_effective = np.logical_or(pv_effective_candidate < -1e-05, pv_effective_candidate > pv_raw + 1e-05)
        if np.any(invalid_effective):
            bad_agent = int(np.nonzero(invalid_effective)[0][0])
            raise ValueError(f'GridEnv received an infeasible PV effective power from the controller for agent {bad_agent}: requested {pv_effective_candidate[bad_agent]:.4f} kW, valid range [0.0000, {pv_raw[bad_agent]:.4f}] kW.')
        pv_effective_clipped = np.clip(pv_effective_candidate, 0.0, pv_raw).astype(np.float32)
        pv_curtail = (pv_raw - pv_effective_clipped).astype(np.float32)
        pv_utilization = np.ones_like(pv_raw, dtype=np.float32)
        valid_mask = pv_raw > 1e-06
        pv_utilization[valid_mask] = (pv_effective_clipped[valid_mask] / pv_raw[valid_mask]).astype(np.float32)
        base_net_load_effective = (np.asarray(signal_state['load_t'], dtype=np.float32) - pv_effective_clipped).astype(np.float32)
        signal_state['pv_effective'] = pv_effective_clipped
        signal_state['pv_curtail'] = pv_curtail
        signal_state['pv_utilization'] = pv_utilization.astype(np.float32)
        signal_state['base_net_load_effective'] = base_net_load_effective
    def _build_signal_state(self, t: int, pv_action: np.ndarray) -> dict[str, Any]:
        wholesale_price_t = float(self.get_signal_step(WHOLESALE_PRICE_SIGNAL, t))
        import_price_t = float(derive_import_price(wholesale_price_t, markup_eur_per_kwh=self.import_price_markup_eur_per_kwh))
        load_t = np.asarray(self.get_signal_step('load', t), dtype=np.float32)
        pv_raw = np.asarray(self.ep_pv[t], dtype=np.float32)
        pv_utilization_req = self._pv_action_to_utilization(pv_action)
        pv_effective_req = (pv_raw * pv_utilization_req).astype(np.float32)
        pv_curtail_req = (pv_raw - pv_effective_req).astype(np.float32)
        base_net_load_raw = (load_t - pv_raw).astype(np.float32)
        signal_state = {'wholesale_price_t': wholesale_price_t, 'import_price_t': import_price_t, 'load_t': load_t, 'pv_raw': pv_raw, 'pv_utilization_req': pv_utilization_req.astype(np.float32), 'pv_effective_req': pv_effective_req, 'pv_curtail_req': pv_curtail_req, 'base_net_load_raw': base_net_load_raw, 'pv_fallback_applied': False, 'pv_fallback_scale': 1.0, 'mu_t': self._future_mean_price(t), 'mu_next': self._future_mean_price_next(t)}
        self._apply_effective_pv_to_signal_state(signal_state, pv_effective_req)
        signal_state['net_load'] = np.asarray(signal_state['base_net_load_effective'], dtype=np.float32).copy()
        signal_state['grid_import_kw'] = np.maximum(signal_state['net_load'], 0.0).astype(np.float32)
        signal_state['grid_export_kw'] = np.maximum(-signal_state['net_load'], 0.0).astype(np.float32)
        return signal_state
    def _has_overvoltage(self, pf_result: Any) -> bool:
        vm_pu = np.asarray(getattr(pf_result, 'vm_pu', np.zeros(0, dtype=np.float32)), dtype=np.float32)
        if vm_pu.size == 0:
            return False
        return bool(np.any(vm_pu > float(self._grid_cfg.v_max_pu) + 1e-06))
    def _run_power_flow_once(self, storage_state: dict[str, np.ndarray], signal_state: dict[str, Any]) -> tuple[Any, str]:
        pf_result = self._grid_core.step(p_batt_kw=storage_state['e_bat'], base_load_kw=signal_state['base_net_load_effective'])
        pf_error = getattr(self._grid_core, 'last_pf_error', '')
        self.vm_pu = pf_result.vm_pu
        signal_state['net_load'] = (np.asarray(signal_state['base_net_load_effective'], dtype=np.float32) + storage_state['e_bat']).astype(np.float32)
        signal_state['grid_import_kw'] = np.maximum(signal_state['net_load'], 0.0).astype(np.float32)
        signal_state['grid_export_kw'] = np.maximum(-signal_state['net_load'], 0.0).astype(np.float32)
        return (pf_result, pf_error)
    def _run_power_flow(self, storage_state: dict[str, np.ndarray], signal_state: dict[str, Any]) -> tuple[Any, str]:
        return self._run_power_flow_once(storage_state, signal_state)
    def _build_step_payload(self, t: int, storage_state: dict[str, np.ndarray], signal_state: dict[str, Any], pf_result: Any, pf_error: str) -> dict[str, Any]:
        return {'t': int(t), **storage_state, 'p_max': self.agent_p_max.astype(np.float32), 'e_min': self.agent_e_min.astype(np.float32), 'e_max': self.agent_e_max.astype(np.float32), 'battery_capacity_kwh': self.agent_c_bat.astype(np.float32), **signal_state, 'pv_action_req': self._utilization_to_pv_action(signal_state['pv_utilization_req']), 'pv_action_exec': self._utilization_to_pv_action(signal_state['pv_utilization']), 'pf_converged': bool(pf_result.converged), 'pf_error': pf_error, 'vm_pu': pf_result.vm_pu, 'agent_vm_pu': pf_result.agent_vm_pu, 'line_loading_pct': pf_result.line_loading_pct, 'trafo_loading_pct': pf_result.trafo_loading_pct, 'trafo_p_signed_kw': pf_result.trafo_p_signed_kw, 'v_violation': pf_result.v_violation, 'line_violation': float(pf_result.line_violation), 'trafo_violation': float(pf_result.trafo_violation), 'l_violation': float(pf_result.l_violation), 'n_buses': int(pf_result.n_buses), 'n_lines': int(pf_result.n_lines), 'n_trafos': int(pf_result.n_trafos), 'psi_v_raw': float(pf_result.psi_v_raw), 'psi_line_raw': float(pf_result.psi_line_raw), 'psi_trafo_raw': float(pf_result.psi_trafo_raw), 'bus_v_excess': pf_result.bus_v_excess, 'line_excess': pf_result.line_excess, 'trafo_excess': pf_result.trafo_excess}
    def _build_reward_state(self, step_state: dict[str, Any]) -> dict[str, Any]:
        return {'e_bat_req': step_state['e_bat_req'], 'e_bat': step_state['e_bat'], 'pv_effective_req': step_state['pv_effective_req'], 'pv_effective': step_state['pv_effective'], 'pv_raw': step_state['pv_raw'], 'p_max': step_state['p_max'], 'wholesale_price_t': step_state['wholesale_price_t'], 'import_price_t': step_state['import_price_t'], 'net_load_t': step_state['base_net_load_raw'], 'actual_grid_power_t': step_state['net_load'], 'dt': self.dt, 'v_violation': step_state['v_violation'], 'psi_v_raw': step_state['psi_v_raw'], 'psi_line_raw': step_state['psi_line_raw'], 'psi_trafo_raw': step_state['psi_trafo_raw']}
    def _build_violation_counts(self, step_state: dict[str, Any]) -> tuple[int, int, int]:
        n_v_viol = int(np.sum(step_state['v_violation'] > 0.0))
        line_limit = float(self._grid_cfg.line_max_loading_pct)
        n_line_viol = int(np.any(np.asarray(step_state['line_loading_pct'], dtype=np.float32) > line_limit))
        n_trafo_viol = int(np.any(np.asarray(step_state['trafo_loading_pct'], dtype=np.float32) > line_limit))
        return (n_v_viol, n_line_viol, n_trafo_viol)
    def _build_step_info(self, step_state: dict[str, Any], components: dict[str, np.ndarray], reward: np.ndarray, done: bool) -> dict[str, Any]:
        if self.mode == 'train':
            info: dict[str, Any] = {'episode_done': bool(done)}
            for key, value in components.items():
                info[str(key)] = np.asarray(value, dtype=np.float32)
            return info
        n_v_viol, n_line_viol, n_trafo_viol = self._build_violation_counts(step_state)
        info: dict[str, Any] = {'episode_done': done, 't': step_state['t'], 'wholesale_price': float(step_state['wholesale_price_t']), 'import_price': float(step_state['import_price_t']), 'load': step_state['load_t'].astype(np.float32), 'pv': step_state['pv_raw'].astype(np.float32), 'pv_raw': step_state['pv_raw'].astype(np.float32), 'pv_effective': step_state['pv_effective'].astype(np.float32), 'pv_curtail': step_state['pv_curtail'].astype(np.float32), 'pv_utilization': step_state['pv_utilization'].astype(np.float32), 'pv_effective_req': step_state['pv_effective_req'].astype(np.float32), 'pv_curtail_req': step_state['pv_curtail_req'].astype(np.float32), 'pv_utilization_req': step_state['pv_utilization_req'].astype(np.float32), 'pv_action_req': step_state['pv_action_req'].astype(np.float32), 'pv_action': step_state['pv_action_exec'].astype(np.float32), 'base_net_load': step_state['base_net_load_raw'].astype(np.float32), 'base_net_load_effective': step_state['base_net_load_effective'].astype(np.float32), 'net_load': step_state['net_load'].astype(np.float32), 'grid_import_kw': step_state['grid_import_kw'].astype(np.float32), 'grid_export_kw': step_state['grid_export_kw'].astype(np.float32), 'e_bat_req': step_state['e_bat_req'].astype(np.float32), 'e_bat': step_state['e_bat'].astype(np.float32), 'p_lower': step_state['p_lower'].astype(np.float32), 'p_upper': step_state['p_upper'].astype(np.float32), 'p_max': step_state['p_max'].astype(np.float32), 'e_min': step_state['e_min'].astype(np.float32), 'e_max': step_state['e_max'].astype(np.float32), 'battery_capacity_kwh': step_state['battery_capacity_kwh'].astype(np.float32), 'soc_min': float(self.soc_min), 'soc_max': float(self.soc_max), 'soc_t': step_state['soc_t'], 'soc_next': step_state['soc_next'], 'available_signals': sorted(self.signals), **components, 'reward': reward.astype(np.float32), 'mu_t': float(step_state['mu_t']), 'mu_next': float(step_state['mu_next']), 'pf_converged': step_state['pf_converged'], 'pf_error': step_state['pf_error'], 'vm_pu': step_state['vm_pu'], 'agent_vm_pu': step_state['agent_vm_pu'], 'line_loading_pct': step_state['line_loading_pct'], 'trafo_loading_pct': step_state['trafo_loading_pct'], 'trafo_p_signed_kw': step_state['trafo_p_signed_kw'], 'v_violation': step_state['v_violation'], 'line_violation': step_state['line_violation'], 'trafo_violation': step_state['trafo_violation'], 'l_violation': step_state['l_violation'], 'n_v_violations': n_v_viol, 'n_l_violations': int(max(n_line_viol, n_trafo_viol)), 'n_line_violations': n_line_viol, 'n_t_violations': n_trafo_viol, 'n_trafo_violations': n_trafo_viol, 'n_buses': step_state['n_buses'], 'n_lines': step_state['n_lines'], 'n_trafos': step_state['n_trafos'], 'psi_v_raw': step_state['psi_v_raw'], 'psi_line_raw': step_state['psi_line_raw'], 'psi_trafo_raw': step_state['psi_trafo_raw'], 'pv_fallback_applied': bool(step_state['pv_fallback_applied']), 'pv_fallback_scale': float(step_state['pv_fallback_scale']), 'bus_v_excess': step_state['bus_v_excess'], 'line_excess': step_state['line_excess'], 'trafo_excess': step_state['trafo_excess']}
        return info
    def step(self, actions: list[np.ndarray]) -> tuple[dict[str, np.ndarray], list[float], list[bool], list[bool], dict[str, Any]]:
        t = self.cur_step
        battery_action, pv_action = self._split_action_components(actions)
        validate_executed_actions_numpy(self._current_safety_local(), np.column_stack([battery_action, pv_action]), efficiency=self.eff, dt_hours=self.dt, soc_min=self.soc_min, soc_max=self.soc_max)
        storage_state = self._apply_storage_dynamics(battery_action)
        signal_state = self._build_signal_state(t, pv_action)
        pf_result, pf_error = self._run_power_flow(storage_state, signal_state)
        step_state = self._build_step_payload(t, storage_state, signal_state, pf_result, pf_error)
        reward_state = self._build_reward_state(step_state)
        reward_per_agent, components = self.reward_fn.compute(reward_state)
        reward = np.asarray(reward_per_agent, dtype=np.float32)
        self.soc = storage_state['soc_next']
        self.cur_step += 1
        done = self.cur_step >= self.episode_length
        terminated_n = [False] * self.n
        truncated_n = [done] * self.n
        obs = self.obs_builder.zeros(self.n) if done else self.obs_builder.build(self)
        info = self._build_step_info(step_state, components, reward, done)
        return (obs, reward.tolist(), terminated_n, truncated_n, info)
    def _cleanup_gurobi_cache(self) -> None:
        for solver in list(self._local_mpc_solver_cache.values()):
            dispose = getattr(solver, 'dispose', None)
            if callable(dispose):
                dispose()
        self._local_mpc_solver_cache.clear()
    def close(self) -> None:
        self._cleanup_gurobi_cache()
