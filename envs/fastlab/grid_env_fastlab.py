"""Fast-lab training environment backed by exact precomputed observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from envs.grid_env import GridEnv
from scripts.utils.madrl_observation_cache_lab import ObservationCacheStore


class GridEnvFastLab(GridEnv):
    """GridEnv variant that indexes a precomputed observation cache."""

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
        observation_cache_dir: str | Path | None = None,
        train_info_mode: str = "minimal",
    ) -> None:
        self._fastlab_cache_dir = Path(
            observation_cache_dir
            or getattr(getattr(cfg, "runtime", None), "fastlab_observation_cache_dir", "")
        ).resolve()
        if not self._fastlab_cache_dir.exists():
            raise FileNotFoundError(
                f"Fast-lab observation cache directory does not exist: {self._fastlab_cache_dir}"
            )
        self._observation_cache = ObservationCacheStore(self._fastlab_cache_dir)
        self._train_info_mode = str(train_info_mode or "minimal").strip().lower()
        self._fastlab_episode_cache: dict[str, np.ndarray] = {}
        super().__init__(
            cfg,
            mode=mode,
            dataset=dataset,
            reward_fn=reward_fn,
            forecaster=forecaster,
            obs_builder=obs_builder,
            grid_core=grid_core,
            data_path=data_path,
        )

    def get_cached_local_feature(self, feature_name: str) -> np.ndarray:
        if feature_name != "calendar_time":
            raise KeyError(f"Unknown cached local feature '{feature_name}'.")
        return np.asarray(self._fastlab_episode_cache["calendar_time"][self.cur_step], dtype=np.float32)

    def get_cached_sequence_feature(self, feature_name: str) -> np.ndarray:
        key = f"{feature_name}_seq"
        if key not in self._fastlab_episode_cache:
            raise KeyError(f"Fast-lab cache does not contain sequence feature '{feature_name}'.")
        return np.asarray(self._fastlab_episode_cache[key][self.cur_step], dtype=np.float32)

    def _future_mean_price(self, t: int) -> float:
        if "mu_t" not in self._fastlab_episode_cache:
            return super()._future_mean_price(t)
        return float(np.asarray(self._fastlab_episode_cache["mu_t"], dtype=np.float32)[int(t)])

    def reset(
        self,
        episode_idx: int | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is None and not self._seeded_once:
            seed = self._default_seed
        super(GridEnv, self).reset(seed=seed)
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
        self._fastlab_episode_cache = self._observation_cache.episode(int(episode_idx))
        self._last_episode_idx = int(episode_idx)
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        self.forecaster.reset()
        self.forecaster.set_episode(self.signals, self.episode_meta)

        base_load = np.asarray(self.ep_load[0], dtype=np.float32)
        base_pv = np.asarray(self.ep_pv[0], dtype=np.float32)
        self._grid_core.reset(base_load, base_pv)

        return self.obs_builder.build(self), self._build_reset_info(int(episode_idx))

    def _build_step_info(
        self,
        step_state: dict[str, Any],
        components: dict[str, np.ndarray],
        reward: np.ndarray,
        done: bool,
    ) -> dict[str, Any]:
        if self.mode != "train" or self._train_info_mode != "minimal":
            return super()._build_step_info(step_state, components, reward, done)

        info: dict[str, Any] = {
            "episode_done": bool(done),
            "price": float(step_state["price_t"]),
            "base_net_load": step_state["base_net_load"].astype(np.float32),
            "e_bat_req": step_state["e_bat_req"].astype(np.float32),
            "e_bat": step_state["e_bat"].astype(np.float32),
            "soc_next": step_state["soc_next"].astype(np.float32),
            "reward": reward.astype(np.float32),
        }
        for key, value in components.items():
            info[str(key)] = np.asarray(value, dtype=np.float32)
        return info

