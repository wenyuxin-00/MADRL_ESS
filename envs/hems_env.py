"""储能多智能体环境。

本环境负责三件事：
1. 从 dataset 读取一个 episode 的 signals。
2. 执行动作并更新 SoC 与奖励。
3. 把内部状态交给 observation builder 组织成结构化观测。

数据层统一采用开放式 `signals` 字典。当前最小实现只依赖 `price` 和 `load`，
未来可以自然扩展到 `pv` 等新信号。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces


class EnergyStorageEnv(gym.Env):
    """面向科研实验的多智能体储能环境。"""

    metadata = {"render.modes": []}

    def __init__(
        self,
        cfg: Any,
        mode: str = "train",
        dataset=None,
        reward_fn=None,
        forecaster=None,
        obs_builder=None,
        data_path: Optional[str] = None,
    ):
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
                f"Invalid SoC range: soc_min={self.soc_min}, soc_max={self.soc_max}. Expected 0 <= min <= max <= 1."
            )
        self.e_min = self.soc_min * self.c_bat
        self.e_max = self.soc_max * self.c_bat
        self.init_soc = float(np.clip(self.init_soc, self.soc_min, self.soc_max))

        self.w_pen = float(reward_cfg.w_pen)
        self.w_soc = float(reward_cfg.w_soc)
        self.lambda_bonus = float(reward_cfg.lambda_bonus)

        from common.rewards import get_reward_fn

        self.reward_fn = reward_fn if reward_fn is not None else get_reward_fn(cfg.reward.type, cfg)

        if dataset is None:
            from datasets.csv_price_load import CsvPriceLoadDataset

            if data_path is None:
                data_dir = Path(__file__).resolve().parent.parent / "data"
                data_path = str(data_dir / ("train_prices.csv" if mode == "train" else "test_prices.csv"))
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

        self.num_available_episodes = self._dataset.num_episodes()
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        self.signals: dict[str, np.ndarray] = {}
        self.episode_meta: dict[str, Any] = {}
        self.ep_price = np.zeros((self.episode_length,), dtype=np.float32)
        self.ep_load = np.zeros((self.episode_length, self.n), dtype=np.float32)

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
        """把 dataset 给出的 signal 规范成 float32 numpy。"""
        signal = np.asarray(value, dtype=np.float32)
        if signal.shape[0] != self.episode_length:
            raise ValueError(
                f"signal '{name}' 的首维长度应为 episode_length={self.episode_length}，实际为 {signal.shape[0]}"
            )
        return signal

    def _load_episode(self, episode_idx: int) -> None:
        """读取一个 episode，并填充开放式 signals。"""
        episode_data = self._dataset.get_episode(episode_idx)
        raw_signals = episode_data.get("signals", {})
        if "price" not in raw_signals or "load" not in raw_signals:
            raise KeyError("环境至少需要 signals['price'] 和 signals['load']。")

        self.signals = {
            name: self._canonicalize_signal(name, signal)
            for name, signal in raw_signals.items()
        }
        self.episode_meta = dict(episode_data.get("meta", {}))

        self.ep_price = self.get_signal("price")
        self.ep_load = self.get_signal("load")
        if self.ep_price.ndim != 1:
            raise ValueError(f"signals['price'] 应为 shape (T,)，当前为 {self.ep_price.shape}")
        if self.ep_load.shape != (self.episode_length, self.n):
            raise ValueError(
                f"signals['load'] 应为 shape {(self.episode_length, self.n)}，当前为 {self.ep_load.shape}"
            )

    def get_signal(self, signal_name: str) -> np.ndarray:
        """读取当前 episode 的某个 signal。"""
        if signal_name not in self.signals:
            available = sorted(self.signals)
            raise KeyError(f"当前 episode 不包含 signal '{signal_name}'，已有 {available}")
        return self.signals[signal_name]

    def get_signal_step(self, signal_name: str, step: int | None = None):
        """读取某个时刻的 signal 值。"""
        step = self.cur_step if step is None else int(step)
        signal = self.get_signal(signal_name)
        value = signal[step]
        if signal.ndim == 1:
            return float(value)
        return np.asarray(value, dtype=np.float32)

    def get_signal_history(self, signal_name: str) -> np.ndarray:
        """读取从 episode 起点到当前步的 signal 历史。"""
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

    def reset(self, episode_idx: Optional[int] = None) -> dict[str, np.ndarray]:
        """重置到一个新 episode。"""
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
            self.forecaster.set_episode(self.ep_price)

        return self.obs_builder.build(self)

    def step(self, actions: List[np.ndarray]) -> Tuple[dict[str, np.ndarray], List[float], List[bool], Dict]:
        """Execute one environment step.
        执行一步环境推进。

        Energy balance pipeline / 能量平衡流程:
            1. Map agent actions [-1, 1] to requested power e_bat_req
            2. Compute feasible charge/discharge limits from current SoC
            3. Clip requested power to feasible range -> e_bat (executed power)
            4. Update stored energy and SoC with efficiency losses
            5. Compute reward from the reward function
        """
        t = self.cur_step

        # --- 1. Action mapping: [-1, 1] -> requested battery power ---
        # 动作映射：将归一化动作转换为请求的电池充放电功率
        action_array = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)[:, 0]
        action_array = np.clip(action_array, -1.0, 1.0)
        e_bat_req = action_array * self.p_max

        soc_t = self.soc.copy().astype(np.float32)
        e_t = soc_t * self.c_bat  # current stored energy (kWh)
        eff = max(self.eff, 1e-6)

        # --- 2. Feasibility projection: compute max charge/discharge power ---
        # 可行性投影：根据当前储能和容量限制，计算最大充/放电功率
        # Charge limit: can't exceed capacity (e_max), accounting for efficiency loss
        # 充电上限 = min(额定功率, (剩余可充容量) / (效率 * 时间步长))
        p_max_chg = np.minimum(
            self.p_max,
            np.maximum(0.0, (self.e_max - e_t) / (eff * self.dt)),
        )
        # Discharge limit: can't go below minimum (e_min), accounting for efficiency
        # 放电上限 = min(额定功率, (可放电量) * 效率 / 时间步长)
        p_max_dis = np.minimum(
            self.p_max,
            np.maximum(0.0, (e_t - self.e_min) * eff / self.dt),
        )

        # --- 3. Clip to feasible range ---
        # 将请求功率裁剪到可行范围 [−p_max_dis, +p_max_chg]
        p_lower = -p_max_dis
        p_upper = p_max_chg
        e_bat = np.clip(e_bat_req, p_lower, p_upper).astype(np.float32)

        # --- 4. SoC update with efficiency losses ---
        # 储能更新：充电时损耗 (×eff)，放电时损耗 (÷eff)
        # delta_e > 0 for charging, < 0 for discharging
        delta_e = np.where(e_bat >= 0.0, e_bat * eff, e_bat / eff) * self.dt
        e_next = np.clip(e_t + delta_e, self.e_min, self.e_max).astype(np.float32)
        soc_next = (e_next / self.c_bat).astype(np.float32)

        price_t = float(self.get_signal_step("price", t))
        load_t = np.asarray(self.get_signal_step("load", t), dtype=np.float32)

        mu_t = self._future_mean_price(t)
        mu_next = self._future_mean_price(min(t + 1, self.episode_length - 1))

        env_state = {
            "e_bat_req": e_bat_req,
            "e_bat": e_bat,
            "soc_t": soc_t,
            "soc_next": soc_next,
            "e_t": e_t,
            "e_next": e_next,
            "e_min": float(self.e_min),
            "e_max": float(self.e_max),
            "price_t": price_t,
            "load_t": load_t,
            "mu_t": mu_t,
            "mu_next": mu_next,
            "gamma": self.gamma,
        }
        reward, components = self.reward_fn.compute(env_state)
        e_net = load_t + e_bat

        self.soc = soc_next
        self.cur_step += 1
        done = self.cur_step >= self.episode_length
        done_n = [done] * self.n

        obs = self.obs_builder.zeros(self.n) if done else self.obs_builder.build(self)

        info = {
            "episode_done": done,
            "t": int(t),
            "price": float(price_t),
            "e_bat_req": e_bat_req.astype(np.float32),
            "e_bat": e_bat.astype(np.float32),
            "p_lower": p_lower.astype(np.float32),
            "p_upper": p_upper.astype(np.float32),
            "e_min": float(self.e_min),
            "e_max": float(self.e_max),
            "soc_min": float(self.soc_min),
            "soc_max": float(self.soc_max),
            "soc_t": soc_t,
            "soc_next": soc_next,
            "net_load": e_net.astype(np.float32),
            "available_signals": sorted(self.signals),
            **components,
            "reward": reward,
            "mu_t": float(mu_t),
            "mu_next": float(mu_next),
        }
        return obs, reward.tolist(), done_n, info
