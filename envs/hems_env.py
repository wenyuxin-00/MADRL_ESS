"""家庭能源管理系统（HEMS）强化学习环境。

封装 Gym 接口的多智能体电池储能调度环境，
不含电网潮流约束。

主要类:
    EnergyStorageEnv -- HEMS 多智能体储能环境

典型用法::
    env = EnergyStorageEnv(cfg, mode="train", dataset=ds)
    obs = env.reset()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gym
import numpy as np
from gym import spaces


class EnergyStorageEnv(gym.Env):
    """基于电价/负荷/光伏信号的多智能体电池储能环境。

    每个智能体控制一个电池储能单元，通过充放电策略套利。
    不包含电网潮流约束（潮流版本见 GridEnv）。

    属性:
        n: 智能体数量
        episode_length: 单集最大步数
        soc: 各智能体当前荷电状态 (State of Charge)，shape=(n,)
        action_space: 各智能体动作空间列表，动作 ∈ [-1, 1]
        observation_space: 结构化观测空间字典
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        cfg: Any,
        mode: str = "train",
        dataset=None,
        reward_fn=None,
        forecaster=None,
        obs_builder=None,
        data_path: str | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        env_cfg = cfg.env
        reward_cfg = cfg.reward

        # 先把配置对象展开成环境内部常用的标量，避免后面频繁写 cfg.xxx.yyy。
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
        self.init_soc = float(np.clip(self.init_soc, self.soc_min, self.soc_max))

        self.w_pen = float(reward_cfg.w_pen)
        self.w_soc = float(reward_cfg.w_soc)
        self.lambda_bonus = float(reward_cfg.lambda_bonus)

        from envs.rewards import get_reward_fn

        # 奖励函数允许外部注入；如果没传，就按配置动态构造默认实现。
        self.reward_fn = reward_fn if reward_fn is not None else get_reward_fn(cfg.reward.type, cfg)

        if dataset is None:
            from data.loaders.csv_price_load import CsvPriceLoadDataset

            if data_path is None:
                data_dir = Path(__file__).resolve().parent.parent / "data"
                data_path = str(data_dir / ("train_prices.csv" if mode == "train" else "test_prices.csv"))
            dataset = CsvPriceLoadDataset(data_path, self.episode_length, self.n)
        self._dataset = dataset

        if forecaster is None:
            from predictors.oracle import PerfectForecaster

            # 默认用“完美预测器”，这样环境本身可以单独运行，不依赖额外模型文件。
            forecaster = PerfectForecaster()
        self.forecaster = forecaster

        if obs_builder is None:
            from envs.observation.default_builder import DefaultObservationBuilder

            # 观测构建器负责把环境内部状态整理成神经网络能消费的结构化输入。
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
        self.ep_pv = np.zeros((self.episode_length, self.n), dtype=np.float32)

        self.agent_c_bat = np.full((self.n,), self.c_bat, dtype=np.float32)
        self.agent_p_max = np.full((self.n,), self.p_max, dtype=np.float32)
        self.agent_e_min = self.soc_min * self.agent_c_bat
        self.agent_e_max = self.soc_max * self.agent_c_bat
        self.e_min = self.agent_e_min.copy()
        self.e_max = self.agent_e_max.copy()

        # 每个智能体动作只有 1 维，表示归一化后的充放电强度。
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for _ in range(self.n)
        ]
        # 观测空间由 obs_builder 给出 schema，这样环境本体不和具体特征设计强耦合。
        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
                for key, shape in self.observation_schema.items()
            }
        )

    def _canonicalize_signal(self, name: str, value: np.ndarray) -> np.ndarray:
        """校验并标准化信号数组的数据类型和第一维长度。

        参数:
            name: 信号名称（用于错误提示）
            value: 原始信号数组

        返回:
            标准化后的 float32 数组

        异常:
            ValueError: 第一维长度与 episode_length 不匹配时抛出
        """
        signal = np.asarray(value, dtype=np.float32)
        if signal.shape[0] != self.episode_length:
            raise ValueError(
                f"signal '{name}' first dimension should match episode_length={self.episode_length}, got {signal.shape[0]}"
            )
        return signal

    def _resolve_meta_vector(self, key: str, default_value: float) -> np.ndarray:
        """从 episode_meta 中解析每智能体向量，缺失时用默认值填充。

        参数:
            key: meta 字典中的键名
            default_value: 缺失或无效时的默认值

        返回:
            shape=(n,) 的 float32 数组
        """
        raw_value = self.episode_meta.get(key)
        if raw_value is None:
            return np.full((self.n,), default_value, dtype=np.float32)

        values = np.asarray(raw_value, dtype=np.float32).reshape(-1)
        if values.size != self.n:
            raise ValueError(f"episode meta '{key}' should have {self.n} values, got shape {values.shape}")
        fallback = np.full((self.n,), default_value, dtype=np.float32)
        return np.where(values > 0.0, values, fallback).astype(np.float32)

    def _apply_episode_storage_config(self) -> None:
        # 如果数据集给了每个 agent 自己的容量/功率参数，就在这里覆盖默认值。
        self.agent_c_bat = self._resolve_meta_vector("ess_capacity_kwh", self.c_bat)
        self.agent_p_max = self._resolve_meta_vector("ess_power_kw", self.p_max)
        self.agent_e_min = (self.soc_min * self.agent_c_bat).astype(np.float32)
        self.agent_e_max = (self.soc_max * self.agent_c_bat).astype(np.float32)
        self.e_min = self.agent_e_min.copy()
        self.e_max = self.agent_e_max.copy()

    def _load_episode(self, episode_idx: int) -> None:
        # 一个 episode 可以理解为一整段时间序列切片，例如一天或两天的价格/负荷轨迹。
        episode_data = self._dataset.get_episode(episode_idx)
        raw_signals = episode_data.get("signals", {})
        if "price" not in raw_signals or "load" not in raw_signals:
            raise KeyError("Environment requires signals['price'] and signals['load'].")

        self.signals = {
            name: self._canonicalize_signal(name, signal)
            for name, signal in raw_signals.items()
        }
        self.episode_meta = dict(episode_data.get("meta", {}))

        # `price` 是一维时间序列；`load` / `pv` 是 `(T, n_agents)` 的矩阵。
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
                f"signals['load'] must have shape {(self.episode_length, self.n)}, got {self.ep_load.shape}"
            )
        if self.ep_pv.shape != (self.episode_length, self.n):
            raise ValueError(
                f"signals['pv'] must have shape {(self.episode_length, self.n)}, got {self.ep_pv.shape}"
            )

        self._apply_episode_storage_config()

    def get_signal(self, signal_name: str) -> np.ndarray:
        if signal_name not in self.signals:
            available = sorted(self.signals)
            raise KeyError(f"Current episode does not contain signal '{signal_name}'. Available: {available}")
        return self.signals[signal_name]

    def get_signal_step(self, signal_name: str, step: int | None = None):
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
        """计算从 t+1 到 t+future_horizon 的未来平均电价。

        参数:
            t: 当前时间步

        返回:
            未来窗口内的平均电价；若已到 episode 末尾则返回当前电价
        """
        start = t + 1
        end = min(t + 1 + self.future_horizon, self.episode_length)
        if start >= self.episode_length:
            return float(self.ep_price[min(t, self.episode_length - 1)])
        seg = self.ep_price[start:end]
        if seg.size == 0:
            return float(self.ep_price[min(t, self.episode_length - 1)])
        return float(np.mean(seg))

    def reset(self, episode_idx: int | None = None) -> dict[str, np.ndarray]:
        if episode_idx is None:
            # 训练时常见做法：随机抽一个 episode，增加样本多样性。
            episode_idx = int(np.random.randint(0, self.num_available_episodes))
        elif episode_idx < 0 or episode_idx >= self.num_available_episodes:
            raise IndexError(
                f"episode_idx={episode_idx} is out of range [0, {self.num_available_episodes - 1}]"
            )

        self._load_episode(int(episode_idx))
        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        # 预测器也需要切换到同一段时间序列上下文。
        self.forecaster.reset()
        if hasattr(self.forecaster, "set_episode"):
            self.forecaster.set_episode(self.signals)

        return self.obs_builder.build(self)

    def step(self, actions: list[np.ndarray]) -> tuple[dict[str, np.ndarray], list[float], list[bool], dict]:
        """执行一步环境交互。

        参数:
            actions: 各智能体的动作列表，每个元素为 shape=(1,) 的数组，取值 [-1, 1]。
                     正值表示充电，负值表示放电。

        返回:
            obs: 下一步的结构化观测字典
            reward: 各智能体的即时奖励列表
            done: 各智能体的终止标志列表
            info: 包含详细步骤信息的字典（电价、SoC、功率等）
        """
        t = self.cur_step

        # 将动作裁剪到 [-1, 1] 并映射为实际功率请求 (kW)
        action_array = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)[:, 0]
        action_array = np.clip(action_array, -1.0, 1.0)
        # 环境先接收“归一化动作”，再乘以各自功率上限，得到实际功率请求。
        e_bat_req = action_array * self.agent_p_max

        # 当前 SoC 和能量状态
        soc_t = self.soc.copy().astype(np.float32)
        e_t = soc_t * self.agent_c_bat
        eff = max(self.eff, 1e-6)  # 防止除零

        # 计算物理允许的最大充/放电功率，确保 SoC 不越界
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
        # 真正执行的功率还要再裁剪一次，保证不会突破物理约束。
        e_bat = np.clip(e_bat_req, p_lower, p_upper).astype(np.float32)

        # 根据充放电效率计算储能变化量：
        #   充电 (e_bat ≥ 0): delta_e = P × η × dt  （储入能量 < 输入功率，损耗在外）
        #   放电 (e_bat < 0): delta_e = P / η × dt  （P<0 故 |delta_e| > |P|×dt，电池多消耗以弥补损耗）
        delta_e = np.where(e_bat >= 0.0, e_bat * eff, e_bat / eff) * self.dt
        e_next = np.clip(e_t + delta_e, self.agent_e_min, self.agent_e_max).astype(np.float32)
        soc_next = (e_next / self.agent_c_bat).astype(np.float32)

        price_t = float(self.get_signal_step("price", t))
        load_t = np.asarray(self.get_signal_step("load", t), dtype=np.float32)
        pv_t = np.asarray(self.ep_pv[t], dtype=np.float32)
        # 对储能来说，真正要面对的是“负荷 - 光伏”后的净负荷。
        net_load_t = (load_t - pv_t).astype(np.float32)

        mu_t = self._future_mean_price(t)
        mu_next = self._future_mean_price(min(t + 1, self.episode_length - 1))

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
        }
        reward, components = self.reward_fn.compute(env_state)
        # 从电网角度看，总功率 = 原始净负荷 + 电池充放电功率。
        grid_power = (net_load_t + e_bat).astype(np.float32)

        self.soc = soc_next
        self.cur_step += 1
        done = self.cur_step >= self.episode_length
        done_n = [done] * self.n

        # 终止时返回零观测，避免下游还去读取一个越界时间步。
        obs = self.obs_builder.zeros(self.n) if done else self.obs_builder.build(self)

        # `info` 不参与训练更新，但非常适合做调试、画图和教学解释。
        info = {
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
            **components,
            "reward": reward,
            "mu_t": float(mu_t),
            "mu_next": float(mu_next),
        }
        return obs, reward.tolist(), done_n, info
