# envs/hems_env.py
import gym
from gym import spaces
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EnergyStorageEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, args: Any, mode: str = "train",
                 dataset=None, reward_fn=None,
                 forecaster=None, obs_builder=None,
                 data_path: Optional[str] = None):
        super().__init__()
        self.args = args
        self.mode = mode

        # ========== 基本配置（从配置文件读取，无默认值） ==========
        self.n = int(args.num_agents)
        self.episode_length = int(args.episode_limit)
        self.future_horizon = int(args.future_horizon)
        self.c_bat = float(args.battery_capacity)
        self.p_max = float(args.max_charge_rate)
        self.eff = float(args.efficiency)
        self.gamma = float(args.gamma)
        self.init_soc = float(args.init_soc)
        self.dt = float(args.dt)

        # ========== SOC 约束 ==========
        self.soc_min = float(args.soc_min)
        self.soc_max = float(args.soc_max)
        self.soc_target = float(args.soc_target)
        self.soc_eps = float(args.soc_eps)

        # ========== 奖励超参数 ==========
        self.w_pen = float(args.w_pen)
        self.w_soc = float(args.w_soc)
        self.lambda_bonus = float(args.lambda_bonus)

        # ========== 奖励函数（可插拔） ==========
        from common.rewards import get_reward_fn
        reward_type = getattr(args, 'reward_type', 'composite')
        self.reward_fn = reward_fn if reward_fn is not None else get_reward_fn(reward_type, args)

        # ========== 数据集（可插拔，向后兼容） ==========
        if dataset is None:
            from datasets.csv_price_load import CsvPriceLoadDataset
            if data_path is None:
                data_dir = Path(__file__).resolve().parent.parent / "data"
                data_path = str(data_dir / ("train_prices.csv" if mode == "train" else "test_prices.csv"))
            dataset = CsvPriceLoadDataset(data_path, self.episode_length, self.n)
        self._dataset = dataset

        # ========== 预测器（可插拔，默认 PerfectForecaster） ==========
        if forecaster is None:
            from forecast.oracle import PerfectForecaster
            forecaster = PerfectForecaster()
        self.forecaster = forecaster

        # ========== 观测构造器（可插拔，默认 DefaultObservationBuilder） ==========
        if obs_builder is None:
            from envs.observation.default_builder import DefaultObservationBuilder
            obs_config = getattr(args, 'obs_config', ['time', 'price', 'load', 'soc'])
            obs_builder = DefaultObservationBuilder(obs_config, self.future_horizon)
        self.obs_builder = obs_builder

        # ========== 可用 episode 数 ==========
        self.num_available_episodes = self._dataset.num_episodes()

        # episode 切片缓存
        self.cur_step: int = 0
        self.ep_price: np.ndarray = np.zeros((self.episode_length,), dtype=np.float32)
        self.ep_load: np.ndarray = np.zeros((self.episode_length, self.n), dtype=np.float32)

        # SoC 状态（(N,)）
        self.soc: np.ndarray = np.full((self.n,), self.init_soc, dtype=np.float32)

        # ========== Gym space ==========
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) for _ in range(self.n)]
        obs_dim = self.obs_builder.get_obs_dim()
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for _ in range(self.n)]

    # ------------------------------------------------------------------
    # 工具：取未来窗口（含当前，共 K+1），不足用 0 padding
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_window_1d(x: np.ndarray, start: int, length: int) -> np.ndarray:
        end = start + length
        if start >= len(x):
            return np.zeros((length,), dtype=np.float32)
        chunk = x[start:end]
        if len(chunk) < length:
            chunk = np.concatenate([chunk, np.zeros((length - len(chunk),), dtype=np.float32)], axis=0)
        return chunk.astype(np.float32)

    @staticmethod
    def _pad_window_2d(x: np.ndarray, start: int, length: int) -> np.ndarray:
        end = start + length
        if start >= x.shape[0]:
            return np.zeros((length, x.shape[1]), dtype=np.float32)
        chunk = x[start:end, :]
        if chunk.shape[0] < length:
            pad = np.zeros((length - chunk.shape[0], x.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk.astype(np.float32)

    # ------------------------------------------------------------------
    # PBRS：未来 K 步平均电价 μ_t（使用真实价格，不经过 forecaster）
    # ------------------------------------------------------------------
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
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, episode_idx: Optional[int] = None) -> List[np.ndarray]:
        if episode_idx is None:
            ep = np.random.randint(0, self.num_available_episodes)
        else:
            if episode_idx < 0 or episode_idx >= self.num_available_episodes:
                raise IndexError(f"episode_idx={episode_idx} 超出范围 [0, {self.num_available_episodes-1}]")
            ep = int(episode_idx)

        episode_data = self._dataset.get_episode(ep)
        self.ep_price = episode_data["price"]   # (T,)
        self.ep_load = episode_data["load"]     # (T,N)

        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        # 初始化预测器
        self.forecaster.reset()
        if hasattr(self.forecaster, 'set_episode'):
            self.forecaster.set_episode(self.ep_price)

        obs = self.obs_builder.build(self)
        return [obs[i] for i in range(self.n)]

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        t = self.cur_step

        # -------- 1) 动作解析（N,） --------
        a = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)[:, 0]
        a = np.clip(a, -1.0, 1.0)

        # 请求功率
        e_bat_req = a * self.p_max  # (N,)

        # -------- 2) 当前能量状态 --------
        soc_t = self.soc.copy().astype(np.float32)     # (N,)
        e_t = soc_t * self.c_bat                        # (N,)
        eff = max(self.eff, 1e-6)

        # -------- 3) 可行域投影：得到执行功率 e_bat_exec --------
        p_max_chg = np.minimum(self.p_max, (self.c_bat - e_t) / (eff * self.dt))
        p_max_dis = np.minimum(self.p_max, (e_t * eff) / self.dt)

        p_lower = -p_max_dis
        p_upper = p_max_chg

        e_bat = np.clip(e_bat_req, p_lower, p_upper).astype(np.float32)

        # -------- 4) SoC 更新（用执行功率） --------
        delta_e = np.where(e_bat >= 0.0, e_bat * eff, e_bat / eff) * self.dt
        e_next = np.clip(e_t + delta_e, 0.0, self.c_bat).astype(np.float32)
        soc_next = (e_next / self.c_bat).astype(np.float32)

        # -------- 5) 奖励计算（委托给可插拔的 reward_fn） --------
        price_t = float(self.ep_price[t])
        load_t = self.ep_load[t, :].astype(np.float32)  # (N,)

        mu_t = self._future_mean_price(t)
        mu_next = self._future_mean_price(min(t + 1, self.episode_length - 1))

        env_state = dict(
            e_bat_req=e_bat_req, e_bat=e_bat,
            soc_t=soc_t, soc_next=soc_next,
            e_t=e_t, e_next=e_next,
            price_t=price_t, load_t=load_t,
            mu_t=mu_t, mu_next=mu_next,
            gamma=self.gamma,
        )
        reward, components = self.reward_fn.compute(env_state)
        e_net = load_t + e_bat

        # -------- 6) 写回状态 / 推进时间 --------
        self.soc = soc_next
        self.cur_step += 1
        done = (self.cur_step >= self.episode_length)
        done_n = [done] * self.n

        # 下一观测
        if not done:
            obs = self.obs_builder.build(self)
            obs_n = [obs[i] for i in range(self.n)]
        else:
            obs_dim = self.observation_space[0].shape[0]
            obs_n = [np.zeros((obs_dim,), dtype=np.float32) for _ in range(self.n)]

        # -------- 7) info --------
        info = {
            "episode_done": done,
            "t": int(t),
            "price": float(price_t),
            "e_bat_req": e_bat_req.astype(np.float32),
            "e_bat": e_bat.astype(np.float32),
            "p_lower": p_lower.astype(np.float32),
            "p_upper": p_upper.astype(np.float32),
            "soc_t": soc_t,
            "soc_next": soc_next,
            "net_load": e_net.astype(np.float32),
            **components,
            "reward": reward,
            "mu_t": float(mu_t),
            "mu_next": float(mu_next),
        }

        return obs_n, reward.tolist(), done_n, info
