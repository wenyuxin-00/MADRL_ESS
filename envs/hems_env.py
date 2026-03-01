# envs/hems_env.py
import gym
from gym import spaces
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EnergyStorageEnv(gym.Env):
    """
    多智能体储能系统 MDP（完全 NumPy 向量化，无 PV 版本）
    ----------------------------------------------------------------------
    观测（每个智能体 1D 向量）：
      [time_sin, time_cos,
       price(t..t+K) padding,
       load_i(t..t+K) padding,
       soc_i]

    动作：
      a_i ∈ [-1, 1]
      e_bat_i = a_i * P_max   (充电>0, 放电<0)

    奖励（每个智能体）：
      r = r_inc - r_pen + r_pbrs - r_soc + r_bonus
    """

    metadata = {"render.modes": []}

    def __init__(self, args: Any, data_path: Optional[str] = None, mode: str = "train"):
        super().__init__()
        self.args = args
        self.mode = mode

        # ========== 基本配置（从配置文件读取，无默认值） ==========
        self.n = int(args.num_agents)
        self.episode_length = int(args.episode_limit)
        self.k = int(args.future_horizon)
        self.c_bat = float(args.battery_capacity)    # 电池容量（能量单位）
        self.p_max = float(args.max_charge_rate)     # 最大充放功率（功率单位）
        self.eff = float(args.efficiency)              # 充放效率
        self.gamma = float(args.gamma)
        self.init_soc = float(args.init_soc)           # 初始 SoC（0~1）
        self.dt = float(args.dt)                       # 时间步长（小时）

        # ========== SOC 约束 ==========
        self.soc_min = float(args.soc_min)
        self.soc_max = float(args.soc_max)
        self.soc_target = float(args.soc_target)       # SoC 正则化的目标值
        self.soc_eps = float(args.soc_eps)             # 动作越限判定阈值

        # ========== 奖励超参数 ==========
        self.w_pen = float(args.w_pen)                 # 动作越限惩罚系数
        self.w_soc = float(args.w_soc)                 # SoC 正则系数
        self.lambda_bonus = float(args.lambda_bonus)  # 吞吐量奖励系数

        # ========== 数据加载 ==========
        self._load_data(data_path)

        # episode 切片缓存
        self.cur_step: int = 0
        self.ep_price: np.ndarray = np.zeros((self.episode_length,), dtype=np.float32)            # (T,)
        self.ep_load: np.ndarray = np.zeros((self.episode_length, self.n), dtype=np.float32)     # (T,N)

        # SoC 状态（(N,)）
        self.soc: np.ndarray = np.full((self.n,), self.init_soc, dtype=np.float32)

        # ========== Gym space ==========
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) for _ in range(self.n)]
        self.observation_space = self._build_observation_space()

    # ------------------------------------------------------------------
    # 数据加载：兼容 load1..loadN / load
    # ------------------------------------------------------------------
    def _load_data(self, data_path: Optional[str]) -> None:
        if data_path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            data_path = data_dir / ("train_prices.csv" if self.mode == "train" else "test_prices.csv")

        with open(data_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")

        raw = np.loadtxt(data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        self.raw = raw
        self.header = header

        # price 必须存在
        if "price" not in header:
            raise ValueError(f"CSV header 必须包含 'price' 列，但当前 header={header}")
        price_idx = header.index("price")
        self.all_price = raw[:, price_idx].astype(np.float32)  # (T_all,)

        # load：优先 load1..loadN，否则 load
        load_cols = []
        for i in range(self.n):
            name = f"load{i+1}"
            if name in header:
                load_cols.append(header.index(name))

        if len(load_cols) == self.n:
            self.all_load = raw[:, load_cols].astype(np.float32)  # (T_all,N)
        elif "load" in header:
            load_idx = header.index("load")
            one = raw[:, load_idx].astype(np.float32)[:, None]    # (T_all,1)
            self.all_load = np.repeat(one, self.n, axis=1)
        else:
            raise ValueError(
                "CSV 必须包含 load 列：要么 load1..loadN，要么单列 load。"
                f"当前 header={header}"
            )

        # 可用 episode 数
        total_steps = len(self.all_price)
        self.num_available_episodes = total_steps // self.episode_length
        if self.num_available_episodes <= 0:
            raise ValueError(f"数据长度不足：len={total_steps}, episode_length={self.episode_length}")

    # ------------------------------------------------------------------
    # 观测空间
    # ------------------------------------------------------------------
    def _build_observation_space(self) -> List[spaces.Box]:
        # time(2) + price(K+1) + load(K+1) + soc(1)
        obs_dim = 2 + 2 * (self.k + 1) + 1  # time + price + load + soc
        return [spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for _ in range(self.n)]

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
    # 观测构造（向量化返回 (N, obs_dim)）
    # ------------------------------------------------------------------
    def _get_obs_matrix(self) -> np.ndarray:
        t = self.cur_step
        T = self.episode_length
        k1 = self.k + 1

        time_sin = np.sin(2.0 * np.pi * t / T).astype(np.float32)
        time_cos = np.cos(2.0 * np.pi * t / T).astype(np.float32)

        price_win = self._pad_window_1d(self.ep_price, t, k1)     # (k1,)
        load_win = self._pad_window_2d(self.ep_load, t, k1).T     # (N,k1)

        obs_dim = self.observation_space[0].shape[0]
        obs = np.zeros((self.n, obs_dim), dtype=np.float32)

        idx = 0
        obs[:, idx] = time_sin
        obs[:, idx + 1] = time_cos
        idx += 2

        obs[:, idx:idx + k1] = price_win[None, :]
        idx += k1

        obs[:, idx:idx + k1] = load_win
        idx += k1

        obs[:, idx] = self.soc
        return obs

    # ------------------------------------------------------------------
    # PBRS：未来 K 步平均电价 μ_t（不含当前 t）
    # ------------------------------------------------------------------
    def _future_mean_price(self, t: int) -> float:
        start = t + 1
        end = min(t + 1 + self.k, self.episode_length)
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

        start = ep * self.episode_length
        end = start + self.episode_length

        self.ep_price = self.all_price[start:end].astype(np.float32)      # (T,)
        self.ep_load = self.all_load[start:end, :].astype(np.float32)     # (T,N)

        self.cur_step = 0
        self.soc = np.full((self.n,), self.init_soc, dtype=np.float32)

        obs = self._get_obs_matrix()
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
        # 充电：E_next = E + p*eff*dt <= C  => p <= (C-E)/(eff*dt)
        p_max_chg = np.minimum(self.p_max, (self.c_bat - e_t) / (eff * self.dt))  # >=0

        # 放电：E_next = E + p/eff*dt >= 0，p为负 => |p| <= E*eff/dt
        p_max_dis = np.minimum(self.p_max, (e_t * eff) / self.dt)                 # >=0

        p_lower = -p_max_dis
        p_upper = p_max_chg

        e_bat = np.clip(e_bat_req, p_lower, p_upper).astype(np.float32)  # 执行功率（可行）

        # -------- 4) SoC 更新（用执行功率） --------
        delta_e = np.where(e_bat >= 0.0, e_bat * eff, e_bat / eff) * self.dt
        e_next = np.clip(e_t + delta_e, 0.0, self.c_bat).astype(np.float32)
        soc_next = (e_next / self.c_bat).astype(np.float32)

        # -------- 5) 奖励分量（全向量化） --------
        price_t = float(self.ep_price[t])
        load_t = self.ep_load[t, :].astype(np.float32)  # (N,)

        # (1) Incremental Cost Reward：e_net = load + e_bat_exec
        e_net = load_t + e_bat
        r = -e_net * price_t
        r_idle = -load_t * price_t
        r_inc = (r - r_idle).astype(np.float32)

        # (2) Action Penalty：惩罚“请求与执行差异”（越界越多罚越多）
        r_pen = (self.w_pen * np.abs(e_bat_req - e_bat) / (self.p_max + 1e-6)).astype(np.float32)

        # (3) PBRS
        mu_t = self._future_mean_price(t)
        mu_next = self._future_mean_price(min(t + 1, self.episode_length - 1))
        phi_t = (mu_t * e_t).astype(np.float32)
        phi_next = (mu_next * e_next).astype(np.float32)
        r_pbrs = (self.gamma * phi_next - phi_t).astype(np.float32)

        # (4) SoC Regularization
        r_soc = (self.w_soc * (soc_t - self.soc_target) ** 2).astype(np.float32)

        # (5) Throughput Bonus（用执行功率）
        r_bonus = (self.lambda_bonus * np.abs(e_bat)).astype(np.float32)

        reward = (r_inc - r_pen + r_pbrs - r_soc + r_bonus).astype(np.float32)

        # -------- 6) 写回状态 / 推进时间 --------
        self.soc = soc_next
        self.cur_step += 1
        done = (self.cur_step >= self.episode_length)
        done_n = [done] * self.n

        # 下一观测
        if not done:
            obs = self._get_obs_matrix()
            obs_n = [obs[i] for i in range(self.n)]
        else:
            obs_dim = self.observation_space[0].shape[0]
            obs_n = [np.zeros((obs_dim,), dtype=np.float32) for _ in range(self.n)]

        # -------- 7) info：增加 e_bat_req，并返回可行域边界便于调试 --------
        info = {
            "episode_done": done,
            "t": int(t),
            "price": float(price_t),

            # 请求 vs 执行
            "e_bat_req": e_bat_req.astype(np.float32),
            "e_bat": e_bat.astype(np.float32),  # 执行功率（保持 key 兼容 Runner）
            "p_lower": p_lower.astype(np.float32),
            "p_upper": p_upper.astype(np.float32),

            "soc_t": soc_t,
            "soc_next": soc_next,
            "net_load": e_net.astype(np.float32),

            "r_inc": r_inc,
            "r_pen": r_pen,
            "r_pbrs": r_pbrs,
            "r_soc": r_soc,
            "r_bonus": r_bonus,
            "reward": reward,

            "mu_t": float(mu_t),
            "mu_next": float(mu_next),
        }

        return obs_n, reward.tolist(), done_n, info