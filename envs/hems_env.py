# envs/hems_env.py
import gym
from gym import spaces
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


class EnergyStorageEnv(gym.Env):
    """
    多智能体电力系统储能协同优化环境 (Multi-Agent Energy Storage Environment)

    说明：
    - 多个储能(ESS)智能体根据电价/负荷等信息进行充放电决策
    - 环境内置 SOC 约束与动作裁剪，并提供“动作可行范围”作为观测，避免裁剪信息被隐藏
    - 支持加入全局摘要特征(net_total_load 与 rolling 统计)以提升协同学习效率
    """

    def __init__(self, args: Any, data_path: Optional[str] = None, mode: str = "train"):
        super().__init__()

        # --- 1. 全局与控制参数 ---
        self.args = args
        self.num_agents = int(args.num_agents)
        self.n = self.num_agents
        self.episode_length = int(args.episode_limit)
        self.future_price = int(args.future_price)
        self.future_load = int(args.future_load)
        self.mode = mode

        # --- 2. 物理参数 ---
        self.battery_capacity = float(args.battery_capacity)
        self.max_charge_rate = float(args.max_charge_rate)  # 功率上限(绝对值)
        self.efficiency = float(args.efficiency)
        self.init_soc = float(args.init_soc)

        # SOC 约束（与 step() 里的裁剪一致，单独抽成常量，避免魔法数字散落）
        self.soc_min = float(getattr(args, "soc_min", 0.05))
        self.soc_max = float(getattr(args, "soc_max", 0.95))

        # rolling window 长度（全局摘要用）
        self.rolling_window_k = int(getattr(args, "rolling_window_k", 8))
        self.rolling_window_k = max(2, self.rolling_window_k)  # 至少2才有 std 的意义

        # --- 3. 观测配置（默认把你要求的新特征加进去，便于直接生效） ---
        self.obs_config = getattr(
            args,
            "obs_config",
            [
                "time",
                "price",
                "load",
                "soc",
                "soc_margin",       # 新增：SOC裕度
                "action_bounds",    # 新增：动作可行范围（避免裁剪信息隐藏）
                "global_summary",   # 新增：全局摘要（总负荷+rolling统计）
                "future_price",
                "future_load",
            ],
        )

        # --- 4. 数据加载与预处理 ---
        self._load_data(data_path)

        # --- 5. 状态记录器初始化 ---
        self.current_step = 0
        self.battery_soc = np.full(self.n, self.init_soc, dtype=np.float32)

        # 用于存储当前 Episode 的切片数据
        self.episode_norm_prices = None
        self.episode_price_mean = 0.0
        self.episode_price_std = 1.0
        self.episode_norm_loads = None
        self.episode_real_loads = None

        # 记录净负荷历史：用于rolling统计（用于奖励/观测）
        self.net_load_history = deque(maxlen=self.episode_length)

        # --- 6. Gym 接口空间定义 ---
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) for _ in range(self.n)
        ]
        self.observation_space = self._create_observation_space()

    # =========================
    # 数据加载
    # =========================
    def _load_data(self, data_path: Optional[str]):
        """读取 CSV 数据并计算归一化参数"""
        if data_path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            data_path = data_dir / ("train_prices.csv" if self.mode == "train" else "test_prices.csv")

        with open(data_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")

        raw_array = np.loadtxt(data_path, delimiter=",", skiprows=1, dtype=np.float32)
        if raw_array.ndim == 1:
            raw_array = raw_array.reshape(1, -1)
        self.raw_array = raw_array

        self.price_col_idx = header.index("price")
        self.load_col_idx = [header.index(f"load{i+1}") for i in range(self.n)]

        self.all_prices = self.raw_array[:, self.price_col_idx]
        self.all_loads = self.raw_array[:, self.load_col_idx]  # shape=(T, n)

        # 负荷的全局归一化参数（单个 agent load 的归一化）
        all_loads_flat = self.all_loads.reshape(-1)
        load_std = float(np.std(all_loads_flat))
        self.load_norm_params = {
            "mean": float(np.mean(all_loads_flat)),
            "std": load_std if load_std > 0 else 1.0,
        }

        # 总负荷的归一化参数（用于 global_summary）
        total_load_series = self.all_loads.sum(axis=1)  # shape=(T,)
        total_std = float(np.std(total_load_series))
        self.total_load_norm_params = {
            "mean": float(np.mean(total_load_series)),
            "std": total_std if total_std > 0 else 1.0,
        }

        # 可用 episode 数（保留你原逻辑：需要“前一天数据”用于滚动归一化）
        self.num_available_episodes = (len(self.all_prices) // self.episode_length) - 1

    # =========================
    # 观测空间与特征
    # =========================
    def _create_observation_space(self) -> List[spaces.Box]:
        """构建多智能体观测空间（维度随 obs_config 改变）"""
        dim = 0
        if "time" in self.obs_config:
            dim += 2
        if "price" in self.obs_config:
            dim += 1
        if "load" in self.obs_config:
            dim += 1
        if "soc" in self.obs_config:
            dim += 1
        if "soc_margin" in self.obs_config:
            dim += 2  # soc_margin_up, soc_margin_down
        if "action_bounds" in self.obs_config:
            dim += 2  # p_chg_max_available, p_dis_max_available
        if "global_summary" in self.obs_config:
            dim += 3  # net_total_load, rolling_mean_k, rolling_std_k
        if "future_price" in self.obs_config:
            dim += self.future_price
        if "future_load" in self.obs_config:
            dim += self.future_load

        return [
            spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32) for _ in range(self.n)
        ]

    def _get_action_power_bounds(self, soc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据当前 SOC 计算“允许的充/放最大功率”。

        返回：
        - p_chg_max_available: shape=(n,), >=0，表示此刻最多能充多少功率（正向）
        - p_dis_max_available: shape=(n,), >=0，表示此刻最多能放多少功率幅值（负向幅值）
          真实允许功率范围为：[-p_dis_max_available, +p_chg_max_available]

        这样 actor 在输出动作前就能知道裁剪边界，不会出现“裁剪信息被隐藏”。
        """
        # 充电：soc + (p*eff)/cap <= soc_max  => p <= (soc_max - soc)*cap/eff
        p_chg_limit_soc = (self.soc_max - soc) * self.battery_capacity / max(self.efficiency, 1e-6)
        p_chg_max_available = np.clip(np.minimum(self.max_charge_rate, p_chg_limit_soc), 0.0, self.max_charge_rate)

        # 放电：soc + (p/eff)/cap >= soc_min，p为负
        # 令 p = -|p|，则 soc - |p|/(eff*cap) >= soc_min  => |p| <= (soc - soc_min)*cap*eff
        p_dis_limit_soc = (soc - self.soc_min) * self.battery_capacity * max(self.efficiency, 1e-6)
        p_dis_max_available = np.clip(np.minimum(self.max_charge_rate, p_dis_limit_soc), 0.0, self.max_charge_rate)

        return p_chg_max_available.astype(np.float32), p_dis_max_available.astype(np.float32)

    def _get_global_summary_features(self) -> Tuple[float, float, float]:
        """
        计算全局摘要特征（用于协同学习）：
        - net_total_load：当前时刻“无动作基线”的总负荷（即原始总负荷，方便决策前用）
        - rolling_mean_k / rolling_std_k：基于最近(k-1)个历史净负荷 + 当前基线总负荷构成窗口的均值/标准差

        注意：这里统一做 total_load 的归一化，避免尺度过大影响学习。
        """
        # 当前时刻原始总负荷（决策前可见的基线）
        current_total_load = float(np.sum(self.episode_real_loads[self.current_step]))

        # rolling窗口：取历史最后k-1个 + 当前基线
        if len(self.net_load_history) > 0:
            hist = np.asarray(self.net_load_history, dtype=np.float32)
            tail = hist[-(self.rolling_window_k - 1) :] if hist.size >= (self.rolling_window_k - 1) else hist
            window = np.concatenate([tail, np.asarray([current_total_load], dtype=np.float32)], axis=0)
        else:
            window = np.asarray([current_total_load], dtype=np.float32)

        rolling_mean = float(np.mean(window))
        rolling_std = float(np.std(window)) if window.size > 1 else 0.0

        # 归一化
        mu = self.total_load_norm_params["mean"]
        sig = self.total_load_norm_params["std"]
        net_total_load_norm = (current_total_load - mu) / sig
        rolling_mean_norm = (rolling_mean - mu) / sig
        rolling_std_norm = rolling_std / sig

        return float(net_total_load_norm), float(rolling_mean_norm), float(rolling_std_norm)

    def _get_single_observation(self, agent_id: int) -> np.ndarray:
        """获取单个智能体当前时刻观测"""
        obs: List[float] = []

        # 时间编码
        if "time" in self.obs_config:
            obs.extend(
                [
                    float(np.sin(2 * np.pi * self.current_step / self.episode_length)),
                    float(np.cos(2 * np.pi * self.current_step / self.episode_length)),
                ]
            )

        # 当前归一化电价/负荷/SOC
        if "price" in self.obs_config:
            obs.append(float(self.episode_norm_prices[self.current_step]))
        if "load" in self.obs_config:
            obs.append(float(self.episode_norm_loads[self.current_step, agent_id]))
        if "soc" in self.obs_config:
            obs.append(float(self.battery_soc[agent_id]))

        # SOC 裕度
        if "soc_margin" in self.obs_config:
            soc = float(self.battery_soc[agent_id])
            obs.append(float(self.soc_max - soc))  # soc_margin_up
            obs.append(float(soc - self.soc_min))  # soc_margin_down

        # 动作可行范围（功率上界），避免裁剪信息隐藏
        if "action_bounds" in self.obs_config:
            p_chg_max_available, p_dis_max_available = self._get_action_power_bounds(self.battery_soc)
            obs.append(float(p_chg_max_available[agent_id]))  # p_chg_max_available
            obs.append(float(p_dis_max_available[agent_id]))  # p_dis_max_available (放电幅值)

        # 全局摘要（协同信号）
        if "global_summary" in self.obs_config:
            net_total_load, rolling_mean_k, rolling_std_k = self._get_global_summary_features()
            obs.extend([net_total_load, rolling_mean_k, rolling_std_k])

        # 未来窗口
        if "future_price" in self.obs_config and self.future_price > 0:
            futures = self.episode_norm_prices[
                self.current_step + 1 : self.current_step + 1 + self.future_price
            ]
            obs.extend(np.pad(futures, (0, self.future_price - len(futures)), "constant").astype(np.float32).tolist())

        if "future_load" in self.obs_config and self.future_load > 0:
            futures = self.episode_norm_loads[
                self.current_step + 1 : self.current_step + 1 + self.future_load, agent_id
            ]
            obs.extend(np.pad(futures, (0, self.future_load - len(futures)), "constant").astype(np.float32).tolist())

        return np.asarray(obs, dtype=np.float32)

    # =========================
    # Gym 标准接口
    # =========================
    def reset(self, episode_idx: Optional[int] = None) -> List[np.ndarray]:
        """
        重置环境，进入新的一天 (Episode)。
        电价标准化采用“前一天滚动窗口”以提升平稳性。
        """
        self.current_step = 0
        self.battery_soc = np.full(self.n, self.init_soc, dtype=np.float32)
        self.net_load_history.clear()

        if episode_idx is None:
            day_idx = np.random.randint(1, self.num_available_episodes + 1)
        else:
            day_idx = episode_idx + 1
            if day_idx > self.num_available_episodes:
                raise IndexError(f"Test episode_idx {episode_idx} is out of valid range.")

        # --- 电价滚动归一化（基于前一天）---
        prev_day_start = (day_idx - 1) * self.episode_length
        prev_day_end = prev_day_start + 2 * self.episode_length
        prev_day_prices = self.all_prices[prev_day_start:prev_day_end]

        self.episode_price_mean = float(np.mean(prev_day_prices))
        price_std = float(np.std(prev_day_prices))
        self.episode_price_std = price_std if price_std > 0 else 1.0

        # --- 当前 Episode 数据切片 ---
        start = day_idx * self.episode_length
        end = start + self.episode_length

        episode_prices = self.all_prices[start:end]
        self.episode_norm_prices = (episode_prices - self.episode_price_mean) / self.episode_price_std

        self.episode_real_loads = self.all_loads[start:end]  # shape=(T, n)
        self.episode_norm_loads = (self.episode_real_loads - self.load_norm_params["mean"]) / self.load_norm_params["std"]

        # 初始化历史负荷（使用前一天“总负荷”作为历史，便于 rolling 统计在 t=0 有意义）
        prev_day_total_load = self.all_loads[prev_day_start:prev_day_end].sum(axis=1)
        self.net_load_history.extend(prev_day_total_load.astype(np.float32).tolist())

        return [self._get_single_observation(i) for i in range(self.n)]

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """
        核心物理推演层：
        - 输入动作 -> 计算功率 -> SOC 更新 -> 计算全局指标/奖励
        """
        actions_arr = np.asarray(actions, dtype=np.float32).reshape(self.n, -1)[:, 0]

        # 当前电价（归一化 & 实值）
        norm_price = float(self.episode_norm_prices[self.current_step])
        real_price = float(norm_price * self.episode_price_std + self.episode_price_mean)

        # ==========================================
        # 1) 物理演化层
        # ==========================================
        # 动作映射到功率
        charge_power = actions_arr * self.max_charge_rate

        # SOC 演化（先按原动作计算 proposed）
        delta_soc = np.where(
            charge_power >= 0,
            charge_power * self.efficiency,
            charge_power / max(self.efficiency, 1e-6),
        ) / self.battery_capacity
        proposed_soc = self.battery_soc + delta_soc

        # ==========================================
        # 2) 约束处理与惩罚
        # ==========================================
        soc_penalties_arr = np.zeros(self.n, dtype=np.float32)
        upper_mask = proposed_soc > self.soc_max
        lower_mask = proposed_soc < self.soc_min

        # 越限惩罚触发
        soc_penalties_arr[upper_mask & (actions_arr > 0)] = -5.0
        soc_penalties_arr[lower_mask & (actions_arr < 0)] = -5.0

        # 功率裁剪到可行域（保证不会越界）
        if np.any(upper_mask):
            charge_power[upper_mask] = np.minimum(
                charge_power[upper_mask],
                ((self.soc_max - self.battery_soc[upper_mask]) * self.battery_capacity) / max(self.efficiency, 1e-6),
            )
        if np.any(lower_mask):
            charge_power[lower_mask] = np.maximum(
                charge_power[lower_mask],
                -((self.battery_soc[lower_mask] - self.soc_min) * self.battery_capacity * max(self.efficiency, 1e-6)),
            )

        # 状态更新（与裁剪一致：最终SOC在边界内）
        self.battery_soc = np.clip(proposed_soc, self.soc_min, self.soc_max)
        total_ess_power = float(np.sum(charge_power))

        # ==========================================
        # 3) 全局指标
        # ==========================================
        original_total_load = float(np.sum(self.episode_real_loads[self.current_step]))
        net_total_load = original_total_load + total_ess_power
        self.net_load_history.append(net_total_load)

        current_net_load_std = float(np.std(self.net_load_history)) if len(self.net_load_history) > 1 else 0.0

        # ==========================================
        # 4) 奖励（保持你现有逻辑）
        # ==========================================
        economic_rewards = -charge_power * norm_price * 10.0  # shape=(n,)

        peak_penalty = 0.0
        if getattr(self.args, "w_pv_rolling", 1.0) > 0 and current_net_load_std > 0:
            peak_penalty = -float(self.args.w_pv_rolling) * current_net_load_std / self.n

        rewards_arr = economic_rewards + soc_penalties_arr + peak_penalty
        rewards = rewards_arr.astype(np.float32).tolist()

        # ==========================================
        # 5) 时钟推进
        # ==========================================
        self.current_step += 1
        done = self.current_step >= self.episode_length
        done_n = [done] * self.n

        if not done:
            obs_n = [self._get_single_observation(i) for i in range(self.n)]
        else:
            obs_n = [np.zeros_like(self.observation_space[i].low, dtype=np.float32) for i in range(self.n)]

        # 额外：把“动作可行范围”也写入 info，便于 debug（不影响训练）
        p_chg_max_available, p_dis_max_available = self._get_action_power_bounds(self.battery_soc)

        info = {
            "ess_power": {i: float(charge_power[i]) for i in range(self.n)},
            "soc": {i: float(self.battery_soc[i]) for i in range(self.n)},
            "price": real_price,
            "original_total_load": original_total_load,
            "net_total_load": net_total_load,
            "original_load_cost": original_total_load * real_price,
            "cost_with_ess": net_total_load * real_price,
            "ess_profit": float(np.sum(-charge_power * real_price)),
            "soc_penalty": float(np.sum(soc_penalties_arr)),
            "peak_valley_penalty": float(peak_penalty),
            # 明确告诉你每步可行域（防止“裁剪信息藏起来”）
            "p_chg_max_available": {i: float(p_chg_max_available[i]) for i in range(self.n)},
            "p_dis_max_available": {i: float(p_dis_max_available[i]) for i in range(self.n)},
        }

        return obs_n, rewards, done_n, info