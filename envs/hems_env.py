# envs/hems_env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Callable

class EnergyStorageEnv(gym.Env):
    def __init__(self, args, reward_fn: Callable, data_path: str = None, mode: str = 'train'):
        self.num_agents = args.num_agents
        self.n = args.num_agents
        self.episode_length = args.episode_limit
        self.future_price = args.future_price
        self.future_load = args.future_load
        self.mode = mode
        
        # --- 物理参数 ---
        self.battery_capacity = args.battery_capacity
        self.max_charge_rate = args.max_charge_rate
        self.efficiency = args.efficiency
        self.init_soc = args.init_soc
        self.obs_config = getattr(args, 'obs_config', ['time', 'price', 'load', 'soc', 'future_price', 'future_load'])
        
        # --- 核心：外部注入的奖励函数 ---
        self.reward_fn = reward_fn
        
        # --- 数据加载 (与你原始逻辑一致) ---
        if data_path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            data_path = data_dir / ("train_prices.csv" if mode == 'train' else "test_prices.csv")
        self.raw_data = pd.read_csv(data_path)
        
        load_cols = [f"load{i+1}" for i in range(self.n)]
        all_loads = self.raw_data[load_cols].values.flatten()
        self.load_norm_params = {'mean': np.mean(all_loads), 'std': np.std(all_loads)}
        self.num_available_episodes = (len(self.raw_data) // self.episode_length) - 1

        self.current_step = 0
        self.battery_soc = {}
        self.load_profiles = {}
        self.net_load_history = deque(maxlen=96)
        
        # --- Gym 接口 ---
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) for _ in range(self.n)]
        self.observation_space = self._create_observation_space()

    def _create_observation_space(self) -> List[spaces.Box]:
        dim = 0
        if 'time' in self.obs_config: dim += 2
        if 'price' in self.obs_config: dim += 1
        if 'load' in self.obs_config: dim += 1
        if 'soc' in self.obs_config: dim += 1
        if 'future_price' in self.obs_config: dim += self.future_price
        if 'future_load' in self.obs_config: dim += self.future_load
        return [spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32) for _ in range(self.n)]

    def _get_single_observation(self, agent_id: int) -> np.ndarray:
        obs = []
        profile = self.load_profiles[agent_id]
        
        if 'time' in self.obs_config:
            obs.extend([
                np.sin(2 * np.pi * self.current_step / self.episode_length),
                np.cos(2 * np.pi * self.current_step / self.episode_length)
            ])
        if 'price' in self.obs_config:
            obs.append(profile['price'].iloc[self.current_step])
        if 'load' in self.obs_config:
            obs.append(profile['load'].iloc[self.current_step])
        if 'soc' in self.obs_config:
            obs.append(self.battery_soc[agent_id])
            
        if 'future_price' in self.obs_config and self.future_price > 0:
            futures = profile['price'].iloc[self.current_step + 1 : self.current_step + 1 + self.future_price].values
            obs.extend(np.pad(futures, (0, self.future_price - len(futures)), 'constant'))
            
        if 'future_load' in self.obs_config and self.future_load > 0:
            futures = profile['load'].iloc[self.current_step + 1 : self.current_step + 1 + self.future_load].values
            obs.extend(np.pad(futures, (0, self.future_load - len(futures)), 'constant'))
            
        return np.array(obs, dtype=np.float32)

    def reset(self, episode_idx=None) -> List[np.ndarray]:
        """重置环境。如果提供了episode_idx，则用于测试模式。"""
        self.current_step = 0
        self.battery_soc = {i: self.init_soc for i in range(self.n)}
        self.net_load_history.clear()

        if episode_idx is None:
            # 训练模式：随机选择一天（确保有前一天的数据）
            day_idx = np.random.randint(1, self.num_available_episodes + 1)
        else:
            # 测试模式：使用指定索引（确保有前一天的数据）
            day_idx = episode_idx + 1
            if day_idx > self.num_available_episodes:
                raise IndexError(f"Test episode_idx {episode_idx} is out of valid range.")

        # --- 计算电价标准化参数 (基于前一天) ---
        prev_day_start = (day_idx - 1) * self.episode_length
        prev_day_end = prev_day_start + 2*self.episode_length
        prev_day_prices = self.raw_data['price'].iloc[prev_day_start:prev_day_end].values
        self.episode_price_mean = np.mean(prev_day_prices)
        self.episode_price_std = np.std(prev_day_prices)
        if self.episode_price_std == 0: self.episode_price_std = 1

        # --- 准备当前回合的数据 ---
        start = day_idx * self.episode_length
        end = start + self.episode_length
        
        episode_prices = self.raw_data['price'].iloc[start:end].values
        normalized_prices = (episode_prices - self.episode_price_mean) / self.episode_price_std
        
        load_cols = [f"load{i+1}" for i in range(self.num_agents)]
        for agent_id in range(self.n):
            agent_load = self.raw_data[load_cols[agent_id]].iloc[start:end].values
            normalized_load = (agent_load - self.load_norm_params['mean']) / self.load_norm_params['std']
            self.load_profiles[agent_id] = pd.DataFrame({'load': normalized_load, 'price': normalized_prices})
        
        # 初始化前一天的总负荷历史用于削峰填谷奖励计算
        prev_day_total_load = self.raw_data[load_cols].iloc[prev_day_start:prev_day_end].sum(axis=1).values
        self.net_load_history.extend(prev_day_total_load)

        return [self._get_single_observation(i) for i in range(self.n)]

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        actions_dict = {i: action[0] for i, action in enumerate(actions)}
        norm_price = self.load_profiles[0]['price'].iloc[self.current_step]
        real_price = norm_price * self.episode_price_std + self.episode_price_mean

        ess_power = {}
        total_ess_power = 0
        soc_penalties = {}
        
        # 1. 物理演化层
        for agent_id, action_val in actions_dict.items():
            charge_power = action_val * self.max_charge_rate
            delta_soc = (charge_power * self.efficiency if charge_power >= 0 else charge_power / self.efficiency) / self.battery_capacity
            new_soc = self.battery_soc[agent_id] + delta_soc

            soc_penalties[agent_id] = 0.0
            if new_soc > 0.95:
                soc_penalties[agent_id] = -5.0 if action_val > 0 else 0
                charge_power = min(charge_power, ((0.95 - self.battery_soc[agent_id]) * self.battery_capacity) / self.efficiency)
                new_soc = 0.95
            elif new_soc < 0.05:
                soc_penalties[agent_id] = -5.0 if action_val < 0 else 0
                charge_power = max(charge_power, -((self.battery_soc[agent_id] - 0.05) * self.battery_capacity * self.efficiency))
                new_soc = 0.05

            self.battery_soc[agent_id] = new_soc
            ess_power[agent_id] = charge_power
            total_ess_power += charge_power

        # 2. 全局协同计算
        original_total_load = 0
        for agent_id in range(self.n):
            normalized_load = self.load_profiles[agent_id]['load'].iloc[self.current_step]
            original_total_load += normalized_load * self.load_norm_params['std'] + self.load_norm_params['mean']
            
        net_total_load = original_total_load + total_ess_power
        self.net_load_history.append(net_total_load)

        # 构建给奖励函数的物理信息包
        physics_info = {
            'ess_power': ess_power,
            'soc': self.battery_soc.copy(),
            'soc_penalties': soc_penalties,
            'original_total_load': original_total_load,
            'net_total_load': net_total_load,
            'net_load_std': np.std(self.net_load_history) if len(self.net_load_history) > 1 else 0
        }

        # 3. ★ 调用外部奖励函数 ★
        rewards = []
        for i in range(self.n):
            rewards.append(self.reward_fn(i, actions_dict[i], physics_info, real_price))

        self.current_step += 1
        done = self.current_step >= self.episode_length
        obs_n = [self._get_single_observation(i) for i in range(self.n)] if not done else [np.zeros_like(self.observation_space[i].low) for i in range(self.n)]
        done_n = [done] * self.n

        # 兼容你的绘图 Info 字典
        total_economic_reward = sum([-ess_power[i] * real_price for i in range(self.n)])
        info = {
            'ess_power': ess_power, 'soc': self.battery_soc, 'price': real_price,
            'original_total_load': original_total_load, 'net_total_load': net_total_load,
            'original_load_cost': original_total_load * real_price,
            'cost_with_ess': net_total_load * real_price,
            'ess_profit': total_economic_reward, 
            'soc_penalty': sum(soc_penalties.values()),
            'peak_valley_penalty': 0 # 将由外部 Reward fn 计算，这里设0保持字典结构
        }
        return obs_n, rewards, done_n, info