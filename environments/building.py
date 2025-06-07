import numpy as np
import pandas as pd
import copy
from collections import namedtuple
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class MultiAgentEnergyStorageEnv:
    """
    Multi-agent reinforcement learning environment for N energy storage systems

    Each agent (energy storage system) observes:
    - Current time (hours)
    - Current electricity price
    - Current room load
    - Current SOC of energy storage system

    Action space:
    - Charging/discharging power (continuous value)

    Reward function:
    - Individual reward: Reduce electricity cost
    - Global reward: Flatten the overall load curve (reduce peak-to-valley difference)
    """
    
    def __init__(self, 
                 data_path: str = r"C:\\Users\\Clouds\Desktop\\Github\\MADRL_ESS\data\\opsd_building.csv",
                 num_agents: int = 3,  # The default is 3 agents (corresponding to load1, load2, load3)
                 episode_length: int = 96,  # 96 15-minute intervals per day
                 battery_capacity: float = 10.0,  # kWh
                 max_charge_rate: float = 5.0,    # kW,The specific capacity of energy storage still needs to be discussed, which depends on the unit of LOAD
                 efficiency: float = 0.95,
                 init_soc: float = 0.5,
                 seed: int = 42):
        """
        Initialize the environment

        Parameters:
        num_agents: number of energy storage systems/agents
        load_profiles: load profiles for each room {agent_id: pd.DataFrame}
        price_schedule: electricity price list (time-of-use electricity price)
        episode_length: time step of each episode (hours)
        battery_capacity: energy storage system capacity (kWh)
        max_charge_rate: maximum charge and discharge power (kW)
        efficiency: charge and discharge efficiency
        init_soc: initial SOC (0-1)
        seed: random seed
        """
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.efficiency = efficiency
        self.init_soc = init_soc
        self.seed = seed
        np.random.seed(seed)
        
        # laod data
        self.raw_data = self._load_data(data_path)
        self.load_profiles = self._process_load_data()
        self.price_schedule = self._process_price_data()

        # Checking data consistency
        self._validate_data()
        
        # Initialization state
        self.current_step = 0
        self.battery_soc = {agent_id: init_soc for agent_id in range(num_agents)}
        self.charge_history = {agent_id: [] for agent_id in range(num_agents)}
        self.load_history = {agent_id: [] for agent_id in range(num_agents)}
        self.price_history = []
        self.net_load_history = []
        
        # Defining the action and observation space
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        
        # Peak and valley smoothing related parameters
        self.peak_threshold = 1.2 * self._calculate_avg_peak()
        self.valley_threshold = 0.8 * self._calculate_avg_valley()
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """from csv"""
        try:
            data = pd.read_csv(data_path)
            print("Data loaded successfully, first 5 rows example:")
            print(data.head())
            return data
        except Exception as e:
            raise ValueError(f"Unable to load data file: {e}")
        

    def _process_load_data(self) -> Dict[int, pd.DataFrame]:
        """Process load data and convert them into load curves for each agent"""
        load_columns = [f"load{i+1}" for i in range(self.num_agents)]
        
        # Verify the data contains all required load columns
        missing_cols = [col for col in load_columns if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing load columns in data file: {missing_cols}")
        
        # Convert timestamp to hour and 15-minute intervals
        self.raw_data['datetime'] = pd.to_datetime(self.raw_data['unixtime'], unit='s')
        self.raw_data['hour'] = self.raw_data['datetime'].dt.hour
        self.raw_data['15min_interval'] = self.raw_data['datetime'].dt.minute // 15
        
        # Create load profile for each agent
        load_profiles = {}
        for agent_id in range(self.num_agents):
            load_col = load_columns[agent_id]
            load_profiles[agent_id] = self.raw_data[['hour', '15min_interval', load_col]].copy()
            load_profiles[agent_id].rename(columns={load_col: 'load'}, inplace=True)
        
        return load_profiles
    
    def _process_price_data(self) -> pd.DataFrame:
        """Process electricity price data"""
        if 'price' not in self.raw_data.columns:
            raise ValueError("Missing 'price' column in data file")
        
        price_data = self.raw_data[['hour', '15min_interval', 'price']].copy()
        
        # Convert price to per kWh (assuming original data is per MWh)
        price_data['price'] = price_data['price'] / 1000  # Convert to $/kWh
        
        return price_data

    def _validate_data(self):
        """Validate data integrity and length"""
        # Check load data
        for agent_id, profile in self.load_profiles.items():
            if len(profile) < self.episode_length:
                raise ValueError(f"Insufficient load data length, requires at least {self.episode_length} timesteps")
        # Check price data
        if len(self.price_schedule) < self.episode_length:
            raise ValueError(f"Insufficient price data length, requires at least {self.episode_length} timesteps")
        
        print(f"Data validation passed, successfully loaded {len(self.raw_data)} rows of data")


    def _get_action_space(self):
        """Define action space - charge/discharge power for each agent"""
        return {'low': -self.max_charge_rate, 'high': self.max_charge_rate}
    
    def _get_observation_space(self):
        """Define observation space"""
        # Observation includes: current hour, current 15-min interval, current price, current load, current SOC
        return {
            'hour': {'low': 0, 'high': 23},
            '15min_interval': {'low': 0, 'high': 3},
            'price': {'low': 0, 'high': np.inf},
            'load': {'low': 0, 'high': np.inf},
            'soc': {'low': 0, 'high': 1}
        }
    
    def _calculate_avg_peak(self) -> float:
        """Calculate average peak load"""
        peak_loads = []
        for agent_id, profile in self.load_profiles.items():
            # Calculate daily peak load for each agent
            daily_peaks = profile.groupby(['hour', '15min_interval'])['load'].mean().groupby('hour').max()
            peak_loads.extend(daily_peaks.values)
        return np.mean(peak_loads) if peak_loads else 0
    
    def _calculate_avg_valley(self) -> float:
        """Calculate average valley load"""
        valley_loads = []
        for agent_id, profile in self.load_profiles.items():
            # Calculate daily valley load for each agent
            daily_valleys = profile.groupby(['hour', '15min_interval'])['load'].mean().groupby('hour').min()
            valley_loads.extend(daily_valleys.values)
        return np.mean(valley_loads) if valley_loads else 0
    

    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset environment state
        
        Returns:
            Initial observation for each agent
        """
        self.current_step = 0
        self.battery_soc = {agent_id: self.init_soc for agent_id in range(self.num_agents)}
        self.charge_history = {agent_id: [] for agent_id in range(self.num_agents)}
        self.load_history = {agent_id: [] for agent_id in range(self.num_agents)}
        self.price_history = []
        self.net_load_history = []
        
        # Get initial observations
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = self._get_agent_observation(agent_id)
        
        return observations
    
    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a single agent"""
        current_data = self.load_profiles[agent_id].iloc[self.current_step]
        current_price = self.price_schedule.iloc[self.current_step]['price']
        current_load = current_data['load']
        current_soc = self.battery_soc[agent_id]
        
        return np.array([
            current_data['hour'],
            current_data['15min_interval'],
            current_price,
            current_load,
            current_soc
        ])
    
    def step(self, actions: Dict[int, float]) -> Tuple:
        """
        Execute one timestep
        
        Modified to use actual data loaded from file
        """
        # Store current loads and electricity price
        current_loads = {}
        for agent_id in range(self.num_agents):
            current_loads[agent_id] = self.load_profiles[agent_id].iloc[self.current_step]['load']
        current_price = self.price_schedule.iloc[self.current_step]['price']
        
        # Execute energy storage actions and update SOC
        net_loads = {}
        individual_costs = {}
        for agent_id, action in actions.items():
            # 1. Constrain action within allowed range
            charge_power = np.clip(action, -self.max_charge_rate, self.max_charge_rate)
            
            # 2. Calculate actual charge/discharge amount (considering efficiency)
            if charge_power > 0:  # Charging
                actual_charge = charge_power * self.efficiency
            else:  # Discharging
                actual_charge = charge_power / self.efficiency
            
            # 3. Update SOC (considering capacity limits)
            delta_soc = actual_charge / self.battery_capacity
            new_soc = self.battery_soc[agent_id] + delta_soc
            self.battery_soc[agent_id] = np.clip(new_soc, 0, 1)
            
            # 4. Calculate net load (actual load - discharge + charge)
            # Note: Discharging is positive, charging is negative
            net_load = current_loads[agent_id] - charge_power  
            net_loads[agent_id] = net_load
            
            # 5. Record historical data
            self.charge_history[agent_id].append(charge_power)
            self.load_history[agent_id].append(current_loads[agent_id])
            
            # 6. Calculate individual cost
            individual_costs[agent_id] = net_load * current_price
        
        # Calculate global net load (sum of all agents)
        global_net_load = sum(net_loads.values())
        self.net_load_history.append(global_net_load)
        self.price_history.append(current_price)
        
        # Get new observations
        observations = {}
        for agent_id in range(self.num_agents):
            observations[agent_id] = self._get_agent_observation(agent_id)
        
        # Calculate rewards
        rewards = self._calculate_rewards(individual_costs, global_net_load)
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Additional information
        info = {
            'global_net_load': global_net_load,
            'average_cost': np.mean(list(individual_costs.values())),
            'peak_load': max(self.net_load_history) if self.net_load_history else 0,
            'valley_load': min(self.net_load_history) if self.net_load_history else 0,
            'current_time': self.load_profiles[0].iloc[self.current_step-1]['datetime']
        }
        
        return observations, rewards, done, info
    
    def _calculate_rewards(self, 
                         individual_costs: Dict[int, float], 
                         global_net_load: float) -> Dict[int, float]:
        """
        Calculate reward function
        
        Args:
            individual_costs: Electricity cost for each agent
            global_net_load: Global net load across all agents
            
        Returns:
            Reward for each agent (combining individual and global components)
        """
        # 1. Individual reward: Minimize electricity cost (negative cost)
        individual_rewards = {}
        for agent_id, cost in individual_costs.items():
            individual_rewards[agent_id] = -cost
        
        # 2. Global reward: Flatten load curve
        # a) Penalize peak-valley difference
        peak_penalty = max(0, global_net_load - self.peak_threshold) ** 2
        valley_penalty = max(0, self.valley_threshold - global_net_load) ** 2
        global_penalty = (peak_penalty + valley_penalty) / 2
        
        # b) Reward load following average (optional)
        avg_load = np.mean(self.net_load_history) if self.net_load_history else global_net_load
        load_following_reward = -abs(global_net_load - avg_load)
        
        # Combine global reward components
        global_reward = load_following_reward - global_penalty
        
        # 3. Combine individual and global rewards
        rewards = {}
        for agent_id in individual_rewards.keys():
            # Use weighted sum (weights can be adjusted as needed)
            rewards[agent_id] = (
                0.7 * individual_rewards[agent_id] + 
                0.3 * global_reward / self.num_agents  # Distribute global reward equally
            )
        
        return rewards
    
    def get_metrics(self) -> Dict:
        """Get evaluation metrics"""
        if not self.net_load_history:
            return {}
        
        peak_load = max(self.net_load_history)
        valley_load = min(self.net_load_history)
        avg_load = np.mean(self.net_load_history)
        peak_to_avg = peak_load / avg_load if avg_load != 0 else 0
        std_dev = np.std(self.net_load_history)
        
        total_cost = sum(
            load * price for load, price in zip(self.net_load_history, self.price_history)
        )
        
        return {
            'peak_load': peak_load,
            'valley_load': valley_load,
            'peak_to_avg_ratio': peak_to_avg,
            'load_std_dev': std_dev,
            'total_cost': total_cost,
            'avg_cost_per_agent': total_cost / self.num_agents
        }
    
    def close(self):
        pass