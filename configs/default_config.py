# configs/default_config.py
import torch

class Config:
    def __init__(self):
        # --- 环境与物理参数 ---
        self.num_agents = 3
        self.episode_limit = 96
        self.future_price = 24
        self.future_load = 24
        self.battery_capacity = 1.0
        self.max_charge_rate = 0.1
        self.efficiency = 1.0
        self.init_soc = 0.05
        
        # --- 观测空间配置 (方便做消融实验) ---
        # 支持: 'time', 'price', 'load', 'soc', 'future_price', 'future_load'
        self.obs_config = ['time', 'price', 'load', 'soc', 'future_price', 'future_load']

        # --- 训练核心参数 ---
        self.algorithm = "MADDPG"  # "MADDPG" 或 "MATD3"
        self.train_episodes = 300
        self.max_train_steps = self.train_episodes * self.episode_limit
        self.batch_size = 96 * 10
        self.max_action = 1.0
        self.buffer_size = int(1e6)
        
        # --- 网络与优化器参数 ---
        self.hidden_dim = 128
        self.lr_a = 1e-4
        self.lr_c = 1e-4
        self.gamma = 0.999
        self.tau = 0.01
        self.use_orthogonal_init = True
        self.use_grad_clip = True
        self.policy_update_freq = 2
        
        # --- 探索噪声参数 ---
        self.noise_std_init = 0.4
        self.noise_std_min = 0.2
        self.noise_decay_steps = 3e5
        self.use_noise_decay = True
        self.noise_std_decay = (self.noise_std_init - self.noise_std_min) / self.noise_decay_steps
        
        # --- MATD3 特有参数 ---
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        
        # 运行时设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")