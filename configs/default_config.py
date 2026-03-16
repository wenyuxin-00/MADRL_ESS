# configs/default_config.py
import torch


class Config:
    def __init__(self):
        # --- 环境与物理参数 ---
        self.num_agents = 3
        self.episode_limit = 96 * 2
        self.future_horizon = 24  # 固定未来视界 K=24（业务要求）
        
        # 电池参数（SoC ∈ [0,1]）
        self.battery_capacity = 1.0   # 电池容量（能量单位）
        self.max_charge_rate = 0.5/4    # 最大充放功率（功率单位）
        self.efficiency = 0.95        # 充放效率
        self.init_soc = 0.5           # 初始 SoC（0~1）
        self.dt = 0.25                 # 时间步长（小时）
        
        # SOC 约束
        self.soc_min = 0.05
        self.soc_max = 0.95
        self.soc_target = 0.5         # SoC 正则化的目标值
        self.soc_eps = 1e-3           # 动作越限判定阈值
        
        # 奖励超参数
        self.w_pv_rolling = 0
        self.w_pen = 5.0              # 动作越限惩罚系数
        self.w_soc = 0.1              # SoC 正则系数
        self.lambda_bonus = 0.01      # 吞吐量奖励系数

        # --- 观测空间配置 (方便做消融实验) ---
        # 支持: 'time'(2), 'price'(K+1), 'load'(K+1), 'soc'(1)
        # 注意：此列表须与 hems_env._build_observation_space / _get_obs_matrix 保持一致
        self.obs_config = ["time", "price", "load", "soc"]

        # --- 训练核心参数 ---
        self.algorithm = "MADDPG"       # "MADDPG" | "MATD3"
        self.reward_type = "composite"  # "composite" | "sparse"（见 common/rewards/）
        self.actor_type  = "mlp"        # "mlp"（见 models/registry.py ACTOR_REGISTRY）
        # 由 core.builder.resolve_algorithm_model_config() 按 algorithm 自动解析。
        self.critic_type = None         # None | "maddpg_mlp" | "matd3_mlp"
        self.forecaster_type = "perfect" # "perfect" | "naive" | "lstm"（见 forecast/）
        # forecaster_type="lstm" 时推荐提供 outputs/best_lstm.pt，
        # 并在同目录放置 best_lstm_meta.json / best_lstm_scaler.pkl。
        self.lstm_model_path = None
        self.lstm_meta_path = None
        self.lstm_scaler_path = None
        self.train_episodes = 1000
        self.max_train_steps = self.train_episodes * self.episode_limit
        self.num_envs = 32
        # self.rollout_steps = 32
        # self.update_epochs = 8
        self.batch_size = 4096
        self.max_action = 1.0
        self.buffer_size = int(1e6)
        self.update_interval = 1
        self.updates_per_step = 1
        
        # --- 网络与优化器参数 ---
        self.hidden_dim = 256
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
        self.N = self.num_agents  # 算法模块通过 args.N 访问智能体数量
