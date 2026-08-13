# 参数配置
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib, json

# 电网拓扑固定
@dataclass
class GridCfg: 
    sb_code: str = "1-LV-rural1--0-sw"
    pf_solver: str = "nr"
    agent_bus_ids: tuple[int, ...] = (12, 4, 2) #三个智能体分别连接在12、4、2号母线上
    v_min_pu: float = 0.95 #电压范围0.95-1.05倍额定电压
    v_max_pu: float = 1.05 
    trafo_limit_kw: float | None = None
    line_limit_kw: float = 80.0 #所有线路的功率限制为80kW
    line_max_loading_pct: float = 100.0 #线路最大负载百分比，100%表示不考虑过载惩罚


@dataclass
class DataCfg:
    data_root: str = "datasets"
    agent_profiles: tuple[str, ...] = ("SFH12", "SFH18", "SFH20")
    share_data_train_test: tuple[int, int] = (2019, 2020)
    train_start_date: str = "2019-01-01"
    train_end_date: str = "2019-12-31"
    eval_start_date: str = "2020-04-01"
    eval_end_date: str = "2020-04-15"
    # eval_end_date: str = "2020-04-02"
    #总负荷 = household负荷 + heatpump负荷，pv发电量不计入总负荷
    load_components: tuple[str, ...] = ("household", "heatpump")
    pv_reference: str = "south"
    load_scale: tuple[float, ...] = (20.0, 20.0, 20.0) #负荷缩放因子，原始数据的负荷较小，乘以20后更符合实际情况
    pv_scale: tuple[float, ...] = (1.0, 1.0, 1.0) #光伏缩放因子，原始数据的光伏发电量较大，乘以1后保持不变，如果需要可以调整
    pv_capacity_kw: tuple[float, ...] = () 


@dataclass
class EnvCfg:
    num_agents: int = 3
    episode_steps: int = 96 #一天96个时间步，每个时间步15分钟
    train_window_days: int = 2 #每次训练一个2天的窗口（调试用） # 原: 7
    dt_hours: float = 0.25
    #电池参数，所有智能体共享相同的电池参数
    soc_min: float = 0.05
    soc_max: float = 0.95
    init_soc: float = 0.05
    train_init_soc_low: float = 0.05
    train_init_soc_high: float = 0.05
    efficiency: float = 0.95
    battery_capacity_kwh: tuple[float, ...] = (100.0, 100.0, 100.0)#每个智能体电池容量100kWh
    max_charge_rate: float = 0.5 #每个时间步最大充电/放电功率，单位为电池容量的倍数，例如0.5表示每个时间步最多充放电50kW
    window_stride_days: int = 1

    # EV parameters
    ev_enabled: bool = True

    # EV battery capacity for each agent
    ev_capacity_kwh: tuple[float, ...] = (60.0, 60.0, 60.0)

    # EV SOC limits
    ev_soc_min: float = 0.10
    ev_soc_max: float = 0.95

    # EV SOC when arriving home at 18:00
    ev_arrival_soc: float = 0.3

    # Required SOC when leaving home at 07:00
    ev_departure_soc_req: float = 0.90

    # Home charger maximum power
    ev_max_charge_kw: tuple[float, ...] = (11.0, 11.0, 11.0)

    # EV charging efficiency
    ev_efficiency: float = 0.95

    # EV connection window
    # With 15-min resolution: 18:00 = step 72, 07:00 = step 28
    ev_arrival_step: int = 72
    ev_departure_step: int = 28
    ev_state_continuity: bool = True
    ev_departure_constraint_mode: str = "soft"
    ev_hard_projection_enabled: bool = False
    ev_emergency_charging_enabled: bool = True
    ev_emergency_window_hours: float = 1.0
    ev_emergency_strategy: str = "required_power"

#强化学习算法参数
@dataclass
class AlgoCfg:
    name: str = "MADRL_PENALTY"
    gamma: float = 0.999
    tau: float = 0.01
    policy_update_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5


@dataclass
class TrainCfg:
    train_episodes: int =20 #原500（7天调试） → 20（2天调试）
    num_envs: int = 1 #原4（7天调试） → 1（2天调试）
    parallel_episode_sampling: str = "unique_active"
    buffer_size: int = 50_000 #100_000 (7天调试) → 50_000 (2天调试)
    batch_size: int = 128 #原来512（7天调试）-> 128（2天调试）
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    learning_starts: int = 1_536  # 原: 5_376 (7天调试) → 1_536 (2天调试)
    actor_learning_starts: int = 2_304  # 原: 8_064 (7天调试) → 2_304 (2天调试)
    n_step_return: int = 48 #原96（7天）-> 48（2天调试），n步回报的n不能大于episode_steps，否则会导致训练过程中计算n步回报时越界
    updates_per_step: int = 1
    update_interval: int = 1
    noise_std_init: float = 0.4
    noise_std_min: float = 0.2
    noise_decay_steps: int = 30_000 #原300_000(7天)-> 30_000（2天调试）
    feasible_random_exploration_start: float = 0.5
    feasible_random_exploration_end: float = 0.05
    feasible_random_exploration_decay_steps: int = 10_000 #原50_000（7天）-> 10_000（2天调试）


@dataclass
class EvalCfg:
    n_episodes: int = 15
    episode_indices: tuple[int, ...] | None = None
    forecast_modes: tuple[str, ...] = ("perfect", "lstm")

#奖励函数参数
@dataclass
class RewardCfg:
    action_boundary_penalty_weight: float = 0.05 #动作边界惩罚权重，超过充放电功率限制时的惩罚力度
    soc_boundary_regularization_weight: float = 0.005 #电池SOC边界正则化权重，鼓励智能体保持SOC在安全范围内
    throughput_bonus_eur_per_kwh_max: float = 0.002 #最大充放电奖励，单位为每kWh的欧元，实际奖励根据充放电量和价格进行缩放
    soc_boundary_epsilon: float = 0.02 #SOC边界epsilon，智能体在SOC接近边界时开始受到惩罚，具体来说，当SOC小于soc_min + epsilon或大于soc_max - epsilon时，开始计算边界惩罚
    soc_boundary_margin: float = 0.02 #SOC边界margin，智能体在SOC接近边界时的安全边距，例如当SOC小于soc_min + margin或大于soc_max - margin时，认为智能体处于危险状态，可以触发安全机制
    import_price_markup_eur_per_kwh: float = 0.0 #进口电价加价，单位为每kWh的欧元，实际进口电价 = wholesale_price + import_price_markup
    export_subsidy_eur_per_kwh: float = 0.0 #出口电价补贴，单位为每kWh的欧元，实际出口电价 = wholesale_price + export_subsidy
    w_voltage_pen: float = 1.0 #电压惩罚权重，电压越偏离额定值，惩罚越大
    w_line_pen: float = 1.0    #线路惩罚权重，线路负载越接近或超过限制，惩罚越大
    w_trafo_pen: float = 1.0   #变压器惩罚权重，变压器负载越接近或超过限制，惩罚越大
    # EV reward parameters
    ev_charging_cost_weight: float = 1.0
    ev_price_aware_penalty_weight: float = 0.0
    ev_price_aware_threshold_quantile: float = 0.70
    ev_departure_penalty_weight: float = 500.0
    ev_departure_target_penalty_weight: float = 0.0
    ev_progress_penalty_weight: float = 0.0
    ev_soc_regularization_weight: float = 0.005
    ev_projection_penalty_weight: float = 0.0
    ev_emergency_penalty_weight: float = 0.0


#观测处理参数
@dataclass
class ObsCfg:
    sequence_length: int = 49 #观测序列长度，包含当前时间步和之前的48个时间步，总共49个时间步
    normalization_enabled: bool = True 
    wholesale_price_tanh_scale: float = 2.0 #批量归一化后价格的tanh缩放因子，原始价格经过批量归一化后可能分布在较小范围内，乘以这个因子后再经过tanh函数，可以增加价格信号的区分度和非线性表达能力
    wholesale_price_spread_scale_eur_per_kwh: float = 0.20 #批量归一化后价格差的缩放因子，单位为每kWh的欧元，价格差 = 预测价格 - 当前价格，乘以这个因子后可以调整价格差在奖励函数中的影响力，例如0.20表示每kWh的价格差最多贡献0.20欧元的奖励或惩罚
    load_tanh_scale: float = 3.0
    normalization_clip_low_quantile: float = 0.01 
    normalization_clip_high_quantile: float = 0.99

#预测模型参数 LSTM 参数
@dataclass
class ForecastCfg:
    mode: str = "perfect"
    target_signals: tuple[str, ...] = ("wholesale_price", "load", "pv")
    lstm_artifact_dir: str | None = None
    history_window: int = 192
    lstm_batch_size: int = 1024
    lstm_train_ratio: float = 0.7
    lstm_val_ratio: float = 0.15
    price_lstm_epochs: int = 50
    price_lstm_hidden_size: int = 128
    price_lstm_num_layers: int = 2
    price_lstm_dropout: float = 0.1
    price_lstm_lr: float = 1e-3
    load_lstm_epochs: int = 20
    load_lstm_hidden_size: int = 64
    load_lstm_num_layers: int = 2
    load_lstm_dropout: float = 0.1
    load_lstm_lr: float = 1e-3
    load_lstm_scaler: str = "robust"
    load_lstm_time_features: str = "hour_week_year"
    load_blend_candidates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    pv_lstm_epochs: int = 20
    pv_lstm_hidden_size: int = 64
    pv_lstm_num_layers: int = 1
    pv_lstm_dropout: float = 0.1
    pv_lstm_lr: float = 8e-4
    pv_lstm_time_features: str = "hour_week_year"
    load_model_mode: str = "per_agent"
    load_component_split: bool = True


@dataclass
class ModelCfg:
    hidden_dim: int = 256
    action_dim: int = 3
    max_action: float = 1.0
    use_grad_clip: bool = True
    grad_clip_norm: float = 10.0

#安全机制参数
@dataclass
class SafetyCfg:
    enabled: bool = False
    projection_iters: int = 3
    voltage_margin_pu: float = 0.005
    line_margin_pct: float = 5.0
    trafo_margin_pct: float = 5.0


@dataclass
class RuntimeCfg:
    device: str = "cuda"
    seed: int = 0

#MPC参数
@dataclass
class MpcCfg:
    admm_max_iter: int = 20
    low_price_quantile: float = 0.35
    high_price_quantile: float = 0.65


@dataclass
class Cfg:
    grid: GridCfg = field(default_factory=GridCfg)
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    algo: AlgoCfg = field(default_factory=AlgoCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    forecast: ForecastCfg = field(default_factory=ForecastCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    mpc: MpcCfg = field(default_factory=MpcCfg)

    def hash8(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def with_algo(self, name: str) -> "Cfg":
        return replace(self, algo=replace(self.algo, name=name))
