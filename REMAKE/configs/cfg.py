from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib, json


@dataclass
class GridCfg:
    sb_code: str = "1-LV-rural1--0-sw"
    pf_solver: str = "nr"
    agent_bus_ids: tuple[int, ...] = (12, 4, 2)
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    trafo_limit_kw: float | None = None
    line_limit_kw: float = 80.0
    line_max_loading_pct: float = 100.0


@dataclass
class DataCfg:
    data_root: str = "REMAKE/datasets"
    agent_profiles: tuple[str, ...] = ("SFH12", "SFH18", "SFH20")
    share_data_train_test: tuple[int, int] = (2019, 2020)
    train_start_date: str = "2019-01-01"
    train_end_date: str = "2019-12-31"
    eval_start_date: str = "2020-04-01"
    eval_end_date: str = "2020-04-15"
    load_components: tuple[str, ...] = ("household", "heatpump")
    pv_reference: str = "south"
    load_scale: tuple[float, ...] = (20.0, 20.0, 20.0)
    pv_scale: tuple[float, ...] = (1.0, 1.0, 1.0)
    pv_capacity_kw: tuple[float, ...] = ()


@dataclass
class EnvCfg:
    num_agents: int = 3
    episode_steps: int = 96
    train_window_days: int = 7
    dt_hours: float = 0.25
    soc_min: float = 0.05
    soc_max: float = 0.95
    init_soc: float = 0.05
    train_init_soc_low: float = 0.05
    train_init_soc_high: float = 0.05
    efficiency: float = 0.95
    battery_capacity_kwh: tuple[float, ...] = (100.0, 100.0, 100.0)
    max_charge_rate: float = 0.5
    window_stride_days: int = 1


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
    train_episodes: int = 500
    num_envs: int = 4
    parallel_episode_sampling: str = "unique_active"
    buffer_size: int = 100_000
    batch_size: int = 512
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    learning_starts: int = 5_376
    actor_learning_starts: int = 8_064
    n_step_return: int = 96
    updates_per_step: int = 1
    update_interval: int = 1
    noise_std_init: float = 0.4
    noise_std_min: float = 0.2
    noise_decay_steps: int = 300_000
    feasible_random_exploration_start: float = 0.5
    feasible_random_exploration_end: float = 0.05
    feasible_random_exploration_decay_steps: int = 50_000


@dataclass
class EvalCfg:
    n_episodes: int = 15
    episode_indices: tuple[int, ...] | None = None
    forecast_modes: tuple[str, ...] = ("perfect", "lstm")


@dataclass
class RewardCfg:
    action_boundary_penalty_weight: float = 0.05
    soc_boundary_regularization_weight: float = 0.005
    throughput_bonus_eur_per_kwh_max: float = 0.002
    soc_boundary_epsilon: float = 0.02
    soc_boundary_margin: float = 0.02
    import_price_markup_eur_per_kwh: float = 0.0
    export_subsidy_eur_per_kwh: float = 0.0
    w_voltage_pen: float = 1.0
    w_line_pen: float = 1.0
    w_trafo_pen: float = 1.0


@dataclass
class ObsCfg:
    sequence_length: int = 49
    normalization_enabled: bool = True
    wholesale_price_tanh_scale: float = 2.0
    wholesale_price_spread_scale_eur_per_kwh: float = 0.20
    load_tanh_scale: float = 3.0
    normalization_clip_low_quantile: float = 0.01
    normalization_clip_high_quantile: float = 0.99


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
    action_dim: int = 2
    max_action: float = 1.0
    use_grad_clip: bool = True
    grad_clip_norm: float = 10.0


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
