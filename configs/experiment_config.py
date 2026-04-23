from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

CANONICAL_AGENT_PROFILES = ["SFH12", "SFH18", "SFH20"]
CANONICAL_AGENT_BUS_IDS = [12, 4, 2]
CANONICAL_LOAD_SCALE = [20.0, 20.0, 20.0]
CANONICAL_PV_SCALE = [1.0, 1.0, 1.0]
CANONICAL_BATTERY_CAPACITY_KWH = [50.0, 50.0, 50.0]
CANONICAL_TEST_START_DATE = "2020-05-01"
CANONICAL_TEST_END_DATE = "2020-05-07"
COMPARE_SCHEME_ORDER = [
    "global_misocp",
    "local_mpc_perfect",
    "local_mpc_lstm",
    "admm_mpc_lstm",
    "madrl_base",
    "madrl_base_safe",
    "madrl_projection_safe",
]
RECORD_SCHEME_CATEGORIES = {
    "global_misocp": "mpc",
    "local_mpc_perfect": "mpc",
    "local_mpc_lstm": "mpc",
    "admm_mpc_lstm": "mpc",
    "madrl_base": "madrl",
    "madrl_base_safe": "madrl",
    "madrl_projection_safe": "madrl",
}
MADRL_NOTEBOOK_SPECS = {
    "train_base": {
        "scheme_name": "madrl_base",
        "display_label": "MADRL + No Safety",
        "algorithm": "MATD3",
        "env_name": "GridTrainBase",
        "experiment_name": "train_base",
        "num_envs": 4,
        "reward": {
            "w_soc_pen": 1,
            "w_voltage_pen": 0.0,
            "w_line_pen": 0.0,
            "w_trafo_pen": 0.0,
        },
    },
    "train_base_safe": {
        "scheme_name": "madrl_base_safe",
        "display_label": "MADRL + Safety Penalty",
        "algorithm": "MATD3",
        "env_name": "GridTrainBaseSafe",
        "experiment_name": "train_base_safe",
        "num_envs": 4,
        "reward": {
            "w_soc_pen": 1,
            "w_voltage_pen": 400.0,
            "w_line_pen": 0.0,
            "w_trafo_pen": 10.0,
        },
    },
    "train_projection_safe": {
        "scheme_name": "madrl_projection_safe",
        "display_label": "MADRL + Safety Projection",
        "algorithm": "MATD3_SAFE_POC",
        "env_name": "GridTrainProjectionSafe",
        "experiment_name": "train_projection_safe",
        "num_envs": 4,
        "reward": {
            "w_soc_pen": 1,
            "w_voltage_pen": 400.0,
            "w_line_pen": 0.0,
            "w_trafo_pen": 10.0,
        },
        "safety": {
            "enabled": True,
        },
    },
}
ADMM_NOTEBOOK_SPEC = {
    "scheme_name": "admm_mpc_lstm",
    "display_label": "ADMM MPC + LSTM Forecast",
    "prediction_mode": "normal",
    "w_soc_pen": 2.0,
    "show_progress": True,
    "rho_init": None,
    "rho_min": 1e-3,
    "rho_max": 1e3,
    "rho_adaptation": "residual_balancing",
    "max_iters": 100,
    "max_iters_first_step": 300,
    "primal_tol": 1e-3,
    "dual_tol": 1e-3,
}


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_managed_signal_training_overrides() -> dict[str, dict[str, object]]:
    return {
        "wholesale_price": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.10,
            "batch_size": 1024,
            "epochs": 20,
            "lr": 1e-3,
        },
        "load": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.10,
            "batch_size": 1024,
            "epochs": 20,
            "lr": 1e-3,
        },
        "pv": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.10,
            "batch_size": 1024,
            "epochs": 20,
            "lr": 8e-4,
        },
    }


@dataclass
class DataConfig:
    data_dir: str | Path | None = None
    agent_profiles: list[str] = field(default_factory=lambda: list(CANONICAL_AGENT_PROFILES))
    train_year: int = 2019
    test_year: int = 2020
    train_start_date: str | None = None
    train_end_date: str | None = None
    test_start_date: str | None = CANONICAL_TEST_START_DATE
    test_end_date: str | None = CANONICAL_TEST_END_DATE
    load_components: list[str] = field(default_factory=lambda: ["household", "heatpump"])
    pv_reference: str = "south"
    pv_capacity_kw: list[float] = field(default_factory=list)
    load_scale: list[float] = field(default_factory=lambda: list(CANONICAL_LOAD_SCALE))
    pv_scale: list[float] = field(default_factory=lambda: list(CANONICAL_PV_SCALE))

    def resolved_load_scale(self, n_agents: int) -> list[float]:
        values = [float(value) for value in self.load_scale]
        return values or [1.0] * int(n_agents)

    def resolved_pv_scale(self, n_agents: int) -> list[float]:
        values = [float(value) for value in self.pv_scale]
        return values or [1.0] * int(n_agents)


@dataclass
class TrainConfig:
    train_episodes: int = 500
    max_train_steps: int | None = None
    num_envs: int = 1
    vec_env_type: str = "dummy"
    parallel_episode_sampling: str = "unique_active"
    batch_size: int = 1024
    buffer_size: int = int(1e6)
    update_interval: int = 1
    updates_per_step: int = 1
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    noise_std_init: float = 0.4
    noise_std_min: float = 0.2
    noise_decay_steps: float = 3e5
    use_noise_decay: bool = True
    show_progress: bool = True
    # The bar advances per interaction step; postfix metrics refresh every N completed episodes.
    progress_postfix_interval: int = 100
    progress_episode_interval: int = 100
    progress_write_interval_seconds: float = 5.0

    def resolved_max_train_steps(self, episode_limit: int) -> int:
        if self.max_train_steps is not None:
            return int(self.max_train_steps)
        return int(self.train_episodes * episode_limit)

    def resolved_noise_std_decay(self) -> float:
        if self.noise_decay_steps <= 0:
            return 0.0
        return float((self.noise_std_init - self.noise_std_min) / self.noise_decay_steps)


@dataclass
class EnvConfig:
    num_agents: int = len(CANONICAL_AGENT_PROFILES)
    episode_limit: int = 96
    future_horizon: int = 24
    battery_capacity: float | list[float] = field(default_factory=lambda: list(CANONICAL_BATTERY_CAPACITY_KWH))
    max_charge_rate: float = 0.5
    efficiency: float = 0.95
    init_soc: float = 0.5
    dt: float = 0.25
    soc_min: float = 0.05
    soc_max: float = 0.95
    soc_target: float = 0.5


@dataclass
class ForecastConfig:
    type: str = "perfect"
    target_signals: list[str] = field(default_factory=lambda: ["wholesale_price", "load", "pv"])
    history_window: int = 96 * 1
    load_model_mode: str = "per_agent"
    load_time_feature_mode: str = "hour_week_year"
    pv_time_feature_mode: str = "hour_week_year"
    load_hybrid_mode: str = "baseline_blend"
    pv_postprocess_mode: str = "physical_clip"
    load_baseline_mode: str = "last_value"
    load_blend_candidates: tuple[float, ...] = field(default_factory=lambda: tuple(index / 10.0 for index in range(11)))
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    lstm_batch_size: int = 512
    lstm_epochs: int = 4
    lstm_lr: float = 1e-3
    lstm_train_ratio: float = 0.7
    lstm_val_ratio: float = 0.15
    auto_train_missing: bool = True
    lstm_artifact_root: str | Path | None = None
    signal_training_overrides: dict[str, dict[str, object]] = field(
        default_factory=_default_managed_signal_training_overrides
    )
    load_component_split: bool = True
    load_scaler_type: str = "robust"


@dataclass
class MpcConfig:
    branch_current_tiebreaker_eur_per_pu_step: float = 1e-4


@dataclass
class RewardConfig:
    w_soc_pen: float = 0.5
    # export_subsidy_eur_per_kwh: float = 0.079
    export_subsidy_eur_per_kwh: float = 0.0
    import_price_markup_eur_per_kwh: float = 0.20
    w_voltage_pen: float = 400.0
    w_line_pen: float = 0.0
    w_trafo_pen: float = 10.0


@dataclass
class ObsConfig:
    local_features: list[str] = field(default_factory=lambda: ["calendar_time", "soc"])
    sequence_features: list[str] = field(default_factory=lambda: ["wholesale_price", "load", "pv"])
    adjacency_type: str = "identity"
    normalization_enabled: bool = True
    wholesale_price_normalization: str = "robust_tanh"
    load_normalization: str = "robust_tanh"
    pv_normalization: str = "capacity"
    soc_normalization: str = "linear_pm1"
    normalization_clip_low_quantile: float = 0.01
    normalization_clip_high_quantile: float = 0.99
    wholesale_price_tanh_scale: float = 2.0
    load_tanh_scale: float = 3.0
    pv_tanh_scale: float = 2.0


@dataclass
class ModelConfig:
    family: str = "mlp"
    actor_head_type: str = "deterministic_continuous"
    critic_head_type: str | None = None
    hidden_dim: int = 256
    max_action: float = 1.0
    use_orthogonal_init: bool = True
    use_grad_clip: bool = True
    grad_clip_norm: float = 10.0


@dataclass
class AlgoConfig:
    name: str = "MADDPG"
    gamma: float = 0.999
    tau: float = 0.01
    policy_update_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5


@dataclass
class RuntimeConfig:
    device: torch.device = field(default_factory=_default_device)
    seed: int = 0
    execution_mode: str = "performance"
    require_cuda: bool = False
    matmul_precision: str = "high"
    allow_tf32: bool | None = None
    cudnn_benchmark: bool | None = None
    cudnn_deterministic: bool | None = None
    use_deterministic_algorithms: bool | None = None
    pin_memory: bool | None = None
    non_blocking_transfers: bool | None = None
    enable_amp: bool | None = None
    amp_dtype: str = "bfloat16"
    enable_compile: bool | None = None
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    worker_rank: int = 0
    observation_schema: dict[str, tuple[int, ...]] | None = None
    observation_layout: dict[str, dict[str, object]] | None = None
    observation_normalization_state: dict[str, object] | None = None
    action_dim: int = 1
    progress_state_path: str | None = None
    shared_data_dir: str | None = None
    shared_data_signature: str | None = None
    forecast_ready: dict[str, object] | None = None
    effective_split_controls: dict[str, object] | None = None
    selected_episode_indices: list[int] | None = None


@dataclass
class GridConfig:
    sb_code: str = "1-LV-rural1--0-sw"
    pf_solver: str = "nr"
    agent_bus_ids: list[int] = field(default_factory=lambda: list(CANONICAL_AGENT_BUS_IDS))
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    line_max_loading_pct: float = 100.0
    train_compact_info: bool = True


@dataclass
class SafetyConfig:
    enabled: bool = False
    projector_mode: str = "joint_linearized"
    projection_iters: int = 3
    voltage_margin_pu: float = 0.005
    line_margin_pct: float = 5.0
    trafo_margin_pct: float = 5.0
    linearization_delta_kw: float = 0.25
    record_diagnostics: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    mpc: MpcConfig = field(default_factory=MpcConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
