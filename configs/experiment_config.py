"""Dataclass-based experiment configuration for the grid training mainline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_managed_signal_training_overrides() -> dict[str, dict[str, object]]:
    return {
        "price": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.10,
            "batch_size": 1024,
            "epochs": 20,
            "lr": 1e-3,
        },
        "load": {
            "hidden_size": 96,
            "num_layers": 2,
            "dropout": 0.10,
            "batch_size": 1024,
            "epochs": 20,
            "lr": 1e-3,
        },
        "pv": {
            "hidden_size": 96,
            "num_layers": 1,
            "dropout": 0.00,
            "batch_size": 1024,
            "epochs": 20,
            "lr": 8e-4,
        },
    }

@dataclass
class DataConfig:
    """Processed prosumer dataset selection."""

    data_dir: str | Path | None = None
    agent_profiles: list[str] = field(default_factory=lambda: ["SFH12", "SFH14", "SFH16", "SFH18", "SFH20"])
    train_year: int = 2019
    test_year: int = 2020
    train_start_date: str | None = None
    train_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None
    load_components: list[str] = field(default_factory=lambda: ["household", "heatpump"])
    pv_reference: str = "south"
    pv_capacity_kw: list[float] = field(default_factory=list)
    load_scale: list[float] = field(default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0])
    pv_scale: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 5.0, 5.0])

    def resolved_load_scale(self, n_agents: int) -> list[float]:
        values = list(self.load_scale)
        if not values:
            return [1.0] * int(n_agents)
        return [float(value) for value in values]

    def resolved_pv_scale(self, n_agents: int) -> list[float]:
        values = list(self.pv_scale)
        if not values:
            return [1.0] * int(n_agents)
        return [float(value) for value in values]


@dataclass
class TrainConfig:
    """Training-loop settings."""

    train_episodes: int = 1000
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
    progress_postfix_interval: int = 10
    progress_episode_interval: int = 10
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
    """Environment settings for the default GridEnv workflow."""

    num_agents: int = 5
    episode_limit: int = 96 * 2
    future_horizon: int = 24
    battery_capacity: float | list[float] = field(default_factory=lambda: [40.0] * 5)
    max_charge_rate: float = 0.5
    efficiency: float = 0.95
    init_soc: float = 0.5
    dt: float = 0.25
    soc_min: float = 0.05
    soc_max: float = 0.95
    soc_target: float = 0.5


@dataclass
class RewardConfig:
    """Reward weights for the default NormalReward."""

    w_soc_pen: float = 2
    export_subsidy_eur_per_kwh: float = 0.079
    import_price_adder_eur_per_kwh: float = 0.20
    w_voltage_pen: float = 400.0
    w_line_pen: float = 0.0
    w_trafo_pen: float = 10.0


@dataclass
class MpcConfig:
    """Solver-side regularization knobs for offline MISOCP analysis."""

    branch_current_tiebreaker_eur_per_pu_step: float = 0.0
    physics_refinement_mode: str = "two_stage_min_branch_l"
    physics_refinement_slack_ratio: float = 2e-2
    physics_refinement_slack_abs_floor_eur: float = 2.0
    physics_refinement_slack_ratio_schedule: list[float] = field(
        default_factory=lambda: [2e-2, 5e-2]
    )
    physics_refinement_slack_abs_floor_schedule_eur: list[float] = field(
        default_factory=lambda: [2.0, 5.0]
    )
    physics_refinement_enable_aggressive_third_tier: bool = False
    physics_refinement_aggressive_third_tier_ratio: float = 1e-1
    physics_refinement_aggressive_third_tier_abs_floor_eur: float = 10.0
    physics_refinement_cap_utilization_trigger: float = 0.95
    physics_refinement_branch_l_gap_ratio_trigger: float = 0.01
    physics_refinement_time_limit_sec: float = 20.0
    physics_refinement_total_time_limit_sec: float = 40.0
    physics_refinement_target_mean_solver_gap_kw: float = 3.0
    physics_refinement_target_max_solver_gap_kw: float = 15.0
    physics_refinement_target_export_gap_ratio: float = 0.05
    physics_refinement_use_full_start: bool = True


@dataclass
class ObsConfig:
    """Observation-builder settings."""

    local_features: list[str] = field(default_factory=lambda: ["calendar_time", "soc"])
    sequence_features: list[str] = field(default_factory=lambda: ["price", "load", "pv"])
    adjacency_type: str = "identity"
    normalization_enabled: bool = True
    price_normalization: str = "robust_tanh"
    load_normalization: str = "robust_tanh"
    pv_normalization: str = "capacity"
    soc_normalization: str = "linear_pm1"
    normalization_clip_low_quantile: float = 0.01
    normalization_clip_high_quantile: float = 0.99
    price_tanh_scale: float = 2.0
    load_tanh_scale: float = 3.0
    pv_tanh_scale: float = 2.0

@dataclass
class ModelConfig:
    """Neural-network family configuration."""

    family: str = "mlp"
    actor_head_type: str = "deterministic_continuous"
    critic_head_type: str | None = None
    hidden_dim: int = 256
    max_action: float = 1.0
    use_orthogonal_init: bool = True
    use_grad_clip: bool = True
    grad_clip_norm: float = 10.0
    transformer_num_heads: int = 4
    transformer_num_layers: int = 1
    graph_num_layers: int = 2


@dataclass
class AlgoConfig:
    """MADRL algorithm configuration."""

    name: str = "MADDPG"
    gamma: float = 0.999
    tau: float = 0.01
    policy_update_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5


@dataclass
class ForecastConfig:
    """Forecasting settings."""

    type: str = "perfect"
    target_signals: list[str] = field(default_factory=lambda: ["price", "load", "pv"])
    history_window: int = 96 * 3
    load_model_mode: str = "per_agent"
    load_time_feature_mode: str = "hour_week_year"
    pv_time_feature_mode: str = "hour_week_year"
    load_hybrid_mode: str = "baseline_blend"
    pv_postprocess_mode: str = "physical_clip"
    load_baseline_mode: str = "last_value"
    load_blend_candidates: tuple[float, ...] = field(default_factory=lambda: tuple(i / 10.0 for i in range(11)))
    heatpump_jump_relief_enabled: bool = False
    heatpump_jump_relief_threshold_kw: float = 0.8
    heatpump_jump_relief_min_weight: float = 0.3
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

    @property
    def lstm_model_path(self) -> None:
        """Legacy single-model entrypoints are no longer supported."""
        return None

    @lstm_model_path.setter
    def lstm_model_path(self, value: str | Path | None) -> None:
        if value is None:
            return
        raise ValueError(
            "forecast.lstm_model_path has been removed. "
            "Use managed LSTM artifacts under forecast.lstm_artifact_root instead."
        )




@dataclass
class RuntimeConfig:
    """Runtime configuration derived before model construction."""

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
    """Power-flow and topology settings for GridEnv."""

    sb_code: str = "1-LV-rural1--0-sw"
    pf_solver: str = "nr"
    agent_bus_ids: list[int] = field(default_factory=lambda: [10, 6, 12, 4, 2])
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    line_max_loading_pct: float = 100.0
    train_compact_info: bool = True


@dataclass
class SafetyConfig:
    """Optional safety-layer controls for isolated safe-policy experiments."""

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
    """Top-level experiment configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    mpc: MpcConfig = field(default_factory=MpcConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
