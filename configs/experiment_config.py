"""Dataclass-based experiment configuration for the grid training mainline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EnvConfig:
    """Environment settings for the default GridEnv workflow."""

    num_agents: int = 3
    episode_limit: int = 96 * 2
    future_horizon: int = 24
    battery_capacity: float = 5.0
    max_charge_rate: float = 2.5
    efficiency: float = 0.95
    init_soc: float = 0.5
    dt: float = 0.25
    soc_min: float = 0.05
    soc_max: float = 0.95
    soc_target: float = 0.5
    storage_power_scale: float = 12.0
    storage_capacity_scale: float = 12.0

@dataclass
class RewardConfig:
    """Reward selection and weights."""

    type: str = "grid_composite"
    w_pen: float = 6.0
    w_soc: float = 0.30
    lambda_bonus: float = 0.001
    w_global_safe: float = 1.0
    w_sens_credit: float = 0.2
    sens_credit_scale: float = 0.05


@dataclass
class ObsConfig:
    """Observation-builder settings."""

    local_features: list[str] = field(default_factory=lambda: ["time", "soc"])
    sequence_features: list[str] = field(default_factory=lambda: ["price", "load", "pv"])
    adjacency_type: str = "identity"

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
    history_window: int = 96
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
class DataConfig:
    """Processed prosumer dataset selection."""

    data_dir: str | Path | None = None
    agent_profiles: list[str] = field(default_factory=lambda: ["SFH12", "SFH14", "SFH16"])
    train_year: int = 2019
    test_year: int = 2020
    train_start_date: str | None = None
    train_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None
    load_components: list[str] = field(default_factory=lambda: ["household", "heatpump"])
    pv_reference: str = "south"
    pv_capacity_kw: list[float] = field(default_factory=list)
    load_scale: list[float] = field(default_factory=list)
    pv_scale: list[float] = field(default_factory=list)
    storage_scale: list[float] = field(default_factory=list)

@dataclass
class TrainConfig:
    """Training-loop settings."""

    train_episodes: int = 1000
    max_train_steps: int | None = None
    num_envs: int = 1
    vec_env_type: str = "dummy"
    batch_size: int = 4096
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

    def resolved_max_train_steps(self, episode_limit: int) -> int:
        if self.max_train_steps is not None:
            return int(self.max_train_steps)
        return int(self.train_episodes * episode_limit)

    def resolved_noise_std_decay(self) -> float:
        if self.noise_decay_steps <= 0:
            return 0.0
        return float((self.noise_std_init - self.noise_std_min) / self.noise_decay_steps)


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
    action_dim: int = 1


@dataclass
class GridConfig:
    """Power-flow and topology settings for GridEnv."""

    sb_code: str = "1-LV-rural1--0-sw"
    pf_solver: str = "nr"
    agent_bus_ids: list[int] = field(default_factory=lambda: [10, 6, 12])
    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    line_max_loading_pct: float = 100.0
    w_v_pen: float = 10.0
    w_line_pen: float = 10.0
    w_trafo_pen: float = 10.0
    sensitivity_delta_kw: float = 1.0
    train_compact_info: bool = True
    sensitivity_trigger_action_delta_kw: float = 1.0
    sensitivity_trigger_load_delta_kw: float = 2.0
    sensitivity_trigger_psi_delta: float = 0.001
    sensitivity_max_staleness_steps: int = 32
    sensitivity_trigger_on_pf_recovery: bool = True
    sensitivity_trigger_on_violation_change: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    grid: GridConfig = field(default_factory=GridConfig)
