"""科研实验使用的统一配置对象。

模块只保留一层轻量 dataclass，不引入额外配置框架。
当前主线需要的默认项尽量直观，同时保留少量明确会扩展的轴：
- `env.env_type`
- `data.dataset_type`
- `obs.builder_type`
- `forecast.type`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EnvConfig:
    """环境主配置。"""

    env_type: str = "energy_storage"
    num_agents: int = 3
    episode_limit: int = 96 * 2
    future_horizon: int = 24
    battery_capacity: float = 1.0
    max_charge_rate: float = 0.5 / 4
    efficiency: float = 0.95
    init_soc: float = 0.5
    dt: float = 0.25
    soc_min: float = 0.05
    soc_max: float = 0.95
    soc_target: float = 0.5


@dataclass
class RewardConfig:
    """奖励函数配置。"""

    type: str = "composite"
    w_pen: float = 5.0
    w_soc: float = 0.1
    lambda_bonus: float = 0.01


@dataclass
class ObsConfig:
    """观测拼装配置。"""

    builder_type: str = "default"
    local_features: list[str] = field(default_factory=lambda: ["time", "price", "load", "soc"])
    sequence_features: list[str] = field(default_factory=lambda: ["price", "load"])
    adjacency_type: str = "identity"


@dataclass
class ModelConfig:
    """模型家族配置。"""

    family: str = "mlp"
    actor_head_type: str = "deterministic_continuous"
    critic_head_type: str | None = None
    hidden_dim: int = 256
    max_action: float = 1.0
    use_orthogonal_init: bool = True
    use_grad_clip: bool = True
    transformer_num_heads: int = 4
    transformer_num_layers: int = 1
    graph_num_layers: int = 2


@dataclass
class AlgoConfig:
    """MADRL 算法配置。"""

    name: str = "MADDPG"
    gamma: float = 0.999
    tau: float = 0.01
    policy_update_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5


@dataclass
class ForecastConfig:
    """预测器配置。"""

    type: str = "perfect"
    naive_window: int = 96
    lstm_model_path: str | Path | None = None


@dataclass
class DataConfig:
    """数据集配置。"""

    dataset_type: str = "csv_price_load"
    data_dir: str | Path | None = None


@dataclass
class TrainConfig:
    """训练循环配置。"""

    train_episodes: int = 1000
    max_train_steps: int | None = None
    num_envs: int = 32
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

    def resolved_max_train_steps(self, episode_limit: int) -> int:
        """返回显式训练步数，或按 episode 自动推导的默认值。"""
        if self.max_train_steps is not None:
            return int(self.max_train_steps)
        return int(self.train_episodes * episode_limit)

    def resolved_noise_std_decay(self) -> float:
        """返回探索噪声的线性衰减斜率。"""
        if self.noise_decay_steps <= 0:
            return 0.0
        return float((self.noise_std_init - self.noise_std_min) / self.noise_decay_steps)


@dataclass
class RuntimeConfig:
    """运行时派生信息。"""

    device: torch.device = field(default_factory=_default_device)
    observation_schema: dict[str, tuple[int, ...]] | None = None
    observation_layout: dict[str, dict[str, object]] | None = None
    action_dim: int = 1


@dataclass
class ExperimentConfig:
    """实验全配置。"""

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
