"""实验统一配置 dataclass 定义。

以纯数据结构形式定义所有实验参数（环境、算法、模型、奖励、预测器等），
不包含任何逻辑，仅供 profiles.py 等组合函数使用。

主要类:
    ExperimentConfig -- 顶层配置，组合以下子配置
    EnvConfig       -- 环境参数
    AlgoConfig      -- MADRL 算法参数
    ModelConfig      -- 神经网络模型参数
    RewardConfig     -- 奖励函数参数
    ObsConfig        -- 观测空间参数
    ForecastConfig   -- 预测器参数
    DataConfig       -- 数据集参数
    TrainConfig      -- 训练循环参数
    RuntimeConfig    -- 运行时配置
    GridConfig       -- 电网潮流约束参数
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
    # 仅保留给历史 price-only 单模型入口，新的主线统一走 artifact_root + target_signals。
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
    """统一的 runtime 配置与派生信息。"""

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
    worker_rank: int = 0
    observation_schema: dict[str, tuple[int, ...]] | None = None
    observation_layout: dict[str, dict[str, object]] | None = None
    action_dim: int = 1


@dataclass
class GridConfig:
    """配电网潮流约束配置。"""

    sb_code: str = "1-LV-rural1--0-sw"
    pf_solver: str = "nr"                                    # "nr"=Newton-Raphson, "dc"=线性化
    agent_bus_ids: list[int] = field(default_factory=lambda: [10, 6, 12])
    v_min_pu: float = 0.95                                   # 节点电压下限 (pu)
    v_max_pu: float = 1.05                                   # 节点电压上限 (pu)
    line_max_loading_pct: float = 100.0                      # 线路热极限 (%)
    w_v_pen: float = 10.0                                    # 电压越界 penalty 权重
    w_l_pen: float = 5.0                                     # 线路越载 penalty 权重


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
    grid: GridConfig = field(default_factory=GridConfig)
