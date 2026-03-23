# MADRL_ESS

面向配电网潮流约束场景的多智能体储能训练工程。当前仓库只保留一条主线：基于 `GridEnv` 的 MADDPG / MATD3 训练与评估。

## 当前主线

默认实验配置已经收束到 grid 主线：

- 环境：`grid_pf`
- 数据集：`csv_prosumer`
- 奖励：`grid_composite`
- 电网 profile：`rural1_phase1`
- 默认入口：`compose_experiment_config()`

这意味着不再需要手动 patch 旧环境、旧控制器或额外的 grid profile，默认配置就是可训练的带潮流环境配置。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

默认训练依赖以下文件：

- `data/simbench_2016_train.csv`
- `data/simbench_2016_test.csv`

如果数据还没准备好，可以执行 [notebooks/data/prepare_simbench_data.ipynb](notebooks/data/prepare_simbench_data.ipynb)。

### 3. 直接跑默认 grid 训练

首选 notebook 入口：

- [notebooks/madrl/train_madrl_grid.ipynb](notebooks/madrl/train_madrl_grid.ipynb)

它只保留最短训练闭环：

1. 生成默认 grid 实验配置
2. 构建 `GridEnv` 训练 runner
3. 执行训练
4. 绘制训练期 reward 分解
5. 保存 checkpoint

如果你更想走脚本入口，可以运行：

```bash
python scripts/run_debug_training.py
```

## 默认配置说明

`configs.compose_experiment_config()` 默认会生成下面这组组合：

```python
cfg = compose_experiment_config()
```

等价于一条默认 grid 主线：

- `cfg.env.env_type == "grid_pf"`
- `cfg.data.dataset_type == "csv_prosumer"`
- `cfg.reward.type == "grid_composite"`
- 自动应用 `rural1_phase1` deployment/profile

常见的最小覆盖方式只有这些：

```python
cfg = compose_experiment_config(profile="debug", algorithm="MADDPG")
cfg.train.train_episodes = 256
```

## 目录概览

```text
configs/
  experiment_config.py      # dataclass 默认配置
  profiles.py               # 组合配置入口 compose_experiment_config()

envs/
  grid_env.py               # GridEnv 外部环境接口
  registry.py               # 只保留 grid_pf
  grid/
    config/                 # grid profile 与 deployment 配置
    core/                   # GridCore / GridStepResult / net builder
    analysis/               # sensitivity 等 grid 分析工具
    topology/               # rural1_fixed 拓扑

controllers/
  madrl_controller.py       # 多智能体控制器封装
  zero_controller.py        # 零动作基线
  madrl/                    # MADDPG / MATD3 实现

scripts/
  builder.py                # build_env / build_train_runner
  train.py                  # 训练主循环
  evaluate.py               # 评估入口
  run_debug_training.py     # 调试训练入口
  plots/                    # 训练/评估可视化
  recorders/                # 轨迹记录

notebooks/
  madrl/train_madrl_grid.ipynb      # 默认 grid 训练 notebook
  madrl/grid_network_analysis.ipynb # grid 网络分析 notebook
```

## 训练产物

默认训练产物保存在 `artifacts/training/`：

- `artifacts/training/checkpoints/`：模型权重
- `artifacts/training/tensorboard/`：TensorBoard 日志

## 保留与删除

当前仓库只保留和 grid 主线一致的内容：

- 保留：`GridEnv`、`GridCore`、`MADDPG`、`MATD3`、`ZeroController`
- 删除：`EnergyStorageEnv / hems_env.py / energy_storage`
- 删除：旧占位控制器与对比 notebook
- 删除：旧主线专用脚本与分析入口

## 建议使用方式

如果你只是想确认主线可跑：

1. 先准备 `simbench_2016_train/test.csv`
2. 直接运行 `notebooks/madrl/train_madrl_grid.ipynb`
3. 检查 `artifacts/training/checkpoints/` 是否生成模型

如果你要继续做算法实验，优先只改：

- `configs/experiment_config.py`
- `configs/profiles.py`
- `envs/grid_env.py`
- `envs/rewards/grid_composite.py`
- `controllers/madrl/`
