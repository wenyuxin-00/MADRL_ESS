# MADRL_ESS

面向配电网潮流约束场景的多智能体储能充放电优化工程。当前仓库已经收束为一条主线：`GridEnv` + processed prosumer dataset + `MADDPG` / `MATD3` 训练与评估。

## Mainline

默认主线只保留一套可直接训练的组合：
- 环境：`GridEnv`
- 数据：`data/processed/prosumer`
- 预测：`perfect` 或 `lstm`
- 默认入口：`compose_experiment_config()`
- 推荐训练 notebook：`notebooks/madrl/train_madrl_grid.ipynb`

现在不再需要在主线配置里切换旧环境、旧数据模式或额外 profile 选择器。常用参数都集中在 `configs/experiment_config.py`，notebook 里再按实验需要覆盖。

## Data

训练主线默认读取以下处理后数据文件：
- `data/processed/prosumer/household.csv`
- `data/processed/prosumer/heatpump.csv`
- `data/processed/prosumer/pv_reference.csv`
- `data/processed/prosumer/price.csv`

如果这些文件已经存在，就可以直接开始训练或训练 LSTM 预测器。

## Quick Start

安装依赖：

```bash
pip install -r requirements.txt
```

打开主训练 notebook：
- `notebooks/madrl/train_madrl_grid.ipynb`

这个 notebook 的逻辑保持为一条主线：
1. 在一个统一参数单元里设置运行时、数据切片和训练超参数。
2. 生成主线实验配置。
3. 构建 runner 并开始训练。
4. 绘制训练结果和测试回放。
5. 保存模型并对比 `perfect` / `normal` 预测模式下的 MPC 基线。

如果需要单独训练 LSTM 预测器：
- `notebooks/forecast/forecast_lstm.ipynb`

## Default Config

`compose_experiment_config()` 返回的默认配置已经是主线配置，常见字段包括：

```python
cfg = compose_experiment_config()

cfg.data.agent_profiles == ["SFH12", "SFH14", "SFH16"]
cfg.data.train_year == 2019
cfg.data.test_year == 2020
cfg.data.load_components == ["household", "heatpump"]
cfg.data.pv_reference == "south"
cfg.reward.type == "grid_composite"
cfg.obs.local_features == ["time", "soc"]
cfg.obs.sequence_features == ["price", "load", "pv"]
```

最常改的参数主要在两处：
- `configs/experiment_config.py`：长期默认值
- `notebooks/madrl/train_madrl_grid.ipynb`：当前实验覆盖值

## Repository Guide

```text
configs/
  experiment_config.py      # dataclass 默认配置
  profiles.py               # notebook 友好的 compose_experiment_config

data/loaders/
  prosumer.py               # processed prosumer 主线数据集
  registry.py               # 主线 dataset 构建入口

envs/
  grid_env.py               # GridEnv 主环境
  grid/                     # 电网拓扑、潮流和分析组件
  observation/              # 默认观测构造器

predictors/
  oracle.py                 # perfect forecaster
  lstm_*.py                 # LSTM 模型与推理
  training.py               # LSTM 训练与 artifact 管理

scripts/
  builder.py                # build_env / build_train_runner
  train.py                  # 训练循环
  evaluate.py               # 评估入口
  plots/                    # 绘图工具
  utils/                    # notebook / runtime 辅助工具

notebooks/
  madrl/train_madrl_grid.ipynb
  madrl/grid_network_analysis.ipynb
  forecast/forecast_lstm.ipynb
```

## Outputs

默认训练产物位于：
- `artifacts/training/checkpoints/`
- `artifacts/training/tensorboard/`
- `artifacts/forecast/lstm/`

## Notes

- 图表文字统一保持英文，避免不同环境下的字体告警。
- 代码注释和文档以 UTF-8 保存，避免乱码。
- 如果只想确认主线可跑，优先使用 `train_madrl_grid.ipynb`，不要从旧兼容路径起步。
