# MADRL_ESS

## 项目简介

`MADRL_ESS` 是一个面向储能调度研究的多智能体强化学习代码仓。项目主线围绕两类算法展开：

- `MADDPG`
- `MATD3`

仓库同时保留了可扩展的模型、观测、预测器、奖励函数和评估接口，目标不是做“功能堆叠”，而是提供一个适合研究生长期维护、复现实验、继续扩展的科研代码基座。

这个仓库适合以下使用场景：

- 学生第一次接触多智能体强化学习，希望从完整实验链路入手
- 研究者想在现有 MADRL 主线下替换模型结构、观测设计或预测器
- 课程或组内项目需要一个结构清晰、便于讲解的实验仓库

## 项目整体架构

主流程可以概括为：

```text
configs
  -> core/builder.py
  -> datasets + forecast + reward + envs/observation + envs
  -> models + algorithms
  -> runners/train_runner.py
  -> evaluation + controllers
```

更具体地说：

```text
ExperimentConfig
  -> build_train_runner()
     -> build_dataset()
     -> build_forecaster()
     -> build_obs_builder()
     -> get_reward_fn()
     -> get_env_cls()
     -> build_actor_network() / build_critic_network()
     -> get_agent_cls()
  -> TrainRunner.run()
  -> evaluate_controller()
  -> plots / comparison
```

设计原则是“组装清晰，职责分开”：

- `configs/` 只描述实验配置，不负责训练
- `core/` 只做总装配，不堆实现细节
- `algorithms/` 只关心算法更新逻辑
- `models/` 只关心网络装配
- `envs/` 只关心环境与观测构造
- `forecast/` 只关心价格预测器
- `runners/` 只关心训练、日志和 checkpoint
- `evaluation/` 只关心评估与可视化

## 目录结构说明

下面只列出最重要的一级目录和关键二级模块：

```text
MADRL_ESS/
├─ algorithms/
│  ├─ base_agent.py
│  ├─ maddpg.py
│  ├─ matd3.py
│  └─ registry.py
├─ common/
│  ├─ project_paths.py
│  ├─ replay_buffer.py
│  ├─ vec_env.py
│  ├─ subproc_vec_env.py
│  └─ rewards/
├─ configs/
│  ├─ experiment_config.py
│  └─ profiles.py
├─ controllers/
├─ core/
│  └─ builder.py
├─ data/
├─ datasets/
├─ envs/
│  ├─ hems_env.py
│  └─ observation/
├─ evaluation/
├─ forecast/
│  ├─ artifacts.py
│  ├─ registry.py
│  ├─ naive.py
│  ├─ oracle.py
│  ├─ lstm_model.py
│  ├─ lstm_forecaster.py
│  └─ notebook_utils.py
├─ madrl/
│  └─ notebook_utils.py
├─ models/
│  ├─ assembly.py
│  ├─ family_adapters.py
│  ├─ registry.py
│  ├─ actors/
│  ├─ critics/
│  ├─ encoders/
│  └─ heads/
├─ notebooks/
│  ├─ data/
│  ├─ forecast/
│  └─ madrl/
├─ runners/
│  ├─ train_runner.py
│  └─ checkpoints.py
├─ scripts/
│  └─ run_debug_training.py
├─ tests/
│  ├─ conftest.py
│  ├─ support/
│  └─ test_*.py
├─ artifacts/
├─ requirements.txt
└─ README.md
```

各目录职责如下：

- `algorithms/`：算法本体。`maddpg.py` 和 `matd3.py` 实现训练更新；`registry.py` 负责算法注册。
- `common/`：跨模块通用工具。包括 replay buffer、向量环境、嵌套数据结构工具、奖励函数框架，以及统一项目路径约定。
- `configs/`：实验配置定义。`experiment_config.py` 放配置数据结构，`profiles.py` 提供适合 notebook 的组合入口。
- `controllers/`：评估阶段统一控制器接口。训练好的 MADRL、零动作基线、MPC 占位控制器和传统 DRL 占位控制器都在这里对齐接口。
- `core/`：总装配入口。`builder.py` 把配置转换成数据集、环境、模型、算法和 runner。
- `data/`：原始数据与整理后的 CSV 数据。
- `datasets/`：把 CSV 数据组织成环境可消费的训练/测试片段。
- `envs/`：强化学习环境。`hems_env.py` 是储能环境主实现；`observation/` 负责观测结构的模块化拼装。
- `evaluation/`：评估、对比、绘图和 episode 记录。
- `forecast/`：价格预测器模块。包含完美预测、朴素预测和 LSTM 预测器，以及 notebook 会复用的工具函数。
- `madrl/`：面向 notebook 的训练/对比辅助函数，不放算法实现本身。
- `models/`：网络装配层。先做 adapter，再做 encoder，最后接 actor/critic head。
- `notebooks/`：教学和实验入口 notebook，按数据整理、预测、MADRL 训练/对比分组。
- `runners/`：训练循环和 checkpoint 管理。
- `scripts/`：非 notebook 的轻量入口脚本。
- `tests/`：全部测试都放在这里，包含 fixture、helper 和 smoke test。
- `artifacts/`：训练日志、checkpoint、预测器产物等生成文件。它不是源码目录。

关键二级模块建议这样理解：

- `configs/profiles.py`：学生最常用的配置入口，负责把“debug / fast_train / model_family / reward_type”这些选择组合成完整实验配置。
- `core/builder.py`：项目主入口。想理解“配置最终怎么变成可训练对象”，先读这里。
- `envs/observation/default_builder.py`：默认观测构造器，最适合扩展新观测特征。
- `models/assembly.py`：模型组装总线，负责把 adapter、encoder、head 接起来。
- `forecast/artifacts.py`：LSTM 预测器产物的统一目录约定。
- `madrl/notebook_utils.py`：训练 notebook 和 compare notebook 共享的辅助函数。
- `runners/checkpoints.py`：checkpoint 保存、加载与 manifest 解析。
- `tests/support/helpers.py`：测试时生成 toy 数据和最小配置的辅助工厂。

## 使用方法

### 1. 环境准备

建议使用独立虚拟环境，然后安装：

```bash
pip install -r requirements.txt
```

如果你主要使用 notebook，还需要自行准备 Jupyter 环境。

### 2. 训练主线

最推荐的入口是 notebook：

- `notebooks/madrl/train_madrl.ipynb`

它适合做这些事情：

- 选择 `MADDPG` 或 `MATD3`
- 切换 `mlp / transformer / graph`
- 调整观测 profile、奖励函数和预测器
- 训练后直接评估并保存 checkpoint

命令行调试入口：

```bash
python scripts/run_debug_training.py
```

这个脚本只用于快速检查训练链路是否能跑通。

### 3. Forecast notebook

使用：

- `notebooks/forecast/forecast_lstm.ipynb`

作用：

- 读取价格序列
- 训练 LSTM 预测器
- 做 one-step rolling forecast
- 做更接近运行时使用方式的 block forecast
- 将模型、meta 和 scaler 保存到 `artifacts/forecast/lstm/`

### 4. Compare notebook

使用：

- `notebooks/madrl/compare_controllers.ipynb`

作用：

- 加载训练好的 MADRL checkpoint
- 与 `ZeroController` 对比
- 预留 `MPCController` 和 `ClassicDRLController` 的统一评估入口

默认 checkpoint 根目录：

- `artifacts/training/checkpoints/`

### 5. 数据整理 notebook

使用：

- `notebooks/data/prepare_data.ipynb`

作用：

- 读取原始 `opsd_building.csv`
- 生成项目训练和测试使用的 `train_prices.csv` 与 `test_prices.csv`

### 6. 测试

运行全部测试：

```bash
pytest
```

只跑某一类测试：

```bash
pytest tests/test_training_smoke.py
pytest tests/test_forecasters.py
pytest tests/test_model_assembly.py
```

测试运行期间产生的临时目录会被限制在 `tests/.tmp/` 下，不会污染仓库主目录。

## 扩展指南

下面给的是“从哪里改”的最短路径。

### 新增算法

1. 在 `algorithms/` 下新增一个实现文件。
2. 继承 `BaseAgent`，补齐动作选择、训练更新和保存加载接口。
3. 在 `algorithms/registry.py` 注册。
4. 在配置中设置 `cfg.algo.name`。

### 新增模型

1. 在 `models/encoders/` 新增 encoder。
2. 在 `models/family_adapters.py` 补 actor/critic 的 adapter。
3. 在 `models/registry.py` 注册 adapter 和 encoder。
4. 如有需要，复用已有 head；否则在 `models/heads/` 扩展 head。

### 新增环境

1. 在 `envs/` 下新增环境文件。
2. 让环境接受 dataset、reward_fn、forecaster、obs_builder。
3. 在 `envs/registry.py` 注册环境类型。
4. 如环境观测结构不同，再同步扩展 `envs/observation/`。

### 新增观测构造器

1. 若只是加特征，优先改 `envs/observation/features.py` 和 `feature_blocks.py`。
2. 若需要全新拼装逻辑，再新增 `ObservationBuilder` 子类。
3. 在 `envs/observation/registry.py` 注册。

### 新增预测器

1. 在 `forecast/` 下新增 forecaster 文件。
2. 实现 `forecast.base.Forecaster` 约定的接口。
3. 在 `forecast/registry.py` 注册。
4. 如需要持久化产物，复用 `forecast/artifacts.py` 的目录约定。

### 新增奖励函数

1. 在 `common/rewards/` 下新增奖励函数实现。
2. 继承 `RewardFn`。
3. 定义好 `component_meta`，保证评估和绘图可以复用。
4. 在 `common/rewards/__init__.py` 中接入选择逻辑。

### 新增测试

1. 所有测试文件都放在 `tests/` 下。
2. 通用 helper 放在 `tests/support/`。
3. 临时目录统一使用 `tmp_path` fixture。
4. 新增功能优先补：
   - 配置构建测试
   - registry 测试
   - smoke 训练测试
   - notebook helper 测试

## 命名规范与代码组织原则

本仓库统一采用以下约定：

- Python 文件和模块目录使用 `snake_case`
- 类名使用 `PascalCase`
- 抽象基类优先使用 `BaseXxx`
- 函数和变量使用 `snake_case`
- 常量使用 `UPPER_SNAKE_CASE`
- notebook 文件名要短、清楚、可直接反映用途

组织原则只有三条：

- 优先保留主链路可读性，不为了“未来也许会用到”做复杂抽象
- 配置、装配、实现、评估分层清楚
- 生成文件一律收进 `artifacts/`，测试辅助内容一律收进 `tests/`

## 建议的阅读顺序

如果你是第一次接触这个仓库，建议按下面顺序看：

1. `README.md`
2. `configs/experiment_config.py`
3. `configs/profiles.py`
4. `core/builder.py`
5. `envs/hems_env.py`
6. `models/assembly.py`
7. `algorithms/maddpg.py` 或 `algorithms/matd3.py`
8. `runners/train_runner.py`
9. `evaluation/evaluator.py`
10. `notebooks/madrl/train_madrl.ipynb`
