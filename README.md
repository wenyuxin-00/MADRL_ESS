# MADRL-ESS：多智能体深度强化学习储能调度系统

> 面向**入门学习者**的完整使用指南
> 适合有一定 Python 基础、此前主要使用 MATLAB 面条式代码的同学

---

## 目录

1. [项目是什么？解决什么问题？](#1-项目是什么)
2. [项目结构总览](#2-项目结构总览)
3. [安装与环境配置](#3-安装与环境配置)
4. [快速开始：跑第一个实验](#4-快速开始)
5. [Notebook 逐格解读](#5-notebook-逐格解读)
6. [核心概念详解](#6-核心概念详解)
7. [所有可调参数手册](#7-所有可调参数手册)
8. [调试指南](#8-调试指南)
9. [如何扩展：添加新奖励 / 新环境 / 新算法 / 新数据集](#9-如何扩展)
10. [MATLAB 用户对照表](#10-matlab-用户对照表)

---

## 1. 项目是什么？

### 1.1 问题背景

想象你家里有一块电池（储能系统），电价一天内会起伏变化——白天用电高峰时贵，夜晚低谷时便宜。
**最优策略**：低价时充电、高价时放电卖给电网，赚取价差（**价格套利**）。

本项目模拟一个小区，由 3 户家庭（智能体 Agent）分别管理各自的电池，通过强化学习**自动学会**套利策略。

```
                     ┌─────────────────────────────────────┐
                     │          电  网 (Grid)               │
                     │   电价随时间变化 (¥/kWh)             │
                     └──────┬──────────┬──────────┬────────┘
                            │          │          │
                购/售电      │          │          │
         ┌──────────────────┤          │          ├───────────┐
         │                  ▼          ▼          ▼           │
         │   Agent 0     Agent 1    Agent 2                   │
         │  (电池+负荷+光伏) (电池+负荷+光伏) (电池+负荷+光伏) │
         │  决策：±充电量  决策：±充电量  决策：±充电量        │
         └────────────────────────────────────────────────────┘
                       3 个智能体协同优化
```

### 1.2 技术路线

| 组件 | 使用技术 | 类比 MATLAB |
|------|----------|-------------|
| 仿真环境 | OpenAI Gym | Simulink 仿真模型 |
| 决策算法 | MADDPG / MATD3（深度强化学习） | `fmincon` 优化求解器 |
| 神经网络 | PyTorch（MLP / Transformer / 图神经网络） | 函数逼近 |
| 实验配置 | Python `dataclass` | MATLAB 结构体 `cfg.field = value` |
| 可视化 | Matplotlib + TensorBoard | MATLAB plot |

### 1.3 训练完成后能做什么

- **输出**：训练好的策略网络（`.pth` 权重文件）
- **评估**：在历史电价数据上测试套利收益
- **分析**：可视化每一步的充放电决策、荷电状态（SoC）曲线、奖励分解

---

## 2. 项目结构总览

```
MADRL_ESS/
│
├── 📓 notebooks/madrl/
│   └── train_madrl.ipynb      ← 【主入口】从这里开始
│
├── ⚙️  configs/
│   ├── experiment_config.py   ← 所有参数定义（数据类型、默认值）
│   └── profiles.py            ← 预设配置组合（debug / base / fast_train）
│
├── 🌍 envs/
│   ├── hems_env.py            ← 电池环境（step / reset 核心逻辑）
│   └── observation/           ← 观测向量的组装方式
│
├── 🤖 algorithms/
│   ├── base_agent.py          ← 所有算法的公共接口（抽象基类 ABC）
│   ├── maddpg.py              ← MADDPG 算法实现
│   ├── matd3.py               ← MATD3 算法实现
│   └── registry.py            ← 注册表（按名称查找算法类）
│
├── 🧠 models/
│   ├── actors/                ← Actor 网络头（输出动作）
│   ├── critics/               ← Critic 网络头（估计价值）
│   └── encoders/              ← 特征编码器（MLP / Transformer / 图网络）
│
├── 🏆 common/rewards/
│   ├── composite.py           ← 多分量复合奖励（默认）
│   ├── sparse.py              ← 稀疏套利奖励
│   └── base.py                ← 奖励函数抽象基类
│
├── 📊 datasets/
│   ├── csv_price_load.py      ← 读取电价 + 负荷 CSV
│   ├── csv_prosumer.py        ← 读取含光伏的 SimBench 数据
│   └── registry.py            ← 数据集注册表
│
├── 🔮 forecast/
│   ├── oracle.py              ← 完美预测（训练阶段使用）
│   └── lstm_forecaster.py     ← LSTM 预测器（可选，需单独训练）
│
├── 🏃 runners/
│   └── train_runner.py        ← 训练主循环（管理 rollout + 更新）
│
├── 🔧 core/
│   └── builder.py             ← 一键构建所有组件的工厂函数
│
├── 📈 evaluation/
│   ├── episode_recorder.py    ← 记录每步历史数据
│   └── plots.py               ← 可视化工具函数
│
├── 🧪 tests/                  ← 自动化测试（pytest）
├── 📁 data/                   ← CSV 数据文件（电价、负荷、光伏）
└── 📁 artifacts/              ← 训练产物（模型权重、TensorBoard 日志）
```

> **关键设计原则**：每种可替换的组件（算法、环境、奖励函数、数据集）都有一个**注册表**。
> 添加新组件 = 写一个类 + 在注册表里加一行，其余代码无需改动。

---

## 3. 安装与环境配置

### 3.1 前置条件

- Python 3.10 及以上
- 推荐 CUDA 12.8（也支持仅 CPU 运行）
- Jupyter Lab 或 VS Code + Jupyter 插件

### 3.2 安装依赖

```bash
# 克隆项目
git clone <仓库地址>
cd MADRL_ESS

# 一键安装全部依赖（包含 PyTorch CUDA 版）
pip install -r requirements.txt
```

> **注意**：如果你的 CUDA 版本不同（如 CUDA 11.8），需要单独安装匹配的 PyTorch：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```
> 安装完成后再运行 `pip install -r requirements.txt` 跳过 torch 行即可。

### 3.3 验证安装

```bash
# 运行最小冒烟测试（约 5 秒，不需要 GPU）
python -m pytest tests/test_encoding_hygiene.py -v
```

看到 `4 passed` 说明基础环境正常。

---

## 4. 快速开始

### 4.1 打开主 Notebook

```bash
cd MADRL_ESS
jupyter lab notebooks/madrl/train_madrl.ipynb
```

### 4.2 第一次运行只需改 3 个参数

打开 Notebook 后，在 **Cell 3** 找到如下代码，按照注释修改：

```python
profile = "debug"      # ← 先用 debug，速度快（约 5-20 秒完成）
algorithm = "MADDPG"   # ← 不用改
seed = 0               # ← 不用改
```

然后在菜单选择 **Run → Run All Cells**（或按 Shift+Enter 依次运行每个格）。

训练完成后你会看到：

1. 控制台输出：`训练完成 episode 数 = 8`
2. 奖励曲线图（各分量随时间的变化）
3. 电价 / 充放电动作 / SoC 对比图
4. 模型保存路径：`artifacts/training/checkpoints/`

---

## 5. Notebook 逐格解读

### Cell 1：定位项目根目录

```python
project_root = Path.cwd().resolve()
while not (project_root / "configs").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
```

**作用**：让 Python 能找到 `configs/`、`algorithms/` 等所有模块。
类比 MATLAB 的 `addpath(genpath('.'))`。**不需要修改这格**。

---

### Cell 2：导入工具函数

```python
from common.experiment_notebook_utils import (
    build_runner,      # 构建训练器（环境 + 算法 + 缓冲区）
    evaluate_runner,   # 评估训练好的策略
    inspect_runner_io, # 检查输入输出维度（调试时很有用）
    summarize_cfg,     # 打印配置摘要
)
from configs import compose_experiment_config  # 组合配置的唯一入口
```

**不需要修改这格**。

---

### Cell 3：实验参数（**你主要在这里修改**）

```python
# ---------- 算法 ----------
profile       = "debug"      # "debug"=8个episode快速调试 | "base"=标准 | "fast_train"=正式训练
algorithm     = "MADDPG"     # "MADDPG" | "MATD3"
model_family  = "mlp"        # "mlp" | "transformer" | "graph"

# ---------- 奖励设计 ----------
reward_type   = "composite"  # "composite"=多分量奖励 | "sparse"=稀疏套利奖励

# ---------- 序列预测来源 ----------
sequence_source = "truth"    # "truth"=完美预知（oracle）| "lstm"=LSTM预测（需单独训练）
forecast_type   = "perfect" if sequence_source == "truth" else "lstm"

# ---------- 观测配置 ----------
observation_profile = "simbench"  # "simbench"=含光伏 | "default"=不含光伏 | "minimal"=最简

# ---------- 其他 ----------
seed               = 0       # 随机种子（用于复现实验）
n_eval_episodes    = 2       # 训练后额外评估的 episode 数
reward_plot_window = 20      # 奖励滑动平均窗口
```

---

### Cell 4：组合完整配置

```python
cfg = compose_experiment_config(
    profile=profile,
    algorithm=algorithm,
    model_family=model_family,
    reward_type=reward_type,
    observation_profile=observation_profile,
    forecast_type=forecast_type,
    ...
)

# --- 任务专属覆盖（直接赋值覆盖 profile 默认值）---
cfg.data.dataset_type       = "csv_prosumer"    # 使用含光伏的 SimBench 数据集
cfg.train.num_envs          = 4                 # 4 个并行环境
cfg.train.train_episodes    = 300               # 训练 300 个 episode
cfg.env.num_agents          = 3                 # 3 个电池智能体
cfg.env.episode_limit       = 96 * 2            # 每个 episode = 2 天 = 192 步
cfg.env.future_horizon      = 24                # 6 小时预测窗口（24 步 × 15 分钟）
cfg.obs.local_features      = ["time", "price", "load", "pv", "soc"]
cfg.obs.sequence_features   = ["price", "load", "pv"]
cfg.forecast.target_signals = ["price", "load", "pv"]

summary = summarize_cfg(cfg)
```

运行后 `summary` 字典会告诉你训练总步数、并行迭代次数、使用的设备等关键信息。

---

### Cell 5：构建训练器

```python
runner = build_runner(cfg, seed=seed, env_name="NotebookTrain", number=1)
```

这一步在后台完成：

1. 创建 4 个并行仿真环境
2. 为每个 Agent 创建 Actor 和 Critic 神经网络
3. 初始化经验回放缓冲区（最多存 100 万条转换）
4. 创建 TensorBoard 日志记录器

成功后控制台打印：

```
Observation schema = {'local': (3, 6), 'adjacency': (3, 3), 'price_seq': (25,), ...}
Action dim = 1
```

---

### Cell 6：IO 健全性检查（**调试时必看**）

```python
sanity = inspect_runner_io(runner, cfg)
```

**输出含义解读**：

```python
{
    # 单个环境的观测形状（无批次维度）
    'observation_schema': {
        'local':     (3, 6),   # 3 个 Agent，每个 6 个局部特征
        'adjacency': (3, 3),   # 3×3 邻接矩阵
        'price_seq': (25,),    # 共享的 25 步电价序列（24+1）
        'load_seq':  (3, 25),  # 每个 Agent 的 25 步负荷序列
        'pv_seq':    (3, 25),  # 每个 Agent 的 25 步光伏序列
    },
    # 批次维度 = num_envs（4 个并行环境）
    'action_batch_shape': (4, 3, 1),  # 4 环境 × 3 Agent × 1 维动作
    'reward_shape':       (4, 3, 1),  # 同上
    'info_keys': ['e_bat', 'soc_t', 'price', 'r_inc', 'r_pen', ...],  # 可调试的中间量
}
```

如果维度不对，在这里就能发现，不必等训练崩溃。

---

### Cell 7：训练主循环

```python
episodes_completed = runner.run()
print("训练完成 episode 数 =", episodes_completed)
runner.perf_summary  # 查看性能报告
```

训练完后 `perf_summary` 提供详细耗时分析：

```python
{
    'total_wall_time_s':      17.3,   # 总耗时（秒）
    'steps_per_sec':          88.7,   # 每秒处理的环境步数
    'avg_update_ms_per_call': 10.9,   # 每次神经网络更新耗时（毫秒）
}
```

---

### Cell 8：评估与可视化

```python
eval_results = evaluate_runner(runner, cfg, n_episodes=2, deterministic=True)
print("平均评估奖励 =", eval_results["mean_episode_reward"])
```

这一格会输出 4 张图：

| 图序 | 内容 |
|------|------|
| 图 1 | 训练期间各奖励分量（r_inc / r_pen / r_pbrs 等）随 episode 的曲线 |
| 图 2 | 训练最近几个 episode 的电价 / 充放电动作 / SoC 轨迹 |
| 图 3 | 评估期间各奖励分量曲线 |
| 图 4 | 评估期间的电价 / 充放电动作 / SoC 轨迹 |

> **如何判断训练是否有效**：查看图 1 中 `r_pen`（惩罚项）是否在下降，`r_inc`（套利收益）是否在上升。

---

### Cell 9：保存模型

```python
runner.save_model(str(save_dir), episode=episodes_completed)
runner.close()  # ← 务必调用！释放环境资源和文件句柄
```

模型保存至 `artifacts/training/checkpoints/MADDPG/`：

```
MADDPG/
├── actor_agent_0_ep_8.pth     # Agent 0 的 Actor 网络权重
├── actor_agent_1_ep_8.pth
├── actor_agent_2_ep_8.pth
├── critic_agent_0_ep_8.pth    # Agent 0 的 Critic 网络权重
├── ...
└── latest_checkpoint.json     # 元信息（episode 数、步数、保存路径等）
```

---

## 6. 核心概念详解

### 6.1 观测（Observation）—— 智能体能看到什么

每步，每个智能体收到一个字典结构的观测：

```python
obs = {
    "local":     shape (3, 6),   # 每个 Agent 的 6 个局部特征（实时量）
    "adjacency": shape (3, 3),   # 邻接矩阵（描述 Agent 间关系）
    "price_seq": shape (25,),    # 共享的 25 步电价时间窗口（过去 + 预测未来）
    "load_seq":  shape (3, 25),  # 每个 Agent 各自的 25 步负荷时间窗口
    "pv_seq":    shape (3, 25),  # 每个 Agent 各自的 25 步光伏时间窗口
}
```

**局部特征 `local` 的 6 个分量**（使用 `simbench` 观测配置时）：

| 索引 | 特征名 | 含义 |
|------|--------|------|
| 0 | `time` | 当前时刻的周期性编码（0~1，一天一周期） |
| 1 | `price` | 当前电价（归一化至合理范围） |
| 2 | `load` | 当前负荷（归一化） |
| 3 | `pv` | 当前光伏发电量（归一化） |
| 4 | `soc` | 电池当前荷电状态（State of Charge，0~1） |
| 5 | （扩展槽位） | 可添加更多自定义特征 |

**时间窗口长度 25 = `future_horizon + 1` = 24 + 1**：

- 使用"完美预知"（`forecast_type="perfect"`）时，窗口内是真实的未来值
- 使用 LSTM 预测器时，窗口内是模型预测值

> **类比**：`local` ≈ Simulink 中的局部传感器输出；`price_seq` ≈ 提前获知的未来电价预报表。

---

### 6.2 动作（Action）—— 智能体能做什么

每个智能体输出 **1 维连续动作**：

```
action ∈ [-1.0, +1.0]
```

| 动作值 | 含义 |
|--------|------|
| `+1.0` | 以最大功率充电（从电网购电存入电池） |
| `-1.0` | 以最大功率放电（从电池向电网售电） |
| `0.0` | 待机（电池不动） |

实际充放电量（kWh）= `action × max_charge_rate × dt`
= `action × 0.125 kW × 0.25 h` = `action × 0.03125 kWh`

> **类比**：动作 = 油门开度（-1 全制动，+1 全力加速）。

---

### 6.3 奖励（Reward）—— 智能体在优化什么

默认的**复合奖励（Composite Reward）** 由 5 个分量叠加：

```
r_total = r_inc - r_pen + r_pbrs - r_soc + r_bonus
```

| 分量 | 颜色标识 | 计算公式 | 作用 |
|------|----------|----------|------|
| `r_inc` | 绿色，符号 + | `-(实际电费 - 基准电费)` | 套利收益：放电替代购电 |
| `r_pen` | 橙色，符号 − | `w_pen × \|请求-实际\| / p_max` | 惩罚越限/不可行动作 |
| `r_pbrs` | 蓝色，符号 + | `γ×φ(t+1) - φ(t)`，φ=均价×电量 | 鼓励在价格上涨前充电 |
| `r_soc` | 紫色，符号 − | `w_soc × (SoC - 目标SoC)²` | 保持电池在安全区间 |
| `r_bonus` | 青色，符号 + | `λ × \|充放电量\| × dt` | 鼓励积极使用电池 |

> **重要理解**：
> - 奖励值通常是**负数**（因为电费是主要成本）。
> - 训练目标是让奖励**尽可能不那么负**，即减少成本、增加套利收益。
> - 图中曲线**上升（向零靠近）= 策略在改进**。

---

### 6.4 训练流程简述

```
每次迭代（interaction step）：

  1. 读取当前观测 obs（来自 4 个并行环境）
  2. Actor 网络计算动作 action，叠加探索噪声
  3. 4 个环境同时执行 env.step(action)
  4. 获得 next_obs, reward, done
  5. 存入经验回放缓冲区 ReplayBuffer
  6. 每隔 update_interval 步，从缓冲区随机采样：
       - 计算 TD 误差 → 更新 Critic（评估网络）
       - 用 Critic 梯度更新 Actor（策略网络）
       - 软更新目标网络（τ = 0.01）
  7. 逐步衰减探索噪声（noise_std: 0.4 → 0.2）
```

---

## 7. 所有可调参数手册

### 7.1 环境参数（`cfg.env`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_agents` | 3 | 智能体（电池）数量 |
| `episode_limit` | 192 | 每轮 episode 的步数（192 = 2 天 × 96 步/天） |
| `battery_capacity` | 1.0 | 电池容量（kWh） |
| `max_charge_rate` | 0.125 | 最大充/放电功率（kW） |
| `efficiency` | 0.95 | 充放电效率（双程） |
| `init_soc` | 0.5 | 初始荷电状态（50%） |
| `soc_min` | 0.05 | 最低安全 SoC（5%） |
| `soc_max` | 0.95 | 最高安全 SoC（95%） |
| `soc_target` | 0.5 | SoC 正则化目标（50%） |
| `dt` | 0.25 | 时间步长（小时，0.25 h = 15 min） |
| `future_horizon` | 24 | 序列观测的预测步数（24 步 = 6 小时） |

### 7.2 训练参数（`cfg.train`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `train_episodes` | 1000 | 训练的总 episode 数 |
| `num_envs` | 32 | 并行环境数（越多越快，受内存限制） |
| `batch_size` | 4096 | 每次从缓冲区抽样的经验条数 |
| `buffer_size` | 1,000,000 | 经验回放缓冲区最大容量 |
| `actor_lr` | 1e-4 | Actor 网络学习率 |
| `critic_lr` | 1e-4 | Critic 网络学习率 |
| `noise_std_init` | 0.4 | 探索噪声初始标准差 |
| `noise_std_min` | 0.2 | 探索噪声最小值 |
| `vec_env_type` | "dummy" | 向量环境类型（"dummy"=单进程，"subproc"=多进程） |

### 7.3 算法参数（`cfg.algo`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `name` | "MADDPG" | 算法名称（"MADDPG" 或 "MATD3"） |
| `gamma` | 0.999 | 折扣因子（越接近 1 = 越重视长期收益） |
| `tau` | 0.01 | 目标网络软更新率（值越小 = 目标网络更新越慢、越稳定） |
| `policy_update_freq` | 2 | MATD3 专用：每 N 次 Critic 更新才更新一次 Actor |

### 7.4 预置配置 profile 速查

| profile | 适用场景 | episode 数 | 环境数 | batch_size |
|---------|---------|------------|--------|------------|
| `"debug"` | 快速验证代码能跑通 | 8 | 1 | 128 |
| `"base"` | 标准配置（未指定时的默认） | 1000 | 32 | 4096 |
| `"fast_train"` | 正式训练，充分利用多核 | 300 | 4-12 | 较大 |

---

## 8. 调试指南

### 8.1 常见报错与解决方法

#### ❌ `RuntimeError: CUDA out of memory`

显存不足，减小批次大小或并行环境数：

```python
cfg.train.batch_size = 512
cfg.train.num_envs   = 2
```

#### ❌ `KeyError: 'pv'` 或 `ValueError: shape mismatch`

原因：选择了 `observation_profile="simbench"`（含光伏），但数据集没有光伏列。

```python
# 解决方案 A：改为不含光伏的配置
cfg.obs.local_features      = ["time", "price", "load", "soc"]
cfg.obs.sequence_features   = ["price", "load"]
cfg.forecast.target_signals = ["price", "load"]

# 解决方案 B：使用正确的含光伏数据集
cfg.data.dataset_type = "csv_prosumer"
```

#### ❌ `ModuleNotFoundError`

```bash
# 确保在项目根目录下启动 Jupyter
cd D:/GithubProject/MADRL_ESS
jupyter lab
```

#### ❌ 训练奖励长时间不上升

这在前期很正常。debug 模式只训练 8 个 episode，远不够收敛。
至少需要 200-500 个 episode 才能看到明显改善：

```python
profile = "base"
cfg.train.train_episodes = 500
```

---

### 8.2 常用调试工具

#### 打印完整配置摘要

```python
from common.experiment_notebook_utils import summarize_cfg
print(summarize_cfg(cfg))
```

#### 检查观测 / 动作维度（Cell 6）

```python
sanity = inspect_runner_io(runner, cfg)
print("观测形状:", sanity['observation_schema'])
print("动作形状:", sanity['action_batch_shape'])   # (num_envs, num_agents, 1)
print("可调试的中间量:", sanity['info_keys'])
```

#### 手动执行一步环境，检查输出

```python
obs = runner.env.reset()
print("obs['local'] 形状:", obs["local"].shape)   # 期望 (num_envs, 3, 6)

actions = runner.format_env_actions(runner.select_action_batch(obs))
next_obs, reward, done, info = runner.env.step(actions)
print("reward 形状:", reward.shape)               # 期望 (num_envs, 3, 1)
print("info[0] 包含的键:", list(info[0].keys()))
```

#### 查看 TensorBoard 训练曲线

```bash
# 在项目根目录运行（需单独开终端）
tensorboard --logdir artifacts/training/tensorboard
# 然后在浏览器访问 http://localhost:6006
```

---

### 8.3 维度速查表

并行向量环境中，所有张量都带有 `num_envs` 批次维度：

| 张量 | 形状（num_envs=4） | 说明 |
|------|-------------------|------|
| `obs["local"]` | `(4, 3, 6)` | 4 个环境 × 3 个 Agent × 6 个局部特征 |
| `obs["price_seq"]` | `(4, 25)` | 4 个环境 × 25 步电价序列（共享） |
| `obs["load_seq"]` | `(4, 3, 25)` | 4 个环境 × 3 个 Agent × 25 步负荷序列 |
| `action_batch` | `(4, 3, 1)` | 4 个环境 × 3 个 Agent × 1 维动作 |
| `reward` | `(4, 3, 1)` | 同上 |

---

## 9. 如何扩展

### 9.1 注册表机制说明

项目所有可替换组件都通过**注册表字典**管理：

```python
# 原理示例（algorithms/registry.py）
AGENT_REGISTRY = {
    "MADDPG": MADDPG,
    "MATD3":  MATD3,
    # 添加新算法：直接在这里加一行
}
```

添加新组件只需三步：**① 写类 → ② 注册 → ③ 用字符串引用**。

---

### 9.2 添加新的奖励函数

**第一步**：新建 `common/rewards/my_reward.py`

```python
import numpy as np
from common.rewards.base import RewardFn, ComponentMeta


class MyReward(RewardFn):
    """我的自定义奖励函数。"""

    def __init__(self, cfg: object) -> None:
        # 从配置中读取你需要的参数
        self.w_my = float(cfg.reward.w_pen)  # 可复用已有配置字段

    @property
    def component_meta(self) -> list[ComponentMeta]:
        """声明奖励分量（驱动可视化）。"""
        return [
            # ComponentMeta(键名, 图例标签, 颜色, 符号)
            # 符号 +1 表示该分量在 total 中是加号，-1 是减号
            ComponentMeta("r_my", "+ r_my（我的分量）", "blue", +1),
        ]

    def compute(self, env_state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        参数说明（env_state 字典的常用键）：
            e_bat      (num_agents,)  实际充放电量（kWh），正=充电，负=放电
            price_t    float          当前电价（元/kWh）
            net_load_t (num_agents,)  净负荷（kW）= 负荷 - 光伏
            soc_t      (num_agents,)  当前荷电状态（0~1）
            dt         float          时间步长（小时）
        返回：
            (total_reward, components_dict)
            total_reward: shape (num_agents,), dtype float32
        """
        e_bat   = np.asarray(env_state["e_bat"],   dtype=np.float32)
        price_t = float(env_state["price_t"])
        dt      = float(env_state.get("dt", 0.25))

        # 示例：充放电量越大，奖励越高
        r_my = (np.abs(e_bat) * price_t * dt).astype(np.float32)

        return r_my, {"r_my": r_my}
```

**第二步**：在 `common/rewards/__init__.py` 里注册

```python
# 找到已有的注册行，仿照格式添加：
from common.rewards.my_reward import MyReward
register_reward("my_reward", MyReward)
```

**第三步**：在 Notebook 里使用

```python
reward_type = "my_reward"
# 重新运行 Cell 4 → Cell 5 → Cell 7 即可
```

---

### 9.3 添加新的环境

**第一步**：新建 `envs/my_env.py`

```python
import gym
import numpy as np
from gym import spaces


class MyEnv(gym.Env):
    """我的自定义仿真环境。"""

    # 框架要求：构造函数签名固定如下
    def __init__(self, cfg, mode="train", dataset=None,
                 reward_fn=None, forecaster=None, obs_builder=None):
        self.cfg        = cfg
        self.num_agents = cfg.env.num_agents
        self.mode       = mode

        # 必须定义：动作空间
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        self._step = 0

    def reset(self, episode_idx=None):
        """重置环境，返回初始观测字典。"""
        self._step = 0
        return {
            "local": np.zeros(
                (self.num_agents, len(self.cfg.obs.local_features)),
                dtype=np.float32
            ),
            # 如果需要序列特征，也在这里加入：
            # "price_seq": np.zeros((self.cfg.env.future_horizon + 1,), dtype=np.float32),
        }

    def step(self, actions):
        """
        执行一步仿真。
        参数：
            actions: list[np.ndarray]，每个 Agent 一个动作数组
        返回：
            obs       : 下一步观测字典
            rewards   : np.ndarray, shape (num_agents, 1)
            done_list : list[dict]，每个 Agent 一个，含 "episode_done" 键
            info_list : list[dict]，每个 Agent 的调试信息
        """
        self._step += 1
        episode_done = (self._step >= self.cfg.env.episode_limit)

        obs     = self.reset() if episode_done else self._build_obs()
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
        done_list = [{"episode_done": episode_done}] * self.num_agents
        info_list = [{"episode_done": episode_done, "reward": 0.0}
                     for _ in range(self.num_agents)]

        return obs, rewards, done_list, info_list

    def _build_obs(self):
        """构建当前步的观测。"""
        return {
            "local": np.zeros(
                (self.num_agents, len(self.cfg.obs.local_features)),
                dtype=np.float32
            ),
        }

    def close(self):
        pass
```

**第二步**：在 `envs/registry.py` 里注册

```python
from envs.my_env import MyEnv
register_env("my_env", MyEnv)
```

**第三步**：在 Notebook 里使用

```python
cfg.env.env_type = "my_env"
```

---

### 9.4 添加新的强化学习算法

> **前提知识**：理解 Actor-Critic 框架（Actor 负责决策，Critic 评估价值）。

**第一步**：新建 `algorithms/my_algo.py`

```python
import copy
import torch
import torch.nn.functional as F
from algorithms.base_agent import BaseAgent
from common.replay_buffer import to_torch_batch
from models import build_actor_network, build_critic_network


class MyAlgo(BaseAgent):
    """
    我的自定义 RL 算法。
    继承 BaseAgent 即可自动获得：
        choose_action()  ← 含噪声的动作采样
        save_model()     ← 保存 actor/critic 权重
        load_model()     ← 加载权重并同步目标网络
        _soft_update()   ← 软更新目标网络
    """

    def __init__(self, cfg: object, agent_id: int) -> None:
        self.cfg        = cfg
        self.device     = cfg.runtime.device
        self.agent_id   = int(agent_id)
        self.num_agents = int(cfg.env.num_agents)
        self.gamma      = float(cfg.algo.gamma)
        self.tau        = float(cfg.algo.tau)
        self.max_action = float(cfg.model.max_action)
        self.use_grad_clip = bool(cfg.model.use_grad_clip)

        # 使用项目内置的网络构建函数
        self.actor         = build_actor_network(cfg, self.agent_id).to(self.device)
        self.critic        = build_critic_network(cfg).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = torch.optim.Adam(
            self.actor.parameters(),  lr=float(cfg.train.actor_lr)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg.train.critic_lr)
        )

    def act_from_torch_obs(self, obs_t: dict, noise_std: float) -> torch.Tensor:
        """从 torch 张量观测计算动作（被继承的 choose_action 调用）。"""
        action = self.actor(obs_t)
        if noise_std > 0.0:
            action = action + torch.randn_like(action) * noise_std
        return action.clamp(-self.max_action, self.max_action)

    def train(self, replay_buffer: object, agent_n: list) -> None:
        """从缓冲区采样并训练。"""
        batch = to_torch_batch(replay_buffer.sample(), self.device)
        self.train_on_batch(batch, agent_n)

    def train_on_batch(self, batch: dict, agent_n: list) -> None:
        """在一个 batch 上更新 Actor 和 Critic 网络。"""
        obs      = batch["obs"]
        action   = batch["action"]
        reward   = batch["reward"]
        next_obs = batch["next_obs"]
        done     = batch["done"]

        # ─── 步骤 1：更新 Critic ─────────────────────────────────────
        with torch.no_grad():
            next_action = torch.stack(
                [a.actor_target(next_obs) for a in agent_n], dim=1
            )
            target_q = (
                reward[:, self.agent_id]
                + self.gamma * (1 - done[:, self.agent_id])
                * self.critic_target(next_obs, next_action)
            )
        current_q = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # ─── 步骤 2：更新 Actor ─────────────────────────────────────
        new_action = action.clone()
        new_action[:, self.agent_id] = self.actor(obs)
        actor_loss = -self.critic(obs, new_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # ─── 步骤 3：软更新目标网络（从 BaseAgent 继承）──────────────
        self._soft_update()
```

**第二步**：在 `algorithms/registry.py` 里注册

```python
from algorithms.my_algo import MyAlgo
register_agent("MyAlgo", MyAlgo)
```

**第三步**：在 Notebook 里使用

```python
algorithm = "MyAlgo"
```

---

### 9.5 添加新的数据集

**第一步**：准备数据

CSV 文件格式要求（以每行一个时刻为单位）：

```
timestamp,electricity_price,load_0,load_1,load_2,pv_0,pv_1,pv_2
2016-01-01 00:00,0.12,1.5,1.2,1.8,0.0,0.0,0.0
2016-01-01 00:15,0.11,1.4,1.1,1.7,0.0,0.0,0.0
...
```

**第二步**：新建 `datasets/my_dataset.py`

```python
import numpy as np
import pandas as pd
from datasets.base import BaseEpisodeDataset


class MyDataset(BaseEpisodeDataset):
    """从自定义 CSV 文件加载数据。"""

    def __init__(self, cfg, mode: str = "train") -> None:
        data_path = cfg.data.data_dir / "my_data.csv"
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        self.episode_length = cfg.env.episode_limit
        self.n_agents       = cfg.env.num_agents

        # 按训练/测试拆分（前 70% 用于训练，后 30% 用于测试）
        split = int(len(df) * 0.7)
        self.df = df.iloc[:split] if mode == "train" else df.iloc[split:]

        self._n_episodes = len(self.df) // self.episode_length

    def num_episodes(self) -> int:
        return self._n_episodes

    def get_episode(self, episode_idx: int) -> dict:
        """
        返回单个 episode 的信号字典。
        键名必须与 cfg.obs.local_features / cfg.obs.sequence_features 中的名称对应。
        """
        start = episode_idx * self.episode_length
        end   = start + self.episode_length
        chunk = self.df.iloc[start:end]

        return {
            "signals": {
                # 电价：shape (T,)
                "price": chunk["electricity_price"].to_numpy(dtype=np.float32),
                # 负荷：shape (T, num_agents)
                "load":  chunk[["load_0", "load_1", "load_2"]].to_numpy(dtype=np.float32),
                # 光伏（可选）：shape (T, num_agents)
                "pv":    chunk[["pv_0", "pv_1", "pv_2"]].to_numpy(dtype=np.float32),
            },
            "meta": {"episode_idx": episode_idx, "source": "my_csv"}
        }
```

**第三步**：在 `datasets/registry.py` 里注册

```python
from datasets.my_dataset import MyDataset
register_dataset("my_dataset", MyDataset)
```

**第四步**：在 Notebook 里使用

```python
cfg.data.dataset_type = "my_dataset"
```

---

## 10. MATLAB 用户对照表

### 10.1 概念映射

| MATLAB 概念 | Python 等价物 | 项目中的具体位置 |
|-------------|---------------|-----------------|
| 结构体 `cfg.field = val` | `dataclass` 字段赋值 | `configs/experiment_config.py` |
| `sim('model.slx')` | `runner.run()` | `runners/train_runner.py` |
| S-Function（自定义模块） | `gym.Env` 子类 | `envs/hems_env.py` |
| Simulink 仿真单步 | `env.step(action)` | 同上 |
| 优化求解 `fmincon` | RL 训练（梯度下降） | `algorithms/` |
| `.mat` 文件保存 | `.pth` 文件保存 | `artifacts/training/checkpoints/` |
| `save('f.mat', 'x')` | `runner.save_model(dir, ep)` | `runners/train_runner.py` |
| `load('f.mat')` | `runner.load_model(dir, ep)` | 同上 |
| `for ep = 1:N` 训练循环 | `runner.run()` 内部的 while 循环 | 同上 |
| `figure; plot(t, y)` | `plot_reward_decomposition(...)` | `evaluation/reward_plots.py` |
| `addpath(genpath('.'))` | `sys.path.insert(0, root)` | Notebook Cell 1 |
| 函数句柄 `@myFunc` | Python 函数 / lambda | 随处可见 |
| `cellfun(@f, C)` | 列表推导 `[f(x) for x in C]` | 随处可见 |

### 10.2 常见语法差异

```python
# ─── MATLAB 风格（不要这样写）─────────────────────────────────
# cfg.train.num_envs = 32;          % 末尾分号
# obs(1, :)                          % 从 1 开始的索引
# A = [1 2; 3 4] * B                 % 矩阵乘法

# ─── Python 等价（正确写法）──────────────────────────────────
cfg.train.num_envs = 32            # Python 不用分号
obs[0, :]                           # Python 从 0 开始索引
A = np.array([[1, 2], [3, 4]]) @ B # @ 表示矩阵乘法，* 是逐元素乘
```

```python
# MATLAB 工作区：变量随处可访问
# result = run_simulation();
# plot(result.x, result.y);

# Python：变量存在于当前作用域，通过返回值传递
eval_results = evaluate_runner(runner, cfg, n_episodes=2)
print(eval_results["mean_episode_reward"])
```

---

## 附录：常用命令速查

```bash
# 运行自动化测试（验证代码完整性）
python -m pytest tests/ -v

# 查看 TensorBoard 训练曲线
tensorboard --logdir artifacts/training/tensorboard
# 然后访问 http://localhost:6006

# 快速冒烟测试（不需要 Jupyter）
python -m pytest tests/test_encoding_hygiene.py -v
```

```python
# 加载已有模型继续训练或评估
runner.load_model(
    model_dir="artifacts/training/checkpoints",
    episode=300   # 对应 save_model 时传入的 episode 数
)
```

---

> **遇到问题时，优先看 Cell 6 的 `inspect_runner_io()` 输出。**
> 90% 的维度不匹配问题可以在那里发现，无需等到训练崩溃。
