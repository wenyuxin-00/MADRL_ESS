# 环境配置与环境系统详细分析

## 📋 第一部分：配置参数详解 (cfg.py)

配置是项目的单一真实源。所有参数按 **owner** 分组，每个配置类对应一个逻辑域。

### 🌐 GridCfg - 电网拓扑与约束

```python
@dataclass
class GridCfg:
    sb_code: str = "1-LV-rural1--0-sw"          # SimBench拓扑编码
    pf_solver: str = "nr"                      # Power flow求解器 (Newton-Raphson)
    agent_bus_ids: tuple[int, ...] = (12, 4, 2)  # 三个agent连接的母线ID
    v_min_pu: float = 0.95                     # 电压下限 (单位值)
    v_max_pu: float = 1.05                     # 电压上限 (单位值)
    trafo_limit_kw: float | None = None        # 变压器功率限额
    line_limit_kw: float = 80.0                # 线路功率限额
    line_max_loading_pct: float = 100.0        # 线路最大负荷百分比
```

**含义强调**：
- `sb_code`: SimBench是真实配电网的标准化数据库。这个编码唯一标识了一个网络拓扑
- `agent_bus_ids`: **核心** - 只有这三个母线上的注入功率会被环境计算。其他所有负荷/发电来自外部模型
- `v_min_pu` / `v_max_pu`: 超出范围 → 违反电压约束 → 奖励惩罚
- `pf_solver="nr"`: Newton-Raphson求解器，快速且精确

---

### 📊 DataCfg - 数据集与时间窗

```python
@dataclass
class DataCfg:
    data_root: str = "datasets"
    agent_profiles: tuple[str, ...] = ("SFH12", "SFH18", "SFH20")  # 三个家庭原型
    share_data_train_test: tuple[int, int] = (2019, 2020)  # 训练/测试年份
    train_start_date: str = "2019-01-01"
    train_end_date: str = "2019-12-31"
    eval_start_date: str = "2020-04-01"
    eval_end_date: str = "2020-04-15"
    load_components: tuple[str, ...] = ("household", "heatpump")  # 两个负荷源
    pv_reference: str = "south"                # PV朝向
    load_scale: tuple[float, ...] = (20.0, 20.0, 20.0)    # 负荷放大系数
    pv_scale: tuple[float, ...] = (1.0, 1.0, 1.0)         # PV放大系数
    pv_capacity_kw: tuple[float, ...] = ()    # 若指定，覆盖真实PV容量
```

**含义强调**：
- `share_data_train_test=(2019, 2020)`: 数据来自两个年份，2019训练，2020评估
- `train_start_date~train_end_date`: **整年** 2019年用于训练
- `eval_start_date~eval_end_date`: 只用2020年4月1-15日（15天）做评估
- `load_components=("household", "heatpump")`: 每个家庭有两个并联的负荷来源，总负荷=household+heatpump
- `load_scale=(20.0, 20.0, 20.0)`: 三个agent的负荷分别放大20倍。为了提高问题规模

---

### ⚙️ EnvCfg - 环境核心参数

```python
@dataclass
class EnvCfg:
    num_agents: int = 3                        # Agent数量
    episode_steps: int = 96                    # 每天时间步数 (15分钟采样 × 96 = 24小时)
    train_window_days: int = 7                 # 训练时单个episode覆盖7天
    dt_hours: float = 0.25                     # 时间步长 (15分钟)
    soc_min: float = 0.05                      # 电池最小SOC (5%)
    soc_max: float = 0.95                      # 电池最大SOC (95%)
    init_soc: float = 0.05                     # 初始SOC
    train_init_soc_low: float = 0.05           # 训练时SOC初始范围下界
    train_init_soc_high: float = 0.05          # 训练时SOC初始范围上界
    efficiency: float = 0.95                   # 充放电效率 (往返95%)
    battery_capacity_kwh: tuple[float, ...] = (100.0, 100.0, 100.0)  # 三个电池容量
    max_charge_rate: float = 0.5               # 最大充放电功率 = 容量 × 比率 (C-rate)
    window_stride_days: int = 1                # 时间窗滑动间隔 (未在代码中使用)
```

**含义强调**：
- **`episode_steps=96` + `dt_hours=0.25`** = 每个step代表15分钟 = 96步 = 1天
- **`train_window_days=7`** = 一个training episode = 7天 = 96×7 = 672 steps
- **计算**: `pmax[i] = battery_capacity_kwh[i] * max_charge_rate` = 100 × 0.5 = **50 kW**
  - 每个agent可充/放电的最大功率是50 kW
  - 15分钟内最多充入/放出 50 × 0.25 × 0.95 = **11.875 kWh** 
- **SOC初始化**:
  - eval时固定 `init_soc=0.05` 
  - train时随机 `uniform(0.05, 0.05)` = 总是0.05（这是设计上的固定值）
- **效率模型**：
  - 充电: `delta_soc = (action_power × dt × efficiency) / capacity`
  - 放电: `delta_soc = (action_power × dt) / (efficiency × capacity)`
  - 充电时能量利用率好，放电时折损率更高

---

### 🧠 AlgoCfg - 强化学习算法

```python
@dataclass
class AlgoCfg:
    name: str = "MADRL_PENALTY"
    gamma: float = 0.999                       # 折扣因子 (接近1 = 长视野)
    tau: float = 0.01                          # 目标网络软更新系数
    policy_update_freq: int = 2                # 每2个critic更新，更新1次policy
    policy_noise: float = 0.2                  # Actor输出的探索噪声标准差
    noise_clip: float = 0.5                    # 噪声裁剪范围
```

**含义强调**：
- `gamma=0.999`: 非常高 → agent关注长期收益 → 适合储能（充放电周期跨度多步）
- `policy_update_freq=2`: Critic更新频率是Policy的2倍 → 稳定性优先
- `policy_noise=0.2`: 目标Policy平滑，提高训练稳定性

---

### 🎓 TrainCfg - 训练超参

```python
@dataclass
class TrainCfg:
    train_episodes: int = 500                  # 训练总episode数
    num_envs: int = 4                          # 并行环境数
    parallel_episode_sampling: str = "unique_active"  # 采样策略
    buffer_size: int = 100_000                 # 经验回放缓冲区大小
    batch_size: int = 512                      # 每次更新的batch大小
    actor_lr: float = 1e-4                     # Actor学习率
    critic_lr: float = 1e-4                    # Critic学习率
    learning_starts: int = 5_376               # 开始更新前收集的步数 (0.5 episodes × 4 envs)
    actor_learning_starts: int = 8_064         # Actor何时开始学习 (0.75 episodes × 4 envs)
    n_step_return: int = 96                    # n-step bootstrapping (1天的长度)
    updates_per_step: int = 1                  # 每环境step更新1次网络
    update_interval: int = 1                   # 更新间隔
    noise_std_init: float = 0.4                # 初始探索噪声标准差
    noise_std_min: float = 0.2                 # 最小噪声标准差
    noise_decay_steps: int = 300_000           # 噪声衰减总步数
    feasible_random_exploration_start: float = 0.5     # 早期可行性探索启用阈值
    feasible_random_exploration_end: float = 0.05      # 结束阈值
    feasible_random_exploration_decay_steps: int = 50_000  # 衰减速度
```

**含义强调**：
- **计算总训练步数**: `500 episodes × 672 steps/episode × 4 envs = 1,344,000 steps`
- `learning_starts=5_376`: 相当于 5376 / (96 × 4) ≈ 0.5个episode → 充分预热缓冲区
- `n_step_return=96`: **关键** - 等于 `episode_steps` → 用整个工作日的回报做bootstrapping
- `noise_decay_steps=300_000`: 噪声在30万步内从0.4衰减到0.2

---

### 🔍 ObsCfg - 观测与归一化

```python
@dataclass
class ObsCfg:
    sequence_length: int = 49                  # 历史序列长度 (49个15分钟步 ≈ 12小时)
    normalization_enabled: bool = True         # 启用robust归一化
    wholesale_price_tanh_scale: float = 2.0    # 价格tanh缩放因子
    wholesale_price_spread_scale_eur_per_kwh: float = 0.20  # 价格波动范围
    load_tanh_scale: float = 3.0               # 负荷tanh缩放因子
    normalization_clip_low_quantile: float = 0.01   # 异常值剪裁下界 (1%)
    normalization_clip_high_quantile: float = 0.99  # 异常值剪裁上界 (99%)
```

**含义强调**：
- `sequence_length=49`: **关键** - 观测包含 49 × 15分钟 = **12.25小时的历史** (包括当前步)
- 这让agent能看到半天前的趋势，做出更好的储能决策
- Robust归一化: 用分位数而不是min/max → 对异常值鲁棒

---

### 🔮 ForecastCfg - 预测器配置

```python
@dataclass
class ForecastCfg:
    mode: str = "perfect"                      # perfect 或 lstm
    target_signals: tuple[str, ...] = ("wholesale_price", "load", "pv")
    lstm_artifact_dir: str | None = None       # LSTM模型存储位置
    history_window: int = 192                  # LSTM输入窗口 (48小时)
    lstm_batch_size: int = 1024
    price_lstm_epochs: int = 50
    price_lstm_hidden_size: int = 128
    price_lstm_num_layers: int = 2
    price_lstm_dropout: float = 0.1
    ...
    load_model_mode: str = "per_agent"         # 每个agent独立的负荷预测
    load_component_split: bool = True          # 分别预测household和heatpump
```

**含义强调**：
- `mode="perfect"`: 使用真实未来数据 (oracle) 作为对标
- `mode="lstm"`: 使用训练好的LSTM预测
- `history_window=192`: LSTM用过去 192×15分钟 = **48小时** 做预测
- 预测任务是困难的，LSTM是为了与oracle对标

---

### 🏭 RewardCfg - 奖励权重

```python
@dataclass
class RewardCfg:
    action_boundary_penalty_weight: float = 0.05        # SOC边界动作惩罚
    soc_boundary_regularization_weight: float = 0.005   # SOC软边界L2正则
    throughput_bonus_eur_per_kwh_max: float = 0.002     # 早期探索吞吐bonus
    soc_boundary_epsilon: float = 0.02                  # SOC边界epsilon
    soc_boundary_margin: float = 0.02                   # 软边界宽度
    import_price_markup_eur_per_kwh: float = 0.0        # 进口加价 (充电时)
    export_subsidy_eur_per_kwh: float = 0.0             # 出口补贴 (放电时)
    w_voltage_pen: float = 1.0                          # 电压惩罚权重
    w_line_pen: float = 1.0                             # 线路惩罚权重
    w_trafo_pen: float = 1.0                            # 变压器惩罚权重
```

**含义强调**：
- **基础收益**: 储能套利 = (放电功率 - 充电功率) × 价格
- **三类惩罚**:
  1. 动作边界惩罚: 在SOC极限附近充放电时惩罚 (防止操作极端)
  2. SOC正则化: 软边界(不严格) - 偏离中心范围就轻微惩罚
  3. 安全惩罚: 电压/线路/变压器违反时惩罚
- `throughput_bonus`: 早期探索奖励 → 鼓励agent探索不同充放电策略

---

### 🎬 ModelCfg - 神经网络结构

```python
@dataclass
class ModelCfg:
    hidden_dim: int = 256                      # 隐层维度 (小而固定)
    action_dim: int = 2                        # 动作维度 (电池功率 + PV利用率)
    max_action: float = 1.0                    # 动作值范围 [-1, 1]
    use_grad_clip: bool = True                 # 梯度剪裁
    grad_clip_norm: float = 10.0               # 剪裁范数
```

---

### 🛡️ SafetyCfg - 安全约束

```python
@dataclass
class SafetyCfg:
    enabled: bool = False                      # 默认关闭
    projection_iters: int = 3                  # 投影迭代数
    voltage_margin_pu: float = 0.005           # 电压裕度
    line_margin_pct: float = 5.0               # 线路裕度 (%)
    trafo_margin_pct: float = 5.0              # 变压器裕度 (%)
```

---

## 📌 第二部分：环境系统 (envs/)

### 🏗️ 三层架构

```
┌─────────────────────────────────────┐
│      GridEnv (环境主类)              │
│  - 管理观测、动作、奖励、SOC         │
│  - 调用 PowerFlowGridCore 做潮流     │
│  - 调用 MadrlObservationStats 归一化 │
└─────────────────────────────────────┘
            ↓ 使用
┌─────────────────────────────────────┐
│    PowerFlowGridCore (潮流计算)      │
│  - 加载SimBench拓扑                  │
│  - 调用pandapower做PF                │
│  - 返回电压/线路/变压器状态          │
└─────────────────────────────────────┘
            ↓ 并行化
┌─────────────────────────────────────┐
│     SubprocVecEnv (多进程采样)       │
│  - 4个并行环境实例                   │
│  - "unique_active"采样 → 每个env不同  │
└─────────────────────────────────────┘
```

---

### 🔌 PowerFlowGridCore - 潮流计算

```python
class PowerFlowGridCore:
    def __init__(self, cfg):
        self.agent_bus_ids = (12, 4, 2)        # 三个agent位置
        self.net = _simbench_net("1-LV-rural1--0-sw")  # 加载真实拓扑
        self.trafo_limit_kw = ...              # 变压器功率限额
        self.load_rows, self.sgen_rows = ...   # 在每个bus创建负荷/发电节点
```

**核心函数: `step(net_load_kw: np.ndarray) → GridStep`**

```python
def step(self, net_load_kw):
    # 1. 将三个agent的净注入功率写入网络
    self._write_net_load(net_load_kw)  # net_load[i] = 负荷 - PV有效利用 + 充电功率
    
    # 2. 运行潮流计算
    self._run_power_flow()             # pandapower Newton-Raphson
    
    # 3. 提取结果
    vm_all = net.res_bus["vm_pu"]      # 所有母线电压 [V/Vnom]
    agent_vm = vm_all[agent_bus_pos]   # 三个agent处的电压
    line_loading = net.res_line["loading_percent"]  # 线路负荷百分比
    trafo_loading = net.res_trafo["loading_percent"] # 变压器负荷百分比
    
    # 4. 计算违反项
    v_violation = max(0, 0.95 - vm) + max(0, vm - 1.05)  # 电压超出范围
    line_excess = max(0, loading - 100%) / 100          # 线路过载
    trafo_excess = max(0, loading - 100%) / 100         # 变压器过载
    
    # 5. 聚合为平方惩罚
    psi_v_raw = sum((v_excess)^2)      # 全网电压平方和
    psi_line_raw = sum((line_excess)^2)
    psi_trafo_raw = sum((trafo_excess)^2)
    
    return GridStep(...)
```

**关键点**：
- **Net Load概念**: `net_load = 负荷 - PV_effective + 充电功率`
  - 正值 = 从网络吸收功率 (充电或消费)
  - 负值 = 注入功率到网络 (放电或PV出力)
- **结果是聚合的**: 所有总线的电压、所有线路的负荷都被考虑，计算全网的平方和惩罚

---

### 🌍 GridEnv - 强化学习环境

#### 初始化

```python
class GridEnv:
    def __init__(self, cfg, split, forecast_mode, share_data):
        self.split = "train" or "eval"
        self.forecast_mode = "perfect" or "lstm"
        self.base_steps = 96              # 每天步数
        self.train_days = 7 if split=="train" else 1   # 单个episode长度
        self.steps = 96 * train_days      # 总步数 (672 or 96)
        self.cap = [100, 100, 100]        # 电池容量 (3个agent)
        self.pmax = [50, 50, 50]          # 最大功率 = cap × 0.5
        self.grid_core = PowerFlowGridCore(cfg)  # 潮流计算器
        self.soc = [0.05, 0.05, 0.05]     # 初始SOC
```

#### Reset - 重置环节

```python
def reset(self, episode_index=None):
    self.episode = random_episode_id()   # 从数据中随机选择一个episode (day)
    self.cur_step = 0                    # 重置当前步
    
    if self.split == "train":
        # 训练时: SOC初始范围 [0.05, 0.05] (固定)
        self.soc = np.random.uniform(0.05, 0.05, size=3)
    else:
        # 评估时: SOC固定
        self.soc = [0.05, 0.05, 0.05]
    
    obs = self._obs()
    return obs, {"episode_idx": episode_id, "initial_soc": [0.05, 0.05, 0.05]}
```

#### Step - 核心时间步

```python
def step(self, action):
    # ===== 1. 解析动作 =====
    battery_action = action[:, 0]        # 电池功率比例 ∈ [-1, 1]
    pv_action = action[:, 1]             # PV利用率 ∈ [-1, 1]
    
    # ===== 2. 获取当前时刻数据 =====
    load = data["load"][episode, step]   # (3,) - 三个家庭的总负荷
    pv = data["pv"][episode, step]       # scalar - 共享PV输出
    price = data["price"][episode, step] # scalar - 电价
    
    # ===== 3. 转换动作为功率 =====
    # 电池功率: [-pmax, pmax]
    charge_kw = battery_action * pmax    # ∈ [-50, 50] kW (正=充，负=放)
    
    # PV利用率: pv_action ∈ [-1, 1] → 利用率 ∈ [0, 1]
    pv_utilization = (pv_action + 1) / 2
    pv_effective = pv * pv_utilization   # 有效PV输出
    
    # ===== 4. SOC裁剪 =====
    # 根据当前SOC,对电池功率进行裁剪 (防止过充/过放)
    delta = charge_kw * dt * eff / cap   # SOC变化
    next_soc_unclamped = soc + delta
    
    # 如果动作会导致SOC超限，裁剪动作
    projected_charge = project_action_to_soc(cfg, soc, battery_action)
    charge_kw = projected_charge * pmax
    
    # ===== 5. 计算净注入功率 =====
    net_load = load - pv_effective + charge_kw
    # 示例: load=30, pv_eff=10, charge=20 → net=40 (从网络吸收40kW)
    
    # ===== 6. 潮流计算 =====
    pf = grid_core.step(net_load)  # 返回GridStep对象
    v_violation = pf.v_violation   # (3,) 电压违反量
    vm = pf.vm_pu                  # (3,) 实际电压
    line_loading = pf.line_loading_pct
    trafo_loading = pf.trafo_loading_pct
    
    # ===== 7. 计算奖励分量 =====
    
    # 7a. 储能套利收益
    storage_profit = (max(0, -charge_kw) - max(0, charge_kw)) * price * dt
    # 放电赚钱: -charge_kw > 0 (负数) → -(-值) = 正利润
    # 充电花钱: charge_kw > 0 → -收益
    
    # 7b. SOC边界动作惩罚 (防极端操作)
    boundary_push = (charge_kw < 0 & soc ≤ soc_min + ε) | (charge_kw > 0 & soc ≥ soc_max - ε)
    action_penalty = 0.05 * |charge_kw| * boundary_push
    
    # 7c. SOC软边界正则 (鼓励在中间范围内)
    soft_low = 0.05 + 0.02 = 0.07
    soft_high = 0.95 - 0.02 = 0.93
    soc_reg = 0.005 * (max(0, soft_low - soc)^2 + max(0, soc - soft_high)^2)
    
    # 7d. 吞吐bonus (早期探索)
    progress = min(1.0, current_steps / total_steps)
    throughput_weight = 0.002 * max(0, (0.80 - progress) / 0.60)
    throughput_bonus = throughput_weight * |charge_kw| * dt
    
    # 7e. 电压/线路/变压器安全惩罚
    v_weights = v_violation / sum(v_violation)  # 分配给超限agent
    safe_v = 3 * 1.0 * psi_v_raw * v_weights
    safe_line = 1.0 * psi_line_raw
    safe_trafo = 1.0 * psi_trafo_raw
    
    # ===== 8. 总奖励 =====
    reward = storage_profit - action_penalty - soc_reg + throughput_bonus - safe_v - safe_line - safe_trafo
    
    # ===== 9. 状态转移 =====
    soc_next = clip(soc + delta, soc_min, soc_max)
    cur_step += 1
    done = (cur_step >= steps)
    
    # ===== 10. 观测 =====
    obs = self._obs() if not done else self._terminal_obs()
    
    return obs, reward, done, False, info
```

**关键计算**:

| 参数 | 公式 | 示例 |
|------|------|------|
| 充放电SOC变化 | δ = P × dt × η / C (充) 或 δ = P × dt / (η × C) (放) | 50 kW × 0.25 h × 0.95 / 100 = 0.011875 |
| 套利收益 | (P_discharge - P_charge) × price × dt | (0 - 50) × 0.1 €/kWh × 0.25 = -1.25 € |
| PV有效输出 | PV_eff = PV_total × utilization | 20 kW × 0.5 = 10 kW |
| 电压违反 | V_viol = max(0, V_min - V) + max(0, V - V_max) | max(0, 0.95-0.93) + max(0, 0.93-1.05) = 0.02 |

---

#### 观测结构

```python
def _obs():
    # 当前时刻本地状态
    local = [soc, load, pv, price]  # (3, 4) for 3 agents
    
    # 未来序列 (49步 = 12.25小时)
    price_seq = price_forecast[t:t+49]  # (49,)
    load_seq = load_forecast[t:t+49]    # (49, 3) per-agent
    pv_seq = pv_forecast[t:t+49]        # (49, 3) broadcast to agents
    sequence = stack[price_seq, load_seq, pv_seq]  # (3, 49, 3)
    
    # 安全约束相关
    safety_local = [soc, load, pv, capacity, pmax]  # (3, 5)
    
    # 价格相对值与波动
    price_relative = normalize(price_seq)  # (49,)
    price_spread = max(price_seq) - min(price_seq)  # scalar
    
    # 归一化的负荷和PV
    load_normalized = robust_tanh(load_seq, stats)  # (3, 49)
    pv_normalized = clip(pv_seq / pv_max, 0, 1.2)  # (3, 49)
    
    # 日历特征
    calendar = [sin(2π×hour/24), cos(...), sin(2π×doy/365.25), cos(...)]  # (3, 4)
    madrl_local = concat[calendar, soc_normalized]  # (3, 5)
    
    return {
        "local": (3, 4),                    # 基础信息
        "sequence": (3, 49, 3),             # 未来序列
        "madrl_local": (3, 5),              # 日历 + SOC
        "safety_local": (3, 5),             # 安全相关
        "load_seq": (3, 49),                # 负荷预测
        "pv_seq": (3, 49),                  # PV预测
        "wholesale_price_relative_seq": (49,),   # 价格相对值
        "wholesale_price_spread_seq": (49,),     # 价格波动
    }
```

---

### 🚀 SubprocVecEnv - 多进程采样

```python
class SubprocVecEnv:
    def __init__(self, num_envs=4, env_factory=..., seed=0):
        # 创建4个子进程，每个进程独立运行一个环境
        for i in range(4):
            proc, child = Pipe()
            worker_proc = Process(target=_subproc_worker, args=(child, env_factory))
            worker_proc.start()
    
    def reset():
        # 1. 采样4个不同的episode index (unique_active采样)
        episodes = sampler.next_wave()  # [e1, e2, e3, e4]
        
        # 2. 并行重置4个环境
        for worker, ep_idx in zip(workers, episodes):
            worker.send(("reset", ep_idx))
        
        # 3. 堆叠返回 (batch维度在第0轴)
        obs_batch = stack([obs1, obs2, obs3, obs4])  # (4, 3, ...) 
        return obs_batch, info_list
    
    def step(action_batch):
        # action_batch: (4, 3, 2) = 4env × 3agent × 2action
        
        for worker, action in zip(workers, action_batch):
            worker.send(("step", action))
        
        obs_batch, reward_batch, done_batch, _, info_list = zip(*results)
        
        # 关键: 同步完整性检查
        assert sum(done_batch) in {0, 4}  # 要么全部继续，要么全部结束
        
        if all(done_batch):
            # 自动采样新episode并重置
            obs_batch = auto_reset()
        
        return stack(obs), stack(rewards), done_batch, ..., info_list
```

**并行采样模式**:
```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  Env#0     │  │  Env#1     │  │  Env#2     │  │  Env#3     │
│  Episode5  │  │  Episode12 │  │  Episode8  │  │  Episode2  │
│  t=0..96   │  │  t=0..96   │  │  t=0..96   │  │  t=0..96   │
└────────────┘  └────────────┘  └────────────┘  └────────────┘
       ↓               ↓               ↓               ↓
       └───────────────┬───────────────┬───────────────┘
                       ↓
          (4, 3, 4) observation batch
          (4, 3) reward batch
          (4,) done batch
```

---

## 🎯 第三部分：工作流串联

### 训练循环

```
1. cfg 初始化
   ↓
2. build_env(cfg, "train", "lstm", share_data) → GridEnv
   ↓
3. SubprocVecEnv(num_envs=4, env_factory=...) → 4个并行环境
   ↓
4. 训练循环 (500 episodes):
   
   episode_start:
   └─ reset() → obs_batch, info
      ↓
   while not done:
      ├─ agent forward(obs) → action (4, 3, 2)
      ├─ step(action) → obs_batch, reward, done, info
      │  ├─ 每个env执行 GridEnv.step()
      │  │  ├─ 解析动作 → 电池功率 + PV利用率
      │  │  ├─ 裁剪SOC
      │  │  ├─ 潮流计算 (PowerFlowGridCore.step)
      │  │  ├─ 计算奖励
      │  │  └─ 返回新观测
      │  └─ 堆叠结果 (4个环境并行)
      ├─ 存储到replay buffer
      └─ 更新网络 (critic + actor)
   
   当 done == True (4个环境全部完成1个episode):
   └─ 自动采样新episode并重置
```

### 数据流

```
CSV数据 (2019-2020年)
  ├─ household.csv (每15分钟)
  ├─ heatpump.csv
  ├─ pv_reference.csv
  └─ price.csv
       ↓
  data/loader.py → SeriesData (按时间窗裁剪)
       ↓
  data/share_data.py → .npz文件
       ├─ train.npz (2019全年 × 7天窗 × 3agents)
       ├─ eval.npz (2020年4月1-15日 × 1天窗)
       └─ manifest.json
            ↓
  GridEnv._obs() → 读取share_data中的
       ├─ train/eval data (负荷、PV、价格)
       ├─ forecast_mode="lstm" → 预测值
       └─ 生成序列观测
```

---

## ⚡ 第四部分：关键参数交互

| 参数 | 含义 | 影响 |
|------|------|------|
| `episode_steps=96` | 每天的离散时间步 | 与`dt=0.25h`相乘→1天 |
| `train_window_days=7` | 训练episode长度 | 7天×672步/天 = 一个4704步的长序列 |
| `sequence_length=49` | 观测的时间地平线 | agent能看12小时前的趋势 |
| `n_step_return=96` | Bootstrapping长度 | 整个工作日的回报累积 |
| `battery_capacity_kwh=100` + `max_charge_rate=0.5` | 电池参数 | pmax=50kW, 充放电速度受限 |
| `noise_decay_steps=300K` | 探索衰减 | 第30万步后探索噪声固定在最小值 |
| `num_envs=4` | 并行环境 | 每个step收集 4 episode × 96 steps = 384条轨迹 |

---

## 🔧 常见调整指南

### 如果想增加问题难度:
- ↑ `train_window_days`: 7→14 (更长的剧集, 更复杂的长期规划)
- ↑ `num_agents`: 3→5 (更多agent, 网络约束更紧张)
- ↓ `line_limit_kw`: 80→60 (更严格的线路约束)
- ↓ `v_max_pu`: 1.05→1.02 (更严格的电压约束)

### 如果想加速训练:
- ↑ `num_envs`: 4→8 (更多并行采样)
- ↑ `batch_size`: 512→1024 (更大的梯度更新)
- ↑ `updates_per_step`: 1→2 (每步更新2次)
- ↓ `learning_starts`: 5376→2000 (更早开始学习)

### 如果奖励信号不清晰:
- ↑ `action_boundary_penalty_weight`: 0.05→0.1 (更强的SOC边界惩罚)
- ↑ `w_voltage_pen`: 1.0→5.0 (更强的电压约束)
- ↑ `throughput_bonus_eur_per_kwh_max`: 0.002→0.005 (鼓励更多探索)

