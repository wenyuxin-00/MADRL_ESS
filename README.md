# MADRL_ESS 项目说明与建模文档

`MADRL_ESS` 是一个面向低压配电网中多 prosumer 储能与电动汽车充电协同控制的研究型代码库。项目主线是一条完整实验链：读取 prosumer 时序数据，训练 LSTM 预测器，构建统一 `share_data` 契约，在 SimBench/pandapower 潮流环境中训练与评估多智能体强化学习控制器，并与 `LOCAL_MPC`、`ADMM_MPC`、全局 `MISOCP` 优化基线比较。

这份 README 说明当前代码实现的完整建模逻辑。配置入口是 `configs/cfg.py` 中的 `Cfg`；实验产物默认写入 `artifacts/runs/<run_id>/`。项目刻意采用显式契约：路径、schema、控制器名称、预测模式、run 目录都必须明确给出，缺文件、缺列、缺 solver 或缺 artifact 时直接失败。

## 一、问题定义

系统研究一个低压配电网中的 3 个 prosumer agent。默认网络为 SimBench `1-LV-rural1--0-sw`，agent 接入母线为 `(12, 4, 2)`。每个 agent 同时具有本地负荷、共享 PV、静态储能电池和 EV 家充需求。每个 15 分钟时间步，控制器决定：

- 静态电池充放电功率；
- PV 利用或弃光比例；
- EV 充电功率。

环境将这些动作转成净负荷，写入 pandapower 网络并执行潮流计算。电网电压、线路负载、变压器负载会反馈到奖励、约束和评估指标中。

核心目标不是单纯最大化储能套利收益，而是在电价、PV、负荷、EV 离家 SoC 要求和配电网安全约束共同作用下，比较不同控制策略的经济性、安全性和可复现行为。

## 二、符号约定

主要集合与索引：

| 符号 | 含义 |
| --- | --- |
| `i in {1,...,N}` | agent/prosumer 索引，默认 `N=3` |
| `t` | 离散时间步，默认 15 分钟 |
| `H` | 预测窗口长度，代码中等于 `cfg.obs.sequence_length`，默认 49 |
| `T` | episode 步数，默认 `cfg.env.episode_steps=96` |
| `Delta t` | 时间步长，默认 `0.25 h` |
| `p_t` | 当前批发电价，单位 EUR/kWh |
| `L_{i,t}` | agent `i` 当前负荷，单位 kW |
| `G_{i,t}` | agent `i` 当前 PV 可用功率，单位 kW |
| `P^b_{i,t}` | 静态电池有符号功率，正为充电，负为放电，单位 kW |
| `P^{ev}_{i,t}` | EV 充电功率，单位 kW |
| `u^{pv}_{i,t}` | PV 利用率，范围 `[0,1]` |
| `SOC^b_{i,t}` | 静态电池 SoC |
| `SOC^{ev}_{i,t}` | EV SoC |
| `C^b_i` | 静态电池容量，单位 kWh |
| `C^{ev}_i` | EV 电池容量，单位 kWh |
| `P^{b,max}_i` | 静态电池最大充放电功率，单位 kW |
| `P^{ev,max}_i` | EV 家充最大功率，单位 kW |
| `eta_b` | 静态电池效率 |
| `eta_ev` | EV 充电效率 |

代码中的默认值来自 `Cfg`：

- `N=3`
- `Delta t=0.25`
- `C^b_i=100 kWh`
- `P^{b,max}_i=C^b_i * max_charge_rate = 50 kW`
- `SOC^b in [0.05, 0.95]`
- `C^{ev}_i=60 kWh`
- `P^{ev,max}_i=11 kW`
- `SOC^{ev} in [0.10, 0.95]`
- EV 到家步 `72`，离家步 `28`，对应 18:00 到家、07:00 离家
- EV 离家目标 `SOC^{ev,req}=0.90`

## 三、数据建模

原始数据位于 `datasets/prosumer/`：

| 文件 | 用途 |
| --- | --- |
| `household.csv` | 住宅负荷分量 |
| `heatpump.csv` | 热泵负荷分量 |
| `pv_reference.csv` | PV 参考出力 |
| `price.csv` | 批发电价 |

`data.loader.load_prosumer_dataset` 将 CSV 统一读成 `SeriesData`。时间戳按 UTC 读取后转换到 `Europe/Berlin`。训练与评估年份默认由 `cfg.data.share_data_train_test=(2019, 2020)` 指定。

对 agent `i`，总负荷建模为：

```text
L_{i,t} = load_scale_i * sum_{c in components} L^{raw}_{c,i,t}
```

其中 `components=("household", "heatpump")`。当前代码中 `load_components` 已经分别乘过 `load_scale`，随后总负荷 `load` 又乘一次 `load_scale`；README 描述的是当前实现效果，而不是物理上唯一可能的定义。

PV 建模分两种情况：

```text
if pv_capacity_kw is empty:
    G_{i,t} = pv_scale_i * G^{ref}_t
else:
    G_{i,t} = pv_scale_i * G^{ref}_t * pv_capacity_kw_i / max_t(G^{ref}_t)
```

默认 `pv_capacity_kw=()`，因此 PV 参考序列按 agent 复制，并乘以 `pv_scale_i`。

## 四、预测建模与 share_data 契约

项目支持两种预测模式：

- `perfect`：直接使用真实未来序列作为预测窗口。
- `lstm`：使用训练好的 LSTM 预测价格、负荷和 PV。

LSTM 输入是长度 `history_window` 的历史窗口，输出长度为 `H=sequence_length` 的未来序列。对任一标量信号 `x_t`，训练样本为：

```text
X_k = [x_k, x_{k+1}, ..., x_{k+h-1}]
y_k = [x_{k+h}, x_{k+h+1}, ..., x_{k+h+H-1}]
```

其中 `h=cfg.forecast.history_window`。价格和 PV 是共享序列；负荷按 agent 和 component 建模，即对每个 `(component, agent)` 单独训练 LSTM。负荷与 PV 可以拼接时间特征，时间特征来自小时、星期/年份周期编码。

LSTM 模型形式为：

```text
z_k = LSTM(feature_window_k)
y_hat_k = Linear(z_k)
```

负荷预测还带有 baseline blend 后处理。一步验证误差会在候选权重中选择 `alpha`：

```text
load_pred = last_value + alpha * (lstm_raw_pred - last_value)
```

对于 heatpump，当前代码固定 `alpha=0`，即使用 last-value baseline，避免明显偏置。

`data/share_data.py` 将真实序列和预测序列写成统一契约：

```text
share_data/
  train.npz
  eval.npz
  manifest.json
```

每个 split 中的核心数组包括：

```text
price[e,t]
load[e,t,i]
pv[e,t]
perfect_price_seq[e,t,h]
perfect_load_seq[e,t,h,i]
perfect_pv_seq[e,t,h]
lstm_price_seq[e,t,h]
lstm_load_seq[e,t,h,i]
lstm_pv_seq[e,t,h]
timestamps[e,t]
```

其中 `e` 为 episode/window 索引，`t` 为 episode 内步，`h` 为预测窗口内偏移。

## 五、环境状态与观测建模

`GridEnv` 是主要环境。每个 episode reset 时：

- 训练模式下，静态电池 SoC 从 `[train_init_soc_low, train_init_soc_high]` 采样；
- 评估模式下，静态电池 SoC 固定为 `init_soc`；
- EV SoC 固定为 `ev_arrival_soc`。

环境内部原始状态可概括为：

```text
s_t = {
  episode_index,
  step,
  SOC^b_{1:N,t},
  SOC^{ev}_{1:N,t},
  L_{1:N,t},
  G_{1:N,t},
  p_t,
  forecast_window_t,
  grid_state_t
}
```

返回给控制器的 observation 包含多种视图：

| 键 | 形状含义 | 用途 |
| --- | --- | --- |
| `local` | 每个 agent 的 `[soc, ev_soc, ev_available, load, pv, price]` | 普通控制器窗口读取 |
| `sequence` | 每个 agent 的 `[price_seq, load_seq, pv_seq]` | MPC/MISOCP 窗口 |
| `madrl_local` | 日周期特征、年周期特征、归一化电池 SoC、归一化 EV SoC、EV 可用标记 | MADRL actor/critic |
| `safety_local` | `[soc, ev_soc, ev_available, load, pv, cap, pmax, ev_cap, ev_pmax]` | 本地可行性与安全投影 |
| `wholesale_price_relative_seq` | 价格窗口相对归一化 | MADRL |
| `wholesale_price_spread_seq` | 价格窗口价差强度 | MADRL |
| `load_seq` | robust-tanh 归一化负荷窗口 | MADRL |
| `pv_seq` | PV 按训练最大值缩放 | MADRL |

日周期和年周期特征为：

```text
hour_feature = [sin(2*pi*hour/24), cos(2*pi*hour/24)]
year_feature = [sin(2*pi*doy/365.25), cos(2*pi*doy/365.25)]
```

静态电池 SoC 和 EV SoC 都线性缩放到 `[-1, 1]`：

```text
soc_norm = clip(2*(soc - soc_min)/(soc_max - soc_min) - 1, -1, 1)
ev_soc_norm = clip(2*(ev_soc - ev_soc_min)/(ev_soc_max - ev_soc_min) - 1, -1, 1)
```

价格窗口相对归一化为：

```text
price_rel_h = 0, if max(price_seq)-min(price_seq) <= eps
price_rel_h = 2*(price_h - min(price_seq))/(max(price_seq)-min(price_seq)) - 1, otherwise
```

## 六、动作空间与物理映射

当前 `cfg.model.action_dim=3`，每个 agent 的动作：

```text
a_{i,t} = [a^b_{i,t}, a^{pv}_{i,t}, a^{ev}_{i,t}] in [-1,1]^3
```

静态电池动作先映射为功率：

```text
P^b_{i,t} = a^b_{i,t} * P^{b,max}_i
```

其中正值表示充电，负值表示放电。环境会根据当前 SoC 裁剪动作，使下一步 SoC 不越过硬边界。可行动作上下界在代码中等价于：

```text
upper_i = (soc_max - SOC^b_{i,t}) * C^b_i / (P^{b,max}_i * Delta t * eta_b)
lower_i = (soc_min - SOC^b_{i,t}) * eta_b * C^b_i / (P^{b,max}_i * Delta t)
a^b_{i,t} <- clip(a^b_{i,t}, max(lower_i,-1), min(upper_i,1))
```

PV 动作映射为利用率：

```text
u^{pv}_{i,t} = clip(0.5*(a^{pv}_{i,t}+1), 0, 1)
G^{use}_{i,t} = u^{pv}_{i,t} * G_{i,t}
G^{curt}_{i,t} = G_{i,t} - G^{use}_{i,t}
```

EV 动作映射为充电功率：

```text
A^{ev}_t = 1, if EV connected at step t; otherwise 0
P^{ev,rl}_{i,t} = A^{ev}_t * clip(0.5*(a^{ev}_{i,t}+1), 0, 1) * P^{ev,max}_i
```

最终 EV 充电功率 `P^{ev}_{i,t}` 还可能被 hard projection 或 emergency charging 修正。

净负荷为：

```text
P^{net}_{i,t} = L_{i,t} - G^{use}_{i,t} + P^b_{i,t} + P^{ev}_{i,t}
```

如果 `P^{net}_{i,t} >= 0`，该 agent 在电网模型中表现为负荷；如果 `P^{net}_{i,t} < 0`，表现为并网发电。

## 七、静态电池动力学

静态电池 SoC 更新为：

```text
if P^b_{i,t} >= 0:
    SOC^b_{i,t+1} = SOC^b_{i,t} + P^b_{i,t} * Delta t * eta_b / C^b_i
else:
    SOC^b_{i,t+1} = SOC^b_{i,t} + P^b_{i,t} * Delta t / (eta_b * C^b_i)
```

之后环境执行硬裁剪：

```text
SOC^b_{i,t+1} <- clip(SOC^b_{i,t+1}, soc_min, soc_max)
```

注意 `P^b` 正为充电、负为放电，因此放电项会降低 SoC。

## 八、EV 连接、动力学与离家约束

EV 连接窗口由 `ev_arrival_step` 和 `ev_departure_step` 定义。默认到家步 72、离家步 28，因此跨午夜连接：

```text
A^{ev}_t = 1, if t_daily >= 72 or t_daily < 28
A^{ev}_t = 0, otherwise
```

到家步会重置 EV SoC：

```text
if t_daily == ev_arrival_step:
    SOC^{ev}_{i,t} = ev_arrival_soc
```

EV SoC 更新为：

```text
SOC^{ev}_{i,t+1} = clip(
    SOC^{ev}_{i,t} + P^{ev}_{i,t} * Delta t * eta_ev / C^{ev}_i,
    ev_soc_min,
    ev_soc_max
)
```

离家步检查目标 SoC：

```text
gap^{ev}_{i,t} = max(0, ev_departure_soc_req - SOC^{ev}_{i,t+1}), if t_daily == ev_departure_step
gap^{ev}_{i,t} = 0, otherwise
```

### Soft 模式

`soft` 模式不改动作，只在离家步加惩罚：

```text
pen^{ev,dep}_{i,t} = w_ev_dep * (gap^{ev}_{i,t})^2
```

### Hard projection 模式

`hard` 模式在 `ev_hard_projection_enabled=True` 时，环境计算“为了保持离家目标仍可达，当前至少应充多少电”：

```text
future_steps = number of future connected steps before departure
max_future_gain_i = future_steps * P^{ev,max}_i * Delta t * eta_ev / C^{ev}_i

P^{ev,req}_{i,t} =
    max(0, (SOC^{ev,req} - max_future_gain_i - SOC^{ev}_{i,t})
           * C^{ev}_i / (Delta t * eta_ev))

P^{ev}_{i,t} = clip(max(P^{ev,rl}_{i,t}, P^{ev,req}_{i,t}), 0, P^{ev,max}_i)
projection_gap_i = max(P^{ev}_{i,t} - P^{ev,rl}_{i,t}, 0)
```

如果 `P^{ev,req}_{i,t} > P^{ev,max}_i`，该步被标记为 hard infeasible。

### Emergency 模式

`emergency` 模式只在离家前的应急窗口内启用自动补电。若策略为 `required_power`：

```text
energy_needed_i = max(0, SOC^{ev,req} - SOC^{ev}_{i,t}) * C^{ev}_i
remaining_time = remaining_emergency_steps * Delta t
P^{ev,req}_{i,t} = clip(energy_needed_i / (eta_ev * remaining_time), 0, P^{ev,max}_i)
P^{ev}_{i,t} = clip(max(P^{ev,rl}_{i,t}, P^{ev,req}_{i,t}), 0, P^{ev,max}_i)
```

若策略为 `max_power`，则应急窗口内低于目标的 EV 直接请求最大功率。

## 九、配电网潮流与安全指标

`PowerFlowGridCore` 使用 SimBench 构造 pandapower 网络，并将 agent 净负荷写入对应母线：

```text
if P^{net}_{i,t} >= 0:
    load.p_mw = P^{net}_{i,t}/1000
    sgen.p_mw = 0
else:
    load.p_mw = 0
    sgen.p_mw = -P^{net}_{i,t}/1000
```

随后执行：

```text
pandapower.runpp(..., algorithm=cfg.grid.pf_solver, numba=False)
```

潮流返回：

- agent 母线电压 `V_{i,t}`；
- 全网线路负载率 `line_loading_pct`;
- 变压器负载率 `trafo_loading_pct`;
- 电压、线路、变压器惩罚原始量。

电压违约按 agent 母线计算：

```text
v_violation_i = max(0, v_min - V_i) + max(0, V_i - v_max)
```

全网电压惩罚原始量为：

```text
psi_v = sum_b (max(0, v_min - V_b) + max(0, V_b - v_max))^2
```

线路与变压器惩罚原始量为：

```text
psi_line = sum_l (max(0, loading_l - line_limit_pct)/100)^2
psi_trafo = sum_m (max(0, loading_m - line_limit_pct)/100)^2
```

代码中 `line_limit_pct` 来自 `cfg.grid.line_max_loading_pct`，默认 100。电压惩罚分摊到 agent 时使用 agent 电压违约占比：

```text
w^v_i = v_violation_i / sum_j v_violation_j, if sum_j v_violation_j > 0
w^v_i = 0, otherwise

safe_v_i = N * w_voltage_pen * psi_v * w^v_i
safe_line_i = w_line_pen * psi_line
safe_trafo_i = w_trafo_pen * psi_trafo
```

## 十、环境奖励函数

环境奖励对每个 agent 单独计算。先定义实际进口电价：

```text
p^{imp}_t = p_t + import_price_markup_eur_per_kwh
```

静态电池套利收益：

```text
r^{bat}_{i,t} =
    (max(-P^b_{i,t},0) - max(P^b_{i,t},0)) * p^{imp}_t * Delta t
```

EV 充电成本与奖励：

```text
cost^{ev}_{i,t} = P^{ev}_{i,t} * p^{imp}_t * Delta t
r^{ev}_{i,t} = -cost^{ev}_{i,t}
```

静态电池边界动作惩罚：

```text
boundary_push_i =
    (P^b_{i,t} < 0 and SOC^b_{i,t} <= soc_min + epsilon)
    or
    (P^b_{i,t} > 0 and SOC^b_{i,t} >= soc_max - epsilon)

pen^{act}_{i,t} = w_action * abs(P^b_{i,t}) * 1(boundary_push_i)
```

静态电池 SoC 软边界正则：

```text
soft_low = soc_min + soc_boundary_margin
soft_high = soc_max - soc_boundary_margin

pen^{soc}_{i,t} =
    w_soc * (max(0, soft_low - SOC^b_{i,t})^2
             + max(0, SOC^b_{i,t} - soft_high)^2)
```

EV SoC 正则：

```text
pen^{ev,soc}_{i,t} =
    w_ev_soc * (max(0, ev_soc_min - SOC^{ev}_{i,t})^2
                + max(0, SOC^{ev}_{i,t} - ev_soc_max)^2)
```

EV hard projection 与 emergency 补电可额外计惩罚，当前默认权重为 0：

```text
pen^{ev,proj}_{i,t} = w_ev_proj * abs(projection_gap_i) * Delta t
pen^{ev,eme}_{i,t} = w_ev_eme * abs(emergency_added_kw_i) * Delta t
```

早期吞吐 bonus：

```text
progress = train_step_calls * num_envs / target_train_steps
w_throughput(t) = throughput_bonus_max * clip((0.80 - progress)/0.60, 0, 1)
bonus^{throughput}_{i,t} = w_throughput(t) * abs(P^b_{i,t}) * Delta t
```

最终 reward：

```text
r_{i,t} =
    r^{bat}_{i,t}
    + r^{ev}_{i,t}
    - pen^{act}_{i,t}
    - pen^{soc}_{i,t}
    - pen^{ev,soc}_{i,t}
    - pen^{ev,dep}_{i,t}
    - pen^{ev,proj}_{i,t}
    - pen^{ev,eme}_{i,t}
    + bonus^{throughput}_{i,t}
    - safe_v_i
    - safe_line_i
    - safe_trafo_i
```

三种 MADRL 主方案的区别主要来自 `safe_v/safe_line/safe_trafo` 权重和是否启用 joint safety projection：

| 控制器 | 安全惩罚权重 | 安全投影 |
| --- | --- | --- |
| `MADRL_BASE` | `(0,0,0)` | 否 |
| `MADRL_PENALTY` | `(400,0,10)` | 否 |
| `MADRL_PROJECTION` | `(400,0,10)` | 是 |

## 十一、MADRL 策略与 MATD3 训练建模

MADRL 使用 centralized training、decentralized actor 的结构。每个 agent 一个 actor：

```text
a_i = pi_i(o_i; theta_i)
```

actor 输入由 `actor_features` 拼接：

```text
actor_input_i =
    [madrl_local_i,
     wholesale_price_relative_seq,
     wholesale_price_spread_seq,
     load_seq_i,
     pv_seq_i]
```

critic 接收联合 observation 与联合 action：

```text
Q_i = Q_i(o_1,...,o_N, a_1,...,a_N; phi_i)
```

`Critic(twin=True)` 为每个 agent 维护 twin Q，用 TD3 的 clipped double Q 降低过估计。经验回放使用 n-step return。对样本 `(o_t, a_t, R^{(n)}_t, o_{t+n}, done)`，目标值为：

```text
a'_{t+n} = guard(pi_target(o_{t+n}) + clipped_noise)
y_i = R^{(n)}_{i,t} + gamma^n * (1-done) * min(Q'_{i,1}(o_{t+n},a'), Q'_{i,2}(o_{t+n},a'))
```

critic 损失：

```text
L_Q = MSE(Q_{i,1}(o_t,a_t), y_i) + MSE(Q_{i,2}(o_t,a_t), y_i)
```

actor 延迟更新。更新 agent `i` 时，只替换联合动作中的第 `i` 个动作，其它 agent 动作来自 batch：

```text
a^{policy}_i = guard(pi_i(o_i))
J_i(theta_i) = mean_batch Q_{i,1}(o_t, [a_1,...,a^{policy}_i,...,a_N])
L_actor = -J_i
```

目标网络软更新：

```text
theta'_i <- (1-tau) * theta'_i + tau * theta_i
phi'_i <- (1-tau) * phi'_i + tau * phi_i
```

`guard` 包含两层动作保护：

1. `map_actor_output_to_soc_feasible_action`：将 actor 原始输出映射到静态电池 SoC 可行区间；
2. 若启用 `JointGridSafetyProjector`，则对静态电池功率和 PV 弃光做联合安全投影；
3. `enforce_local_action_feasibility`：再次强制静态电池本地 SoC 可行。

## 十二、MADRL 联合安全投影

`JointGridSafetyProjector` 只投影静态电池功率和 PV 弃光，不投影 EV 充电。设投影变量为：

```text
x = [P^b_1,...,P^b_N, G^{curt}_1,...,G^{curt}_N]
```

本地边界：

```text
lower_i <= P^b_i <= upper_i
0 <= G^{curt}_i <= G_i
```

投影器先在零注入附近用有限差分估计电压和线路负载灵敏度：

```text
S^v_{k,i} = (V_k(+delta e_i) - V_k(-delta e_i)) / (2*delta)
S^l_{m,i} = (loading_m(+delta e_i) - loading_m(-delta e_i)) / (2*delta)
```

给定基础净负荷：

```text
base_net = load - pv_raw + ev_charge
```

构造近似线性安全约束：

```text
v_min + voltage_margin <= V_base + S^v * net_adjustment <= v_max - voltage_margin
line_loading_base + S^l * net_adjustment <= line_limit_pct - line_margin_pct
abs(sum(base_net + net_adjustment)) <= trafo_limit_kw * (1 - trafo_margin_pct/100)
```

其中 `net_adjustment` 来自电池功率与 PV 弃光的联合改变。代码使用逐行半空间投影迭代 `projection_iters` 次：

```text
if row * x > bound:
    x <- x - (row*x - bound) * row / ||row||^2
```

每轮后重新裁剪本地功率边界与弃光边界。

## 十三、LOCAL_MPC 建模

`LOCAL_MPC` 为每个 agent 独立建立滚动优化问题。决策变量包括：

```text
c_t >= 0          电池充电功率
d_t >= 0          电池放电功率
curt_t >= 0       PV 弃光功率
ev_t >= 0         EV 充电功率
E_t               静态电池能量
E^{ev}_t          EV 电池能量
grid_import_t >= 0
grid_export_t >= 0
grid_mode_t in {0,1}
```

静态电池能量约束：

```text
E_{t+1} = E_t + eta_b * c_t * Delta t - d_t * Delta t / eta_b
soc_min*C^b <= E_t <= soc_max*C^b
0 <= c_t,d_t <= P^{b,max}
```

EV 能量约束：

```text
E^{ev}_{t+1} = E^{ev}_t + eta_ev * ev_t * Delta t
ev_soc_min*C^{ev} <= E^{ev}_t <= ev_soc_max*C^{ev}
0 <= ev_t <= P^{ev,max}, only when EV connected
```

PV 弃光约束：

```text
0 <= curt_t <= G_t
```

局部功率平衡写成进口/出口拆分：

```text
grid_import_t - grid_export_t
    = (L_t - G_t) + curt_t + c_t - d_t + ev_t
```

二进制 `grid_mode_t` 防止同一步同时进口和出口。目标函数是局部购电成本、卖电收益、EV 充电成本和一个很小的电池吞吐 tie-breaker：

```text
min sum_t [
    p_t * Delta t * c_t
    - p_t * Delta t * d_t
    + p_t * Delta t * ev_t
    + eps_throughput * Delta t * (c_t + d_t)
] + w_ev_dep * gap_ev^2
```

`LOCAL_MPC` 不显式约束电压、线路或变压器，只通过环境执行后得到实际潮流后果。

## 十四、ADMM_MPC 建模

`ADMM_MPC` 将每个 agent 的局部优化和一个聚合变压器功率边界连接起来。每个 agent 的本地变量与 Local MPC 类似，但增加本地净负荷变量：

```text
n_{i,t} = L_{i,t} - G_{i,t} + curt_{i,t} + c_{i,t} - d_{i,t} + ev_{i,t}
```

协调变量使用贡献副本：

```text
q_{i,t} = alpha_{i,t} * n_{i,t}
```

当前代码中：

```text
alpha_{i,t} = 1
```

变压器聚合边界：

```text
-trafo_limit_kw <= sum_i q_{i,t} <= trafo_limit_kw
```

ADMM 迭代中，每个 agent 解带二次协调项的本地问题：

```text
min local_energy_cost_i
    + (rho/2) * ||alpha_i*n_i - z_i + u_i||_2^2
```

聚合投影步骤把所有 agent 的贡献副本投影到变压器边界：

```text
z <- projection({q_i + u_i}, lower=-trafo_limit_kw, upper=trafo_limit_kw)
u <- u + q - z
```

收敛残差：

```text
primal_residual = ||q - z||_2
dual_residual = rho * ||z_new - z_old||_2
```

若启用 residual balancing：

```text
if primal > mu * dual: rho <- rho * tau
if dual > mu * primal: rho <- rho / tau
```

最终第一步动作还会经过一个小型联合 QP，尽量接近 ADMM 得到的电池、弃光、EV 计划，同时满足第一步变压器聚合边界和本地能量边界。

## 十五、MISOCP 全局优化建模

`MISOCP` 是全局优化基线。它在 episode 开始时对完整剩余 horizon 建模，决策变量包括：

- 每个 agent 的电池充电 `c_{i,t}`、放电 `d_{i,t}`、PV 弃光 `curt_{i,t}`、EV 充电 `ev_{i,t}`；
- 每个 agent 的静态电池能量 `E_{i,t}` 和 EV 能量 `E^{ev}_{i,t}`；
- 每条支路的有功 `P_{l,t}`、无功 `Q_{l,t}`、电流平方 `l_{l,t}`；
- 每个节点的电压平方 `v_{b,t}`；
- 根节点进口/出口功率和二进制模式。

本地设备约束与前述 MPC 一致：

```text
E_{i,t+1} = E_{i,t} + eta_b*c_{i,t}*Delta t - d_{i,t}*Delta t/eta_b
E^{ev}_{i,t+1} = E^{ev}_{i,t} + eta_ev*ev_{i,t}*Delta t
0 <= curt_{i,t} <= G_{i,t}
0 <= c_{i,t}, d_{i,t} <= P^{b,max}_i
0 <= ev_{i,t} <= P^{ev,max}_i, only when connected
```

EV 到家重置：

```text
if t+1 is arrival step:
    E^{ev}_{i,t+1} = ev_arrival_soc * C^{ev}_i
```

EV 离家约束：

```text
soft:
    gap_i >= SOC^{ev,req} - E^{ev}_{i,dep}/C^{ev}_i
    objective += w_ev_dep * gap_i^2

hard or emergency:
    E^{ev}_{i,dep} >= SOC^{ev,req} * C^{ev}_i
```

全局电网模型使用径向 DistFlow/SOCP 近似。对支路 `l=(parent, child)`：

```text
P_{l,t} = sum_{k in children(child)} P_{k,t} + r_l * l_{l,t} + p^{inj}_{child,t}
Q_{l,t} = sum_{k in children(child)} Q_{k,t} + x_l * l_{l,t}
v_{child,t} = v_{parent,t}
             - 2*(r_l*P_{l,t} + x_l*Q_{l,t})
             + (r_l^2 + x_l^2)*l_{l,t}
```

其中 agent 注入项按 MVA 基准缩放：

```text
p^{inj}_{bus,t} =
    (L_{i,t} - G_{i,t} + curt_{i,t} + c_{i,t} - d_{i,t} + ev_{i,t})
    / (1000 * S_base)
```

二阶锥松弛为：

```text
P_{l,t}^2 + Q_{l,t}^2 <= v_{parent,t} * l_{l,t}
```

代码用等价 rotated cone 形式写入 Gurobi。电压和支路限额：

```text
v_min^2 <= v_{b,t} <= v_max^2
l_{line,t} <= line_limit
l_{trafo,t} <= trafo_limit
root_import_t <= S_base * loading_scale
root_export_t <= S_base * loading_scale
```

目标函数：

```text
min sum_{i,t} [
    p_t * Delta t * (c_{i,t} - d_{i,t})
    + p_t * Delta t * ev_{i,t}
    + 1e-4 * Delta t * (c_{i,t} + d_{i,t})
] + EV departure penalty if soft
```

`MISOCP` 使用 perfect forecast。控制器在 episode 首步求解完整计划，后续每步从计划中取当前 offset 的动作。

## 十六、评估指标与 rollout record

评估脚本对每个 controller 和 forecast mode 运行若干 episode，统计：

```text
episode_reward_mean
storage_profit_eur_mean
voltage_violation_steps_mean
min_vm_pu
max_vm_pu
trafo_loading_max_pct
trafo_overload_steps
act_time_s_mean
```

完整 rollout record 写入：

```text
results/<forecast_mode>/<controller>/record/
  step.parquet
  agent.parquet
  grid.parquet
  summary.parquet
  metrics.parquet
```

经济指标主要按以下关系计算：

```text
storage_charge_cost = max(P^b,0) * price * Delta t
storage_discharge_revenue = max(-P^b,0) * price * Delta t
storage_profit = storage_discharge_revenue - storage_charge_cost
storage_objective = -storage_profit
ev_cost = P^{ev} * price * Delta t
total_cost = -storage_profit + ev_cost + pv_cost
```

当前 PV cost 默认是 0，除非后续显式配置弃光价格或惩罚。

## 十七、目录职责

```text
configs/      Cfg dataclass 和所有实验参数
data/         原始 prosumer 数据读取与 share_data 构建
predictors/   LSTM 特征、训练、加载、评估与 artifact 保存
envs/         pandapower 电网核心、强化学习环境、多进程向量环境
models/       MADRL actor/critic 网络与 observation 拼接
controllers/  MADRL、Local MPC、ADMM MPC、MISOCP 控制器
scripts/      notebook 调用的实验 owner 函数
utils/        路径、价格、torch runtime、产物、rollout record 与绘图
notebooks/    主实验入口和 EV 场景 notebook
tests/        契约、smoke、encoding hygiene 与关键行为测试
```

## 十八、推荐实验流程

主线 notebook 顺序：

1. `notebooks/predict.ipynb`：训练 LSTM，写入 forecast artifacts，构建 `share_data`。
2. `notebooks/madrl_base.ipynb`：训练并评估 `MADRL_BASE`。
3. `notebooks/madrl_base_safe.ipynb`：训练并评估 `MADRL_PENALTY`。
4. `notebooks/madrl_projection_safe.ipynb`：训练并评估 `MADRL_PROJECTION`。
5. `notebooks/misocp.ipynb`：运行全局 `MISOCP` perfect-forecast 基线。
6. `notebooks/mpc.ipynb`：运行 `LOCAL_MPC` perfect/lstm 和 `ADMM_MPC` lstm 基线。
7. `notebooks/compare.ipynb`：读取缓存结果和 rollout record，生成对比表与图。

EV 扩展实验位于：

```text
notebooks/notebooks_EV/soft/
notebooks/notebooks_EV/hard/
notebooks/notebooks_EV/emergency_charging/
```

## 十九、程序化入口

notebook 外可用 Python 串起主线：

```python
from dataclasses import replace
from pathlib import Path

from configs.cfg import Cfg
from data.share_data import build_share_data, load_share_data
from predictors.lstm_training import train_forecasters
from scripts.compare import compare_all
from scripts.eval import eval_all
from scripts.madrl import SCHEMES, train_madrl_scheme
from utils.run_artifacts import create_run_dir, write_config_json

cfg = Cfg()
run_dir = create_run_dir(cfg)

forecast = train_forecasters(cfg, run_dir, overwrite=True)
cfg = replace(cfg, forecast=replace(cfg.forecast, lstm_artifact_dir=str(forecast["artifact_dir"])))
write_config_json(cfg, run_dir)

share_dir = build_share_data(cfg, run_dir, forecast["artifact_dir"], overwrite=True)
share_data = load_share_data(share_dir, cfg)

madrl_models = {}
for spec in SCHEMES:
    trained = train_madrl_scheme(cfg, run_dir, share_data, spec)
    madrl_models[str(spec["controller"])] = Path(trained["model_path"])

eval_all(cfg, madrl_models, run_dir, share_data=share_data)
compare_all(cfg, run_dir)
```

长实验默认使用 `cfg.runtime.device="cuda"`。若机器没有可用 GPU，应显式改为 CPU。仓库默认 Python 环境是 Conda 环境 `MADRL_ESS`。

## 二十、产物布局

默认运行目录：

```text
artifacts/runs/<timestamp>_<cfg_hash>/
```

核心产物：

```text
config.json
run.json
forecast/
  artifacts/
  tables/
  figures/
share_data/
  train.npz
  eval.npz
  manifest.json
models/
  madrl/<scheme>/model.pt
  madrl/<scheme>/meta.json
results/
  <forecast_mode>/<controller>/
    metrics.json
    traces.npz
    record/
      manifest.json
      meta.json
      step.parquet
      agent.parquet
      grid.parquet
      summary.parquet
      metrics.parquet
tables/
figures/
```

`metrics.json` 和 `traces.npz` 是轻量评估缓存；`record/` 是论文表格和图使用的完整 rollout 记录。

## 二十一、环境与依赖

推荐使用仓库约定的 Conda 环境：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_remake_smoke.py
```

核心依赖包括 PyTorch、NumPy、pandas、scikit-learn、pandapower、simbench、Gurobi、matplotlib、openpyxl、pytest 等。MPC 和 MISOCP 需要 `gurobipy` 和可用 Gurobi license；没有 Gurobi 时，预测、数据加载和部分环境测试仍可运行，但优化基线不可用。

## 二十二、测试

常用检查：

```powershell
conda run -n MADRL_ESS python -m pytest
conda run -n MADRL_ESS python -m pytest tests/test_encoding_hygiene.py
```

`test_remake_smoke.py` 用小配置跑预测、share_data、MADRL、eval、compare 的端到端冒烟流程。`test_encoding_hygiene.py` 检查 README、Markdown、Python 文件和 notebook 文本是否为 UTF-8、无 BOM、无明显乱码标记。

## 二十三、维护边界

新增功能时优先保持这些边界：

- 一个概念只有一个 owner：数据契约归 `data/`，环境动力学归 `envs/`，控制策略归 `controllers/`，实验编排归 `scripts/`。
- 不保留旧 schema、旧路径、旧 checkpoint 的兼容读取。
- 不做 sibling/latest 扫描，不用前缀模糊匹配 artifact。
- 配置从 `Cfg` 进入，产物从显式 `run_dir` 读取。
- wrapper 只有在保护真实边界时才保留。
- 合同失败要说明预期和实际，不隐藏数据、路径、solver 或 artifact 的缺失。

这套代码的价值在于研究链条明确：数据如何进入、预测如何形成、动作如何映射到物理功率、SoC 和 EV 如何演化、潮流如何反馈、电网和 EV 约束如何进入奖励或优化、结果如何落盘，都能从主线代码直接追到。
