# MADRL_ESS 中文建模说明

`MADRL_ESS` 是一个用于研究低压配电网中多 prosumer 储能、电动汽车充电和光伏弃光协同控制的实验项目。当前主线把数据读取、LSTM 预测、共享数据契约、pandapower 潮流环境、多智能体强化学习、MPC/ADMM/MISOCP 优化基线和结果记录整合在同一个可复现实验链中。

本文按当前代码实现重新整理整个项目，重点说明 EV 和静态电池的物理建模、数学公式、奖励项、`progress` 项和 `costweight` 项。公式使用 LaTeX 书写，便于直接阅读或复制到论文文档中。

## 1. 项目研究问题

项目研究对象是一个低压配电网中的 \(N=3\) 个 prosumer agent。默认 SimBench 网络为：

$$
\texttt{1-LV-rural1--0-sw}
$$

三个 agent 默认接入母线：

$$
\mathcal{B}_{agent}=(12,4,2)
$$

每个 agent 在每个 15 分钟时间步同时具有：

- 本地负荷；
- 共享光伏可用功率；
- 静态储能电池；
- EV 家庭充电负荷；
- 批发电价；
- 配电网电压、线路、变压器潮流约束。

控制器需要在每个时间步决定静态电池充放电、PV 利用率和 EV 充电功率。环境将这些决策转化为净负荷并写入 pandapower 网络进行潮流计算，然后把经济收益、EV 约束、电池约束和电网安全约束反馈为奖励或优化目标。

## 2. 主要符号

| 符号 | 代码对应 | 含义 |
| --- | --- | --- |
| \(i\in\{1,\dots,N\}\) | `agent_id` | prosumer/agent 索引 |
| \(t\) | `step` | episode 内离散时间步 |
| \(\Delta t\) | `cfg.env.dt_hours` | 时间步长，默认 \(0.25\) 小时 |
| \(T\) | `cfg.env.episode_steps` | 一天步数，默认 \(96\) |
| \(H\) | `cfg.obs.sequence_length` | 预测窗口长度，默认 \(49\) |
| \(p_t\) | `price` | 批发电价，单位 EUR/kWh |
| \(\tilde p_t\) | `storage_price` | 进口电价，\(\tilde p_t=p_t+\text{markup}\) |
| \(L_{i,t}\) | `load` | agent 负荷，单位 kW |
| \(G_{i,t}\) | `pv` | agent PV 可用功率，单位 kW |
| \(G^{use}_{i,t}\) | `pv_effective` | 实际利用 PV，单位 kW |
| \(G^{curt}_{i,t}\) | `pv_curtail` | 弃光功率，单位 kW |
| \(P^b_{i,t}\) | `charge_kw` / `e_bat` | 静态电池有符号功率，正充电，负放电 |
| \(P^{ev}_{i,t}\) | `ev_charge_kw` | EV 实际充电功率，单位 kW |
| \(P^{ev,rl}_{i,t}\) | `ev_charge_kw_rl` | actor/MPC 原始请求的 EV 充电功率 |
| \(SOC^b_{i,t}\) | `soc` | 静态电池 SoC |
| \(SOC^{ev}_{i,t}\) | `ev_soc` | EV SoC |
| \(C^b_i\) | `battery_capacity_kwh` | 静态电池容量，默认 100 kWh |
| \(C^{ev}_i\) | `ev_capacity_kwh` | EV 电池容量，默认 60 kWh |
| \(P^{b,\max}_i\) | `pmax` | 静态电池最大充放电功率 |
| \(P^{ev,\max}_i\) | `ev_pmax` | EV 家充最大功率，默认 11 kW |
| \(\eta_b\) | `efficiency` | 静态电池效率，默认 0.95 |
| \(\eta_{ev}\) | `ev_efficiency` | EV 充电效率，默认 0.95 |

当前默认 EV 参数为：

$$
SOC^{ev}_{arr}=0.30,\quad SOC^{ev}_{req}=0.90,\quad
t_{arr}=72,\quad t_{dep}=28
$$

即 EV 在 18:00 到家，次日 07:00 离家，跨午夜连接。

## 3. 数据建模

原始数据位于 `datasets/prosumer/`：

- `household.csv`
- `heatpump.csv`
- `pv_reference.csv`
- `price.csv`

`data.loader.load_prosumer_dataset` 负责将 CSV 转换为 `SeriesData`。时间戳先按 UTC 读取，再转换到 `Europe/Berlin`。

### 3.1 负荷建模

配置中 agent profile 默认为：

$$
(\text{SFH12}, \text{SFH18}, \text{SFH20})
$$

负荷由 household 和 heatpump 两类分量相加：

$$
L_{i,t}
= s^L_i\sum_{c\in\{\text{household},\text{heatpump}\}}L^{raw}_{c,i,t}
$$

其中 \(s^L_i\) 是 `cfg.data.load_scale`。默认：

$$
s^L_i=20,\quad i=1,2,3
$$

代码还保存了分量级序列：

$$
L^{component}_{c,i,t}=s^L_i L^{raw}_{c,i,t}
$$

这些分量用于按 component、按 agent 训练负荷 LSTM。

### 3.2 PV 建模

PV 来自 `pv_reference.csv` 中的参考列，默认 `ref_south`。如果没有指定 `pv_capacity_kw`，则每个 agent 使用同一条参考 PV，并按 `pv_scale` 缩放：

$$
G_{i,t}=s^G_i G^{ref}_{t}
$$

如果指定了 agent 级容量 \(C^{pv}_i\)，则代码按参考序列峰值归一化：

$$
G_{i,t}
=s^G_iG^{ref}_{t}\frac{C^{pv}_i}{\max_t G^{ref}_{t}}
$$

当前默认 `pv_capacity_kw=()`，因此环境中 PV 是共享序列，再复制到每个 agent。

### 3.3 电价建模

批发电价为：

$$
p_t
$$

环境奖励和优化器中使用进口电价：

$$
\tilde p_t=p_t+\mu_{import}
$$

其中 \(\mu_{import}\) 对应 `cfg.reward.import_price_markup_eur_per_kwh`，当前默认 0。

## 4. LSTM 预测与 share_data 契约

项目支持两种预测模式：

- `perfect`：未来窗口直接取真实序列；
- `lstm`：未来窗口由训练好的 LSTM 产生。

对于任意一条标量时间序列 \(x_t\)，LSTM 训练样本为：

$$
X_k=(x_k,x_{k+1},\dots,x_{k+h-1})
$$

$$
y_k=(x_{k+h},x_{k+h+1},\dots,x_{k+h+H-1})
$$

其中：

$$
h=\texttt{cfg.forecast.history_window},\quad
H=\texttt{cfg.obs.sequence_length}
$$

模型形式为：

$$
\hat y_k=f_{\theta}(X_k,\phi_k)
$$

\(\phi_k\) 是可选时间特征。负荷和 PV 使用小时、周/年周期特征；价格默认不使用时间特征。

负荷预测按 `(component, agent)` 训练：

$$
\hat L_{c,i,t:t+H-1}=f^{load}_{c,i,\theta}(L_{c,i,t-h:t-1},\phi_t)
$$

负荷一步预测还带 baseline blend：

$$
\hat x^{blend}_{t+1}=x_t+\alpha(\hat x^{raw}_{t+1}-x_t)
$$

\(\alpha\) 从 `load_blend_candidates` 中按验证集 MAE 选择。heatpump 分量当前固定 \(\alpha=0\)，即使用 last-value baseline。

`data/share_data.py` 将预测和真实序列写入：

```text
share_data/
  train.npz
  eval.npz
  manifest.json
```

核心数组包括：

$$
\text{price}[e,t],\quad
\text{load}[e,t,i],\quad
\text{pv}[e,t]
$$

$$
\text{perfect\_price\_seq}[e,t,h],\quad
\text{perfect\_load\_seq}[e,t,h,i],\quad
\text{perfect\_pv\_seq}[e,t,h]
$$

$$
\text{lstm\_price\_seq}[e,t,h],\quad
\text{lstm\_load\_seq}[e,t,h,i],\quad
\text{lstm\_pv\_seq}[e,t,h]
$$

其中 \(e\) 是 episode/window 索引，\(t\) 是 episode 内步，\(h\) 是预测窗口偏移。

## 5. 环境状态与观测

`GridEnv` 的内部状态包括：

$$
s_t=\{
e,t,SOC^b_{1:N,t},SOC^{ev}_{1:N,t},
L_{1:N,t},G_{1:N,t},p_t,\hat x_{t:t+H-1},
\text{grid state}
\}
$$

reset 时：

- 训练模式：\(SOC^b_{i,0}\sim U(\texttt{train\_init\_soc\_low},\texttt{train\_init\_soc\_high})\)；
- 评估模式：\(SOC^b_{i,0}=\texttt{init\_soc}\)；
- EV：\(SOC^{ev}_{i,0}=\texttt{ev\_arrival\_soc}\)。

返回 observation 时包含多种视图：

| observation key | 内容 | 用途 |
| --- | --- | --- |
| `local` | \([SOC^b,SOC^{ev},A^{ev},L,G,p]\) | 普通控制器窗口 |
| `sequence` | \([\hat p,\hat L,\hat G]\) | MPC/MISOCP 窗口 |
| `madrl_local` | 时间周期、归一化 SoC、EV 可用性 | MADRL actor/critic |
| `safety_local` | \([SOC^b,SOC^{ev},A^{ev},L,G,C^b,P^{b,\max},C^{ev},P^{ev,\max}]\) | 本地动作可行性和安全投影 |
| `load_seq` | robust-tanh 归一化负荷预测 | MADRL |
| `pv_seq` | PV 缩放预测 | MADRL |
| `wholesale_price_relative_seq` | 价格窗口相对归一化 | MADRL |
| `wholesale_price_spread_seq` | 价格窗口价差强度 | MADRL |

时间特征为：

$$
\phi^{hour}_t=
\left[
\sin\frac{2\pi hour_t}{24},
\cos\frac{2\pi hour_t}{24}
\right]
$$

$$
\phi^{year}_t=
\left[
\sin\frac{2\pi doy_t}{365.25},
\cos\frac{2\pi doy_t}{365.25}
\right]
$$

SoC 归一化为：

$$
\bar{SOC}^b_{i,t}
=\operatorname{clip}
\left(
2\frac{SOC^b_{i,t}-SOC^b_{\min}}{SOC^b_{\max}-SOC^b_{\min}}-1,
-1,1
\right)
$$

$$
\bar{SOC}^{ev}_{i,t}
=\operatorname{clip}
\left(
2\frac{SOC^{ev}_{i,t}-SOC^{ev}_{\min}}{SOC^{ev}_{\max}-SOC^{ev}_{\min}}-1,
-1,1
\right)
$$

## 6. 动作空间和物理映射

当前 `cfg.model.action_dim=3`。每个 agent 的动作是：

$$
a_{i,t}=
\left[
a^b_{i,t},
a^{pv}_{i,t},
a^{ev}_{i,t}
\right]\in[-1,1]^3
$$

### 6.1 静态电池动作

actor 输出的静态电池动作先按最大功率映射：

$$
P^b_{i,t}=a^b_{i,t}P^{b,\max}_i
$$

其中：

$$
P^{b,\max}_i=C^b_i\cdot r^{b,\max}
$$

`max_charge_rate` 当前为 0.5，因此 100 kWh 电池对应 50 kW 最大充放电功率。

环境会根据当前 SoC 裁剪动作，防止下一步越界。可行动作上下界为：

$$
a^{b,upper}_{i,t}
=\frac{(SOC^b_{\max}-SOC^b_{i,t})C^b_i}
{P^{b,\max}_i\Delta t\eta_b}
$$

$$
a^{b,lower}_{i,t}
=\frac{(SOC^b_{\min}-SOC^b_{i,t})\eta_b C^b_i}
{P^{b,\max}_i\Delta t}
$$

最终：

$$
a^b_{i,t}\leftarrow
\operatorname{clip}
\left(
a^b_{i,t},
\max(a^{b,lower}_{i,t},-1),
\min(a^{b,upper}_{i,t},1)
\right)
$$

物理意义：当电池接近上边界时限制继续充电，当电池接近下边界时限制继续放电。

### 6.2 PV 动作

PV 动作表示利用率：

$$
u^{pv}_{i,t}
=\operatorname{clip}\left(\frac{a^{pv}_{i,t}+1}{2},0,1\right)
$$

$$
G^{use}_{i,t}=u^{pv}_{i,t}G_{i,t}
$$

$$
G^{curt}_{i,t}=G_{i,t}-G^{use}_{i,t}
$$

物理意义：\(a^{pv}=-1\) 表示全弃光，\(a^{pv}=1\) 表示全利用。

### 6.3 EV 动作

先定义 EV 连接标记：

$$
A^{ev}_t=
\begin{cases}
1,& \text{EV connected}\\
0,& \text{otherwise}
\end{cases}
$$

原始 EV 充电请求为：

$$
P^{ev,rl}_{i,t}
=A^{ev}_t
\operatorname{clip}\left(\frac{a^{ev}_{i,t}+1}{2},0,1\right)
P^{ev,\max}_i
$$

如果当前模式是 `soft`，则：

$$
P^{ev}_{i,t}=P^{ev,rl}_{i,t}
$$

如果当前模式是 `hard` 或 `emergency`，环境可能进一步修正 \(P^{ev}_{i,t}\)，见第 8 节。

### 6.4 净负荷

环境最终写入电网的 agent 净负荷为：

$$
P^{net}_{i,t}
=L_{i,t}-G^{use}_{i,t}+P^b_{i,t}+P^{ev}_{i,t}
$$

当 \(P^{net}_{i,t}\ge 0\) 时，agent 表现为负荷；当 \(P^{net}_{i,t}<0\) 时，agent 表现为并网发电。

## 7. 静态电池建模

静态电池功率符号约定：

- \(P^b_{i,t}>0\)：充电；
- \(P^b_{i,t}<0\)：放电。

SoC 动力学为：

$$
SOC^b_{i,t+1}=
\begin{cases}
SOC^b_{i,t}
\dfrac{P^b_{i,t}\Delta t\eta_b}{C^b_i},
&P^b_{i,t}\ge 0\\
SOC^b_{i,t}
\dfrac{P^b_{i,t}\Delta t}{\eta_b C^b_i},
&P^b_{i,t}<0
\end{cases}
$$

然后硬裁剪：

$$
SOC^b_{i,t+1}
\leftarrow
\operatorname{clip}(SOC^b_{i,t+1},SOC^b_{\min},SOC^b_{\max})
$$

物理意义：充电时只有 \(\eta_b\) 比例的电能进入电池；放电时为了向外输出同样电能，电池内部能量下降更多。

### 7.1 静态电池经济收益

静态电池的即时收益为：

$$
r^{bat}_{i,t}
=
\left[
\max(-P^b_{i,t},0)
-
\max(P^b_{i,t},0)
\right]
\tilde p_t\Delta t
$$

放电收益为正，充电成本为负。

### 7.2 静态电池边界惩罚

若电池已经接近下边界仍放电，或接近上边界仍充电，则触发边界动作惩罚：

$$
\mathbb{I}^{bd}_{i,t}
=
\mathbb{I}
\left[
P^b_{i,t}<0
\land SOC^b_{i,t}\le SOC^b_{\min}+\epsilon
\right]
+
\mathbb{I}
\left[
P^b_{i,t}>0
\land SOC^b_{i,t}\ge SOC^b_{\max}-\epsilon
\right]
$$

$$
\ell^{act}_{i,t}
=w_{act}|P^b_{i,t}|\mathbb{I}^{bd}_{i,t}
$$

其中 \(\epsilon=\texttt{soc\_boundary\_epsilon}\)。

### 7.3 静态电池 SoC 软边界正则

定义软边界：

$$
SOC^{soft}_{low}=SOC^b_{\min}+m,\quad
SOC^{soft}_{high}=SOC^b_{\max}-m
$$

其中 \(m=\texttt{soc\_boundary\_margin}\)。正则为：

$$
\ell^{soc}_{i,t}
=w_{soc}
\left[
\max(0,SOC^{soft}_{low}-SOC^b_{i,t})^2
+
\max(0,SOC^b_{i,t}-SOC^{soft}_{high})^2
\right]
$$

物理意义：即使没有真正越界，也鼓励电池远离硬边界，避免训练时长期卡在不可操作区。

### 7.4 训练早期电池吞吐 bonus

训练中还存在一个早期探索用的电池吞吐 bonus。训练进度为：

$$
\rho^{train}_t
=
\operatorname{clip}
\left(
\frac{(\texttt{train\_step\_calls}+1)\cdot \texttt{num\_envs}}
{\texttt{train\_episodes}\cdot \texttt{episode\_length}},
0,1
\right)
$$

吞吐权重随训练进度衰减：

$$
w^{throughput}_t
=
w^{throughput}_{max}
\operatorname{clip}
\left(
\frac{0.80-\rho^{train}_t}{0.60},
0,1
\right)
$$

bonus 为：

$$
r^{throughput}_{i,t}
=
w^{throughput}_t |P^b_{i,t}|\Delta t
$$

物理意义：训练初期鼓励电池真的发生充放电，减少 actor 学到“完全不动”的策略。后期该 bonus 自动衰减。

## 8. EV 建模

EV 默认启用，具有独立的容量、SoC 边界、到家 SoC、离家目标 SoC 和充电功率上限。

### 8.1 EV 连接窗口

默认：

$$
t_{arr}=72,\quad t_{dep}=28,\quad T=96
$$

由于 \(t_{arr}>t_{dep}\)，连接窗口跨午夜。连接标记为：

$$
A^{ev}_t=
\begin{cases}
1,& t_{day}\ge t_{arr}\ \text{or}\ t_{day}<t_{dep}\\
0,& \text{otherwise}
\end{cases}
$$

其中：

$$
t_{day}=t\bmod T
$$

物理意义：EV 晚上 18:00 回家并接入，次日 07:00 离家断开。

### 8.2 到家 SoC 重置

到家步重置 EV SoC：

$$
SOC^{ev}_{i,t}\leftarrow SOC^{ev}_{arr}
\quad \text{if } t_{day}=t_{arr}
$$

当前默认：

$$
SOC^{ev}_{arr}=0.30
$$

物理意义：每次车辆回家时，假设路上消耗后剩余 30% 电量。

### 8.3 EV SoC 动力学

EV 只充电，不建模 V2G 放电。SoC 更新为：

$$
SOC^{ev}_{i,t+1}
=
\operatorname{clip}
\left(
SOC^{ev}_{i,t}
\frac{P^{ev}_{i,t}\Delta t\eta_{ev}}{C^{ev}_i},
SOC^{ev}_{\min},
SOC^{ev}_{\max}
\right)
$$

物理意义：EV 家充功率经过充电效率转化为电池能量增量。

### 8.4 离家 SoC 约束

离家目标为：

$$
SOC^{ev}_{req}=0.90
$$

离家步缺口为：

$$
g^{dep}_{i,t}
=
\begin{cases}
\max(0,SOC^{ev}_{req}-SOC^{ev}_{i,t+1}),
& t_{day}=t_{dep}\\
0,& \text{otherwise}
\end{cases}
$$

soft 模式下离家惩罚为：

$$
\ell^{dep}_{i,t}
=
w_{dep}^{ev}
\left(g^{dep}_{i,t}\right)^2
$$

当前 EV soft 实验 notebook 中常使用：

$$
w_{dep}^{ev}=1000
$$

### 8.5 EV progress 线性目标轨迹

这是当前项目新增的 `progress` 项，只在以下条件同时满足时启用：

$$
\text{mode}=\text{soft},\quad A^{ev}_t=1,\quad w^{ev}_{prog}>0
$$

代码函数为 `ev_progress_target_soc(cfg, step)`。

先定义 EV 连接窗口内进度 \(\alpha_t\)。若 EV 未连接：

$$
\alpha_t=0
$$

若连接且跨午夜，即 \(t_{arr}>t_{dep}\)，总连接步数为：

$$
T^{conn}=T-t_{arr}+t_{dep}
$$

已连接步数为：

$$
\tau_t=
\begin{cases}
t_{day}-t_{arr},&t_{day}\ge t_{arr}\\
T-t_{arr}+t_{day},&t_{day}<t_{arr}
\end{cases}
$$

进度为：

$$
\alpha_t=
\operatorname{clip}
\left(
\frac{\tau_t}{\max(T^{conn},1)},
0,1
\right)
$$

EV progress 目标 SoC 是从到家 SoC 到离家目标 SoC 的线性插值：

$$
SOC^{ev,prog}_t
=
SOC^{ev}_{arr}
+
\alpha_t
\left(
SOC^{ev}_{req}-SOC^{ev}_{arr}
\right)
$$

progress 缺口为：

$$
g^{prog}_{i,t}
=
\max(0,SOC^{ev,prog}_t-SOC^{ev}_{i,t+1})
$$

progress 惩罚为：

$$
\ell^{prog}_{i,t}
=
w^{ev}_{prog}
\left(g^{prog}_{i,t}\right)^2
$$

物理意义：离家惩罚只在最后一步出现，学习信号很稀疏；progress 项把“应该逐步充到目标”的要求分摊到整个夜间连接窗口。它不是硬约束，而是一个软引导，使 EV SoC 不要拖到临近离家才补电。

当前新增 notebook `madrl_base_EV_soft _progress.ipynb` 中设置：

$$
w^{ev}_{prog}=50
$$

同时使用：

$$
SOC^{ev}_{arr}=0.30,\quad w^{ev}_{dep}=1000,\quad
w^{throughput}_{max}=0.001
$$

### 8.6 EV charging cost weight

这是当前项目新增的 `costweight` 项，对应：

```text
cfg.reward.ev_charging_cost_weight
```

它只在 EV soft 模式下改变环境奖励中的 EV 充电成本：

$$
w^{ev}_{cost}
=
\begin{cases}
\texttt{ev\_charging\_cost\_weight},
& \text{mode}=\text{soft}\\
1,
& \text{mode}\in\{\text{hard},\text{emergency}\}
\end{cases}
$$

EV 充电成本为：

$$
c^{ev}_{i,t}
=
w^{ev}_{cost}P^{ev}_{i,t}\tilde p_t\Delta t
$$

EV reward 为：

$$
r^{ev}_{i,t}=-c^{ev}_{i,t}
$$

物理意义：`costweight` 不是物理功率约束，而是经济偏好系数。增大它会让 RL 更讨厌 EV 充电成本，倾向于把 EV 充电安排到更便宜的时段，但仍需满足离家 SoC 惩罚或 progress 惩罚。

当前新增 notebook `madrl_base_EV_soft_costweight.ipynb` 中设置：

$$
w^{ev}_{cost}=1.5,\quad w^{ev}_{dep}=1000
$$

注意：Local MPC、ADMM MPC 和 MISOCP 当前优化目标中直接使用真实进口电价 \( \tilde p_t \)，没有乘这个 RL reward 里的 `ev_charging_cost_weight`。因此 costweight 是 MADRL soft EV 奖励实验变体，不是所有优化基线的共同物理模型参数。

### 8.6.1 EV price-aware high-price charging penalty

这是当前项目新增的 EV 高电价充电 reward shaping 项，对应：

```text
cfg.reward.ev_price_aware_penalty_weight
cfg.reward.ev_price_aware_threshold_quantile
```

该项只用于 MADRL 奖励塑形，不进入物理 total cost，不改变 EV SoC 动力学，不改变 hard projection、emergency charging、headroom clipping、MPC 或 MISOCP。

当前步使用含进口加价的有效电价：

$$
\tilde p_t
=
p_t+\mu_{\mathrm{import}}
$$

对当前训练或评估窗口内的价格序列

$$
\mathcal P_e=\{p_\tau\}_{\tau=0}^{T_e-1}
$$

取高价阈值：

$$
p^{\mathrm{thr}}_e
=
Q_q(\mathcal P_e),
\qquad
q=\texttt{ev\_price\_aware\_threshold\_quantile}
$$

其中 \(Q_q(\cdot)\) 表示 \(q\) 分位数。当前实验使用：

$$
q=0.70
$$

只惩罚超过阈值的高价部分：

$$
\Delta p^+_{e,t}
=
\max\!\left(0,\tilde p_t-p^{\mathrm{thr}}_e\right)
$$

归一化高价超额为：

$$
\bar{\Delta p}_{e,t}
=
\frac{
\max\!\left(0,\tilde p_t-p^{\mathrm{thr}}_e\right)
}{
\max\!\left(p^{\mathrm{thr}}_e,\epsilon\right)
},
\qquad
\epsilon=10^{-6}
$$

设最终实际执行的 EV 充电功率为 \(P^{ev,exe}_{i,t}\)。这里的 \(P^{ev,exe}_{i,t}\) 是经过 hard projection、emergency charging 和 SoC headroom clipping 之后真正进入 net load、EV SoC update 和 EV charging cost 的功率。price-aware penalty 定义为：

$$
\ell^{ev,price}_{i,t}
=
w^{ev}_{price}
P^{ev,exe}_{i,t}
\bar{\Delta p}_{e,t}
\Delta t
$$

其中：

$$
w^{ev}_{price}
=
\texttt{ev\_price\_aware\_penalty\_weight}
$$

因此：

$$
\ell^{ev,price}_{i,t}=0
\quad
\text{if}
\quad
\tilde p_t\le p^{\mathrm{thr}}_e
$$

而在高价时段：

$$
\ell^{ev,price}_{i,t}>0
\quad
\text{if}
\quad
\tilde p_t>p^{\mathrm{thr}}_e
\ \text{and}\
P^{ev,exe}_{i,t}>0
$$

EV 基础充电成本仍为：

$$
c^{ev}_{i,t}
=
w^{ev}_{cost}
P^{ev,exe}_{i,t}
\tilde p_t
\Delta t
$$

price-aware 项只是在 reward 中额外扣除：

$$
r_{i,t}
=
r^{storage}_{i,t}
-c^{ev}_{i,t}
-\ell^{action}_{i,t}
-\ell^{bat,soc}_{i,t}
-\ell^{ev,soc}_{i,t}
-\ell^{ev,dep}_{i,t}
-\ell^{ev,over}_{i,t}
-\ell^{ev,prog}_{i,t}
-\ell^{ev,price}_{i,t}
-\ell^{ev,proj}_{i,t}
-\ell^{ev,eme}_{i,t}
+r^{throughput}_{i,t}
-\ell^V_{i,t}
-\ell^{line}_{i,t}
-\ell^{trafo}_{i,t}
$$

真实 total cost 不包含该 reward shaping 项，仍然定义为：

$$
C^{total}_t
=
\sum_i
\left(
C^{bat}_{i,t}
+
C^{ev}_{i,t}
+
C^{pv}_{i,t}
\right)
$$

因此：

$$
\ell^{ev,price}_{i,t}
\notin
C^{total}_t
$$

当前 `madrl_projection_safe_EV_hard costweight.ipynb` 已改为 price-aware 实验设置：

$$
w^{ev}_{cost}=1.0,
\qquad
w^{ev}_{price}=0.05,
\qquad
q=0.70
$$

该实验的建模含义是：不再整体放大所有 EV 充电成本，而是只在高电价区间对 EV 充电增加额外奖励惩罚，从而鼓励 EV 在满足 hard departure constraint 的前提下避开高价时段。

### 8.7 EV SoC 正则

EV SoC 正则为：

$$
\ell^{ev,soc}_{i,t}
=
w^{ev}_{soc}
\left[
\max(0,SOC^{ev}_{\min}-SOC^{ev}_{i,t})^2
+
\max(0,SOC^{ev}_{i,t}-SOC^{ev}_{\max})^2
\right]
$$

由于 SoC 更新后已经 clip，这一项主要是额外的边界行为提示。

### 8.8 EV hard projection

hard 模式用于把 EV 离家 SoC 要求从“奖励中的软惩罚”提升为“环境执行层的可行性投影”。它的目标不是让策略自己承担所有离家失败惩罚，而是在每个连接时间步检查：如果当前少充电会导致后续即使满功率充电也达不到离家目标，则环境立即把 EV 充电功率补到最低可行值。

代码入口为 `project_ev_action_to_departure_soc`。该逻辑只有在以下条件同时满足时才会执行：

- `cfg.env.ev_enabled=True`；
- 当前时间步 EV 已连接，即 \(A^{ev}_t=1\)；
- 当前步不是离家步；
- `cfg.env.ev_departure_constraint_mode="hard"`；
- `cfg.env.ev_hard_projection_enabled=True`。

因此，`ev_departure_constraint_mode="hard"` 表示选择 hard 约束语义，而 `ev_hard_projection_enabled=True` 表示实际启用投影执行器。当前 hard 实验 notebook 中会显式设置这两个配置。默认配置里 `ev_hard_projection_enabled=False`，避免普通 soft 实验被意外改写动作。

#### 8.8.1 剩余可充电时间

从当前步 \(t\) 开始，环境先计算直到离家步之前仍然连接的未来步数：

$$
K_t=\text{future connected steps before departure}
$$

这里的 \(K_t\) 由 `count_future_connected_steps_until_departure(cfg, step)` 得到。注意它表示“当前步之后、离家前还剩多少个连接充电步”，不包括当前步本身。因为 hard 投影正在决定当前步该至少充多少电，当前步的功率单独由下面的 \(P^{ev,req}_{i,t}\) 表示。

若 EV 连接窗口跨午夜，例如当前项目默认：

$$
t^{arr}=72,\qquad t^{dep}=28,\qquad N_{day}=96
$$

则计数会按日内步循环处理。物理意义是：EV 晚上 18:00 接入，次日 07:00 离开；hard 规则始终看“从现在到离家之前还剩多少次机会可以充电”。

#### 8.8.2 未来最大可补 SoC

若当前步之后的所有剩余连接步都以最大家充功率充电，则未来最多还能增加的 SoC 为：

$$
\Delta SOC^{ev,max}_{i,t}
=
\frac{K_tP^{ev,\max}_i\Delta t\eta_{ev}}{C^{ev}_i}
$$

其中：

- \(K_t\)：当前步之后到离家前的剩余连接步数；
- \(P^{ev,\max}_i\)：第 \(i\) 个家庭 EV 家充桩最大功率；
- \(\Delta t\)：单个仿真步长，当前为 \(0.25\) 小时；
- \(\eta_{ev}\)：EV 充电效率；
- \(C^{ev}_i\)：EV 电池容量。

物理意义：\(\Delta SOC^{ev,max}_{i,t}\) 是“如果从下一步开始全力补救，理论上最多还能补多少电”。它是 hard 约束判断当前是否还能拖延的核心。

#### 8.8.3 当前步最低必要充电功率

离家目标为 \(SOC^{ev}_{req}\)。如果当前 EV SoC 是 \(SOC^{ev}_{i,t}\)，未来最多还能补 \(\Delta SOC^{ev,max}_{i,t}\)，那么当前步必须至少补上的 SoC 缺口为：

$$
G^{hard}_{i,t}
=
\max
\left(
0,
SOC^{ev}_{req}
-SOC^{ev}_{i,t}
-\Delta SOC^{ev,max}_{i,t}
\right)
$$

把这个 SoC 缺口换算为当前步最低必要充电功率：

$$
P^{ev,req}_{i,t}
=
\max
\left(
0,
\frac{
\left(
SOC^{ev}_{req}
-\Delta SOC^{ev,max}_{i,t}
-SOC^{ev}_{i,t}
\right)C^{ev}_i
}
{\Delta t\eta_{ev}}
\right)
$$

也就是代码中的：

```python
required_min_charge_kw = max(
    0,
    (ev_departure_soc_req - max_future_soc_gain - ev_soc) * ev_cap / (dt * ev_efficiency),
)
```

物理意义：如果 \(P^{ev,req}_{i,t}=0\)，说明当前可以不充或少充，因为后面还有足够时间追上离家目标；如果 \(P^{ev,req}_{i,t}>0\)，说明当前已经不能继续完全听从低充电请求，否则即便未来满功率也可能无法达标。

#### 8.8.4 原始动作到 hard 可行动作的投影

actor 或控制器给出的原始 EV 充电请求为：

$$
P^{ev,rl}_{i,t}
=
A^{ev}_t
\operatorname{clip}
\left(
\frac{a^{ev}_{i,t}+1}{2},
0,
1
\right)
P^{ev,\max}_i
$$

hard 投影后的实际 EV 充电功率为：

$$
P^{ev}_{i,t}
=
\operatorname{clip}
\left(
\max(P^{ev,rl}_{i,t},P^{ev,req}_{i,t}),
0,
P^{ev,\max}_i
\right)
$$

这条公式有三层含义：

- \(\max(P^{ev,rl}_{i,t},P^{ev,req}_{i,t})\)：环境尊重策略的更高充电请求，但不允许它低于当前最低必要功率；
- 下界 \(0\)：EV 不建模 V2G，不能反向放电；
- 上界 \(P^{ev,\max}_i\)：家充桩物理功率上限。

投影增加的功率为：

$$
\Delta P^{ev,proj}_{i,t}
=
\max(0,P^{ev}_{i,t}-P^{ev,rl}_{i,t})
$$

对应记录字段为 `ev_projection_gap_kw`。若该值为正，说明环境替策略额外补了电，原因不是经济最优，而是离家可行性要求。

#### 8.8.5 不可行判定

如果当前步最低必要功率已经超过家充桩物理上限：

$$
P^{ev,req}_{i,t}>P^{ev,\max}_i
$$

则记录：

$$
\texttt{ev\_hard\_infeasible}_{i,t}=\text{True}
$$

同时实际功率仍会被 clip 到：

$$
P^{ev}_{i,t}\le P^{ev,\max}_i
$$

物理意义：不可行标记并不代表代码会突破充电桩功率上限，而是说明在当前 SoC、剩余时间和最大功率条件下，系统已经没有足够物理能力保证离家 SoC 目标。此时 hard 投影只能“尽力满充”，不能创造额外时间或额外功率。

#### 8.8.6 hard 模式下的 EV SoC 更新

投影后的 \(P^{ev}_{i,t}\) 会进入真实 EV SoC 动力学：

$$
SOC^{ev}_{i,t+1}
=
\operatorname{clip}
\left(
SOC^{ev}_{i,t}
+
\frac{P^{ev}_{i,t}\Delta t\eta_{ev}}{C^{ev}_i},
SOC^{ev}_{\min},
SOC^{ev}_{\max}
\right)
$$

因此 hard 模式改变的不是 SoC 方程本身，而是进入 SoC 方程的实际充电功率。它把动作空间中的 EV 维度从“任意 \(0\sim P^{ev,\max}\) 请求”改成“不得低于当前最低必要功率”的状态相关可行集合：

$$
\mathcal{U}^{ev,hard}_{i,t}
=
\left[
\min(P^{ev,req}_{i,t},P^{ev,\max}_i),
P^{ev,\max}_i
\right]
$$

当 \(P^{ev,req}_{i,t}=0\) 时，hard 可行集合退化为普通物理功率范围：

$$
\mathcal{U}^{ev,hard}_{i,t}=[0,P^{ev,\max}_i]
$$

当 \(P^{ev,req}_{i,t}>0\) 时，低于该值的 EV 动作会被环境抬高。

#### 8.8.7 hard 模式与 reward 的关系

hard 模式下 EV 充电成本仍按真实功率计入：

$$
C^{ev}_{i,t}=P^{ev}_{i,t}\tilde p_t\Delta t
$$

其中 hard 和 emergency 模式不会使用 soft 里的 `ev_charging_cost_weight`：

$$
w^{ev}_{cost}=
\begin{cases}
\texttt{ev\_charging\_cost\_weight},& \text{soft}\\
1,& \text{hard or emergency}
\end{cases}
$$

所以 hard 模式的经济含义是：如果为了保证离家可行性必须多充电，这部分充电会以真实电价进入成本。策略可以通过更早在低价时段主动充电来减少后续被 hard 投影强行补电的概率，但一旦进入不可拖延状态，环境会优先保证物理可行性。

投影惩罚项为：

$$
R^{ev,proj}_{i,t}
=
w^{ev}_{proj}
\left|
\Delta P^{ev,proj}_{i,t}
\right|
\Delta t
$$

在总奖励中以负项扣除：

$$
r_{i,t}\leftarrow r_{i,t}-R^{ev,proj}_{i,t}
$$

当前默认 `ev_projection_penalty_weight=0.0`，因此 hard 投影通常主要作为执行层约束和诊断字段存在，而不是额外奖励惩罚。记录字段包括：

- `ev_required_min_charge_kw`：当前步最低必要充电功率；
- `ev_projection_gap_kw`：环境相对原始请求额外增加的功率；
- `ev_hard_infeasible`：是否已物理不可行；
- `ev_charge_kw_rl`：投影前请求；
- `ev_charge_kw`：投影后实际执行功率。

#### 8.8.8 hard 与 soft/progress/emergency/safety projector 的区别

soft 模式只在离家步计算：

$$
w^{ev}_{dep}
\max(0,SOC^{ev}_{req}-SOC^{ev}_{i,t+1})^2
$$

它允许策略失败，只是失败会被扣分。progress 项仍属于 soft reward shaping，它在连接窗口内给出线性目标轨迹，但不会改写动作：

$$
SOC^{ev,prog}_t
=
SOC^{ev}_{arr}
+
\alpha_t(SOC^{ev}_{req}-SOC^{ev}_{arr})
$$

hard 模式不同：它直接改写低于最低必要功率的 EV 动作，使未来离家约束尽量保持可行。emergency 模式则只在离家前短窗口内补救，更短视；hard 模式从整个连接窗口持续检查未来可行性。

还要注意，`JointGridSafetyProjector` 只投影静态电池功率和 PV 弃光，不投影 EV 动作。EV hard projection 在环境 `GridEnv.step` 中独立执行，并且 EV 充电功率会作为外生净负荷进入电网安全投影和潮流：

$$
P^{net}_{i,t}
=
L_{i,t}
-P^{pv,eff}_{i,t}
+P^{bat}_{i,t}
+P^{ev}_{i,t}
$$

总结起来，EV hard 模型的物理意义是：家充 EV 必须在有限连接时间、有限充电功率、有限电池容量和充电效率下达到离家 SoC。hard projection 用一个逐步可行性约束把“不能错过充电时机”编码进环境执行层，防止学习策略仅依赖末端惩罚而把充电拖延到物理上已经无法完成的时间点。

### 8.9 EV emergency charging

emergency 模式只在离家前的应急窗口内启用。应急窗口步数为：

$$
K^{eme}=
\left\lceil
\frac{\texttt{ev\_emergency\_window\_hours}}{\Delta t}
\right\rceil
$$

当策略为 `required_power` 时：

$$
E^{need}_{i,t}
=
\max(0,SOC^{ev}_{req}-SOC^{ev}_{i,t})C^{ev}_i
$$

$$
P^{ev,req}_{i,t}
=
\operatorname{clip}
\left(
\frac{E^{need}_{i,t}}{\eta_{ev}K^{rem}_t\Delta t},
0,
P^{ev,\max}_i
\right)
$$

实际功率：

$$
P^{ev}_{i,t}
=
\operatorname{clip}
\left(
\max(P^{ev,rl}_{i,t},P^{ev,req}_{i,t}),
0,
P^{ev,\max}_i
\right)
$$

当策略为 `max_power` 时，应急窗口内低于目标的 EV 直接请求 \(P^{ev,\max}_i\)。

物理意义：emergency 是临近离家时的补救规则，比 progress 更硬，但比全局优化更短视。

## 9. 潮流与电网安全建模

`PowerFlowGridCore` 使用 SimBench/pandapower 网络。对每个 agent 的净负荷：

$$
P^{net}_{i,t}=L_{i,t}-G^{use}_{i,t}+P^b_{i,t}+P^{ev}_{i,t}
$$

写入 pandapower 时：

$$
\text{load.p\_mw}_{i,t}
=
\frac{\max(P^{net}_{i,t},0)}{1000}
$$

$$
\text{sgen.p\_mw}_{i,t}
=
\frac{\max(-P^{net}_{i,t},0)}{1000}
$$

随后执行 AC 潮流：

$$
\texttt{pandapower.runpp}
$$

电压越界量：

$$
\nu_{i,t}
=
\max(0,V_{\min}-V_{i,t})
+
\max(0,V_{i,t}-V_{\max})
$$

全网电压惩罚原始量：

$$
\psi^V_t
=
\sum_{b\in\mathcal{B}}
\left[
\max(0,V_{\min}-V_{b,t})
+
\max(0,V_{b,t}-V_{\max})
\right]^2
$$

线路和变压器惩罚原始量：

$$
\psi^{line}_t
=
\sum_{\ell}
\left[
\frac{\max(0,\lambda_{\ell,t}-\lambda_{\max})}{100}
\right]^2
$$

$$
\psi^{trafo}_t
=
\sum_m
\left[
\frac{\max(0,\lambda^{trafo}_{m,t}-\lambda_{\max})}{100}
\right]^2
$$

电压惩罚按 agent 电压违约占比分摊：

$$
\omega^V_{i,t}
=
\begin{cases}
\dfrac{\nu_{i,t}}{\sum_j\nu_{j,t}},
&\sum_j\nu_{j,t}>0\\
0,&\text{otherwise}
\end{cases}
$$

$$
\ell^V_{i,t}
=
Nw_V\psi^V_t\omega^V_{i,t}
$$

线路和变压器惩罚对每个 agent 相同：

$$
\ell^{line}_{i,t}=w_{line}\psi^{line}_t
$$

$$
\ell^{trafo}_{i,t}=w_{trafo}\psi^{trafo}_t
$$

## 10. 总奖励函数

环境中每个 agent 的最终奖励为：

$$
\begin{aligned}
r_{i,t}={}&
r^{bat}_{i,t}
r^{ev}_{i,t}
-\ell^{act}_{i,t}
-\ell^{soc}_{i,t}
-\ell^{ev,soc}_{i,t}
-\ell^{dep}_{i,t}
-\ell^{prog}_{i,t}\\
&-\ell^{ev,proj}_{i,t}
-\ell^{ev,eme}_{i,t}
+r^{throughput}_{i,t}
-\ell^{V}_{i,t}
-\ell^{line}_{i,t}
-\ell^{trafo}_{i,t}
\end{aligned}
$$

其中：

$$
\ell^{ev,proj}_{i,t}
=
w^{ev}_{proj}
|\Delta P^{ev,proj}_{i,t}|\Delta t
$$

$$
\ell^{ev,eme}_{i,t}
=
w^{ev}_{eme}
|\Delta P^{ev,eme}_{i,t}|\Delta t
$$

当前默认 `ev_projection_penalty_weight=0`、`ev_emergency_penalty_weight=0`，因此它们通常只用于记录诊断，不实际改变 reward。

MADRL 三个主方案的安全配置为：

| 方案 | 电压惩罚 | 线路惩罚 | 变压器惩罚 | 动作投影 |
| --- | --- | --- | --- | --- |
| `MADRL_BASE` | 0 | 0 | 0 | 否 |
| `MADRL_PENALTY` | 400 | 0 | 10 | 否 |
| `MADRL_PROJECTION` | 400 | 0 | 10 | 是 |

新增 EV soft 变体：

| 变体 | 关键参数 | 物理意义 |
| --- | --- | --- |
| `MADRL_BASE_PROGRESS` | \(w^{ev}_{prog}=50\) | 用夜间线性 SoC 目标缓解离家惩罚稀疏 |
| `MADRL_BASE_COSTWEIGHT` | \(w^{ev}_{cost}=1.5\) | 放大 EV 充电成本，引导低价充电 |

## 11. MADRL 建模

MADRL 使用 centralized training、decentralized execution。每个 agent 有一个 actor：

$$
a_{i,t}=\pi_i(o_{i,t};\theta_i)
$$

actor 输入为：

$$
o^{actor}_{i,t}
=
\left[
o^{local}_{i,t},
\bar p_{t:t+H-1},
\Delta \bar p_{t:t+H-1},
\bar L_{i,t:t+H-1},
\bar G_{i,t:t+H-1}
\right]
$$

每个 agent 有 twin critic：

$$
Q_{i,1}(o_{1:N,t},a_{1:N,t};\phi_{i,1}),
\quad
Q_{i,2}(o_{1:N,t},a_{1:N,t};\phi_{i,2})
$$

经验回放使用 n-step return。对 batch 样本：

$$
(o_t,a_t,R^{(n)}_t,o_{t+n},d_t)
$$

目标动作加入 TD3 平滑噪声并经过动作保护：

$$
a'_{t+n}
=
\operatorname{guard}
\left(
\pi' (o_{t+n})+\epsilon
\right)
$$

$$
\epsilon\sim\operatorname{clip}(\mathcal{N}(0,\sigma^2),-\epsilon_{clip},\epsilon_{clip})
$$

critic 目标：

$$
y_i
=
R^{(n)}_{i,t}
\gamma^n(1-d_t)
\min
\left(
Q'_{i,1}(o_{t+n},a'_{t+n}),
Q'_{i,2}(o_{t+n},a'_{t+n})
\right)
$$

critic 损失：

$$
\mathcal{L}_{Q_i}
=
\operatorname{MSE}(Q_{i,1}(o_t,a_t),y_i)
+
\operatorname{MSE}(Q_{i,2}(o_t,a_t),y_i)
$$

actor 延迟更新。更新 agent \(i\) 时，只替换联合动作中的第 \(i\) 个动作：

$$
a^{policy}_{i,t}
=
\operatorname{guard}(\pi_i(o_{i,t}))
$$

$$
\mathcal{L}_{\pi_i}
=
-
\mathbb{E}
\left[
Q_{i,1}(o_t,a_{1,t},\dots,a^{policy}_{i,t},\dots,a_{N,t})
\right]
$$

目标网络软更新：

$$
\theta'_i\leftarrow(1-\tau)\theta'_i+\tau\theta_i
$$

$$
\phi'_i\leftarrow(1-\tau)\phi'_i+\tau\phi_i
$$

`guard` 包含：

1. `map_actor_output_to_soc_feasible_action`：把 actor 原始输出映射到静态电池 SoC 可行功率区间；
2. 可选 `JointGridSafetyProjector`：投影静态电池功率和 PV 弃光；
3. `enforce_local_action_feasibility`：再次强制静态电池本地可行。

## 12. 联合安全投影

`JointGridSafetyProjector` 只投影静态电池功率和 PV 弃光，不投影 EV 充电。投影变量为：

$$
x_t=
\left[
P^b_{1,t},\dots,P^b_{N,t},
G^{curt}_{1,t},\dots,G^{curt}_{N,t}
\right]
$$

本地约束：

$$
P^{b,lower}_{i,t}
\le
P^b_{i,t}
\le
P^{b,upper}_{i,t}
$$

$$
0\le G^{curt}_{i,t}\le G_{i,t}
$$

投影器在零注入附近用有限差分估计灵敏度：

$$
S^V_{k,i}
=
\frac{V_k(+\delta e_i)-V_k(-\delta e_i)}{2\delta}
$$

$$
S^{line}_{m,i}
=
\frac{\lambda_m(+\delta e_i)-\lambda_m(-\delta e_i)}{2\delta}
$$

基础净负荷包含 EV 充电：

$$
P^{base}_{i,t}=L_{i,t}-G_{i,t}+P^{ev}_{i,t}
$$

线性化约束为：

$$
V_{\min}+m_V
\le
V^{base}+S^V\Delta P
\le
V_{\max}-m_V
$$

$$
\lambda^{base}+S^{line}\Delta P
\le
\lambda_{\max}-m_{line}
$$

$$
\left|
\sum_i(P^{base}_{i,t}+\Delta P_{i,t})
\right|
\le
S^{trafo}_{\max}(1-m_{trafo})
$$

代码按半空间逐行投影：

$$
x\leftarrow
x-
\frac{\max(0,r^\top x-b)}{\|r\|_2^2}r
$$

迭代次数为 `cfg.safety.projection_iters`。

## 13. Local MPC 建模

`LOCAL_MPC` 每个 agent 独立求解滚动优化。变量包括：

$$
c_t,d_t,curt_t,ev_t,E_t,E^{ev}_t,
P^{imp}_t,P^{exp}_t,z_t
$$

其中 \(z_t\) 是进口/出口模式二进制变量。

电池约束：

$$
E_{t+1}=E_t+\eta_bc_t\Delta t-\frac{d_t\Delta t}{\eta_b}
$$

$$
SOC^b_{\min}C^b\le E_t\le SOC^b_{\max}C^b
$$

$$
0\le c_t,d_t\le P^{b,\max}
$$

EV 约束：

$$
E^{ev}_{t+1}=E^{ev}_t+\eta_{ev}ev_t\Delta t
$$

$$
SOC^{ev}_{\min}C^{ev}\le E^{ev}_t\le SOC^{ev}_{\max}C^{ev}
$$

$$
0\le ev_t\le P^{ev,\max}
\quad\text{only if connected}
$$

功率平衡：

$$
P^{imp}_t-P^{exp}_t
=
L_t-G_t+curt_t+c_t-d_t+ev_t
$$

目标函数：

$$
\min
\sum_t
\left[
\tilde p_t\Delta t c_t
-
\tilde p_t\Delta t d_t
+
\tilde p_t\Delta t ev_t
+
\epsilon_b\Delta t(c_t+d_t)
\right]
+
w^{ev}_{dep}g_{dep}^2
$$

Local MPC 不显式约束电压、线路或变压器，只在环境执行后通过潮流评价实际影响。

## 14. ADMM MPC 建模

`ADMM_MPC` 在 local 子问题外增加变压器聚合约束。agent 净负荷为：

$$
n_{i,t}=L_{i,t}-G_{i,t}+curt_{i,t}+c_{i,t}-d_{i,t}+ev_{i,t}
$$

贡献副本：

$$
q_{i,t}=\alpha_{i,t}n_{i,t}
$$

当前实现：

$$
\alpha_{i,t}=1
$$

变压器聚合边界：

$$
-\bar P^{trafo}
\le
\sum_iq_{i,t}
\le
\bar P^{trafo}
$$

其中：

$$
\bar P^{trafo}=0.97P^{trafo}_{limit}
$$

每个 agent 解：

$$
\min
J_i^{local}
+
\frac{\rho}{2}
\left\|
\alpha_i n_i-z_i+u_i
\right\|_2^2
$$

然后投影：

$$
z\leftarrow
\Pi_{\left[-\bar P^{trafo},\bar P^{trafo}\right]}
(q+u)
$$

$$
u\leftarrow u+q-z
$$

残差为：

$$
r^{pri}=\|q-z\|_2
$$

$$
r^{dual}=\rho\|z^{new}-z^{old}\|_2
$$

最后第一步动作再解一个小 QP，使执行动作接近 ADMM 计划并满足第一步变压器聚合边界。

## 15. MISOCP 全局优化建模

`MISOCP` 是全局优化基线，在 episode 开始时对完整剩余 horizon 求解。变量包括：

- agent 设备变量：\(c_{i,t},d_{i,t},curt_{i,t},ev_{i,t},E_{i,t},E^{ev}_{i,t}\)；
- 支路变量：\(P_{\ell,t},Q_{\ell,t},\ell_{\ell,t}\)；
- 节点电压平方：\(v_{b,t}\)；
- 根节点进口/出口功率和二进制模式。

设备约束同 Local MPC。EV 到家重置：

$$
E^{ev}_{i,t+1}
=
SOC^{ev}_{arr}C^{ev}_i
\quad
\text{if }t+1=t_{arr}
$$

EV 离家约束：

soft 模式：

$$
g_i\ge
SOC^{ev}_{req}
-
\frac{E^{ev}_{i,t_{dep}}}{C^{ev}_i}
$$

$$
J\leftarrow J+w^{ev}_{dep}g_i^2
$$

hard/emergency 模式：

$$
E^{ev}_{i,t_{dep}}
\ge
SOC^{ev}_{req}C^{ev}_i
$$

DistFlow 约束对支路 \(\ell=(u,v)\)：

$$
P_{\ell,t}
=
\sum_{k\in\mathcal{C}(v)}P_{k,t}
+
r_{\ell}\ell_{\ell,t}
+
p^{inj}_{v,t}
$$

$$
Q_{\ell,t}
=
\sum_{k\in\mathcal{C}(v)}Q_{k,t}
+
x_{\ell}\ell_{\ell,t}
$$

$$
v_{v,t}
=
v_{u,t}
-2(r_{\ell}P_{\ell,t}+x_{\ell}Q_{\ell,t})
+
(r_{\ell}^2+x_{\ell}^2)\ell_{\ell,t}
$$

二阶锥松弛：

$$
P_{\ell,t}^2+Q_{\ell,t}^2
\le
v_{u,t}\ell_{\ell,t}
$$

agent 注入：

$$
p^{inj}_{v,t}
=
\frac{
L_{i,t}-G_{i,t}+curt_{i,t}+c_{i,t}-d_{i,t}+ev_{i,t}
}
{1000S_{base}}
$$

目标函数：

$$
\min
\sum_{i,t}
\left[
\tilde p_t\Delta t(c_{i,t}-d_{i,t})
+
\tilde p_t\Delta t ev_{i,t}
+
10^{-4}\Delta t(c_{i,t}+d_{i,t})
\right]
+
\text{soft EV departure penalty}
$$

MISOCP 使用 perfect forecast，并在 episode 首步求完整计划，后续按 offset 执行。

## 16. 结果记录和指标

rollout record 包含：

```text
results/<forecast_mode>/<controller>/record/
  step.parquet
  agent.parquet
  grid.parquet
  summary.parquet
  metrics.parquet
  meta.json
  manifest.json
```

新增 EV progress 字段包括：

$$
SOC^{ev,prog}_t,\quad g^{prog}_{i,t},\quad \ell^{prog}_{i,t}
$$

对应列：

- `ev_progress_target_soc`
- `ev_progress_gap`
- `ev_progress_penalty`
- `madrl_r_ev_progress_penalty`

经济记录中：

$$
\text{battery cost}
=-r^{bat}
$$

$$
\text{EV cost}
=w^{ev}_{cost}P^{ev}\tilde p_t\Delta t
\quad\text{for soft mode records}
$$

$$
\text{total cost}
=
\text{battery cost}
+
\text{EV cost}
+
\text{PV cost}
$$

当前 PV cost 默认为 0，除非未来显式配置弃光价格或惩罚。

主要评估指标：

- `episode_reward_mean`
- `storage_profit_eur_mean`
- `voltage_violation_steps_mean`
- `min_vm_pu`
- `max_vm_pu`
- `trafo_loading_max_pct`
- `trafo_overload_steps`
- `feeder_netload_ramp_mean_abs_kw`
- `battery_net_power_kw_mean_abs`
- EV departure/progress/cost summary

## 17. 目录职责

```text
configs/      所有 dataclass 配置
data/         原始数据读取与 share_data 构建
predictors/   LSTM 特征、训练、加载和评估
envs/         GridEnv、pandapower 潮流核心、多进程环境
models/       actor/critic 网络结构
controllers/  MADRL、Local MPC、ADMM MPC、MISOCP 控制器
scripts/      notebook 调用的实验编排函数和诊断脚本
utils/        路径、价格、artifact、record、绘图
notebooks/    主实验与 EV soft/hard/emergency 变体
tests/        契约测试、smoke 测试、encoding hygiene
```

新增 EV soft notebook 变体：

```text
notebooks/notebooks_EV/soft/madrl_base_EV_soft _progress.ipynb
notebooks/notebooks_EV/soft/madrl_base_EV_soft_costweight.ipynb
```

## 18. 推荐实验流程

主线流程：

1. `notebooks/predict.ipynb`：训练 LSTM，构建 `share_data`。
2. `notebooks/madrl_base.ipynb`：训练 `MADRL_BASE`。
3. `notebooks/madrl_base_safe.ipynb`：训练 `MADRL_PENALTY`。
4. `notebooks/madrl_projection_safe.ipynb`：训练 `MADRL_PROJECTION`。
5. `notebooks/misocp.ipynb`：运行全局 MISOCP。
6. `notebooks/mpc.ipynb`：运行 Local MPC 和 ADMM MPC。
7. `notebooks/compare.ipynb`：读取缓存并生成比较表/图。

EV 场景：

```text
notebooks/notebooks_EV/soft/
notebooks/notebooks_EV/hard/
notebooks/notebooks_EV/emergency_charging/
```

其中 soft 目录现在包含 base、progress、costweight 等 reward 变体。

## 19. 程序化入口

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

长实验默认使用：

$$
\texttt{cfg.runtime.device="cuda"}
$$

若没有 GPU，应显式改成 CPU。

## 20. 产物布局

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

## 21. 环境与测试

仓库默认使用 Conda 环境 `MADRL_ESS`：

```powershell
conda run -n MADRL_ESS python -m pytest
```

编辑中文 README、notebook 或注释后，至少运行：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_encoding_hygiene.py
```

MPC、ADMM MPC 和 MISOCP 依赖 `gurobipy` 与可用 Gurobi license。没有 Gurobi 时，数据、预测、部分环境和编码测试仍可运行，但优化基线不可用。

## 22. 维护边界

- 数据契约归 `data/` 和 `share_data`。
- 物理环境、SoC、EV progress/costweight reward 归 `envs/grid_env.py`。
- MADRL 训练归 `scripts/madrl.py`。
- 优化基线归 `controllers/`。
- 结果 schema 和图表归 `utils/records.py`。
- 不做旧路径扫描或 latest artifact 模糊查找。
- 缺 artifact、缺字段、缺 solver 时直接失败。

这套代码的核心是显式复现实验：负荷、PV、电价如何进入；电池和 EV 如何演化；progress 和 costweight 如何改变 reward；潮流如何反馈安全惩罚；MADRL 与优化基线如何比较；所有结果如何落盘，都应能从代码主线直接追到。
