# MADRL_ESS

面向配电网潮流约束场景的多智能体储能与弃光协同控制工程。当前仓库的主线工作流是：

- 用 `LSTM` 预测 `price / load / pv`
- 在 `GridEnv` 中训练和评估多智能体控制器
- 在同一测试窗口下比较三种 `MADRL` 方案与两种 `MPC` 基线

这份 README 不再以旧训练入口为中心，而是按当前代码状态，优先说明：

1. 现在真正对比的三种 `MADRL` 方案是什么
2. 环境、动作、约束、奖励和测试目标的数学模型是什么
3. 三种方案的差异发生在哪一层

## 当前代码中的三种 MADRL 方案

| 方案名 | Notebook | Algorithm | 电网安全惩罚 | 本地 SoC 约束 | 安全投影层 | 核心机制说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `MADRL + No Safety` | `notebooks/madrl/train_base.ipynb` | `MATD3` | 否，默认 `w_voltage_pen = w_line_pen = w_trafo_pen = 0` | 是 | 否 | 训练和评估前都保留本地电池可行性裁剪，并保留 `r_soc_pen`；这里只是“不对电网越限额外罚分”，不是“完全无约束” |
| `MADRL + Safety Penalty` | `notebooks/madrl/train_base_safe.ipynb` | `MATD3` | 是，默认三项安全惩罚权重均为 `10.0` | 是 | 否 | 动作链与 `MADRL + No Safety` 相同，但把电压、线路、变压器越限写进优化目标 |
| `MADRL + Safety Projection` | `notebooks/madrl/train_projection_safe.ipynb` | `MATD3_SAFE_POC` | 是，默认三项安全惩罚权重均为 `10.0` | 是 | 是 | 在 `rollout`、`target-Q` 和 `actor-loss` 三处都引入 joint safety projector，并在 projector 后再做一次本地 SoC 可行化 |

`MADRL + No Safety` 这个名字需要特别注意：它的含义是“无电网安全惩罚”，不是“智能体动作完全不受约束”。当前代码里三种方案都会保留：

- 本地电池 `SoC` 可行性裁剪 `enforce_local_action_feasibility_torch`
- 储能越限惩罚 `r_soc_pen`
- 默认 `w_soc_pen = 10.0`

## 问题与数学模型

### 1. 索引、变量与符号约定

设：

- 智能体集合为 $\mathcal{N} = \{1,2,\dots,N\}$
- 离散时间步为 $t = 0,1,\dots,T-1$
- 预测窗口长度为 $H$，观测中包含未来一段 `price / load / pv` 序列

对任意智能体 $i \in \mathcal{N}$ 和时间步 $t$，定义：

- $P^{load}_{i,t}$：负荷功率，单位 `kW`
- $P^{pv,raw}_{i,t}$：原始光伏功率，单位 `kW`
- $\pi_t$：电价，单位 `EUR/kWh`
- $E_{i,t}$：电池能量，单位 `kWh`
- $SoC_{i,t} = E_{i,t}/E_i^{cap}$：电池荷电状态
- $P_i^{max}$：电池充放电功率上限
- $\eta$：充放电效率
- $\Delta t$：离散时间步长，单位 `h`

当前代码中的电池功率符号约定必须单独强调：

- $P^{bat} > 0$ 表示充电
- $P^{bat} < 0$ 表示放电

这个约定与部分文献中“正值表示放电”的写法不同，阅读公式时请以本仓库实现为准。

### 2. 动作空间与物理量映射

每个智能体的动作是二维向量：

$$
a_{i,t} = \left[a^{bat}_{i,t}, a^{pv}_{i,t}\right], \qquad a^{bat}_{i,t}, a^{pv}_{i,t} \in [-1,1]
$$

其中：

- $a^{bat}_{i,t}$ 是归一化电池动作
- $a^{pv}_{i,t}$ 是归一化光伏动作

动作到物理量的映射为：

$$
P^{bat,req}_{i,t} = a^{bat}_{i,t} P_i^{max}
$$

$$
u^{pv}_{i,t} = \frac{a^{pv}_{i,t} + 1}{2}
$$

$$
P^{pv,eff}_{i,t} = u^{pv}_{i,t} P^{pv,raw}_{i,t}
$$

$$
P^{pv,curt}_{i,t} = P^{pv,raw}_{i,t} - P^{pv,eff}_{i,t}
$$

其中：

- $P^{bat,req}_{i,t}$ 是控制器申请的储能功率
- $P^{pv,eff}_{i,t}$ 是实际被利用的光伏功率
- $P^{pv,curt}_{i,t}$ 是弃光功率

### 3. 本地 SoC 与电池约束

对每个时间步，环境先根据当前 `SoC` 计算电池可行的本地充放电范围。令：

$$
E_i^{min} = SoC^{min} E_i^{cap}, \qquad E_i^{max} = SoC^{max} E_i^{cap}
$$

则本地上界和下界为：

$$
P^{upper}_{i,t} = \min \left(P_i^{max}, \max \left(0, \frac{E_i^{max} - E_{i,t}}{\eta \Delta t}\right)\right)
$$

$$
P^{lower}_{i,t} = - \min \left(P_i^{max}, \max \left(0, \frac{(E_{i,t} - E_i^{min}) \eta}{\Delta t}\right)\right)
$$

最终送入环境的电池功率为：

$$
P^{bat,exec}_{i,t} = \mathrm{clip}\left(P^{bat,req}_{i,t}, P^{lower}_{i,t}, P^{upper}_{i,t}\right)
$$

电池能量更新与代码中的效率方向完全一致：

充电时：

$$
\Delta E_{i,t} = \eta P^{bat,exec}_{i,t} \Delta t, \qquad P^{bat,exec}_{i,t} \ge 0
$$

放电时：

$$
\Delta E_{i,t} = \frac{P^{bat,exec}_{i,t}}{\eta} \Delta t, \qquad P^{bat,exec}_{i,t} < 0
$$

由于放电时 $P^{bat,exec}_{i,t}$ 为负值，所以 $\Delta E_{i,t}$ 也为负，电池能量会下降。于是：

$$
E_{i,t+1} = E_{i,t} + \Delta E_{i,t}
$$

$$
SoC_{i,t+1} = \frac{E_{i,t+1}}{E_i^{cap}}
$$

控制器侧储能越限惩罚的核心未加权量记为：

$$
\psi^{soc}_{i,t} = \frac{\left|P^{bat,req}_{i,t} - P^{bat,exec}_{i,t}\right|}{P_i^{max}}
$$

在 `MADRL + Safety Projection` 中，`r_soc_pen` 的来源略有不同：它主要记录 `projected -> executed` 的残差，而不是 `raw -> executed` 的差值。

### 4. 电网功率平衡与潮流量

对任意智能体，实际净负荷为：

$$
P^{net}_{i,t} = P^{load}_{i,t} - P^{pv,eff}_{i,t} + P^{bat,exec}_{i,t}
$$

由此得到本地购电与上网功率：

$$
P^{grid,imp}_{i,t} = \max(P^{net}_{i,t}, 0)
$$

$$
P^{grid,exp}_{i,t} = \max(-P^{net}_{i,t}, 0)
$$

然后环境将全体智能体的净负荷送入 `GridCore` 做潮流计算，得到：

- 节点电压 $v_t$
- 线路负载率 $l_t$
- 变压器负载率 $\tau_t$

电压、线路和变压器越限惩罚都来自这些潮流结果。

在测试阶段的总功率平衡图中，当前代码使用的严格堆叠口径为：

$$
P^{load,tot}_t + P^{charge,tot}_t + P^{export,tot}_t + P^{curt,tot}_t
=
P^{pv,raw,tot}_t + P^{import,tot}_t + P^{discharge,tot}_t
$$

也就是说，弃光被视作“消纳侧的一部分”进入堆叠图。

### 5. 训练 reward 与测试 objective 的双口径

当前训练阶段优化的是逐步 reward 最大化问题。按代码中的实现，reward 分量为：

$$
r = -r_{purchase\_cost} + r_{export\_subsidy} - r_{soc\_pen} - r_{safe\_v} - r_{safe\_line} - r_{safe\_trafo}
$$

其中：

- $r_{purchase\_cost}$：购电成本
- $r_{export\_subsidy}$：向上级电网售电补贴
- $r_{soc\_pen}$：储能本地可行性惩罚
- $r_{safe\_v}$：电压越限惩罚
- $r_{safe\_line}$：线路越限惩罚
- $r_{safe\_trafo}$：变压器越限惩罚

需要特别说明的是：`NormalReward` 负责生成奖励分量模板，但 `r_soc_pen` 的实际数值由 controller-side postprocessing 注入。

测试阶段汇总的是成本口径的目标值，当前代码定义为：

$$
objective\_total =
purchase\_cost\_total
- export\_subsidy\_total
+ soc\_penalty\_total
+ voltage\_penalty\_total
+ line\_penalty\_total
+ trafo\_penalty\_total
$$

这里要注意两点：

1. `reward` 是最大化口径，`objective_total` 是最小化运行成本口径
2. `soc_penalty_total`、`voltage_penalty_total`、`line_penalty_total`、`trafo_penalty_total` 都是带权后的惩罚累计值，不是未加权原始 violation

例如，`soc_penalty_total` 可以理解为测试期内所有时间步、所有智能体的带权储能惩罚求和：

$$
soc\_penalty\_total = \sum_{t=0}^{T-1} \sum_{i=1}^{N} w_{soc} \psi^{soc}_{i,t}
$$

其余三项安全惩罚总量也是类似的“带权 violation 累计”。

### 6. `MADRL + Safety Projection` 的 projector 数学描述

`MATD3_SAFE_POC` 的 joint safety projector 工作在物理量 `kW` 空间，而不是归一化动作空间。对 $N$ 个智能体，定义：

$$
x_t \in \mathbb{R}^{2N}
$$

$$
x_t =
\left[
P^{bat}_{1,t}, \dots, P^{bat}_{N,t},
P^{curt}_{1,t}, \dots, P^{curt}_{N,t}
\right]
$$

也就是说，projector 的优化变量前半部分是电池功率，后半部分是弃光功率。

在一个时间步内，projector 基于 `GridCore` 的线性化灵敏度构造近似约束：

$$
v_t \approx v^{aff}_t + S_v x_t
$$

$$
l_t \approx l^{aff}_t + S_l x_t
$$

$$
\tau_t \approx \tau^{aff}_t + S_{\tau} x_t
$$

其中：

- $v^{aff}_t$、$l^{aff}_t$、$\tau^{aff}_t$ 是由当前负荷与原始光伏确定的仿射基点
- $S_v$、$S_l$、$S_{\tau}$ 是对节点电压、线路负载率、变压器负载率的灵敏度矩阵

projector 施加的安全边界近似写为：

$$
v^{min} + margin_v \le v_t \le v^{max} - margin_v
$$

$$
l_t \le l^{max} - margin_l
$$

$$
\tau_t \le \tau^{max} - margin_{\tau}
$$

同时还要满足本地物理边界：

$$
P^{lower}_{i,t} \le P^{bat}_{i,t} \le P^{upper}_{i,t}
$$

$$
0 \le P^{curt}_{i,t} \le P^{pv,raw}_{i,t}
$$

在代码里，projector 输出的联合动作会先从 `kW` 空间转换回归一化动作，再经过一次本地 `SoC` feasibility clamp，然后才真正送入环境。这也是当前 `projection_safe` 路径和另外两种方案最大的实现差异。

## 三种方案的实现差异

### `MADRL + No Safety`

- 算法：`MATD3`
- 交互动作链：`raw action -> local SoC feasibility -> env`
- 电网安全惩罚权重默认关闭：`w_voltage_pen = w_line_pen = w_trafo_pen = 0`
- 仍然保留 `r_soc_pen`

它可以回答的问题是：如果只保留本地储能物理约束，而不把电网安全越限写进目标函数，策略会学成什么样。

### `MADRL + Safety Penalty`

- 算法：`MATD3`
- 交互动作链：`raw action -> local SoC feasibility -> env`
- 与 `MADRL + No Safety` 的主要差别不在动作链，而在 reward
- 默认打开三项电网安全惩罚：`w_voltage_pen = w_line_pen = w_trafo_pen = 10.0`

它对应的是“通过惩罚项学习安全”的思路：不直接改动作，只通过目标函数把越限行为变得更贵。

### `MADRL + Safety Projection`

- 算法：`MATD3_SAFE_POC`
- 交互动作链：`raw action -> joint safety projector -> local SoC feasibility -> env`
- 不仅 rollout 前会投影，`target-Q` 和 `actor-loss` 也会使用 projector 处理 joint action
- `r_soc_pen` 在该方案下主要记录 `projected -> executed` 残差，因此更像 projector 健壮性诊断，而不是 base 方案里的原始申请惩罚

它对应的是“先把动作投到近似安全集合，再拿去训练和执行”的思路。

### 三种方案的动作进入环境口径对比

| 环节 | `MADRL + No Safety` | `MADRL + Safety Penalty` | `MADRL + Safety Projection` |
| --- | --- | --- | --- |
| rollout 前动作处理 | 本地 `SoC` 可行化 | 本地 `SoC` 可行化 | 先 projector，再本地 `SoC` 可行化 |
| `target-Q` 动作 | 标准 `MATD3` 路径 | 标准 `MATD3` 路径 | 使用 projector 后的 joint action |
| `actor-loss` 动作 | 标准 `MATD3` 路径 | 标准 `MATD3` 路径 | 使用 projector 后的 joint action |
| 电网安全信息进入优化 | 不进入 | 通过 reward 惩罚进入 | 通过 reward 惩罚和动作投影同时进入 |

## 对比与复现入口

当前对比入口是 `notebooks/madrl/compare.ipynb`。它会在同一测试窗口下评估五个控制器：

- `MADRL + No Safety`
- `MADRL + Safety Penalty`
- `MADRL + Safety Projection`
- `MPC + Perfect Forecast`
- `MPC + LSTM Forecast`

这份 README 主体只详细展开前三种 `MADRL` 方案；两种 `MPC` 基线主要作为统一测试窗口下的参考对象。

建议的复现顺序是：

1. `notebooks/forecast/forecast_lstm.ipynb`
2. `notebooks/madrl/train_base.ipynb`
3. `notebooks/madrl/train_base_safe.ipynb`
4. `notebooks/madrl/train_projection_safe.ipynb`
5. `notebooks/madrl/compare.ipynb`

## 工程入口与目录

### 关键 notebook

- `notebooks/forecast/forecast_lstm.ipynb`
- `notebooks/madrl/train_base.ipynb`
- `notebooks/madrl/train_base_safe.ipynb`
- `notebooks/madrl/train_projection_safe.ipynb`
- `notebooks/madrl/compare.ipynb`

### 关键代码位置

- `configs/experiment_config.py`：默认配置
- `envs/grid_env.py`：环境、动作映射、储能动力学
- `envs/rewards/NormalReward.py`：购电成本、售电补贴和电网安全惩罚
- `controllers/action_feasibility.py`：本地 `SoC` 可行化与 `r_soc_pen`
- `controllers/madrl/matd3.py`：`MATD3`
- `controllers/madrl/matd3_safe_poc.py`：`MATD3_SAFE_POC`
- `controllers/madrl/safety_projector.py`：joint safety projector
- `scripts/train.py`：训练主循环
- `scripts/utils/grid_notebook_workflow.py`：评估、汇总与画图

### 主要产物目录

- `artifacts/forecast/lstm/`
- `artifacts/training/shared_data/`
- `artifacts/training/checkpoints/`
- `artifacts/training/tensorboard/`

## 说明

- 当前文档只描述现有主线代码，不再保留旧版 reward、旧版 notebook 或不存在的入口。
- 所有数学公式均按 GitHub Markdown 可渲染的 `$...$` 与 `$$...$$` 形式书写。
- 如果你只想快速开始，请先确认 `data/processed/prosumer/` 下的数据已准备好，再按“复现顺序”依次运行 notebook。
