# Comparison Models for PPT

本文档整理 7 种对比方案的统一建模表达，便于直接提炼到 PPT 中。建议讲述顺序为：

1. 统一说明所有方案共享的系统模型
2. 再说明各方案在预测方式、求解结构、安全处理上的差异

## 1. 公用模型

### 1.1 记号定义

- \(i \in \mathcal N\)：用户/智能体编号
- \(t = 0,\dots,H-1\)：优化时域内的时间步
- \(\Delta t\)：时间步长
- \(L_{i,t}\)：负荷功率
- \(PV_{i,t}\)：光伏可用功率
- \(\pi_t\)：购电价格
- \(\pi^{\mathrm{exp}}\)：售电补贴或上网电价
- \(P^{\mathrm{ch}}_{i,t}\)：储能充电功率
- \(P^{\mathrm{dis}}_{i,t}\)：储能放电功率
- \(P^{\mathrm{curt}}_{i,t}\)：光伏弃电功率
- \(E_{i,t}\)：储能能量状态
- \(P^g_{i,t}\)：用户与电网的净交换功率
- \(P^{\mathrm{imp}}_{i,t}\)、\(P^{\mathrm{exp}}_{i,t}\)：购电功率、售电功率

### 1.2 公共功率平衡

对任意用户 \(i\) 和时刻 \(t\)，净交换功率满足

\[
P^g_{i,t}
=
L_{i,t}-PV_{i,t}+P^{\mathrm{ch}}_{i,t}-P^{\mathrm{dis}}_{i,t}+P^{\mathrm{curt}}_{i,t}
\]

同时定义

\[
P^g_{i,t}=P^{\mathrm{imp}}_{i,t}-P^{\mathrm{exp}}_{i,t},
\quad
P^{\mathrm{imp}}_{i,t}\ge 0,\;
P^{\mathrm{exp}}_{i,t}\ge 0
\]

### 1.3 公共储能动态

\[
E_{i,t+1}
=
E_{i,t}
+\eta_i^{\mathrm{ch}} P^{\mathrm{ch}}_{i,t}\Delta t
-\frac{P^{\mathrm{dis}}_{i,t}}{\eta_i^{\mathrm{dis}}}\Delta t
\]

### 1.4 公共设备约束

\[
0 \le P^{\mathrm{ch}}_{i,t} \le \bar P_i^{\mathrm{ch}}
\]

\[
0 \le P^{\mathrm{dis}}_{i,t} \le \bar P_i^{\mathrm{dis}}
\]

\[
0 \le P^{\mathrm{curt}}_{i,t} \le PV_{i,t}
\]

\[
\underline E_i \le E_{i,t} \le \bar E_i
\]

如需终端 SoC 约束，还可写为

\[
E_{i,H} \ge E_i^{\mathrm{ref}}
\quad \text{或} \quad
(E_{i,H}-E_i^{\mathrm{ref}})^2
\]

### 1.5 公共经济目标

所有方案都围绕同一个经济目标展开，即最小化购电成本并最大化售电收益。统一可写为

\[
\min J
=
\sum_{t=0}^{H-1}\sum_{i\in\mathcal N}
\left(
\pi_t P^{\mathrm{imp}}_{i,t}\Delta t
- \pi^{\mathrm{exp}} P^{\mathrm{exp}}_{i,t}\Delta t
\right)
+\lambda_{\mathrm{soc}} J_{\mathrm{soc}}
+\lambda_v J_v
+\lambda_l J_l
+\lambda_{tr} J_{tr}
\]

其中：

- \(J_{\mathrm{soc}}\)：终端 SoC 偏差或电池使用正则项
- \(J_v\)：电压越限项
- \(J_l\)：线路越限项
- \(J_{tr}\)：变压器越限项

### 1.6 公共网络安全约束

统一可抽象为

\[
V_n^{\min} \le V_{n,t} \le V_n^{\max}
\]

\[
|S_{\ell,t}| \le \bar S_{\ell}
\]

\[
-\bar P_{tr}^{\mathrm{exp}}
\le
P_{tr,t}
\le
\bar P_{tr}^{\mathrm{imp}}
\]

不同方案的差异在于：

- 有的方案把这些约束作为硬约束直接写进优化问题
- 有的方案把它们改写成惩罚项
- 有的方案在策略输出后再做安全投影

## 2. 各方案数学模型

## 2.1 MISOCP

### 建模思想

MISOCP 是集中式全局优化模型，在同一个优化问题中同时决定所有用户的储能、弃光和网络潮流变量，并显式考虑配电网物理约束。

### 决策变量

- 用户侧：\(P^{\mathrm{ch}}_{i,t}, P^{\mathrm{dis}}_{i,t}, P^{\mathrm{curt}}_{i,t}, E_{i,t}\)
- 网络侧：\(P_{\ell,t}, Q_{\ell,t}, l_{\ell,t}, v_{n,t}\)
- 根节点购售电：\(P_t^{\mathrm{imp}}, P_t^{\mathrm{exp}}\)
- 二进制变量：\(u_t \in \{0,1\}\)，用于限制同一时刻主网购电/售电模式

### 目标函数

\[
\min
\sum_t
\left(
\pi_t P_t^{\mathrm{imp}}\Delta t
-\pi^{\mathrm{exp}}P_t^{\mathrm{exp}}\Delta t
\right)
+ \epsilon_1 \sum_{i,t}\left(P^{\mathrm{ch}}_{i,t}+P^{\mathrm{dis}}_{i,t}\right)
+ \epsilon_2 \sum_{\ell,t} l_{\ell,t}
\]

其中后两项通常是正则项或物理细化项。

### 主要约束

用户侧满足公共功率平衡、储能动态和功率边界约束。

网络侧满足 DistFlow/SOCP 约束，例如：

\[
P_{ij,t}
=
\sum_{k:(j,k)} P_{jk,t}
+ p_{j,t}
+ r_{ij} l_{ij,t}
\]

\[
Q_{ij,t}
=
\sum_{k:(j,k)} Q_{jk,t}
+ q_{j,t}
+ x_{ij} l_{ij,t}
\]

\[
v_{j,t}
=
v_{i,t}
-2(r_{ij}P_{ij,t}+x_{ij}Q_{ij,t})
+(r_{ij}^2+x_{ij}^2)l_{ij,t}
\]

\[
P_{ij,t}^2+Q_{ij,t}^2 \le v_{i,t} l_{ij,t}
\]

再加上

\[
0 \le P_t^{\mathrm{imp}} \le \bar P_{tr} u_t
\]

\[
0 \le P_t^{\mathrm{exp}} \le \bar P_{tr}(1-u_t)
\]

### PPT 可强调

- 集中式全局最优
- 网络约束最完整
- 计算复杂度最高

## 2.2 Single-Agent MPC + Perfect Prediction

### 建模思想

每个用户独立求解自己的滚动优化问题，不显式联立整个网络潮流。未来负荷、光伏和价格均使用真实未来值。

### 优化问题

在时刻 \(k\)，第 \(i\) 个用户求解

\[
\min_{x_{i,k:k+H-1}}
\sum_{\tau=k}^{k+H-1}
\left(
\pi_{\tau} P^{\mathrm{imp}}_{i,\tau}\Delta t
- \pi^{\mathrm{exp}} P^{\mathrm{exp}}_{i,\tau}\Delta t
\right)
+ \omega_i (E_{i,k+H}-E_i^{\mathrm{ref}})^2
\]

满足

\[
P^{\mathrm{imp}}_{i,\tau}-P^{\mathrm{exp}}_{i,\tau}
=
L_{i,\tau}-PV_{i,\tau}
+P^{\mathrm{ch}}_{i,\tau}
-P^{\mathrm{dis}}_{i,\tau}
+P^{\mathrm{curt}}_{i,\tau}
\]

以及公共储能和设备约束。

### Perfect prediction 假设

\[
\hat L_{i,\tau|k}=L_{i,\tau},\quad
\hat PV_{i,\tau|k}=PV_{i,\tau},\quad
\hat \pi_{\tau|k}=\pi_{\tau}
\]

即预测值等于真实未来值。

### PPT 可强调

- 单体局部优化
- 预测误差为 0
- 可作为 MPC 上界参考

## 2.3 Single-Agent MPC + LSTM Prediction

### 建模思想

优化结构与上一种方案完全相同，只是把真实未来量换成 LSTM 预测序列。

### 优化问题

\[
\min_{x_{i,k:k+H-1}}
\sum_{\tau=k}^{k+H-1}
\left(
\hat \pi_{\tau|k} P^{\mathrm{imp}}_{i,\tau}\Delta t
- \pi^{\mathrm{exp}} P^{\mathrm{exp}}_{i,\tau}\Delta t
\right)
+ \omega_i (E_{i,k+H}-E_i^{\mathrm{ref}})^2
\]

满足

\[
P^{\mathrm{imp}}_{i,\tau}-P^{\mathrm{exp}}_{i,\tau}
=
\hat L_{i,\tau|k}-\hat PV_{i,\tau|k}
+P^{\mathrm{ch}}_{i,\tau}
-P^{\mathrm{dis}}_{i,\tau}
+P^{\mathrm{curt}}_{i,\tau}
\]

### 预测模型输入

- \(\hat L_{i,\tau|k}\)：LSTM 负荷预测
- \(\hat PV_{i,\tau|k}\)：LSTM 光伏预测
- \(\hat \pi_{\tau|k}\)：LSTM 或给定预测价格

### PPT 可强调

- 与 perfect prediction 的唯一区别是预测来源
- 能体现预测误差对 MPC 性能的影响

## 2.4 ADMM MPC + LSTM Prediction

### 建模思想

将多用户耦合优化问题分解成多个局部 MPC 子问题，通过 ADMM 迭代实现一致性协调。未来信息来自 LSTM 预测。

### 全局形式

\[
\min_{x_1,\dots,x_N} \sum_{i=1}^{N} f_i(x_i)
\quad
\text{s.t.}\quad
\sum_{i=1}^{N} A_i x_i \in \mathcal C
\]

其中：

- \(f_i(x_i)\) 是第 \(i\) 个用户的局部 MPC 成本
- \(\mathcal C\) 表示变压器或网络安全可行域

### 局部子问题

定义用户净注入

\[
n_{i,t}
=
\hat L_{i,t}-\hat PV_{i,t}
+P^{\mathrm{ch}}_{i,t}
-P^{\mathrm{dis}}_{i,t}
+P^{\mathrm{curt}}_{i,t}
\]

局部成本为

\[
f_i(x_i)
=
\sum_t
\left(
\pi_t n_{i,t}^{+}\Delta t
- \pi^{\mathrm{exp}} n_{i,t}^{-}\Delta t
\right)
+ \omega_i(E_{i,H}-E_i^{\mathrm{ref}})^2
\]

### ADMM 迭代

第 \(r+1\) 轮迭代：

\[
x_i^{r+1}
=
\arg\min_{x_i \in \mathcal X_i}
\left(
f_i(x_i)
+ \frac{\rho}{2}\|A_i x_i - z_i^r + u_i^r\|_2^2
\right)
\]

\[
z^{r+1} = \Pi_{\mathcal C}(A x^{r+1}+u^r)
\]

\[
u^{r+1} = u^r + A x^{r+1} - z^{r+1}
\]

### PPT 可强调

- 分布式求解
- 局部优化并行
- 通过 ADMM 实现网络协调

## 2.5 DRL No Safe

### 建模思想

将问题建模为马尔可夫决策过程，策略网络直接输出控制动作，不显式加入网络安全机制。

### 策略形式

\[
a_t = \pi_{\theta}(s_t)
\]

其中：

- \(s_t\)：状态，包含 SoC、负荷、光伏、价格、可能的网络观测
- \(a_t\)：动作，包含储能控制和弃光控制

### 优化目标

\[
\max_{\theta}
\mathbb E_{\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t^{\mathrm{econ}}
\right]
\]

其中经济回报为

\[
r_t^{\mathrm{econ}}
=
-\pi_t P_t^{\mathrm{imp}}\Delta t
+\pi^{\mathrm{exp}} P_t^{\mathrm{exp}}\Delta t
\]

### 特点

- 不进行安全投影
- 不显式处理电压、线路、变压器越限
- 更关注经济性

### PPT 可强调

- 纯策略学习
- 作为无安全机制基线

## 2.6 DRL + Penalty Safe

### 建模思想

仍采用 DRL 直接输出动作，但把安全违约写进奖励函数中，用软约束方式抑制越限。

### 策略形式

\[
a_t = \pi_{\theta}(s_t)
\]

### 奖励函数

\[
\max_{\theta}
\mathbb E_{\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\right]
\]

其中

\[
r_t
=
r_t^{\mathrm{econ}}
-\lambda_v \psi_t^v
-\lambda_l \psi_t^l
-\lambda_{tr} \psi_t^{tr}
\]

例如可定义

\[
\psi_t^v
=
\sum_n
\left(
[V_{n,t}-V_n^{\max}]_+
+[V_n^{\min}-V_{n,t}]_+
\right)
\]

\[
\psi_t^l
=
\sum_{\ell}
[|S_{\ell,t}|-\bar S_\ell]_+
\]

\[
\psi_t^{tr}
=
[|P_{tr,t}|-\bar P_{tr}]_+
\]

### PPT 可强调

- 安全通过惩罚项体现
- 不能严格保证每一步都可行
- 属于软约束方法

## 2.7 DRL + Projection Safe

### 建模思想

策略先输出原始动作，再通过安全投影层投影到当前时刻的线性化安全可行域中，从而实现硬安全修正。

### 原始策略输出

\[
a_t^{\mathrm{raw}} = \pi_{\theta}(s_t)
\]

### 安全投影

\[
a_t
=
\Pi_{\mathcal S(s_t)}(a_t^{\mathrm{raw}})
=
\arg\min_{a \in \mathcal S(s_t)} \|a-a_t^{\mathrm{raw}}\|_2^2
\]

其中

\[
\mathcal S(s_t)
=
\{a \mid A(s_t)a \le b(s_t),\; a \in \mathcal X_{\mathrm{local}}\}
\]

### 线性化安全约束

可将网络量在当前点附近线性化为

\[
\hat V_t = V_t^0 + H_t^V a
\]

\[
\hat S_t = S_t^0 + H_t^S a
\]

\[
\hat P_{tr,t} = P_{tr,t}^0 + H_t^{tr} a
\]

并施加

\[
V^{\min}+\delta_v \le \hat V_t \le V^{\max}-\delta_v
\]

\[
\hat S_t \le \bar S - \delta_l
\]

\[
-\bar P_{tr}^{\mathrm{exp}}
\le
\hat P_{tr,t}
\le
\bar P_{tr}^{\mathrm{imp}}
\]

### 最终训练目标

\[
\max_{\theta}
\mathbb E
\left[
\sum_t \gamma^t r_t(s_t,a_t)
\right]
\quad
\text{s.t.}\quad
a_t = \Pi_{\mathcal S(s_t)}(\pi_\theta(s_t))
\]

### PPT 可强调

- 策略负责经济性
- 投影层负责安全性
- 比 penalty safe 更接近硬约束控制

## 3. 一句话区分 7 个方案

- `MISOCP`：集中式、全网物理约束最完整的混合整数二阶锥优化
- `single-agent MPC + perfect prediction`：单用户局部滚动优化，未来信息完全准确
- `single-agent MPC + LSTM prediction`：单用户局部滚动优化，未来信息由 LSTM 预测
- `ADMM MPC + LSTM prediction`：多用户分布式 MPC，通过 ADMM 实现协调
- `DRL no safe`：仅追求经济性的 DRL 基线
- `DRL + penalty safe`：通过奖励惩罚项抑制安全违约的软约束 DRL
- `DRL + projection safe`：通过安全投影层实时修正动作的硬安全 DRL

## 4. PPT 排版建议

每个方案一页时，建议固定成 4 个模块：

1. 核心思想
2. 目标函数
3. 约束条件
4. 与其他方案的关键区别

如果你后面需要，我可以继续把这个文档再精简成“每页 PPT 直接可贴的 5 行版”。 
