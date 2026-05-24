# 项目目标函数与奖励函数完整总结

## 📌 第一部分：整体研究目标

### 问题定义
**多智能体储能协控问题 (MADRL_ESS)**

```
给定：
  - 低压配电网拓扑 (SimBench "1-LV-rural1--0-sw")
  - N=3个prosumer (家庭+电池+PV)
  - 24小时的负荷、PV、电价序列
  - 电网物理约束 (电压、线路、变压器)

目标：
  最大化 联合储能套利收益 + 满足安全约束
  同时 学习分布式策略（不依赖全局通信）
```

### 优化目标层级

```
┌─────────────────────────────────────────────────────┐
│ 层级1: 研究目标 (Research Objective)                │
│ "对比不同控制策略的经济性和安全性"                   │
│  - MISOCP (全局最优基线)                            │
│  - MPC (滚动优化基线)                                │
│  - MADRL (多智能体强化学习)                          │
└─────────────────────────────────────────────────────┘
           ↓ 具体化为
┌─────────────────────────────────────────────────────┐
│ 层级2: 控制目标 (Control Objective)                 │
│ 最大化: J(π) = E[∑ γ^t (R_economic - R_penalty)]   │
│                                                      │
│ 其中:                                               │
│  - R_economic: 储能套利收益                         │
│  - R_penalty: 安全约束违反的惩罚                     │
│  - γ: 折扣因子 (0.999)                              │
│  - π: agent的策略                                   │
└─────────────────────────────────────────────────────┘
           ↓ 每一步具体化为
┌─────────────────────────────────────────────────────┐
│ 层级3: 即时奖励 (Step Reward)                       │
│ r_t = R_storage - R_safety - R_boundary - ...      │
│                                                      │
│ (每一步计算的标量奖励，累积后形成轨迹回报)          │
└─────────────────────────────────────────────────────┘
```

---

## 📐 第二部分：即时奖励函数数学形式

### 🎯 **总奖励公式**

```
r_i(t) = r_storage(t) 
       - r_action_boundary(t) 
       - r_soc_regularization(t) 
       + r_throughput_bonus(t)
       - r_voltage_safety(t)
       - r_line_safety(t)
       - r_trafo_safety(t)

其中 i ∈ {0,1,2} 表示三个agent的索引
```

### 📊 **六大奖励分量详解**

---

#### **1️⃣ 储能套利收益** (Primary Objective)

$$r_{storage,i}(t) = (\max(0, -P_{charge,i}) - \max(0, P_{charge,i})) \cdot (p_t + m_{import}) \cdot \Delta t$$

其中：
- $P_{charge,i} \in [-50, 50]$ kW：电池充放电功率（正=充，负=放）
- $p_t$：当前时刻批发电价 €/kWh
- $m_{import}$：进口加价 (默认0)
- $\Delta t = 0.25$ h：时间步长

**逻辑**：
- 放电时 ($P_{charge} < 0$)：$-P_{charge} > 0$ → 正收益
- 充电时 ($P_{charge} > 0$)：$-P_{charge} < 0$ → 负收益（花钱）
- 高价时应该放电，低价时应该充电 → **套利激励**

**数值示例**：
```
场景1: 低价充电
  P_charge = 50 kW (充电)
  p_t = 0.05 €/kWh
  r_storage = (0 - 50) × (0.05 + 0) × 0.25 = -0.625 €
  
场景2: 高价放电
  P_charge = -50 kW (放电)
  p_t = 0.15 €/kWh
  r_storage = (50 - 0) × (0.15 + 0) × 0.25 = +1.875 €
```

---

#### **2️⃣ SOC动作边界惩罚** (Constraint Protection)

$$r_{action\_boundary,i}(t) = -w_{action} \cdot |P_{charge,i}(t)| \cdot \mathbb{1}_{boundary\_push,i}(t)$$

其中：
- $w_{action} = 0.05$：惩罚权重
- $\mathbb{1}_{boundary\_push}$：二进制标志

$$\mathbb{1}_{boundary\_push,i}(t) = \begin{cases} 
1 & \text{if } (P_{charge,i} < 0 \text{ AND } SOC_i \leq 0.05 + 0.02) \\
  & \text{OR } (P_{charge,i} > 0 \text{ AND } SOC_i \geq 0.95 - 0.02) \\
0 & \text{otherwise}
\end{cases}$$

**逻辑**：
- **防止极限操作**：在SOC极端附近（距离极限值仅2%）不允许充放电
- $\epsilon = 0.02$：硬约束的缓冲区

**数值示例**：
```
场景1: 正常操作（允许）
  SOC = 0.5, P_charge = 30 kW
  boundary_push = 0
  r_action_boundary = 0
  
场景2: 不允许的操作（被惩罚）
  SOC = 0.06 (接近下限0.05)
  P_charge = -40 kW (想放电)
  0.06 ≤ 0.05 + 0.02 = 0.07 ✓
  boundary_push = 1
  r_action_boundary = -0.05 × 40 = -2.0 €
```

---

#### **3️⃣ SOC软边界L2正则** (Incentive for Balanced Operation)

$$r_{soc\_reg,i}(t) = -w_{soc\_reg} \cdot \left[ \max(0, S_{low} - SOC_i)^2 + \max(0, SOC_i - S_{high})^2 \right]$$

其中：
- $w_{soc\_reg} = 0.005$：正则化权重
- $S_{low} = 0.05 + 0.02 = 0.07$：软下界
- $S_{high} = 0.95 - 0.02 = 0.93$：软上界

**逻辑**：
- **鼓励在中间范围操作**：不硬约束，但有二次惩罚
- SOC离中心越远，惩罚越大（二次项）

**数值示例**：
```
场景1: SOC在软范围内 [0.07, 0.93]
  SOC = 0.5
  r_soc_reg = 0
  
场景2: SOC低于软下界
  SOC = 0.04
  r_soc_reg = -0.005 × (0.07 - 0.04)^2 = -0.000045 €
  
场景3: SOC过高
  SOC = 0.94
  r_soc_reg = -0.005 × (0.94 - 0.93)^2 = -0.000005 €
```

---

#### **4️⃣ 吞吐奖励** (Early Exploration Bonus)

$$r_{throughput,i}(t) = w_{throughput}(p) \cdot |P_{charge,i}(t)| \cdot \Delta t$$

其中：
- 权重随训练进度衰减：
$$w_{throughput}(p) = w_{max} \cdot \max\left(0, \frac{0.80 - p}{0.60}\right)$$

- $w_{max} = 0.002$ €/kWh
- $p = \frac{t_{current} \cdot N_{envs}}{t_{total}}$：训练进度 ∈ [0, 1]

**逻辑**：
- **早期鼓励探索**：前期提供额外收益激励充放电
- **随训练衰减**：当p > 0.8时，bonus完全消失
- **帮助冷启动**：初期buffer不足时还有探索动力

**数值示例**：
```
场景1: 早期训练 (p=0.2)
  w_throughput = 0.002 × (0.80 - 0.2) / 0.60 = 0.002 €/kWh
  P_charge = 40 kW
  r_throughput = 0.002 × 40 × 0.25 = 0.02 €
  
场景2: 中期训练 (p=0.5)
  w_throughput = 0.002 × (0.80 - 0.5) / 0.60 = 0.001 €/kWh
  P_charge = 40 kW
  r_throughput = 0.001 × 40 × 0.25 = 0.01 €
  
场景3: 后期训练 (p=0.85)
  w_throughput = 0.002 × max(0, (0.80 - 0.85) / 0.60) = 0 €/kWh
  r_throughput = 0 (完全消失)
```

---

#### **5️⃣ 电压安全惩罚** (Voltage Constraint)

$$r_{voltage,i}(t) = -n \cdot w_v \cdot \psi_v \cdot \frac{v_{viol,i}(t)}{\sum_j v_{viol,j}(t)}$$

其中：
- $n = 3$：agent数量
- $w_v = 1.0$：电压权重
- $\psi_v$：全网电压平方和惩罚
$$\psi_v = \sum_{\text{all buses}} \left[\max(0, V_{min} - V) + \max(0, V - V_{max})\right]^2$$
- $v_{viol,i}$：agent i处的电压违反量
$$v_{viol,i} = \max(0, 0.95 - V_i) + \max(0, V_i - 1.05)$$

**逻辑**：
- **集中惩罚造成问题的agent**：电压违反多的agent获得更大负奖励
- **全网聚合**：$\psi_v$计算所有总线的平方和，所以大规模违反被二次放大
- **奖励分配**：按比例分配给造成问题的agent

**数值示例**：
```
场景1: 电压正常
  所有V ∈ [0.95, 1.05]
  v_violation = 0
  ψ_v = 0
  r_voltage = 0
  
场景2: Agent0导致电压低
  V_0 = 0.93 (低0.02), V_1=0.98, V_2=0.96
  v_viol_0 = 0.95 - 0.93 = 0.02
  v_viol_1 = 0, v_viol_2 = 0
  ψ_v = 0.02² + 0 + 0 = 0.0004
  r_voltage_0 = -3 × 1.0 × 0.0004 × 0.02/0.02 = -0.0012 €
  r_voltage_1 = -3 × 1.0 × 0.0004 × 0/0.02 = 0
  r_voltage_2 = -3 × 1.0 × 0.0004 × 0/0.02 = 0
```

---

#### **6️⃣ 线路/变压器安全惩罚** (Similar to Voltage)

$$r_{line,i}(t) = -w_{line} \cdot \psi_{line}(t)$$
$$r_{trafo,i}(t) = -w_{trafo} \cdot \psi_{trafo}(t)$$

其中：
- $w_{line} = 1.0$, $w_{trafo} = 1.0$
- $\psi_{line} = \sum_{\text{all lines}} \left[\max(0, L - L_{max})\right]^2 / 100^2$
- $\psi_{trafo} = \sum_{\text{all trafos}} \left[\max(0, T - T_{max})\right]^2 / 100^2$

**逻辑**：
- 不分配给specific agent（全部agent平等受罚）
- 任何线路或变压器过载都影响全局

---

## 📋 第三部分：三种MADRL方案的区别

### 方案对比矩阵

| 方案 | 奖励构成 | 约束方式 | 安全性 | 经济性 |
|------|--------|---------|--------|--------|
| **MADRL_BASE** | 仅`r_storage` | 无（通过环境物理约束） | 低 | 高 |
| **MADRL_PENALTY** | 全部6项 | 通过惩罚项 | 中 | 中 |
| **MADRL_PROJECTION** | 全部6项 + 动作投影 | 惩罚 + 动作裁剪 | 高 | 低 |

### 方案1: MADRL_BASE (Baseline)

```
r_i = r_storage - 0 - 0 + 0 - 0 - 0 - 0
    = r_storage only

特点:
  ✓ 简单 - agent只学习套利
  ✗ 无约束 - 可能严重违反电网
  ✗ 故障风险高
```

### 方案2: MADRL_PENALTY (论文主方案)

```
r_i = r_storage 
    - r_action_boundary 
    - r_soc_regularization 
    + r_throughput_bonus
    - r_voltage_safety
    - r_line_safety
    - r_trafo_safety

特点:
  ✓ 完整约束 - 通过奖励惩罚
  ✓ 可学习性好 - agent学会权衡
  ✓ 现实可行 - 逐渐学会安全
  ✗ 初期可能不安全 - 学习过程中可能违反
```

### 方案3: MADRL_PROJECTION (安全保证)

```
π' = argmin_π ||π - π_actor||²  s.t. 动作满足线性化约束

特点:
  ✓ 最安全 - 强制满足约束
  ✓ 零违反 - 不可能违反约束
  ✗ 学习变难 - 动作空间被限制
  ✗ 经济性下降 - 无法探索"临界"操作
```

---

## 🧮 第四部分：完整计算流程演示

### 场景设置

```
时刻 t = 某个下午3点15分
配置:
  - 3个agent，SOC分别为 [0.4, 0.5, 0.6]
  - 负荷: [15, 20, 18] kW
  - PV总产出: 25 kW
  - 批发电价: 0.12 €/kWh
  - 训练进度: p = 0.3 (30%)

Agent输出动作: 
  a_0 = [0.8, -0.5]   (想充满，不用PV)
  a_1 = [0.0,  0.5]   (不动电池，用50%PV)
  a_2 = [-0.6, 1.0]   (放电，用100%PV)
```

### 逐步计算

#### Step 1: 转换动作

```
电池功率:
  P_charge_0 = 0.8 × 50 = 40 kW (充电)
  P_charge_1 = 0.0 × 50 = 0 kW
  P_charge_2 = -0.6 × 50 = -30 kW (放电)

PV利用率:
  util_0 = (-0.5 + 1) / 2 = 0.25 → PV_eff_0 = 25 × 0.25 × (1/3) = 2.08 kW
  util_1 = (0.5 + 1) / 2 = 0.75 → PV_eff_1 = 25 × 0.75 × (1/3) = 6.25 kW
  util_2 = (1.0 + 1) / 2 = 1.0 → PV_eff_2 = 25 × 1.0 × (1/3) = 8.33 kW
  
净注入功率:
  net_0 = 15 - 2.08 + 40 = 52.92 kW (吸收)
  net_1 = 20 - 6.25 + 0 = 13.75 kW (吸收)
  net_2 = 18 - 8.33 - 30 = -20.33 kW (注入)
```

#### Step 2: 潮流计算

```
潮流结果(假设):
  V_0 = 1.04 pu, V_1 = 0.98 pu, V_2 = 0.96 pu
  线路负荷: [80%, 85%, 90%]
  变压器负荷: [75%]
  
电压违反:
  v_viol_0 = max(0, 0.95-1.04) + max(0, 1.04-1.05) = 0 + 0.01 = 0.01
  v_viol_1 = max(0, 0.95-0.98) + max(0, 0.98-1.05) = 0 + 0 = 0
  v_viol_2 = max(0, 0.95-0.96) + max(0, 0.96-1.05) = 0 + 0 = 0
  
全网平方和:
  ψ_v = 0.01² + 0 + 0 = 0.0001
  ψ_line = 0 (未超过100%)
  ψ_trafo = 0 (未超过100%)
```

#### Step 3: 奖励计算

**Agent 0**:
```
1. r_storage_0 = (0 - 40) × (0.12 + 0) × 0.25 = -1.20 €
   (充电，花钱)

2. boundary_push_0 = (40 > 0 AND 0.4 ≥ 0.93) ? 1 : 0 = 0
   r_action_boundary_0 = 0
   (SOC 0.4 远离上界，允许充电)

3. r_soc_reg_0 = -0.005 × [max(0, 0.07-0.4)² + max(0, 0.4-0.93)²]
               = -0.005 × [0 + 0.2809]
               = -0.001405 €

4. w_throughput = 0.002 × (0.80-0.3)/0.60 = 0.00167 €/kWh
   r_throughput_0 = 0.00167 × 40 × 0.25 = 0.0167 €

5. r_voltage_0 = -3 × 1.0 × 0.0001 × 0.01/0.01 = -0.0003 €

6. r_line_0 = -1.0 × 0 = 0
   r_trafo_0 = -1.0 × 0 = 0

TOTAL:
  r_0 = -1.20 - 0 - 0.001405 + 0.0167 - 0.0003 - 0 - 0
      = -1.1850 €
```

**Agent 1** (类似计算):
```
1. r_storage_1 = (0 - 0) × 0.12 × 0.25 = 0 €
2. boundary_push_1 = 0
   r_action_boundary_1 = 0
3. r_soc_reg_1 = -0.005 × 0 = 0 (SOC 0.5 在软范围内)
4. r_throughput_1 = 0 (不充放电)
5. r_voltage_1 = 0
6. r_line_1 = 0, r_trafo_1 = 0

TOTAL: r_1 = 0 €
```

**Agent 2** (放电):
```
1. r_storage_2 = (30 - 0) × 0.12 × 0.25 = +0.90 €
   (放电，赚钱!)
2. boundary_push_2 = 0 (SOC 0.6不近边界)
   r_action_boundary_2 = 0
3. r_soc_reg_2 = -0.005 × 0 = 0 (SOC 0.6在范围内)
4. r_throughput_2 = 0.00167 × 30 × 0.25 = 0.0125 €
5. r_voltage_2 = 0
6. r_line_2 = 0, r_trafo_2 = 0

TOTAL: r_2 = +0.90 + 0.0125 = +0.9125 €
```

---

## 🎯 第五部分：学习过程中的权衡

### 奖励成分的相对大小

```
典型情况下，各奖励分量的量级 (€/step):

r_storage:           -0.5 ~ +1.5  (最大！取决于价格和功率)
r_throughput_bonus:   0 ~ +0.05  (前80%训练阶段)
r_action_boundary:    0 ~ -0.5  (仅在边界附近)
r_soc_regularization: 0 ~ -0.001 (总是小)
r_voltage/line/trafo: 0 ~ -0.001 (仅有大规模违反时)

比例: 经济收益 >> 约束惩罚
```

### 学习动态

```
早期 (p=0~0.3):
  ✓ throughput_bonus活跃 → 鼓励探索充放
  ✗ 可能探索不安全操作 → 获得电压/线路惩罚
  → Agent学会权衡: "高收益但风险" vs "安全但低收"

中期 (p=0.3~0.8):
  ✓ throughput_bonus逐渐衰减 → 探索激励减弱
  ✓ 约束惩罚持续 → Agent更重视安全
  → Agent找到"最优前沿" (Pareto frontier)

后期 (p>0.8):
  ✓ throughput_bonus完全消失
  ✓ 只有经济性和约束惩罚 → 稳定策略
  → Agent倾向保守的、满足约束的策略
```

---

## 📊 第六部分：与基线方法的对标

### MISOCP (全局最优基线)

```
问题:
  maximize ∑_t ∑_i (放电_i - 充电_i) × p_t × dt
  subject to:
    所有时刻所有agent的硬约束 (电压、线路、变压器)
    所有SOC约束 (0.05 ≤ SOC ≤ 0.95)

特点:
  ✓ 全局最优 (MISOCP找到理论最优解)
  ✓ 完全满足约束 (不可能违反)
  ✗ 计算昂贵 O(N! × T)
  ✗ 集中式 (需要全局信息)
```

### LOCAL_MPC (独立MPC基线)

```
每个agent独立求解:
  maximize ∑_τ=t^{t+H} (discharge_i - charge_i) × p_τ × dt
  subject to:
    agent i局部约束 (SOC、功率限制)
    假设其他agent不变

特点:
  ✓ 计算快 (分布式，每个N个小问题)
  ✓ 简单
  ✗ 可能网络约束冲突 (多个agent同时吸收功率)
  ✗ 可能有电压/线路违反
```

### ADMM_MPC (分布式MPC基线)

```
使用ADMM加入网络协调项:
  L(x,y,λ) = 各agent局部目标 + 协调项 + Lagrange乘子

特点:
  ✓ 分布式+协调
  ✓ 能处理网络约束
  ✗ 收敛需要多次迭代
  ✗ 滚动视界可能冲突
```

### MADRL_PENALTY (论文方案)

```
多智能体强化学习:
  π_i = neural_network(observation_i)
  
  r_i = r_storage - r_penalty - r_safety
  
  学习目标:
    max E[∑_t γ^t r_i(t) | π_i]

特点:
  ✓ 完全分布式 (无需通信)
  ✓ 快速推理 (前向一次)
  ✓ 学习适应性 (可对新场景泛化)
  ✗ 初期不安全 (学习过程中可能违反)
  ✗ 最优性无保证 (只是局部最优)
```

---

## 🔑 核心要点总结

| 问题 | 答案 |
|------|------|
| **主要优化目标** | 最大化储能套利收益 (€) |
| **约束方式** | 多项惩罚权重 (安全性换经济性) |
| **奖励来源** | 6项分量累积 |
| **学习策略** | 从探索→权衡→稳定 |
| **与基线区别** | 分布式、快速、适应强 |
| **权衡点** | MADRL_PENALTY最平衡 |

