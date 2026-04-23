# MADRL 训练停滞修复计划

## 0. 目标

当前 `train_base` 的 MADRL 训练出现两个直接症状：

```text
1. 训练奖励长期为负。
2. 评估 rollout 中电池几乎不充电也不放电，storage_profit_total_eur = 0。
```

本计划的目标不是重新定义 compare 排名口径，而是在保持七个方案统计口径对齐 MISOCP 的前提下，让 MADRL 真正学到可执行的储能套利策略。

最终 compare 主指标仍然保持：

```text
maximize storage_profit

storage_profit =
    storage_discharge_revenue
  - storage_charge_cost
```

所有 controller 和 notebook 输出必须继续对齐这些列：

```text
storage_charge_cost_total_eur
storage_discharge_revenue_total_eur
storage_profit_total_eur
storage_objective_total_eur
storage_purchase_cost_eur
storage_sale_revenue_eur
storage_total_profit_eur
storage_objective_eur
```

如果后续决定改变储能使用的价格口径，例如从 `import_price = wholesale_price + 0.20` 改成纯 `wholesale_price`，必须同时修改 MADRL、Local MPC、ADMM MPC、Global MISOCP 和 compare 统计。禁止只改 MADRL reward。

## 1. 当前诊断结论

### 1.1 电价归一化不是唯一主因

当前 `train_base` 使用 2019 训练集拟合 `wholesale_price` 的 `robust_tanh` 归一化：

```text
q01    = -0.00305
median =  0.07566
iqr    =  0.05914
q99    =  0.36835

normalized_price =
    tanh((clip(price, q01, q99) - median) / iqr / 2.0)
```

2020 测试窗口 `2020-04-01` 到 `2020-04-15` 的批发电价明显高于 2019 训练集：

```text
2019 train median wholesale = 0.07566 EUR/kWh
2020 test  median wholesale = 0.16764 EUR/kWh
```

归一化后 2020 价格整体偏高：

```text
2020 normalized median   = 0.651
2020 normalized >= 0.80  = 23.6%
2020 normalized >= 0.90  = 9.4%
2020 normalized >= 0.95  = 1.4%
```

结论：

```text
2020 电价确实在 2019 normalizer 下整体偏高，但并没有全部饱和到 +1。
归一化会削弱低价和高价之间的可分辨性，但不是电池完全不动的唯一原因。
```

### 1.2 `+0.20` import markup 显著削弱储能套利

当前 reward 使用：

```text
storage_price_t = import_price_t
import_price_t = wholesale_price_t + 0.20
```

在 2020 测试窗口中：

```text
2020 import price min    = 0.19751 EUR/kWh
2020 import price median = 0.36764 EUR/kWh
2020 import price max    = 0.50993 EUR/kWh
```

电池效率为：

```text
eta = 0.95
round_trip_factor = eta^2 = 0.9025
required_sell_buy_ratio = 1 / 0.9025 = 1.108
```

因此完整一充一放要赚钱，卖价至少需要比买价高约 `10.8%`。

诊断统计：

```text
真实未来电价窗口，2019 train 可套利比例：
  markup = 0.0  -> 88.97%
  markup = 0.2  -> 46.88%

LSTM shared_data 预测窗口，2019 train 可套利比例：
  markup = 0.2  -> 27.83%
```

结论：

```text
训练时 actor 看到的是 LSTM 预测序列。
这些预测序列比真实电价更平滑，高价峰值被压低。
在当前 import_price 口径下，超过 72% 的训练窗口没有明显储能套利价值。
```

### 1.3 训练负奖励主要来自本地 SoC action penalty

当前保存的 `madrl_base` rollout 显示：

```text
battery_charge_total       = 0
battery_discharge_total    = 0
storage_profit_total_eur   = 0
battery_net_power_mean_abs = 0
```

但 agent 不是输出真正的 idle，而是大量请求在 SOC 下界继续放电：

```text
soc_t     = 0.05
soc_min   = 0.05
e_bat_req ~= -50 kW
e_bat     = 0 kW
```

因为初始 SOC 等于下界，放电请求会被本地可行性裁剪为 0。

训练最后 100 个 episode 的 reward 分解：

```text
episode_total_reward mean_last100 ~= -247.53
r_storage_profit     mean_last100 ~= -0.033
r_soc_pen            mean_last100 ~= -247.50
r_safe_v             mean_last100 = 0
r_safe_line          mean_last100 = 0
r_safe_trafo         mean_last100 = 0
```

结论：

```text
当前训练差的直接原因不是电池执行了很多亏损套利，而是 policy 持续请求本地不可行动作。
执行动作被裁剪成 0，统计上电池摆烂；请求动作和执行动作之间的 gap 通过 r_soc_pen 持续扣分。
```

## 2. 根因排序

按优先级排序，当前 MADRL 训练停滞的根因是：

```text
P0. 非法放电请求被裁剪，r_soc_pen 成为主要负奖励。
P1. 初始 SOC = soc_min，训练一开始放电不可行，只能先充电。
P2. 当前 import markup = 0.20 使储能套利窗口大幅减少。
P3. LSTM 预测序列平滑峰谷，进一步削弱可套利信号。
P4. 绝对电价用 2019 robust_tanh 拟合，2020 测试期整体偏高，低价信号不够清晰。
P5. reward 是真实利润，充电先扣分、放电后收益，credit assignment 难。
```

因此修复顺序应当是：

```text
先修动作可行性学习问题，再增强价格特征，最后再决定是否改变储能价格口径。
```

不建议第一步就只改归一化，因为归一化无法解决 `soc_min` 上持续请求放电的问题。

## 3. 修复原则

### 3.1 保持统计口径对齐 MISOCP

所有输出必须继续使用同一套储能收益定义：

```text
e_bat > 0 表示充电
e_bat < 0 表示放电

charge_kw    = max(e_bat, 0)
discharge_kw = max(-e_bat, 0)

storage_charge_cost =
    charge_kw * dt * storage_price

storage_discharge_revenue =
    discharge_kw * dt * storage_price

storage_profit =
    storage_discharge_revenue - storage_charge_cost
```

`storage_price` 的选择必须是全局配置项，不能在不同 controller 中隐式分叉。

### 3.2 训练 reward 可以有 shaping，但 compare 不能混入 shaping

允许为了训练加入辅助项，例如 terminal SoC value、potential shaping、动作可行性辅助损失。

但以下 compare 输出只能记录真实经济口径：

```text
storage_profit_total_eur
storage_objective_total_eur
returned_primary_objective_eur
```

训练辅助项必须单独输出，例如：

```text
r_terminal_soc_value
r_price_spread_shaping
r_action_gap_pen
```

禁止把 shaping 后的训练 reward 当成 MISOCP 对齐指标。

### 3.3 训练集拟合 normalizer，不用测试集泄漏

归一化 state 仍然只能用 train split 拟合。

可以增加更稳健的特征，例如相对电价、窗口价差、窗口 rank，但不能用 2020 测试集重新拟合 normalizer 来提升测试表现。

## 4. 阶段 A：冻结诊断和基线

### 4.1 新增价格诊断入口

建议新增一个明确 owner：

```text
scripts/diagnostics/madrl_price_diagnostics.py
```

输出：

```text
notebooks/record/diagnostics/madrl_price_diagnostics.json
notebooks/record/diagnostics/madrl_price_diagnostics.parquet
```

统计项：

```text
normalization_state.wholesale_price
raw_wholesale_train quantiles
raw_wholesale_test quantiles
normalized_train quantiles
normalized_test quantiles
normalized saturation ratio
import_price quantiles
perfect future window arbitrage ratio
LSTM shared_data window arbitrage ratio
```

验收标准：

```text
1. 能复现当前诊断数字。
2. 诊断输出记录 shared_data_signature。
3. 诊断输出记录 storage_price_mode 和 import_price_markup_eur_per_kwh。
```

### 4.2 新增训练动作诊断

建议新增或扩展训练 summary：

```text
requested_battery_power_kw_mean_abs
executed_battery_power_kw_mean_abs
controller_action_gap_mean
soc_penalty_total
storage_profit_total
charge_steps
discharge_steps
idle_steps
```

当前必须能明确显示：

```text
requested_battery_power_kw_mean_abs >> executed_battery_power_kw_mean_abs
r_soc_pen dominates episode_total_reward
```

验收标准：

```text
1. train_reward_summary.json 能看到 r_soc_pen、r_storage_profit 的 last100 平均值。
2. rollout record 中能区分 e_bat_req 和 e_bat。
3. 电池 idle 不能只看 e_bat，必须同时看 e_bat_req。
```

## 5. 阶段 B：修复本地动作可行性学习

### 5.0 执行状态（2026-04-23）

已落地第一版：

```text
1. 新增 local_action_penalty_mode / local_action_penalty_weight 配置。
2. train_base 默认使用 diagnostic_only，不再把本地动作 gap 扣进训练 reward。
3. 训练和 rollout 仍保留 e_bat_req、e_bat、controller_action_gap、r_soc_pen 统计口。
4. notebook 外部训练 payload 和 compare 模型包校验已纳入 local action penalty 配置。
```

仍留到下一阶段：

```text
把本地可行动作映射统一接入 rollout、target-Q、actor-loss，使 critic 和 actor loss 看到的动作完全等于 env 实际执行动作。
```

### 5.1 当前问题

`TrainRunner` 对所有 MADRL 都会执行本地 SoC 可行性裁剪：

```text
raw actor action -> enforce_local_action_feasibility_torch -> executed action
```

然后 `merge_action_info_into_step_info(... apply_action_penalty=True)` 会把请求动作和执行动作的 gap 作为 `r_soc_pen` 扣分。

这导致：

```text
1. 初始 SOC 在下界时，请求放电必然被裁剪。
2. policy 可能持续输出放电。
3. 执行动作是 0，所以经济收益为 0。
4. r_soc_pen 持续为负，训练奖励长期为负。
```

### 5.2 配置拆分

建议把本地动作 gap penalty 从 `w_soc_pen` 中拆出来：

文件：

```text
configs/experiment_config.py
```

新增字段：

```python
local_action_penalty_weight: float = 0.0
local_action_penalty_mode: str = "diagnostic_only"
```

推荐模式：

```text
diagnostic_only:
    记录 e_bat_req、e_bat、action_gap，但不扣训练 reward。

penalty:
    按 local_action_penalty_weight 扣 reward。

strict:
    非法动作直接失败，用于 debug，不用于正式训练。
```

`train_base` 推荐：

```text
local_action_penalty_mode = "diagnostic_only"
local_action_penalty_weight = 0.0
```

`train_base_safe` 可先保持低权重：

```text
local_action_penalty_mode = "penalty"
local_action_penalty_weight = 0.1
```

`train_projection_safe` 由 projector 处理安全约束，保留单独 projector diagnostics。

### 5.3 让 actor 学到可执行动作，而不是只在 env 外裁剪

只把 penalty 设为 0 还不够，因为 actor 仍然可能输出不可行动作。

推荐增加一个统一的本地可行动作 owner：

```text
controllers/madrl/local_action_feasibility.py
```

职责：

```text
1. 根据 safety_local 计算 battery lower/upper bound。
2. 将 actor 的 normalized action 映射到当前 SOC 下可行区间。
3. 同一个函数用于 rollout、target-Q、actor-loss。
```

第一版保守做法：

```text
rollout action:
    actor raw action -> local feasibility projection -> env

critic target action:
    target actor raw action + noise -> local feasibility projection -> target Q

actor loss action:
    actor raw action -> local feasibility projection -> critic
```

这样 policy 优化看到的动作和环境实际执行动作一致，减少“训练的是 raw action，环境执行的是 clipped action”的错位。

受影响符号：

```text
scripts/train.py::select_action_batch_with_info
scripts/train.py::build_shared_update_ctx
controllers/madrl/base_agent.py 中 actor loss 相关路径
controllers/madrl/safety_projector.py 中现有 local bounds 逻辑
```

编辑这些符号前必须按 AGENTS.md 运行 GitNexus impact analysis。

### 5.4 验收标准

训练 500 episodes 后：

```text
r_soc_pen mean_last100 明显下降，目标 < 10
requested_battery_power_kw_mean_abs 和 executed_battery_power_kw_mean_abs 接近
battery_net_power_kw_mean_abs > 1 kW
storage_profit_total_eur 不再恒等于 0
```

如果 `storage_profit_total_eur` 仍为 0，但 `r_soc_pen` 已经下降，说明动作可行性问题已修复，下一步重点转向价格信号和 reward shaping。

## 6. 阶段 C：增强电价观测，不破坏 normalizer 合同

### 6.1 当前观测问题

actor 当前主要看到：

```text
wholesale_price_seq
```

这个序列是绝对批发电价，经 2019 train 的 `robust_tanh` 归一化。

问题：

```text
1. actor 不直接看到 import_price。
2. actor 不直接看到当前窗口内的价差。
3. 2020 整体偏高时，绝对价格特征更像“都很高”。
4. LSTM 预测平滑后，峰谷套利信号更弱。
```

### 6.2 新增相对价格特征

建议新增 sequence features：

```text
wholesale_price_rel_seq
wholesale_price_spread_seq
wholesale_price_rank_seq
```

定义：

```text
wholesale_price_rel_seq[h] =
    wholesale_price_seq[h] - wholesale_price_seq[0]

wholesale_price_spread_seq[h] =
    wholesale_price_seq[h] - mean(wholesale_price_seq)

wholesale_price_rank_seq[h] =
    rank(wholesale_price_seq[h] within horizon) / (H - 1)
```

可选新增：

```text
storage_price_seq
storage_price_rel_seq
```

如果 `storage_price_mode = real_time_price`，则：

```text
storage_price_seq = wholesale_price_seq + import_price_markup
```

注意：

```text
rank 和 rel 特征只使用当前预测窗口内信息，不使用测试集全局统计，不构成数据泄漏。
```

### 6.3 归一化策略

推荐：

```text
absolute wholesale_price_seq:
    继续使用 train-fitted robust_tanh。

relative price seq:
    用 train-fitted price_iqr 缩放后 tanh。

rank seq:
    直接落在 [0, 1]，不再 robust_tanh。
```

新增 normalizer state 字段：

```text
price_iqr_for_relative_features
```

不要用 2020 test 重新拟合。

### 6.4 shared_data 合同升级

由于 `train_base` 当前使用 precomputed shared_data，新增观测字段必须同步升级：

```text
predictors/shared_data.py
envs/observation/default_builder.py
envs/observation/normalization.py
scripts/utils/grid_notebook_workflow.py
notebooks/forecast/forecast_lstm.ipynb
```

必须提升 shared-data schema 或 signature，使旧 shared_data 不能静默复用。

合同失败信息必须说明：

```text
旧对象：当前 shared_data_dir/signature
新合同：需要包含 price relative/rank features
重跑入口：notebooks/forecast/forecast_lstm.ipynb
```

### 6.5 验收标准

训练前诊断应输出：

```text
relative price median ~= 0
rank seq min/max in [0, 1]
window spread 与 storage_profit 有正相关
```

训练后：

```text
charge 动作更集中在低 rank 区间
discharge 动作更集中在高 rank 区间
```

## 7. 阶段 D：明确储能价格口径

### 7.1 当前口径

当前口径是：

```text
storage_price = import_price = wholesale_price + 0.20
```

这个口径已经被 MADRL、Local MPC、ADMM MPC、MISOCP 和 compare 统计使用。

优点：

```text
统计已经对齐。
```

缺点：

```text
0.20 EUR/kWh 的常数 markup 明显减少训练集内可套利窗口。
如果这是模拟用户零售购电价，储能套利本来就应该少。
如果这是批发市场储能，markup 不应该进入储能充放电价格。
```

### 7.2 决策分支

必须在项目层面二选一。

方案 D1：保持当前 real_time import price

```text
storage_price_mode = "real_time_price"
storage_price = wholesale_price + import_price_markup
```

含义：

```text
储能按终端用户购电价套利。
预期可套利机会少。
MADRL 应重点解决 action feasibility 和 reward shaping。
```

方案 D2：切换为 wholesale storage price

```text
storage_price_mode = "wholesale_price"
storage_price = wholesale_price
```

含义：

```text
储能按批发市场价格套利。
可套利窗口大幅增加。
必须同时更新 MADRL、Local MPC、ADMM MPC、Global MISOCP、compare 统计和 notebook meta。
```

### 7.3 推荐

推荐先执行：

```text
第一轮修复：保持 D1，不改 MISOCP 价格口径。
```

原因：

```text
当前最大故障是 actor 请求不可行动作导致 r_soc_pen 爆炸。
如果此时改价格口径，会混淆问题来源。
```

当阶段 B 和 C 完成后，再做一组对照实验：

```text
Experiment A: storage_price_mode = real_time_price
Experiment B: storage_price_mode = wholesale_price
```

两组实验必须都重跑：

```text
global_MISOCP.ipynb
local_MPC.ipynb
ADMM_mpc.ipynb
train_base.ipynb
train_base_safe.ipynb
train_projection_safe.ipynb
mainline_compare.py
```

## 8. 阶段 E：加入训练 shaping，但不污染 compare

### 8.1 当前 reward 的 credit assignment 问题

当前真实储能利润：

```text
charge at t:
    immediate reward < 0

discharge at t+k:
    delayed reward > 0
```

在初始 SOC 低、低价窗口少、预测平滑的情况下，actor 很容易学到：

```text
不充电可以避免即时负收益。
不可行放电被裁剪后执行 idle。
```

### 8.2 推荐 shaping

第一优先级：terminal SoC value

训练 reward 增加：

```text
r_terminal_soc_value =
    terminal_reference_price * final_stored_energy
```

只在 episode 末尾加入，或者使用 potential-based shaping：

```text
phi_t = reference_price_t * stored_energy_t

r_shaping_t =
    gamma * phi_{t+1} - phi_t
```

推荐 `reference_price_t` 选择：

```text
horizon_mean_storage_price
```

这样低价充电不会只表现为即时负奖励，stored energy 的未来价值会进入训练信号。

第二优先级：price spread shaping

```text
r_price_spread =
    discharge_energy * max(price_t - horizon_mean_price, 0)
  + charge_energy    * max(horizon_mean_price - price_t, 0)
```

这会改变训练目标，因此必须作为辅助 shaping，并单独输出。

### 8.3 配置字段

建议新增：

```python
training_reward_shaping_mode: str = "none"
terminal_soc_value_weight: float = 0.0
price_spread_shaping_weight: float = 0.0
```

第一轮建议：

```text
training_reward_shaping_mode = "terminal_soc_value"
terminal_soc_value_weight = 0.1
price_spread_shaping_weight = 0.0
```

### 8.4 验收标准

训练后：

```text
charge 在低 price_rank 区间出现。
final_soc 不再长期停在 soc_min。
storage_profit_total_eur > idle baseline。
compare 指标仍只使用 unshaped storage_profit。
```

## 9. 阶段 F：调整初始 SOC 训练分布

### 9.1 当前问题

当前默认：

```text
init_soc = 0.05
soc_min  = 0.05
```

这意味着 episode 开始时：

```text
放电不可行。
```

如果 actor 初期探索偏向放电，会立即进入：

```text
请求放电 -> 被裁剪 -> penalty -> 执行 idle
```

### 9.2 推荐配置

训练时支持随机初始 SOC：

```python
train_init_soc_mode: str = "random_uniform"
train_init_soc_low: float = 0.20
train_init_soc_high: float = 0.80
eval_init_soc_mode: str = "fixed"
```

评估仍可固定为 MISOCP 对齐值。

如果 compare 要严格同初始 SOC：

```text
eval_init_soc = 0.05
```

如果目标是训练稳定性优先：

```text
train 随机 SOC，eval 固定 SOC。
```

### 9.3 验收标准

训练诊断应显示：

```text
episode_start_soc 分布覆盖 [0.2, 0.8]
非法放电请求比例下降
充放电动作都出现
```

## 10. 阶段 G：训练流程和记录更新

### 10.1 强制记录训练合同

每次训练结果必须记录：

```text
shared_data_signature
observation_feature_set
observation_normalization_signature
storage_price_mode
import_price_markup_eur_per_kwh
local_action_penalty_mode
local_action_penalty_weight
training_reward_shaping_mode
terminal_soc_value_weight
price_spread_shaping_weight
train_init_soc_mode
```

旧 checkpoint 如果缺字段，必须拒绝复用，并提示重跑：

```text
notebooks/madrl/train_base.ipynb
```

### 10.2 notebook 输出对齐

以下 notebook 都要输出同一套经济列和 meta：

```text
notebooks/madrl/train_base.ipynb
notebooks/madrl/train_base_safe.ipynb
notebooks/madrl/train_projection_safe.ipynb
notebooks/madrl/local_MPC.ipynb
notebooks/madrl/ADMM_mpc.ipynb
notebooks/madrl/global_MISOCP.ipynb
```

必须在输出表中显式展示：

```text
controller
storage_discharge_revenue_total_eur
storage_charge_cost_total_eur
storage_profit_total_eur
battery_net_power_kw_mean_abs
requested_battery_power_kw_mean_abs
executed_battery_power_kw_mean_abs
soc_penalty_total
storage_price_mode
objective_mode
```

## 11. 推荐实施顺序

### Step 1：只加诊断，不改训练行为

目标：

```text
把当前坏行为量化并保存。
```

涉及：

```text
scripts/diagnostics/madrl_price_diagnostics.py
scripts/utils/grid_notebook_workflow.py
notebooks/madrl/train_base.ipynb
```

验收：

```text
能复现当前 price diagnosis 和 action gap diagnosis。
```

### Step 2：拆分 local action penalty

目标：

```text
让 r_soc_pen 不再压倒 storage_profit。
```

涉及：

```text
configs/experiment_config.py
scripts/train.py
controllers/madrl/safety_projector.py
tests/test_train_mainline_launcher.py
tests/test_grid_notebook_workflow.py
```

验收：

```text
r_soc_pen mean_last100 < 10
storage_profit_total_eur 不再恒等于 0
```

### Step 3：统一 rollout、target-Q、actor-loss 的本地可行动作处理

目标：

```text
policy 优化的动作和环境执行的动作一致。
```

涉及：

```text
controllers/madrl/local_action_feasibility.py
scripts/train.py
controllers/madrl/base_agent.py
tests/test_madrl_action_feasibility.py
```

验收：

```text
requested/executed action gap 明显下降。
actor 不再长期请求 soc_min 下的满功率放电。
```

### Step 4：新增相对价格特征

目标：

```text
让 actor 看到低价和高价的相对关系，而不是只看到绝对归一化价格。
```

涉及：

```text
configs/experiment_config.py
envs/observation/default_builder.py
envs/observation/normalization.py
predictors/shared_data.py
notebooks/forecast/forecast_lstm.ipynb
tests/test_observation_normalization.py
tests/test_grid_notebook_workflow.py
```

验收：

```text
shared_data schema 升级。
旧 shared_data 明确失败。
新 observation 包含 relative/rank price features。
```

### Step 5：加入 terminal SoC value shaping

目标：

```text
降低“充电先亏、未来才赚”的学习难度。
```

涉及：

```text
envs/rewards/NormalReward.py
configs/experiment_config.py
envs/grid_env.py
tests/test_normal_reward.py
```

验收：

```text
training reward components 中有 r_terminal_soc_value。
compare metrics 不包含 shaping。
```

### Step 6：做 storage_price_mode 对照实验

目标：

```text
判断当前 import markup 是否符合研究目标。
```

实验：

```text
A. storage_price_mode = real_time_price
B. storage_price_mode = wholesale_price
```

验收：

```text
两组实验的 MADRL、MPC、MISOCP 都使用同一价格口径。
compare 表中明确显示 storage_price_mode。
```

## 12. 测试计划

每阶段至少运行：

```text
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_normal_reward.py -q
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_grid_notebook_workflow.py -q
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_train_mainline_launcher.py -q
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_madrl_notebook_smoke.py -q -k train_base
```

涉及 MPC 价格口径时额外运行：

```text
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_global_socp_mpc.py -q
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_admm_mpc_notebook_helpers.py -q
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_misocp_global_notebook.py -q
```

训练层面的最低验收：

```text
1. train_reward_summary 中 r_soc_pen 不再主导总 reward。
2. rollout 中 e_bat 不再全为 0。
3. storage_profit_total_eur 优于 idle baseline = 0。
4. requested 和 executed action gap 可解释且有下降。
5. compare 输出列和 MISOCP 完全一致。
```

## 13. GitNexus 和风险控制

编辑任何函数、类或方法前，必须按 AGENTS.md 运行 impact analysis。

高风险符号包括：

```text
NormalReward.compute
GridEnv.step
DefaultObservationBuilder._sequence_feature
DefaultObservationBuilder._build
fit_observation_normalization_state
build_observation_normalizer
TrainRunner.select_action_batch_with_info
TrainRunner.build_shared_update_ctx
merge_action_info_into_step_info
enforce_local_action_feasibility_torch
compute_action_gap_metrics_torch
```

如果 GitNexus 对这些符号返回 HIGH 或 CRITICAL，必须先向用户说明 blast radius，再继续编辑。

完成代码改动后必须运行：

```text
gitnexus_detect_changes()
```

并确认变化只覆盖预期 owner：

```text
reward owner
observation owner
MADRL action feasibility owner
shared_data contract owner
notebook workflow output owner
tests
```

## 14. 最小可行修复版本

如果只做一轮最小修复，推荐范围如下：

```text
1. 新增诊断输出。
2. 拆分 local_action_penalty_weight，让 train_base 不再用 r_soc_pen 压倒训练 reward。
3. 统一 rollout、target-Q、actor-loss 的本地可行动作处理。
4. 增加 wholesale_price_rel_seq 和 wholesale_price_rank_seq。
5. 不改 storage_price_mode，仍保持 real_time_price 与 MISOCP 对齐。
6. 强制旧 checkpoint 和旧 shared_data 失败，不静默复用。
```

这个版本的成功标准：

```text
训练不再长期只学到 soc_min 下的不可行放电。
评估 rollout 中电池有实际充放电。
storage_profit_total_eur 不再恒为 0。
所有输出仍能和 MISOCP 的 storage_profit 统计口径对齐。
```
