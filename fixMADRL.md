# MADRL 跨时储能套利修复计划

## 0. 目标

让 MADRL 学到真实的跨时段储能套利，而不是利用每天重置 SoC 的漏洞做虚空套利。

本轮计划只解决 MADRL 链路的两个核心问题：

```text
1. MADRL rollout 必须和 MPC / MISOCP 一样使用 continuous SoC。
2. 删除 terminal SoC shaping，重写 MADRL 内部训练 reward。
```

最终输出必须同时保留两个口径：

```text
口径 A：MISOCP 对齐口径
用于七个方案横向比较。只用实际执行动作和实际价格计算，不包含 MADRL 内部 reward shaping。

口径 B：MADRL 内部 reward 分解口径
用于诊断智能体为什么充电、放电、等待或撞边界。包含 r_inc、r_pen、r_soc、r_bonus、safe penalties 等内部项。
```

## 0.1 硬约束

本计划的每一步都必须遵守：

```text
1. 不保留 dual-path：任何字段都不再有“新行为 / 旧行为”两条活路。
2. 不保留 deprecated alias：同一份数据不再以两个不同字段名导出。
3. 不引入 mode 枚举充当 on/off 开关：能用 weight=0 关闭的，就不做 mode。
   删除旧项时必须从 `component_meta`、summary schema 和 parquet schema 一起删除，禁止保留列并输出 0。
4. 不保留兜底：遇到旧 checkpoint / 旧 shared_data / 旧 rollout record / 旧 config，第一道边界直接 raise。
   异常信息必须说明旧对象、期望的新合同、要重跑的 notebook。
5. 不新增 owner 除非有明确价值：稳定 API、类型边界、命名对齐、协议隔离、CLI 或 notebook 入口。
6. 比较口径和训练 reward 必须分离：compare 不得读取 MADRL 内部 shaping 项作为主排名指标。
7. reward 字段语义必须显式：raw action、projected action、executed battery power 不能在不同 MADRL 分支里悄悄换含义。
```

## 1. 当前诊断结论

### 1.1 虚空套利来自 daily SoC reset

当前 `GridEnv.reset(...)` 每次都会重置 SoC：

```text
envs/grid_env.py
GridEnv.reset -> self.soc = self._initial_soc_for_reset()
```

训练时，vec env 在每个 daily episode 结束后继续 reset；评估时，MADRL rollout 默认也没有传入 `soc_mode='continuous'`。

因此当前策略面对的任务不是“跨天管理一块电池”，而是：

```text
每天开局免费获得一块带电电池
当天尽量放电卖钱
明天 SoC 自动恢复
```

这会自然诱导“一味放电”的策略。继续加训练代数不能解决这个问题，只会让策略更熟练地利用漏洞。

### 1.2 MPC / MISOCP 和 MADRL 的 SoC 口径不一致

Local MPC、ADMM MPC、Global MISOCP 的 rollout 已经按 continuous SoC 处理；MADRL rollout 仍然走 reset SoC。

这导致 MADRL 在横向比较中拿到了 MPC 没有的免费能量，结果不可信。

### 1.3 terminal SoC shaping 不是正确修复

当前 terminal SoC shaping 试图给 SoC 库存一个未来价值，但它有三个问题：

```text
1. 它没有修复 daily reset 漏洞。
2. 它和真实横向比较口径混在一起，容易让 reward 好看但经济指标不可比。
3. 它依赖 horizon mean price，不能稳定表达“什么时候应该为未来高价保留电量”。
```

本轮必须删除 terminal SoC shaping，而不是继续调权重。

## 2. 新目标状态

### 2.1 横向比较口径：MISOCP 对齐

所有方案最终 compare 只使用实际执行轨迹计算：

```text
storage_charge_cost_total_eur
storage_discharge_revenue_total_eur
storage_profit_total_eur
storage_objective_total_eur
```

本轮统一选择这一组作为 aggregate canonical columns。

以下历史别名必须删除，不能继续双写或兼容读取：

```text
storage_purchase_cost_eur
storage_sale_revenue_eur
storage_total_profit_eur
```

如果旧 summary / parquet / notebook output 只包含历史别名，读取边界必须失败并提示重跑产生该 record 的 notebook。

定义保持：

```text
battery_power_kw > 0  => charge
battery_power_kw < 0  => discharge

charge_energy_kwh_t    = max(battery_power_kw_t, 0) * dt
discharge_energy_kwh_t = max(-battery_power_kw_t, 0) * dt

storage_charge_cost_t       = price_t * charge_energy_kwh_t
storage_discharge_revenue_t = price_t * discharge_energy_kwh_t
storage_profit_t            = storage_discharge_revenue_t - storage_charge_cost_t
storage_objective_t         = -storage_profit_t
```

这个口径用于：

```text
global_misocp
local_mpc_perfect
local_mpc_lstm
admm_mpc_lstm
madrl_base
madrl_base_safe
madrl_projection_safe
```

compare 主表数值来源的硬契约：

```text
所有方案的 storage_charge_cost_total_eur / storage_discharge_revenue_total_eur /
storage_profit_total_eur / storage_objective_total_eur 必须从 step_df 的
battery_power_kw 与实际 price 逐步重算、再聚合，七个方案使用同一个重算函数。

Global MISOCP solver 自己的 objective 值可以作为诊断字段
`returned_primary_objective_eur` 保留，但不得作为 compare 的
storage_profit_total_eur。
```

这样避免 solver 内部含松弛项 / tie-breaking 微权重 / simultaneous charge-discharge 处理差异时，
compare 主表数值与 step-wise 累加不恒等，造成排名不稳定。

注意：

```text
MADRL 内部 reward 的 r_inc、r_pen、r_soc、r_bonus 不进入 compare 主排名。
SAFE penalty 也不进入 compare 主排名，只作为安全诊断。
```

### 2.2 MADRL 内部 reward 口径：训练专用

MADRL 环境每一步反馈给智能体的内部奖励改为：

```text
r_t = r_t_inc - r_t_pen - r_t_soc + r_t_bonus - r_t_safe
```

其中 `r_t_safe` 只在 safety penalty / projection 方案中非零。

内部 reward 的目标不是替代 MISOCP compare，而是提供更强、更稳定的学习信号。

## 3. Phase A：MADRL rollout 改为 continuous SoC

### 3.1 修改方向

`collect_madrl_rollout(...)` 必须调用：

```text
collect_controller_rollout(..., soc_mode="continuous")
```

并且必须沿用当前 continuous SoC 合同：

```text
1. episode_indices 必须按时间升序，且必须来自同一 contiguous date range。
2. 禁止跨 gap 做 SoC carry。如果相邻 episode 的 start 时间戳不等于前一 episode 的 end 时间戳，
   continuous SoC 路径必须在 gap 处 raise，而不是静默 carry，否则 SoC 会从 Day N 直接跳到 Day N+k 的开局，
   成为物理不可行轨迹。
3. 第 N+1 个 episode 的初始 SoC 必须等于第 N 个 episode 的结束 SoC。
4. rollout.meta["soc_mode"] 必须等于 "continuous"。
5. rollout.meta 必须额外记录 episode_indices 的 start/end 时间戳，便于 compare 层抽查 gap。
```

### 3.2 禁止行为

```text
1. 禁止 MADRL rollout 默认为 reset SoC。
2. 禁止 compare notebook 接收 soc_mode="reset" 的 MADRL record。
3. 禁止用旧 MADRL rollout record 参与新 compare。
```

### 3.3 验收

新增或更新测试：

```text
1. collect_madrl_rollout passes soc_mode="continuous" to collect_controller_rollout.
2. madrl_base / madrl_base_safe / madrl_projection_safe 的 record meta 中 soc_mode == "continuous"。
3. compare 读取 MADRL record 时，如果 soc_mode != "continuous"，直接失败并提示重跑对应 notebook。
```

必须覆盖 compare 的失败路径和成功路径：

```text
tests/test_compare_notebook.py::test_compare_rejects_madrl_reset_record
tests/test_compare_notebook.py::test_compare_accepts_madrl_continuous_record
```

## 4. Phase B：训练也要消除 daily reset 漏洞

只改 rollout 不够。评估会真实，但策略仍然是在错误 MDP 上训练出来的。

### 4.1 训练目标状态

MADRL 训练不再把每一天当成一块免费补电的独立电池。训练 episode 必须覆盖连续多天，或在日边界 carry SoC 且不把日边界当成终止状态。

当前执行目标改为：

```text
MADRL train episode = contiguous multi-day window
train_window_days = 3
window_stride_days = 1
episode_limit = 96 * train_window_days = 288
```

这样 reset 只发生在多日窗口开始，而不是每天发生；同时把第一版训练时长控制在可接受范围内，优先保证 `train_base.ipynb` 端到端跑通。

### 4.1.1 window 边界价值合同

`train_window_days = 3` 只把采样窗口改长，不自动解决“窗口最后几步把库存放光”的问题。必须显式规定 window 边界的价值语义：

```text
如果当前 step 的物理下一时刻仍然存在，且仍属于同一个 contiguous train split，
则 window 末端只能记为 truncated，不能记为 terminal。

只有以下两种情况允许记为 terminal：
    1. 到达 train split 的真实物理末端；
    2. 遇到 train split 内部 gap（本计划要求直接失败，见 §4.3.1）。
```

当前计划选择：

```text
window boundary mode = truncated_bootstrap_v1
```

合同细节：

```text
1. replay / batch 中必须同时保留 terminated 与 truncated。
2. critic / TD target 只能在 terminated=True 时把 continuation value 归零。
3. truncated=True 时必须继续 bootstrap V(s_{t+1}) / Q(s_{t+1}, a_{t+1})。
4. 这个 s_{t+1} 必须来自物理时间轴上的真实下一时刻，而不是 window 内 padding。
5. 如果当前实现做不到 terminated / truncated 分离，则 Step 1 视为未完成，禁止进入重训。
```

### 4.2 为什么不用 daily reset + terminal shaping

daily reset + terminal shaping 仍然会让策略看到“每天结束后库存价值消失”的不一致环境。多日窗口能让策略直接经历低价充电、高价放电、跨天保留电量这些真实行为。

### 4.3 shared_data 合同

如果训练 episode 从 daily 改为 multi-day，则 shared_data 必须硬切换；但不再保留“rolling window 全量物化”方案。

```text
shared_data schema version bump
shared_data owner 改为 timeline-cache
train/test split 都只缓存物理时间轴上的每个 timestamp 一次
window / episode 只记录 slice 或 index，不重复存储重叠 forecast 数组
test rollout 仍按时间排序 carry SoC
```

旧 shared_data 不得被扫描或兼容读取。

### 4.3.1 影响的合同清单

这一阶段不能只改 `episode_limit`。必须逐一处理以下合同：

```text
predictors/shared_data.py
    必须 bump shared_data SCHEMA_VERSION。
    train split 不再把 rolling windows 物化成完整 episode 数组。
    改为：
        1. 先按物理时间轴缓存 forecast / calendar / price-rank 等基础数组；
        2. 再用 window manifest 记录
           (history_start_idx, active_start_idx, active_end_idx, next_active_idx)。
    当前目标 train_window_days = 3，window_stride_days = 1。
    末尾不足 3 天的 tail window 直接丢弃，并在 metadata 记录 dropped_tail_steps。
    禁止 padding、wrap-around、扫描相邻 split 补齐。
    禁止同时保留“window 全量物化”和“timeline-cache”双路径。
    如果 train split 内部存在 gap，必须在 shared-data 边界显式失败并提示重选 date range，
    不做跨 gap 拼接，也不做“每段各自训练窗口”的静默退化。

envs/subproc_vec_env.py
    每个 worker 只负责自己采样到的完整 multi-day window。
    SoC carry 只发生在同一个 worker 的同一个 window 内部。
    worker 之间不做 SoC 同步；跨 worker 同步会把随机训练采样和物理时间顺序混在一起。
    evaluation / compare 的 continuous SoC 必须走按时间排序的 rollout coordinator，不依赖 SubprocVecEnv 随机 worker 顺序。

configs/profiles.py
    如果存在 profile / training contract schema version，必须 bump。
    如果没有独立 SCHEMA_VERSION，也必须更新 summarize_experiment 输出合同和测试。
    summary 中删除 terminal_soc_value_weight，新增 train_window_days、window_stride_days、reward_contract、rollout_soc_contract。

normalization contract
    load / pv / price normalizer 仍只用 train split 拟合。
    normalizer 必须在原始 train date range 的时间序列上 fit，每个物理时刻只出现一次。
    即使 train windows 使用 stride=1，也禁止在 rolling windows 的 concatenated 样本上 fit。
    如果 train split 内部存在 gap，normalization 直接失败，不做跨 gap 拼接。
    任何带 daily-segment 假设的 normalization signature 都必须失效并重算。
```

训练集和测试集的分割策略：

```text
train:
    使用 train date range 内的 3-day rolling windows。
    stride = 1 day。
    incomplete tail dropped with explicit metadata。
    这些 window 只在 manifest 中表达，不重复存 forecast tensor。

test:
    测试窗口仍可按 daily episodes 选择，便于 notebook 输出和对齐。
    但底层 shared-data 仍走 timeline-cache。
    MADRL rollout 必须按时间升序串起来，并在 day boundary carry SoC。
```

### 4.4 训练合同

checkpoint / result metadata 必须记录：

```text
reward_contract = "madrl_incremental_storage_reward_v1"
rollout_soc_contract = "continuous_soc_v1"
train_window_days = 3
window_stride_days = 1
episode_limit = 288
shared_data_schema_version = new version
normalization_signature = new signature
global_interaction_step = 累计环境交互步数 (用于 §6.5 resume 退火)
target_total_steps      = 训练目标总步数
window_boundary_mode    = "truncated_bootstrap_v1"
```

旧 checkpoint 如果含有 `terminal_soc_value_weight`，或缺少上述新合同字段，必须在加载边界失败，并提示重跑：

```text
notebooks/madrl/train_base.ipynb
notebooks/madrl/train_base_safe.ipynb
notebooks/madrl/train_projection_safe.ipynb
```

resume 约束：

```text
resume 时 target_total_steps 只能保持不变或缩短。
禁止在同一训练 run 上把 target_total_steps 向后延长。

原因：
    §6.5 throughput bonus 的退火进度依赖
    global_interaction_step / target_total_steps。
    延长 target_total_steps 会把已接近 0 的 bonus 人为拉回非零区间，破坏退火合同。

如果确实要延长：
    视为新训练，而不是 resume。
    必须 bump shared_data SCHEMA_VERSION / shared_data signature，
    生成新的 training_contract_signature，
    并以新 checkpoint run 重新开始。
```

## 5. Phase C：删除 terminal SoC shaping

### 5.1 删除范围

删除以下概念：

```text
terminal_soc_value_weight
r_terminal_soc_value
horizon_mean_storage_price_t
horizon_mean_storage_price_next
terminal SoC potential shaping
```

涉及位置：

```text
configs/experiment_config.py
envs/rewards/NormalReward.py
envs/grid_env.py reward_state
scripts/checkpoints.py training_contract
scripts/mainline_madrl.py reward_controls
scripts/utils/storage_profit.py
scripts/utils/grid_notebook_workflow.py reward_controls
configs/profiles.py summarize_experiment
tests
notebooks/madrl/*.ipynb
notebooks/record/madrl/*/agent.parquet schema
```

执行前必须先做全仓搜索并逐项清除：

```text
rg -n "terminal_soc_value|r_terminal_soc_value|horizon_mean_storage_price" .
```

清除标准：

```text
1. `NormalReward.component_meta` 不再包含 r_terminal_soc_value。
2. 训练 summary 不再包含 terminal_soc_value_weight。
3. reward_state 不再生产 horizon_mean_storage_price_t / horizon_mean_storage_price_next。
4. tests/test_normal_reward.py 删除旧 terminal shaping 断言，改为断言旧字段不存在。
5. tests/test_train_mainline_launcher.py、tests/test_grid_notebook_workflow.py 等所有 contract 测试同步删除旧字段。
6. notebooks/record/madrl/*/agent.parquet 如果包含旧列，读取边界直接失败，不能 drop column 后继续用。
```

### 5.2 禁止兼容

旧对象如果包含 `terminal_soc_value_weight`，不能静默忽略。第一道边界要报错：

```text
Old MADRL reward contract includes terminal_soc_value_weight.
New contract expects madrl_incremental_storage_reward_v1 without terminal SoC shaping.
Re-run notebooks/madrl/train_base.ipynb.
```

## 6. Phase D：新 MADRL reward 设计

### 6.1 总公式

每个 agent 的内部 reward：

```text
r_i,t =
    r_i,t_inc
  - r_i,t_pen
  - r_i,t_soc
  + r_i,t_bonus
  - r_i,t_safe
```

其中：

```text
r_safe = r_safe_v + r_safe_line + r_safe_trafo
```

`madrl_base` 的 safety weights 为 0，所以 `r_safe = 0`。

#### 6.1.1 executed_battery_power_kw 硬契约

本章所有 reward 分项（`r_inc`、`r_pen`、`r_bonus`）必须共用同一个 executed battery power：

```text
executed_battery_power_kw_i,t :=
    env 内部做完 SoC / P_max / 充放电互斥等全部 feasibility clamp 后、
    真正用于 SoC 更新（soc_{t+1} = soc_t + e_bat * dt / C_bat）的那笔 e_bat。

符号约定（与 §2.1 一致）：
    executed_battery_power_kw > 0  => 充电
    executed_battery_power_kw < 0  => 放电
```

硬约束：

```text
1. r_inc_i,t、r_pen_i,t、r_bonus_i,t 必须都使用同一个 executed_battery_power_kw_i,t。
2. 禁止某一项用 raw actor output、另一项用 pre-env projected action、第三项用 post-env executed。
   这种混用会让同一个 step 的 reward 分解自相矛盾（例如 env 把 -5 kW 夹到 -2 kW，
   r_inc 按 -2 记收益，r_pen 按 -5 记惩罚）。
3. raw actor output 与 pre-env projected action 只允许出现在诊断字段
   madrl_raw_action_boundary_violation / madrl_action_projection_gap 中，
   禁止进入任一 reward 分项。
4. 三条 MADRL 分支（madrl_base / madrl_base_safe / madrl_projection_safe）的 r_pen / r_inc / r_bonus
   使用的 executed_battery_power_kw 必须是同一个语义：env 落地那一笔。
   分支差异只体现在"从 actor output 到 executed_battery_power_kw 中间走了哪些 projection 环节"
   的诊断字段上，不体现在 reward 数值来源上。
```

`r_safe` 的 per-agent 分摊合同沿用当前安全 reward 逻辑，不在本轮重新定义：

```text
r_safe_v     按现有 voltage_violation 分摊逻辑给各 agent，不改成均分。
r_safe_line  按现有 line violation 逻辑。
r_safe_trafo 按现有 trafo violation 逻辑。
```

如果安全分摊公式要改，必须作为独立 owner 变更处理，不能混在本轮储能套利 reward 里。

### 6.2 核心经济驱动项：r_inc

目的：

```text
直接衡量“电池这一步让系统多赚/少花了多少钱”，避免负载和 PV 的大额背景成本淹没电池信号。
```

第一版 `madrl_r_inc` 采用和 MISOCP compare 一致的对称 storage price 口径，不把 import markup / tax 的进口侧非对称性写入训练 reward：

```text
storage_price_t = compare_storage_price_t

# e_bat_i,t := executed_battery_power_kw_i,t，参见 §6.1.1。
charge_kwh_i,t    = max(e_bat_i,t, 0)  * dt
discharge_kwh_i,t = max(-e_bat_i,t, 0) * dt

r_inc_i,t =
    storage_price_t * discharge_kwh_i,t
  - storage_price_t * charge_kwh_i,t
```

其中 `e_bat_i,t` 严格是 §6.1.1 定义的 `executed_battery_power_kw_i,t`，不是 actor raw output，
也不是 pre-env projected action。

`import_price_markup_eur_per_kwh` 和 `export_subsidy_eur_per_kwh` 可以进入 observation 或系统成本诊断，但本轮不进入 `madrl_r_inc`。原因：

```text
如果 r_inc 只在 net > 0 时加 tax，而 net < 0 出口侧不加 tax，
策略会被诱导过量放电到出口状态，以赚取 import markup 的非对称差。
这会和 MISOCP storage_profit = price * (discharge - charge) 的横向比较口径打架。
```

如果未来明确要改成“系统 net import/export cost delta”作为训练信号，必须另起新 reward contract，且成本函数必须显式包含 export subsidy：

```text
cost(net, lambda, tax, subsidy) =
    dt * max(net, 0)  * (lambda + tax)
  - dt * max(-net, 0) * subsidy
```

禁止使用“进口加 tax、出口按 lambda 卖电”的半边公式。

重要合同：

```text
madrl_r_inc 是 per-agent 训练信号和诊断信号。
它采用 bus-level separability assumption，只把本 agent 的执行电池功率归因给本 agent。
它不是潮流耦合后的全局系统成本改变量证明。
```

因此：

```text
1. 不能写测试要求 sum(madrl_r_inc) == feeder_net_cost_reduction。
2. 不能写测试要求 sum(madrl_r_inc) == storage_profit_total_eur。
3. 不能把 sum(madrl_r_inc) 当作全局优化目标。
4. compare 必须独立从实际执行轨迹重算 storage_profit_total_eur。
5. 如果因为相同 storage_price 和 e_bat 出现 accounting-level 接近，也只能视为会计近似，不是 power-flow economics 恒等。
```

注意：

```text
r_inc 是训练信号，不是 compare 主指标。
compare 仍然用 MISOCP 对齐的 storage_profit_total_eur。
```

必须额外输出一个显式诊断项：

```text
madrl_r_inc_vs_storage_profit_gap_eur =
    sum(madrl_r_inc) - storage_profit_total_eur
```

用途：

```text
它不是正确性测试，也不是收敛目标。
它只是帮助诊断 bus-level separability assumption 在潮流耦合较强时偏离多大。
```

### 6.3 动作边界惩罚：r_pen

目的：

```text
惩罚 env 落地后仍在 SoC 边界附近执行的充放电动作，避免策略把边界当成常态操作点。
```

本轮选择一个统一语义：

```text
r_pen 使用 §6.1.1 定义的 executed_battery_power_kw_i,t。
它不使用 actor raw output，也不使用 pre-env projected action。
```

三条 MADRL 分支的 r_pen 数值来源完全一致，均为 env 落地后那一笔 executed battery power。
分支差异只体现在"从 actor output 到 executed_battery_power_kw 之间走了哪些 projection 环节"
的诊断字段上：

```text
madrl_base:
    诊断链：actor_raw -> local SoC feasibility projection -> env clamp -> executed
madrl_base_safe:
    诊断链：actor_raw -> local SoC feasibility projection + safety logic -> env clamp -> executed
madrl_projection_safe:
    诊断链：actor_raw -> safety_projector.py -> env clamp -> executed
```

raw action / pre-env projected action 只能作为诊断字段记录：

```text
madrl_raw_action_boundary_violation
madrl_action_projection_gap
```

它们禁止进入 `madrl_r_action_penalty`。如果未来要惩罚“意图”，必须新建显式字段
`madrl_r_intent_penalty`，不能复用 `madrl_r_action_penalty`。

惩罚（使用 executed battery power 的符号约定，参见 §6.1.1）：

```text
r_pen_i,t =
    w_pen * |e_bat_i,t|, if e_bat_i,t < 0 and soc_i,t <= soc_min + eps   (边界放电)
    w_pen * |e_bat_i,t|, if e_bat_i,t > 0 and soc_i,t >= soc_max - eps   (边界充电)
    0, otherwise
```

`eps` 第一版使用一个小常数，例如：

```text
soc_boundary_epsilon = 0.02
```

不新增 mode，只新增权重：

```text
action_boundary_penalty_weight
```

### 6.4 SoC 灵活性正则：r_soc

目的：

```text
抑制长期贴边，但不把电池强行拉回 0.5。
```

禁止使用第一版原始中点 MSE：

```text
r_soc = w_soc * (soc - 0.5)^2
```

原因：

```text
低价充电、高价放电要求 SoC 在日 / 周内扫过大范围。
中点 MSE 会持续惩罚有效套利路径，容易把 policy 拉成惰性保持 0.5。
```

第一版改为 flat-bottom boundary regularization：

```text
soc_lower_soft = soc_min + soc_boundary_margin
soc_upper_soft = soc_max - soc_boundary_margin

below = max(0, soc_lower_soft - soc)
above = max(0, soc - soc_upper_soft)

r_soc_i,t = w_soc * (below^2 + above^2)
```

推荐：

```text
soc_boundary_margin = 0.10
```

必须显式切开 `r_pen` 与 `r_soc` 的作用区：

```text
r_pen:
    只管最边缘 epsilon 带内、且 executed action 继续向边界外推的动作。
    它回答的是“这一步是不是在撞边界还继续推”。

r_soc:
    管更宽的 soft margin 带，不看动作方向。
    它回答的是“电池是不是长期贴边、失去灵活性”。

硬约束：
    soc_boundary_margin > soc_boundary_epsilon
```

因此第一版默认不允许：

```text
soc_boundary_margin == soc_boundary_epsilon
```

否则 `r_soc` 与 `r_pen` 会在同一条窄带内重叠，`r_pen` 按权重完全压过 `r_soc`，`r_soc` 基本失去独立信号。

新增权重：

```text
soc_boundary_regularization_weight
```

### 6.5 吞吐量激励：r_bonus

目的：

```text
防止策略因为害怕边界惩罚和 SoC 正则而学成永远不动。
```

定义（`e_bat_i,t` 严格为 §6.1.1 定义的 `executed_battery_power_kw_i,t`）：

```text
throughput_kwh_i,t = |e_bat_i,t| * dt
r_bonus_i,t = throughput_bonus_eur_per_kwh(step) * throughput_kwh_i,t
```

注意：

```text
bonus 只用于内部训练。
compare 不得把 bonus 算进收益。
```

throughput bonus 必须带退火合同，不能长期污染最终 policy：

```text
0%   <= training_progress < 20%:
    throughput_bonus_eur_per_kwh = 0.002

20%  <= training_progress < 80%:
    throughput_bonus_eur_per_kwh linearly decays from 0.002 to 0.0

80%  <= training_progress <= 100%:
    throughput_bonus_eur_per_kwh = 0.0
```

`training_progress` 以 global environment interaction steps 计算，不以 episode 数计算。

resume-from-checkpoint 的退火合同：

```text
training_progress = (restored_global_interaction_step + incremental_steps) / target_total_steps

1. checkpoint 必须持久化 global_interaction_step 字段。
2. resume 训练时，annealing schedule 必须从 restored step 继续，禁止重置为 0。
3. 如果 checkpoint 缺少 global_interaction_step 字段，加载边界直接失败，
   并提示重跑产生该 checkpoint 的 notebook。
```

否则 resume 会把已退火到 0 的 bonus 重新注入高值区间，破坏 annealing 合同。

训练 summary 必须记录：

```text
madrl_throughput_bonus_weight_mean
madrl_throughput_bonus_weight_final
madrl_r_throughput_bonus
throughput_kwh_total
```

evaluation rollout 中如果只重算最终 contract reward，则 throughput bonus 权重应为 0；训练历史里的 reward summary 才是观察 bonus 如何起作用的主要入口。

### 6.6 推荐初始权重

第一版建议保守设置：

```text
action_boundary_penalty_weight = 0.05
soc_boundary_regularization_weight = 0.005
throughput_bonus_eur_per_kwh_max   = 0.002
```

这些不是最终调参结论，只是让新 reward 能跑起来并暴露诊断信号。

验收时不能只看总 reward，要看 reward 分解是否合理。

## 7. Phase E：两个输出口径的字段设计

### 7.1 口径 A：MISOCP 对齐字段

这些字段写入 step / agent / summary / metrics，用于横向比较：

```text
aggregate summary fields:
storage_charge_cost_total_eur
storage_discharge_revenue_total_eur
storage_profit_total_eur
storage_objective_total_eur

per-row step / agent fields:
storage_charge_cost_eur
storage_discharge_revenue_eur
storage_profit_eur
storage_objective_eur
```

数值来源（与 §2.1 一致）：

```text
aggregate 字段必须由七个方案共用的同一个重算函数从 step_df 聚合产出。
禁止任何方案直接把 solver 内部 objective 或 episode 累积中间变量塞进 aggregate。
Global MISOCP solver 的 objective 只能以 returned_primary_objective_eur 作为诊断。
```

owner 显式指定为：

```text
scripts/compare/storage_profit_recompute.py
```

合同：

```text
1. compare 主表、MADRL notebook summary、MPC notebook summary 都只能调用这个 owner。
2. 禁止在 notebook cell 内各自手写 aggregate 公式。
3. 如果未来要迁移 owner，只允许整块迁移，禁止复制出第二份实现。
```

禁止继续输出或消费以下历史别名：

```text
storage_purchase_cost_eur
storage_sale_revenue_eur
storage_total_profit_eur
```

compare 主表只按：

```text
storage_profit_total_eur
```

降序排序。

### 7.2 口径 B：MADRL 内部 reward 字段

这些字段只用于 MADRL 诊断：

```text
madrl_r_inc
madrl_r_action_penalty
madrl_r_soc_regularization
madrl_r_throughput_bonus
madrl_r_safe_v
madrl_r_safe_line
madrl_r_safe_trafo
madrl_r_safe_total
madrl_r_total_internal
madrl_action_projection_gap
madrl_raw_action_boundary_violation
madrl_throughput_bonus_weight_mean
madrl_throughput_bonus_weight_final
madrl_r_inc_vs_storage_profit_gap_eur
```

推荐落盘位置：

```text
train_reward_summary.json
rollout step_df / agent_df diagnostics columns
optional: notebooks/record/madrl/<scheme>/reward_components.parquet
```

字段命名必须带 `madrl_` 前缀，避免和 MISOCP compare 字段混淆。

### 7.3 compare 禁止读取内部 reward

`compare.ipynb` 和 `build_compare_economic_table(...)` 不允许使用：

```text
madrl_r_total_internal
madrl_r_inc
madrl_r_action_penalty
madrl_r_soc_regularization
madrl_r_throughput_bonus
madrl_r_safe_total
```

这些只能在 MADRL notebook 或 diagnostics 图中显示。

## 8. Phase F：可视化和诊断

MADRL notebook 需要新增内部 reward 分解图：

```text
1. cumulative madrl_r_inc
2. cumulative madrl_r_action_penalty
3. cumulative madrl_r_soc_regularization
4. cumulative madrl_r_throughput_bonus
5. cumulative madrl_r_total_internal
```

还需要新增行为诊断：

```text
1. SoC boundary hit rate
2. discharge while low SoC rate
3. charge while high SoC rate
4. charge price rank distribution
5. discharge price rank distribution
6. daily boundary SoC continuity check
```

验收目标：

```text
低价分位更常充电
高价分位更常放电
跨天 SoC 连续
策略不再每天从固定 SoC 开始清空电池
```

## 9. Phase G：实施顺序

以下只保留剩余工作。任何一步没有通过验收，都禁止进入后续 notebook 重跑或重训；否则会产生旧合同产物并污染 compare。

### 9.1 当前完成度快照

```text
已完成：
1. MADRL rollout 已显式请求 continuous SoC。
2. terminal SoC shaping 删除与旧对象拒绝。
3. 新 reward 字段、summary、diagnostics 基本合同。
4. 基础 multi-day train plumbing、tests 与 notebook launcher 控制面。
5. replay / critic 已分离 terminated 与 truncated；truncated window 末端保留 bootstrap continuation value。
6. continuous SoC rollout 已按 timestamp / bootstrap timestamp 校验 gap，并在 meta 记录 selected_episode_start_timestamps / selected_episode_end_timestamps。
7. shared-data owner 已改为 schema 6 timeline-cache，episode manifest 显式记录 active_start_idx / active_end_idx / next_active_idx。
8. compare storage profit owner 已落位，aggregate storage profit 只从 step_df.battery_power_kw 与实际 price 重算，旧经济别名在保存/读取边界直接拒绝。

仍未完成：
1. canonical forecast shared-data 需要用 notebooks/forecast/forecast_lstm.ipynb 重新生成 schema 6 timeline-cache 产物。
2. train_base / train_base_safe / train_projection_safe 需要按新 shared-data 与 compare 合同重训。
3. 最终 compare 需要只读取新 record；任何旧 schema / 旧字段 record 必须在读取边界失败。
```

### Step 0：先关闭合同漏洞，禁止直接重训

Step 0 必须先完成，不能和 notebook 重跑交错。

#### Step 0.1：训练 bootstrap 合同闭合

文件范围：

```text
envs/grid_env.py
envs/subproc_vec_env.py
scripts/train.py
scripts/utils/replay_buffer.py
controllers/madrl/base_agent.py
tests
```

验收：

```text
replay buffer 同时保存 terminated 与 truncated
critic / TD target 只在 terminated=True 时清零 continuation value
truncated=True 时继续 bootstrap V(s_next) / Q(s_next, a_next)
训练 window 到达采样边界但物理下一时刻存在时，只能标记 truncated=True、terminated=False
只有 train split 真实末端允许 terminated=True
测试覆盖 MATD3 / MADDPG / projection-safe target 的 bootstrap mask
```

#### Step 0.2：continuous SoC rollout gap 合同闭合

文件范围：

```text
scripts/utils/grid_notebook_workflow.py
scripts/mainline_compare.py
tests
```

验收：

```text
continuous SoC rollout 读取每个 episode 的 first_timestamp / last_timestamp
相邻 episode 的下一物理时间戳必须连续，否则直接 raise
rollout.meta 记录 selected_episode_start_timestamps / selected_episode_end_timestamps
compare 读取 MADRL record 时校验 soc_mode == continuous 且 timestamp meta 存在
旧 record 缺 timestamp continuity meta 时直接失败并提示重跑对应 notebook
```

#### Step 0.3：compare 经济口径 owner 先落位

文件范围：

```text
scripts/compare/storage_profit_recompute.py
scripts/mainline_compare.py
scripts/utils/grid_notebook_workflow.py
scripts/utils/admm_mpc_notebook_helpers.py
controllers/mpc/global_socp_mpc.py
tests
```

验收：

```text
所有 aggregate storage profit 都由 scripts/compare/storage_profit_recompute.py 从 step_df 重算
compare 不再消费 storage_purchase_cost_eur / storage_sale_revenue_eur / storage_total_profit_eur
rollout step / agent / summary / metrics 不再输出上述历史别名
旧 record 如果只有历史别名，读取边界直接失败并提示重跑源 notebook
Global MISOCP solver objective 只作为 returned_primary_objective_eur 诊断字段，不进入 storage_profit_total_eur
```

### Step 1：shared-data owner 改为 timeline-cache

文件范围：

```text
predictors/shared_data.py
data/loaders/prosumer.py
data/loaders/registry.py
scripts/builder.py
envs/observation/normalization.py
configs/experiment_config.py
envs/grid_env.py
scripts/train.py
scripts/utils/replay_buffer.py
tests
```

验收：

```text
train episode_limit = 96 * train_window_days
当前 train_window_days = 3，window_stride_days = 1
shared-data 不再把 rolling windows 物化成完整 forecast episode 张量
每个物理 timestamp 在 shared-data 中只缓存一次
window manifest 记录 history_start_idx / active_start_idx / active_end_idx / next_active_idx
normalization signature 升级
旧 daily train shared_data 和旧 window-materialized shared_data 都显式失败
window 末端在有真实下一时刻时记为 truncated，并继续 bootstrap
train split 内部若有 gap，则 shared-data / normalization 在边界显式失败
forecast shared-data 生成时间和磁盘占用相对当前方案显著下降
```

### Step 2：重跑 forecast notebook，产出新的 canonical shared-data

重跑：

```text
notebooks/forecast/forecast_lstm.ipynb
```

验收：

```text
notebooks/record/forecast/lstm/shared_data_record.json 更新到新 schema / signature
shared_data_record 指向 timeline-cache 产物
train split 与 test split 都通过新 shared-data 合同校验
共享数据中每个物理 timestamp 只出现一次；window manifest 只保存索引
```

### Step 3：先只跑通 train_base.ipynb

执行：

```text
notebooks/madrl/train_base.ipynb
```

验收：

```text
train_base 使用 train_window_days = 3 跑通
训练 result / checkpoint / rollout 全部满足新 shared-data 合同
rollout continuous SoC 正常
record 同时包含：
    1. MISOCP compare 口径字段
    2. MADRL 内部 reward 字段
训练 replay / critic 已使用 terminated-only bootstrap mask
优先确认端到端可运行，再决定是否继续拉长训练时长
```

### Step 4：扩展到另外两条 MADRL

执行：

```text
notebooks/madrl/train_base_safe.ipynb
notebooks/madrl/train_projection_safe.ipynb
```

验收：

```text
三条训练 result 都有新 reward_contract / rollout_soc_contract / shared_data_schema_version
三条 rollout 都是 continuous SoC
三条 record 都使用新的 timeline-cache shared-data
```

### Step 5：重新 compare

重跑：

```text
notebooks/madrl/compare.ipynb
```

验收：

```text
七个方案全部使用 storage_profit_total_eur 横向比较
compare 不再消费 storage_purchase_cost_eur / storage_sale_revenue_eur / storage_total_profit_eur 历史别名
MADRL 内部 reward 只作为诊断输出，不进入排名
所有 aggregate storage profit 都来自 scripts/compare/storage_profit_recompute.py
```

## 10. 最终验收指标

### 10.1 必须通过的硬指标

```text
1. MADRL rollout soc_mode == continuous。
2. 第 N+1 天初始 SoC == 第 N 天最终 SoC。
3. compare 不接收旧 reset MADRL record。
4. terminal_soc_value_weight 完全删除。
5. r_terminal_soc_value 完全删除。
6. MADRL reward summary 输出全部 9 项内部 reward 本体字段：
   madrl_r_inc
   madrl_r_action_penalty
   madrl_r_soc_regularization
   madrl_r_throughput_bonus
   madrl_r_safe_v
   madrl_r_safe_line
   madrl_r_safe_trafo
   madrl_r_safe_total
   madrl_r_total_internal
   （以上为 reward 本体字段。§7.2 定义的诊断字段
   madrl_action_projection_gap / madrl_raw_action_boundary_violation /
   madrl_throughput_bonus_weight_mean / madrl_throughput_bonus_weight_final
   由 Phase F 可视化任务验收，不计入本硬指标清单。）
7. compare 主表字段必须从 step_df 重算，Global MISOCP solver 的 objective 值
   只能以 returned_primary_objective_eur 保留，不得进入 storage_profit_total_eur。
8. compare 主表只按 storage_profit_total_eur 排序。
9. compare 不再消费 storage_purchase_cost_eur / storage_sale_revenue_eur / storage_total_profit_eur 历史别名。
10. continuous SoC rollout 的 episode_indices 必须连续（禁止跨 gap carry SoC）。
11. 所有 reward 分项（r_inc / r_pen / r_bonus）使用的 e_bat 必须是 §6.1.1 定义的
    executed_battery_power_kw，三分支语义一致。
12. 训练 window 末端在存在真实下一时刻时必须走 truncated，不得把价值清零；
    只有 train split 真末端或 gap 才允许 terminal。
13. 旧 shared_data / 旧 checkpoint / 旧 record 在各自边界必须显式报错，
    且失败路径有测试覆盖，禁止 dual-path 偷跑。
14. 全量 pytest 通过。
15. GitNexus detect_changes 确认影响范围，HIGH / CRITICAL 风险必须在报告中说明。
```

### 10.2 学习行为指标

训练和 rollout 后至少报告：

```text
1. charge_energy_total_kwh
2. discharge_energy_total_kwh
3. storage_profit_total_eur
4. boundary_penalty_total
5. soc_boundary_regularization_total
6. throughput_bonus_total
7. charge_price_rank_mean
8. discharge_price_rank_mean
9. daily_soc_jump_max
10. final_soc_by_agent
11. madrl_throughput_bonus_weight_final
12. madrl_r_inc_vs_storage_profit_gap_eur
13. window_end_discharge_share
```

期望方向：

```text
daily_soc_jump_max 接近 0
charge_price_rank_mean < discharge_price_rank_mean
boundary_penalty_total 不长期主导 reward
discharge 不再集中发生在每天开头的免费 SoC 阶段
window_end_discharge_share 不应在窗口最后一小段异常升高
```

## 11. 风险和取舍

### 11.1 multi-day 训练会更慢

即使把 `train_window_days` 从 7 降到 3，`episode_limit` 仍会从 96 变成 288。训练轮数不能简单沿用 “100 episodes” 来比较，应改看 interaction steps。

建议后续验收使用：

```text
train_steps
completed_windows
wall_time
storage_profit_total_eur
```

而不是只看 episode 数。

### 11.2 throughput bonus 可能诱导无意义充放电

如果 `throughput_bonus_eur_per_kwh` 太高，智能体会为了 bonus 反复充放电。

因此必须同时监控：

```text
round_trip_loss_cost
charge/discharge price rank
simultaneous or oscillatory behavior
```

第一版 bonus 必须小，只用于打破不动作局部最优。

并且 bonus 必须按 §6.5 退火到 0。验收不能接受“100k step 后 throughput bonus 仍为正”的训练合同。

### 11.3 SoC 正则可能抑制套利

中点 MSE 已从计划中删除，但 flat-bottom boundary regularization 仍可能在权重过高时让策略害怕接近边界。

因此验收不能只看 SoC 漂亮，还要看 storage_profit、price rank、boundary hit rate 和实际充放电量。

### 11.4 overlapping train windows 样本相关性

当前目标 `train_window_days = 3, window_stride_days = 1` 仍会让每个物理时刻最多出现在 3 个窗口里。

```text
on-policy 算法（PPO 等）：影响有限，每次 rollout 按 policy 重新采样，窗口重叠主要影响初始 state 分布。
off-policy 算法（SAC / MATD3 等）：replay buffer 采样会对重叠时段过采样，放大 Q overestimation。
timeline-cache 只能消除重复计算和重复存储，不能消除统计相关性。
```

验收必须监控：

```text
critic loss 震荡
action entropy 震荡
Q value 的分布漂移
```

如果出现显著震荡，调整方案按优先级：

```text
1. 调大 window_stride_days，直到相关性可接受为止。
2. 限制 replay buffer 对同一物理时刻的重复采样（例如按 timestamp 去重）。
3. 进一步缩短 train_window_days。
```

调整必须重新 bump shared_data SCHEMA_VERSION 和 normalization_signature，禁止就地修改。

## 12. 本轮不做的事

```text
1. 不改 MISOCP 口径。
2. 不把 MADRL 内部 reward 当作横向 compare 目标。
3. 不保留 terminal SoC shaping 的开关。
4. 不新增“旧 reward / 新 reward”mode。
5. 不用旧 checkpoint 继续训练。
6. 不用旧 rollout record 混入新 compare。
7. 不在 compare 中消费 storage_purchase_cost_eur / storage_sale_revenue_eur / storage_total_profit_eur 历史别名。
8. 不在 r_inc 中使用非对称 import tax / export subsidy 口径。
9. 不使用 actor raw output 或 pre-env projected action 计算 r_inc / r_pen / r_bonus 任何一项。
10. 不在 continuous SoC rollout 中跨 disjoint date range 做 SoC carry。
11. 不把 Global MISOCP solver 自身的 objective 值作为 compare 主表 storage_profit_total_eur。
```

## 13. 推荐先做的最小闭环

当前剩余最小闭环：

```text
1. 重跑 notebooks/forecast/forecast_lstm.ipynb，产出新的 schema 6 timeline-cache shared-data。
2. 用 train_window_days = 3 跑通 notebooks/madrl/train_base.ipynb。
3. 检查两个口径：
   - storage_profit_total_eur 用于 compare
   - madrl reward components 用于诊断
4. train_base 成功后，再重跑 train_base_safe / train_projection_safe。
5. 最终 compare 只读取新 record；旧 shared_data / 旧 rollout record / 旧经济字段直接失败。
```

只有这个闭环确认后，再扩展到：

```text
train_base_safe
train_projection_safe
完整 100+ 训练
最终 compare
```
