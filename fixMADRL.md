# fixMADRL: 提升套利质量与 SoC 下边界停滞修复计划

## 0. 当前状态

上一阶段已经解决了最紧急的问题：三个储能 agent 不再只有一个在动。

当前已实现并验证的基础训练机制：

```text
1. actor raw action -> SoC-aware feasible mapping -> env step / replay。
2. target-Q 和 actor-loss 使用同一套 SoC-aware feasible mapping。
3. rollout 端加入 feasible random exploration。
4. replay 支持按同一个并行 env 时间线计算 n-step return。
5. critic learning starts 和 actor learning starts 分离。
6. 训练健康检查记录 loss / grad / nonfinite / near-zero grad。
```

50 episode perfect forecast + train_init_soc=soc_min 结果：

```text
steps = 33600
update_calls = 7057
steps/s = 72.91
nonfinite_failure_count = 0
actor_gradient_collapse_agents = []
projection_gap_mean = 0
```

eval_socmin：

```text
total throughput = 13981 kWh
total profit = 121.14 EUR

agent 0: throughput 5425 kWh, profit 68.12 EUR
agent 1: throughput 4285 kWh, profit 21.13 EUR
agent 2: throughput 4271 kWh, profit 31.89 EUR

final_soc = [0.05, 0.05, 0.277]
```

结论：

```text
“agent 不动”已经不是当前主问题。
当前主问题变成：
1. agent 1/2 的 profit per kWh 偏低；
2. SoC 仍然长期贴近 soc_min；
3. 策略会动，但套利质量还不够稳定。
```

## 1. 硬约束

本阶段仍然保持从零训练，不引入专家知识。

必须遵守：

```text
1. 不使用 warm start。
2. 不使用 MPC / oracle imitation。
3. 不改成规则控制器。
4. 不加 terminal SoC shaping。
5. Phase 1 不改 reward 语义。
6. 不加兼容旧字段、旧合同、旧 artifact 的 fallback。
7. 不加开关保留旧训练路径。
8. 不在 notebook 外层临时拼训练逻辑；逻辑归属必须清楚。
```

允许修改：

```text
1. gamma。
2. n_step_return。
3. learning_starts_transitions。
4. actor_learning_starts_transitions。
5. feasible_random_exploration schedule。
6. 诊断与验收指标。
```

只有当 Phase 1 失败时，才进入 Phase 2：potential-based energy inventory value。

## 2. 问题诊断

当前环境奖励的主项是储能利润：

```text
madrl_r_inc
  = discharge_kw * dt * price
  - charge_kw * dt * price
```

总奖励：

```text
r = madrl_r_inc
  - madrl_r_action_penalty
  - madrl_r_soc_regularization
  + madrl_r_throughput_bonus
  - madrl_r_safe_total
```

这意味着：

```text
充电这一步一定先付钱；
放电这一步才赚钱；
电池库存本身没有显式价值；
评估窗口结束时，剩余 SoC 没有残值。
```

所以在有限窗口里，策略倾向于最后把电卖掉并贴近 `soc_min`。这不是 NaN，也不是梯度消失，而是目标函数和有限 horizon 的自然结果。

但当前还不应该立刻改 reward，因为 50 episode 还很短。需要先确认：只靠更长的信用传播和更持续的可行探索，能不能让 agent 1/2 学出更好的低买高卖。

## 3. 数学动机

充电动作的 Q 梯度近似为：

```text
dQ / dc_t
  = -p_t * dt
    + gamma * dV/dsoc_{t+1} * eta * dt / C
```

第一项是即时负收益。第二项是未来库存价值。

如果 critic 学不到足够长的未来价值，则 actor 会认为：

```text
dQ / dc_t < 0
```

于是学成少充电或只在探索推动下充电。

要让 actor 自己学到跨时间套利，必须让 critic 更直接看到：

```text
低价充电 -> 未来高价放电 -> 总回报为正
```

因此本阶段优先改：

```text
1. 更大的 gamma：提高远期收益权重。
2. 更长的 n-step return：让未来放电收益更直接进入 target。
3. 更晚的 actor start：先让 critic 看够样本，再让 actor 追 Q。
4. 更慢衰减的 feasible random exploration：持续覆盖真实可执行充放电轨迹。
```

## 4. Phase 1：不改 reward 的训练强化方案

### 4.1 推荐配置

把当前主实验配置从：

```text
gamma = 0.999
n_step_return = 96
learning_starts = 2 episode windows
actor_learning_starts = 3 episode windows
feasible_random_exploration = 0.50 -> 0.05 over 50000 transitions
```

改成：

```text
gamma = 0.9995
n_step_return = 192
learning_starts = 2 episode windows
actor_learning_starts = 4 episode windows
feasible_random_exploration = 0.70 -> 0.10 over 150000 transitions
```

当前 7 天训练窗口下：

```text
train_episode_limit = 96 * 7 = 672
num_envs = 4

learning_starts_transitions = 672 * 4 * 2 = 5376
actor_learning_starts_transitions = 672 * 4 * 4 = 10752
```

### 4.2 为什么这样调

`gamma=0.999` 时：

```text
gamma^672 = 0.999^672 ≈ 0.51
```

7 天后的价值只剩约一半。

`gamma=0.9995` 时：

```text
gamma^672 = 0.9995^672 ≈ 0.71
```

跨天套利收益更容易被 critic 保留下来。

`n_step_return=96` 只覆盖 1 天：

```text
96 * 15 min = 24 h
```

`n_step_return=192` 覆盖 2 天：

```text
192 * 15 min = 48 h
```

这更适合训练出“今天低价充，明天或后天高价放”的行为。

actor start 从 3 个 episode windows 延后到 4 个 episode windows，是为了降低早期 critic 误导 actor 的风险。

exploration 终值从 0.05 提高到 0.10，衰减步数从 50000 提到 150000，是为了避免 replay 后期过早失去可行充电样本。

### 4.3 不允许的替代

不要用这些方式替代 Phase 1：

```text
1. 不给 agent 手写低价充高价放规则。
2. 不用 MPC 动作填 replay。
3. 不给 actor 加 imitation loss。
4. 不在 reward 里加入 terminal SoC penalty。
5. 不把 soc_min 附近的 action 人工改成强制充电。
```

## 5. 需要修改的 owner

### 5.1 Config Owner

文件：

```text
configs/experiment_config.py
```

修改：

```python
AlgoConfig.gamma = 0.9995
TrainConfig.n_step_return = 192
TrainConfig.feasible_random_exploration_start = 0.70
TrainConfig.feasible_random_exploration_end = 0.10
TrainConfig.feasible_random_exploration_decay_steps = 150000
```

`actor_learning_starts_transitions` 的默认解析从 3 个 episode windows 改成 4 个 episode windows：

```python
return max(critic_start, train_episode_limit * num_envs * 4)
```

### 5.2 Training Contract Owner

文件：

```text
scripts/checkpoints.py
scripts/mainline_madrl.py
scripts/utils/grid_notebook_workflow.py
scripts/mainline_compare.py
```

要求：

```text
1. training_contract 必须继续记录 gamma / n_step_return / learning starts / exploration schedule。
2. 旧 checkpoint 缺少这些字段时继续 fail explicitly。
3. 不加旧字段兼容。
```

如果当前合同尚未记录 `gamma`，需要新增：

```text
discount_gamma
```

这样不同 gamma 训练出来的 checkpoint 不会被错误混用。

### 5.3 Replay Owner

文件：

```text
scripts/utils/replay_buffer.py
```

要求：

```text
1. n_step_return=192 不能跨并行 env lane。
2. 不能跨 terminated / truncated。
3. bootstrap_discount = gamma^k，k 为实际可 bootstrap 的步数。
4. buffer_size 必须仍然能容纳足够的 n-step 采样窗口。
```

### 5.4 Agent Owner

文件：

```text
controllers/madrl/base_agent.py
```

要求：

```text
1. critic 使用 replay 给出的 n-step reward 和 bootstrap_discount。
2. actor 继续使用 feasible mapped action 进入 critic。
3. actor update gate 仍由 shared_ctx["allow_actor_update"] 控制。
4. 训练健康统计继续记录每个 agent 的 grad/loss。
```

## 6. 诊断指标

不能只看 throughput。必须输出以下 per-agent 指标：

```text
throughput_kwh
storage_profit_eur
profit_per_kwh = storage_profit_eur / throughput_kwh
charge_steps
discharge_steps
idle_fraction
charge_avg_price
discharge_avg_price
price_spread = discharge_avg_price - charge_avg_price
soc_min_fraction
soc_max_fraction
final_soc
actor_action_mapping_gap_mean
projection_gap_mean
feasible_random_exploration_source_ratio
actor_grad_norm_mean
actor_near_zero_grad_ratio_last_window
critic_loss_mean
nonfinite_failure_count
```

新增诊断脚本或 helper 时，owner 应属于：

```text
scripts/diagnostics/
```

不要把大段统计逻辑塞进 notebook cell。

## 7. Acceptance Criteria

主验收使用 perfect forecast，训练窗口仍为 7 天，评估区间仍为：

```text
2020-04-01 到 2020-04-15
```

必须同时跑：

```text
eval_socmin
eval_0p5
```

### 7.1 最低通过线

```text
nonfinite_failure_count = 0
actor_gradient_collapse_agents = []
projection_gap_mean ≈ 0

每个 agent:
  throughput_kwh > 3000
  charge_steps > 0
  discharge_steps > 0
  profit_per_kwh > 0
  discharge_avg_price > charge_avg_price
```

### 7.2 SoC 下边界验收

`soc_min_fraction` 不要求立刻很低，因为有限窗口套利自然会在某些高价后放空。

但不能长期完全贴边：

```text
每个 agent soc_min_fraction < 0.45
```

如果 agent 1 或 agent 2：

```text
profit_per_kwh > 0
discharge_avg_price > charge_avg_price
soc_min_fraction 仍然高
```

说明套利方向对了，但库存残值缺失仍然存在。

如果：

```text
profit_per_kwh <= 0
或 discharge_avg_price <= charge_avg_price
```

说明训练信号仍不足，先不要进入 Phase 2，要继续检查 exploration / n-step / gamma。

## 8. 实验矩阵

### 8.1 主实验 A：100 episodes

```text
prediction_mode = perfect
train_episodes = 100
train_init_soc_low = soc_min
train_init_soc_high = soc_min
init_soc = soc_min
gamma = 0.9995
n_step_return = 192
actor_learning_starts = 4 episode windows
feasible_random_exploration = 0.70 -> 0.10 over 150000 transitions
reward 不变
```

目标：

```text
看 agent 1/2 的 profit_per_kWh 是否明显改善。
```

### 8.2 主实验 B：150 episodes

如果 100 episodes 仍在改善，则跑 150 episodes。

目标：

```text
验证 SoC 下边界停滞是否随训练变少，而不是只在 50 episodes 过早收敛。
```

### 8.3 对照实验 C：n_step_return=96

保持其他参数不变，只改：

```text
n_step_return = 96
```

目标：

```text
确认 192-step 的改善来自更长信用传播，而不是探索 schedule 单独造成。
```

### 8.4 对照实验 D：gamma=0.999

保持其他参数不变，只改：

```text
gamma = 0.999
```

目标：

```text
确认 gamma 提高是否确实改善跨天套利。
```

## 9. Phase 2：Potential-Based Energy Value

只有在 Phase 1 满足以下情况时，才进入 Phase 2：

```text
1. 每个 agent 都能正 profit_per_kWh；
2. discharge_avg_price > charge_avg_price；
3. 但 soc_min_fraction 仍长期过高；
4. final_soc 仍系统性贴 soc_min；
5. 继续增加训练 episode 没有改善。
```

### 9.1 动机

当前 reward 没有库存残值：

```text
窗口末端留电 = 没奖励
窗口末端卖电 = 有奖励
```

因此最终贴 `soc_min` 是有限 horizon 的自然最优倾向。

更合理的修复不是 terminal SoC penalty，而是 potential-based shaping：

```text
r'_t = r_t + gamma * Phi(soc_{t+1}, t+1) - Phi(soc_t, t)
```

`Phi` 表示电池库存的未来套利价值。

### 9.2 Phi 的候选定义

只允许使用当前训练窗口内可观测价格统计，不使用专家动作：

```text
Phi(soc_t, t)
  = stored_energy_kwh(t) * max(0, future_price_quantile_75(t:t+H) - current_price_t)
```

或更保守：

```text
Phi(soc_t, t)
  = stored_energy_kwh(t) * future_price_spread_value_t
```

其中 `future_price_spread_value_t` 必须从 observation / shared-data owner 明确生成，不能在 reward 里临时偷看不可用信息。

### 9.3 Phase 2 硬约束

```text
1. 不允许 terminal SoC target。
2. 不允许固定要求 final_soc=0.5。
3. 不允许用 MPC value function。
4. 不允许使用 test period 统计量训练。
5. 必须新增明确 reward contract，例如:
   reward_contract = "madrl_storage_profit_with_potential_inventory_v1"
6. 旧 checkpoint 必须 fail explicitly。
```

Phase 2 是单独任务，不与 Phase 1 混在同一个实验里。

## 10. GitNexus Impact Commands

实施 Phase 1 前必须跑：

```powershell
.\node_modules\.bin\gitnexus.cmd impact "TrainConfig" --direction upstream --repo MADRL_ESS --include-tests
.\node_modules\.bin\gitnexus.cmd impact "AlgoConfig" --direction upstream --repo MADRL_ESS --include-tests
.\node_modules\.bin\gitnexus.cmd impact "build_training_contract" --direction upstream --repo MADRL_ESS --include-tests
.\node_modules\.bin\gitnexus.cmd impact "validate_checkpoint_training_contract" --direction upstream --repo MADRL_ESS --include-tests
.\node_modules\.bin\gitnexus.cmd impact "ReplayBuffer" --direction upstream --repo MADRL_ESS --include-tests
.\node_modules\.bin\gitnexus.cmd impact "MATD3" --direction upstream --repo MADRL_ESS --include-tests
.\node_modules\.bin\gitnexus.cmd impact "MADDPG" --direction upstream --repo MADRL_ESS --include-tests
.\node_modules\.bin\gitnexus.cmd impact "select_action_batch_with_info" --direction upstream --repo MADRL_ESS --include-tests
```

完成前必须跑：

```powershell
.\node_modules\.bin\gitnexus.cmd detect_changes --scope all --json --repo MADRL_ESS
```

## 11. Targeted Tests

Phase 1 修改后至少跑：

```powershell
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest `
  tests/test_replay_buffer_batch.py `
  tests/test_model_assembly.py `
  tests/test_train_mainline_launcher.py `
  tests/test_checkpoints.py `
  tests/test_notebook_utils.py `
  tests/test_grid_notebook_workflow.py `
  tests/test_mainline_train_runner_equivalence.py `
  tests/test_normal_reward.py
```

新增或更新测试：

```text
1. training_contract 记录 gamma。
2. actor_learning_starts 默认解析为 4 episode windows。
3. n_step_return=192 时 replay 不跨 env lane。
4. feasible random exploration schedule 使用 0.70 -> 0.10 / 150000。
5. 诊断输出 profit_per_kWh、charge/discharge avg price、soc_min_fraction。
```

## 12. 最终判断逻辑

如果 Phase 1 成功：

```text
说明 agent 1/2 的套利质量问题主要来自 credit assignment 和探索不足。
下一步扩大到 300 episodes，并恢复 normal/LSTM shared-data 评估。
```

如果 Phase 1 部分成功：

```text
profit_per_kWh 变正，但 soc_min_fraction 仍高。
进入 Phase 2，单独设计 potential-based energy inventory value。
```

如果 Phase 1 失败：

```text
profit_per_kWh 仍不稳定，或 discharge_avg_price <= charge_avg_price。
不要急着改 reward；先检查 critic loss、exploration source ratio、n-step target、price observation。
```

本计划的原则是：

```text
先让 RL 自己学出跨时套利；
只有确认目标函数缺库存价值时，才改 reward contract。
```
