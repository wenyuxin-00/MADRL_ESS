# MADRL_ESS

这份 README 是当前代码状态的重新学习地图。它不追溯历史实验分支，也不描述已经回退的中间方案；下面所有说明都以当前主线代码为准。

项目目标是比较多户 prosumer 场景下的储能控制方法：集中或分布式 MPC 作为强基线，MADRL 作为可学习控制器。所有方法最终都通过同一个 `GridEnv` 和同一套物理、电价、储能、潮流评估逻辑来落地，因此比较结果主要来自控制器本身，而不是环境口径差异。

## 1. 当前主要入口

| 入口 | 作用 |
| --- | --- |
| `notebooks/forecast/forecast_lstm.ipynb` | 生成 LSTM 预测结果和 shared-data 缓存。perfect forecast 场景可以不依赖 LSTM 权重，但仍可使用 shared-data 加速。 |
| `notebooks/madrl/train_base.ipynb` | 当前默认 MADRL 训练入口：`MATD3`，perfect data，无遮挡安全惩罚，使用 SoC-aware actor action mapping。 |
| `notebooks/madrl/train_base_safe.ipynb` | `MATD3` 加安全 reward penalty，不做 projection。 |
| `notebooks/madrl/train_projection_safe.ipynb` | `MATD3_SAFE_POC`，actor action 先做 SoC-aware mapping，再做 grid safety projection。 |
| `notebooks/madrl/local_MPC.ipynb` | 本地 MPC rollout，每个 agent 用本地预测和本地储能约束独立求解。 |
| `notebooks/madrl/global_MISOCP.ipynb` | 全局网络约束 MPC / oracle rollout。 |
| `notebooks/madrl/ADMM_mpc.ipynb` | ADMM MPC rollout，默认面向 LSTM forecast workflow。 |
| `notebooks/madrl/compare.ipynb` | 汇总 MADRL、MPC、ADMM 记录并做横向对比。 |

代码化入口主要在：

- `configs/experiment_config.py`: 实验、环境、reward、模型、训练、预测、MPC 的配置源。
- `scripts/builder.py`: 把配置装配成 dataset、observation builder、env、runner。
- `scripts/train.py`: MADRL 训练主循环。
- `scripts/mainline_compare.py`: MADRL / MPC / oracle 对比 rollout 主线。
- `scripts/utils/grid_notebook_workflow.py`: notebook 共享的 rollout 记录、保存、summary 逻辑。

## 2. 当前实验默认事实

当前 canonical 场景是 3 个 prosumer agent：

- agent profiles: `SFH12`, `SFH18`, `SFH20`
- bus ids: `12`, `4`, `2`
- load scale: `20.0`
- PV scale: `1.0`
- battery capacity: `100 kWh` per agent
- test window: `2020-04-01` 到 `2020-04-15`
- control time step: `dt = 0.25 h`
- one episode: `96` steps，也就是一天
- train window: 默认 `7` 天滚动窗口
- MADRL action per agent: `[battery_action, pv_action]`
- environment action range: 每个维度仍是 `[-1, 1]`

`train_base` 当前 notebook spec 的关键设置：

- algorithm: `MATD3`
- prediction mode: `perfect`
- parallel envs: `4`
- training episodes: 当前配置里默认 `50`
- initial SoC: 固定 `0.05`
- SoC range: `[0.05, 0.95]`
- battery max charge/discharge rate: `0.5 C`
- discount factor: `gamma = 0.999`
- actor / critic learning rate: `1e-4`
- replay batch size: `512`
- n-step return: `96`

这些默认值都集中在 `configs/experiment_config.py`。如果 notebook 里覆盖了配置，以 notebook spec 为准。

## 3. 数据架构

数据链路可以按下面顺序理解：

```text
raw csv
  -> ProsumerDataset
  -> forecast / shared-data
  -> DefaultObservationBuilder
  -> GridEnv
  -> controller action
  -> reward / grid metrics / rollout records
```

### 3.1 原始数据和 ProsumerDataset

数据 owner 是 `data/loaders/prosumer.py` 里的 `ProsumerDataset`。它从数据目录读取：

- `household.csv`
- `heatpump.csv`
- `pv_reference.csv`
- `price.csv`

然后根据配置选择 agent profile、年份、日期范围和 scale：

- household + heatpump 组成用户 load；
- PV reference 通过 profile scale / capacity 变成 agent PV；
- wholesale price 作为电价基础；
- train split 使用滚动多日窗口；
- test split 使用测试日期窗口；
- 每个 episode 记录 `active_start_idx`, `active_end_idx`, `next_active_idx` 等时间索引。

### 3.2 Forecast 和 shared-data

预测层在 `predictors/` 下。当前主要有两类：

- `perfect`: 直接使用真实未来数据，适合先验证控制逻辑上限。
- `lstm`: 使用 LSTM 预测 artifact，适合现实预测误差场景。

shared-data owner 是 `predictors/shared_data.py`。它把观测序列预先生成成 memmap / manifest 缓存，避免训练时重复构造 forecast sequence。当前 shared-data 合同包括：

- schema version: `8`
- layout: `timeline_cache_v1`
- price contract: `oracle_window_relative_price_v1`

缓存中保存的典型字段：

- `calendar_time`
- `wholesale_price_seq`
- `wholesale_price_rank_seq`
- `wholesale_price_relative_seq`
- `wholesale_price_spread_seq`
- `load_seq`
- `pv_seq`
- `mu_t`
- `mu_next`

重要区别：

- 价格归一化特征只进入神经网络 observation；
- reward 仍使用物理单位的真实电价 EUR/kWh；
- 所以“电价归一化影响策略学习”，但不会直接改写真实经济收益计算。

### 3.3 ObservationBuilder

`DefaultObservationBuilder` 把 dataset 或 shared-data 行转换成模型观测。当前配置中关键 observation features 是：

- local features: `calendar_time`, `soc`
- sequence features: `wholesale_price_relative`, `wholesale_price_spread`, `load`, `pv`

直觉上：

- `soc` 告诉 actor 当前电池状态；
- `price_relative` / `price_spread` 告诉 actor 当前和未来价格形状；
- `load` / `pv` 告诉 actor 本地供需和可用 PV；
- calendar time 帮助模型学习日内周期。

## 4. GridEnv 和 action 语义

`GridEnv` 在 `envs/grid_env.py`，是所有控制器共同进入物理世界的地方。

每个 agent 的 action 是二维：

```text
[battery_action, pv_action]
```

### 4.1 Battery action

环境 action space 仍是归一化的 `[-1, 1]`：

```text
requested_battery_power_kw = battery_action * p_max_kw
```

符号约定：

- `battery_power_t > 0`: 充电，增加 net load，产生购电成本；
- `battery_power_t < 0`: 放电，降低 net load 或上网，产生售电收益；
- `battery_power_t = 0`: 电池不动作。

储能状态更新：

```text
if battery_power_t >= 0:
    delta_e = battery_power_t * efficiency * dt
else:
    delta_e = battery_power_t / efficiency * dt

energy_next = energy_current + delta_e
soc_next = energy_next / capacity
```

`GridEnv` 会验证最终执行动作是否满足本地 SoC 约束。现在 MADRL actor 在进环境前已经做 SoC-aware mapping，所以正常情况下不应该再依赖环境硬裁剪来“救”动作。

### 4.2 PV action

PV action 也是 `[-1, 1]`：

```text
pv_effective_kw = pv_raw_kw * 0.5 * (pv_action + 1)
```

所以：

- `pv_action = -1`: 全 curtail；
- `pv_action = +1`: 全部使用可用 PV；
- PV 不与电池 SoC 耦合，clamp 到 `[-1, 1]` 就是本地 feasible。

### 4.3 Net load 和潮流

环境先计算本地净负荷：

```text
base_net_load_effective = load_kw - pv_effective_kw
net_load_kw = base_net_load_effective + battery_power_kw
```

然后通过 `GridCore` / pandapower 网络计算：

- bus voltage
- line loading
- transformer loading
- voltage / line / transformer violation

这些 grid metrics 会进入 safety reward、rollout records 和 compare summary。

## 5. 当前奖励模型

奖励 owner 是 `envs/rewards/NormalReward.py`。当前 reward 不是 action feasibility regularization 版本；当前有效字段仍是：

- `action_boundary_penalty_weight`
- `soc_boundary_regularization_weight`
- `throughput_bonus_eur_per_kwh_max`
- `w_voltage_pen`
- `w_line_pen`
- `w_trafo_pen`

`NormalReward` 会显式拒绝一些旧实验字段，例如 `action_feasibility_regularization_weight`、`terminal_soc_value_weight` 等，避免旧合同悄悄混进当前主线。

### 5.1 总公式

每个 agent 每一步的 reward 是：

```text
reward
  = madrl_r_inc
  - madrl_r_action_penalty
  - madrl_r_soc_regularization
  + madrl_r_throughput_bonus
  - madrl_r_safe_total
```

内部记录字段 `madrl_r_total_internal` 与上式一致。

### 5.2 madrl_r_inc: 储能套利收益

当前核心目标是最大化 storage profit：

```text
madrl_r_inc = discharge_revenue - charge_cost
```

也就是：

- 放电时，`battery_power_t < 0`，赚当前价格；
- 充电时，`battery_power_t > 0`，付当前价格；
- 价格使用真实物理价格，不使用归一化后的 observation price。

这解释了一个重要现象：如果初始 SoC 有电，而模型还没学会跨时间等待高价，它很容易先学到“现在卖掉立刻拿正 reward”。后续要靠高折扣因子、n-step return、价格序列、探索和动作映射让它学到跨时间套利。

### 5.3 madrl_r_action_penalty: SoC 边界动作惩罚

当前这个字段保留的是边界动作惩罚，不是 action feasibility regularization。

它惩罚的是：

- SoC 已经接近 `soc_min`，还继续放电；
- SoC 已经接近 `soc_max`，还继续充电。

权重来自：

```text
RewardConfig.action_boundary_penalty_weight = 0.05
```

由于当前 MADRL actor 会先做 SoC-aware mapping，正常情况下 actor 不应该频繁产生这种边界硬撞行为。

### 5.4 madrl_r_soc_regularization: SoC 边界软正则

这是一个靠近 SoC 上下边界时的二次惩罚，鼓励电池不要长期贴边：

```text
soc_boundary_regularization_weight = 0.005
soc_boundary_margin = 0.02
```

它不是 terminal value，也不是强制回到 `soc_target`；它只是边界附近的软惩罚。

### 5.5 madrl_r_throughput_bonus: 早期吞吐探索奖励

当前代码里保留了一个早期 throughput bonus：

```text
madrl_r_throughput_bonus = weight_t * abs(battery_power_kw) * dt
```

这个 bonus 会随训练进度衰减：

- 训练早期帮助 agent 探索充放电；
- 中后期逐渐消失；
- 最终策略仍主要由 storage profit 和 safety penalty 决定。

### 5.6 madrl_r_safe_total: 电网安全惩罚

安全惩罚由三部分组成：

```text
madrl_r_safe_total
  = madrl_r_safe_v
  + madrl_r_safe_line
  + madrl_r_safe_trafo
```

对应：

- voltage violation
- line loading violation
- transformer loading violation

在 `train_base` 里这些 safety 权重为 0；在 `train_base_safe` 和 `train_projection_safe` 里会启用 voltage / transformer penalty。

## 6. 神经网络架构

当前模型装配 owner 是 `models/assembly.py`。主线只保留 `model.family = "mlp"`。

### 6.1 Actor

Actor 结构：

```text
actor observation
  -> ActorObservationAdapter
  -> MLPEncoder
  -> DeterministicContinuousActorHead
  -> raw action in [-1, 1]
```

`MLPEncoder` 是两层 MLP：

```text
Linear(input_dim, hidden_dim)
ReLU
Linear(hidden_dim, hidden_dim)
ReLU
```

默认：

- `hidden_dim = 256`
- actor output head: deterministic continuous
- output activation: `tanh`
- raw action dim: 2, 对应 battery 和 PV

Actor 是 decentralised policy：每个 agent 的 actor 主要看自己的 local observation 和与自己相关的 sequence observation。

### 6.2 Critic

Critic 结构：

```text
joint observation + joint action
  -> CriticObservationActionAdapter
  -> MLPEncoder
  -> Q head
```

Critic 是 centralized critic：它看所有 agents 的 observation 和所有 agents 的 actions。

算法差异：

- `MADDPG`: single Q critic
- `MATD3`: twin Q critic
- `MATD3_SAFE_POC`: twin Q critic + safety projector 路径

### 6.3 Actor raw output 和最终动作不是同一个东西

这是当前代码最容易混乱的点。

Actor 网络输出的是 raw action：

```text
raw_action in [-1, 1]
```

但 MADRL actor-generated action 在进入环境、replay、target-Q 和 actor-loss 之前，会先做 SoC-aware mapping。

当前 mapping 在 `controllers/madrl/safety_projector.py`：

```text
mapped_battery_kw
  = lower_feasible_kw
  + 0.5 * (raw_battery_action + 1.0)
    * (upper_feasible_kw - lower_feasible_kw)

mapped_battery_action = mapped_battery_kw / p_max_kw
```

其中：

- `lower_feasible_kw <= 0`: 当前 SoC 下还能放多少电；
- `upper_feasible_kw >= 0`: 当前 SoC 下还能充多少电；
- raw `-1` 表示选当前最大可行放电；
- raw `+1` 表示选当前最大可行充电；
- raw `0` 表示选当前可行区间中点。

PV branch 不走 SoC mapping，只做 clamp，因为 PV feasible range 与 SoC 无关。

这套 mapping 被一致用于三个地方：

- rollout action selection
- target-Q action
- actor-loss policy action

这保证 critic 看到的 action 分布、replay 里的 action 分布、actor 被优化的 action 分布是一致的。

## 7. MADRL 训练逻辑链

MADRL 训练主线在 `notebooks/madrl/train_base.ipynb`、`scripts/mainline_madrl.py`、`scripts/train.py` 和 `controllers/madrl/base_agent.py`。如果只想重新建立脑图，可以先看下面这个调用树。

### 7.1 训练主调用树

这棵树按“从外到内”的顺序写：越上层越接近 notebook，越下层越接近每个 step 的神经网络和环境交互。

```text
notebooks/madrl/train_base.ipynb
|-- Cell 3: force_retrain_madrl
|   |-- False: 严格读取当前 canonical checkpoint
|   `-- True: 重新启动 mainline 训练
|
|-- Cell 4: 构造 cfg
|   |-- MADRL_NOTEBOOK_SPECS["train_base"]
|   |-- compose_experiment_config(...)
|   |-- apply_notebook_experiment_settings(...)
|   |-- cfg.algo.name = "MATD3"
|   |-- cfg.train.num_envs = 4
|   |-- cfg.train.vec_env_type = "subproc"
|   |-- cfg.reward.* = spec["reward"]
|   `-- configure_torch_runtime(...)
|
`-- Cell 5: resolve_madrl_notebook_training(...)
    |-- if force_retrain_madrl is False
    |   `-- load_madrl_training_result(...)
    |       |-- find_latest_training_run(...)
    |       |-- validate training_contract
    |       `-- return model_root + train_result
    |
    `-- if force_retrain_madrl is True
        `-- run_external_train_mainline(...)
            |-- write JSON controls
            |   |-- experiment_controls.json
            |   |-- data_controls.json
            |   |-- battery_controls.json
            |   |-- train_controls.json
            |   `-- checkpoint_controls.json
            |
            `-- python -m scripts.mainline_madrl
                |-- main(...)
                |   |-- compose_experiment_config(...)
                |   |-- _apply_model_controls(...)
                |   |-- _apply_runtime_controls(...)
                |   |-- _apply_reward_controls(...)
                |   |-- _apply_safety_controls(...)
                |   |-- apply_mainline_experiment_settings(...)
                |   |-- _apply_train_controls(...)
                |   |-- configure_torch_runtime(...)
                |   |-- ensure_mainline_forecast_ready(...)
                |   |-- build_train_runner(...)
                |   |-- runner.run()
                |   |-- runner.save_model(...)
                |   |-- runner.build_reward_summary()
                |   `-- write train_result.json
                |
                `-- build_train_runner(cfg)
                    |-- build_env(cfg, mode="train")
                    |   |-- ProsumerDataset
                    |   |-- PerfectForecaster or LSTM forecaster
                    |   |-- PrecomputedObservationStore if shared-data is enabled
                    |   |-- DefaultObservationBuilder
                    |   |-- GridCore
                    |   |-- GridEnv
                    |   `-- DummyVecEnv or SubprocVecEnv
                    |
                    |-- build_env(cfg, mode="test")
                    |   `-- single evaluation GridEnv
                    |
                    `-- TrainRunner(cfg, train_env, eval_env)
                        |-- get_agent_cls(cfg.algo.name)
                        |-- create agent_0 / agent_1 / agent_2
                        |   |-- ActorNetwork
                        |   |-- CriticNetwork
                        |   |-- target ActorNetwork
                        |   `-- target CriticNetwork
                        |-- ReplayBuffer
                        `-- SummaryWriter
```

训练真正发生在 `TrainRunner.run()` 里面。它可以继续展开成第二棵树：

```text
TrainRunner.run()
|-- resolve train budget
|   |-- train_episode_limit = 96
|   |-- target_total_steps = train_episodes * episode_limit
|   |-- learning_starts_transitions
|   `-- actor_learning_starts_transitions
|
|-- env.reset()
|
`-- while total_steps < target_total_steps
    |-- select_action_batch_with_info(...)
    |   |-- each agent actor outputs raw_action in [-1, 1]
    |   |-- map_actor_output_to_soc_feasible_action_torch(...)
    |   |   |-- battery raw -> current feasible charge/discharge interval
    |   |   `-- PV raw -> clamp only
    |   |-- feasible random exploration inside current feasible interval
    |   |-- optional JointGridSafetyProjector for MATD3_SAFE_POC
    |   |-- enforce_local_action_feasibility_torch(...)
    |   `-- compute_action_gap_metrics_torch(...)
    |
    |-- env.step(mapped_actions)
    |   |-- GridEnv.step(...)
    |   |   |-- normalized action -> battery_power_kw
    |   |   |-- normalized action -> pv_effective_kw
    |   |   |-- update battery energy and SoC
    |   |   |-- GridCore.step(...)
    |   |   |   `-- pandapower power-flow metrics
    |   |   |-- NormalReward.compute(...)
    |   |   `-- return next_obs, reward, done, info
    |   |
    |   `-- vector env aggregates per-env outputs
    |
    |-- merge action_info into step info
    |
    |-- ReplayBuffer.add(...)
    |   |-- obs
    |   |-- mapped action actually sent to env
    |   |-- adjusted reward
    |   |-- next_obs
    |   `-- done / bootstrap metadata
    |
    |-- if replay is ready
    |   `-- for each agent: agent.update(replay_batch)
    |       |-- MATD3 critic update
    |       |   |-- target actor raw action
    |       |   |-- SoC-aware mapping for target-Q
    |       |   |-- target policy smoothing noise
    |       |   |-- twin target critics
    |       |   |-- Bellman target
    |       |   `-- critic optimizer step
    |       |
    |       `-- delayed actor update
    |           |-- policy raw action
    |           |-- SoC-aware mapping for actor-loss
    |           |-- critic evaluates mapped joint action
    |           |-- actor loss = -Q
    |           `-- actor optimizer step
    |
    |-- update tqdm progress and episode summaries
    |
    `-- when an env episode ends
        |-- record total reward
        |-- record reward components
        |-- record throughput bonus weight
        `-- reset that env to the next sampled training window
```

训练结束后，notebook 还会继续走 evaluation / record 链路：

```text
train_result + model_root
`-- collect_madrl_rollout(...)
    |-- load_madrl_controller(...)
    |   |-- build_env(cfg, mode="test")
    |   |-- resolve_checkpoint_to_load(...)
    |   |-- create agents
    |   `-- load actor / critic weights
    |
    |-- collect_controller_rollout(...)
    |   |-- env.reset()
    |   |-- MADRLController.act(...)
    |   |   |-- actor raw action
    |   |   |-- SoC-aware mapping
    |   |   |-- optional safety projection
    |   |   `-- residual local guard
    |   |-- env.step(...)
    |   `-- build step_df / agent_df / grid_df
    |
    |-- compare_rollout_metrics(...)
    `-- save_rollout_record(...)
        |-- step.parquet
        |-- agent.parquet
        |-- grid.parquet
        |-- summary.parquet
        |-- metrics.parquet
        |-- meta.json
        `-- manifest.json
```

所以，最短的“金字塔记忆法”是：

```text
Notebook
  -> mainline_madrl
    -> build_train_runner
      -> build_env + build_agents
        -> TrainRunner.run
          -> action mapping
            -> GridEnv.step
              -> NormalReward
          -> ReplayBuffer
          -> MATD3 critic/actor update
    -> save checkpoint/result
  -> collect rollout
  -> save notebook record
```

### 7.2 每一步怎么选动作

在训练 rollout 中：

1. 每个 agent actor 根据 observation 输出 raw action。
2. raw battery action 通过 SoC-aware mapping 变成当前 SoC 下可行的 battery action。
3. exploration 会在当前可行区间内采样，而不是随便采样后硬裁剪。
4. `MATD3_SAFE_POC` 额外调用 grid safety projector。
5. 最后 residual guard 做数值保险。
6. 最终 action 进入 `GridEnv.step`，并存入 replay。

因此 replay 里保存的是控制器实际请求并执行的可行动作，不是 actor raw action。

### 7.3 每一步怎么更新

当前训练不是每个 episode 结束后才更新，而是在每个 env step 后，只要满足 warmup 和 batch 条件，就从 replay buffer 采样更新。

大致是：

```text
for episode:
  for t in episode:
    select action
    env.step
    replay.add
    if enough samples:
      sample batch
      update critics
      update actors every policy_update_freq steps
```

`MATD3` 还包含：

- twin critic target 取 min；
- target policy smoothing noise；
- delayed actor update；
- soft update target networks。

### 7.4 Target-Q 和 actor-loss 为什么也要 mapping

如果只在 rollout 时 mapping，而 target-Q / actor-loss 里仍把 raw action 直接交给 critic，就会出现训练语义错位：

- replay 中 critic 学的是 feasible mapped action；
- target-Q 里 critic 看到的是 raw infeasible action；
- actor-loss 优化的是另一个动作空间。

当前代码避免了这个问题：actor raw action 在 rollout、target-Q、actor-loss 三个路径里都会先变成同一种 SoC-feasible action。

### 7.5 Safe POC 路径

`MATD3_SAFE_POC` 的 action path 是：

```text
actor raw
  -> SoC-aware local mapping
  -> grid safety projection
  -> residual local feasibility guard
  -> env step / critic
```

其中 safety projector 会基于当前 grid 状态和灵敏度近似，把 joint battery/PV action 投影到更安全的网络约束区域。

### 7.6 训练健康检查

`BaseAgent` 里有 loss / gradient finite 检查和梯度诊断。出现 NaN、Inf 或极端梯度问题时，应该优先看：

- critic loss
- actor loss
- actor grad norm
- critic grad norm
- actor near-zero grad ratio
- action mapping gap
- action projection gap
- per-agent battery throughput

如果只看总 reward，很容易误判为“模型没学会”，但其实可能是某个 agent 动作空间、SoC 边界、价格特征或 critic target 已经先出问题。

## 8. MPC 逻辑链

MPC 和 MADRL 共用环境与记录系统，但控制器不同。

### 8.1 Local MPC

入口：

- `notebooks/madrl/local_MPC.ipynb`
- `scripts/mainline_compare.py`

逻辑：

```text
config
  -> build comparison env
  -> choose prediction mode
  -> per-agent local MPC solver
  -> solve finite-horizon battery/PV schedule
  -> take first action
  -> env.step
  -> record rollout
```

Local MPC 的特点：

- 每个 agent 主要根据自己的 load、PV、price forecast 和 battery constraints 做优化；
- 它不学习，不依赖 replay；
- perfect forecast 代表理想预测；
- LSTM forecast 代表现实预测；
- 它可以作为 MADRL 的“局部理性”参考。

Local MPC 不等于全局电网最优。它可以给出合理储能套利，但不会像 global MISOCP 那样完整考虑所有网络约束。

### 8.2 Global MISOCP / SOCP MPC

入口：

- `notebooks/madrl/global_MISOCP.ipynb`
- `controllers/mpc/global_socp_mpc.py`

逻辑：

```text
grid model + forecast horizon
  -> build global optimization problem
  -> battery / PV / voltage / line / transformer variables
  -> network constraints
  -> objective
  -> solve
  -> take first-step action
  -> env.step
  -> record rollout
```

它的优化变量包括：

- battery charge / discharge power
- battery energy
- PV curtailment
- branch active/reactive power
- branch current square
- bus voltage square
- root import/export
- import/export binary gate

它的约束包括：

- battery power and energy bounds
- SoC dynamics
- PV curtailment bounds
- voltage bounds
- branch and transformer loading limits
- radial network flow constraints

它更接近 oracle / upper-bound reference，但求解成本也更高，并且依赖 solver 状态。

### 8.3 ADMM MPC

入口：

- `notebooks/madrl/ADMM_mpc.ipynb`
- `scripts/utils/admm_mpc_notebook_helpers.py`

逻辑：

```text
forecast + env state
  -> initialize ADMM controller
  -> local subproblem per agent
  -> grid coupling / consensus update
  -> residual balancing rho update
  -> iterate until max_iters or tolerance
  -> take first action
  -> env.step
  -> record rollout
```

默认 ADMM notebook spec 是 `ADMM MPC + LSTM Forecast`。它的价值是把全局耦合问题拆成局部优化加协调迭代，在成本和全局性之间折中。

## 9. MADRL 和 MPC 的核心差别

| 维度 | MADRL | Local MPC | Global MISOCP | ADMM MPC |
| --- | --- | --- | --- | --- |
| 是否学习 | 是 | 否 | 否 | 否 |
| 是否需要 replay | 是 | 否 | 否 | 否 |
| 控制策略来源 | neural policy + critic learning | 每步优化 | 全局优化 | 分布式优化 |
| 使用预测 | observation sequence | forecast horizon | forecast horizon | forecast horizon |
| 是否显式优化未来 | 通过 critic / return 学出来 | 是 | 是 | 是 |
| 网络约束 | reward penalty 或 projection | 主要通过 env 评估 | 显式进入优化 | 通过协调近似进入 |
| 速度 | 训练慢，推理快 | 每步中等 | 每步慢 | 每步中等到慢 |

对当前调试最重要的一句话：

MPC 每一步都显式知道未来 horizon，并直接解“现在充电以后能不能高价卖”的优化问题；MADRL 需要通过 reward、discount、n-step return、critic target 和 exploration 自己学出这个跨时间价值。

## 10. Rollout 记录和比较结果

rollout 记录 owner 是 `scripts/utils/grid_notebook_workflow.py`。

典型输出目录：

- `notebooks/record/madrl/madrl_base`
- `notebooks/record/madrl/madrl_base_safe`
- `notebooks/record/madrl/madrl_projection_safe`
- `notebooks/record/mpc/local_mpc_perfect`
- `notebooks/record/mpc/local_mpc_lstm`
- `notebooks/record/mpc/global_misocp`
- `notebooks/record/mpc/admm_mpc_lstm`

每个 rollout 通常包含：

- `step.parquet`: step-level reward、price、grid summary 等；
- `agent.parquet`: agent-level action、SoC、storage profit、throughput 等；
- `grid.parquet`: grid safety metrics；
- `summary.parquet`: 聚合结果；
- `metrics.parquet`: notebook / compare 使用的指标；
- `meta.json` / `manifest.json`: 合同、配置签名和文件索引。

比较逻辑在 `scripts/mainline_compare.py`，它会把不同 controller 的结果统一成同一套 metric columns，例如：

- storage throughput
- storage profit
- charge cost
- discharge revenue
- voltage violation
- transformer violation
- forecast MAE
- ramp / action gap diagnostics

## 11. 重新读代码的推荐顺序

如果你现在觉得架构已经乱了，建议按这个顺序重新建立脑图。

1. 先读 `configs/experiment_config.py`

   只看 `MADRL_NOTEBOOK_SPECS`、`RewardConfig`、`TrainConfig`、`ObservationConfig`、`MPCConfig`。先明确“当前实验到底在跑什么”。

2. 再读 `data/loaders/prosumer.py`

   理解 3 个 agent 的 load、PV、price、episode window 是怎么来的。

3. 再读 `predictors/shared_data.py`

   理解为什么 notebook 可以直接复用 shared-data，price sequence / load sequence / PV sequence 是怎么缓存的。

4. 再读 `envs/grid_env.py`

   重点看 action 如何变成 battery power / PV effective power / net load，以及 env.step 返回什么。

5. 再读 `envs/rewards/NormalReward.py`

   对照 README 里的 reward 公式，看每个 component 是如何进入 `reward` 的。

6. 再读 `controllers/madrl/safety_projector.py`

   重点看 SoC-aware mapping、local feasibility guard、action gap metrics、grid safety projection。

7. 再读 `models/assembly.py`

   理解 actor / critic 的输入输出维度和 centralized critic 结构。

8. 再读 `controllers/madrl/base_agent.py`

   理解 MADDPG / MATD3 / Safe POC 的 critic update、target-Q、actor-loss。

9. 再读 `scripts/train.py`

   把 vectorized env、replay、update schedule、checkpoint 串起来。

10. 最后读 MPC：

   - local MPC: `scripts/mainline_compare.py`
   - global MPC: `controllers/mpc/global_socp_mpc.py`
   - ADMM MPC: `scripts/utils/admm_mpc_notebook_helpers.py`

这个顺序的好处是：先搞清楚“数据和环境真实发生了什么”，再看“神经网络怎么学”，最后看“MPC 为什么更像 oracle”。

## 12. 当前最需要记住的调试原则

1. Reward 用真实物理电价，网络 observation 用归一化价格。

   如果模型不套利，不能只看 reward，要同时看 observation 里的价格峰谷是否清晰。

2. Actor raw action 不是最终 action。

   当前 MADRL 的最终 action 来自 raw action 经过 SoC-aware mapping、可选 safety projection、residual guard 之后的结果。

3. Replay、target-Q、actor-loss 必须使用同一套 action 语义。

   当前代码已经按这个原则组织，这是修复 storage agent 摆烂问题的关键之一。

4. MADRL 趴在 SoC 下边界不一定是 NaN。

   它可能是局部最优：初始有电时立刻卖掉给正 reward，而充电需要先付成本，未来收益要靠 critic 学出来。

5. MPC 强是因为它显式看未来，不是因为 reward 不同。

   Local MPC / global MPC 每一步都把未来 horizon 放进优化问题；MADRL 需要从数据里学出这个未来价值。

6. 判断 agent 是否真的动起来，不要只看总 reward。

   同时看 per-agent throughput、SoC trace、charge/discharge revenue、action mapping gap、projection gap、critic loss 和 actor grad。

7. 不要引入兼容兜底。

   当前代码风格是合同优先：旧字段、旧 artifact、旧 schema 应该在入口失败，而不是静默 fallback。

## 13. 常用命令

优先使用 `MADRL_ESS` conda 环境：

```powershell
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_normal_reward.py
```

运行一组主线测试：

```powershell
C:\Users\10856\miniconda3\envs\MADRL_ESS\python.exe -m pytest tests/test_model_assembly.py tests/test_train_mainline_launcher.py tests/test_grid_notebook_workflow.py tests/test_checkpoints.py tests/test_normal_reward.py
```

查看当前 Git 改动：

```powershell
git status --short
```

提交前检查 GitNexus 影响范围：

```powershell
.\node_modules\.bin\gitnexus.cmd detect_changes --scope all --json --repo MADRL_ESS
```
