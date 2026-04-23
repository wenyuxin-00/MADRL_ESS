# 七个方案目标函数改造计划：最大化储能收益

## 0. 当前基线

当前已提交版本为：

```text
1b3f093 最小化运行成本
```

当前核心目标是最小化运行成本：

```text
cost = import_price * grid_import_energy
     - export_subsidy * grid_export_energy
     + safety / SoC / regularization penalties
```

现在统一 compare 主指标要改为：

```text
maximize storage_profit
```

也就是七个方案都以储能收益为目标，而不是以购电成本最小为目标。
其中 Local MPC、ADMM MPC、Global MISOCP 的优化目标直接按这个口径写；MADRL SAFE 方案可以继续在训练 reward 中保留原有安全惩罚，但最终 compare 仍只按储能收益排序。

## 1. 推荐的新统一目标定义

我建议先明确一个全项目统一的收益口径，避免七个方案看起来都叫“最大化收益”，但实际算的不是同一个东西。

推荐定义：

```text
price_t = wholesale_price_t + fixed_price_markup

storage_profit_t =
    price_t * battery_discharge_energy_t
  - price_t * battery_charge_energy_t
```

其中：

```text
battery_charge_energy_t    = max(e_bat_t, 0) * dt
battery_discharge_energy_t = max(-e_bat_t, 0) * dt
```

注意当前代码里 `e_bat > 0` 表示充电，`e_bat < 0` 表示放电。

购电价和售电价都使用同一个实时价格：

```text
real_time_price_t = wholesale_price_t + cfg.reward.import_price_markup_eur_per_kwh
```

因此应移除“固定 export_subsidy 作为售电价格”的优化含义。`export_subsidy_eur_per_kwh` 可以保留为兼容字段，但新目标中不再作为主要售电价。

## 2. 先做一个共享目标口径

### 2.1 新增配置字段

文件：

```text
configs/experiment_config.py
```

建议在 `RewardConfig` 中增加：

```python
storage_objective_mode: str = "max_storage_profit"
storage_price_mode: str = "real_time_price"
storage_profit_weight: float = 1.0
```

第一版不加入额外成本项，也不把安全惩罚放进统一 compare 主指标。统一目标只衡量储能套利本身。

### 2.2 新增共享计算函数

推荐新增一个明确 owner，例如：

```text
scripts/utils/storage_profit.py
```

包含：

```python
compute_storage_profit(...)
compute_storage_profit_components(...)
```

统一输出：

```text
storage_charge_cost
storage_discharge_revenue
storage_profit
storage_objective
```

这样七个方案都引用同一个收益口径，减少“同名不同义”的风险。

## 3. 修改方案一：MADRL + No Safety

方案名：

```text
madrl_base
```

涉及文件：

```text
envs/rewards/NormalReward.py
configs/experiment_config.py
notebooks/madrl/train_base.ipynb
```

当前 reward：

```text
reward = - purchase_cost
       + export_subsidy
       - safety_penalties
```

修改后 reward：

```text
reward = storage_profit
```

对 `madrl_base`，安全惩罚权重仍保持 0：

```text
w_voltage_pen = 0
w_line_pen = 0
w_trafo_pen = 0
```

步骤：

1. 在 `NormalReward.compute(...)` 中从 `actual_grid_power_t` 改为读取电池功率 `e_bat` 或 `battery_power_t`。
2. 如果 env_state 还没有传入电池功率，需要在 `GridEnv.step(...)` 的 `reward_state` 中加入：

   ```python
   "battery_power_t": storage_state["e_bat"]
   ```

3. 用实时价格计算：

   ```text
   charge_cost = price_t * max(e_bat, 0) * dt
   discharge_revenue = price_t * max(-e_bat, 0) * dt
   storage_profit = discharge_revenue - charge_cost
   ```

4. `total reward = storage_profit`。
5. 更新 reward component：

   ```text
   r_storage_discharge_revenue
   r_storage_charge_cost
   r_storage_profit
   ```

6. 重新训练 `train_base.ipynb`。

## 4. 修改方案二：MADRL + Safety Penalty

方案名：

```text
madrl_base_safe
```

涉及文件：

```text
envs/rewards/NormalReward.py
configs/experiment_config.py
notebooks/madrl/train_base_safe.ipynb
```

训练 reward：

```text
storage_profit - voltage_penalty - line_penalty - trafo_penalty
```

统一 compare 主指标仍然只用 `storage_profit = 储能售电收益 - 储能购电成本`，不扣这些安全惩罚。

步骤：

1. 复用方案一的 `storage_profit`。
2. 保留当前安全惩罚：

   ```text
   w_voltage_pen = 400
   w_line_pen = 0
   w_trafo_pen = 10
   ```

3. 新 reward：

   ```text
   reward = storage_profit
          - r_safe_v
          - r_safe_line
          - r_safe_trafo
   ```

4. 确认 `component_meta` 里收益项是正号，惩罚项是负号。
5. 重新训练 `train_base_safe.ipynb`。

## 5. 修改方案三：MADRL + Safety Projection

方案名：

```text
madrl_projection_safe
```

涉及文件：

```text
envs/rewards/NormalReward.py
controllers/madrl/safety_projector.py
configs/experiment_config.py
notebooks/madrl/train_projection_safe.ipynb
```

训练 reward：

```text
storage_profit - projection/action_penalty - safety_penalties
```

统一 compare 主指标仍然只用 `storage_profit = 储能售电收益 - 储能购电成本`。

步骤：

1. 复用方案二的 reward。
2. 保留 safety projector 对动作的修正。
3. 当前 projection 会通过 `r_soc_pen` / `action_penalty` 记录动作修正惩罚。
4. 推荐将这个惩罚重命名或显式解释为：

   ```text
   projection_penalty
   ```

   但第一版不需要重命名；SAFE 方案保留原有安全惩罚作为训练约束，不进入统一 compare 主收益口径。

   ```text
   reward = storage_profit
          - projection_penalty
          - safety_penalties
   ```

5. 重新训练 `train_projection_safe.ipynb`。

## 6. 修改方案四：Local MPC + Perfect Forecast

方案名：

```text
local_mpc_perfect
```

涉及文件：

```text
controllers/mpc/gurobi_agent_mpc.py
scripts/mainline_compare.py
notebooks/madrl/local_MPC.ipynb
```

当前 Local MPC 是最小化 net-grid 成本：

```text
min price_t * grid_import
  - export_subsidy * grid_export
```

修改为最大化储能收益。Gurobi 仍用 minimization 表达：

```text
min - storage_profit
```

即：

```text
min price_t * charge_energy
  - price_t * discharge_energy
  + tiny_regularization
```

步骤：

1. 在 `_ReusableLocalMPCSolver._apply_problem_data(...)` 中修改 objective。
2. 不再让 `grid_export.Obj` 使用固定 `export_subsidy` 作为售电价。
3. 对电池变量直接设置目标：

   ```python
   charge.Obj = price_t * dt
   discharge.Obj = -price_t * dt
   ```

4. PV curtailment 第一版建议保持 0 成本，避免和储能收益目标混在一起。
5. `solve_result.objective_eur` 改成新目标的经济值，或者新增：

   ```text
   storage_profit_eur
   returned_primary_objective_eur
   ```

6. 用 perfect forecast rerun `local_MPC.ipynb`。

## 7. 修改方案五：Local MPC + LSTM Forecast

方案名：

```text
local_mpc_lstm
```

涉及文件同方案四：

```text
controllers/mpc/gurobi_agent_mpc.py
scripts/mainline_compare.py
notebooks/madrl/local_MPC.ipynb
predictors/shared_data.py
envs/observation/default_builder.py
```

目标函数和方案四完全一致，唯一差异是 forecast 输入来自 LSTM shared-data。

步骤：

1. 先完成方案四。
2. 确保 LSTM shared-data horizon 与 `cfg.env.future_horizon` 一致。
3. 用相同目标函数跑 LSTM forecast。
4. Compare 时检查：

   ```text
   local_mpc_perfect >= local_mpc_lstm
   ```

   这里的 `>=` 是指储能收益应该通常不低于 LSTM 预测方案。

## 8. 修改方案六：ADMM MPC + LSTM Forecast

方案名：

```text
admm_mpc_lstm
```

涉及文件：

```text
scripts/utils/admm_mpc_solver.py
scripts/utils/admm_mpc_notebook_helpers.py
notebooks/madrl/ADMM_mpc.ipynb
```

当前 ADMM 局部目标是运行成本形式：

```text
objective += dt * export_subsidy * net_load
objective += dt * (import_price - export_subsidy) * n_pos
objective += ADMM quadratic penalty
```

修改为：

```text
min - storage_profit + ADMM quadratic penalty
```

也就是：

```text
objective += dt * price_t * charge
objective += -dt * price_t * discharge
objective += ADMM quadratic consensus penalty
```

步骤：

1. 在 `_ReusableAdmmLocalSolver.solve(...)` 中替换经济目标项。
2. 保留 ADMM 协调项：

   ```text
   0.5 * rho * ...
   ```

3. 不恢复 terminal SoC shaping。
4. `build_admm_mpc_window_data(...)` 中仍然使用实时价格：

   ```text
   import_price = wholesale_price + markup
   ```

   但命名上建议后续改成 `real_time_price_eur_per_kwh`。

5. 更新 ADMM rollout meta：

   ```text
   admm_objective_mode = "max_storage_profit"
   admm_terminal_cost_mode = "none"
   ```

6. rerun `ADMM_mpc.ipynb`。

## 9. 修改方案七：Global MISOCP

方案名：

```text
global_misocp
```

涉及文件：

```text
controllers/mpc/global_socp_mpc.py
scripts/mainline_compare.py
notebooks/madrl/global_MISOCP.ipynb
```

当前目标在 `GlobalMISOCPProblem._build_sequence_model(...)`：

```text
agent purchase cost
- agent export subsidy
+ throughput regularization
+ branch current tiebreaker
```

修改为：

```text
min - storage_profit
  + throughput_regularization
  + branch_current_tiebreaker
```

具体：

```text
storage_profit =
  Σ price_t * discharge_mwh * 1000
- Σ price_t * charge_mwh * 1000
```

步骤：

1. 在 `_build_sequence_model(...)` 中删除或旁路基于 `agent_abs_grid` 的经济目标项。
2. 直接对 `p_charge` / `p_discharge` 加目标：

   ```python
   objective_terms.append(1e3 * dt * price_t * p_charge)
   objective_terms.append(-1e3 * dt * price_t * p_discharge)
   ```

3. 保留物理网络硬约束：

   ```text
   voltage
   line loading
   transformer loading
   SoC bounds
   PV curtailment bounds
   ```

4. 保留 throughput regularization，避免同时充放电。
5. 结果结构里新增或重命名：

   ```text
   storage_charge_cost_eur
   storage_discharge_revenue_eur
   storage_profit_eur
   ```

6. rerun `global_MISOCP.ipynb`。

## 10. 修改统一评价与 compare

涉及文件：

```text
scripts/utils/grid_notebook_workflow.py
scripts/mainline_compare.py
scripts/plots/grid_notebook_plotting.py
notebooks/madrl/compare.ipynb
```

当前 compare 主要看：

```text
purchase_cost_total
export_subsidy_total
total_cost_eur
objective_total
```

新 compare 推荐改为：

```text
storage_charge_cost_total_eur
storage_discharge_revenue_total_eur
storage_profit_total_eur
```

其中：

```text
storage_profit_total_eur =
    storage_discharge_revenue_total_eur
  - storage_charge_cost_total_eur
```

如果 `objective_total` 保持“越小越好”的旧语义，会非常容易误读。推荐 compare 主表不再使用 `objective_total`，只使用：

```text
storage_profit_total_eur   # 统一总目标函数，越大越好
```

solver 内部的 minimization 值可以单独记录为诊断字段，例如 `storage_objective_total_eur = -storage_profit_total_eur`，但不参与 compare 排名。

## 11. 推荐实施顺序

### Phase 1：只改评价，不改控制器

目的：确认新收益口径能从现有 rollout 中正确算出来。

步骤：

1. 新增 `storage_profit.py`。
2. 修改 `_build_rollout_records(...)`，增加 storage profit columns。
3. 修改 `compare_rollout_metrics(...)`，增加 storage profit metrics。
4. 不改七个控制器。
5. 用旧记录跑 compare，确认数值和符号正确。

推荐先做这个阶段。因为它风险最低，而且能先统一“收益到底怎么算”。

### Phase 2：改 Local MPC perfect/LSTM

目的：先改最简单、最容易验证的优化器。

步骤：

1. 改 `gurobi_agent_mpc.py`。
2. 跑 Local MPC perfect。
3. 跑 Local MPC LSTM。
4. 对比 perfect 和 LSTM。

通过标准：

```text
perfect storage_profit 通常 >= LSTM storage_profit
```

### Phase 3：改 ADMM MPC

目的：让 ADMM 和 Local MPC 使用同一储能收益目标。

步骤：

1. 改 `admm_mpc_solver.py`。
2. 保持 terminal SoC shaping = none。
3. rerun ADMM notebook。
4. 检查 ADMM 是否不再异常优于 perfect oracle。

### Phase 4：改 Global MISOCP

目的：建立新目标下的全局上界/物理 oracle。

步骤：

1. 改 `global_socp_mpc.py`。
2. 加收益字段到 `MISOCPResult`。
3. rerun global MISOCP。
4. 如果仍 infeasible，先解决物理可行性，不要把 infeasible 和目标函数问题混在一起。

### Phase 5：改三个 MADRL reward 并重新训练

目的：让学习策略真的学习最大化储能收益。

步骤：

1. 改 `NormalReward.compute(...)`。
2. 更新三个 MADRL notebook spec。
3. 重新训练：

   ```text
   train_base.ipynb
   train_base_safe.ipynb
   train_projection_safe.ipynb
   ```

4. 重新保存 rollout records。

### Phase 6：最终 compare

目的：生成七个方案在同一目标下的比较。

步骤：

1. 确认七个 scheme 的 record 都是新目标版本。
2. rerun `compare.ipynb`。
3. 主表按：

   ```text
   storage_profit_total_eur
   ```

   降序排序。

## 12. 推荐决策

我推荐采用：

```text
先统一评价口径，再逐个改优化器，最后重新训练 MADRL。
```

原因：

1. 当前七个方案来自三类机制：RL reward、单体 MPC、ADMM、Global MISOCP。
2. 如果直接一起改，很难判断收益变化来自目标函数、预测 horizon、物理约束，还是记录口径。
3. 先加统一 storage profit 指标，可以立刻发现符号错误，例如充电被算成收入、放电被算成成本。
4. Local MPC 最容易验证，所以先改它；MADRL 最耗时，所以最后重训。

## 13. 需要审阅确认的问题

请先确认这三个定义：

1. 售电价是否完全等于：

   ```text
   wholesale_price + import_price_markup
   ```

   而不再使用 `export_subsidy_eur_per_kwh`。

2. 储能收益是否只计算电池充放电：

   ```text
   discharge_revenue - charge_cost
   ```

   不把 PV 自发自用收益单独算进储能收益。

3. MADRL safe / projection safe 是否保留原有安全惩罚，但不把安全惩罚纳入统一 compare 主收益口径：

   ```text
   统一 compare 主指标 = 储能售电收益 - 储能购电成本
   ```

   当前计划按这个口径执行。

## 14. 行级修改清单

下面行号基于当前工作区文件。真正实施时行号会随前面的编辑略微移动，但函数名和替换位置保持不变。

### 14.1 配置层

文件：

```text
configs/experiment_config.py
```

当前位置：

```text
RewardConfig: lines 239-246
MADRL_NOTEBOOK_SPECS: lines 33-79
ADMM_NOTEBOOK_SPEC: lines 80-90
COMPARE_SCHEME_ORDER: lines 15-23
```

修改：

1. 在 `RewardConfig` 的 line 246 后新增：

   ```python
   storage_objective_mode: str = "max_storage_profit"
   storage_price_mode: str = "real_time_price"
   storage_profit_weight: float = 1.0
   ```

2. 保留：

   ```python
   import_price_markup_eur_per_kwh: float = 0.20
   ```

   这是新目标下购电和售电共用的固定电价加成。

3. `export_subsidy_eur_per_kwh` 第一版不删除，但在计划中降级为历史字段；所有新目标函数不再使用它作为售电价。

4. `MADRL_NOTEBOOK_SPECS` 的三组 reward 配置不需要重复写 `storage_objective_mode`，除非需要方案级覆盖。第一版推荐只用全局 `RewardConfig`。

### 14.2 新增共享收益口径

新增文件：

```text
scripts/utils/storage_profit.py
```

新增函数：

```python
def compute_storage_profit_components(
    *,
    battery_power_kw,
    price_eur_per_kwh,
    dt_hours,
):
    ...
```

统一符号：

```text
battery_power_kw > 0  => charge
battery_power_kw < 0  => discharge
charge_kw             = max(battery_power_kw, 0)
discharge_kw          = max(-battery_power_kw, 0)
charge_cost_eur       = price * charge_kw * dt
discharge_revenue_eur = price * discharge_kw * dt
storage_profit_eur    = discharge_revenue_eur - charge_cost_eur
solver_objective_eur  = -storage_profit_eur
```

输出字段固定为：

```text
storage_charge_cost_eur
storage_discharge_revenue_eur
storage_profit_eur
storage_objective_eur
```

所有七个方案的记录、汇总、compare 都只认这组字段。这样统计口径不会漂。

### 14.3 GridEnv 给 reward 传入电池功率

文件：

```text
envs/grid_env.py
```

当前位置：

```text
GridEnv.step: line 81
```

当前 `reward_state` 是：

```python
reward_state = {
    "import_price_t": signal_state["import_price_t"],
    "actual_grid_power_t": signal_state["net_load"],
    "dt": self.dt,
    ...
}
```

修改为新增：

```python
"battery_power_t": storage_state["e_bat"],
"storage_price_t": signal_state["import_price_t"],
```

保留 `actual_grid_power_t`，因为安全、旧诊断和部分测试可能还需要它。

### 14.4 MADRL reward 目标

文件：

```text
envs/rewards/NormalReward.py
```

当前位置：

```text
NormalReward.__init__: line 7-9
NormalReward.component_meta: line 10-11
NormalReward.compute: line 12-17
```

修改：

1. line 9 读取新配置：

   ```python
   self.storage_profit_weight = float(getattr(cfg.reward, "storage_profit_weight", 1.0))
   ```

2. line 11 替换 component meta，至少包含：

   ```text
   r_storage_discharge_revenue
   r_storage_charge_cost
   r_storage_profit
   r_soc_pen
   r_safe_v
   r_safe_line
   r_safe_trafo
   ```

3. line 13 不再用 `actual_grid_power_t` 计算 purchase/export 作为 reward 主项。改为：

   ```python
   battery_power_t = np.asarray(env_state["battery_power_t"], dtype=np.float32)
   price_t = float(env_state.get("storage_price_t", env_state["import_price_t"]))
   charge_kw = np.maximum(battery_power_t, 0.0)
   discharge_kw = np.maximum(-battery_power_t, 0.0)
   r_storage_charge_cost = charge_kw * dt * price_t
   r_storage_discharge_revenue = discharge_kw * dt * price_t
   r_storage_profit = r_storage_discharge_revenue - r_storage_charge_cost
   ```

4. line 17 的总 reward 改为：

   ```python
   total = (
       self.storage_profit_weight * r_storage_profit
       - r_soc_pen
       - r_safe_v
       - r_safe_line
       - r_safe_trafo
   ).astype(np.float32)
   ```

   SAFE 相关惩罚只保留在 MADRL 的训练 reward 中，不进入七个方案统一 compare 主指标。

5. 三个 MADRL 方案区别仍来自 `configs/experiment_config.py`：

   ```text
   madrl_base: lines 41-46, safety weights = 0
   madrl_base_safe: lines 55-60, voltage/trafo penalties enabled
   madrl_projection_safe: lines 69-77, projection enabled
   ```

### 14.5 Local MPC Perfect / LSTM 目标

文件：

```text
controllers/mpc/gurobi_agent_mpc.py
scripts/mainline_compare.py
```

当前位置：

```text
_ReusableLocalMPCSolver._apply_problem_data: lines 64-70
_ReusableLocalMPCSolver._solve_prepared result economics: line 89
collect_local_mpc_rollout postprocess: scripts/mainline_compare.py lines 94-99
```

修改：

1. `gurobi_agent_mpc.py:70` 当前：

   ```python
   self.charge[step_idx].Obj = throughput_penalty
   self.discharge[step_idx].Obj = throughput_penalty
   self.grid_import[step_idx].Obj = (price_t + GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH) * self.dt
   self.grid_export[step_idx].Obj = (-self.export_subsidy_eur_per_kwh + GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH) * self.dt
   ```

   改为：

   ```python
   self.charge[step_idx].Obj = price_t * self.dt + throughput_penalty
   self.discharge[step_idx].Obj = -price_t * self.dt + throughput_penalty
   self.grid_import[step_idx].Obj = GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH * self.dt
   self.grid_export[step_idx].Obj = GRID_EXPORT_TIEBREAKER_EPS_EUR_PER_KWH * self.dt
   ```

   这里 Gurobi 仍然最小化，所以 `discharge` 是负成本，等价于最大化放电收入。

2. 不新增额外成本参数。第一版目标只看储能充电成本和放电收益。

3. `gurobi_agent_mpc.py:89` 当前 `objective_eur` 仍是旧的网侧购售电成本：

   ```python
   objective_eur = sum((prices * grid_import_kw - subsidy * grid_export_kw) * dt)
   ```

   改成统一 storage profit 口径：

   ```python
   charge_cost = np.sum(prices * solution.charge_kw * self.dt)
   discharge_revenue = np.sum(prices * solution.discharge_kw * self.dt)
   storage_profit = discharge_revenue - charge_cost
   objective_eur = -storage_profit
   ```

4. `mainline_compare.py:96-99` 当前会强制把 Local MPC 的 `objective_total` 改回旧成本口径。必须删除或替换为：

   ```python
   rollout.meta["local_mpc_objective_mode"] = "max_storage_profit"
   ```

   不再重写 `objective_total = purchase_cost - export_subsidy`。

Local MPC Perfect 和 Local MPC LSTM 共用同一套代码，区别只在 `prediction_mode`。

### 14.6 ADMM MPC 目标

文件：

```text
scripts/utils/admm_mpc_solver.py
scripts/utils/admm_mpc_notebook_helpers.py
```

当前位置：

```text
build_admm_mpc_window_data: lines 38-49
_ReusableAdmmLocalSolver.solve: lines 64-73
ADMM economic objective: lines 67-70
```

修改：

1. line 45 当前仍读取 `export_subsidy` 并做 `import_price - subsidy` 检查：

   ```python
   export_subsidy = ...
   min_gap = min(import_price - export_subsidy)
   ```

   新目标不再需要这个约束。删除 `min_gap` 检查，保留：

   ```python
   real_time_price = derive_import_price_seq(...)
   ```

2. `AdmmMpcWindowData` line 13 中字段建议从：

   ```text
   import_price_eur_per_kwh
   export_subsidy_eur_per_kwh
   ```

   改为：

   ```text
   real_time_price_eur_per_kwh
   ```

   如果担心改动太大，第一版可暂时保留字段名 `import_price_eur_per_kwh`，但文档和 meta 必须标注它现在是实时购售电价。

3. line 67-69 当前：

   ```python
   import_minus_subsidy = ...
   objective += dt * export_subsidy * net_load
   objective += dt * import_minus_subsidy * n_pos
   ```

   改为直接对电池变量计价：

   ```python
   price_t = float(window_data.import_price_eur_per_kwh[step_idx])
   objective += float(window_data.dt_hours * price_t) * self.charge[step_idx]
   objective += float(window_data.dt_hours * -price_t) * self.discharge[step_idx]
   objective += float(window_data.dt_hours * _THROUGHPUT_TIEBREAKER_EUR_PER_KWH) * (self.charge[step_idx] + self.discharge[step_idx])
   ```

4. line 70 的 ADMM quadratic consensus penalty 保留不变。

5. `scripts/utils/admm_mpc_notebook_helpers.py` 的 rollout meta 增加：

   ```python
   meta["admm_objective_mode"] = "max_storage_profit"
   meta["admm_price_mode"] = "real_time_price"
   meta["admm_terminal_cost_mode"] = "none"
   ```

### 14.7 Global MISOCP 目标

文件：

```text
controllers/mpc/global_socp_mpc.py
scripts/mainline_compare.py
```

当前位置：

```text
GlobalMISOCPProblem._build_sequence_model: lines 104-134
subsidy and price constraint: lines 110-111
objective terms: lines 113-118
result economics extraction: around line 197 in current compact file
```

修改：

1. line 110-111 当前用 `subsidy` 约束：

   ```python
   subsidy = ...
   if min(import_price_seq) <= subsidy: raise ...
   ```

   新目标不需要 `price > subsidy`。删除这个检查。

2. line 112 当前仍创建：

   ```python
   agent_abs_grid = model.addVars(...)
   ```

   新目标不需要为了经济目标引入 `agent_abs_grid`，但如果其他诊断还引用它，可以保留变量但不再进目标。推荐第一版保留，降低结构改动。

3. line 114 保留 throughput regularization。

4. line 115 当前经济项是：

   ```python
   objective_terms.append(0.5 * (price - subsidy) * agent_abs_grid)
   objective_terms.append(0.5 * (price + subsidy) * agent_net_grid_expr)
   ```

   替换为：

   ```python
   objective_terms.append(1e3 * self.dt_hours * price_value * p_charge[agent_idx, step_idx])
   objective_terms.append(1e3 * self.dt_hours * -price_value * p_discharge[agent_idx, step_idx])
   ```

5. `agent_net_grid_expr` 仍需要保留用于网络功率平衡相关表达，但不再作为收益目标主项。

6. 结果提取处新增：

   ```text
   storage_charge_cost_eur
   storage_discharge_revenue_eur
   storage_profit_eur
   ```

   并让 `stage1_primary_objective_eur` 代表 solver objective，即 `-storage_profit + regularization + tiebreaker`。

7. `scripts/mainline_compare.py` 的 `collect_global_full_horizon_rollout(...)` meta 增加：

   ```python
   "global_objective_mode": "max_storage_profit",
   "global_price_mode": "real_time_price",
   "storage_profit_eur": float(result.storage_profit_eur),
   ```

## 15. 统计口径一致性

统一原则：所有方案最终比较时只按实际执行动作和实际实时电价计算收益，不按预测电价计算收益。

### 15.1 统一 step/agent/summary 字段

文件：

```text
scripts/utils/grid_notebook_workflow.py
```

当前位置：

```text
_build_rollout_records: lines 103-108
```

line 104 当前已拿到：

```python
battery_power
battery_charge
battery_discharge
info[IMPORT_PRICE_COLUMN]
```

在 line 104 经济计算附近新增：

```python
storage_charge_cost_per_agent = battery_charge * dt_hours * info[IMPORT_PRICE_COLUMN]
storage_discharge_revenue_per_agent = battery_discharge * dt_hours * info[IMPORT_PRICE_COLUMN]
storage_profit_per_agent = (
    storage_discharge_revenue_per_agent
    - storage_charge_cost_per_agent
)
```

注意：这里不纳入负荷购电成本、PV 收益或安全惩罚。负荷只作为背景影响潮流与约束，compare 主目标只看储能本身。

### 15.2 step_df 必须新增字段

在 `step_row` line 104 的 dict 中新增：

```text
storage_charge_cost_total_eur
storage_discharge_revenue_total_eur
storage_profit_total_eur
storage_objective_total_eur
```

定义：

```text
storage_profit_total_eur = sum(storage_profit_per_agent)
storage_objective_total_eur = -storage_profit_total_eur
```

安全惩罚字段可以继续作为 MADRL/SAFE 诊断列存在，但不再派生“扣惩罚后收益”作为统一比较指标。

### 15.3 agent_df 必须新增字段

在 `agent_row` line 108 中新增：

```text
storage_charge_cost_eur
storage_discharge_revenue_eur
storage_profit_eur
```

`summary` 不再只 groupby：

```text
purchase_cost, export_subsidy, objective_total
```

而是 groupby：

```text
storage_charge_cost_eur
storage_discharge_revenue_eur
storage_profit_eur
purchase_cost
export_subsidy
```

旧字段可以保留为诊断列，但不再作为主比较目标。

### 15.4 objective_total 的新语义

为了避免“越小越好”和“越大越好”混淆，推荐这样处理：

```text
storage_profit_total_eur               # compare 统一总目标函数，越大越好
storage_objective_total_eur             # solver 诊断字段，等于 -storage_profit，越小越好
```

`objective_total` 第一版推荐保留为旧兼容字段，但 compare 不再用它排序。

## 16. Compare 修改清单

### 16.1 compare 指标汇总

文件：

```text
scripts/mainline_compare.py
```

当前位置：

```text
COMPARE_METRIC_COLUMNS: line 8
ECONOMIC_TABLE_COLUMNS: line 9
summarize_rollout_metrics: lines 34-48
build_compare_economic_table: lines 50-54
```

修改：

1. line 8 的 `COMPARE_METRIC_COLUMNS` 新增：

   ```text
   storage_charge_cost_total_eur
   storage_discharge_revenue_total_eur
   storage_profit_total_eur
   storage_objective_total_eur
   ```

2. line 9 的 `ECONOMIC_TABLE_COLUMNS` 替换为：

   ```python
   ECONOMIC_TABLE_COLUMNS = [
       "controller",
       "storage_discharge_revenue_total_eur",
       "storage_charge_cost_total_eur",
       "storage_profit_total_eur",
   ]
   ```

3. line 38 从 `step_df` 汇总新字段：

   ```python
   storage_profit_total = float(step_df["storage_profit_total_eur"].sum())
   ```

4. line 45 的 `metrics` dict 增加这些字段。

5. line 50-54 的 `build_compare_economic_table(...)` 改为：

   ```python
   table = metrics_df.loc[:, ECONOMIC_TABLE_COLUMNS].copy()
   return table.sort_values("storage_profit_total_eur", ascending=False)
   ```

   统一比较口径固定按 `storage_profit_total_eur` 降序。SAFE 的安全惩罚不进入经济排名。

### 16.2 compare notebook

文件：

```text
notebooks/madrl/compare.ipynb
```

当前位置：

```text
line 70: load_rollout_record(...)
line 71: metrics_df = compare_rollout_metrics(*rollouts)
line 72: economic_table = build_compare_economic_table(metrics_df)
```

修改：

1. 保留 line 70-72 的主流程。
2. 在显示 `economic_table` 时，主排序列必须是：

   ```text
   storage_profit_total_eur
   ```

3. 在 compare notebook 中增加一个显式检查：

   ```python
   required_profit_columns = {
       "storage_profit_total_eur",
   }
   missing = required_profit_columns.difference(metrics_df.columns)
   if missing:
       raise ValueError(f"Compare requires storage-profit metrics. Missing: {sorted(missing)}. Rerun all seven rollout notebooks after the max-profit objective migration.")
   ```

这样可以防止旧 record 混进新 compare。

### 16.3 compare 绘图

文件：

```text
scripts/plots/grid_notebook_plotting.py
```

当前位置：

```text
plot_price_prediction_comparison: lines 64-68
plot_battery_power_and_soc_comparison: lines 94-108
```

新增函数：

```python
def plot_storage_profit_comparison(*rollouts, figsize=None):
    ...
```

要求 step_df 必须包含：

```text
timestamp
storage_charge_cost_total_eur
storage_discharge_revenue_total_eur
storage_profit_total_eur
```

图建议三行：

```text
1. cumulative storage_profit_total_eur
2. charge cost vs discharge revenue
3. per-step storage_profit_total_eur
```

## 17. 七个方案完成标准

每个方案 rerun 后，record 必须满足：

```text
step.parquet has storage_profit_total_eur
agent.parquet has storage_profit_eur
summary.parquet has storage_profit_eur
metrics.parquet has storage_profit_total_eur
meta.json has objective_mode = max_storage_profit
```

七个 scheme：

```text
global_misocp
local_mpc_perfect
local_mpc_lstm
admm_mpc_lstm
madrl_base
madrl_base_safe
madrl_projection_safe
```

最终 compare 只允许读取全部包含新收益字段的 record。否则直接失败并提示重跑对应 notebook。

## 18. 推荐审阅顺序

推荐你按下面顺序逐步审阅：

1. 先审阅第 15 节：统计口径是否正确定义了“储能收益”。
2. 再审阅第 16 节：compare 是否按你想看的收益列排序。
3. 然后审阅第 14.5 节：Local MPC 的目标函数，因为它最容易验证。
4. 再审阅第 14.6 节：ADMM 目标函数。
5. 再审阅第 14.7 节：Global MISOCP 目标函数。
6. 最后审阅第 14.4 节：MADRL reward，因为这个改完需要重新训练。
