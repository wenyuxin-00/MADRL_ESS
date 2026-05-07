# Notebook 重跑与训练加速验收计划

目标：在 MPC/MISOCP 文件整合、MADRL 并行训练、artifact 合同清理之后，重新跑通主线 notebook，确认环节、逻辑和算法结果没有因为代码修改变坏；同时记录训练时间，重点验证 MADRL 改为并行环境后是否真的变快。

## 0. 审阅结论

原计划方向正确，但需要补强四个关键点：

```text
1. run_dir 必须是单一契约，否则 compare 可能读到旧结果。
2. MADRL notebook 不能走 retrain=False 的缓存分支，否则无法验证训练速度。
3. scheme 名和 controller 名必须分清：madrl_base_safe 对应 MADRL_PENALTY。
4. compare.ipynb 当前调用 compare_records，不是 compare_all；验收产物应按真实输出检查。
```

因此本计划把“执行前 notebook 状态”“结果来源”“Go / No-Go 阈值”都写死，避免跑完以后才发现结果不可比。

## 1. 硬约束

本轮只验证主线，不引入新实验分叉：

```text
1. train/eval 默认都跑潮流计算。
2. 不新增关闭潮流、旧缓存、checkpoint resume、latest lookup、fallback schema。
3. notebook 用 nbconvert --execute --inplace 执行，结果直接写回 ipynb。
4. 如需改代码，保持 owner 明确、主线唯一、缺文件或缺字段直接报错。
5. 若代码改动涉及 public symbol 或主执行流，先跑 GitNexus impact，最后跑 detect_changes。
```

运行环境固定：

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python ...
```

## 2. Run Dir 与数据契约

当前主线 notebook 都指向同一个 run：

```text
artifacts/runs/20260504_234339_01e73bd1
```

本轮验收必须保证所有 notebook 使用同一个 `run_dir`：

```text
misocp.ipynb              -> same run_dir
mpc.ipynb                 -> same run_dir
madrl_base.ipynb          -> same run_dir
madrl_base_safe.ipynb     -> same run_dir
madrl_projection_safe.ipynb -> same run_dir
compare.ipynb             -> same run_dir
```

执行前检查：

```powershell
rg -n "run_dir = Path|load_experiment_context|retrain =" notebooks -g "*.ipynb"
```

如果当前 `Cfg`、forecast artifact、share_data 与该 run 不匹配，不做 latest/sibling scan。正确处理方式只有两种：

```text
1. 继续使用该 run_dir，并确认 cfg_hash/share_data 合同匹配。
2. 先重跑 predict.ipynb 生成新 run_dir，然后显式替换六个 notebook 里的 run_dir 字面量。
```

compare 只允许读取本轮同一个 run_dir 下的 records，不能混用旧 run。

## 3. 执行对象与真实名称

需要重新执行并写回结果的 notebook：

| notebook | scheme | controller | 主要产物 |
|---|---|---|---|
| `notebooks/misocp.ipynb` | `global_misocp` | `MISOCP` | `results/perfect/MISOCP/record` |
| `notebooks/mpc.ipynb` | `local_mpc_perfect` | `LOCAL_MPC` | `results/perfect/LOCAL_MPC/record` |
| `notebooks/mpc.ipynb` | `local_mpc_lstm` | `LOCAL_MPC` | `results/lstm/LOCAL_MPC/record` |
| `notebooks/mpc.ipynb` | `admm_mpc_lstm` | `ADMM_MPC` | `results/lstm/ADMM_MPC/record` |
| `notebooks/madrl_base.ipynb` | `madrl_base` | `MADRL_BASE` | `results/lstm/MADRL_BASE/record` |
| `notebooks/madrl_base_safe.ipynb` | `madrl_base_safe` | `MADRL_PENALTY` | `results/lstm/MADRL_PENALTY/record` |
| `notebooks/madrl_projection_safe.ipynb` | `madrl_projection_safe` | `MADRL_PROJECTION` | `results/lstm/MADRL_PROJECTION/record` |
| `notebooks/compare.ipynb` | compare | all above | `tables/compare_metrics.csv` |

`compare.ipynb` 必须最后执行，因为它读取前面所有 record。

## 4. 执行前检查

先确认没有断引用：

```powershell
rg -n "scripts/train|train_all|local_mpc_solver|admm_mpc_solver|misocp_solver|mpc_common" . -g "*.py" -g "*.md" -g "*.ipynb" -g "!artifacts/**" -g "!.git/**" -g "!.conda/**"
```

跑主线测试：

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m pytest tests
```

检查 MADRL notebook 是否会真实重训：

```powershell
rg -n "retrain =|if retrain|load_madrl_record" notebooks/madrl_base.ipynb notebooks/madrl_base_safe.ipynb notebooks/madrl_projection_safe.ipynb
```

本轮要求：

```text
1. 三个 MADRL notebook 都必须执行 500 episodes 训练。
2. 不允许使用 load_madrl_record 分支作为本轮结果来源。
3. 如果 notebook 里保留 retrain 开关，执行前必须设为 retrain=True。
4. 更干净的做法是删除本轮 notebook 中的缓存分支，直接调用 run_madrl_scheme_experiment。
```

如果中途发现结果来自旧 record，本轮验收无效，必须重新执行。

## 5. Notebook 执行命令

逐个执行，便于定位失败点和记录单项耗时：

```powershell
$ErrorActionPreference = "Stop"

$nb = "notebooks/misocp.ipynb"
Measure-Command { & "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb }

$nb = "notebooks/mpc.ipynb"
Measure-Command { & "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb }

$nb = "notebooks/madrl_base.ipynb"
Measure-Command { & "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb }

$nb = "notebooks/madrl_base_safe.ipynb"
Measure-Command { & "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb }

$nb = "notebooks/madrl_projection_safe.ipynb"
Measure-Command { & "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb }
```

前五个 notebook 全部成功后，再跑 compare：

```powershell
$nb = "notebooks/compare.ipynb"
Measure-Command { & "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb }
```

每个 notebook 的 `TotalSeconds` 写入第 10 节。

## 6. MADRL 并行提速验收

从下列文件读取训练时间：

```text
tables/madrl_train_summary_madrl_base.csv
tables/madrl_train_summary_madrl_base_safe.csv
tables/madrl_train_summary_madrl_projection_safe.csv
```

至少记录：

```text
controller
scheme
episodes
elapsed_s
s_per_episode
steps
updates
env_step_s
sample_update_s
other_s
num_envs
device
model_path
record_dir
```

当前主线预期：

```text
vec_env       = SubprocVecEnv
replay_buffer = array-backed ReplayBuffer
power_flow    = enabled
train_episodes = 500
```

历史基准：

```text
old MADRL_BASE 500 episodes:
  elapsed_s     = 9289.66
  s_per_episode = 18.58

old shared critic 500 episodes:
  elapsed_s     = 9520.67
  s_per_episode = 19.04

50 episodes acceleration benchmark:
  sync_current              = 17.36 s/episode
  array_replay              = 16.62 s/episode
  subproc_vecenv            =  8.34 s/episode
  subproc_plus_array_replay =  7.33 s/episode
```

速度判断：

```text
MADRL_BASE <= 12.5 s/episode: Go，说明并行训练收益明确。
12.5 < MADRL_BASE < 16.0: Warning，需要结合 timers 判断瓶颈。
MADRL_BASE >= 16.0: No-Go，基本没有吃到并行收益。
```

safe/projection safe 可以慢于 base，但不能退回旧串行水平。若 projection safe 明显慢，需要额外记录 safety projection 耗时是否进入瓶颈。

## 7. MPC/MISOCP 验收

`misocp.ipynb`、`mpc.ipynb` 主要验证整合后的 controller 逻辑没有坏。

必须记录：

```text
notebook
controller
forecast_mode
run_dir
record_dir
wall_time_s
act_time_s_mean or solve_time_s_mean
episode_reward_mean
storage_profit_eur_mean
voltage_violation_steps_mean
min_vm_pu
max_vm_pu
trafo_loading_max_pct
trafo_overload_steps
```

验收判断：

```text
1. MISOCP、LOCAL_MPC、ADMM_MPC 都能生成 record 和 manifest。
2. controller import 不再依赖已删除 solver 文件。
3. action 转换统一来自 GridEnv 动作语义。
4. solve_time / act_time 不出现空列或旧字段。
5. 电压、变压器、安全指标没有明显异常。
```

## 8. Compare 验收

`compare.ipynb` 当前调用：

```text
compare_records(cfg, run_dir)
```

它读取的固定方案来自 `scripts.compare.COMPARE_SCHEMES`：

```text
perfect/MISOCP
perfect/LOCAL_MPC
lstm/LOCAL_MPC
lstm/ADMM_MPC
lstm/MADRL_BASE
lstm/MADRL_PENALTY
lstm/MADRL_PROJECTION
```

核心输出：

```text
tables/compare_metrics.csv
tables/economic_table.csv
tables/safety_table.csv
tables/compare.xlsx
figures/compare_*.png
```

必须记录：

```text
controller
forecast_mode
total_reward
storage_profit_eur
voltage_violation_steps
trafo_overload_steps
trafo_loading_max_pct
min_vm_pu
max_vm_pu
```

验收判断：

```text
1. compare 缺任一 record 时应直接失败。
2. 七个 compare scheme 都进入 compare_metrics。
3. MADRL 三个 controller 都进入 economic/safety table。
4. 若收益或安全指标明显变坏，回查对应 notebook 的 record_dir 和 metrics_df。
```

不要把 `compare_records` 和 `compare_all` 混用：前者比较 rollout record，后者读取 eval result。当前 `compare.ipynb` 验收以前者为准。

## 9. 额外记录项

执行时同步记录：

```text
1. 是否出现 worker 死锁、子进程未关闭、notebook 重跑残留进程。
2. Gurobi/MISOCP 是否出现 infeasible、time limit、异常 solve_time。
3. MADRL 是否出现 reward 曲线突然崩坏、safe 项异常增大。
4. compare 图是否有空图、缺曲线、缺 controller。
5. CPU/GPU 占用是否仍然都很低；若低，结合 env_step_s/sample_update_s 判断瓶颈。
6. artifacts/runs 中本轮 run_id、cfg_hash、share_data 路径。
7. 当前 git diff 是否包含与本轮重跑无关的代码变化。
```

## 10. 本轮结果记录表

本轮执行 run：

```text
run_dir = artifacts/runs/20260504_234339_01e73bd1
cfg_hash = 98aff7d4
share_data = artifacts/runs/20260504_234339_01e73bd1/share_data
```

收尾校验：

```text
pytest tests = 22 passed
GitNexus detect_changes = completed, risk critical
critical 原因 = 本轮工作区包含 controller/eval/GridEnv/MADRL notebook 主执行流改动；notebook 已重跑验证。
```

结论摘要：

```text
1. 六个 notebook 均已执行并写回输出。
2. MADRL_BASE 和 MADRL_PENALTY 都实际重训 500 episodes，速度约 8 s/episode，明显快于旧 18.58 s/episode。
3. MADRL_PROJECTION 也完成 500 episodes，但速度约 19.97 s/episode，未吃到端到端提速；瓶颈从 env_step 转移到 sample_update/action_sample。
4. compare 读取七个主线 record 成功，MISOCP/MPC/MADRL 都进入 compare_metrics/economic_table/safety_table。
5. 安全结果整体正常：projection 电压越限为 0，变压器过载 6 步；penalty 电压越限 1 步，但收益几乎归零。
```

### 10.1 Notebook 执行耗时

| notebook | status | wall_time_s | run_dir | note |
|---|---:|---:|---|---|
| misocp.ipynb | OK | 47.61 | `20260504_234339_01e73bd1` | 写回 MISOCP 输出 |
| mpc.ipynb | OK | 390.40 | `20260504_234339_01e73bd1` | 写回 LOCAL_MPC/ADMM_MPC 输出 |
| madrl_base.ipynb | OK | 4030.37 | `20260504_234339_01e73bd1` | 真实重训 500 episodes |
| madrl_base_safe.ipynb | OK | 4050.05 | `20260504_234339_01e73bd1` | 真实重训 500 episodes |
| madrl_projection_safe.ipynb | OK | 10014.99 | `20260504_234339_01e73bd1` | nbconvert 外层 7204s 超时，但原进程继续完成并写回 |
| compare.ipynb | OK | 50.80 | `20260504_234339_01e73bd1` | 读取七个主线 record 成功 |

### 10.2 MADRL 训练速度

| scheme | controller | episodes | elapsed_s | s_per_episode | old_s_per_episode | speedup | env_step_s | sample_update_s | other_s | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| madrl_base | MADRL_BASE | 500 | 3971.08 | 7.94 | 18.58 | 2.34x | 1780.23 | 1916.49 | 13.70 | Go，`subproc + array` 生效 |
| madrl_base_safe | MADRL_PENALTY | 500 | 3988.70 | 7.98 | 18.58 ref | 2.33x | 1786.52 | 1926.90 | 13.78 | Go，耗时与 base 基本一致 |
| madrl_projection_safe | MADRL_PROJECTION | 500 | 9983.57 | 19.97 | 18.58 ref | 0.93x | 1844.21 | 7107.82 | 14.79 | Speed No-Go，瓶颈在 update/action，不在 env |

补充计时：

```text
MADRL_BASE action_sample_s       = 221.96
MADRL_PENALTY action_sample_s    = 222.83
MADRL_PROJECTION action_sample_s = 977.02

MADRL_BASE replay_add_s          = 38.70
MADRL_PENALTY replay_add_s       = 38.67
MADRL_PROJECTION replay_add_s    = 39.73
```

可以看出 projection 的 replay 写入不慢，主要是 action 生成和 sample/update 慢。

### 10.3 MPC/MISOCP 结果

| forecast_mode | controller | reward_mean | profit_mean | violation_mean | trafo_overload_steps | trafo_max_pct | act_time_s_mean | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| perfect | MISOCP | 42.30 | 40.30 | 1.60 | 0 | 97.86 | 0.0013 | 全局 MISOCP 正常 |
| perfect | LOCAL_MPC | 38.86 | 38.43 | 1.60 | 157 | 170.05 | 0.0126 | 收益高，但变压器过载明显 |
| lstm | LOCAL_MPC | 17.48 | 17.68 | 1.33 | 106 | 170.44 | 0.0108 | LSTM 预测下收益下降且仍过载 |
| lstm | ADMM_MPC | 13.39 | 12.58 | 0.07 | 0 | 99.37 | 0.1612 | 安全性明显好，收益较低 |

### 10.4 Compare 关键结果

| forecast_mode | controller | episode_reward_mean | profit_total_eur | voltage_violations | trafo_overload_steps | trafo_max_pct | status |
|---|---|---:|---:|---:|---:|---:|---|
| perfect | MISOCP | 42.30 | 604.49 | 24 | 0 | 97.86 | OK |
| perfect | LOCAL_MPC | 38.86 | 576.41 | 24 | 157 | 170.05 | Profit OK / Safety weak |
| lstm | LOCAL_MPC | 17.48 | 265.17 | 20 | 106 | 170.44 | Forecast degradation |
| lstm | ADMM_MPC | 13.39 | 188.68 | 1 | 0 | 99.37 | Safety OK |
| lstm | MADRL_BASE | 19.43 | 271.40 | 5 | 38 | 138.27 | Better profit than LSTM MPC, worse safety than ADMM |
| lstm | MADRL_PENALTY | 0.42 | 1.46 | 1 | 13 | 149.06 | Safety improves, profit collapses |
| lstm | MADRL_PROJECTION | 18.64 | 274.50 | 0 | 6 | 113.22 | Best MADRL safety/profit balance, but slow training |

## 11. 如果中途需要改代码

只允许为“跑通当前主线”修改代码，不趁机加新功能。

代码修改原则：

```text
1. 不加兼容旧 schema 的分支。
2. 不加 checkpoint/resume/latest/sibling scan。
3. 不新增一层只转发参数的 wrapper。
4. 不加隐藏 fallback；缺依赖、缺文件、缺列让它直接报错。
5. 私有单用 helper 能内联就内联。
6. 资源释放代码可以保留，例如 close()/dispose()。
```

修改前：

```powershell
node tools\gitnexus-shim\gitnexus.mjs impact <symbol> --direction upstream --repo MADRL_ESS
```

修改后：

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m pytest tests
node tools\gitnexus-shim\gitnexus.mjs detect_changes --scope all --repo MADRL_ESS
```

如果改了 notebook 或中文文档，还要跑：

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n MADRL_ESS python -m pytest tests/test_encoding_hygiene.py
```

## 12. 下一步提速建议输出格式

compare 完成后给出下一步建议，按收益优先级排序：

```text
1. 当前并行训练实际 speedup 是多少。
2. 主要瓶颈现在落在 env_step_s、sample_update_s 还是 other_s。
3. 是否值得把 num_envs 从当前值提高到 8/16。
4. pandapower backend benchmark 是否值得做。
5. 是否需要进一步减少 Python/NumPy/Torch 转换。
6. 是否需要把安全 projection 的耗时单独计时。
```

建议必须基于本轮真实表格，不基于体感。若 `num_envs=16` 的收益判断缺数据，先跑 50 episodes benchmark，不直接上 500 episodes。

### 12.1 本轮建议

基于本轮 500 episodes 真实结果，优先级如下：

```text
P1. 单独优化 MADRL_PROJECTION 的 sample/update 路径。
    证据：projection 的 env_step_s=1844.21，和 base 的 1780.23 接近；
    但 sample_update_s=7107.82，是 base 的 3.71x。

P2. 给 safety projection 训练增加细粒度计时。
    当前只知道 action_sample_s 和 sample_update_s 慢；
    下一步要拆成 actor forward、projection、to_torch_obs、critic update、actor update、soft update。

P3. 先做 num_envs=8/16 的 50 episodes benchmark，不直接上 500。
    对 base/penalty，env_step_s 仍占约 45%，提高 num_envs 可能继续有收益；
    对 projection，瓶颈主要不在 env_step，盲目加 num_envs 不一定解决。

P4. 检查 projection notebook 的 sequence_length=49 是否是必要设置。
    projection 当前 cfg 改了 obs.sequence_length=49；
    这可能显著放大 actor/critic 输入和 sample/update 成本。

P5. pandapower backend benchmark 暂时排在 projection profile 之后。
    因为 base/penalty 已经明显提速，而 projection 的主要瓶颈不是潮流。
```

不建议现在优先做：

```text
1. 继续加 shared critic 当作提速主线。
2. 直接把 num_envs 拉到 16 跑完整 500 episodes。
3. 关闭潮流或加 no-safety fast path。
4. 为了跑通而恢复旧 record/cache 分支。
```

## 13. Go / No-Go

Go 条件：

```text
1. 六个 notebook 全部执行成功并写回输出。
2. pytest tests 通过。
3. 三个 MADRL notebook 都真实执行 500 episodes，不使用旧 record。
4. MADRL_BASE s_per_episode <= 12.5，或至少相对 18.58 达到明确 speedup。
5. compare_metrics 中七个主线 scheme 都存在。
6. 安全指标没有出现无法解释的明显退化。
```

本轮判定：

```text
Overall Go for pipeline correctness.
Speed Go for MADRL_BASE and MADRL_PENALTY.
Speed No-Go for MADRL_PROJECTION; needs targeted profiling before继续大规模训练。
Compare Go: 七个主线 scheme 均已进入 compare_metrics/economic_table/safety_table。
```

No-Go 条件：

```text
1. 任一 notebook 失败且需要旧路径或 fallback 才能继续。
2. 任一 MADRL notebook 走了 load_madrl_record 缓存分支。
3. compare 缺少关键 controller 或画出空图。
4. MADRL 并行训练速度没有改善，且 summary 显示仍在走串行环境。
5. MISOCP/MPC 结果缺失 solve/eval/record 关键字段。
6. 需要靠 checkpoint、latest lookup 或旧缓存才能复现实验。
```
