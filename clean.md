# MADRL_ESS 学习与清理手册

这份文档不是一次性重构计划，而是给不熟悉项目的人用的学习路线。目标是每次只理解一个文件或一个 owner，然后做很小、可验证的清理。

你每完成一个文件，都应该能回答四个问题：

- 这个文件拥有什么概念？
- 输入从哪里来，输出交给谁？
- 哪些合同必须严格失败，不能兼容或兜底？
- 我改动后跑了哪些测试，影响范围是否符合预期？

## 0. 工作方式

每一轮只选一个文件。不要同时清理 notebook、controller、env、predictor、plot。

推荐节奏：

```text
1. 读本文件在 clean.md 里的学习目标。
2. 用 GitNexus 查这个文件相关流程。
3. 如果要改函数、类、方法，先跑 impact analysis。
4. 只读这个文件、直接调用点、直接测试。
5. 写下输入、输出、owner、合同。
6. 做最小清理。
7. 跑 targeted tests。
8. 跑 gitnexus detect_changes。
9. 在 clean.md 或提交信息里记录本轮学到了什么。
```

GitNexus 索引如果提示 stale，先运行：

```powershell
gitnexus analyze
```

如果本地只有 npx 入口，再用：

```powershell
npx gitnexus analyze
```

文档、notebook、中文注释改完后，跑编码卫生测试：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_encoding_hygiene.py
```

如果这个解释器在当前机器不存在，先试：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_encoding_hygiene.py
```

## 1. 不要先碰这些

刚开始不要清理这些目录：

- `artifacts/`
- `notebooks/record/`
- `.gitnexus/`
- `.claude/`
- `data/processed/`
- 任何 checkpoint run 目录

这些是运行结果、索引或缓存。它们可以帮助理解当前状态，但不是第一轮清理对象。

也不要一开始删除 notebook。Notebook 里虽然有重复和旧输出，但它们是理解主线入口最直接的线索。先学代码 owner，再瘦身 notebook。

## 2. 每个文件的学习模板

读每个文件时，在草稿里填这个模板。

```text
文件:
owner:
主入口:
直接上游:
直接下游:
核心数据结构:
必须拒绝的旧合同:
可以删除的复杂性:
不能删除的原因:
测试:
```

如果填不出来，不要急着改。先用 GitNexus：

```text
query({query: "文件名或概念", repo: "MADRL_ESS"})
context({name: "函数名或类名", repo: "MADRL_ESS"})
impact({target: "函数名或类名", direction: "upstream", repo: "MADRL_ESS"})
```

## 3. 当前主线地图

现在项目的主线可以压成这几条：

```text
forecast_lstm.ipynb
  -> predictors.training
  -> predictors.shared_data
  -> shared-data manifest

train_base*.ipynb
  -> scripts.utils.grid_notebook_workflow
  -> scripts.mainline_madrl
  -> scripts.builder
  -> envs.grid_env.GridEnv
  -> controllers.madrl.base_agent
  -> checkpoint + rollout record

local_MPC.ipynb
  -> scripts.mainline_compare
  -> controllers.mpc.gurobi_agent_mpc
  -> rollout record

global_MISOCP.ipynb
  -> controllers.mpc.global_socp_mpc
  -> rollout record

ADMM_mpc.ipynb
  -> scripts.utils.admm_mpc_notebook_helpers
  -> scripts.utils.admm_mpc_solver
  -> rollout record

compare.ipynb
  -> scripts.mainline_compare
  -> notebooks/record/*
```

学习时始终问：这个文件在哪条主线上？

## 4. 文件学习顺序

### 4.1 先读入口和配置

| 顺序 | 文件 | 学习目标 | 第一轮清理只做什么 | 推荐测试 |
|---:|---|---|---|---|
| 1 | `README.md` | 建立项目总脑图。只读，不急着改。 | 发现 README 和真实代码不一致时记下来。 | 不需要。 |
| 2 | `configs/experiment_config.py` | 理解所有 dataclass、默认日期、reward、MPC、notebook spec。 | 标注哪些是配置 owner，哪些像 notebook spec。不要拆文件。 | `tests/test_registries.py`, `tests/test_madrl_notebook_defaults.py` |
| 3 | `configs/profiles.py` | 理解 profile 如何修改 config，summary 如何展示当前实验事实。 | 删除重复 summary 字段前先确认 README 是否依赖。 | `tests/test_madrl_notebook_defaults.py` |
| 4 | `scripts/utils/project_paths.py` | 理解项目根目录、artifact 路径、checkpoint 路径。 | 修正绝对路径污染时优先在这里或调用点做精确定位。 | 路径相关 targeted tests。 |

读完这一组，你应该能回答：

```text
当前默认 train_year/test_year 是什么？
测试窗口为什么是 2020-04-01 到 2020-04-15？
normal、perfect、safe、projection_safe 的差异来自哪里？
路径应该从 project_root 推导，还是从旧 JSON 里的绝对路径读取？
```

### 4.2 再读数据和预测

| 顺序 | 文件 | 学习目标 | 第一轮清理只做什么 | 推荐测试 |
|---:|---|---|---|---|
| 5 | `data/loaders/constants.py` | 看数据列名和时间约定。 | 只确认常量是否仍被使用。 | `tests/test_prosumer_dataset.py` |
| 6 | `data/loaders/prosumer.py` | 理解 raw CSV 如何变成 episode、load、PV、price。 | 不改窗口逻辑，先画出 train/test 日期进入点。 | `tests/test_prosumer_dataset.py` |
| 7 | `data/loaders/registry.py` | 理解 dataset owner 如何被构建。 | 如果只是薄入口，保留为稳定边界。 | `tests/test_registries.py` |
| 8 | `predictors/mainline_forecast.py` | 理解 canonical forecast controls。 | 只删明显未引用的旧 forecast control。 | `tests/test_forecast_scaling.py` |
| 9 | `predictors/training.py` | 理解 LSTM artifact 训练、签名、评估。 | 不做大拆分；先找旧 artifact fallback。 | forecast 相关 tests |
| 10 | `predictors/lstm_forecaster.py` | 理解 LSTM artifact 加载和推理。 | 只删除旧 schema 接受逻辑，不能改预测语义。 | `tests/test_lstm_load_per_agent.py` |
| 11 | `predictors/shared_data.py` | 理解 shared-data schema、manifest、runtime date selection。 | 合同错误必须在这里失败，不外移到 notebook。 | `tests/test_madrl_shared_data.py`, `tests/test_builder_vec_env.py` |
| 12 | `predictors/registry.py` | 理解 perfect/lstm forecaster 如何实例化。 | 保留 registry 边界，不塞 artifact 搜索逻辑。 | `tests/test_registries.py` |

读完这一组，你应该能回答：

```text
full-year shared-data 和 0401-0415 runtime selection 是什么关系？
LSTM artifact 缺失时应该报什么错？
哪些旧 schema 应该被拒绝？
```

### 4.3 再读环境和奖励

| 顺序 | 文件 | 学习目标 | 第一轮清理只做什么 | 推荐测试 |
|---:|---|---|---|---|
| 13 | `envs/observation/normalization.py` | 理解 observation normalizer 的 train/test 统计。 | 不改数值口径，只标出 owner。 | `tests/test_observation_normalization.py` |
| 14 | `envs/observation/default_builder.py` | 理解 local features、sequence features 如何组装。 | 删除重复 feature 映射前先确认 shared-data 字段。 | observation tests |
| 15 | `envs/grid/net_builder.py` | 理解 pandapower 网络如何建。 | 不优化网络结构，只确认输入合同。 | grid/env tests |
| 16 | `envs/grid/grid_core.py` | 理解潮流、voltage、line、trafo metrics。 | 不改物理量单位。 | grid/env tests |
| 17 | `envs/grid/deployments.py` | 理解 agent bus/profile 部署。 | 只清理未用 deployment helper。 | grid/env tests |
| 18 | `envs/grid_env.py` | 理解 `reset`、`step`、action 到 battery/PV/net load。 | 改前必须 impact；这是高风险 owner。 | `tests/test_grid_env*.py`, reward/env tests |
| 19 | `envs/rewards/NormalReward.py` | 理解 reward component 和旧 reward key 拒绝逻辑。 | 旧字段只允许报错，不允许兼容。 | `tests/test_normal_reward.py` |

读完这一组，你应该能回答：

```text
action[0] 正负分别代表什么？
PV action 如何转成 curtailment？
reward 用真实价格还是归一化价格？
grid safety penalty 在哪里算？
```

### 4.4 再读模型和 MADRL

| 顺序 | 文件 | 学习目标 | 第一轮清理只做什么 | 推荐测试 |
|---:|---|---|---|---|
| 20 | `models/assembly.py` | 理解 actor/critic 输入输出、MLP family、centralized critic。 | 不新增 model family。 | `tests/test_model_assembly.py` |
| 21 | `controllers/madrl/safety_projector.py` | 理解 SoC-aware mapping、local guard、projection。 | 改前必须 impact；先写清 rollout/target-Q/actor-loss 三处语义。 | MADRL action mapping tests |
| 22 | `controllers/madrl/base_agent.py` | 理解 MADDPG/MATD3 更新、target-Q、actor-loss。 | 不先拆文件；先删重复旧分支。 | `tests/test_matd3*.py`, train tests |
| 23 | `controllers/madrl_controller.py` | 理解 notebook rollout 如何加载 agent 并 act。 | 只修路径和 checkpoint 精确定位。 | rollout/controller tests |
| 24 | `controllers/madrl/__init__.py` | 理解 public imports。 | 只保留真正需要的导出。 | import tests |

读完这一组，你应该能回答：

```text
actor raw action 和进入 env 的 action 是否相同？
replay 里存的是 raw action 还是 mapped action？
target-Q 和 actor-loss 是否使用同一套 action mapping？
MATD3_SAFE_POC 比 MATD3 多了什么？
```

### 4.5 再读训练和 checkpoint 主线

| 顺序 | 文件 | 学习目标 | 第一轮清理只做什么 | 推荐测试 |
|---:|---|---|---|---|
| 25 | `scripts/builder.py` | 理解 config 如何装配 dataset、forecaster、obs_builder、env、runner。 | 改前必须 impact；不要把业务逻辑塞进 builder。 | `tests/test_builder_vec_env.py` |
| 26 | `scripts/train.py` | 理解 TrainRunner、replay、update schedule、reward summary。 | 只清理无 owner 的统计重复。 | training tests |
| 27 | `scripts/checkpoints.py` | 理解 checkpoint path、label、manifest。 | 删除 latest/sibling fallback 要非常谨慎，先确认 notebook 入口。 | `tests/test_checkpoints.py` |
| 28 | `scripts/mainline_madrl.py` | 理解外部训练进程如何读取 controls 并写 result。 | 让它保持 launcher，不承载算法逻辑。 | `tests/test_train_mainline_launcher.py` |

读完这一组，你应该能回答：

```text
force_retrain_madrl=False 时加载哪里？
train_result.json 里哪些路径不能相信为跨机器绝对路径？
reward_summary_path 应该如何从 result_json_path 精确推导？
```

建议第一个小修复练习：

```text
在 notebook 读取 train_reward_summary.json 时，
不要相信旧机器写入 train_result["reward_summary_path"] 的绝对路径。
优先用 Path(result_json_path).with_name("train_reward_summary.json")。
```

这类修复边界小，适合熟悉 notebook 和 checkpoint 元数据。

### 4.6 再读 notebook workflow 和图表

| 顺序 | 文件 | 学习目标 | 第一轮清理只做什么 | 推荐测试 |
|---:|---|---|---|---|
| 29 | `scripts/utils/grid_notebook_workflow.py` | 理解 notebook 编排、rollout record、save/load、plot wrapper。 | 防止它继续变成杂物桶；能回 owner 的逻辑回 owner。 | `tests/test_grid_notebook_workflow.py` |
| 30 | `scripts/plots/grid_notebook_plotting.py` | 理解图表 owner。 | 只保留 notebook 真正使用的图。 | plot tests |
| 31 | `scripts/mainline_compare.py` | 理解 compare metric、bundle validation、MADRL/MPC 对齐。 | 不允许 fuzzy result discovery。 | compare tests |
| 32 | `scripts/compare/storage_profit_recompute.py` | 判断是正式 compare owner 还是一次性脚本。 | 若只被一次性使用，列为删除候选。 | compare tests |

读完这一组，你应该能回答：

```text
rollout record 由哪些 parquet/json 组成？
compare metric 的统一口径在哪里？
旧结果 bundle 不匹配时在哪里失败？
```

### 4.7 最后读 MPC

| 顺序 | 文件 | 学习目标 | 第一轮清理只做什么 | 推荐测试 |
|---:|---|---|---|---|
| 33 | `controllers/mpc/gurobi_agent_mpc.py` | 理解 local MPC 每个 agent 如何求解。 | 不改优化模型，只标注输入输出。 | MPC tests |
| 34 | `controllers/mpc/global_socp_mpc.py` | 理解 global MISOCP/oracle 的网络约束。 | 这是高风险文件，第一轮只读。 | Gurobi/license 条件测试 |
| 35 | `scripts/utils/admm_mpc_solver.py` | 理解 ADMM 子问题、consensus、residual。 | 不先优化性能，先确认数学合同。 | ADMM tests |
| 36 | `scripts/utils/admm_mpc_notebook_helpers.py` | 理解 ADMM notebook controller 和 diagnostics。 | notebook glue 保持薄，solver 逻辑回 solver。 | `tests/test_admm_mpc_notebook_helpers.py` |

读完这一组，你应该能回答：

```text
Local MPC、Global MISOCP、ADMM MPC 的目标函数有什么差异？
它们和 MADRL 共用了哪些 env/record 口径？
哪些 solver 失败应该显式报错？
```

### 4.8 最后处理 notebooks

Notebook 清理放到最后，因为你需要先知道逻辑应该属于哪个 Python owner。

| 顺序 | Notebook | 学习目标 | 第一轮清理只做什么 |
|---:|---|---|---|
| 37 | `notebooks/forecast/forecast_lstm.ipynb` | 生成 forecast artifacts 和 shared-data。 | cell 只保留参数、运行、展示。 |
| 38 | `notebooks/madrl/train_base.ipynb` | normal MADRL 主训练入口。 | 去掉旧输出、跨机器绝对路径。 |
| 39 | `notebooks/madrl/train_base perfect.ipynb` | perfect 上限训练入口。 | 后续考虑改名，但第一轮不做大迁移。 |
| 40 | `notebooks/madrl/train_base_safe.ipynb` | safety penalty 训练入口。 | 判断是否仍是当前主线。 |
| 41 | `notebooks/madrl/train_projection_safe.ipynb` | projection safe POC 训练入口。 | 修复路径读取，确认是否还要保留。 |
| 42 | `notebooks/madrl/local_MPC.ipynb` | local MPC rollout。 | 不写 solver 逻辑。 |
| 43 | `notebooks/madrl/global_MISOCP.ipynb` | global oracle rollout。 | 不写 solver 逻辑。 |
| 44 | `notebooks/madrl/ADMM_mpc.ipynb` | ADMM MPC rollout。 | 确认默认测试窗口和 record 输出。 |
| 45 | `notebooks/madrl/compare.ipynb` | 汇总所有 record。 | 不做 fuzzy result search。 |

Notebook 最终目标形态：

```text
cell 1: 参数和开关
cell 2: 构造 cfg 并展示 summary
cell 3: 调用 Python owner 执行
cell 4: 读取 record 并画图
```

## 5. 清理判断规则

### 5.1 保留

保留一个函数、文件或 notebook cell，必须满足至少一个条件：

- 是当前主线 fresh kernel 必经路径；
- 是稳定 public boundary；
- 是明确 owner 的核心逻辑；
- 是合同校验的第一边界；
- 是测试 helper；
- 是 README 或论文结果复现明确需要的记录入口。

### 5.2 删除

默认删除这些复杂性：

- old key 到 new key 的静默兼容；
- schema 旧版本自动升级；
- 找不到精确路径后扫描 sibling；
- 找不到指定 checkpoint 后自动找 latest；
- notebook 里复制 Python owner 逻辑；
- 单调用点且不隔离边界的 wrapper；
- 一次性诊断脚本；
- 只为“以后可能用”存在的开关；
- 已经不在 README 主线里的实验入口。

### 5.3 暂停

出现这些情况先暂停，不继续删：

- GitNexus impact 是 HIGH 或 CRITICAL；
- d=1 调用点没有全部理解；
- targeted tests 缺失；
- notebook fresh kernel 可能跑不通；
- 你发现当前结果只靠 fallback 才能加载；
- 某个文件正在变成新的 `utils` 杂物桶。

## 6. 每轮清理的验证

每轮最少做三件事：

```powershell
git status --short
```

然后跑对应 targeted tests。示例：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_builder_vec_env.py
```

最后跑 GitNexus 变更分析：

```text
detect_changes({scope: "all", repo: "MADRL_ESS"})
```

如果改了 notebook 或文档，再跑：

```powershell
conda run -n MADRL_ESS python -m pytest tests/test_encoding_hygiene.py
```

提交信息建议写成：

```text
clean shared-data runtime date selection
clean reward contract boundary
clean notebook reward summary path resolution
```

不要写：

```text
clean misc
refactor utils
remove old stuff
```

## 7. 第一周建议路线

如果你现在完全不熟，第一周不要追求删很多代码。按下面做：

| 天 | 文件 | 目标 |
|---:|---|---|
| 1 | `README.md`, `configs/experiment_config.py` | 知道当前实验到底跑什么。 |
| 2 | `data/loaders/prosumer.py`, `predictors/shared_data.py` | 知道数据、日期窗口、shared-data 如何进入 env。 |
| 3 | `envs/grid_env.py`, `envs/rewards/NormalReward.py` | 知道 action、SoC、reward、grid metric。 |
| 4 | `controllers/madrl/safety_projector.py`, `controllers/madrl/base_agent.py` | 知道 actor raw action 如何变成可行动作。 |
| 5 | `scripts/builder.py`, `scripts/train.py` | 知道训练循环如何串起来。 |
| 6 | `scripts/utils/grid_notebook_workflow.py`, 一个 MADRL notebook | 知道 rollout record 如何保存。 |
| 7 | `ADMM_mpc.ipynb`, `local_MPC.ipynb`, `compare.ipynb` | 知道 MPC 和 compare 如何对齐。 |

每一天只允许做一个小清理：

- 修一个跨机器路径；
- 删除一个未使用 helper；
- 移除一个旧字段 fallback；
- 更新一个错误信息；
- 给一个 owner 增加 targeted test。

## 8. 当前优先清理候选

这些不是立即删除清单，而是读到对应 owner 后重点检查：

| 候选 | 为什么可疑 | 处理方式 |
|---|---|---|
| notebook 里旧电脑绝对路径 | 跨机器不可复现。 | 从 `project_root` 或 `result_json_path` 推导。 |
| `notebooks/record/*` 改动 | 运行产物，不是主线源码。 | 不手工清理，必要时重新生成。 |
| `scripts/diagnostics/madrl_price_diagnostics.py` | 可能是一次性诊断。 | 若仍需要，归入 diagnostics owner；否则删除。 |
| `scripts/compare/storage_profit_recompute.py` | 可能是一次性修复脚本。 | 若 compare 仍需要，迁入 compare owner。 |
| `train_base perfect.ipynb` 文件名 | 空格不利于脚本和文档。 | 后续单独 rename，不能顺手改。 |
| `grid_notebook_workflow.py` 继续膨胀 | notebook glue 容易变杂物桶。 | 核心逻辑回 predictor/env/controller/compare owner。 |

## 9. 完成一个文件的标准

一个文件算“学完并清理过”，必须满足：

- 你能用两句话说明 owner；
- 你知道它的直接上游和下游；
- 你知道最关键的合同失败点；
- 你没有引入兼容兜底；
- 你没有顺手改生成结果；
- targeted tests 已跑或已说明为什么不能跑；
- `detect_changes` 没有显示意外流程；
- 如果有中文或 notebook 改动，编码卫生测试已跑。

## 10. 最终目标

清理完成后，项目应该能用唯一主线解释：

```text
数据从 raw CSV 来；
forecast/shared-data 生成 observation sequence；
GridEnv 负责 step 和物理量；
NormalReward 负责 reward；
MADRL actor 输出 raw action；
action mapping 把 raw action 变成当前 SoC 下可行动作；
TrainRunner 负责 replay 和 update；
MPC controllers 作为对比基线；
notebook 只负责配置、运行、展示和保存 record；
compare 只读取精确 record。
```

如果某段代码不能放进这条解释，也没有测试保护，就进入删除候选。
