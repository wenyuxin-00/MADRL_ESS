# MADRL_ESS 清理计划

## 目标

把当前代码库整理成一个清晰、干净、可重新学习的研究工程。

核心原则是：**只保留当前主线，不保留历史迁移层**。

这次清理不追求“为了少几行而少几行”，而是追求：

- 一个概念只有一个 owner；
- 一个入口只走一条主线；
- 一个 artifact 只通过一个精确 locator 找到；
- 一个合同错误只在第一个边界失败；
- notebook 只负责配置、运行和展示；
- 算法、数据、奖励、环境、checkpoint 逻辑都回到 Python owner；
- 不写兼容；
- 不写兜底；
- 不写套壳；
- 不保留旧字段别名；
- 不用大段 `if/else` 同时照顾旧方案和新方案。

## 当前代码规模

统计口径：

- 排除 `tests/`；
- 排除 `.conda/`、`artifacts/`、`data/`、`node_modules/`、Git 元数据和缓存；
- notebook 只统计 code cell 行数，不统计 `.ipynb` 的 JSON 外壳。

| 范围 | 物理行数 | 非空行数 |
|---|---:|---:|
| Python 源码 | 5,443 | 5,205 |
| Notebook code cells | 2,022 | 1,836 |
| 小脚本和工具 | 301 | 258 |
| **非测试总代码** | **7,766** | **7,299** |

当前最大的非测试文件：

| 文件 | 行数 | 判断 |
|---|---:|---|
| `controllers/madrl/base_agent.py` | 508 | MADRL 核心 owner，可以清理重复 update 分支，但不能粗暴拆散。 |
| `notebooks/forecast/forecast_lstm.ipynb` | 508 | notebook 里逻辑过多，应瘦身。 |
| `configs/experiment_config.py` | 452 | dataclass、常量、notebook spec 混在一起。 |
| `notebooks/forecast/forecast_test.ipynb` | 387 | 很可能是非主线 notebook。 |
| `predictors/training.py` | 383 | forecast training owner，但模式和分支偏多。 |
| `scripts/utils/grid_notebook_workflow.py` | 303 | notebook glue 有变成杂物桶的风险。 |
| `scripts/diagnostics/madrl_price_diagnostics.py` | 284 | 诊断价值有，但不应长期留在主线代码里。 |
| `notebooks/madrl/train_base.ipynb` | 275 | 与 perfect notebook 重复。 |
| `notebooks/madrl/train_base perfect.ipynb` | 275 | 与 normal notebook 重复，且文件名需要规范。 |

## 能缩到多少行

如果保留当前有价值的完整研究能力：

- LSTM forecast shared data；
- MADRL normal 训练；
- MADRL perfect 上限训练；
- local MPC；
- ADMM MPC；
- global oracle / MISOCP；
- checkpoint 读写；
- compare notebook；
- 必要的图表和结果记录；

那么我认为合理目标是：

| 清理强度 | 非测试物理行数 | 含义 |
|---|---:|---|
| 保守清理 | 6,200-6,700 | 删除明显重复 notebook 和套壳，主体结构不大动。 |
| 强清理 | 4,800-5,400 | notebook 变薄，主线唯一，删除旧分支和临时诊断。 |
| 激进但保留完整功能 | **4,200-4,800** | 一个 owner 一个概念，所有非主线路径删除。 |
| MADRL-only 极简版 | 2,800-3,400 | 删除 MPC/oracle/ADMM 对比，不适合当前研究目标。 |

我的建议估算：

```text
当前非测试代码：       7,766 行
推荐清理目标：         约 4,500 行
合理目标区间：         4,200-5,400 行
预计减少：             2,300-3,500 行
预计减少比例：         30%-45%
```

如果还要保留完整 MPC/MADRL/forecast/compare 研究闭环，**4,000 行左右就是比较硬的下限**。再往下压，通常不是清理，而是在删除研究能力。

## 清理后的主线

只承认下面这些主线。

```text
notebooks/forecast/forecast_lstm.ipynb
  -> predictors.training
  -> predictors.shared_data
  -> artifacts/training/shared_data/mainline/<signature>

notebooks/madrl/train_base.ipynb
  -> scripts.utils.grid_notebook_workflow
  -> scripts.mainline_madrl
  -> scripts.builder
  -> envs.grid_env.GridEnv
  -> controllers.madrl.base_agent
  -> scripts.checkpoints

notebooks/madrl/train_base_perfect.ipynb
  -> same path as train_base
  -> prediction_mode = perfect

notebooks/madrl/compare.ipynb
  -> scripts.mainline_compare
  -> controllers.mpc.*
  -> scripts.utils.grid_notebook_workflow rollout records
```

不在这些主线里的东西，要么是测试，要么要明确说明为什么还存在。否则删除。

## 裁决规则

遇到任何文件、函数、notebook cell、配置字段，按下面顺序裁决。

### 1. 是否在当前主线

只问一个问题：

```text
fresh kernel 从 canonical notebook 跑到结果，是否必经这里？
```

如果答案是否：

- 测试保留；
- 文档按需要保留；
- 一次性诊断删除；
- 旧实验入口删除；
- 未来可能用到的代码删除；
- 只有论文复现实验需要的代码，必须写入 README 的主线或 comparison 表。

### 2. 是否拥有完整概念

一个 owner 必须拥有一个概念的完整生命周期。

坏味道：

```text
函数 A 只检查参数，然后调用函数 B；
函数 B 又转发到函数 C；
函数 C 才做真正工作。
```

处理方式：

- 如果 A 是 CLI/notebook 边界，允许保留为薄入口；
- 如果 A 只是套壳，删除；
- 如果 B 是稳定协议边界，保留；
- 否则把逻辑移回真正 owner。

### 3. 是否在替旧合同兜底

任何代码只要做这些事，就默认删除：

- old key -> new key；
- schema 6/7 -> schema 8；
- 找不到精确路径后扫描 sibling；
- 找不到指定 checkpoint 后找 latest；
- 旧 notebook 名称映射到新 notebook；
- 旧 artifact layout 映射到新 artifact layout。

唯一例外：错误信息可以提到旧对象，但不能接受旧对象。

## Owner 边界

| Owner | 负责 | 不负责 |
|---|---|---|
| `configs/experiment_config.py` | dataclass 和基础默认值 | notebook spec、artifact 查找、迁移逻辑 |
| `configs/experiment_specs.py` | canonical notebook/experiment specs | dataclass 定义 |
| `configs/profiles.py` | profile transform 和 summary | 运行主线逻辑 |
| `predictors/training.py` | 训练和校验 LSTM artifact | shared-data 写入、notebook 展示 |
| `predictors/shared_data.py` | 生成和校验 MADRL shared data | 训练 LSTM、接受旧 schema |
| `predictors/registry.py` | 实例化 forecaster | 搜索旧 artifact 布局 |
| `envs/grid_env.py` | env step、状态推进、reward 调用 | notebook 记录格式 |
| `envs/observation/*` | observation 构造和归一化 | forecast artifact 加载 |
| `envs/rewards/NormalReward.py` | reward 数学和 component 合同 | 训练循环 |
| `controllers/madrl/base_agent.py` | actor/critic update 语义 | notebook rollout 格式 |
| `controllers/madrl/safety_projector.py` | action projection 和 feasibility | reward、checkpoint、notebook |
| `controllers/mpc/*` | MPC 求解和动作组装 | MADRL checkpoint |
| `scripts/train.py` | 训练 runner | artifact schema 迁移 |
| `scripts/checkpoints.py` | checkpoint manifest 读写 | 模糊搜索旧 checkpoint |
| `scripts/mainline_*.py` | 精确解析 controls 并调用 owner | 业务逻辑 |
| `scripts/utils/grid_notebook_workflow.py` | notebook 编排 glue | 子系统核心逻辑 |

## 必须删除的复杂性

优先删除这些东西：

1. 旧字段别名。
2. 旧 schema 接受逻辑。
3. artifact fallback 搜索。
4. sibling/latest-compatible 查找。
5. 同一个 notebook 逻辑复制到多个文件。
6. 只包一层函数名、不缩小边界的 wrapper。
7. 一次性诊断脚本。
8. 旧实验路径。
9. 未被当前 README 或 notebook 调用的入口。
10. 为了“也许以后用”保留的配置开关。

## 不能为了少行数删除的东西

这些是研究闭环的一部分，不能为了好看直接删：

- `controllers/mpc/global_socp_mpc.py`：如果还要 global oracle 上限；
- `controllers/mpc/gurobi_agent_mpc.py`：如果还要 local MPC；
- `scripts/utils/admm_mpc_solver.py`：如果还要 ADMM MPC；
- reward component key：结果记录和画图依赖它们；
- checkpoint contract：用来拒绝旧结果；
- shared-data schema contract：用来拒绝旧缓存；
- tests：不计入目标行数，但必须保留。

## 执行协议

每一轮清理都按这个节奏做，避免“大扫除式失控”。

```text
1. 选一个 owner 或一个 notebook，不跨多个概念。
2. 用 GitNexus 查主线和 impact。
3. 如果 impact 是 HIGH/CRITICAL，先写明影响范围，再动代码。
4. 删除旧路径，不写兼容路径。
5. 更新直接调用点。
6. 跑该 owner 的 targeted tests。
7. 跑最小 notebook/CLI smoke。
8. 跑 gitnexus detect_changes。
9. 记录本轮减少了什么复杂性，而不是只记录减少了多少行。
```

每轮清理的提交粒度：

```text
一个 commit = 一个 owner 的结构变化 + 对应调用点 + 对应测试
```

不要做这种提交：

```text
clean miscellaneous files
refactor utils
remove old stuff
```

这些名字说明改动边界已经失控。

## 分阶段计划

### Phase 0：存档当前结果

在动结构前，先保留当前可运行状态：

- commit 当前代码；
- 记录 normal/perfect checkpoint 目录；
- 记录 shared data signature；
- 记录当前 README 里的主线说明；
- 记录当前可复现的训练设置。

验收：

- 当前结果可以通过明确 checkpoint 路径重新加载；
- 不依赖“latest”或模糊目录搜索。

### Phase 1：冻结合同

先把当前真正有效的合同写清楚：

- forecast artifact format；
- shared-data schema version；
- price observation contract；
- MADRL training contract；
- checkpoint manifest fields；
- reward component keys；
- notebook spec fields；
- comparison result fields。

验收：

- 旧字段在第一个 owner 边界失败；
- 错误信息说明旧对象、新合同、要重跑哪个 notebook；
- 没有第二路径查找。

### Phase 2：瘦身 notebook

目标 notebook 形态：

```text
cell 1: 参数
cell 2: 构建 config，展示 summary
cell 3: 调用 owner 执行
cell 4: 读取结果并画图
```

动作：

- `train_base perfect.ipynb` 改名为 `train_base_perfect.ipynb`；
- normal 和 perfect notebook 只差 `NOTEBOOK_KEY`；
- 删除重复 setup cell；
- forecast notebook 不再定义大量滚动预测 helper；
- `forecast_test.ipynb` 若不是主线，删除。

预计减少：**900-1,300 行**。

### Phase 3：拆分 config 和 spec

目标：

```text
configs/experiment_config.py   dataclass only
configs/experiment_specs.py    notebook specs and scheme specs
configs/profiles.py            profile transforms and summaries
```

动作：

- dataclass 留在 `experiment_config.py`；
- notebook spec 移到 `experiment_specs.py`；
- 删除旧 import 兼容；
- 更新所有调用点。

预计减少：**100-250 行**，更重要的是降低认知负担。

### Phase 4：清理 forecast owner

动作：

- 只保留 managed multi-signal LSTM artifact；
- 删除非主线 artifact mode；
- 保留一个精确 artifact path convention；
- forecast evaluation 只保留当前 notebook 真正展示的部分；
- shared-data 只接受当前 schema；
- 派生价格 observation 到 `wholesale_price` 的依赖规则保持显式。

预计减少：**500-800 行**。

### Phase 5：清理 MADRL action/update 路径

只保留当前语义：

```text
rollout:
actor raw -> SoC-aware mapping -> residual guard -> env step/replay

target-Q:
target actor raw -> SoC-aware mapping -> residual guard -> critic target

actor-loss:
policy raw -> SoC-aware mapping -> residual guard -> critic policy loss
```

动作：

- 删除旧 action transform；
- 如果 MADDPG 不再比较，删除或降级为非主线；
- SafePOC 若不再跑，删除对应 notebook 和入口；
- 合并 MATD3/MADDPG 中重复的 helper；
- 保留 NaN/gradient health check，但删除只包装 PyTorch 报错的防御层。

预计减少：**300-600 行**。

### Phase 6：清理诊断和一次性脚本

候选：

- `scripts/diagnostics/madrl_price_diagnostics.py`；
- `scripts/compare/storage_profit_recompute.py`；
- 旧 notebook plots；
- 临时价格归一化验证脚本。

规则：

- 如果还要保留，必须接入明确 owner 和 manifest；
- 如果只是一次性分析，删除；
- 不保留“以后可能有用”的脚本。

预计减少：**250-500 行**。

### Phase 7：让 scripts 变薄

目标：

```text
scripts/mainline_*.py
  -> parse exact controls
  -> call owner
  -> write manifest/result
```

动作：

- `mainline_compare.py` 的业务逻辑回到 comparison owner；
- `mainline_madrl.py` 只做 launcher；
- `scripts/utils/` 里单调用点且不隔离边界的 helper 直接内联或移回 owner。

预计减少：**300-500 行**。

## 删除候选清单

| 候选 | 初步动作 |
|---|---|
| `notebooks/forecast/forecast_test.ipynb` | 非主线则删除。 |
| `notebooks/madrl/train_base_safe.ipynb` | safety penalty 不再作为当前实验则删除。 |
| `notebooks/madrl/train_projection_safe.ipynb` | SafePOC 不再跑则删除。 |
| `scripts/diagnostics/madrl_price_diagnostics.py` | 价格诊断完成后删除，或并入正式 report owner。 |
| `scripts/compare/storage_profit_recompute.py` | 并入 compare owner 或删除。 |
| `tools/gitnexus-shim/*` | 只有 GitNexus 运行仍需要时保留。 |
| 空 `__init__.py` | 只在包导入需要时保留。 |

## 验收标准

每个阶段结束都必须满足：

- canonical notebook 能 fresh kernel 运行；
- normal 和 perfect 保存到不同精确目录；
- shared-data 只通过 signature 复用；
- checkpoint loader 明确拒绝旧合同；
- compare notebook 能加载当前结果；
- targeted tests 通过；
- `gitnexus detect_changes --scope all --json --repo MADRL_ESS` 只显示预期影响。

## 停止条件

出现下面任一情况，就停止继续删代码，先重新审阅：

- normal 或 perfect notebook 不能 fresh kernel 跑通；
- shared-data 不能通过当前 schema 生成或复用；
- checkpoint loader 需要 fallback 才能找到当前结果；
- compare notebook 需要手工改路径才能读结果；
- GitNexus impact 显示 HIGH/CRITICAL，但没有明确更新所有 d=1 调用点；
- 某个 owner 变成新的 `utils` 杂物桶；
- 行数下降了，但 README 调用链更难解释了。

## 清理后的 README 要求

清理完成后，README 必须能用树状调用链说明：

```text
forecast_lstm
train_base normal
train_base perfect
local MPC
ADMM MPC
global oracle
compare
checkpoint load
```

README 不需要解释历史方案，只解释当前主线。

如果某段代码不能出现在 README 的主线树里，也没有被测试明确保护，它就不应该留在非测试代码里。

## 建议执行顺序

1. commit 当前可运行版本。
2. 冻结合同和主线说明。
3. notebook 改名和瘦身。
4. 删除非主线 notebook。
5. 拆 config/spec。
6. 清 forecast owner。
7. 清 MADRL action/update owner。
8. 清诊断和一次性脚本。
9. 清 scripts/utils。
10. 跑完整 normal/perfect/compare smoke workflow。
11. commit 为 `clean canonical workflow`。

## 最终判断

我认为这个项目最合理的清理目标是：

```text
非测试代码从 7,766 行
降到约 4,500 行
保守目标 5,400 行以内
激进目标 4,800 行以内
完整功能硬下限约 4,000 行
```

真正的收益不是行数本身，而是你重新看代码时能直接回答：

```text
数据从哪里来？
预测在哪里生成？
环境在哪里 step？
奖励在哪里算？
actor 动作在哪里变成可行动作？
replay 存的是什么？
checkpoint 合同在哪里验证？
compare 读取的是哪一份结果？
```

如果这些问题都能沿着唯一主线回答，代码库就算清理成功。
