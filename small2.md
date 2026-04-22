# MADRL_ESS 第二轮极限收口计划（5000 行冲刺版，第四次收束后修订版，2026-04-22）

## 说明

- 本版**不改变终局原则，不改变验收指标，不改变六条主线与 notebook 保留清单**。
- 本版只做两件事：
  - 按第四次收束后的真实仓库快照刷新 `small2.md`。
  - 把“下一阶段”从旧版里已经过期的 `N2`/旧 `N3` 战场，改成当前仍然存在且仍然超重的 owner。
- 相比 `2026-04-21` 版本，当前仓库已经额外完成：
  - `CURRENT` 从 `10651` 降到 `8526`，新增净减 `-2125` 行。
  - `PY_FILE_COUNT` 从 `79` 降到 `57`，已经提前压到 `N3` / `N4` 的文件数硬指标之内，并逼近 `N5`。
  - `TESTS_LINES` 从 `9875` 降到 `8911`，已经提前压到本版统一执行线 `<= 8911`。
  - `controllers/action_feasibility.py`、`controllers/madrl/{maddpg.py, matd3.py, matd3_safe_poc.py, registry.py}`、`envs/vec_env.py`、`envs/parallel_episode_sampling.py`、`scripts/mainline_mpc.py`、`scripts/mainline_grid_analysis.py`、`scripts/mainline_artifacts.py`、`scripts/utils/misocp_notebook_helpers.py` 等旧壳层和分裂 support 已经收掉。
- 当前真正卡住的不是文件数，也不是 tests，而是 `predictors/`、`scripts/` 这两个大桶，以及 `predictors/training.py`、`scripts/mainline_compare.py`、`controllers/mpc/global_socp_mpc.py` 这类残余大 owner。
- 如果后续执行中证明 `5000` 在不破坏代码纯洁性的前提下不可达，**仍然不能自行降格**，必须停下来单独上报，由用户决策。

## 当前严格快照

- 统计口径：
  - 只统计 `configs/`、`controllers/`、`data/`、`envs/`、`models/`、`predictors/`、`scripts/` 下当前实际存在的 `.py` 文件。
  - `tests/` 不计入总行数目标，但要单独跟随维护。
  - 工作区中的未跟踪 `.py` 文件也算数。
  - 不统计 `.conda/`、第三方依赖目录、notebook 本体。
- 当前基线：
  - `BASELINE=33323`
  - `CURRENT=8526`
  - `DELTA=-24797`
  - `PY_FILE_COUNT=57`
  - `TESTS_LINES=8911`
  - 距离 `5000` 目标还差 `3526` 行
  - `PY_FILE_COUNT` 已提前满足 `N3` / `N4` 的文件数要求，后续阶段不得反弹。
  - `TESTS_LINES` 已提前满足本版统一执行线 `<= 8911`，后续阶段不得反弹。

### 一级目录现状

| 目录 | 当前行数 | 当前文件数 | 终局预算 |
| ---- | -------: | ---------: | -------: |
| `configs/` | 381 | 2 | <= 240 |
| `controllers/` | 1908 | 7 | <= 1300 |
| `data/` | 338 | 5 | <= 190 |
| `envs/` | 1193 | 10 | <= 860 |
| `models/` | 242 | 6 | <= 140 |
| `predictors/` | 1713 | 11 | <= 900 |
| `scripts/` | 2751 | 16 | <= 1370 |

- 以上终局预算合计正好是 `5000`。

### 当前最重且最该优先收口的文件

| 文件 | 当前行数 | 阶段收口参考 |
| ---- | -------: | -------: |
| `predictors/training.py` | 842 | <= 680 |
| `controllers/mpc/global_socp_mpc.py` | 795 | <= 760 |
| `scripts/mainline_compare.py` | 593 | <= 420 |
| `controllers/madrl/safety_projector.py` | 427 | <= 360 |
| `predictors/lstm_forecaster.py` | 384 | <= 300 |
| `envs/subproc_vec_env.py` | 305 | <= 240 |
| `controllers/mpc/gurobi_agent_mpc.py` | 292 | <= 260 |
| `scripts/train.py` | 292 | <= 220 |
| `scripts/mainline_madrl.py` | 291 | <= 220 |
| `scripts/utils/grid_notebook_workflow.py` | 288 | <= 260 |
| `envs/grid_env.py` | 287 | <= 250 |
| `scripts/utils/admm_mpc_solver.py` | 285 | <= 260 |
| `controllers/madrl/base_agent.py` | 280 | <= 220 |
| `scripts/plots/grid_notebook_plotting.py` | 274 | <= 240 |
| `data/loaders/prosumer.py` | 272 | <= 150 |
| `predictors/shared_data.py` | 264 | <= 200 |
| `configs/experiment_config.py` | 263 | <= 220 |

- 上表“阶段收口参考”只是为了排优先级而给出的收口值，不强制等同于最近 checkpoint；**真正执行时仍以各阶段的 checkpoint 与硬要求为准**。

### 当前最碎的区域

| 区域 | 当前文件数 | 当前行数 | 终局方向 |
| ---- | ---------: | -------: | ---- |
| `scripts/` 顶层（`__init__.py` + `mainline_compare.py` / `mainline_madrl.py` / `train.py` / `builder.py` / `checkpoints.py`） | 6 | 1441 | 新 `N1` 主战场，先压到 <= 1100，再继续打到终局 <= 220 |
| `scripts/utils/` | 9 | 1036 | 新 `N1` 共战场，先压到 <= 900，再继续打到终局 <= 850 |
| `scripts/plots/` | 1 | 274 | 新 `N1` 顺带收口，保持单 owner，不允许重新长出子树 |
| `predictors/` | 11 | 1713 | 新 `N2` 主战场，先压到 <= 1250，再继续打到终局 <= 900 |
| `controllers/mpc/` | 3 | 1087 | 新 `N3` 主战场，保留主解法，先压到 <= 900 |
| `controllers/madrl/` | 2 | 707 | 新 `N3` 共战场，先压到 <= 520，再继续打到终局 <= 340 |
| `envs/vec_stack` | 1 | 305 | 新 `N3` 必须补齐到 <= 240 |
| `envs/observation/` | 2 | 270 | 新 `N3` 补齐到 <= 220，终局 <= 200 |
| `envs/grid/` | 5 | 249 | 新 `N3` 补到 <= 180，终局 <= 120 |
| `models/` | 6 | 242 | 新 `N4` 主战场，压到 <= 3 个文件、<= 140 行 |

## 最终目标

### 终局目标

- `tests/` 之外全部 `.py` 文件总行数压到 **`<= 5000`**。
- `tests/` 之外 `.py` 文件总数压到 **`<= 55`**，冲刺目标 **`<= 50`**。
- notebook 继续全部保留，但只能作为薄入口与展示层，不再承载主控制流。
- 目录结构继续固定在：
  - `configs/`
  - `controllers/`
  - `data/`
  - `envs/`
  - `models/`
  - `predictors/`
  - `scripts/`
  - `tests/`

### 六条主线必须保留

- `MISOCP`
- `local MPC`
- `ADMM MPC`
- `MADRL base`
- `MADRL base safe`
- `MADRL projection safe`

### notebook 必须保留

- `notebooks/forecast/forecast_lstm.ipynb`
- `notebooks/madrl/global_MISOCP.ipynb`
- `notebooks/madrl/local_MPC.ipynb`
- `notebooks/madrl/ADMM_mpc.ipynb`
- `notebooks/madrl/train_base.ipynb`
- `notebooks/madrl/train_base_safe.ipynb`
- `notebooks/madrl/train_projection_safe.ipynb`
- `notebooks/madrl/compare.ipynb`
- `notebooks/madrl/grid_network_analysis.ipynb`

## 不变且更严格的硬原则

### 1. 单场景优先

- 项目只服务一套固定主线场景。
- 唯一测试窗口固定为 `2020-06-01` 到 `2020-06-07`。
- 六条主线共用同一套 agent、reward、forecast、battery、grid、MPC 参数。
- 所有共享参数只允许集中存放在 `configs/experiment_config.py`。

### 2. 不兼容，不兜底，保证代码纯洁性

- 命中旧 schema、旧 key、旧缓存包、旧 artifact 命名时：
  - 不做 fallback。
  - 不做 sibling 扫描。
  - 不做 latest-compatible 搜索。
  - 不做 prefix 模糊匹配。
  - 不保留 deprecated alias。
  - 不保留 bridge adapter。
  - 不保留旧新双路径并存的过渡逻辑。
- 唯一允许的处理：
  1. 直接失败。
  2. 明确指出命中的旧对象。
  3. 明确提示应该重跑哪个主入口或 notebook。

### 3. 不允许用投机取巧的方式刷减量

- **不允许通过缩减空行、压扁导入、把多行字典/参数硬挤成一行来冒充减量。**
- 以下情况不计入“有效减量”：
  - 只删空行或只改换行风格。
  - 只把注释改短，但逻辑未简化。
  - 只把变量名缩短。
  - 只把多行表达式硬折成一行。
  - 只靠拆成更多小文件制造“视觉变薄”。
- 只有以下减量算数：
  - 删除死代码、旧壳层、重复路径、兼容路径。
  - 合并重复 owner。
  - 把 workflow 从多跳 helper 压成单跳主入口。
  - 删除冗余 schema、diagnostics、artifact、cache 流程。
  - 把只服务单一路径的散 helper 收回 canonical owner。

### 4. 主目录不变，但目录内强制收敛

- 不新增新的一级业务目录。
- 允许在既有主目录内部合并、搬迁、内联文件，但必须同时满足：
  - 文件数减少或不增加。
  - 调用链更短。
  - owner 更清晰。
  - 总量真实下降。

### 5. 文件数量越少越好，默认策略是合并

- 默认策略是**合并**，不是拆分。
- 新增 `.py` 文件只有在满足以下全部条件时才允许：
  - 能明显缩短控制流跳转。
  - 能直接删除至少 `2` 个旧文件。
  - 本阶段净减量仍然达标。
- 本版之后新增的 support 文件，不默认视为终局结构；后续必须继续吸收或合并。

### 6. 单文件大小不再是硬顶，但必须服从 owner 纯洁性

- `500` 行不再是硬性上限。
- 如果合并能减少文件数量、缩短跳转、让 owner 更清晰，单文件允许超过 `500` 行。
- 但以下情况一律不允许：
  - 为了避免拆分，把多个无关 owner 硬塞进同一文件。
  - 把 wrapper、compatibility、debug/export、diagnostics 堆进核心文件里。
  - 把 `scripts/utils/` 变成新的“杂货间”。
- 运行期约束：
  - `> 1200` 的文件必须进入下一阶段主战场。
  - `> 800` 的文件只能是清晰的核心 owner，不能是 wrapper 或 utils 汇总桶。

### 7. 快照刷新纪律

- 每个阶段开始前，必须重新按本计划的统计口径测量一次：
  - 总行数
  - 文件数
  - 七个一级目录行数
  - 当前前十重文件
- 如果阶段开始时实际快照与本文不一致：
  - 先刷新本文快照。
  - 再重锁该阶段预算。
  - **不得拿着旧预算继续推进。**

### 8. tests 不计入总量，但必须同步收口

- `tests/` 不进入 `5000` 行目标，但不是“可以无限长胖”的豁免区。
- `TESTS_BASELINE=11850`，后续只允许下降，不允许阶段性反增。
- `tests/` 软上限：
  - 新 `N1` 末：`TESTS_LINES <= 8911`
  - 新 `N2` 末：`TESTS_LINES <= 8911`
  - 新 `N3` 末：`TESTS_LINES <= 8911`
  - 新 `N4` 末：`TESTS_LINES <= 8911`
  - `N5` 末：`TESTS_LINES <= 8911`
- 当前起始快照已经是 `8911`，所以从本版开始的所有新阶段默认**不得高于 `8911`**。
- 某条 owner 路径在本阶段被删掉、合并掉、去兼容化时：
  - 对应旧测试要在**同阶段**一起删或改。
  - 不允许保留只为旧兼容层服务的测试。
- 阶段验收必须同时给出本阶段代码 smoke 与对应测试状态。

### 9. notebook 只做薄入口

- notebook 只允许承担三件事：
  - 读取中心配置。
  - 调用单一主入口。
  - 展示结果或回读结果包。
- notebook 内不再保留：
  - 重复参数块。
  - schema 判断。
  - cache/package 搜索逻辑。
  - shared_data 生成主流程。
  - compare 重算逻辑。
  - solver、trainer、artifact 清理主控制流。

### 10. GitNexus 高风险改动纪律

- 任意函数、类、方法编辑前，都必须先做 GitNexus impact analysis。
- 命中 `HIGH` 或 `CRITICAL` 时，必须先说明 blast radius，再决定是否编辑。
- 提交前必须运行 `gitnexus detect_changes()`。

## 结构收口的终局约束

### owner 约束

- notebook 调用链必须尽量收敛为：
  - `notebook -> mainline_*.py -> canonical owner`
- 默认禁止：
  - `notebook -> mainline_*.py -> scripts/utils/* -> scripts/utils/* -> owner`
  - `notebook -> helper -> helper -> helper -> owner`
- 任何只做重导出的 wrapper，如果没有真正收回 ownership，都不算完成。

### 主入口约束

- 终局允许保留的 notebook / CLI 主入口，当前只应围绕以下四个文件收敛：
  - `predictors/mainline_forecast.py`
  - `scripts/mainline_madrl.py`
  - `scripts/mainline_compare.py`
  - `scripts/train.py`
- `scripts/builder.py`、`scripts/checkpoints.py` 只允许作为基础设施残量存在，不算独立业务入口；`N5` 前必须继续合并或进一步变薄。
- 保留下来的入口可以很薄，但必须是**唯一且稳定**的入口，不能再挂第二层 support 树。

### `scripts/utils/` 终局口径

- `scripts/utils/` 最终只允许保留小而纯的通用工具。
- 终局允许保留的类型：
  - 路径工具
  - price protocol
  - 极少量 torch/runtime 工具
  - 极少量真正跨主线共享且稳定的小函数
- 终局不应保留：
  - notebook workflow 主流程
  - compare orchestration
  - artifact orchestration
  - train launcher orchestration
  - 大体量 diagnostics helper

### `envs/` 终局子预算

- `envs/grid_env.py <= 240`
- `envs/grid/ <= 120`
- `envs/observation/ <= 200`
- 剩余 vec stack（当前只剩 `envs/subproc_vec_env.py`）`<= 240`
- `envs/rewards/ + 其余 residual <= 60`
- `envs/rewards/NormalReward.py` 必须在终局前压到 `<= 60`，否则就吸收进 `envs/grid_env.py` 或其它 retained owner；不允许总桶达标但 residual 子桶超线。
- 上述子预算合计对应 `envs/ <= 860`，后续阶段必须按这个拆分推进，不能只盯总桶。

### `controllers/` 终局子预算

- `controllers/mpc/ <= 920`
  - `controllers/mpc/global_socp_mpc.py <= 650`
  - `controllers/mpc/gurobi_agent_mpc.py <= 220`
  - `controllers/mpc/__init__.py` + 残余 `<= 50`
- `controllers/madrl/ <= 340`
  - `controllers/madrl/safety_projector.py <= 190`
  - `controllers/madrl/base_agent.py <= 150`
- `controllers/` 顶层 `<= 40`
  - `controllers/action_feasibility.py` 已不存在，不允许以任何形式回潮
  - `controllers/madrl_controller.py` 继续向 `base_agent.py` 或稳定入口吸收，终局不得保留第二层调度壳
  - 只允许保留 `controllers/__init__.py` 与精简后的 `madrl_controller.py`
- 上述子预算合计对应 `controllers/ <= 1300`，N5 必须按子桶单独校验，不允许只看总桶达标。

### `predictors/` 终局子预算

- `predictors/training.py <= 470`
- `predictors/lstm_forecaster.py <= 240`
- `predictors/shared_data.py <= 130`
- `predictors/mainline_forecast.py <= 30`
- `predictors/time_features.py <= 20`
- `predictors/oracle.py + registry.py + artifacts.py + lstm_model.py + base.py + __init__.py` 合并到 `<= 1` 个聚合文件、`<= 10` 行桩
- 终局 `predictors/` 文件数 `<= 6`
- 上述子预算合计 `470 + 240 + 130 + 30 + 20 + 10 = 900`，对应 `predictors/ <= 900`。

### `scripts/` 终局子预算

- `scripts/utils/ <= 850`（见 `scripts/utils/` 终局口径）
- `scripts/plots/ <= 300`
- `scripts/` 顶层 `<= 220`
  - `scripts/train.py <= 60`
  - `scripts/mainline_madrl.py <= 60`
  - `scripts/mainline_compare.py <= 80`
  - `scripts/builder.py` + `scripts/checkpoints.py` + `scripts/__init__.py <= 20`
- 上述子预算合计对应 `scripts/ <= 1370`。
- 其中 `scripts/plots/ <= 300` 只是总桶上限；由于当前只剩 `grid_notebook_plotting.py` 单文件，执行口径仍按 `N1` 收到 `<= 240` 后不得回弹。

### `configs/` 终局子预算

- 方案 A（保留 `profiles.py`）：`experiment_config.py <= 180` + `profiles.py <= 60`。
- 方案 B（合并）：`profiles.py` 内容并入 `experiment_config.py`，只留一个 `.py`，总量 `<= 240`。
- 终局二选一，不允许保留 `profiles.py` 同时 `experiment_config.py` 超 200。

### 已完成的壳层处置

- `scripts/mainline_mpc.py`、`scripts/mainline_grid_analysis.py`、`scripts/mainline_artifacts.py` 已经删除。
- `controllers/action_feasibility.py`、`controllers/base.py`、`controllers/madrl/registry.py` 及旧算法分裂文件已被吸收或删除。
- 上述对象**不得以新的转发壳或兼容壳形式重新出现**。

### 现有 residual support 的处理原则

- `scripts/utils/grid_notebook_workflow.py`
- `scripts/utils/admm_mpc_solver.py`
- `scripts/utils/admm_mpc_notebook_helpers.py`
- `scripts/utils/torch_runtime.py`
- `scripts/utils/replay_buffer.py`

以上文件当前可以短期存在，但**不预设它们是终局结构**。后续必须判断：

- 能吸回 `scripts/mainline_madrl.py`、`scripts/mainline_compare.py`、`scripts/train.py` 或 canonical owner，就吸回。
- 吸不回时，也必须继续合并成更少的 residual support 文件。
- 绝不允许围绕这些 residual support 再长出第二层 helper。

## 前序收束完成记录

### 已完成的三次收束

- `34c388a 第二次收束`
- `93b80a1 第三次收束`
- `e42f072 第四次收束`

### 已兑现的量化结果

- `CURRENT: 10651 -> 8526`，本轮前序收束累计新增净减 `-2125` 行。
- `PY_FILE_COUNT: 79 -> 57`，已经提前把文件数压到离终局 `<= 55` 只差 `2` 个文件。
- `TESTS_LINES: 9875 -> 8911`，已经提前进入本版统一执行线 `<= 8911` 区间。
- 一级目录里，`controllers/`、`envs/`、`scripts/` 都已经先吃掉了一轮大块减量，后续不能允许反弹。

### 已完成的结构动作

- 已删掉入口纯壳：
  - `scripts/mainline_mpc.py`
  - `scripts/mainline_grid_analysis.py`
  - `scripts/mainline_artifacts.py`
- 已删掉旧 support / workflow 壳：
  - `scripts/utils/experiment_notebook_utils.py`
  - `scripts/utils/mainline_setup.py`
  - `scripts/utils/train_progress.py`
  - `scripts/utils/train_runner_support.py`
  - `scripts/utils/train_mainline_support.py`
  - `scripts/utils/misocp_notebook_helpers.py`
  - `scripts/utils/misocp_notebook_diagnostics.py`
  - `scripts/utils/misocp_plan_packages.py`
  - `scripts/utils/rollout_package_utils.py`
- 已删掉 controller / MADRL 分裂 owner：
  - `controllers/action_feasibility.py`
  - `controllers/base.py`
  - `controllers/madrl/maddpg.py`
  - `controllers/madrl/matd3.py`
  - `controllers/madrl/matd3_safe_poc.py`
  - `controllers/madrl/registry.py`
- 已删掉旧 env / observation / vec 分裂层：
  - `envs/vec_env.py`
  - `envs/parallel_episode_sampling.py`
  - `envs/observation/registry.py`
  - `envs/observation/precomputed_builder.py`
  - `envs/observation/features.py`
  - `envs/observation/feature_blocks.py`
  - `envs/grid/core/grid_types.py`
- 已完成的 owner 回收：
  - `MADDPG` / `MATD3` / `MATD3_SAFE_POC` 已经并回 `controllers/madrl/base_agent.py`
  - `action_feasibility` 的核心能力已吸回 `controllers/madrl/safety_projector.py`
  - mainline / train / notebook 相关 support 已经完成第一轮吸收

### 从现在开始的阶段重排原则

- 旧版里“为了补 `126` 行差额而单列一个阶段”的想法到此为止。
- 新版阶段体系从**当前快照 `CURRENT=8526`** 重新编号，直接写成新的 `N1-N5`。
- 新版 `N1-N5` 只围绕**仍然存在且仍然超重**的 owner 展开，不再把已删除文件写进后续计划。

## 新的 N1-N5 总览

### 阶段目标线

| 新阶段 | 总行数目标 | 文件数目标 | 说明 |
| ---- | ---------: | ---------: | ---- |
| `N1` | `<= 7800` | `<= 57` | 主战场：`scripts/` 顶层 + `mainline_compare` + `scripts/utils/` + plotting |
| `N2` | `<= 7150` | `<= 56` | 主战场：`predictors/` + forecast/data 主线 |
| `N3` | `<= 6300` | `<= 56` | 主战场：`controllers/` + `envs/` 主体 |
| `N4` | `<= 5600` | `<= 55` | 主战场：`models/` + `configs/` + `data/` + residual 集成 |
| `N5` | `<= 5000` | `<= 55`，冲刺 `<= 50` | 全仓终局验收 |

### 分阶段净减预算

| 新阶段 | 计划净减 | 主要来源 |
| ---- | -------: | ---- |
| `N1` | `>= 726` | `scripts/` 顶层、`mainline_compare`、`scripts/utils/`、`scripts/plots/` |
| `N2` | `>= 650` | `predictors/training.py`、`lstm_forecaster.py`、`shared_data.py`、forecast/data 路径 |
| `N3` | `>= 850` | `controllers/mpc/`、`controllers/madrl/`、`envs/`、vec/observation 主路径 |
| `N4` | `>= 700` | `models/`、`configs/`、`data/`、residual scripts/predictors |
| `N5` | `>= 600` | 最终 wrapper、残余大文件、精确对桶、文件数最后两步 |

- `726 + 650 + 850 + 700 + 600 = 3526`，与当前距离终局的差额完全对齐。
- 当前 `PY_FILE_COUNT=57`，所以新阶段的主任务是**行数减量**，不是为了做文件数而乱拆结构。
- 任一新阶段只要净减不足预算的 `70%`，就必须暂停并重排，不得机械推进。

## 阶段 N1：优先打 `scripts/` 顶层、compare、utils、plots

### 目标

- 先吃掉当前最大且最碎的 `scripts/` 桶，不再单独为旧版那 `126` 行缺口开战场。
- 把 `scripts/` 顶层从“多入口 + orchestration 混合区”压回真正的 canonical owner。
- 让 compare、workflow、plotting 只保留单跳路径，不再继续承担第二控制中心。

### 主战场

- `scripts/mainline_compare.py`
- `scripts/mainline_madrl.py`
- `scripts/train.py`
- `scripts/builder.py`
- `scripts/checkpoints.py`
- `scripts/utils/grid_notebook_workflow.py`
- `scripts/utils/admm_mpc_solver.py`
- `scripts/utils/admm_mpc_notebook_helpers.py`
- `scripts/utils/torch_runtime.py`
- `scripts/plots/grid_notebook_plotting.py`

### 必删 / 必收路径

- `scripts/mainline_compare.py`
  - 删内联的 MISOCP 重跑 / validation artifact 组装侧链。
  - 删重复的 safety local / action gap 计算侧链。
  - compare 入口只保留“读结果包”和“调用单一 rollout owner”两类职责。
- `scripts/mainline_madrl.py` 与 `scripts/train.py`
  - 吸回当前仍散落在 residual support 中、且只服务单一路径的 orchestration。
  - 不再重复维护 run metadata、checkpoint、summary、tensorboard、目录协议的第二份逻辑。
- `scripts/builder.py` 与 `scripts/checkpoints.py`
  - 只允许保留基础设施职责。
  - `N1` 结束时若仍存在，必须已经非常薄，不能再像独立业务 owner。
- `scripts/utils/grid_notebook_workflow.py`
  - 只保留 notebook 单跳 workflow。
  - 不再内含第二层 compare/helper/packaging 树。
- `scripts/utils/admm_mpc_solver.py`
  - 删 notebook-only package、重复 diagnostics、重复汇总路径。
- `scripts/plots/grid_notebook_plotting.py`
  - 只保留“读结果包 -> dataframe -> 绘图”。
  - 不允许再承担 solver 重跑、schema 兼容、artifact 清理。

### 阶段内 checkpoint

- `N1a`
  - `CURRENT <= 8150`
  - `scripts/ <= 2450`
  - `scripts/mainline_compare.py <= 500`
  - `scripts/utils/grid_notebook_workflow.py <= 270`
- `N1b`
  - `CURRENT <= 7800`
  - `scripts/ <= 2100`
  - `scripts/utils/ <= 900`
  - `scripts/plots/ <= 240`
  - `scripts/mainline_compare.py <= 420`
  - `scripts/train.py <= 240`
  - `scripts/mainline_madrl.py <= 240`

### 本阶段硬要求

- `CURRENT <= 7800`
- `PY_FILE_COUNT <= 57`
- `scripts/ <= 2100`
- `scripts/` 顶层 `<= 1100`
- `scripts/utils/ <= 900`
- `scripts/plots/ <= 240`
- `scripts/mainline_compare.py <= 420`
- `scripts/train.py <= 240`
- `scripts/mainline_madrl.py <= 240`
- `scripts/utils/grid_notebook_workflow.py <= 260`
- `scripts/utils/admm_mpc_solver.py <= 260`
- `scripts/plots/grid_notebook_plotting.py <= 240`

### 本阶段策略

- 不允许新建 compare helper 树、workflow helper 树、plot helper 树。
- `scripts/` 的减量必须来自真实 owner 回收和重复 orchestration 删除，不接受格式性减量。
- `TESTS_LINES` 在本阶段结束时必须仍然 `<= 8911`。

## 阶段 N2：集中处理 `predictors/` 与 forecast/data 主线

### 目标

- 把 `predictors/` 从当前 `1713` 行压到明显更低的单主线路径。
- 让 `training.py` 成为单一路径训练 owner，不再兼任多模式工厂、artifact inventory 平台、split 选择器集合。
- 把 forecast/data 相关的残余兼容逻辑一起收掉，避免 predictor 清完后又从 data/config 回流。

### 主战场

- `predictors/training.py`
- `predictors/lstm_forecaster.py`
- `predictors/shared_data.py`
- `predictors/mainline_forecast.py`
- `predictors/oracle.py`
- `predictors/registry.py`
- `predictors/artifacts.py`
- `predictors/base.py`
- `predictors/lstm_model.py`
- `predictors/time_features.py`
- `data/loaders/prosumer.py`

### 必删 / 必收路径

- `predictors/training.py`
  - 合并重复 loader：matrix / segments / frames 只留一条 canonical 路径。
  - 合并重复 split / window builder：只留一个 canonical builder。
  - 删 artifact inventory / validate / ensure 的重复 orchestration。
  - 删只为多模式泛化服务的参数分支。
- `predictors/shared_data.py`
  - 只保留当前主线 manifest / signature / path 协议。
  - 删旧 schema、旧命名、旧导出约定的兼容判断。
- `predictors/oracle.py`、`registry.py`、`artifacts.py`、`base.py`、`lstm_model.py`、`time_features.py`
  - 优先并成更少文件。
  - 不保留为“以后也许有别的 predictor”而存在的壳层。
- `data/loaders/prosumer.py`
  - 删重复数据清洗与路径兼容分支。
  - 只保留当前单场景数据入口。

### 阶段内 checkpoint

- `N2a`
  - `CURRENT <= 7500`
  - `predictors/ <= 1450`
  - `predictors/training.py <= 760`
  - `predictors/shared_data.py <= 220`
- `N2b`
  - `CURRENT <= 7150`
  - `predictors/ <= 1250`
  - `PY_FILE_COUNT <= 56`
  - `predictors/training.py <= 650`
  - `predictors/lstm_forecaster.py <= 300`

### 本阶段硬要求

- `CURRENT <= 7150`
- `PY_FILE_COUNT <= 56`
- `predictors/ <= 1250`
- `predictors/` 文件数 `<= 9`
- `predictors/training.py <= 650`
- `predictors/lstm_forecaster.py <= 300`
- `predictors/shared_data.py <= 180`
- `predictors/mainline_forecast.py <= 30`
- `data/ <= 300`
- `data/loaders/prosumer.py <= 220`

### 本阶段策略

- 不允许通过新建 `predictors/support_*.py` 转移复杂度。
- predictor 的减量必须来自重复 loader / split / artifact orchestration 的真实删除。
- `TESTS_LINES` 在本阶段结束时必须仍然 `<= 8911`。

## 阶段 N3：压扁 `controllers/` 与 `envs/` 主体

### 目标

- 这是新的“大块核心 owner 阶段”，重点是 `controllers/` 与 `envs/`。
- 把 `mpc`、`madrl`、`grid_env`、`vec stack`、`observation` 从“还能工作但仍偏厚”的状态推到稳定终局形态附近。
- 不再保留 diagnostics / debug export / retry / compatibility 的副路径。

### 主战场

- `controllers/mpc/global_socp_mpc.py`
- `controllers/mpc/gurobi_agent_mpc.py`
- `controllers/madrl/safety_projector.py`
- `controllers/madrl/base_agent.py`
- `controllers/madrl_controller.py`
- `envs/grid_env.py`
- `envs/subproc_vec_env.py`
- `envs/grid/core/grid_core.py`
- `envs/observation/default_builder.py`
- `envs/observation/normalization.py`

### 必删 / 必收路径

- `controllers/mpc/global_socp_mpc.py`
  - 删 debug/export 侧链。
  - 删 retry / fallback / chunk telemetry 侧链。
  - 删过厚的 refinement / diagnostics / result baggage。
- `controllers/madrl/safety_projector.py`
  - 删重复 feasibility / diagnostics 路径。
  - 只保留当前 projection-safe 主线需要的那一条路径。
- `controllers/madrl/base_agent.py` 与 `controllers/madrl_controller.py`
  - 删第二层 orchestration 壳。
  - 训练、动作、共享上下文只保留当前 retained 算法路径。
- `envs/grid_env.py`、`envs/subproc_vec_env.py`
  - 删历史缓存、旧观测、旧 vec 语义兼容。
  - vec stack 只保留剩余这一个 canonical 文件。
- `envs/observation/*`
  - observation builder / normalization 只保留单 schema 路径。
  - 不再回潮 registry / feature block / precomputed builder 思路。

### 阶段内 checkpoint

- `N3a`
  - `CURRENT <= 6700`
  - `controllers/ <= 1700`
  - `envs/ <= 1050`
  - `controllers/mpc/global_socp_mpc.py <= 740`
  - `controllers/madrl/safety_projector.py <= 360`
- `N3b`
  - `CURRENT <= 6300`
  - `controllers/ <= 1500`
  - `envs/ <= 950`
  - `PY_FILE_COUNT <= 56`

### 本阶段硬要求

- `CURRENT <= 6300`
- `PY_FILE_COUNT <= 56`
- `controllers/ <= 1500`
- `controllers/mpc/ <= 900`
- `controllers/madrl/ <= 520`
- `controllers/mpc/global_socp_mpc.py <= 700`
- `controllers/mpc/gurobi_agent_mpc.py <= 240`
- `controllers/madrl/safety_projector.py <= 300`
- `controllers/madrl/base_agent.py <= 220`
- `envs/ <= 950`
- `envs/grid_env.py <= 240`
- 剩余 vec stack（当前只剩 `envs/subproc_vec_env.py`）`<= 240`
- `envs/grid/ <= 180`
- `envs/observation/ <= 220`

### 本阶段策略

- 这一阶段的减量必须来自 owner 纯化，不允许把复杂度迁移到新的 helper 文件。
- `GlobalMISOCPProblem`、`GridEnv`、`build_env`、`collect_controller_rollout` 等入口一律先做 GitNexus impact。
- `TESTS_LINES` 在本阶段结束时必须仍然 `<= 8911`。

## 阶段 N4：对齐 `models/`、`configs/`、`data/`，并清 residual 集成

### 目标

- 这一阶段不是单纯“抹边角”，而是把小桶和 residual 集成全部对齐到终局预算附近。
- 模型装配必须回到更少文件、更少抽象层。
- 配置、数据、基础设施残量必须停止作为回潮入口。

### 主战场

- `models/assembly.py`
- `models/family_adapters.py`
- `models/utils.py`
- `models/encoders/mlp_encoder.py`
- `models/heads/actor_head.py`
- `models/heads/critic_head.py`
- `configs/experiment_config.py`
- `configs/profiles.py`
- `data/loaders/prosumer.py`
- `scripts/mainline_compare.py`
- `scripts/mainline_madrl.py`
- `scripts/train.py`
- `scripts/builder.py`
- `scripts/checkpoints.py`
- `scripts/plots/grid_notebook_plotting.py`
- `predictors/` 与 `scripts/` 剩余 residual 文件

### 必删 / 必收路径

- `models/`
  - tiny files 必须吸回 retained owner。
  - family / adapter / utils 只能保留当前 retained `mlp` 所必需的那一层。
- `configs/experiment_config.py` / `profiles.py`
  - 合并重复 profile、bus、window、artifact path 组。
  - 删备用参数块与镜像字段。
- `scripts/mainline_compare.py` / `scripts/mainline_madrl.py` / `scripts/train.py`
  - `N4` 的 `scripts/ <= 1550` 不能只靠 `builder.py` / `checkpoints.py` 两个基础设施文件兑现。
  - 顶层 mainline orchestration 必须继续从 `N1` 的中间收口线往终局推进，不允许在 `N4` 停在 `420 / 240 / 240` 这种半成品状态。
- `scripts/builder.py` / `scripts/checkpoints.py`
  - 继续向真正的 retained owner 吸收。
  - 不允许保留“只是因为以前有过独立文件”而存在的基础设施壳。
- `scripts/plots/grid_notebook_plotting.py`
  - 若 `N1` 已压到 `<= 240`，则 `N4` 只能保持或继续变薄，不得回弹。
- `data/loaders/prosumer.py`
  - 和 N2 残量一起对齐到终局预算，不留下第二轮清理尾巴。

### 阶段内 checkpoint

- `N4a`
  - `CURRENT <= 5900`
  - `models/` 文件数 `<= 4`
  - `configs/ <= 300`
  - `data/ <= 260`
  - `scripts/ <= 1750`
  - `scripts/` 顶层 `<= 700`
- `N4b`
  - `CURRENT <= 5600`
  - `PY_FILE_COUNT <= 55`
  - `models/ <= 170`
  - `configs/ <= 260`
  - `data/ <= 230`
  - `scripts/ <= 1550`
  - `scripts/utils/ <= 860`
  - `scripts/plots/ <= 240`
  - `scripts/` 顶层 `<= 450`

### 本阶段硬要求

- `CURRENT <= 5600`
- `PY_FILE_COUNT <= 55`
- `models/ <= 170`
- `models/` 文件数 `<= 4`
- `configs/ <= 260`
- `data/ <= 230`
- `scripts/ <= 1550`
- `scripts/utils/ <= 860`
- `scripts/plots/ <= 240`
- `scripts/` 顶层 `<= 450`
- `predictors/ <= 1000`
- `configs/experiment_config.py <= 220`
- `data/loaders/prosumer.py <= 180`

### 本阶段策略

- `models/` 只能通过吸回 owner 变薄，不允许换名字继续分层。
- `configs/` 和 `data/` 的减量必须来自重复字段、重复协议、重复清洗路径的删除。
- `PY_FILE_COUNT` 在本阶段必须第一次真正压进 `<= 55`。
- `TESTS_LINES` 在本阶段结束时必须仍然 `<= 8911`。

## 阶段 N5：最终收尾与全量验收

### 目标

- 从 `5600` 打到 **`<= 5000`**。
- 让仓库进入真正终局状态：
  - 代码纯
  - 文件少
  - 没有壳层
  - 没有兼容债
  - 没有只是为了“显得模块化”而存在的杂乱 support 树

### 启动前门槛

- 必须列出当前剩余全部 `>= 150` 行文件。
- 对每个文件逐个标注：
  - `删`
  - `压到 X`
  - `保留`
- 必须按文件列出并加总，证明剩余可减量 `>= 650` 行。
- 其中 `650` 相比 `N5` 计划净减 `600` 多出的 `50` 行，是给最后一公里反弹和统计误差的安全带。
- 证明不出来，不得启动 `N5`。

### 终局硬验收

- `CURRENT <= 5000`
- `PY_FILE_COUNT <= 55`，冲刺 `<= 50`
- `scripts/utils/ <= 850` 且文件数 `<= 7`
- `scripts/ <= 1370`
- `configs/ <= 240`
- `controllers/ <= 1300`
- `data/ <= 190`
- `envs/ <= 860`
- `envs/grid/ <= 120`
- `envs/observation/ <= 200`
- 剩余 vec stack（当前只剩 `envs/subproc_vec_env.py`）`<= 240`
- `predictors/ <= 900`
- `models/ <= 140`
- `TESTS_LINES <= 8911`
- 命中旧 schema、旧 key、旧缓存包、旧 artifact 名称时统一直接失败
- `tests/` 只覆盖当前主线，不再维护兼容层
- 若终局仍存在 `> 800` 的文件，该文件必须是明确的核心 owner，且不得含有 wrapper / debug export / compatibility 负担

## 当前最优先的 17 个目标文件

1. `predictors/training.py`
2. `controllers/mpc/global_socp_mpc.py`
3. `scripts/mainline_compare.py`
4. `controllers/madrl/safety_projector.py`
5. `predictors/lstm_forecaster.py`
6. `envs/subproc_vec_env.py`
7. `controllers/mpc/gurobi_agent_mpc.py`
8. `scripts/train.py`
9. `scripts/mainline_madrl.py`
10. `scripts/utils/grid_notebook_workflow.py`
11. `envs/grid_env.py`
12. `scripts/utils/admm_mpc_solver.py`
13. `controllers/madrl/base_agent.py`
14. `scripts/plots/grid_notebook_plotting.py`
15. `data/loaders/prosumer.py`
16. `predictors/shared_data.py`
17. `configs/experiment_config.py`

## 每个阶段都必须遵守的执行纪律

### 1. 优先打大头，不许刷边角

- 同一阶段如果已经改了 3 个以上主文件，但净减量还不到该阶段预算的 `25%`，必须暂停。
- 禁止靠删空行、压一行、修格式来汇报“本阶段完成”。

### 2. 不允许文件数失控

- 任一阶段如果 `.py` 文件数不降反升，必须说明新增文件对应删除了哪些旧文件。
- 解释不清，视为阶段失败。

### 3. 不允许已收口区域反弹

- 已完成收口的入口、utils、plots、predictor、env 区域不得重新长胖。
- 已压下去的 notebook 逻辑不得再被搬回 notebook 单元里。

### 4. 任何高风险代码改动前必须先做 GitNexus impact

- 特别是：
  - `GridEnv`
  - `TrainRunner`
  - `build_env`
  - `collect_controller_rollout`
  - `build_misocp_validation_artifacts`
  - `GlobalMISOCPProblem` 相关入口

## 分阶段验证矩阵

### `N1`

- `tests/test_train_mainline_launcher.py`
- `tests/test_train_progress.py`
- `tests/test_checkpoints.py`
- `tests/test_builder_vec_env.py`
- `tests/test_grid_notebook_workflow.py`
- `tests/test_evaluation_plots.py`
- `tests/test_compare_notebook.py`
- `scripts/mainline_compare.py` smoke
- `scripts/mainline_madrl.py` smoke
- `scripts/train.py` smoke

### `N2`

- `tests/test_forecast_scaling.py`
- `tests/test_lstm_load_per_agent.py`
- `tests/test_madrl_shared_data.py`
- `tests/test_prosumer_dataset.py`
- `tests/test_prosumer_integration.py`
- `tests/test_registries.py`
- `predictors/mainline_forecast.py` smoke
- forecast / shared_data / dataset 主线 smoke

### `N3`

- `tests/test_global_socp_mpc.py`
- `tests/test_global_socp_mpc_orchestration.py`
- `tests/test_gurobi_agent_mpc.py`
- `tests/test_safety_projector.py`
- `tests/test_grid_env.py`
- `tests/test_parallel_episode_sampling.py`
- `tests/test_observation_schema.py`
- `tests/test_observation_normalization.py`
- `tests/test_grid_core.py`
- `tests/test_local_mpc_notebook.py`
- `tests/test_misocp_global_notebook.py`
- controller / env 主线 smoke

### `N4`

- `tests/test_model_assembly.py`
- `tests/test_madrl_notebook_defaults.py`
- `tests/test_normal_reward.py`
- `tests/test_action_dtype_casts.py`
- `tests/test_experiment_notebook_utils.py`
- `tests/test_notebook_utils.py`
- `tests/test_registries.py`
- config / model assembly / residual integration smoke

### `N5`

- `tests/test_madrl_notebook_smoke.py`
- `tests/test_admm_mpc_notebook.py`
- `tests/test_admm_mpc_notebook_helpers.py`
- `tests/test_compare_notebook.py`
- `tests/test_local_mpc_notebook.py`
- `tests/test_misocp_global_notebook.py`
- `tests/test_encoding_hygiene.py`
- `forecast_lstm.ipynb -> shared_data -> MADRL train/eval -> result package` 一条龙 smoke
- compare 只读结果包 smoke
- `grid_network_analysis.ipynb` smoke
- 六条主线主入口 smoke
- 全部目标测试集合回归

## 以后每次阶段汇报的固定模板

- `BASELINE=33323`
- `CURRENT=<当前值>`
- `DELTA=<净变化>`
- `PY_FILE_COUNT=<当前文件数>`
- `TESTS_LINES=<当前 tests 总行数 / 相对上一阶段变化>`
- `PHASE=<当前阶段>`
- `TARGET_NEXT=<下一个 checkpoint>`
- `OVER_800_FILES=<文件列表 + 各自 owner 合法性说明>`
- `TOP_BLOCKERS=<当前最大的 3 个文件/区域>`
- `REAL_REDUCTION=<本阶段真实删掉了哪些 owner / 兼容 / 重复逻辑>`

- 没有 `REAL_REDUCTION` 的汇报，不算有效汇报。
