# MADRL_ESS 第二轮极限收口计划（5000 行冲刺版，重铸修订版，2026-04-21）

## 说明

- 本版继续保持原方向：一级主目录不变，六条主线不变，notebook 保留，原则仍然是“单场景、单主线、不兼容、不兜底、真实减量”。
- 本版修正上一版最关键的问题：**计划必须以当前仓库真实快照为准**。本次已按当前工作区重新测量，并据此重锁全部阶段预算。
- 本版还做了六个结构性修订：
  - 去掉“单个文件最终必须 <= 500 行”的硬顶，改成“少文件、短调用链、清晰 owner 优先”，允许单文件超过 `500` 行。
  - `N2` 不再同时并行打太多战场；`scripts/plots/` 和 `scripts/utils/grid_notebook_workflow.py` 延后到 `N3`。
  - 为 `controllers/mpc/global_socp_mpc.py`、`predictors/training.py`、`models/`、`scripts/plots/grid_notebook_plotting.py`、`configs/experiment_config.py` 增加可直接执行的删减路径清单。
  - `tests/` 虽然不计入总行数目标，但必须跟随 owner 删除在同阶段同步收口，不能拖到最后，且给出软上限阶段曲线。
  - 补齐 **`controllers/` / `envs/` / `predictors/` / `scripts/` / `configs/` 终局子桶预算**，让子文件预算加总对得上一级桶预算；原版 `controllers/ 1300`、`envs/ 760`、`predictors/ 920`、`scripts/ 1410` 与已列子目标存在加总超桶或子文件没预算的问题，本版已重分配并显式列出子桶（含 `controllers/ 顶层` 与 `controllers/action_feasibility.py` 的合并归宿）。
  - 为 `N2` 补一份 **净减量按文件列账表**（预计 `-1775` 行，对 `>= 1764` 目标留 11 行安全余量），防止阶段机械推进却无实际来源。
- 如果后续执行中证明 `5000` 在不破坏代码纯洁性的前提下不可达，**不能自行降格**，必须停下来单独上报，由用户决策。

## 当前严格快照

- 统计口径：
  - 只统计 `configs/`、`controllers/`、`data/`、`envs/`、`models/`、`predictors/`、`scripts/` 下当前实际存在的 `.py` 文件。
  - `tests/` 不计入总行数目标，但要单独跟随维护。
  - 工作区中的未跟踪 `.py` 文件也算数。
  - 不统计 `.conda/`、第三方依赖目录、notebook 本体。
- 当前基线：
  - `BASELINE=33323`
  - `CURRENT=10651`
  - `DELTA=-22672`
  - `PY_FILE_COUNT=79`
  - `TESTS_LINES=9875`
  - 距离 `5000` 目标还差 `5651` 行

### 一级目录现状

| 目录 | 当前行数 | 当前文件数 | 终局预算 |
| ---- | -------: | ---------: | -------: |
| `configs/` | 390 | 2 | <= 240 |
| `controllers/` | 2739 | 13 | <= 1300 |
| `data/` | 338 | 5 | <= 190 |
| `envs/` | 1832 | 23 | <= 860 |
| `models/` | 242 | 6 | <= 140 |
| `predictors/` | 1751 | 11 | <= 900 |
| `scripts/` | 3359 | 19 | <= 1370 |

- 以上终局预算合计正好是 `5000`。

### 当前最重的文件

| 文件 | 当前行数 | 终局目标 |
| ---- | -------: | -------: |
| `controllers/mpc/global_socp_mpc.py` | 1185 | <= 650 |
| `predictors/training.py` | 842 | <= 470 |
| `scripts/utils/grid_notebook_workflow.py` | 569 | <= 240 |
| `scripts/utils/admm_mpc_solver.py` | 463 | <= 200 |
| `envs/grid_env.py` | 385 | <= 260 |
| `predictors/lstm_forecaster.py` | 384 | <= 240 |
| `controllers/mpc/gurobi_agent_mpc.py` | 361 | <= 240 |
| `scripts/train.py` | 345 | <= 180 |
| `scripts/utils/misocp_notebook_helpers.py` | 323 | <= 200 |
| `controllers/madrl/safety_projector.py` | 318 | <= 180 |
| `scripts/plots/grid_notebook_plotting.py` | 308 | <= 240 |
| `configs/experiment_config.py` | 272 | <= 180 |
| `controllers/action_feasibility.py` | 198 | 并入 `controllers/madrl/safety_projector.py` |
| `data/loaders/prosumer.py` | 272 | <= 150 |
| `predictors/shared_data.py` | 264 | <= 130 |
| `scripts/mainline_madrl.py` | 231 | <= 120 |

### 当前最碎的区域

| 区域 | 当前文件数 | 当前行数 | 终局方向 |
| ---- | ---------: | -------: | ---- |
| `scripts/utils/` | 10 | 1895 | 压到 <= 7 个文件、<= 850 行 |
| `scripts/plots/` | 3 | 493 | 压到 <= 2 个文件、<= 300 行 |
| `scripts/` 顶层（`mainline_*.py` + `train.py` + `builder.py` + `checkpoints.py`） | 5 | 971 | 合并到 <= 5 个文件、<= 220 行 |
| `controllers/mpc/` | 3 | 1559 | 保留主解法，压到 <= 3 个文件、<= 920 行 |
| `controllers/madrl/` | 6 | 854 | 吸收 `action_feasibility.py` 后，压到 <= 4 个文件、<= 340 行 |
| `controllers/` 顶层（`base.py` + `madrl_controller.py` + `action_feasibility.py`） | 3 | 324 | 并入 madrl owner 或删除，终局 <= 40 行 |
| `envs/grid/` | 8 | 414 | 压到 <= 3 个文件、<= 120 行 |
| `envs/observation/` | 8 | 550 | 压到 <= 4 个文件、<= 200 行 |
| `envs/vec_stack` | 2 | 345 | 合并到 <= 2 个文件、<= 240 行 |
| `predictors/` | 11 | 1751 | 压到 <= 6 个文件、<= 900 行 |
| `models/` | 6 | 242 | 压到 <= 3 个文件、<= 140 行 |

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
- N1 之后新增的 support 文件，不默认视为终局结构；后续必须继续吸收或合并。

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
  - `N2` 末：`TESTS_LINES <= 11850`
  - `N3` 末：`TESTS_LINES <= 11000`
  - `N4` 末：`TESTS_LINES <= 9800`
  - `N5` 末：`TESTS_LINES <= 9000`
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

- 终局允许保留的主入口最多包含以下六个文件，但 **`mainline_mpc.py` 与 `mainline_grid_analysis.py` 是可删的候选**（当前为 4 行纯壳，见"空壳主入口处置"）：
  - `predictors/mainline_forecast.py`
  - `scripts/mainline_madrl.py`
  - `scripts/mainline_compare.py`
  - `scripts/mainline_artifacts.py`
  - `scripts/mainline_mpc.py`（可删 / 可长成 owner 二选一）
  - `scripts/mainline_grid_analysis.py`（可删 / 可长成 owner 二选一）
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

- `envs/grid_env.py <= 260`
- `envs/grid/ <= 120`
- `envs/observation/ <= 200`
- `envs/subproc_vec_env.py + envs/vec_env.py + envs/parallel_episode_sampling.py <= 240`
- `envs/rewards/ + 其余 residual <= 40`
- 上述子预算合计对应 `envs/ <= 860`，后续阶段必须按这个拆分推进，不能只盯总桶。

### `controllers/` 终局子预算

- `controllers/mpc/ <= 920`
  - `controllers/mpc/global_socp_mpc.py <= 650`
  - `controllers/mpc/gurobi_agent_mpc.py <= 240`
  - `controllers/mpc/__init__.py` + 残余 `<= 30`
- `controllers/madrl/ <= 340`
  - 吸收 `controllers/action_feasibility.py` 的 feasibility 核心后，`safety_projector.py <= 180`
  - `base_agent.py <= 80`、`matd3.py + matd3_safe_poc.py` 合并 `<= 60`
  - `maddpg.py + registry.py + __init__` 合并 `<= 20`
- `controllers/` 顶层 `<= 40`
  - `controllers/action_feasibility.py` 不再独立存在（并入 `controllers/madrl/safety_projector.py`）
  - `madrl_controller.py` 合并到 `base_agent.py` 或直接删除
  - 只允许保留 `controllers/__init__.py` 与精简后的 `base.py`
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
  - `scripts/train.py <= 100`
  - `scripts/mainline_madrl.py <= 60`
  - `scripts/mainline_compare.py <= 40`
  - `scripts/mainline_artifacts.py <= 80`（与其它顶层互相借位，不得超顶层 220 总额）
  - 其余 `scripts/builder.py`、`scripts/checkpoints.py` 需合并到 `train.py` 或彼此合并成单文件 `<= 20`
- 上述子预算合计对应 `scripts/ <= 1370`。

### `configs/` 终局子预算

- 方案 A（保留 `profiles.py`）：`experiment_config.py <= 180` + `profiles.py <= 60`。
- 方案 B（合并）：`profiles.py` 内容并入 `experiment_config.py`，只留一个 `.py`，总量 `<= 240`。
- 终局二选一，不允许保留 `profiles.py` 同时 `experiment_config.py` 超 200。

### 空壳主入口处置（必须在 `N2` 收口）

- `scripts/mainline_mpc.py`（当前 4 行，纯壳）、`scripts/mainline_grid_analysis.py`（当前 4 行，纯壳）只能二选一：
  - A. 直接删除，notebook 改为调用 `scripts/utils/grid_notebook_workflow.py` 或对应 owner 作为稳定入口。
  - B. 长成唯一 owner，但必须明确吸回多少行、吸回的来源文件必须在同阶段被删掉（避免净行数反增）。
- 不允许两个入口继续以"4 行转发壳"状态进入 `N3`。

### 现有 support 文件的处理原则

- `scripts/utils/mainline_setup.py`
- `scripts/utils/train_progress.py`
- `scripts/utils/train_runner_support.py`
- `scripts/utils/train_mainline_support.py`

以上文件当前可以短期存在，但**不预设它们是终局结构**。后续必须判断：

- 能吸回 `scripts/train.py` 或 `scripts/mainline_madrl.py`，就吸回。
- 吸不回时，也必须继续合并成更少的 support 文件。
- 绝不允许围绕这些 support 文件再长出第二层 helper。

## 剩余阶段总览

### 阶段目标线

| 阶段 | 总行数目标 | 文件数目标 | 说明 |
| ---- | ---------: | ---------: | ---- |
| `N1` | 已完成 `11594` | `81` | 当前基准点 |
| `N2` | `<= 10700` | `<= 86` | 回收 N1 碎片，只打入口 support 与 utils 第一层 |
| `N3` | `<= 8400` | `<= 72` | 主战场：controller/mpc/env + workflow/plots |
| `N4` | `<= 6400` | `<= 61` | predictor/models/data/notebook 配套收口 |
| `N5` | `<= 5000` | `<= 55`，冲刺 `<= 50` | 全仓终局验收 |

### 分阶段净减预算

| 阶段 | 计划净减 | 主要来源 |
| ---- | -------: | ---- |
| `N2` | >= 1764 | 入口 support 回收、`scripts/utils/` 第一层合并、纯壳删除 |
| `N3` | >= 2300 | `controllers/mpc/`、`controllers/madrl/`、`envs/`、`grid_notebook_workflow.py`、`scripts/plots/` |
| `N4` | >= 2000 | `predictors/`、`models/`、`data/loaders/`、`scripts/` 残量 |
| `N5` | >= 1400 | 最终 wrapper、tests 同步瘦身、残余大文件收尾 |

- 任一阶段只要净减不足预算的 `70%`，就必须暂停并重排，不得机械推进。

## 阶段 N2：回收 N1 期间新增碎片，只打入口 support 与 utils 第一层

### 目标

- 先把 N1 为了快速过线而新增的 support 分裂重新收口。
- 把 `scripts/utils/` 从 “21 个文件 / 3151 行” 压到明显更少、更短，但**暂不**同时并行处理 `scripts/plots/` 与 `grid_notebook_workflow.py`。
- 让 mainline 入口成为真正的 owner 边界，而不是新的中转站。

### 主战场

- `scripts/mainline_madrl.py`
- `scripts/train.py`
- `scripts/mainline_mpc.py`
- `scripts/mainline_grid_analysis.py`
- `scripts/utils/mainline_setup.py`
- `scripts/utils/train_progress.py`
- `scripts/utils/train_runner_support.py`
- `scripts/utils/train_mainline_support.py`
- `scripts/utils/experiment_notebook_utils.py`
- `scripts/utils/project_paths.py`
- `scripts/utils/nested.py`
- `scripts/utils/rollout_package_utils.py`

### 阶段内 checkpoint

- `N2a`：
  - 先吃掉 4 个新增 support 文件的第一轮合并。
  - `PY_FILE_COUNT <= 89`
  - `scripts/utils/ <= 2700`
- `N2b`：
  - 再清掉入口纯壳与重复 setup / path / progress 分支。
  - `CURRENT <= 10700`
  - `PY_FILE_COUNT <= 86`

### 本阶段硬要求

- `CURRENT <= 10700`
- `PY_FILE_COUNT <= 86`
- `scripts/ <= 3900`
- `scripts/utils/ <= 2200`
- `scripts/utils/` 文件数 `<= 14`
- `scripts/mainline_madrl.py` 与 `scripts/train.py` 不再依赖第二层 support 跳转
- `scripts/mainline_mpc.py` / `scripts/mainline_grid_analysis.py` 若保留，必须是唯一稳定入口，不能继续只是多跳转发壳

### 本阶段策略

- 优先减少文件数，而不是继续长辅助支持模块。
- 优先把只服务单一路径的 support 吸回主 owner。
- `grid_notebook_workflow.py` 和 `scripts/plots/` 延后到 `N3`，避免本阶段主战场并行过多。
- `scripts/utils/project_paths.py`（44 行）和 `scripts/utils/nested.py`（24 行）只承担文件数收口作用，不作为 `N2` 主要减量来源；`N2` 的主减量必须来自 train / mainline support 回收与纯壳删除。
- 本阶段改掉的 support 对应旧测试，同阶段一起收口。

### `N2` 净减量按文件列账（`>= 1764`）

| 文件 / 动作 | 当前 | N2 末 | 净减 | 备注 |
| ---- | ---: | ---: | ---: | ---- |
| 删除 `scripts/mainline_mpc.py` 或吸收成 owner | 4 | 0/+X | -4 或净 0 | 选 A 就直接删 |
| 删除 `scripts/mainline_grid_analysis.py` 或吸收成 owner | 4 | 0/+X | -4 或净 0 | 选 A 就直接删 |
| 删除 `scripts/utils/misocp_notebook_diagnostics.py` | 150 | 0 | -150 | diagnostics helper 不留 |
| 删除 `scripts/utils/experiment_notebook_utils.py` | 71 | 0 | -71 | 吸回 mainline |
| 删除 `scripts/utils/nested.py` | 24 | 0 | -24 | 吸回调用点 |
| 删除 `scripts/utils/grid_surrogate_notebook_helpers.py` | 38 | 0 | -38 | notebook 专属 helper 不留 |
| 吸收 `scripts/utils/mainline_setup.py` 回 `mainline_madrl.py` | 43 | 0 | -43 | 收 ownership |
| 吸收 `scripts/utils/train_progress.py` 回 `scripts/train.py` | 44 | 0 | -44 | 收 ownership |
| 压缩 `scripts/utils/train_runner_support.py` | 181 | 90 | -91 | 主体吸回，残留保留 |
| 压缩 `scripts/utils/train_mainline_support.py` | 167 | 80 | -87 | 同上 |
| 压缩 `scripts/utils/rollout_package_utils.py` | 170 | 60 | -110 | 吸回 mainline compare / artifacts |
| 压缩 `scripts/utils/misocp_plan_packages.py` | 154 | 40 | -114 | 吸回 misocp_notebook_helpers |
| 压缩 `scripts/utils/admm_mpc_notebook_helpers.py` | 127 | 60 | -67 | 吸回 admm_mpc_solver |
| 压缩 `scripts/utils/admm_mpc_rollout_packages.py` | 101 | 40 | -61 | 吸回 admm_mpc_solver / mainline |
| 压缩 `scripts/utils/local_mpc_rollout_packages.py` | 69 | 20 | -49 | 吸回 mainline |
| 压缩 `scripts/utils/torch_runtime.py` | 185 | 110 | -75 | 去掉不再用的 helper |
| 压缩 `scripts/utils/project_paths.py` | 44 | 20 | -24 | 删兼容导入路径 |
| `scripts/mainline_madrl.py` 吸回 support 后净变 | 171 | 120 | -51 | 吸收 setup 但删去重复 support wrapper |
| `scripts/train.py` 吸回 support 后净变 | 189 | 150 | -39 | 吸收 progress 后删 wrapper |
| `scripts/mainline_artifacts.py` 精简 | 250 | 150 | -100 | 去 artifact orchestration 兼容层 |
| `scripts/mainline_compare.py` 精简 | 174 | 120 | -54 | 去 compare orchestration 冗余 |
| `scripts/checkpoints.py` 并入 `scripts/train.py` | 120 | 0 | -120 | 主入口统一 |
| `scripts/builder.py` 小幅精简 | 145 | 100 | -45 | 去未用分支 |
| 合计预计 | | | **-1775** | 比 `>= 1764` 目标多 `11` 行安全余量 |

- 以上表格是启动 `N2` 前必须完成的 **"净减量来源清单"**；执行中某一行净减不达预期时，必须补另一行，不允许总额失配。
- 该清单只依赖 `scripts/` 内部的真实删除和吸收，不要求从 `controllers/` / `envs/` / `predictors/` 提前借位，保证 `N2` 边界纯净。

## 阶段 N3：压扁 controller / mpc / env 主体，并处理 workflow/plots

### 目标

- 这是总减量的最大主战场，必须从最大、最厚、最危险的文件里兑现减量。
- 同时把上一阶段故意延后的 `grid_notebook_workflow.py` 与 `scripts/plots/` 一并收掉，避免 `scripts/` 尾部长拖。

### 主战场

- `controllers/mpc/global_socp_mpc.py`
- `controllers/mpc/gurobi_agent_mpc.py`
- `controllers/action_feasibility.py`
- `controllers/madrl/safety_projector.py`
- `controllers/madrl/base_agent.py`
- `envs/grid_env.py`
- `envs/subproc_vec_env.py`
- `envs/vec_env.py`
- `envs/observation/*`
- `scripts/utils/grid_notebook_workflow.py`
- `scripts/utils/misocp_notebook_helpers.py`
- `scripts/utils/admm_mpc_solver.py`
- `scripts/plots/grid_notebook_plotting.py`

### 阶段内 checkpoint

- `N3a`：
  - 先只打 `controllers/mpc/global_socp_mpc.py`、`scripts/utils/grid_notebook_workflow.py`、`scripts/plots/grid_notebook_plotting.py`
  - `CURRENT <= 9700`
  - `controllers/mpc/global_socp_mpc.py <= 1000`
  - `scripts/utils/grid_notebook_workflow.py <= 400`
  - `scripts/plots/grid_notebook_plotting.py <= 360`
- `N3b`：
  - 再推进 `envs/`、`controllers/madrl/`、`controllers/mpc/gurobi_agent_mpc.py` 的结构收口
  - `CURRENT <= 8400`
  - `PY_FILE_COUNT <= 72`

### `scripts/plots/grid_notebook_plotting.py` 必删路径清单

- 删除 MISOCP-专属诊断/比较绘图，只保留当前主线使用的公共面板：
  - MISOCP 状态对比、refinement tier 可视化
  - ADMM 收敛轨迹 debug panel
  - per-chunk summary 可视化
- 删除重复的 dataframe reshape / pivot helper，统一到单个 `_to_panel_df` 纯函数：
  - `_reshape_*_panels`、`_pivot_for_*`、`_build_bus_frame` 一类的多版本 helper 收敛为一条路径。
- 删除只为过期 artifact schema 服务的绘图入口（旧 key、旧包名、旧目录约定）。
- `grid_notebook_plotting.py` 终局必须只做：加载已落盘的 result package → 统一 dataframe → 绘图。**不允许**在这里做 solver 再跑、artifact 清理、schema 判定。
- 合并 `scripts/plots/rollout_plots.py` 与 `scripts/plots/reward_plots.py` 到同一个 `scripts/plots/shared_plots.py`（或直接并入 `grid_notebook_plotting.py`），`scripts/plots/` 终局 `<= 2` 个文件。
- **不允许**为绘图新建 `scripts/plots/_helpers/` 之类的子树。

### `controllers/mpc/global_socp_mpc.py` 必删路径清单

- 删除 debug/export 侧链：
  - `_maybe_export_debug_artifacts`
  - `_sanitize_debug_tag`
  - `debug_artifacts`
  - controller 的 `export_debug` / `debug_tag_prefix`
- 删除 retry / fallback / chunk telemetry 侧链：
  - `default_retry_solve_config`
  - `_run_window_with_retry`
  - `_make_chunk_summary`
  - `_concatenate_chunk_results` 里只为 retry / fallback 服务的聚合字段
  - `chunk_retry_count`
  - `no_retry_or_fallback_used`
  - `chunk_summaries`
- 删除 physics refinement 的多层梯度与高诊断开销路径：
  - `physics_refinement_*_schedule`
  - aggressive third tier 相关配置
  - `physics_refinement_attempt_caps_eur`
  - `floor_accepted_tier`
  - `used_physics_refinement_tier`
  - `refinement_status_counts`
- 同步删除只为这些路径服务的 `MISOCPResult` 字段和 diagnostics 输出。
- **不允许**把这些逻辑简单搬去新的 helper 文件里，只能删或并入单一路径。

### 本阶段硬要求

- `CURRENT <= 8400`
- `PY_FILE_COUNT <= 72`
- `controllers/ <= 1900`
- `envs/ <= 1200`
- `envs/grid/ <= 250`
- `envs/observation/ <= 300`
- `envs/subproc_vec_env.py + envs/vec_env.py + envs/parallel_episode_sampling.py <= 300`
- `scripts/utils/ <= 1500`
- `scripts/plots/ <= 320`
- `controllers/mpc/global_socp_mpc.py <= 900`
- `controllers/mpc/gurobi_agent_mpc.py <= 300`
- `envs/grid_env.py <= 320`
- `scripts/utils/grid_notebook_workflow.py <= 300`
- `scripts/utils/admm_mpc_solver.py <= 280`
- `scripts/utils/misocp_notebook_helpers.py <= 280`
- `scripts/plots/grid_notebook_plotting.py <= 260`

### 本阶段策略

- 不再保留“未来也许支持另一类实验”的代码。
- 同类约束、诊断、安全逻辑只保留一条路径。
- `GridEnv` 不再承担历史缓存、旧观测、兼容解析职责。
- `grid_notebook_workflow.py` 必须继续收口为 notebook 的单跳工作流，而不是 notebook 专属 orchestrator 森林。

## 阶段 N4：predictor / models / data / notebook 配套代码收口

### 目标

- 把 forecast、shared_data、predictor 训练链路和 notebook 配套 `.py` 压到接近终局。
- 同时收掉 `models/` 的抽象壳层，让模型装配回到更少文件、更清晰 owner。

### 主战场

- `predictors/training.py`
- `predictors/lstm_forecaster.py`
- `predictors/shared_data.py`
- `predictors/mainline_forecast.py`
- `data/loaders/prosumer.py`
- `configs/experiment_config.py`
- `models/assembly.py`
- `models/family_adapters.py`
- `models/utils.py`
- `scripts/mainline_compare.py`
- `scripts/mainline_artifacts.py`
- `scripts/utils/` 剩余文件

### `predictors/training.py` 必删路径清单

- 合并重复的数据读取路径，只保留一个 canonical source loader：
  - `_load_signal_matrix_from_source`
  - `load_signal_matrix_from_source`
  - `_load_signal_segments_from_source`
  - `_load_signal_segment_frames_from_source`
- 合并重复的 split / window builder，只保留一个 canonical builder：
  - `_temporal_split_items`
  - `temporal_split_segment_frames`
  - `temporal_split_segments`
  - `build_supervised_windows_from_matrix`
  - `build_supervised_windows_from_segments`
  - `build_supervised_windows_from_time_feature_segments`
- 压缩 artifact inventory / validation orchestration：
  - `expected_lstm_artifact_meta`
  - `compare_lstm_artifact_meta`
  - `validate_lstm_artifact`
  - `_collect_lstm_artifact_inventory`
  - `_raise_lstm_artifact_requirements_error`
  - `collect_available_lstm_artifacts`
  - `ensure_lstm_artifacts`
- 删除只为多模式泛化服务的分支，按当前主线配置保留单一路径：
  - `component_split`
  - `_build_signal_training_specs` 里的多组件展开
  - `_select_load_blend_weight_from_validation`
  - `_select_heatpump_blocked_blend_weight`
- **不允许**通过新建 `predictors/support_*.py` 来转移复杂度。

### `models/` 必删路径清单

- 把 `models/assembly.py`、`models/family_adapters.py`、`models/utils.py` 收敛到更少文件。
- `family` 抽象如果只剩 `mlp` 一条主线，就不再单独保留“family”层。
- `flatten_*` 这类只服务当前 retained mainline 的工具，要么并回 assembly owner，要么并回 retained adapter owner。
- `encoders/mlp_encoder.py`、`heads/actor_head.py`、`heads/critic_head.py` 合计仅 61 行，`N4` 直接并回 retained owner，不再保留独立判定空间。
- `models/` 终局目标是：
  - `<= 3` 个文件
  - `<= 140` 行
  - 不保留仅用于转发的抽象壳层

### `configs/experiment_config.py` 必删路径清单

- 删除不再被六条主线引用的备用参数块、重复窗口、重复路径常量。
- 删除只为旧 notebook、旧 key、旧 artifact 路径兼容服务的镜像字段。
- 把重复出现的 profile / bus / date window / artifact path 组收敛为单一来源，不再在多个配置段重复写死。
- 对真正只是“当前主线一个固定值”的字段，直接收成单值配置，不再保留伪通用参数分层。

### 阶段内 checkpoint

- `N4a`：
  - `predictors/training.py <= 680`
  - `models/` 文件数 `<= 4`
  - `predictors/ <= 1200`
- `N4b`：
  - `CURRENT <= 6400`
  - `PY_FILE_COUNT <= 61`
  - `predictors/training.py <= 530`
  - `predictors/shared_data.py <= 150`

### 本阶段硬要求

- `CURRENT <= 6400`
- `PY_FILE_COUNT <= 61`
- `predictors/ <= 980`
- `predictors/` 文件数 `<= 7`
- `models/ <= 140`
- `models/` 文件数 `<= 3`
- `configs/ <= 260`
- `data/ <= 200`
- `scripts/ <= 1800`
- `configs/experiment_config.py <= 200`
- `predictors/training.py <= 530`
- `predictors/lstm_forecaster.py <= 260`
- `predictors/shared_data.py <= 150`
- `controllers/action_feasibility.py` 已不存在（并入 `controllers/madrl/safety_projector.py`）
- `scripts/mainline_mpc.py` 与 `scripts/mainline_grid_analysis.py` 已在 `N2` 结束前二选一处置，`N4` 末不得以任何形式重新出现为纯壳

### 本阶段策略

- `predictors/training.py` 必须通过删重复路径和删多模式分支来变薄，不允许靠拆 support。
- notebook 配套逻辑要么进 canonical owner，要么删除，不再挂在 utils 森林里。
- compare / artifact / forecast 只保留当前主线协议，不保留旧包名兼容层。

## 阶段 N5：最终收尾与全量验收

### 目标

- 从 `6400` 打到 **`<= 5000`**。
- 让仓库进入真正终局状态：
  - 代码纯
  - 文件少
  - 没有壳层
  - 没有兼容债
  - 没有只是为了“显得模块化”而存在的杂乱 support 树

### 启动前门槛

- 必须列出当前剩余全部 `>= 200` 行文件。
- 对每个文件逐个标注：
  - `删`
  - `压到 X`
  - `保留`
- 必须按文件列出并加总，证明剩余可减量 `>= 1500` 行。
- 其中 `1500` 相比 `N5` 计划净减 `1400` 多出的 `100` 行，是专门预留给最后一公里反弹与统计误差的安全带。
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
- `envs/subproc_vec_env.py + envs/vec_env.py + envs/parallel_episode_sampling.py <= 240`
- `predictors/ <= 900`
- `models/ <= 140`
- `TESTS_LINES <= 9000`
- 命中旧 schema、旧 key、旧缓存包、旧 artifact 名称时统一直接失败
- `tests/` 只覆盖当前主线，不再维护兼容层
- 若终局仍存在 `> 800` 的文件，该文件必须是明确的核心 owner，且不得含有 wrapper / debug export / compatibility 负担

## 当前最优先的 14 个目标文件

1. `controllers/mpc/global_socp_mpc.py`
2. `predictors/training.py`
3. `scripts/utils/grid_notebook_workflow.py`
4. `scripts/utils/admm_mpc_solver.py`
5. `scripts/plots/grid_notebook_plotting.py`
6. `scripts/utils/misocp_notebook_helpers.py`
7. `predictors/lstm_forecaster.py`
8. `controllers/mpc/gurobi_agent_mpc.py`
9. `envs/grid_env.py`
10. `controllers/madrl/safety_projector.py`
11. `configs/experiment_config.py`
12. `predictors/shared_data.py`
13. `scripts/mainline_artifacts.py`
14. `data/loaders/prosumer.py`

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

### `N2`

- `tests/test_train_mainline_launcher.py`
- `tests/test_train_progress.py`
- `tests/test_action_dtype_casts.py`
- `scripts/mainline_madrl.py` smoke
- `scripts/train.py` smoke

### `N3`

- `tests/test_grid_env.py`
- `tests/test_parallel_episode_sampling.py`
- `tests/test_local_mpc_notebook_helpers.py`
- `tests/test_misocp_global_notebook.py`
- `tests/test_grid_notebook_workflow.py`
- controller/mpc 主线 smoke

### `N4`

- `tests/test_forecast_scaling.py`
- `tests/test_prosumer_dataset.py`
- `tests/test_model_assembly.py`
- `tests/test_builder_vec_env.py`
- predictor / forecast / compare smoke

### `N5`

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
