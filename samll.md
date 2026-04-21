# MADRL_ESS 压缩与清理计划

## 目标

在不动当前最新训练结果和当前最新计算结果的前提下，完成一次偏“主线收口”的压缩整理：

- 允许彻底删除旧 artifact
- 允许彻底删除旧 reward key 兼容
- 允许彻底删除旧 notebook 兼容层
- 允许彻底删除事实上的死代码（无 import、只有自家测试调用的模块）
- 允许在不改根目录结构、也不改现有 `.ipynb` 文件结构的前提下，重组内部目录和模块拆分

这次整理的核心原则不是“尽量兼容”，而是“先保护当前主线结果，再删掉历史包袱”。

### 量化目标

- 当前非测试 Python 总量：约 34,850 行
- 本次压缩落点目标：**16,000 ~ 17,000 行**（除 `tests/` 外）
- 即削减 ~18,000 行，约 51%
- 测试代码也跟随主线模块的删除一起精简，但不在 16k 计算内

## 不可突破的边界

- 不改根目录一级结构，保留当前的 `artifacts/`、`configs/`、`controllers/`、`envs/`、`models/`、`notebooks/`、`predictors/`、`scripts/`、`tests/` 等入口。
- 不改现有 notebook 文件本身的结构，不改 cell 顺序，不改 notebook 路径。
- **三套 MPC 主线全部保留**：`local MPC`、`ADMM MPC`、`MISOCP` 三条计算流水线及其 notebook 入口都必须保持可运行，不得整体删除任一条。
- 当前最新训练产物、当前最新计算产物必须先冻结、建档、可回读，然后才允许删除旧东西。
- 任何删除动作都必须先有“保留清单”和“待删清单”，不能直接凭感觉清。

## 这次采用的标准

### 1. 只保留当前主线接口

reward 只保留下面这些正式键：

- `w_soc_pen`
- `w_voltage_pen`
- `w_line_pen`
- `w_trafo_pen`
- `export_subsidy_eur_per_kwh`
- `import_price_markup_eur_per_kwh`

forecast / artifact 只保留当前主线语义：

- 信号名只保留 `wholesale_price`、`load`、`pv`
- forecast artifact 根目录只认 `artifacts/forecast/lstm/`
- training artifact 根目录只认 `artifacts/training/checkpoints/`、`artifacts/training/shared_data/`、`artifacts/training/tensorboard/`

### 2. 旧兼容一律按“失败即重建”处理

只要命中旧 schema、旧 key、旧缓存包、旧 artifact 命名，不再做温和兼容或自动兜底，而是：

- 明确报错
- 指向当前主线路径
- 要求重建当前 artifact / cache / shared data

### 3. notebook 只保留“当前结构 + 当前主线依赖”

notebook 自身不改结构，但 notebook 背后的 Python 依赖允许收口。也就是说：

- 可以改 notebook import 到的 Python 实现
- 不能靠修改 notebook cell 结构来绕过整理工作

## 已识别的主线包袱点

当前仓库里最值得优先收口的点（行数为本次扫描的实际值）：

| 文件 | 当前行数 | 处理策略 | 目标行数 |
| ---- | ---- | ---- | ---- |
| `scripts/utils/forecast_diagnostics.py` | 2180 | **整体删除**（事实上死代码，无任何 notebook / 主线 import） | 0 |
| `scripts/utils/grid_notebook_workflow.py` | 4080 | 拆分 + 删兼容层 | ~1500 |
| `predictors/training.py` | 3184 | 拆分 + 删 legacy schema 分支 | ~1800 |
| `scripts/utils/misocp_notebook_helpers.py` | 3058 | 删 latest-compatible + 收画图/诊断 | ~1500 |
| `scripts/utils/admm_mpc_notebook_helpers.py` | 1818 | 删 latest-compatible + 收画图/诊断 | ~1000 |
| `predictors/lstm_forecaster.py` | 1121 | 删 legacy attribute 同步 | ~700 |
| `scripts/utils/local_mpc_notebook_helpers.py` | 374 | 删 latest-compatible | ~250 |
| `controllers/mpc/global_socp_mpc.py` | 3362 | **保留**（MISOCP 求解器主体），仅做边角清理 | ~3200 |
| `controllers/mpc/gurobi_agent_mpc.py` | 1032 | **保留**（local/ADMM 主体），仅做边角清理 | ~1000 |
| `scripts/train.py` | 819 | 边角清理 | ~750 |
| `envs/grid_env.py` | 807 | 边角清理 | ~750 |
| `scripts/utils/madrl_shared_data.py` | 799 | 删 legacy schema 兜底 | ~500 |
| `scripts/run_train_mainline.py` | 482 | 删旧 reward key 兼容 | ~400 |
| `envs/rewards/NormalReward.py` | (待测) | 删 `w_action_pen -> w_soc_pen` 回退 | 收紧 |

`forecast_diagnostics.py` 已被本次扫描确认：除 `tests/test_forecast_diagnostics.py` 外，无任何文件 import 它的任何符号，整个 notebook 目录也未引用其公开 API（如 `collect_load_step1_diagnostics`、`run_blocked_blend_experiment` 等）。视为本次清理的“低风险大头”。

notebook helper 里最典型的历史兼容是：

- `resolve_latest_compatible_local_mpc_rollout_package_dir`
- `resolve_latest_compatible_admm_mpc_rollout_package_dir`
- `resolve_latest_compatible_misocp_plan_package_dir`

这类“按前缀找最新兼容包”的逻辑，是本次清理优先要移除的对象。

## 分阶段执行计划

### 阶段 0：冻结基线，保护最新结果

目标：先把“绝对不能误伤的结果”从混乱目录里标出来。

要做的事：

- 扫描并登记当前最新 forecast artifact、shared data、checkpoint、notebook cache package。
- 生成一份保护清单，至少覆盖：
  - 当前主线训练入口对应的最新 checkpoint
  - 当前主线 shared data 的最新 signature 目录
  - 当前主线 forecast artifact 的最新可用版本
  - 当前 notebook 仍在使用的最新 cache/package
- 为保护清单生成 manifest 或 hash 摘要，确保后面能校验“没被动过”。
- 在真正清理前，先做一次 dry-run 清单输出：哪些保留，哪些待删。

完成标准：

- 我们能明确回答“现在仓库里哪些结果必须保，哪些只是历史残留”。

### 阶段 0.5：删除事实死代码（forecast_diagnostics）

目标：在动主线之前，先把零依赖的死代码移走，立刻拿到 ~2400 行的削减空间。

要做的事：

- 再次确认 `scripts/utils/forecast_diagnostics.py` 的所有公开符号在 `notebooks/`、`scripts/`、`predictors/`、`controllers/`、`envs/`、`models/`、`data/`、`tools/` 下没有任何引用。
- 删除文件 `scripts/utils/forecast_diagnostics.py`（约 2180 行）。
- 删除测试 `tests/test_forecast_diagnostics.py`（约 166 行）。
- 检查并清理 `tests/conftest.py`、`pytest.ini`、CI 配置中是否对该测试有显式引用，有则一并删。
- 在 git commit message 中写明：本模块为预测调参期遗留实验脚手架，主线无引用，整体删除；如需恢复请回 git 历史。

完成标准：

- 仓库内无任何文件 import `forecast_diagnostics` 中的符号。
- 测试集中无 `test_forecast_diagnostics` 相关用例。
- `pytest` 在删除后仍能正常 collect 和运行其他测试。

预期削减：~2180 行（主线）+ 166 行（测试）。

### 阶段 1：先砍旧 reward key 兼容

目标：把 reward 输入面收紧到当前主线，停止接受旧字段。

重点处理文件：

- `envs/rewards/NormalReward.py`
- `scripts/run_train_mainline.py`
- `scripts/utils/grid_notebook_workflow.py`
- 对应测试 `tests/test_train_mainline_launcher.py`、`tests/test_normal_reward.py`、`tests/test_registries.py`

计划动作：

- 删除 `w_action_pen` 兼容入口，只接受 `w_soc_pen`
- 删除 `lambda_throughput` 的忽略分支，改为显式拒绝未知旧键
- 清掉 reward 相关的旧别名测试，改成严格主线测试

完成标准：

- 主线代码不再接受旧 reward key
- 所有 reward 参数入口行为一致
- 遇到旧 key 会明确报错，而不是悄悄吞掉

### 阶段 2：砍掉旧 notebook cache 兼容层

目标：不再帮旧 notebook 缓存“自动找一个差不多能用的包”，改为精确匹配当前 schema。

重点处理文件：

- `scripts/utils/local_mpc_notebook_helpers.py`
- `scripts/utils/admm_mpc_notebook_helpers.py`
- `scripts/utils/misocp_notebook_helpers.py`
- 对应 notebook 测试和 helper 测试

计划动作：

- 删除 `resolve_latest_compatible_*` 这类“兼容搜索”函数
- 改为“精确路径 + 精确 signature/schema 校验”
- 命中旧缓存时直接失败，并提示重新生成当前 cache
- 同步改写测试，不再验证“兼容候选回退”，而是验证“严格失败 + 重新生成提示”

完成标准：

- 当前 notebook 仍能按现有结构运行
- 旧 cache 不再被偷偷复用
- notebook helper 行为从“模糊兼容”变成“严格校验”

### 阶段 3：砍掉旧 artifact / 旧 schema 兼容

目标：把 artifact 管理从“兼容历史”切到“只认当前主线格式”。

重点处理文件：

- `predictors/lstm_forecaster.py`
- `predictors/training.py`
- `scripts/utils/madrl_shared_data.py`
- `scripts/utils/cleanup_shared_data.py`
- `scripts/utils/price_protocol.py`

计划动作：

- 清理 legacy signal / legacy attribute / legacy schema 兜底分支
- 保留“识别旧 artifact 并拒绝加载”的能力，但删除“继续兼容跑”的逻辑
- shared data 清理脚本改成面向当前 schema，不再围绕旧 schema 继续兜底
- 对于当前仍必须读取的最新结果，如果它本身是旧格式，先重建，再删兼容代码

完成标准：

- 当前主线 artifact 的读写路径唯一且清晰
- 旧 artifact 只能触发报错和重建提示，不能继续混入主线

### 阶段 4：模块压缩和内部重组

目标：在不动 notebook 文件结构、不动根目录结构、且三套 MPC 主线都保留的前提下，压缩大文件，降低主线复杂度。

优先拆分对象（带量化目标）：

- `scripts/utils/grid_notebook_workflow.py`：4080 → ~1500
- `predictors/training.py`：3184 → ~1800
- `scripts/utils/misocp_notebook_helpers.py`：3058 → ~1500
- `scripts/utils/admm_mpc_notebook_helpers.py`：1818 → ~1000
- `predictors/lstm_forecaster.py`：1121 → ~700
- `scripts/utils/local_mpc_notebook_helpers.py`：374 → ~250
- `scripts/utils/madrl_shared_data.py`：799 → ~500

不在本阶段动结构，仅做边角清理的对象：

- `controllers/mpc/global_socp_mpc.py`（MISOCP 主体）
- `controllers/mpc/gurobi_agent_mpc.py`（local / ADMM 主体）
- `controllers/madrl/safety_projector.py`
- `envs/grid_env.py`、`scripts/train.py` 等核心运行时

建议拆分方向：

- `grid_notebook_workflow.py` 拆成控制参数、forecast 准备、rollout/compare、绘图/汇总几个内部模块
- `predictors/training.py` 拆成 artifact 校验、artifact inventory、单 signal 训练、主线 ensure 流程
- `misocp_notebook_helpers.py` 拆成加载、验证、绘图三块；删除“latest compatible”兼容搜索整块
- `admm_mpc_notebook_helpers.py` 同上：加载、验证、绘图三块；删兼容搜索
- `local_mpc_notebook_helpers.py` 删兼容搜索后基本只剩纯加载
- `lstm_forecaster.py` 删 legacy attribute 同步路径

执行原则：

- notebook-facing API 名称尽量先保持不变
- 真正的重排放在模块内部完成
- 必要时保留很薄的一层 facade，但不再承载历史兼容逻辑
- **三套 MPC 求解器实现保持算法完整**：不为追求行数砍掉数学正确性相关的分支

完成标准：

- 上述大文件全部达到目标行数 ±10% 以内
- notebook 入口不需要因为这次整理而改结构
- 三套 MPC 主线（local / ADMM / MISOCP）的 notebook 跑通验证集

### 阶段 5：artifact 目录清理

目标：在保护清单之外，彻底删掉历史残留。

当前高优先级待清对象已经很明显：

- `artifacts/misocp_debug/` 下的大量历史调试导出
- `artifacts/tmp_oracle_smoke/`
- `artifacts/tmp_oracle_smoke_2/`
- 旧的 `misocp_cached_plan/` 历史包
- 旧 schema 的 `shared_data/` 目录
- 非当前主线需要的旧 forecast artifact

执行方式：

- 先按“最新保留、其余候选删除”出清单
- 对每一类目录先做 dry-run
- 再做真正删除

完成标准：

- `artifacts/` 根结构还在，但内部明显收紧
- 当前主线需要的结果都还可加载
- 历史调试垃圾和旧 cache 被真正清掉

### 阶段 6：验证和收口

目标：确认这次是“清理成功”，不是“清理出一个看起来更干净但更脆的仓库”。

最低验证集合：

- `tests/test_train_mainline_launcher.py`
- `tests/test_normal_reward.py`
- `tests/test_registries.py`
- `tests/test_grid_notebook_workflow.py`
- `tests/test_local_mpc_notebook_helpers.py`
- `tests/test_admm_mpc_notebook_helpers.py`
- `tests/test_misocp_global_notebook.py`
- `tests/test_compare_notebook.py`
- `tests/test_madrl_notebook_smoke.py`
- `tests/test_notebook_utils.py`

额外验证：

- 保护清单里的最新 artifact / checkpoint / shared_data 重新加载一次
- notebook compile / smoke 测试必须通过
- 清理后的错误信息必须直接告诉用户“该重建什么”

完成标准：

- 主线训练、主线 notebook、主线 artifact 全部按当前标准可用
- 历史兼容已经不再悄悄参与运行

## 验收标准

这次整理完成后，应该满足以下结果：

- 当前最新训练结果和当前最新计算结果都还存在、可定位、可回读
- 根目录结构没动，现有 `.ipynb` 文件结构没动
- 三套 MPC 主线（local / ADMM / MISOCP）的 notebook 全部能跑
- `forecast_diagnostics.py` 及其测试已整体删除
- reward 旧 key 兼容被删干净
- notebook cache 的“latest compatible”回退逻辑被删干净
- artifact 旧 schema / 旧格式兼容被删到只剩“拒绝并提示重建”
- 关键超大模块完成第一轮拆分，主线更短、更清晰
- `artifacts/` 内部历史残留显著下降
- **非测试 Python 总行数落在 16,000 ~ 17,000 行区间**

### 行数账本（预期）

| 来源 | 削减量 |
| ---- | ---- |
| 删除 `forecast_diagnostics.py` | ~2180 |
| `grid_notebook_workflow.py` 拆分 + 删兼容 | ~2580 |
| `predictors/training.py` 拆分 + 删 legacy | ~1380 |
| `misocp_notebook_helpers.py` 收口 | ~1560 |
| `admm_mpc_notebook_helpers.py` 收口 | ~820 |
| `lstm_forecaster.py` 删 legacy | ~420 |
| `madrl_shared_data.py` 收口 | ~300 |
| 其他边角清理（reward / launcher / shared utils） | ~500 |
| **合计预期削减** | **~9,740** |
| 进一步拆分时连带的死代码清理（保守估计） | ~7,000 ~ 8,000 |
| **从 34,850 起目标落点** | **~17,000 ~ 18,000** |

如果一轮压缩后仍高于 17k，再启动一轮“二次精简”：聚焦在还没碰过的 helper、未使用的 plotting 分支、未引用的 utility 函数。

## 需要特别盯住的风险

- 如果当前 notebook 源码里硬编码依赖某些旧 helper 名称，第一轮不能硬删到 import 断裂，只能先变成薄 facade。
- 如果当前“最新结果”本身仍带旧 schema，那么顺序必须是“先重建最新结果，再删兼容”，不能反过来。
- `predictors/lstm_forecaster.py` 的 legacy attribute 同步逻辑是否还能被主线调用，需要先做调用面清点，再决定本轮是否彻底砍掉。

## 建议的实际落地顺序

按风险最小、收益最大的顺序推进，实际执行顺序是：

1. 冻结基线和保护清单（阶段 0）
2. **删除 `forecast_diagnostics.py` 及其测试**（阶段 0.5，零依赖一次性收益 ~2400 行）
3. 删 reward 旧 key 兼容（阶段 1）
4. 删 notebook cache 兼容搜索（阶段 2）
5. 删 artifact / schema 旧兼容（阶段 3）
6. 拆 `grid_notebook_workflow.py`、`predictors/training.py`、`misocp_notebook_helpers.py`、`admm_mpc_notebook_helpers.py`（阶段 4）
7. artifact 目录物理清理（阶段 5）
8. 三套 MPC notebook 跑通验证 + 行数核对，未达标则启动二次精简（阶段 6）

这个顺序的好处是：先把“零依赖死代码”一次性删干净，再把“行为边界”收紧，再做“代码体积压缩”，最后才做“物理删除”和验证，回退和定位都会更稳。
