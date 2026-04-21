# MADRL_ESS 第二轮极限收口计划（6500~7000 行版）

## 目标

在**不改根目录一级结构**的前提下，把项目收敛成一个只服务当前主线实验的极简仓库。

本轮最终验收目标调整为：

- `tests/` 之外的 `.py` 源码总量压到 **6500~7000 行**
- `tests/` 允许同步删除、合并、重写，不再为了旧 helper 结构保留测试壳
- 保留并可运行以下 notebook 文件：
  - `notebooks/forecast/forecast_lstm.ipynb`
  - `notebooks/madrl/global_MISOCP.ipynb`
  - `notebooks/madrl/local_MPC.ipynb`
  - `notebooks/madrl/ADMM_mpc.ipynb`
  - `notebooks/madrl/train_base.ipynb`
  - `notebooks/madrl/train_base_safe.ipynb`
  - `notebooks/madrl/train_projection_safe.ipynb`
  - `notebooks/madrl/compare.ipynb`
  - `notebooks/madrl/grid_network_analysis.ipynb`
- 六条主线必须保留：
  - `MISOCP`
  - `local MPC`
  - `ADMM MPC`
  - `MADRL base`
  - `MADRL base safe`
  - `MADRL projection safe`

可选 stretch goal：

- 如果在不伤主线可用性的前提下还能继续收缩，可以继续向 **5000 行**逼近
- 但 **5000 不再是本轮硬验收标准**
- 真正冲 5000 时，notebook 代码 cell 上限也要按同比例再下调一档，否则主口径会被逼着吃掉 solver 的数学正确性

## 现实约束

当前非测试 Python 总量约为 **33323 行**。目标改为 `6500~7000` 后，仍然意味着要删除约 **26300 行左右**，规模依然是一次架构级裁剪，而不是普通重构。

要做到这一步，必须接受下面这些现实前提：

- 仓库不再按“通用研究框架”维护，只服务当前这一套固定主线
- 不再保留任何“也许以后还会用”的兼容层、备用路径、自动兜底
- notebook 允许直接改代码，优先把 notebook 改薄，而不是保旧 helper API
- compare 必须尽早改成“只读结果包”，否则大量 helper 不敢删
- artifact 协议一旦统一，旧 artifact 必须立刻删，不再拖到最后

## 统计口径

本计划中“总代码量”定义为：

- **主口径**：`tests/` 之外所有 `.py` 文件总行数
- **次口径**：`.ipynb` 不按 JSON 原文计入主指标，否则在保留 9 个 notebook 文件的前提下目标失真
- **测试口径**：`tests/` 不计入 6500~7000 硬目标，但允许同步大幅删改

如果未来要求把 `.ipynb` 原文 JSON 也计入总代码量，那么在“所有文件都要保留”的前提下，应重新定义目标，而不是沿用本计划。

### 严格执行口径（新增硬规则）

本轮后续所有进度汇报、阶段验收、是否“代码变少了”的判断，一律只认下面这一套严格口径：

- **唯一硬指标**：仓库内 `tests/` 之外的全部 `.py` 文件当前总行数
- **基线值固定**：以计划启动时的非测试 `.py` 总量 **33323 行** 作为统一基线，后续所有阶段都只汇报“当前总量”和“相对 33323 的净变化”
- **新增文件必须计入**：任何新建的 `mainline_*.py`、共享工具、过渡包装、清理脚本，只要是 `.py`，无论是否已 `git add`，都必须立刻计入当前总量
- **删除后又搬回别处，不算减量**：如果删除旧文件、但把同一批逻辑以新文件或新共享层的形式重新写回仓库，总量没有下降，就**不得**宣称“已完成收缩”
- **局部 diff 不得冒充总减量**：`git diff --stat`、单文件 `-N +M`、删除了几个旧 helper，这些都只能作为辅助说明，**不能**替代总量对账
- **未跟踪文件必须纳入统计**：严格总行数必须按当前工作树扫描得到，而不是只看 `git diff HEAD`，因为未跟踪的新 `.py` 文件同样占用主口径
- **只有净总量下降，才算阶段有效**：任何阶段结束时，如果非测试 `.py` 当前总量没有实质下降，则该阶段视为“结构迁移”而非“收缩完成”
- **所有阶段汇报必须同时给出**：`BASELINE=33323`、`CURRENT=<当前值>`、`DELTA=<净变化>`，三者缺一不可
- **禁止模糊表述**：以后不允许再用“代码更集中”“结构更干净”“旧文件删了很多”替代严格总量结果；如果 `DELTA` 不够大，就必须诚实承认收缩效果不足
- **计数实现必须按 raw text 行数执行**：严格总行数必须使用“逐文件按原始文本逐行计数”的脚本口径，空白行也必须计入；`PowerShell Get-Content ... | Measure-Object -Line` 这类会漏算空白行的办法**禁止**再用于阶段验收

执行含义：

1. 先看仓库当前非测试 `.py` 总量有没有下降
2. 再看下降是否达到本阶段 checkpoint
3. 若未达标，优先继续删减或重做方案，而不是进入下一阶段

## 不可突破的边界

- 不改根目录一级结构，保留当前的 `artifacts/`、`configs/`、`controllers/`、`envs/`、`models/`、`notebooks/`、`predictors/`、`scripts/`、`tests/` 等入口
- notebook 文件路径必须保留
- 六条主线必须都能产出或重放结果
- `grid_network_analysis.ipynb`、`compare.ipynb`、`forecast_lstm.ipynb` 必须保留
- 旧 schema、旧 key、旧缓存包、旧 artifact 命名一旦命中，统一直接失败，不再温和兼容或自动兜底

## 本轮硬原则

### 1. 单场景优先

整个项目只服务一套固定主线场景：

- 唯一测试窗口：`2020-06-01` 到 `2020-06-07`
- 六条主线共用同一组 agent、reward、forecast、battery、grid、MPC 参数
- 所有共享参数只允许集中存在于 `configs/experiment_config.py`

### 2. 不兼容不兜底

只要命中旧 schema、旧 key、旧缓存包、旧 artifact 命名：

- 不做 fallback
- 不做 sibling 扫描
- 不做 latest-compatible 搜索
- 不做 prefix 模糊匹配
- 不做“能跑就先跑”的历史兼容

统一行为只有三步：

1. 直接报错
2. 明确指出命中的旧对象
3. 明确提示应该重新运行哪个 notebook 重建

### 3. notebook 只做薄入口

notebook 允许修改，但只允许承担三件事：

- 读取中心配置
- 调用单一主函数
- 展示结果或读回结果包

不再允许 notebook 内部保留大量：

- 重复参数块
- schema 兼容判断
- cache/package 搜索逻辑
- shared data 兜底生成逻辑
- 重算 compare 依赖的逻辑

### 4. 优先修改 notebook，而不是保 facade

上一轮为了平滑迁移保留了一些 facade。本轮原则改为：

- 优先直接改 notebook import
- 优先删除旧 helper 文件
- 除非 wrapper 少于 10 行且确实必要，否则不再保留 facade

### 5. 每阶段必须过行数 checkpoint

本轮不接受“先做到最后再看总行数”。

规则如下：

- 每个阶段结束都必须核对一次非测试 `.py` 总行数
- **未达到该阶段硬指标，不进入下一阶段**
- 如果某阶段偏离目标超过 10%，必须先重做计划，而不是继续往下删

## notebook 瘦身硬指标

本轮给 notebook 加硬约束，不再写“明显瘦身”这种模糊要求。

### 代码 cell 总量约束

- `forecast_lstm.ipynb`：代码 cell 合并后源码总量 **≤ 260 行**
- `compare.ipynb`：代码 cell 合并后源码总量 **≤ 240 行**
- `global_MISOCP.ipynb` / `local_MPC.ipynb` / `ADMM_mpc.ipynb`：每个 **≤ 180 行**
- `train_base.ipynb` / `train_base_safe.ipynb` / `train_projection_safe.ipynb`：每个 **≤ 180 行**
- `grid_network_analysis.ipynb`：**≤ 180 行**

### 结构约束

- 单个 notebook 代码 cell 数量尽量 **≤ 8**
- 任一 notebook 不允许再出现大段重复参数定义
- compare notebook 不允许直接包含 solver / trainer / shared-data 构建主逻辑

## 统一主线配置

所有共享参数统一收敛到 `configs/experiment_config.py`。

建议收敛出唯一入口，例如：

- `build_mainline_experiment_config()`
- `MAINLINE_COMPARE_WINDOW`
- `MAINLINE_AGENT_PROFILES`
- `MAINLINE_ARTIFACT_PATHS`
- `MAINLINE_REWARD`
- `MAINLINE_MPC`
- `MAINLINE_SAFE_POC`

这份配置必须覆盖：

- `agent_profiles`
- `agent_bus_ids`
- `test_start_date="2020-06-01"`
- `test_end_date="2020-06-07"`
- `load_scale`
- `pv_scale`
- `pv_capacity_kw`
- `future_horizon`
- `history_window`
- reward 六个正式 key
- `local MPC / ADMM / MISOCP` 所需参数
- `MADRL base / safe / projection safe` 所需训练与评估参数
- 所有主线 artifact 的精确路径

## 目标主线链路

### 结果生产链路

1. `forecast_lstm.ipynb`
2. 训练当前主线 forecast
3. **立即生成当前主线 shared_data**
4. `train_base.ipynb`
5. `train_base_safe.ipynb`
6. `train_projection_safe.ipynb`
7. `global_MISOCP.ipynb`
8. `local_MPC.ipynb`
9. `ADMM_mpc.ipynb`
10. 六条主线全部产出统一结果包
11. `compare.ipynb` 只读取六个结果包做对比

### 结果消费链路

- forecast notebook 只负责产出 forecast + shared_data
- 三个 MADRL notebook 只负责训练、评估、写包
- 三个 MPC notebook 只负责求解、评估、写包
- compare notebook 只负责读包和画图
- `grid_network_analysis.ipynb` 只负责读取当前主线 grid analysis 结果并展示

任何 notebook 都不再负责：

- 找最近兼容目录
- 在缺包时自己补算
- 自动切换旧路径
- 自动兼容旧 schema

### 结果保护与回滚

为保证”当前最新训练和计算结果不被动”，旧 artifact 的删除必须带保护清单和回读校验，不允许直接清库。

保护清单的边界：

- 保护清单是**新产出白名单**，只列当前主线真正依赖的产物
- 旧 artifact（历史命名、非主线实验、旧 schema 包）**默认不在清单、默认可删**
- 阶段 3 的删除对象就是”保护清单之外的一切旧物”，两者不重叠

硬规则：

- 阶段 2 末必须在 `artifacts/_protection/` 生成保护清单
- 保护清单至少覆盖：forecast artifact、forecast 导出数据、`shared_data`、六条主线结果包、compare 直接消费的 manifest / 签名文件、三个 MADRL 主线**各自最新一套可运行训练包**（即当前 `model_root` 目录及其算法子目录、`_meta` 元数据、训练结果摘要等运行所需文件，而不是只保单个权重点）、`grid_network_analysis.ipynb` 当前主线分析结果
- 三条 MADRL 主线的**历史旧 run 目录和历史中间步数 checkpoint**不在保护清单内，阶段 3 允许清理，不允许以”以后也许会用”为由保留
- 保护清单必须记录精确路径、产物类型、所属主线、schema 版本、必要的完整性信息
- 阶段 3 只能删除不在保护清单中的旧 artifact
- 阶段 3 的删除必须先移动到 `artifacts/_trash/` 隔离区，不允许第一步就物理不可恢复删除
- 阶段 3 启动前，必须完成一次”按保护清单逐项回读”并验证通过
- 保护 / 回读 / 隔离区管理的 `.py` 代码应优先收敛进 `mainline_artifacts.py`，并尽量压在其建议体量 350 行附近；但无论最终写在哪个文件，都必须**诚实计入阶段 2 的非测试 `.py` 总行数**，再由同阶段删减去抵消

回读”验证通过”的判据（全部满足才算通过）：

- 清单中每一项都能被读取而不抛异常
- schema 字段齐全，与 `mainline_artifacts.py` 声明的当前版本匹配
- 签名 / 版本号匹配，不允许出现旧 schema 兼容回退
- 六条主线结果包能够被 `compare.ipynb` 组合消费一次（compare 能生成完整对比表格，不缺包）
- `forecast_lstm.ipynb -> shared_data -> MADRL train/eval -> result package` 一条龙可以基于保护清单中的产物完成至少一次 smoke
- `grid_network_analysis.ipynb` 能基于保护清单中的当前主线结果成功读取或生成一次 grid analysis 展示数据
- 三条 MADRL 主线各自最新一套可运行训练包都可被定位，并完成至少一次加载 smoke（历史中间 checkpoint 不参与验证）

固定顺序：

1. 统一新结果包协议
2. 生成保护清单
3. 对保护对象做回读验证
4. 将待删旧 artifact 移入 `artifacts/_trash/` 隔离区并生成删除批次清单
5. 基于”只剩保护对象 + 新主线代码”的状态完成一次删除后 smoke
6. smoke 通过后再物理删除隔离区内容

隔离区物理删除的时间窗口：

- 阶段 3 末尾必须完成物理删除，`artifacts/_trash/` 清空
- 阶段 4 启动前，`artifacts/_trash/` 不允许有残留
- 不允许把隔离区物理删除推迟到阶段 5 或阶段 6，避免长期占磁盘和”旧物还没真的消失”的心理负担

保护清单在阶段 4 / 阶段 5 的维护规则：

- 如果 `mainline_*.py` 的重构**没有改变任何结果包 schema**，保护清单沿用阶段 2 版本，不必重做
- 如果 `mainline_*.py` 的重构**改变了任一结果包的 schema、签名格式或路径布局**，必须：
  1. 重新产出受影响的主线结果包
  2. 重新生成保护清单
  3. 重新完成一次按新清单的全量回读验证
- 未完成上述三步前，不允许继续删除任何被影响的旧 artifact 或旧兼容代码

回读验证失败时的回滚路径：

- **中止阶段 3**，不允许以”部分通过”为由继续删除
- 回到阶段 2，找出回读失败的具体项并重产相应结果包
- 重产完成后重新生成 / 更新保护清单，再次完成全量回读
- 只有全量回读一次性通过，才允许进入阶段 3
- 这条回滚路径允许在阶段 2 与回读验证之间反复循环

阶段 3 删除后 smoke 失败时的回滚路径：

- 不做物理删除，直接从 `artifacts/_trash/` 隔离区原样恢复
- 恢复后重新跑 compare / grid analysis / MADRL checkpoint 加载 smoke，直到恢复后的状态重新可用
- 找出误删规则或保护清单漏项，修正后重新生成保护清单和删除批次清单
- 只有删除后 smoke 一次性通过，才允许清空本批次隔离区

## 主线文件职责边界

为避免动作 D 与动作 F 冲突，本轮先明确 `mainline_*.py` 的职责边界。

### `mainline_*.py` 骨架的时序约定

本计划里 `mainline_*.py` 分两步登场，不要把它们一次性堆到阶段 4：

- **阶段 1 建骨架**：创建 `mainline_forecast.py / mainline_artifacts.py / mainline_madrl.py / mainline_mpc.py / mainline_compare.py / mainline_grid_analysis.py` 的空壳或薄包装，把现有主函数直接搬进来、不做删减。目的是让 notebook 在阶段 1 就能把 import 切到最终路径，后续阶段只改内部实现、不再动 notebook import。
- **阶段 2 实装**：`mainline_artifacts.py` 必须在阶段 2 末尾真正承担结果包的写入、读取、manifest 校验，否则阶段 3 "删旧兼容层" 没有替代品可用。
- **阶段 4 收缩**：骨架里的内部逻辑被大规模合并与删除，最终落到职责边界下方给出的建议体量。

规则：notebook 的 import 在阶段 1 定型，之后不再改。所有内部重构都发生在 `mainline_*.py` 的实现里。

### `predictors/mainline_forecast.py`

职责：

- 训练 `wholesale_price/load/pv` 三个主线 forecast
- 校验当前主线 artifact
- 生成当前主线 `shared_data`

不负责：

- MADRL 训练
- MPC 求解
- compare
- 结果图表汇总

建议体量：**≤ 700 行**

### `scripts/mainline_artifacts.py`

职责：

- 统一定义主线 artifact 路径
- 统一写入和读取结果包 manifest
- 校验结果包 schema 和签名

不负责：

- 训练
- 求解
- compare 业务逻辑

建议体量：**≤ 350 行**

### `scripts/mainline_madrl.py`

职责：

- 执行 `base/base_safe/projection_safe` 三条 MADRL 主线
- 触发统一评估
- 写出结果包

不负责：

- forecast 训练
- shared_data 生成
- compare 画图
- solver 数学逻辑

建议体量：**≤ 750 行**

### `scripts/mainline_mpc.py`

职责：

- 分发 `MISOCP/local MPC/ADMM MPC` 三条主线执行
- 调用对应 solver
- 写出统一结果包

不负责：

- compare
- notebook 展示
- artifact 清库

建议体量：**≤ 550 行**

### `scripts/mainline_compare.py`

职责：

- 读取六个结果包
- 生成统一对比表格
- 生成最少必要图表

不负责：

- 重算任何主线
- shared_data 构建
- 目录扫描

建议体量：**≤ 450 行**

### `scripts/mainline_grid_analysis.py`

职责：

- 作为 `grid_network_analysis.ipynb` 的唯一脚本入口
- 读取中心配置
- 生成或读取当前主线 grid analysis 结果
- 输出最少必要展示数据

不负责：

- compare 汇总
- 六条主线结果包聚合
- 历史 notebook helper 兼容
- latest-compatible / 自动兜底逻辑

建议体量：**≤ 250 行**

## 允许彻底删除的对象

- 旧 artifact
- 旧 reward key 兼容
- 旧 notebook 兼容层
- 无 import 的死代码
- 只有自家测试调用、主线完全不 import 的模块
- 只服务历史实验的 plotting / diagnostics / report helper
- `resolve_latest_compatible_*`
- prefix-search / latest-compatible / fuzzy-match 逻辑

## 第二轮关键动作

### 动作 A：把整个项目切换成单配置驱动

目标：

- 所有 notebook 共用同一份参数
- 删除 notebook 内部重复参数块
- 清空与当前主线无关的 `profiles.py` 变体

### 动作 B：forecast 前移并接管 shared data 生成

目标：

- `forecast_lstm.ipynb` 在训练完成后直接生成当前主线训练所需 `shared_data`
- MADRL notebook 不再负责 shared data 构建

### 动作 C：统一六条主线结果包，尽早释放 compare 简化红利

目标：

- 六条主线统一结果包协议
- `compare.ipynb` 只读结果包
- compare 相关 helper 立刻开始删除

这一步必须前移，不再拖到后面。

### 动作 D：删除通用 helper 思维，改成少量主线入口

目标：

- 从大量 `scripts/utils/*` 切换到少量 `mainline_*.py`
- helper 只保留真有必要的底层函数

### 动作 E：solver 只保留当前主线数学路径

目标：

- `gurobi_agent_mpc.py` 只保留当前 local/ADMM 实际用到的 formulation
- `global_socp_mpc.py` 只保留当前 MISOCP 实际用到的 formulation

### 动作 F：MADRL 训练链路收敛成一条最短流程

目标：

- 不再由 `scripts/train.py`、`scripts/run_train_mainline.py`、`predictors/training.py` 各承担一层流程
- 训练主流程收敛到 `scripts/mainline_madrl.py`
- `scripts/train.py` 与 `run_train_mainline.py` 要么删除，要么降为薄 wrapper

### 动作 G：旧 artifact 在协议稳定后立即删除

目标：

- 一旦阶段 2/3 稳定了统一结果包协议，并且保护清单生成与回读校验通过，立刻删除旧 artifact
- 不允许阶段 4、5 继续维护两套路径

### 动作 H：测试同步瘦身

目标：

- 删除只验证旧 helper 和旧兼容层的测试
- 把测试改成只覆盖：
  - 主线 notebook smoke
  - 主线 artifact contract
  - `artifacts/_protection` 保护清单生成 / 回读验证 / `_trash` 恢复 smoke
  - `forecast_lstm.ipynb -> shared_data -> MADRL train/eval -> result package` 一条龙集成 smoke
  - compare 读包
  - `grid_network_analysis.ipynb` 主线 smoke
  - 三条 MADRL 主线最新可运行训练包加载 smoke
  - solver 基础正确性

## 分阶段执行

### 阶段 0：锁定基线与口径

目标：

- 锁定保留 notebook
- 锁定唯一主线窗口
- 锁定当前必须保护的 artifact
- 锁定统计口径

阶段末硬指标：

- 非测试 `.py` 总行数应 **≤ 33323**

完成标准：

- 以后的一切判断都围绕“当前单一主线是否能跑”

### 阶段 1：统一配置入口 + notebook 薄入口改造 + mainline_*.py 骨架

目标：

- 所有共享参数搬进 `configs/experiment_config.py`
- notebook 删除各自定义的测试窗口、reward、agent、shared-data 参数块
- 创建 `mainline_forecast.py / mainline_artifacts.py / mainline_madrl.py / mainline_mpc.py / mainline_compare.py / mainline_grid_analysis.py` 骨架（允许仅做薄包装、搬现有主函数进来、不做删减）
- notebook 改成调用 `mainline_*.py` 入口，import 在本阶段定型
- 允许同步删除已经无 import、主线完全不触达的死 helper（此时不要求重构，只要求删）
- 允许同步删除至少一批已被 `mainline_*.py` 替代的旧入口 / 旧 helper，不允许只增不减
- 阶段 1 净减量下限：**≥ -3323 行**（即阶段末非测试 `.py` 总量必须 ≤ 30000），新增骨架与保留 wrapper 必须被同阶段删除的旧代码抵消

阶段末硬指标：

- 非测试 `.py` 总行数应 **≤ 30000**

完成标准：

- notebook 中不再散落 `2020-06-01` / `2020-06-07` 之外的主线参数定义
- 9 个 notebook 全部通过 `mainline_*.py` 入口运行
- notebook import 之后不再变动，后续阶段只在 `mainline_*.py` 内部重构
- `profiles.py` 与非主线配置分支被清理
- 至少一批被替代的旧入口 / 旧 helper 已被实际删除，而不是仅保留并停止引用

### 阶段 2：forecast 前移 + 六条主线结果包统一 + mainline_artifacts 实装

本阶段把动作 B 和动作 C 合并并行，并强制 `mainline_artifacts.py` 真正落地。

目标：

- `forecast_lstm.ipynb` 训练后自动生成 `shared_data`
- 六条主线统一结果包协议，路径/签名/manifest 由 `mainline_artifacts.py` 统一管理
- `compare.ipynb` 改成只读包
- `grid_network_analysis.ipynb` 改成只调用 `mainline_grid_analysis.py`
- 生成 `artifacts/_protection/` 保护清单，并对新的 forecast / shared_data / 六条主线结果包 / 当前 MADRL 最新可运行训练包 / 当前 grid analysis 结果完成首轮回读校验
- 与新结果包协议重复的 `grid_notebook_compare_reports.py`、`madrl_shared_data.py`、旧 rollout package helper 开始被替代

阶段末硬指标：

- 非测试 `.py` 总行数应 **≤ 24000**

完成标准：

- `mainline_artifacts.py` 已实装，并被六条主线与 compare 实际使用
- `mainline_artifacts.py` 在阶段 2 内允许暂时超过其最终建议体量 350 行（因为此时要吸收 `grid_notebook_compare_reports.py` / `madrl_shared_data.py` / 旧 rollout helper 的逻辑），但必须在阶段 4 压回 350 附近；其它 `mainline_*.py` 同理，阶段 2 允许短期膨胀，阶段 4 必须回到各自建议体量
- compare 不再重算，不再自己找路径
- MADRL notebook 不再生成 shared_data
- 六条主线结果包可直接被 compare 读取
- 保护清单已生成，且受保护输出全部完成至少一次回读成功
- `grid_network_analysis.ipynb` 已通过 `mainline_grid_analysis.py` 成功读取或生成一次当前主线结果
- 三条 MADRL 主线各自最新一套可运行训练包已被纳入保护清单并完成至少一次加载 smoke（历史旧 run 和历史中间 checkpoint 不在保护范围）
- 阶段 2 净减量 **≥ -6000 行**（即阶段末非测试 `.py` 总量必须 ≤ 24000）；减量主要来源：`grid_notebook_compare_reports.py`（约 1450 行）、`madrl_shared_data.py`、旧 rollout package helper、旧 compare helper 已被实际删除或收编进 `mainline_artifacts.py`，不允许只替换不删除

### 阶段 3：旧 artifact 与旧兼容层立即删除

前提：阶段 2 所有完成标准必须全部达成，否则本阶段没有替代品可用，不允许启动。

目标：

- 既然结果包协议已经统一，立刻删除旧 artifact（数据文件不贡献 `.py` 减量，但释放后续删除 `.py` 代码的空间）
- 立刻删除 notebook cache 兼容层、旧 reward 兼容、旧 artifact 命名兼容
- `grid_network_analysis.ipynb` 完全脱离历史 notebook helper，只保留 `mainline_grid_analysis.py` 入口
- 删除 `resolve_latest_compatible_*`、sibling 扫描、prefix 搜索等一切 fuzzy-match `.py` 代码

阶段末硬指标：

- 非测试 `.py` 总行数应 **≤ 18500**

完成标准：

- 仓库里不再同时维护新旧两套路径
- 命中旧 schema / 旧 key / 旧缓存包 / 旧 artifact 命名时统一直接失败
- `artifacts/` 目录下除 `_protection/`（保护清单文件）与 `_trash/`（阶段 3 末已清空的隔离区目录）之外，只保留当前主线结果
- 旧 artifact 的删除范围严格以保护清单为边界
- 阶段 3 的删除已经过”隔离区移动 -> 删除后 smoke -> 物理删除”三步，而不是直接不可恢复删除
- 阶段 3 末 `artifacts/_trash/` 已清空，不允许把物理删除推迟到后续阶段
- 此处"`artifacts/` 目录只保留当前主线结果"的豁免：`artifacts/_protection/` 保护清单文件允许保留；`artifacts/_trash/` 目录本身允许保留以便后续阶段复用，但其内容在阶段 3 末必须清空

### 阶段 4：helper 大规模合并与删除

目标：

- 删除 `scripts/utils/*` 中大部分通用层
- `mainline_forecast.py / mainline_artifacts.py / mainline_madrl.py / mainline_mpc.py / mainline_compare.py / mainline_grid_analysis.py` 内部完成实质重构，落到职责边界给出的建议体量附近
- 本阶段是本轮最大的减量阶段，必须完成 -6500 行

阶段末硬指标：

- 非测试 `.py` 总行数应 **≤ 12000**

完成标准：

- 大部分 notebook helper、rollout helper、compare helper 被删除
- `scripts/utils/grid_notebook_*`、`scripts/utils/misocp_*`、`scripts/utils/admm_*`、`scripts/utils/local_*` 中大部分文件已消失或收敛
- 剩余 `scripts/utils/*` 只保留确实无法内联到 `mainline_*.py` 的底层函数

### 阶段 5：solver / env / trainer 极限收缩

目标：

- 压缩 `global_socp_mpc.py`
- 压缩 `gurobi_agent_mpc.py`
- 压缩 `grid_env.py`
- 压缩训练链路

阶段末硬指标：

- 非测试 `.py` 总行数应 **≤ 8000**

完成标准：

- solver 与 env 只保留当前主线真实走到的数学路径

### 阶段 6：最终收口与验收

本阶段定位为收口 + 验收，不再以"冲数字"为优先目标。因为阶段 5 硬上限已经收到 8000，本阶段最多还需要再收 **1000 行** 才能落到 7000；如果阶段 5 已经进入 7500 附近，这里通常只需要做少量回调与验收修补。

目标：

- 剩余死代码清零
- 测试同步瘦身（`tests/` 不计入主口径，但要删除只验证旧 helper / 旧兼容层的测试，改为覆盖主线 notebook smoke、artifact contract、`artifacts/_protection` 保护清单生成 / 回读验证 / `_trash` 恢复 smoke、`forecast_lstm.ipynb -> shared_data -> MADRL train/eval -> result package` 一条龙集成 smoke、compare 读包、`grid_network_analysis.ipynb` 主线 smoke、三条 MADRL 主线最新可运行训练包加载 smoke、solver 基础正确性）
- notebook 结构与结果包最终确认
- 六条主线完整重放验收

阶段末硬指标：

- 非测试 `.py` 总行数应 **≤ 7000**（理想落点 6500~7000）
- 允许为正确性补回代码，但总量不得超过 7000

完成标准：

- 项目进入 6500~7000 行区间
- 六条主线、compare、forecast、grid analysis 全部可用
- `tests/` 通过，且只覆盖上面八类场景
- 命中旧 schema / 旧 key / 旧缓存包 / 旧 artifact 命名时统一直接失败

## 阶段性行数 checkpoint 总表

| 阶段 | 阶段末非测试 `.py` 总行数应 ≤ | 本阶段减量 |
| ---- | ----: | ----: |
| 阶段 0 | 33323 | 基线 |
| 阶段 1 | 30000 | -3323 |
| 阶段 2 | 24000 | -6000 |
| 阶段 3 | 18500 | -5500 |
| 阶段 4 | 12000 | -6500 |
| 阶段 5 | 8000 | -4000 |
| 阶段 6 | 7000 | -1000 |

规则：

- 任一阶段未达标，不进入下一阶段
- 任一阶段超标超过 10%，先重排方案，再继续执行

## 最终验收标准

- 非测试 `.py` 源码总量 **≤ 7000**，理想落点 **6500~7000**
- `tests/` 已同步收缩，不再维护旧兼容层测试
- 9 个 notebook 文件全部保留
- `forecast_lstm.ipynb` 能训练 forecast 并自动生成当前主线 shared data
- 六条主线都能产出或重放结果包
- `compare.ipynb` 只读结果包即可完成对比
- `grid_network_analysis.ipynb` 通过 `mainline_grid_analysis.py` 运行，不再依赖历史 notebook helper
- 三条 MADRL 主线各自最新一套可运行训练包仍然完整可加载（历史旧 run 和历史中间 checkpoint 已在阶段 3 清理）
- 所有共享参数集中在 `configs/experiment_config.py`
- 唯一主线测试窗口固定为 `2020-06-01` 到 `2020-06-07`
- 旧 artifact 的删除始终遵循“先生成保护清单、再回读验证、移入隔离区、删除后 smoke、最后物理删除旧物”的顺序
- 命中旧 schema、旧 key、旧缓存包、旧 artifact 命名时直接失败

## 建议的执行顺序

1. 先改 `configs/experiment_config.py`，把单一主线配置钉死。
2. 再改 9 个 notebook，把它们全部改成薄入口，并满足 notebook 行数上限。
3. 立刻并行做 forecast 前移、六条主线结果包统一，以及 `grid_network_analysis.ipynb` 的主线入口切换。
4. 生成 `artifacts/_protection/` 保护清单，并完成新结果包的首轮回读验证。
5. 将保护清单之外的旧 artifact 先移入 `artifacts/_trash/`，跑一次删除后 smoke，再决定是否物理删除。
6. compare 一旦变成纯读包且删除后 smoke 通过，立即按保护清单边界删除旧兼容层。
7. 再大规模删除 `scripts/utils/*`、`predictors/*` 的通用层。
8. 对 `GridEnv`、`gurobi_agent_mpc.py`、`global_socp_mpc.py` 做极限收缩。
9. 最后同步瘦身 `tests/`（删除只验证旧 helper / 旧兼容层的测试，收敛到动作 H 列出的七类覆盖），并完成六条主线完整重放验收。

这个顺序的核心原因是：

- 先钉住中心配置，才能放心删 notebook 内部逻辑
- 先钉住结果包协议，compare 简化红利才能尽早释放
- 旧 artifact 越早删，后面越不会被两套路径拖住
- solver 最后动，风险最可控
