# MADRL_ESS Notebook 收束与重构计划（small3，2026-04-22）

## 说明

- 本计划只处理 notebook 体系的整理、重构、参数回收、结果落盘与 compare 读链路收束。
- 目标是在**尽量不损失现有功能**的前提下，把 notebook 压回薄入口与展示层。
- 默认允许删除、合并、重写。
- 默认**不兼容旧 notebook helper 结构**，不为了保留旧壳层继续维护过渡逻辑。

## 范围与非目标

### 范围

- notebook 参数回收
- notebook 主链路收束
- compare 读链路改为纯记录读取
- `notebooks/record/` 标准记录合同
- 与 notebook 主链路直接耦合的 helper、tests、导出协议整理

### 非目标

- 不主动扩展新的实验场景、新的测试窗口、新的 notebook 入口
- 不为了“顺便优化”去改算法定义或训练目标，除非 notebook 收束本身要求删除旧壳层
- 不保留旧 notebook 参数结构、旧 compare 重算路径、旧 helper 壳层兼容

## 源文件与减量纪律

### 源文件纪律

- 本轮执行**不允许新增源码文件**，包括新的 `.py`、`.ipynb`、源码型 helper 文件或中转壳文件。
- 允许修改、合并、重写、删除已有源码文件。
- `notebooks/record/` 属于运行时输出目录，不属于新增源码文件范畴。
- 若某个修复动作只有通过新增源码文件才能完成，默认说明当前方案不合格，必须改为更短的主链路方案。

### 总代码量硬约束

- 代码量统计口径固定为：
  - `configs/`
  - `controllers/`
  - `data/`
  - `envs/`
  - `models/`
  - `predictors/`
  - `scripts/`
  - `tests/`
  - `notebooks/` 中所有 code cell
- 当前基线：
  - 非 notebook 项目总行数：`12360`
    - 该数字**包含 `tests/`**
  - notebook code 总行数：`2934`
  - 合计：`15294`
- 本轮完成后的 notebook code 总行数必须 **<= 当前的 70%**。
- 对应 notebook 硬上限：`2934 * 70% = 2053.8`，向下取整后本计划按 **`<= 2053`** 执行。
- notebook 之外的项目总行数必须 **<= 当前基线 `12360`**，不允许高于当前量。

### 分阶段减量预算

| 阶段 | notebook code 净减量要求 | 非 notebook 项目总行数要求 |
| ---- | -----------------------: | -------------------------: |
| `P0` | `<= 0` 净增量，至少不允许变厚 | `<= 12360` |
| `P1` | `= 0` 或更少 | `<= 12360` |
| `P2` | `>= 150` | `<= 12360` |
| `P3` | `>= 250` | `<= 12360` |
| `P4` | `>= 250` | `<= 12360` |
| `P5` | `>= 150` | `<= 12360` |
| `P6` | `>= 81` | `<= 12360` |
| `P7` | `>= 0` | `<= 12360` |

- 上表 notebook 净减量要求合计为 `>= 881`，与最终 `<= 2053` 的 notebook 硬上限对齐。
- 任一阶段如果未达到本阶段 notebook 减量预算的 `70%`，必须暂停并重排，不得机械推进。
- 若某阶段因链路修复短暂增加 notebook code，只允许在**同阶段内**用更大规模删除抵消，阶段总量仍必须满足预算。
- 非 notebook 项目总行数在任一阶段收口后都不得高于 `12360`。

## 本轮硬目标

1. 所有 notebook 共享参数回收至 `configs/experiment_config.py`。
2. 统一测试窗口为完整一周：`2020-06-01` 到 `2020-06-07`。
3. `notebooks/forecast/forecast_lstm.ipynb` 只保留“是否重训”一个 notebook 参数。
4. `notebooks/madrl/train_base.ipynb`、`train_base_safe.ipynb`、`train_projection_safe.ipynb` 只保留“是否重训”一个 notebook 参数。
5. `notebooks/madrl/global_MISOCP.ipynb`、`local_MPC.ipynb`、`ADMM_mpc.ipynb` 统一把测试结果写入 `notebooks/record/`。
6. `notebooks/madrl/compare.ipynb` 只读 `notebooks/record/`，不再触发重训、重建 shared data、重跑 MPC。
7. notebook 不再保留只为旧 helper、旧缓存、旧兼容路径服务的壳层测试或辅助逻辑。
8. 三个 MPC notebook 与 `compare.ipynb` 默认不再暴露任何 notebook 级业务参数。
9. `notebooks/record/` 采用固定 canonical 路径，不使用 timestamp、latest、run_id 或模糊查找。
10. 提高 `configs/experiment_config.py` 可读性，不再把配置定义挤成一坨。
11. `forecast / madrl / mpc` 相关参数前置到 `experiment_config.py` 更靠前的位置，便于集中修改。
12. notebook 代码在不改变功能与输出的前提下，默认继续精简；不保留兼容、套壳、兜底结构。
13. notebook 在关键步骤补中文注释，提升阅读成本最低化。
14. 若 notebook 主链路已在前几轮改动中被打乱，必须先修复到最短 canonical 链路，再继续做减量与整理。
15. 本轮执行后 notebook code 总行数必须 `<= 2053`。
16. 本轮执行后非 notebook 项目总行数必须 `<= 12360`。
17. 本轮执行中不得新增任何源码文件。

## 当前问题快照

### 已确认的窗口不一致

- `notebooks/forecast/forecast_lstm.ipynb` 已写成 `2020-06-01` 到 `2020-06-07`，但 notebook 内仍承担 shared-data 清理与生成 orchestration。
- `notebooks/madrl/global_MISOCP.ipynb` 当前窗口是 `2020-06-01` 到 `2020-06-07`。
- `notebooks/madrl/local_MPC.ipynb` 当前窗口是 `2020-06-01` 到 `2020-06-04`。
- `notebooks/madrl/ADMM_mpc.ipynb` 当前窗口是 `2020-06-01` 到 `2020-06-30`。
- `notebooks/madrl/train_base.ipynb`、`train_base_safe.ipynb`、`train_projection_safe.ipynb` 当前仍保留 notebook 级 `test_start_date` / `test_end_date`，且不少输出还是 `2020-06-05`。
- `notebooks/madrl/compare.ipynb` 当前仍保留 notebook 级 compare 窗口，且还会现场对 ADMM horizon 做对齐处理。

### 已确认的 notebook 过重问题

- `forecast_lstm.ipynb` 仍在 notebook 中直接设置 runtime/data/forecast 控制项，并直接清理 `artifacts/training/shared_data/mainline/*`。
- 三个 MADRL 训练 notebook 仍保留：
  - `test_start_date`
  - `test_end_date`
  - `REUSE_MODEL_ROOT`
  - shared-data 准备与 rollout 组织逻辑
- `compare.ipynb` 仍直接：
  - `ensure_madrl_shared_data`
  - `collect_*_rollout`
  - 读取 / 调整 ADMM 配置
  - 把导出结果写到 `artifacts/compare`
- `configs/experiment_config.py` 当前 dataclass 与字段定义仍然高度压扁，`forecast / MADRL / MPC` 的主线参数不够前置，不利于 notebook 收参数后的集中维护。
- 这些 notebook 很可能在前几轮大改中已经出现“主链路断裂、路径变长、入口与实际执行不一致”的问题，执行本计划时不能只删代码，必须同步修复主链路。

这些都说明 notebook 还没有收回到“薄入口 + 展示层”。

## 主线依赖顺序

1. 先识别并修复已经断裂的 notebook canonical 链路，但修复必须走**最短路径**，不允许新增文件或再包一层 helper。
2. 先固化固定周输出基线，避免“代码更短了，但结果语义变了”。
3. 先整理 `experiment_config.py`，再回收 notebook 参数。
4. 先完成 `forecast_lstm.ipynb`，产出 canonical forecast 记录与 shared-data 记录。
5. 再完成三类结果生产 notebook：
   - `global_MISOCP.ipynb`
   - `local_MPC.ipynb`
   - `ADMM_mpc.ipynb`
   - 三个 MADRL 训练 notebook
6. 最后再改 `compare.ipynb` 为纯读取器。
7. `grid_network_analysis.ipynb` 不在 compare 主链路上，放到最后处理；若无独立价值则直接删除。

## 目标终态

### 1. 参数单一事实源

- notebook 不再定义业务参数。
- 所有共享参数统一收回 `configs/experiment_config.py`。
- `configs/profiles.py` 只允许做组合与命名，不允许重新定义 notebook 专属业务常量。
- `configs/experiment_config.py` 必须同步做可读性整理：
  - dataclass 字段按主题分块
  - 避免多字段长串挤在一行
  - `forecast / MADRL / MPC` 高频修改项前置
  - notebook 主线会改的字段优先靠前
- 默认测试窗口固定为：
  - `test_start_date = "2020-06-01"`
  - `test_end_date = "2020-06-07"`
- `forecast`、`madrl`、`mpc`、`compare` 全部读取同一套测试窗口。

### 2. notebook 角色重定义

- `forecast_lstm.ipynb`
  - notebook 只保留 `force_retrain_forecast`
  - `True`：按 canonical 配置重训 LSTM，随后输出固定周测试结果与 shared data 结果
  - `False`：默认读取 canonical 模型，直接输出固定周测试结果与 shared data 结果
- `train_base.ipynb`、`train_base_safe.ipynb`、`train_projection_safe.ipynb`
  - notebook 只保留 `force_retrain_madrl`
  - `True`：按 canonical 配置重训，再输出固定周测试结果
  - `False`：默认读取 canonical 模型，直接输出固定周测试结果
- `global_MISOCP.ipynb`、`local_MPC.ipynb`、`ADMM_mpc.ipynb`
  - 不再为 compare 临时重算或临时导出
  - 统一负责把固定周测试结果写入 `notebooks/record/`
- `compare.ipynb`
  - 只读取 `notebooks/record/`
  - 不再负责训练、重跑 rollout、重建 shared data、对齐 ADMM horizon
- `grid_network_analysis.ipynb`
  - 只做读结果与展示
  - 若没有独立工程价值，则允许删除；若保留，必须改成只读 `notebooks/record/` 或固定结果文件

### 3. 统一结果目录

- 新增统一结果根目录：`notebooks/record/`
- 该目录成为 notebook 侧唯一的标准化结果出口。
- `compare.ipynb` 以后只从该目录读取，不再从 `artifacts/compare` 或 notebook 临时变量链路拼装数据。
- `notebooks/record/` 是 notebook 结果记录层，不是新的运行时业务 owner。
- 核心 owner 仍保留自己的 canonical 产物路径；notebook 记录层只保存 compare-ready 记录与精确 locator。
- `shared_data` 的真实 payload 仍保留在 runtime owner 的 canonical 路径；`notebooks/record/` 不复制大体量 payload，只记录 manifest 快照和精确 locator。

### 4. 参数暴露上限

- `forecast_lstm.ipynb` 只允许保留 `force_retrain_forecast`。
- `train_base.ipynb`、`train_base_safe.ipynb`、`train_projection_safe.ipynb` 只允许保留 `force_retrain_madrl`。
- `global_MISOCP.ipynb`、`local_MPC.ipynb`、`ADMM_mpc.ipynb`、`compare.ipynb` 默认不应保留任何 notebook 级业务参数。
- 若某个 notebook 仍要求用户手填测试窗口、模型根目录、导出目录、shared-data 路径，则视为未完成收束。

### 5. compare 方案清单必须固定

- compare 以后只认这 7 个 canonical scheme：
  - `global_misocp`
  - `local_mpc_perfect`
  - `local_mpc_lstm`
  - `admm_mpc_lstm`
  - `madrl_base`
  - `madrl_base_safe`
  - `madrl_projection_safe`
- 不允许 compare 通过 notebook 名称、目录扫描、文件个数去推断 scheme。
- 每个 scheme 都必须有自己的固定记录目录和独立 manifest。

### 6. notebook 代码风格终态

- notebook 在不改变功能和输出的前提下，默认继续删减。
- 若 notebook 当前已经跑不通或链路已乱，优先修复最短 canonical 链路；修复本身也必须遵守“无新增文件、无兜底、无套壳”。
- notebook 不再保留：
  - 兼容分支
  - 套壳 helper
  - 兜底查找
  - 多层 orchestration 中转
  - 只为旧路径服务的冗余展示块
- notebook 的关键步骤必须补中文注释，至少覆盖：
  - 入口参数说明
  - 主入口调用点
  - 结果写入 `notebooks/record/` 的位置
  - compare 读取记录并校验合同的位置
- 中文注释必须解释“为什么这里要这样做”，不是复述代码字面含义。
- notebook 收束后的代码应尽量靠近“单参数 -> 组装 cfg -> 调主入口 -> 展示结果 -> 写记录”的骨架。

## 统一记录合同

### 目录建议

```text
notebooks/record/
  forecast/
    lstm/
      manifest.json
      metrics.parquet
      predictions.parquet
      shared_data_record.json
      config_snapshot.json
  madrl/
    madrl_base/
      manifest.json
      metrics.parquet
      rollout.parquet
      config_snapshot.json
    madrl_base_safe/
      ...
    madrl_projection_safe/
      ...
  mpc/
    global_misocp/
      manifest.json
      metrics.parquet
      rollout.parquet
      config_snapshot.json
    local_mpc_perfect/
      manifest.json
      metrics.parquet
      rollout.parquet
      config_snapshot.json
    local_mpc_lstm/
      manifest.json
      metrics.parquet
      rollout.parquet
      config_snapshot.json
    admm_mpc_lstm/
      ...
  compare/
      manifest.json
      economic_table.parquet
      safety_table.parquet
      compare_timeseries.parquet
```

### 固定路径策略

- 单场景单窗口下，`notebooks/record/` 默认使用固定 canonical 文件名。
- 同一 notebook 再次运行时，直接覆盖同一路径，不额外生成 `latest`、时间戳目录、随机目录或副本链。
- compare 只读固定路径 manifest，不做目录枚举，不做“选最新一次运行”。
- 一个 notebook 若产出多个 scheme，则必须写入多个固定 scheme 目录，而不是把多份结果混在一个目录里。

### 最低合同要求

- 每个方案都必须有 `manifest.json`。
- `manifest.json` 至少记录：
  - `schema_version`
  - `producer_notebook`
  - `scheme_name`
  - `test_start_date`
  - `test_end_date`
  - `config_snapshot_path`
  - `metrics_path`
  - `rollout_path` 或 `predictions_path`
- 若结果依赖 shared data，记录文件必须额外提供精确 locator：
  - `shared_data_dir`
  - `shared_data_signature`
  - `shared_data_manifest_path` 或复制后的 manifest 快照
- compare 只接受完整 manifest，不做模糊猜测，不做兜底扫描，不做“找最近的一份”。

### shared data 记录原则

- `shared_data` 属于运行时 owner，不应为了 notebook 记录层而被 compare 重新生成或挪动主存放位置。
- `forecast_lstm.ipynb` 在 `notebooks/record/forecast/lstm/` 中记录：
  - 固定周测试结果
  - shared-data manifest 快照或精确 locator
  - config snapshot
- compare 只读取 notebook record 中登记过的 shared-data locator，不直接扫描 `artifacts/training/shared_data/mainline/`。

### 输出不变原则

- notebook 收束允许改变代码组织、记录目录与调用链，但不应改变既有方案的运行语义。
- 对同一固定窗口与同一 canonical 配置：
  - 指标定义不变
  - compare 表格语义不变
  - rollout / prediction 的时间范围不变
  - 结果路径可以变，但结果含义不能偷换
- 若确实需要改变某个输出合同，必须显式写入 manifest schema 版本，并同步改 compare 读取逻辑；不允许半新半旧共存。

## 计划分阶段

## P0. 先固化当前输出基线与记录合同

### 目标

- 在改 notebook 之前，先修复明显断裂的 canonical 链路，再把当前固定周输出基线固化下来，然后把 `notebooks/record/` 的目录、文件名、manifest 字段、写入 owner、读取 owner 固定下来。

### 主要动作

- 先检查 producer notebook 是否已被前几轮大改打乱主链路；若已断裂，先做最短路径修复。
- 链路修复时禁止：
  - 新增源码文件
  - 新增兼容层
  - 新增中转 helper
  - 为了跑通而恢复旧壳路径
- 用固定窗口 `2020-06-01` 到 `2020-06-07` 固化当前 canonical 输出基线，至少覆盖：
  - forecast 固定周指标
  - 7 个 compare scheme 的主要 metrics
  - compare 经济表与安全表的列定义
- 明确每个 notebook 的记录出口目录与固定文件名。
- 明确 notebook 到 scheme 的映射关系：
  - `forecast_lstm.ipynb` -> `forecast/lstm`
  - `global_MISOCP.ipynb` -> `global_misocp`
  - `local_MPC.ipynb` -> `local_mpc_perfect` + `local_mpc_lstm`
  - `ADMM_mpc.ipynb` -> `admm_mpc_lstm`
  - `train_base.ipynb` -> `madrl_base`
  - `train_base_safe.ipynb` -> `madrl_base_safe`
  - `train_projection_safe.ipynb` -> `madrl_projection_safe`
- 明确 compare 的输入只来自 record manifest。
- 明确 shared-data 只通过精确 locator 进入 compare，不通过目录扫描进入 compare。
- 若需要新增 writer/reader helper，必须属于记录协议 owner，而不是再造一个 notebook orchestration 杂货层。

### 完成标志

- 先有稳定记录合同，再开始改 forecast / MADRL / MPC / compare notebook。
- compare 不再依赖 notebook 运行时对象结构，只依赖 record manifest。
- 已有输出基线可用于后续回归对照。
- 本阶段 notebook code `净增量 <= 0`。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## P1. 先整理 experiment_config 可读性

### 目标

- 在把 notebook 参数大规模收回配置层之前，先把 `configs/experiment_config.py` 从“压扁配置表”整理成可阅读、可定位、可集中修改的配置入口。

### 主要动作

- 将 dataclass 字段改为按主题分块排布，不再多字段长串同排。
- 将 notebook 主线会频繁调整的配置提前，包括：
  - `forecast`
  - `madrl` 训练主线直接相关配置
  - `mpc`
  - 测试窗口
  - `notebooks/record` 路径或记录协议相关配置
- 将默认窗口 `2020-06-01` 到 `2020-06-07` 放在清晰可见的位置。
- 保持配置语义不变；本阶段目标是提可读性与可维护性，不引入新的兼容层。

### 完成标志

- `experiment_config.py` 不再是大段单行字段堆叠。
- 开发者能在文件前部较快找到 `forecast / MADRL / MPC / test window` 主线配置。
- 参数回收后，不需要再去 notebook 里翻业务常量。
- `algo / model / train / obs / reward / safety` 这组 MADRL 主线配置在结构上相邻可读，而不是散落查找。
- 本阶段 notebook code 净减量要求：`= 0` 或更少。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## P2. 收回 notebook 参数到 experiment_config

### 目标

- 清空 notebook 中除重训 flag 以外的业务参数。
- 把测试窗口、forecast 控制、MADRL/MPC 测试设置、record 根目录全部收回配置层。

### 主要动作

- 在 `configs/experiment_config.py` 中建立 notebook 主线需要的 canonical 参数。
- 如需 profile，只允许在 `configs/profiles.py` 中组合，不允许再复制常量。
- 删除 notebook 内的以下参数定义：
  - `test_start_date`
  - `test_end_date`
  - `TEST_START_DATE`
  - `TEST_END_DATE`
  - `REUSE_MODEL_ROOT`
  - `EXPORT_DIR`
  - notebook 内重复的 `agent_profiles` / `load_scale` / `pv_scale` / `future_horizon` 等控制项

### 完成标志

- `rg` 搜 notebook 时，不再出现 notebook 内定义的测试窗口常量。
- notebook 内只剩 `force_retrain_forecast` 或 `force_retrain_madrl` 这类单一入口布尔量。
- MPC 与 compare notebook 中不再出现 notebook 级 `EXPORT_DIR`、测试窗口、shared-data 路径或模型根目录常量。
- 本阶段 notebook code 净减量要求：`>= 150`。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## P3. forecast notebook 收束

### 目标

- `forecast_lstm.ipynb` 从“训练 + 验证 + shared-data orchestration notebook”收成“单一入口 + 展示 + 记录导出”。

### 主要动作

- 删除 notebook 里的 shared-data 清理逻辑。
- 删除 notebook 里的 runtime/data/signal 手工拼装块。
- 把训练、验证、shared-data 输出串成单一主入口调用。
- 删除 notebook 中只为旧 helper、旧 shared-data 清理流程服务的壳层代码。
- `force_retrain_forecast=False` 时，默认直接读取 canonical LSTM artifact，再输出：
  - 固定周测试结果
  - shared data locator 与 manifest 快照
  - `notebooks/record/forecast/lstm/*`
- 在关键步骤补中文注释：
  - 唯一 notebook 参数
  - 主入口调用
  - 结果记录写出

### 完成标志

- `forecast_lstm.ipynb` 参数区只剩一个重训 flag。
- notebook 不再直接删除目录，不再手工重建 shared-data 根结构。
- notebook 默认读 canonical artifact；找不到就显式失败并提示应先运行哪个 notebook。
- notebook 主体明显缩短，执行链条短于当前版本。
- 结果可与 P0 固化的 forecast 基线做回归对照。
- 本阶段 notebook code 净减量要求：`>= 250`。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## P4. 三个 MADRL 训练 notebook 收束

### 目标

- `train_base.ipynb`、`train_base_safe.ipynb`、`train_projection_safe.ipynb` 结构统一，角色统一，输出统一。

### 主要动作

- 删除三个 notebook 中的：
  - `test_start_date`
  - `test_end_date`
  - `REUSE_MODEL_ROOT`
  - notebook 内部日期归一化逻辑
  - notebook 内为旧 shared-data/helper 链路保留的分叉
  - 只做转发与拼装的中间壳函数
- 保留唯一参数：
  - `force_retrain_madrl`
- `False` 分支默认：
  - 读取 canonical model
  - 运行固定周测试
  - 写入 `notebooks/record/madrl/<scheme>/`
- `True` 分支默认：
  - 重训 canonical model
  - 紧接固定周测试
  - 写入同一记录目录
- 在关键步骤补中文注释：
  - 当前 notebook 对应的 scheme
  - 重训 / 复用分支
  - 测试结果记录写出

### 完成标志

- 三个 notebook 的参数区与执行骨架高度一致。
- 三个 notebook 不再接受 notebook 级窗口与模型根目录参数。
- 三个 notebook 的测试结果可被 compare 直接读取。
- 三个 notebook 输出的 manifest 字段结构一致，只有 `scheme_name` 与产物路径不同。
- 三个 notebook 相比当前版本明显减掉分叉、辅助块与旧兼容逻辑。
- 结果可与 P0 固化的 3 个 MADRL scheme 基线做回归对照。
- 本阶段 notebook code 净减量要求：`>= 250`。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## P5. 三个 MPC notebook 收束

### 目标

- `global_MISOCP.ipynb`、`local_MPC.ipynb`、`ADMM_mpc.ipynb` 统一成为 compare-ready 结果生产者。

### 主要动作

- 统一测试窗口到 `2020-06-01` 到 `2020-06-07`。
- 删除 notebook 内对 compare 的临时适配逻辑。
- 删除 notebook 内只为 compare 服务的临时导出路径。
- 删除 notebook 内只为旧 notebook helper 或旧缓存结构服务的转发代码。
- 统一把固定周结果写入：
  - `notebooks/record/mpc/global_misocp/`
  - `notebooks/record/mpc/local_mpc_perfect/`
  - `notebooks/record/mpc/local_mpc_lstm/`
  - `notebooks/record/mpc/admm_mpc_lstm/`

### 特别要求

- ADMM 不再在 compare 里做 horizon 对齐。
- ADMM 若与统一窗口不兼容，应在自己的主入口显式失败，而不是把补丁逻辑留给 compare。
- Local / Global / ADMM 三个 MPC notebook 不再暴露 notebook 级导出目录或测试窗口参数。
- 在关键步骤补中文注释：
  - 固定周窗口来源
  - 主求解调用
  - 记录写入位置

### 完成标志

- 三个 MPC notebook 都能独立生成 compare-ready 记录。
- compare 不再调用任何 MPC rollout 收集逻辑。
- 三个 MPC notebook 主体没有多余兜底和中转壳层。
- `local_MPC.ipynb` 明确产出两份独立 scheme 记录，而不是一个混合目录。
- 结果可与 P0 固化的 4 个 MPC-related scheme 基线做回归对照。
- 本阶段 notebook code 净减量要求：`>= 150`。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## P6. compare notebook 改成纯读取器

### 目标

- `compare.ipynb` 只负责读、验、汇总、画图。

### 主要动作

- 删除 compare 中的：
  - `ensure_madrl_shared_data`
  - `collect_global_full_horizon_rollout`
  - `collect_local_mpc_rollout`
  - `collect_madrl_rollout`
  - compare 内部 ADMM horizon 对齐
  - compare 内部导出到 `artifacts/compare`
- compare 只做：
  - 读取 `notebooks/record/` 下各方案 manifest
  - 校验窗口一致
  - 校验 schema 一致
  - 汇总 metrics / timeseries
  - 输出 compare 表格与图
- 在关键步骤补中文注释：
  - 当前读取哪些记录
  - 校验哪些合同字段
  - 输出哪些比较结果

### 完成标志

- compare 不再触发任何训练或求解。
- compare 在缺少某个方案记录时直接失败，并明确指出缺的是哪一个 notebook 的记录。
- compare 不再持有任何 notebook 级测试窗口、导出目录或 shared-data 生成参数。
- compare 代码主体比当前明显更短，不再保留重算链路与兼容链路。
- compare 读取的 scheme 列表与 P0 固定的 7 个 canonical scheme 完全一致。
- 本阶段 notebook code 净减量要求：`>= 81`。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## P7. tests 与旧 helper 清理

### 目标

- 测试与 notebook 结构一起收束，不再维护旧 helper 壳层。

### 主要动作

- 删除只服务旧 notebook 参数结构、旧临时导出路径、旧 compare 重算链路的测试。
- 新增或改写测试，覆盖：
  - notebook 参数是否已收束到单一 flag
  - `experiment_config.py` 是否按新结构保持可读性与主线参数前置
  - 固定周窗口是否统一为 `2020-06-01` 到 `2020-06-07`
  - 7 个 canonical scheme 的记录目录与 manifest 是否齐全
  - `notebooks/record/` 合同是否完整
  - compare 是否只读记录、不再重算
- 若存在旧 notebook helper，仅在它们有独立 owner 价值时保留；否则直接删。

### 完成标志

- tests 不再为旧 helper、旧参数块、旧兼容路径背书。
- notebook 收束后新增测试数量可以增加，但不允许为了保留旧行为继续挂测试壳。
- 本阶段 notebook code 净减量要求：`>= 0`。
- 本阶段非 notebook 项目总行数 `<= 12360`。

## 明确删除项

- 删除 notebook 内自定义测试窗口参数。
- 删除 notebook 内 `REUSE_MODEL_ROOT`。
- 删除 notebook 内 `EXPORT_DIR = PROJECT_ROOT / 'artifacts' / 'compare'` 这类 compare 专属临时出口。
- 删除 forecast notebook 中直接清理 shared-data 目录的逻辑。
- 删除 compare notebook 中的 on-the-fly shared-data 构造与 rollout 重算逻辑。
- 删除只为旧 notebook helper 结构保留的测试与 helper 壳层。
- 删除 `latest`、时间戳、随机 `run_id` 记录目录方案。
- 删除将 `local_MPC` 两种方案混放在单目录的记录结构。
- 删除为修复链路而临时加上的过渡壳；若某修复需要靠过渡壳存在，则该修复方案本身不合格。

## 验收标准

1. `forecast_lstm.ipynb` 参数区只剩 `force_retrain_forecast`。
2. 三个 MADRL 训练 notebook 参数区只剩 `force_retrain_madrl`。
3. `experiment_config.py` 已完成可读性整理，且 `forecast / MADRL / MPC / test window` 主线参数前置可见。
4. `forecast / madrl / mpc / compare` 全部使用 `2020-06-01` 到 `2020-06-07`。
5. 三个 MPC notebook 与 `compare.ipynb` 不再暴露 notebook 级业务参数。
6. `compare.ipynb` 不再训练、不再求解、不再构造 shared data。
7. 所有 compare 所需输入都能在 `notebooks/record/` 固定路径下找到。
8. 缺少记录时显式失败，不做任何模糊查找、兜底或旧路径兼容。
9. notebook 主体只承担：
   - 读取 canonical config
   - 调用单一主入口
   - 展示结果
   - 写标准记录
10. `notebooks/record/` 中不存在 `latest`、时间戳目录、随机运行目录等二义性路径。
11. notebook 关键步骤均有中文注释，且中文注释解释的是关键设计点而不是复述代码字面含义。
12. notebook 代码较当前版本明显更短，不再保留旧兼容、套壳、兜底链路。
13. compare 固定读取且只读取这 7 个 scheme：`global_misocp`、`local_mpc_perfect`、`local_mpc_lstm`、`admm_mpc_lstm`、`madrl_base`、`madrl_base_safe`、`madrl_projection_safe`。
14. `local_MPC.ipynb` 的 perfect / lstm 结果已拆成两份独立记录，而不是混在单目录。
15. 收束后的主要指标与 compare 表语义相对 P0 输出基线不发生偷换。
16. 本轮执行过程中未新增任何源码文件。
17. 本轮执行后的 notebook code 总行数 `<= 2053`，即不高于当前 notebook 基线 `2934` 的 `70%`。
18. 本轮执行后的非 notebook 项目总行数 `<= 12360`，即不高于当前非 notebook 基线。

## 一句话总纲

把所有 notebook 收成薄入口，把参数收回 `experiment_config.py`，把结果统一写成固定合同的 `notebooks/record/`，让 compare 只读记录、不再重跑任何训练或求解。
