# MADRL_ESS 主项目说明

`MADRL_ESS` 当前主线来自 REMAKE 重构：保留数据、预测、环境、控制器、训练、评估、比较这条闭环，删除历史兼容层、检查点恢复、防御式兜底、重复封装和与论文实验无关的旁路代码。它的目标不是做一个通用框架，而是把当前储能多智能体实验用最直接、可读、可复现实验的方式跑通。

## 核心问题

项目研究的是低压配电网中多个 prosumer 储能单元的协同控制。每个 agent 有本地负荷、共享 PV、批发电价和电池 SoC；动作包含电池充放电与 PV 利用/弃光。环境用 SimBench + pandapower 做潮流计算，并把电压、线路、变压器约束反馈到奖励和安全指标里。

实验比较两类方法：

- 优化基线：`MISOCP`、`LOCAL_MPC`、`ADMM_MPC`
- 多智能体强化学习：`MADRL_BASE`、`MADRL_PENALTY`、`MADRL_PROJECTION`

预测设置也保持显式：

- `perfect`：直接使用真实未来序列
- `lstm`：使用训练好的 LSTM 预测价格、负荷和 PV

## 主线流程

推荐按 notebook 顺序运行：

1. `notebooks/predict.ipynb`
   - 创建 `artifacts/runs/<run_id>/`
   - 训练 price/load/PV LSTM
   - 写入 forecast artifacts
   - 构造 `share_data`
   - 导出预测质量表和图
2. `notebooks/madrl.ipynb`
   - 读取同一个 `run_dir`
   - 训练三种 MADRL 方案
   - 保存模型、训练曲线、rollout record
3. `notebooks/misocp.ipynb`
   - 运行全局 MISOCP perfect-forecast 基线
4. `notebooks/mpc.ipynb`
   - 运行 local MPC 的 perfect/lstm 基线
   - 运行 ADMM MPC 的 lstm 基线
5. `notebooks/compare.ipynb`
   - 只读取已经缓存好的 record/result
   - 生成最终对比表、Excel 和图

`remake_test.md` 保留了本次升格和清理的执行计划。

## 目录职责

`configs/`

唯一配置入口是 `configs/cfg.py` 里的 `Cfg` dataclass。所有实验参数按 owner 分组：grid、data、env、algo、train、eval、reward、obs、forecast、model、safety、runtime、mpc。`Cfg.hash8()` 用当前配置生成短 hash，参与 run id 和产物标识。

`datasets/prosumer/`

放实验使用的 CSV 数据：`household.csv`、`heatpump.csv`、`pv_reference.csv`、`price.csv`。读取逻辑只接受这些显式路径和列，不做旧布局搜索。

`data/`

`loader.py` 负责把原始 CSV 转成按 train/eval 时间窗裁剪的 `SeriesData`。`share_data.py` 负责把真实序列和 LSTM 预测序列整理成环境直接使用的 `.npz`，并写 `manifest.json`。`share_data` 是后续 MADRL、MPC、MISOCP、compare 的共同数据契约。

`predictors/`

LSTM 预测 owner。`lstm_training.py` 训练价格、负荷、PV 模型；`lstm_loader.py` 在构造 `share_data` 时批量推理；`lstm_eval.py` 生成预测误差表。负荷按 agent 和 component 训练，PV 和价格是共享序列。

`envs/`

`grid_core.py` 封装 SimBench/pandapower 潮流计算。`grid_env.py` 是强化学习环境：生成 observation、执行动作、更新 SoC、计算储能收益和安全惩罚。`vec_env.py` 提供同步多环境采样，用于 MADRL 训练。

`models/`

`assembly.py` 定义 MADRL 的 actor/critic 网络和 observation 拼接。网络结构是小而固定的 MLP；actor 是每个 agent 一个，critic 接收联合 observation/action。

`controllers/`

统一控制器入口在 `protocol.py`：

- `MISOCP`：全局混合整数二阶锥优化，使用 perfect forecast
- `LOCAL_MPC`：每个 agent 独立滚动优化
- `ADMM_MPC`：带网络协调项的分布式 MPC
- `MADRL_BASE`：无安全惩罚、无投影
- `MADRL_PENALTY`：奖励中加入安全惩罚
- `MADRL_PROJECTION`：奖励惩罚 + 动作投影

MPC/MISOCP 求解器集中在 `*_solver.py`；控制器类只负责从环境 observation 取窗口、调用 solver、把功率计划转成环境动作。

`scripts/`

这里不是命令行包装层，而是 notebook 调用的实验 owner：

- `madrl.py`：MADRL 训练、rollout、训练表和图
- `train.py`：训练三种 MADRL 控制器的薄入口
- `eval.py`：统一评估所有控制器并写 `eval_summary.csv`
- `misocp.py`：全局 MISOCP 实验
- `mpc.py`：local/admm MPC 实验
- `compare.py`：读取缓存结果并生成最终对比
- `plot.py`：预测结果图

`utils/`

只放跨 owner 的稳定基础设施：路径、运行目录、价格转换、torch 设备/随机种子、rollout record 与对比图表。`records.py` 是结果表和图的 owner，因此较大，但它保存的是统一 record schema，不是杂项桶。

`tests/`

`test_remake_smoke.py` 是 REMAKE 的端到端冒烟测试：用很小配置跑预测、share_data、MADRL、eval、compare，并确认核心缓存产物写出。它还约束非测试 Python 代码量，防止 REMAKE 再次膨胀。

## 产物布局

所有运行产物默认写到：

```text
artifacts/runs/<timestamp>_<cfg_hash>/
```

核心子目录：

```text
config.json
run.json
forecast/
  artifacts/
  tables/
  figures/
share_data/
  train.npz
  eval.npz
  manifest.json
models/
  madrl/<scheme>/model.pt
results/
  <forecast_mode>/<controller>/
    metrics.json
    traces.npz
    record/
      manifest.json
      meta.json
      step.parquet
      agent.parquet
      grid.parquet
      summary.parquet
      metrics.parquet
tables/
figures/
```

`metrics.json`/`traces.npz` 是轻量评估缓存；`record/` 是用于论文表格和图的完整 rollout 记录。`compare.py` 依赖这些缓存存在，缺失就直接失败。

## 关键实验语义

动作空间：

- `action[..., 0]`：电池功率比例，映射到 `[-pmax, pmax]` 并受 SoC 约束裁剪
- `action[..., 1]`：PV 利用率，`-1` 表示全弃光，`1` 表示全利用

奖励主要由以下部分组成：

- 储能套利收益
- SoC 边界动作惩罚
- SoC 软边界正则
- 早期探索吞吐 bonus
- 电压、线路、变压器安全惩罚

`MADRL_PROJECTION` 在 actor 输出后增加联合动作投影。投影用线性化电压/线路灵敏度和变压器功率限额，把电池功率与 PV curtailment 投到更安全的可行动作区域。

比较主表默认覆盖 7 个 scheme：

```text
perfect/MISOCP
perfect/LOCAL_MPC
lstm/LOCAL_MPC
lstm/ADMM_MPC
lstm/MADRL_BASE
lstm/MADRL_PENALTY
lstm/MADRL_PROJECTION
```

## 设计边界

这个目录刻意遵守几条硬边界：

- 没有旧 schema、旧路径、旧 checkpoint 的兼容读取。
- 没有“找不到就扫 sibling/latest”的模糊发现。
- 没有训练中断恢复逻辑；高成本产物靠明确 run_dir 和 manifest 复用。
- 缺文件、缺列、缺 solver、缺 artifact 时让异常自然暴露。
- notebook 是实验入口；Python 模块负责真实逻辑。
- 配置从 `Cfg` 进入，结果从 `artifacts/runs/<run_id>` 出来。

这让 REMAKE 更像一份干净的研究实验代码：路径短、owner 明确、失败清楚、复现实验时要重跑哪个入口也清楚。

## 最小程序化入口

notebook 之外，也可以用 Python 串起主线：

```python
from dataclasses import replace

from configs.cfg import Cfg
from data.share_data import build_share_data, load_share_data
from predictors.lstm_training import train_forecasters
from scripts.compare import compare_all
from scripts.eval import eval_all
from scripts.train import train_all
from utils.run_artifacts import create_run_dir, write_config_json

cfg = Cfg()
run_dir = create_run_dir(cfg)

forecast = train_forecasters(cfg, run_dir, overwrite=True)
cfg = replace(cfg, forecast=replace(cfg.forecast, lstm_artifact_dir=str(forecast["artifact_dir"])))
write_config_json(cfg, run_dir)

share_dir = build_share_data(cfg, run_dir, forecast["artifact_dir"], overwrite=True)
share_data = load_share_data(share_dir, cfg)

madrl_models = train_all(cfg, run_dir, share_data)
eval_all(cfg, madrl_models, run_dir, share_data=share_data)
compare_all(cfg, run_dir)
```

长实验默认走 `cuda`。如果机器没有可用 GPU，需要显式改 `cfg.runtime.device`，不要依赖隐式 fallback。

## 维护建议

新增实验时优先回答三个问题：

1. 新逻辑的 owner 是谁？
2. 输入契约在哪里显式进入？
3. 产物是否能通过明确路径或 manifest 被找到？

如果答案不清楚，就先不要加包装层。REMAKE 的价值正在于它短、硬、直：一条科研主线，少量清楚的 owner，结果可复查。
