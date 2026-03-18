"""Add module-level docstrings to all refactored files.

Ensures every .py file in the refactored project has a clear
module-level docstring describing:
1. Core responsibility (one line)
2. Main classes / functions
3. Typical usage (optional)
"""
import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─── Docstrings for all files ────────────────────────────────────────

DOCSTRINGS = {
    # ── configs/ ──
    "configs/experiment_config.py": (
        '"""实验统一配置 dataclass 定义。\n'
        '\n'
        '以纯数据结构形式定义所有实验参数（环境、算法、模型、奖励、预测器等），\n'
        '不包含任何逻辑，仅供 profiles.py 等组合函数使用。\n'
        '\n'
        '主要类:\n'
        '    ExperimentConfig -- 顶层配置，组合以下子配置\n'
        '    EnvConfig       -- 环境参数\n'
        '    AlgoConfig      -- MADRL 算法参数\n'
        '    ModelConfig      -- 神经网络模型参数\n'
        '    RewardConfig     -- 奖励函数参数\n'
        '    ObsConfig        -- 观测空间参数\n'
        '    ForecastConfig   -- 预测器参数\n'
        '    DataConfig       -- 数据集参数\n'
        '    TrainConfig      -- 训练循环参数\n'
        '    RuntimeConfig    -- 运行时配置\n'
        '    GridConfig       -- 电网潮流约束参数\n'
        '"""\n'
    ),

    # ── controllers/ ──
    "controllers/base.py": (
        '"""控制器抽象基类。\n'
        '\n'
        '定义所有控制器（MADRL、MPC、经典 DRL 等）的统一接口，\n'
        '包含动作选择、模型保存加载等基本方法。\n'
        '\n'
        '主要类:\n'
        '    BaseController -- 控制器抽象基类\n'
        '"""\n'
    ),
    "controllers/zero_controller.py": (
        '"""零动作基线控制器。\n'
        '\n'
        '始终输出零动作的简单基线控制器，用于与其他算法对比。\n'
        '\n'
        '主要类:\n'
        '    ZeroController -- 零动作基线控制器\n'
        '"""\n'
    ),
    "controllers/madrl_controller.py": (
        '"""多智能体深度强化学习（MADRL）控制器。\n'
        '\n'
        '封装 MADRL 算法（MADDPG / MATD3）的高层控制器接口，\n'
        '负责多智能体联合动作选择与训练调度。\n'
        '\n'
        '主要类:\n'
        '    MADRLController -- MADRL 控制器\n'
        '"""\n'
    ),
    "controllers/madrl/base_agent.py": (
        '"""MADRL 单智能体基类。\n'
        '\n'
        '定义单个智能体的 Actor-Critic 网络结构与更新逻辑，\n'
        '供 MADDPG 和 MATD3 等具体算法继承。\n'
        '\n'
        '主要类:\n'
        '    BaseAgent -- 单智能体抽象基类\n'
        '"""\n'
    ),
    "controllers/madrl/maddpg.py": (
        '"""MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 实现。\n'
        '\n'
        '基于集中式训练、分布式执行（CTDE）范式的多智能体连续动作算法。\n'
        '\n'
        '主要类:\n'
        '    MADDPG -- MADDPG 算法智能体\n'
        '"""\n'
    ),
    "controllers/madrl/matd3.py": (
        '"""MATD3 (Multi-Agent Twin Delayed DDPG) 实现。\n'
        '\n'
        '在 MADDPG 基础上引入 twin critic 和延迟策略更新，提高训练稳定性。\n'
        '\n'
        '主要类:\n'
        '    MATD3 -- MATD3 算法智能体\n'
        '"""\n'
    ),
    "controllers/madrl/registry.py": (
        '"""MADRL 算法注册表。\n'
        '\n'
        '根据算法名称（如 "MADDPG"、"MATD3"）返回对应的智能体类。\n'
        '\n'
        '主要函数:\n'
        '    get_agent_cls -- 根据名称获取算法类\n'
        '"""\n'
    ),
    "controllers/mpc/mpc_controller.py": (
        '"""模型预测控制（MPC）控制器。\n'
        '\n'
        '基于线性规划的 MPC 基线控制器，使用未来预测信息\n'
        '求解最优充放电策略。\n'
        '\n'
        '主要类:\n'
        '    MPCController -- MPC 控制器\n'
        '"""\n'
    ),
    "controllers/drl/classic_drl_controller.py": (
        '"""经典单智能体 DRL 控制器。\n'
        '\n'
        '将多智能体问题退化为单智能体处理的基线控制器。\n'
        '\n'
        '主要类:\n'
        '    ClassicDRLController -- 单智能体 DRL 控制器\n'
        '"""\n'
    ),

    # ── data/loaders/ ──
    "data/loaders/base.py": (
        '"""数据集抽象基类。\n'
        '\n'
        '定义所有数据集加载器的统一 API，子类需实现 __len__ 和 __getitem__。\n'
        '\n'
        '主要类:\n'
        '    BaseDataset -- 数据集抽象基类\n'
        '"""\n'
    ),
    "data/loaders/csv_price_load.py": (
        '"""CSV 电价-负荷数据集加载器。\n'
        '\n'
        '从 SimBench 格式的 CSV 文件加载电价和负荷时间序列数据。\n'
        '\n'
        '主要类:\n'
        '    CsvPriceLoadDataset -- CSV 电价负荷数据集\n'
        '"""\n'
    ),
    "data/loaders/csv_prosumer.py": (
        '"""CSV 产消者数据集加载器。\n'
        '\n'
        '加载含负荷、光伏、电价等多信号的产消者时间序列数据，\n'
        '支持多智能体场景。\n'
        '\n'
        '主要类:\n'
        '    CsvProsumerDataset -- 产消者数据集\n'
        '"""\n'
    ),
    "data/loaders/registry.py": (
        '"""数据集注册表。\n'
        '\n'
        '根据数据集类型名称（如 "csv_price_load"、"csv_prosumer"）\n'
        '返回对应的数据集类或实例。\n'
        '\n'
        '主要函数:\n'
        '    build_dataset -- 根据配置构建数据集实例\n'
        '"""\n'
    ),
    "data/loaders/simbench_export.py": (
        '"""SimBench 电网数据导出与预处理工具。\n'
        '\n'
        '从 SimBench 电网模型中提取负荷、发电等时间序列数据，\n'
        '导出为标准 CSV 格式供训练使用。\n'
        '\n'
        '主要函数:\n'
        '    export_simbench_data -- 导出 SimBench 数据到 CSV\n'
        '"""\n'
    ),

    # ── envs/ ──
    "envs/hems_env.py": (
        '"""家庭能源管理系统（HEMS）强化学习环境。\n'
        '\n'
        '封装 Gym 接口的多智能体电池储能调度环境，\n'
        '不含电网潮流约束。\n'
        '\n'
        '主要类:\n'
        '    EnergyStorageEnv -- HEMS 多智能体储能环境\n'
        '\n'
        '典型用法::\n'
        '    env = EnergyStorageEnv(cfg, mode="train", dataset=ds)\n'
        '    obs = env.reset()\n'
        '"""\n'
    ),
    "envs/grid_env.py": (
        '"""电网潮流环境实现。\n'
        '\n'
        '在 pandapower 电网模型上封装 Gym 接口，支持多智能体电池调度，\n'
        '包含电压和线路负载约束。\n'
        '\n'
        '主要类:\n'
        '    GridEnv -- 带潮流约束的多智能体储能环境\n'
        '\n'
        '典型用法::\n'
        '    env = GridEnv(cfg, mode="train", dataset=ds)\n'
        '    obs = env.reset()\n'
        '"""\n'
    ),
    "envs/registry.py": (
        '"""环境注册表。\n'
        '\n'
        '根据环境类型名称（如 "energy_storage"、"grid_pf"）\n'
        '返回对应的环境类。\n'
        '\n'
        '主要函数:\n'
        '    build_env -- 根据配置创建环境实例\n'
        '"""\n'
    ),
    "envs/vec_env.py": (
        '"""向量化环境封装（DummyVecEnv）。\n'
        '\n'
        '在单进程中串行运行多个环境副本，提供统一的批量接口。\n'
        '\n'
        '主要类:\n'
        '    DummyVecEnv -- 单进程向量化环境\n'
        '"""\n'
    ),
    "envs/subproc_vec_env.py": (
        '"""多进程向量化环境封装（SubprocVecEnv）。\n'
        '\n'
        '使用多进程并行运行多个环境副本，适合 CPU 密集的环境\n'
        '（如 pandapower 潮流计算）。\n'
        '\n'
        '主要类:\n'
        '    SubprocVecEnv -- 多进程向量化环境\n'
        '"""\n'
    ),
    "envs/grid/core/grid_core.py": (
        '"""电网核心模型。\n'
        '\n'
        '封装 pandapower 电网的创建、潮流计算和状态查询，\n'
        '供 GridEnv 调用。\n'
        '\n'
        '主要类:\n'
        '    GridCore -- pandapower 电网核心封装\n'
        '"""\n'
    ),
    "envs/grid/core/grid_types.py": (
        '"""电网数据类型定义。\n'
        '\n'
        '定义电网模型使用的数据结构，如节点类型、线路参数等。\n'
        '"""\n'
    ),
    "envs/grid/core/net_builder.py": (
        '"""pandapower 电网构建器。\n'
        '\n'
        '根据 SimBench 编码或自定义拓扑构建 pandapower 网络对象。\n'
        '\n'
        '主要函数:\n'
        '    build_net -- 构建 pandapower 网络\n'
        '"""\n'
    ),
    "envs/grid/config/grid_config.py": (
        '"""电网拓扑与约束配置。\n'
        '\n'
        '定义电网 SimBench 编码、求解器类型、电压约束等参数。\n'
        '\n'
        '主要类:\n'
        '    GridConfig -- 电网配置数据类\n'
        '"""\n'
    ),
    "envs/grid/topology/rural1_fixed.py": (
        '"""Rural1 固定拓扑定义。\n'
        '\n'
        '为 SimBench "1-LV-rural1--0-sw" 拓扑提供预定义的\n'
        '智能体安装位置和网络参数。\n'
        '"""\n'
    ),
    "envs/rewards/base.py": (
        '"""奖励函数基类。\n'
        '\n'
        '定义所有奖励函数的统一接口。\n'
        '\n'
        '主要类:\n'
        '    BaseReward -- 奖励函数抽象基类\n'
        '"""\n'
    ),
    "envs/rewards/composite.py": (
        '"""复合奖励函数。\n'
        '\n'
        '组合电价套利奖励、SoC 惩罚等多个分量的加权奖励函数。\n'
        '\n'
        '主要类:\n'
        '    CompositeReward -- 复合奖励函数\n'
        '"""\n'
    ),
    "envs/rewards/grid_composite.py": (
        '"""带电网约束的复合奖励函数。\n'
        '\n'
        '在 CompositeReward 基础上增加电压越界和线路过载惩罚。\n'
        '\n'
        '主要类:\n'
        '    GridCompositeReward -- 电网约束复合奖励\n'
        '"""\n'
    ),
    "envs/rewards/sparse.py": (
        '"""稀疏奖励函数。\n'
        '\n'
        '仅在 episode 结束时给出奖励的稀疏奖励方案。\n'
        '\n'
        '主要类:\n'
        '    SparseReward -- 稀疏奖励函数\n'
        '"""\n'
    ),

    # ── predictors/ ──
    "predictors/base.py": (
        '"""预测器抽象基类。\n'
        '\n'
        '定义所有预测器（perfect、naive、LSTM 等）的统一接口。\n'
        '\n'
        '主要类:\n'
        '    BaseForecaster -- 预测器抽象基类\n'
        '"""\n'
    ),
    "predictors/naive.py": (
        '"""朴素预测器。\n'
        '\n'
        '基于历史窗口复制的简单基线预测器。\n'
        '\n'
        '主要类:\n'
        '    NaiveForecaster -- 朴素预测器\n'
        '"""\n'
    ),
    "predictors/oracle.py": (
        '"""Oracle（完美预知）预测器。\n'
        '\n'
        '直接返回真实未来值，用作预测上界基线。\n'
        '\n'
        '主要类:\n'
        '    OracleForecaster -- 完美预知预测器\n'
        '"""\n'
    ),
    "predictors/lstm_forecaster.py": (
        '"""LSTM 时序预测器。\n'
        '\n'
        '使用训练好的 LSTM 模型对电价、负荷、光伏等信号进行多步预测。\n'
        '\n'
        '主要类:\n'
        '    LSTMForecaster -- LSTM 预测器\n'
        '"""\n'
    ),
    "predictors/lstm_model.py": (
        '"""LSTM 预测网络定义。\n'
        '\n'
        '定义 LSTM 回归网络的 PyTorch 实现。\n'
        '\n'
        '主要类:\n'
        '    LSTMModel -- LSTM 网络模型\n'
        '"""\n'
    ),
    "predictors/training.py": (
        '"""LSTM 预测器训练与数据准备。\n'
        '\n'
        '提供 LSTM 预测模型的训练循环、数据切分、评估指标计算等功能。\n'
        '\n'
        '主要函数:\n'
        '    ensure_lstm_artifacts -- 确保 LSTM 模型已训练，否则自动训练\n'
        '    train_lstm_model     -- 训练 LSTM 预测模型\n'
        '"""\n'
    ),
    "predictors/artifacts.py": (
        '"""预测器模型产物（artifact）路径管理。\n'
        '\n'
        '管理 LSTM 等预测模型的保存/加载路径。\n'
        '\n'
        '主要函数:\n'
        '    get_default_lstm_artifact_dir -- 获取默认 LSTM 产物目录\n'
        '"""\n'
    ),
    "predictors/registry.py": (
        '"""预测器注册表。\n'
        '\n'
        '根据预测器类型名称（如 "perfect"、"naive"、"lstm"）\n'
        '返回对应的预测器类。\n'
        '\n'
        '主要函数:\n'
        '    build_forecaster -- 根据配置构建预测器实例\n'
        '"""\n'
    ),

    # ── scripts/ ──
    "scripts/builder.py": (
        '"""实验组件构建入口。\n'
        '\n'
        '根据 ExperimentConfig 构建环境、控制器、预测器、数据集等\n'
        '训练所需的各个组件，是训练流程的核心装配逻辑。\n'
        '\n'
        '主要函数:\n'
        '    build_env          -- 构建环境\n'
        '    build_train_runner -- 构建完整训练运行器\n'
        '"""\n'
    ),
    "scripts/train.py": (
        '"""训练主循环。\n'
        '\n'
        '实现 MADRL 训练的主循环逻辑，包括环境交互、经验收集、\n'
        '网络更新、checkpoint 保存等。\n'
        '\n'
        '主要类:\n'
        '    TrainRunner -- 训练运行器\n'
        '\n'
        '典型用法::\n'
        '    runner = TrainRunner(cfg, env, controller, ...)\n'
        '    runner.run()\n'
        '"""\n'
    ),
    "scripts/checkpoints.py": (
        '"""Checkpoint 保存与加载。\n'
        '\n'
        '管理训练过程中模型权重、优化器状态、训练进度的持久化。\n'
        '\n'
        '主要函数:\n'
        '    save_checkpoint -- 保存训练 checkpoint\n'
        '    load_checkpoint -- 加载训练 checkpoint\n'
        '"""\n'
    ),
    "scripts/evaluate.py": (
        '"""评估器。\n'
        '\n'
        '在测试集上评估训练好的控制器性能，收集 episode 统计数据。\n'
        '\n'
        '主要函数:\n'
        '    evaluate_controller -- 评估控制器性能\n'
        '"""\n'
    ),
    "scripts/comparison.py": (
        '"""多控制器对比评估。\n'
        '\n'
        '同时评估多个控制器（如 MADRL vs MPC vs Zero）并生成对比报告。\n'
        '\n'
        '主要函数:\n'
        '    compare_controllers -- 对比多个控制器\n'
        '"""\n'
    ),
    "scripts/run_debug_training.py": (
        '"""快速调试训练入口。\n'
        '\n'
        '使用 debug profile 快速启动一次小规模训练，\n'
        '用于验证训练流程是否正常。\n'
        '"""\n'
    ),
    "scripts/export_project_code.py": (
        '"""项目代码导出工具。\n'
        '\n'
        '将项目所有源代码文件合并导出为单个文本文件，\n'
        '方便代码审查或提交。\n'
        '"""\n'
    ),
    "scripts/plots/plots.py": (
        '"""通用训练曲线绘制。\n'
        '\n'
        '绘制训练过程中的奖励曲线、损失曲线等可视化图表。\n'
        '\n'
        '主要函数:\n'
        '    plot_training_curves -- 绘制训练曲线\n'
        '"""\n'
    ),
    "scripts/plots/grid_plots.py": (
        '"""电网潮流结果可视化。\n'
        '\n'
        '绘制电压分布、线路负载、功率注入等电网相关图表。\n'
        '\n'
        '主要函数:\n'
        '    plot_voltage_profile -- 绘制电压分布\n'
        '    plot_line_loading    -- 绘制线路负载\n'
        '"""\n'
    ),
    "scripts/plots/reward_plots.py": (
        '"""奖励分量分析图。\n'
        '\n'
        '将复合奖励拆分为各分量并可视化，便于调试奖励设计。\n'
        '\n'
        '主要函数:\n'
        '    plot_reward_breakdown -- 绘制奖励分量分解图\n'
        '"""\n'
    ),
    "scripts/recorders/episode_recorder.py": (
        '"""Episode 数据记录器。\n'
        '\n'
        '记录每个 episode 的状态、动作、奖励等轨迹数据。\n'
        '\n'
        '主要类:\n'
        '    EpisodeRecorder -- Episode 数据记录器\n'
        '"""\n'
    ),
    "scripts/recorders/grid_recorder.py": (
        '"""电网潮流数据记录器。\n'
        '\n'
        '记录每步的电压、线路负载等电网潮流计算结果。\n'
        '\n'
        '主要类:\n'
        '    GridRecorder -- 电网数据记录器\n'
        '"""\n'
    ),
    "scripts/utils/project_paths.py": (
        '"""项目路径工具。\n'
        '\n'
        '提供项目根目录、数据目录等路径发现函数。\n'
        '\n'
        '主要函数:\n'
        '    project_root -- 返回项目根目录 Path\n'
        '    get_data_root -- 返回数据目录 Path\n'
        '"""\n'
    ),
    "scripts/utils/nested.py": (
        '"""嵌套 tensor 操作工具。\n'
        '\n'
        '提供对嵌套字典结构中 tensor 的批量操作（stack、squeeze 等）。\n'
        '\n'
        '主要函数:\n'
        '    stack_nested -- 批量堆叠嵌套 tensor\n'
        '"""\n'
    ),
    "scripts/utils/torch_runtime.py": (
        '"""PyTorch 运行时配置。\n'
        '\n'
        '管理设备选择、随机种子、TF32 精度、cuDNN 后端等\n'
        '运行时参数的统一配置入口。\n'
        '\n'
        '主要函数:\n'
        '    resolve_device       -- 解析设备（CPU/CUDA）\n'
        '    resolve_runtime_mode -- 解析运行时模式\n'
        '    setup_runtime        -- 应用运行时配置\n'
        '\n'
        '主要常量:\n'
        '    PERFORMANCE_RUNTIME_MODE  -- 性能模式标识\n'
        '    STRICT_REPRO_RUNTIME_MODE -- 严格复现模式标识\n'
        '"""\n'
    ),
    "scripts/utils/replay_buffer.py": (
        '"""经验回放缓冲区。\n'
        '\n'
        '为 off-policy MADRL 算法提供经验存储与批量采样功能。\n'
        '\n'
        '主要类:\n'
        '    ReplayBuffer -- 经验回放缓冲区\n'
        '"""\n'
    ),
    "scripts/utils/experiment_notebook_utils.py": (
        '"""Jupyter notebook 实验辅助工具。\n'
        '\n'
        '提供 notebook 中常用的实验管理函数，如结果加载、进度显示等。\n'
        '"""\n'
    ),
}


def update_docstring(filepath: Path, new_docstring: str) -> bool:
    """Replace or add a module-level docstring."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        print(f"  [SKIP] Cannot read {filepath}")
        return False

    # Try to parse and find existing docstring
    try:
        tree = ast.parse(text)
    except SyntaxError:
        print(f"  [SKIP] Syntax error in {filepath}")
        return False

    # Check if there's an existing module docstring
    has_docstring = (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    )

    if has_docstring:
        # Replace existing docstring
        old_ds = tree.body[0]
        # Find the old docstring in the text
        lines = text.split('\n')
        # The docstring starts at old_ds.lineno (1-indexed) and ends at old_ds.end_lineno
        start_line = old_ds.lineno - 1
        end_line = old_ds.end_lineno
        # Replace those lines
        new_lines = lines[:start_line] + [new_docstring.rstrip()] + lines[end_line:]
        new_text = '\n'.join(new_lines)
    else:
        # Prepend the docstring
        new_text = new_docstring + '\n' + text

    if new_text != text:
        filepath.write_text(new_text, encoding="utf-8")
        print(f"  [DOCSTRING] {filepath.relative_to(ROOT)}")
        return True
    return False


def main():
    print("Adding module-level docstrings to all refactored files")
    print("=" * 60)

    updated = 0
    for rel_path, docstring in DOCSTRINGS.items():
        filepath = ROOT / rel_path
        if not filepath.exists():
            print(f"  [MISS] {rel_path} not found")
            continue
        if update_docstring(filepath, docstring):
            updated += 1

    print(f"\nUpdated docstrings in {updated} files")


if __name__ == "__main__":
    main()
