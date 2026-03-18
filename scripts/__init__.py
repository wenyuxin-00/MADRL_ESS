"""入口脚本、评估工具和通用基础设施。

本模块集中了所有执行入口和辅助脚本：

子模块:
    builder      -- 实验组件构建入口
    train        -- 训练主循环（TrainRunner）
    checkpoints  -- checkpoint 保存与加载
    evaluate     -- 评估器
    comparison   -- 多控制器对比评估
    plots/       -- 训练曲线与电网结果可视化
    recorders/   -- Episode 与电网数据录制
    utils/       -- 通用工具（路径、tensor 操作、运行时、回放缓冲区）
"""
