"""Core builder module -- the single entry point for assembling experiments.
构建工厂模块 -- 实验组装的统一入口。

The builder orchestrates the full pipeline:
    ExperimentConfig -> [Dataset, Env, Forecaster, ObsBuilder, Reward, Models, Algo]
                     -> TrainRunner

Typical usage::

    from configs import compose_experiment_config
    from core import build_train_runner

    cfg = compose_experiment_config(train_profile="debug")
    runner = build_train_runner(cfg)
    runner.run()
"""

from core.builder import build_env, build_train_runner

__all__ = [
    "build_env",
    "build_train_runner",
]
