"""预测器模块（原 forecast/）。

为强化学习环境提供多步时序预测，包括电价、负荷、光伏等信号。

已实现预测器:
    PerfectForecaster -- Oracle 完美预知（上界基线）
    NaiveForecaster   -- 基于历史窗口复制的朴素预测
    LSTMForecaster    -- LSTM 深度学习预测器

主要函数:
    build_forecaster -- 根据配置构建预测器实例

如何添加新预测器:
    1. 创建 ``predictors/your_forecaster.py``，继承 ``Forecaster``
    2. 在 ``predictors/registry.py`` 中注册
    3. 在配置中使用: ``cfg.forecast.type = "your_type"``
"""

from predictors.base import Forecaster
from predictors.naive import NaiveForecaster
from predictors.oracle import PerfectForecaster
from predictors.registry import FORECASTER_REGISTRY, build_forecaster

__all__ = [
    "Forecaster",
    "FORECASTER_REGISTRY",
    "NaiveForecaster",
    "PerfectForecaster",
    "build_forecaster",
]
