"""Price / load forecasting module.
价格/负荷预测模块。

Provides three forecaster implementations:
    - PerfectForecaster (oracle)  -- uses ground-truth future prices (upper bound)
    - NaiveForecaster             -- repeats recent history as prediction (baseline)
    - LSTMForecaster              -- learned LSTM-based multi-step forecaster

How to add a new forecaster / 如何添加新预测器:
    1. Create a new file, e.g. ``forecast/transformer_forecaster.py``
    2. Implement the ``Forecaster`` base class from ``forecast.base``
    3. Register it in ``forecast/registry.py``::

           from forecast.transformer_forecaster import TransformerForecaster
           FORECASTER_REGISTRY["transformer"] = TransformerForecaster

    4. Use in config: ``cfg.forecast.type = "transformer"``

See also:
    - ``forecast/forecast.ipynb``     -- notebook for training & evaluating the LSTM
    - ``forecast/notebook_utils.py``  -- helper functions used by the notebook
    - ``forecast/artifacts.py``       -- unified artifact path conventions
"""

from forecast.base import Forecaster
from forecast.naive import NaiveForecaster
from forecast.oracle import PerfectForecaster
from forecast.registry import FORECASTER_REGISTRY, build_forecaster

__all__ = [
    "Forecaster",
    "FORECASTER_REGISTRY",
    "NaiveForecaster",
    "PerfectForecaster",
    "build_forecaster",
]
