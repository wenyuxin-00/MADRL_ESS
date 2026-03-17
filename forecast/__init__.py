"""Forecast package for price/load/pv observation windows."""

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
