from __future__ import annotations

from scripts.builder import build_env
from scripts.utils.grid_notebook_workflow import apply_notebook_experiment_settings, ensure_forecast_ready


def build_mainline_env(cfg):
    return build_env(cfg)


def apply_mainline_experiment_settings(cfg, **kwargs):
    return apply_notebook_experiment_settings(cfg, **kwargs)


def ensure_mainline_forecast_ready(cfg):
    return ensure_forecast_ready(cfg)


def build_mainline_mpc_env(cfg):
    return build_mainline_env(cfg)


def apply_mainline_mpc_settings(cfg, **kwargs):
    return apply_mainline_experiment_settings(cfg, **kwargs)


def ensure_mainline_mpc_forecast_ready(cfg):
    return ensure_mainline_forecast_ready(cfg)


def build_mainline_grid_analysis_env(cfg):
    return build_mainline_env(cfg)


def apply_mainline_grid_analysis_settings(cfg, **kwargs):
    return apply_mainline_experiment_settings(cfg, **kwargs)


def ensure_mainline_grid_analysis_forecast_ready(cfg):
    return ensure_mainline_forecast_ready(cfg)


__all__ = ['apply_mainline_experiment_settings', 'apply_mainline_grid_analysis_settings', 'apply_mainline_mpc_settings', 'build_mainline_env', 'build_mainline_grid_analysis_env', 'build_mainline_mpc_env', 'ensure_mainline_forecast_ready', 'ensure_mainline_grid_analysis_forecast_ready', 'ensure_mainline_mpc_forecast_ready']
