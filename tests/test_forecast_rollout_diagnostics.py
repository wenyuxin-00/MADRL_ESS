from __future__ import annotations

from predictors.training import train_signal_lstm
from scripts.utils.forecast_diagnostics import run_sfh14_load_repair_report
from tests.support.helpers import make_smoke_config


def _make_load_only_cfg(tmp_path):
    cfg = make_smoke_config(tmp_path)
    cfg.forecast.type = "lstm"
    cfg.forecast.target_signals = ["load"]
    cfg.obs.sequence_features = ["load"]
    cfg.forecast.lstm_artifact_root = tmp_path / "artifacts" / "forecast" / "lstm"
    cfg.forecast.history_window = 4
    cfg.env.future_horizon = 2
    cfg.forecast.lstm_epochs = 1
    cfg.forecast.lstm_batch_size = 4
    cfg.forecast.lstm_hidden_size = 8
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0
    cfg.forecast.load_model_mode = "per_agent"
    cfg.forecast.load_time_feature_mode = "hour_week_year"
    cfg.forecast.load_hybrid_mode = "baseline_blend"
    cfg.forecast.load_baseline_mode = "last_value"
    cfg.forecast.auto_train_missing = False
    return cfg


def test_run_sfh14_load_repair_report_includes_rollout_tables(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    load_result = train_signal_lstm(cfg, "load", show_progress=False)

    report = run_sfh14_load_repair_report(
        cfg,
        load_result=load_result,
        load_overrides={"epochs": 1, "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "batch_size": 4, "lr": 1e-3},
        output_dir=tmp_path / "analysis",
        show_progress=False,
    )

    assert {"horizon_metrics", "sfh14_month_horizon_metrics", "rollout_experiment_summary"} <= set(report)
    assert {"val2019", "test2020"} == set(report["horizon_metrics"]["split"])
    assert {
        "current_default",
        "E1_time_feature_none",
        "E2_blocked_blend",
        "E3_component_split",
    } <= set(report["horizon_metrics"]["experiment"])
    assert {1, 2} == set(report["sfh14_month_horizon_metrics"]["horizon"])
    assert {
        "current_default",
        "E1_time_feature_none",
        "E2_blocked_blend",
        "E3_component_split",
    } == set(report["rollout_experiment_summary"]["experiment"])
    assert {"horizon_metrics", "sfh14_month_horizon_metrics", "rollout_experiment_summary"} <= set(
        report["written_paths"]
    )
