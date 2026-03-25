from __future__ import annotations

import numpy as np
import pandas as pd

from predictors.training import train_signal_lstm
from scripts.utils.forecast_diagnostics import (
    Step1WindowPayload,
    build_component_drift_report,
    build_profile_monthly_error_report,
    collect_load_step1_diagnostics,
    run_sfh14_load_repair_report,
    select_blocked_blend_weight,
)
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


def test_build_component_drift_report_includes_total_rows(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)

    report = build_component_drift_report(cfg)

    assert set(report["component"]) == {"household", "heatpump", "total"}
    assert set(report["split"]) == {"train", "test"}
    assert set(report["profile"]) == set(cfg.data.agent_profiles)


def test_select_blocked_blend_weight_prefers_bias_safe_candidate() -> None:
    timestamps = pd.date_range("2019-01-01", periods=12, freq="MS", tz="UTC")
    target = np.ones(12, dtype=np.float32)
    baseline = np.ones(12, dtype=np.float32)
    raw = np.ones(12, dtype=np.float32) * 1.4
    payload = Step1WindowPayload(
        profile="SFH14",
        timestamps=pd.DatetimeIndex(timestamps),
        target=target,
        baseline=baseline,
        raw=raw,
        blended=raw.copy(),
        blend_weight=1.0,
    )

    selection = select_blocked_blend_weight(payload, candidate_weights=(0.0, 0.5, 1.0), bias_threshold_kw=0.15)

    assert selection["selected_weight"] == 0.0
    assert bool(selection["bias_guard_satisfied"]) is True


def test_collect_load_step1_diagnostics_smoke(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    load_result = train_signal_lstm(cfg, "load", show_progress=False)

    diagnostics = collect_load_step1_diagnostics(cfg, load_result=load_result)
    monthly = build_profile_monthly_error_report(diagnostics["test_payloads"]["SFH14"], profile="SFH14")

    assert {"val2019", "test2020"} == set(diagnostics["step1_metrics"]["split"])
    assert {"baseline", "raw", "blended"} == set(diagnostics["step1_metrics"]["mode"])
    assert set(diagnostics["step1_metrics"]["profile"]) == set(cfg.data.agent_profiles)
    assert set(monthly["month"]) == {1}


def test_run_sfh14_load_repair_report_smoke(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    load_result = train_signal_lstm(cfg, "load", show_progress=False)

    report = run_sfh14_load_repair_report(
        cfg,
        load_result=load_result,
        load_overrides={"epochs": 1, "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "batch_size": 4, "lr": 1e-3},
        output_dir=tmp_path / "analysis",
        show_progress=False,
    )

    assert {"step1_metrics", "sfh14_monthly_metrics", "component_drift", "experiment_summary"} <= set(report)
    assert "recommendation" in report
    assert "written_paths" in report
