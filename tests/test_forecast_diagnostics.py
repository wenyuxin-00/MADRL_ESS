from __future__ import annotations

import numpy as np
import pandas as pd

from predictors.training import train_signal_lstm
from scripts.utils.forecast_diagnostics import (
    Step1WindowPayload,
    apply_heatpump_jump_relief,
    build_component_drift_report,
    build_profile_monthly_error_report,
    collect_load_step1_diagnostics,
    select_blocked_blend_weight,
)
from tests.support.helpers import DEFAULT_TEST_BUSES, make_smoke_config, write_prosumer_processed_dataset


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
    cfg.forecast.load_component_split = True
    cfg.forecast.auto_train_missing = False
    return cfg


def _make_five_agent_load_cfg(tmp_path):
    cfg = _make_load_only_cfg(tmp_path)
    agent_profiles = ["SFH12", "SFH14", "SFH16", "SFH18", "SFH20"]
    data_dir = tmp_path / "data_five_agent"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=agent_profiles,
        train_steps=12,
        test_steps=12,
    )
    cfg.data.data_dir = data_dir
    cfg.data.agent_profiles = agent_profiles
    cfg.env.num_agents = len(agent_profiles)
    cfg.grid.agent_bus_ids = DEFAULT_TEST_BUSES[: cfg.env.num_agents]
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


def test_apply_heatpump_jump_relief_disabled_is_identity() -> None:
    timestamps = pd.date_range("2020-01-04", periods=4, freq="15min", tz="UTC")
    payload = Step1WindowPayload(
        profile="SFH14",
        timestamps=pd.DatetimeIndex(timestamps),
        target=np.array([0.2, 0.3, 1.2, 0.4], dtype=np.float32),
        baseline=np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float32),
        raw=np.array([0.2, 0.3, 1.6, 0.4], dtype=np.float32),
        blended=np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float32),
        blend_weight=0.0,
    )

    result = apply_heatpump_jump_relief(payload, enabled=False, threshold_kw=0.8, min_weight=0.3)

    assert result["triggered_count"] == 0
    assert np.array_equal(result["trigger_mask"], np.array([False, False, False, False]))
    assert np.allclose(result["effective_weights"], 0.0)
    assert np.allclose(result["payload"].blended, payload.blended)


def test_apply_heatpump_jump_relief_raises_only_triggered_points() -> None:
    timestamps = pd.date_range("2020-01-04", periods=4, freq="15min", tz="UTC")
    payload = Step1WindowPayload(
        profile="SFH14",
        timestamps=pd.DatetimeIndex(timestamps),
        target=np.array([0.2, 0.3, 1.2, 0.4], dtype=np.float32),
        baseline=np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float32),
        raw=np.array([0.2, 0.9, 1.6, 0.6], dtype=np.float32),
        blended=np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float32),
        blend_weight=0.0,
    )

    result = apply_heatpump_jump_relief(payload, enabled=True, threshold_kw=0.8, min_weight=0.3)

    assert result["triggered_count"] == 1
    assert np.array_equal(result["trigger_mask"], np.array([False, False, True, False]))
    assert np.allclose(result["effective_weights"], np.array([0.0, 0.0, 0.3, 0.0], dtype=np.float32))
    expected_blended = np.array([0.2, 0.3, 0.55, 0.4], dtype=np.float32)
    assert np.allclose(result["payload"].blended, expected_blended)
    assert list(result["triggered_timestamps"]) == [timestamps[2]]


def test_collect_load_step1_diagnostics_smoke(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    load_result = train_signal_lstm(cfg, "load", show_progress=False)

    diagnostics = collect_load_step1_diagnostics(cfg, load_result=load_result)
    monthly = build_profile_monthly_error_report(diagnostics["test_payloads"]["SFH14"], profile="SFH14")

    assert {"val2019", "test2020"} == set(diagnostics["step1_metrics"]["split"])
    assert {"baseline", "raw", "blended"} == set(diagnostics["step1_metrics"]["mode"])
    assert {"household", "heatpump", "total"} == set(diagnostics["step1_metrics"]["component"])
    assert set(diagnostics["step1_metrics"]["profile"]) == set(cfg.data.agent_profiles)
    assert set(monthly["month"]) == {1}


def test_collect_load_step1_diagnostics_accepts_notebook_load_overrides(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    load_overrides = {"hidden_size": 12, "num_layers": 2, "dropout": 0.1, "epochs": 1}

    train_signal_lstm(cfg, "load", overrides=load_overrides, show_progress=False)
    diagnostics = collect_load_step1_diagnostics(cfg, load_overrides=load_overrides)

    assert set(diagnostics["step1_metrics"]["component"]) == {"household", "heatpump", "total"}
    assert set(diagnostics["test_payloads"]) == set(cfg.data.agent_profiles)


def test_collect_load_step1_diagnostics_supports_sfh20_component_split(tmp_path) -> None:
    cfg = _make_five_agent_load_cfg(tmp_path)
    load_result = train_signal_lstm(cfg, "load", show_progress=False)

    diagnostics = collect_load_step1_diagnostics(cfg, load_result=load_result)
    focus_rows = diagnostics["step1_metrics"].loc[
        (diagnostics["step1_metrics"]["profile"] == "SFH20")
        & (diagnostics["step1_metrics"]["split"] == "test2020")
    ]

    assert set(focus_rows["component"]) == {"household", "heatpump", "total"}
    assert "SFH20" in diagnostics["test_payloads"]
