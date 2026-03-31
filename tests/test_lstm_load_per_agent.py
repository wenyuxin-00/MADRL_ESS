from __future__ import annotations

import json
from pathlib import Path

import pytest

from predictors.lstm_forecaster import LSTMForecaster
from predictors.registry import build_forecaster
from predictors.training import collect_available_lstm_artifacts, train_signal_lstm
from tests.support.helpers import DEFAULT_TEST_BUSES, make_smoke_config, write_prosumer_processed_dataset


def _make_load_only_cfg(tmp_path):
    cfg = make_smoke_config(tmp_path)
    cfg.forecast.type = "lstm"
    cfg.forecast.target_signals = ["load"]
    cfg.obs.sequence_features = ["load"]
    cfg.forecast.lstm_artifact_root = Path(tmp_path) / "artifacts" / "forecast" / "lstm"
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


def _make_load_only_cfg_for_profiles(tmp_path, agent_profiles):
    cfg = _make_load_only_cfg(tmp_path)
    data_dir = Path(tmp_path) / "data_five_agent"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=list(agent_profiles),
        train_steps=12,
        test_steps=12,
    )
    cfg.data.data_dir = data_dir
    cfg.data.agent_profiles = list(agent_profiles)
    cfg.env.num_agents = len(agent_profiles)
    cfg.grid.agent_bus_ids = DEFAULT_TEST_BUSES[: cfg.env.num_agents]
    return cfg


def test_train_signal_lstm_load_returns_per_agent_results(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)

    result = train_signal_lstm(cfg, "load", show_progress=False)

    assert result["settings"]["model_mode"] == "per_agent"
    assert result["settings"]["time_feature_mode"] == "hour_week_year"
    assert result["settings"]["input_size"] == 7
    assert result["settings"]["postprocess_mode"] == "baseline_blend"
    assert result["settings"]["baseline_mode"] == "last_value"
    assert len(result["agent_results"]) == cfg.env.num_agents

    for agent_index, agent_result in enumerate(result["agent_results"]):
        meta_path = Path(agent_result["artifact_paths"]["meta_path"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert meta["artifact_format"] == "lstm_forecaster_v5"
        assert meta["agent_index"] == agent_index
        assert meta["agent_profile"] == cfg.data.agent_profiles[agent_index]
        assert meta["model_mode"] == "per_agent"
        assert meta["time_feature_mode"] == "hour_week_year"
        assert meta["input_size"] == 7
        assert meta["postprocess_mode"] == "baseline_blend"
        assert meta["baseline_mode"] == "last_value"
        assert 0.0 <= float(meta["blend_weight"]) <= 1.0
        assert meta["optimized_metric"] == "mae_step1"
        assert 0.0 <= float(agent_result["hybrid"]["blend_weight"]) <= 1.0
        assert meta_path.parent.name == f"agent_{agent_index}"

    artifact_map = collect_available_lstm_artifacts(cfg)
    assert "load" in artifact_map
    assert isinstance(artifact_map["load"], list)
    assert len(artifact_map["load"]) == cfg.env.num_agents


def test_train_signal_lstm_rejects_agent_profile_count_mismatch(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    cfg.env.num_agents = 5
    cfg.data.agent_profiles = ["SFH12", "SFH14", "SFH16"]

    with pytest.raises(ValueError, match="cfg.env.num_agents=5"):
        train_signal_lstm(cfg, "load", show_progress=False)


def test_train_signal_lstm_load_supports_five_agent_config(tmp_path) -> None:
    agent_profiles = ["SFH12", "SFH14", "SFH16", "SFH18", "SFH20"]
    cfg = _make_load_only_cfg_for_profiles(tmp_path, agent_profiles)

    result = train_signal_lstm(cfg, "load", show_progress=False)

    assert len(result["agent_results"]) == 5
    assert [entry["agent_profile"] for entry in result["agent_results"]] == agent_profiles
    artifact_map = collect_available_lstm_artifacts(cfg)
    assert len(artifact_map["load"]) == 5


def test_heatpump_component_uses_blocked_bias_guard_selection(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    cfg.forecast.load_component_split = True

    result = train_signal_lstm(cfg, "load", show_progress=False)
    heatpump_results = [entry for entry in result["agent_results"] if entry.get("component") == "heatpump"]
    household_results = [entry for entry in result["agent_results"] if entry.get("component") == "household"]

    assert heatpump_results
    assert household_results

    for entry in heatpump_results:
        meta_path = Path(entry["artifact_paths"]["meta_path"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["optimized_metric"] == "blocked_bias_guard_step1"
        assert entry["hybrid"]["optimized_metric"] == "blocked_bias_guard_step1"

    for entry in household_results:
        meta_path = Path(entry["artifact_paths"]["meta_path"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["optimized_metric"] == "mae_step1"


def test_build_forecaster_uses_per_agent_load_bundle(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    train_signal_lstm(cfg, "load", show_progress=False)

    forecaster = build_forecaster(cfg)

    assert isinstance(forecaster, LSTMForecaster)
    assert "load" in forecaster.signal_runtimes
    assert len(forecaster.signal_runtimes["load"]) == cfg.env.num_agents
    assert all(runtime.input_size == 7 for runtime in forecaster.signal_runtimes["load"])
    assert all(runtime.model_mode == "per_agent" for runtime in forecaster.signal_runtimes["load"])
    assert all(runtime.postprocess_mode == "baseline_blend" for runtime in forecaster.signal_runtimes["load"])
    assert all(runtime.baseline_mode == "last_value" for runtime in forecaster.signal_runtimes["load"])


def test_build_forecaster_reuses_cfg_signal_training_overrides(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    overrides = {"hidden_size": 12, "num_layers": 2, "dropout": 0.1, "epochs": 1}
    cfg.forecast.signal_training_overrides = {"load": dict(overrides)}
    cfg.forecast.auto_train_missing = False
    train_signal_lstm(cfg, "load", overrides=overrides, show_progress=False)

    forecaster = build_forecaster(cfg)

    assert isinstance(forecaster, LSTMForecaster)
    assert "load" in forecaster.signal_runtimes
    assert len(forecaster.signal_runtimes["load"]) == cfg.env.num_agents


def test_collect_available_lstm_artifacts_accepts_notebook_overrides(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    overrides = {"hidden_size": 12, "num_layers": 2, "dropout": 0.1, "epochs": 1}

    train_signal_lstm(cfg, "load", overrides=overrides, show_progress=False)

    artifact_map = collect_available_lstm_artifacts(cfg, overrides_by_signal={"load": overrides})
    assert "load" in artifact_map
    assert len(artifact_map["load"]) == cfg.env.num_agents


def test_collect_available_lstm_artifacts_uses_cfg_signal_training_overrides(tmp_path) -> None:
    cfg = _make_load_only_cfg(tmp_path)
    overrides = {"hidden_size": 12, "num_layers": 2, "dropout": 0.1, "epochs": 1}
    cfg.forecast.signal_training_overrides = {"load": dict(overrides)}

    train_signal_lstm(cfg, "load", overrides=overrides, show_progress=False)

    artifact_map = collect_available_lstm_artifacts(cfg)
    assert "load" in artifact_map
    assert len(artifact_map["load"]) == cfg.env.num_agents
