from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.loaders.registry import build_dataset
from envs.grid_env import GridEnv
from envs.observation.registry import build_obs_builder
from envs.rewards import NormalReward
from predictors.lstm_forecaster import (
    BASELINE_MODE_LAST_VALUE,
    LSTMForecaster,
    PHYSICAL_NORMALIZATION_LOAD_SCALE,
    PHYSICAL_NORMALIZATION_PV_PEAK,
    POSTPROCESS_MODE_BASELINE_BLEND,
    _SignalForecasterRuntime,
)
from predictors.training import (
    _select_load_blend_weight_from_validation,
    build_supervised_windows_from_matrix,
    compare_lstm_artifact_meta,
    expected_lstm_artifact_meta,
    fit_signal_scaler,
)
from predictors.time_features import TIME_FEATURE_MODE_HOUR_WEEK_YEAR
from tests.support.helpers import make_smoke_config


class EchoLastInputModel(torch.nn.Module):
    def __init__(self, pred_len: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)

    def forward(self, x):
        if x.ndim == 3:
            last_load = x[:, -1, 0].reshape(-1, 1)
        else:
            last_load = x[:, -1].reshape(-1, 1)
        return last_load.repeat(1, self.pred_len)


class ConstantChunkModel(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32).reshape(1, -1)

    def forward(self, x):
        batch = x.shape[0]
        return self.values.repeat(batch, 1)


class SpyForecaster:
    def __init__(self) -> None:
        self.last_episode_meta: dict[str, object] | None = None
        self.last_predict_timestamps: list[object] | None = None
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def set_episode(
        self,
        episode_signals: dict[str, np.ndarray] | np.ndarray,
        episode_meta: dict[str, object] | None = None,
    ) -> None:
        del episode_signals
        self.last_episode_meta = dict(episode_meta or {})

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        *,
        signal_name: str = "price",
        history_timestamps=None,
    ) -> np.ndarray:
        del signal_name
        self.last_predict_timestamps = None if history_timestamps is None else list(history_timestamps)
        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 1:
            return np.repeat(history[-1:], horizon).astype(np.float32)
        return np.repeat(history[-1:, :].T, horizon, axis=1).astype(np.float32)


class PassiveGridCore:
    def __init__(self, n_agents: int) -> None:
        self.n_buses = 4 + int(n_agents)
        self.n_lines = 3
        self.n_trafos = 1
        self.last_pf_error = ""

    def reset(self, base_load_kw, base_pv_kw) -> None:
        del base_load_kw, base_pv_kw


def _make_runtime_forecaster(
    signal_name: str,
    normalization_mode: str,
    *,
    input_size: int = 1,
    time_feature_mode: str = "none",
    model_mode: str = "shared",
    postprocess_mode: str = "none",
    baseline_mode: str = "none",
    blend_weight: float | None = None,
    model: torch.nn.Module | None = None,
) -> LSTMForecaster:
    runtime = _SignalForecasterRuntime(
        signal_name=signal_name,
        seq_len=3,
        pred_len=2,
        hidden_size=1,
        num_layers=1,
        dropout=0.0,
        model=EchoLastInputModel(pred_len=2) if model is None else model,
        scaler=None,
        input_size=input_size,
        time_feature_mode=time_feature_mode,
        model_mode=model_mode,
        physical_normalization_mode=normalization_mode,
        postprocess_mode=postprocess_mode,
        baseline_mode=baseline_mode,
        blend_weight=blend_weight,
    )
    return LSTMForecaster(device="cpu", signal_runtimes={signal_name: [runtime]})


def test_training_windows_are_scale_invariant_after_physical_normalization() -> None:
    base = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
            [3.0, 5.0],
            [4.0, 6.0],
            [5.0, 7.0],
        ],
        dtype=np.float32,
    )
    scale = np.array([5.0, 0.5], dtype=np.float32)
    scaled = base * scale[None, :]

    base_scaler = fit_signal_scaler(base, physical_scale_by_column=[1.0, 1.0])
    scaled_scaler = fit_signal_scaler(scaled, physical_scale_by_column=scale)
    base_x, base_y = build_supervised_windows_from_matrix(
        base,
        seq_len=2,
        pred_len=1,
        scaler=base_scaler,
        physical_scale_by_column=[1.0, 1.0],
    )
    scaled_x, scaled_y = build_supervised_windows_from_matrix(
        scaled,
        seq_len=2,
        pred_len=1,
        scaler=scaled_scaler,
        physical_scale_by_column=scale,
    )

    assert np.allclose(base_x, scaled_x)
    assert np.allclose(base_y, scaled_y)


def test_lstm_forecaster_load_prediction_scales_with_load_scale() -> None:
    forecaster = _make_runtime_forecaster(
        "load",
        PHYSICAL_NORMALIZATION_LOAD_SCALE,
        input_size=7,
        time_feature_mode=TIME_FEATURE_MODE_HOUR_WEEK_YEAR,
        model_mode="per_agent",
    )
    base_history = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=np.float32,
    )
    scaled_history = base_history * np.float32(5.0)
    history_timestamps = np.array(
        [
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:15:00+00:00",
            "2020-01-01T00:30:00+00:00",
        ]
    )

    forecaster.set_episode({"load": base_history}, {"load_scale": np.array([1.0, 1.0], dtype=np.float32)})
    base_prediction = forecaster.predict(
        base_history,
        horizon=3,
        signal_name="load",
        history_timestamps=history_timestamps,
    )

    forecaster.reset()
    forecaster.set_episode({"load": scaled_history}, {"load_scale": np.array([5.0, 5.0], dtype=np.float32)})
    scaled_prediction = forecaster.predict(
        scaled_history,
        horizon=3,
        signal_name="load",
        history_timestamps=history_timestamps,
    )

    assert np.allclose(scaled_prediction, base_prediction * np.float32(5.0))


def test_load_runtime_tracks_time_feature_input_shape() -> None:
    forecaster = _make_runtime_forecaster(
        "load",
        PHYSICAL_NORMALIZATION_LOAD_SCALE,
        input_size=7,
        time_feature_mode=TIME_FEATURE_MODE_HOUR_WEEK_YEAR,
        model_mode="per_agent",
    )

    runtime = forecaster.signal_runtimes["load"][0]
    assert runtime.input_size == 7
    assert runtime.time_feature_mode == TIME_FEATURE_MODE_HOUR_WEEK_YEAR
    assert runtime.model_mode == "per_agent"


def test_lstm_forecaster_pv_prediction_scales_with_episode_pv_peak() -> None:
    forecaster = _make_runtime_forecaster("pv", PHYSICAL_NORMALIZATION_PV_PEAK)
    base_history = np.array(
        [
            [0.4, 0.8],
            [0.8, 1.6],
            [1.2, 2.4],
        ],
        dtype=np.float32,
    )
    scaled_history = base_history * np.float32(5.0)

    forecaster.set_episode({"pv": base_history}, {"pv_peak_kw": np.array([2.0, 4.0], dtype=np.float32)})
    base_prediction = forecaster.predict(base_history, horizon=3, signal_name="pv")

    forecaster.reset()
    forecaster.set_episode({"pv": scaled_history}, {"pv_peak_kw": np.array([10.0, 20.0], dtype=np.float32)})
    scaled_prediction = forecaster.predict(scaled_history, horizon=3, signal_name="pv")

    assert np.allclose(scaled_prediction, base_prediction * np.float32(5.0))


def test_lstm_forecaster_zero_load_scale_collapses_prediction_to_zero() -> None:
    forecaster = _make_runtime_forecaster("load", PHYSICAL_NORMALIZATION_LOAD_SCALE)
    zero_history = np.zeros((3, 1), dtype=np.float32)

    forecaster.set_episode({"load": zero_history}, {"load_scale": np.array([0.0], dtype=np.float32)})
    prediction = forecaster.predict(zero_history, horizon=3, signal_name="load")

    assert np.allclose(prediction, np.zeros((1, 3), dtype=np.float32))


def test_load_baseline_blend_weight_zero_matches_last_value_baseline() -> None:
    forecaster = _make_runtime_forecaster(
        "load",
        PHYSICAL_NORMALIZATION_LOAD_SCALE,
        input_size=7,
        time_feature_mode=TIME_FEATURE_MODE_HOUR_WEEK_YEAR,
        model_mode="per_agent",
        postprocess_mode=POSTPROCESS_MODE_BASELINE_BLEND,
        baseline_mode=BASELINE_MODE_LAST_VALUE,
        blend_weight=0.0,
        model=ConstantChunkModel([10.0, 50.0]),
    )
    history = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    history_timestamps = np.array(
        [
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:15:00+00:00",
            "2020-01-01T00:30:00+00:00",
        ]
    )

    forecaster.set_episode({"load": history}, {"load_scale": np.array([1.0], dtype=np.float32)})
    prediction = forecaster.predict(
        history,
        horizon=4,
        signal_name="load",
        history_timestamps=history_timestamps,
    )

    assert np.allclose(prediction.reshape(-1), np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32))


def test_load_baseline_blend_weight_one_matches_raw_lstm_rollout() -> None:
    raw_forecaster = _make_runtime_forecaster(
        "load",
        PHYSICAL_NORMALIZATION_LOAD_SCALE,
        input_size=7,
        time_feature_mode=TIME_FEATURE_MODE_HOUR_WEEK_YEAR,
        model_mode="per_agent",
        model=ConstantChunkModel([10.0, 10.0]),
    )
    hybrid_forecaster = _make_runtime_forecaster(
        "load",
        PHYSICAL_NORMALIZATION_LOAD_SCALE,
        input_size=7,
        time_feature_mode=TIME_FEATURE_MODE_HOUR_WEEK_YEAR,
        model_mode="per_agent",
        postprocess_mode=POSTPROCESS_MODE_BASELINE_BLEND,
        baseline_mode=BASELINE_MODE_LAST_VALUE,
        blend_weight=1.0,
        model=ConstantChunkModel([10.0, 10.0]),
    )
    history = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    history_timestamps = np.array(
        [
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:15:00+00:00",
            "2020-01-01T00:30:00+00:00",
        ]
    )

    raw_forecaster.set_episode({"load": history}, {"load_scale": np.array([1.0], dtype=np.float32)})
    hybrid_forecaster.set_episode({"load": history}, {"load_scale": np.array([1.0], dtype=np.float32)})
    raw_prediction = raw_forecaster.predict(
        history,
        horizon=4,
        signal_name="load",
        history_timestamps=history_timestamps,
    )
    hybrid_prediction = hybrid_forecaster.predict(
        history,
        horizon=4,
        signal_name="load",
        history_timestamps=history_timestamps,
    )

    assert np.allclose(hybrid_prediction, raw_prediction)


def test_load_baseline_blend_rolls_forward_one_step_at_a_time() -> None:
    forecaster = _make_runtime_forecaster(
        "load",
        PHYSICAL_NORMALIZATION_LOAD_SCALE,
        input_size=7,
        time_feature_mode=TIME_FEATURE_MODE_HOUR_WEEK_YEAR,
        model_mode="per_agent",
        postprocess_mode=POSTPROCESS_MODE_BASELINE_BLEND,
        baseline_mode=BASELINE_MODE_LAST_VALUE,
        blend_weight=0.5,
        model=ConstantChunkModel([10.0, 50.0]),
    )
    history = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    history_timestamps = np.array(
        [
            "2020-01-01T00:00:00+00:00",
            "2020-01-01T00:15:00+00:00",
            "2020-01-01T00:30:00+00:00",
        ]
    )

    forecaster.set_episode({"load": history}, {"load_scale": np.array([1.0], dtype=np.float32)})
    prediction = forecaster.predict(
        history,
        horizon=4,
        signal_name="load",
        history_timestamps=history_timestamps,
    )

    expected = np.array([3.0, 6.5, 8.25, 9.125], dtype=np.float32)
    assert np.allclose(prediction.reshape(-1), expected)


def test_lstm_forecaster_rejects_non_positive_pv_peak() -> None:
    forecaster = _make_runtime_forecaster("pv", PHYSICAL_NORMALIZATION_PV_PEAK)

    with pytest.raises(ValueError, match="pv_peak_kw"):
        forecaster.set_episode(
            {"pv": np.ones((3, 1), dtype=np.float32)},
            {"pv_peak_kw": np.array([0.0], dtype=np.float32)},
        )


def test_artifact_meta_ignores_scale_changes_but_rejects_source_signature_changes(tmp_path) -> None:
    cfg = make_smoke_config(tmp_path)
    cfg.env.future_horizon = 2
    expected = expected_lstm_artifact_meta(cfg, "load")

    scaled_cfg = copy.deepcopy(cfg)
    scaled_cfg.data.load_scale = [5.0, 0.25]
    scaled_cfg.data.pv_scale = [4.0, 2.0]
    scaled_meta = expected_lstm_artifact_meta(scaled_cfg, "load")
    assert compare_lstm_artifact_meta(scaled_meta, expected)["compatible"]

    changed_cfg = copy.deepcopy(cfg)
    changed_cfg.data.agent_profiles = ["SFH12", "SFH16"]
    changed_meta = expected_lstm_artifact_meta(changed_cfg, "load")
    comparison = compare_lstm_artifact_meta(changed_meta, expected)
    assert not comparison["compatible"]
    assert "source_signature" in comparison["mismatches"]

    old_meta = dict(expected)
    old_meta["artifact_format"] = "lstm_forecaster_v4"
    old_meta.pop("postprocess_mode", None)
    old_meta.pop("baseline_mode", None)
    old_meta.pop("blend_weight", None)
    old_meta.pop("optimized_metric", None)
    old_comparison = compare_lstm_artifact_meta(old_meta, expected)
    assert not old_comparison["compatible"]
    assert "artifact_format" in old_comparison["mismatches"]
    assert "postprocess_mode" in old_comparison["mismatches"]
    assert "baseline_mode" in old_comparison["mismatches"]
    assert "blend_weight" in old_comparison["mismatches"]


def test_blend_weight_search_can_fallback_to_conservative_baseline() -> None:
    x_val = np.array(
        [
            [[0.5], [0.5], [0.5]],
            [[1.0], [1.0], [1.0]],
            [[1.5], [1.5], [1.5]],
        ],
        dtype=np.float32,
    )
    y_val = np.array(
        [
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
        ],
        dtype=np.float32,
    )
    runtime_state = SimpleNamespace(device=torch.device("cpu"))

    selection = _select_load_blend_weight_from_validation(
        model=ConstantChunkModel([5.0, 5.0]),
        x_val=x_val,
        y_val=y_val,
        scaler=None,
        physical_scale=None,
        runtime_state=runtime_state,
        batch_size=2,
        candidate_weights=(0.0, 0.5, 1.0),
    )

    assert selection["blend_weight"] == pytest.approx(0.0)
    assert selection["blended_mae_step1"] == pytest.approx(selection["baseline_mae_step1"])


def test_grid_env_reset_passes_episode_meta_to_forecaster(tmp_path) -> None:
    cfg = make_smoke_config(tmp_path)
    cfg.forecast.type = "perfect"
    dataset = build_dataset(cfg, mode="test")
    spy_forecaster = SpyForecaster()
    env = GridEnv(
        cfg,
        mode="test",
        dataset=dataset,
        reward_fn=NormalReward(cfg),
        forecaster=spy_forecaster,
        obs_builder=build_obs_builder(cfg),
        grid_core=PassiveGridCore(cfg.env.num_agents),
    )

    try:
        _, reset_info = env.reset(episode_idx=0)
        assert spy_forecaster.reset_calls == 1
        assert spy_forecaster.last_episode_meta is not None
        assert np.allclose(
            np.asarray(spy_forecaster.last_episode_meta["pv_peak_kw"], dtype=np.float32),
            np.asarray(reset_info["episode_meta"]["pv_peak_kw"], dtype=np.float32),
        )
        assert np.allclose(
            np.asarray(spy_forecaster.last_episode_meta["load_scale"], dtype=np.float32),
            np.asarray(reset_info["episode_meta"]["load_scale"], dtype=np.float32),
        )
        assert spy_forecaster.last_predict_timestamps is not None
        assert len(spy_forecaster.last_predict_timestamps) >= 1
    finally:
        env.close()
