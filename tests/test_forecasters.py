import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from configs.experiment_config import ExperimentConfig
from forecast.lstm_forecaster import (
    LSTMForecaster,
    load_lstm_forecaster_artifacts,
    save_lstm_forecaster_artifacts,
)
from forecast.lstm_model import LSTMPricePredictor
from forecast.naive import NaiveForecaster
from forecast.oracle import PerfectForecaster
from forecast.registry import build_forecaster
from forecast.training import (
    SignalForecastEvaluation,
    collect_available_lstm_artifacts,
    compare_lstm_artifact_meta,
    ensure_lstm_artifacts,
    expected_lstm_artifact_meta,
    plot_signal_training_report,
    plot_weekly_forecasts,
    validate_lstm_artifact,
)

def test_perfect_forecaster_predict_length():
    forecaster = PerfectForecaster(np.array([1, 2, 3, 4], dtype=np.float32))
    pred = forecaster.predict(np.array([1, 2], dtype=np.float32), horizon=5)

    assert pred.shape == (5,)
    assert pred.dtype == np.float32
    assert pred[0] == np.float32(2.0)


def test_perfect_forecaster_supports_per_agent_signals():
    forecaster = PerfectForecaster(
        {
            "load": np.array(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                ],
                dtype=np.float32,
            )
        }
    )

    history = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    pred = forecaster.predict(history, horizon=3, signal_name="load")

    assert pred.shape == (2, 3)
    assert np.allclose(pred[:, 0], np.array([2.0, 20.0], dtype=np.float32))


def test_naive_forecaster_predict_length():
    forecaster = NaiveForecaster(window=3)
    pred = forecaster.predict(np.array([1, 2, 3, 4], dtype=np.float32), horizon=6)

    assert pred.shape == (6,)
    assert pred.dtype == np.float32
    assert pred[0] == np.float32(4.0)
    assert np.allclose(pred[1:], np.array([3, 3, 3, 3, 3], dtype=np.float32))


def test_lstm_forecaster_rolls_predictions_forward():
    forecaster = LSTMForecaster(
        model_path=None,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        pred_len=2,
        seq_len=4,
        device="cpu",
        scaler=None,
    )

    class FakeRolloutModel(torch.nn.Module):
        def forward(self, x):
            last = x[:, -1]
            return torch.stack([last + 1.0, last + 2.0], dim=1)

    forecaster.model = FakeRolloutModel()
    pred = forecaster.predict(np.array([1.0, 2.0, 3.0], dtype=np.float32), horizon=5)

    assert pred.shape == (5,)
    assert pred.dtype == np.float32
    assert np.allclose(pred, np.array([3, 4, 5, 6, 7], dtype=np.float32))


def test_lstm_forecaster_supports_per_agent_histories():
    forecaster = LSTMForecaster(
        model_path=None,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        pred_len=2,
        seq_len=4,
        device="cpu",
        scaler=None,
    )

    class FakeRolloutModel(torch.nn.Module):
        def forward(self, x):
            last = x[:, -1]
            return torch.stack([last + 1.0, last + 2.0], dim=1)

    forecaster.model = FakeRolloutModel()
    history = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    pred = forecaster.predict(history, horizon=4, signal_name="load")

    assert pred.shape == (2, 4)
    assert np.allclose(pred[0], np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32))
    assert np.allclose(pred[1], np.array([30.0, 31.0, 32.0, 33.0], dtype=np.float32))


def test_build_forecaster_loads_lstm_artifacts(tmp_path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [10.0]], dtype=np.float32))

    model = LSTMPricePredictor(hidden_size=8, num_layers=1, dropout=0.0, pred_len=2)
    artifact_paths = save_lstm_forecaster_artifacts(
        model_path=tmp_path / "best_lstm.pt",
        state_dict=model.state_dict(),
        scaler=scaler,
        seq_len=6,
        pred_len=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )

    cfg = ExperimentConfig()
    cfg.forecast.type = "lstm"
    cfg.forecast.lstm_model_path = artifact_paths["model_path"]
    cfg.runtime.device = "cpu"
    forecaster = build_forecaster(cfg)

    assert isinstance(forecaster, LSTMForecaster)
    assert forecaster.seq_len == 6
    assert forecaster.pred_len == 2
    assert forecaster.scaler is not None


def test_build_forecaster_prefers_managed_multi_signal_artifacts(tmp_path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [1.0]], dtype=np.float32))
    cfg = ExperimentConfig()
    cfg.forecast.type = "lstm"
    cfg.forecast.lstm_artifact_root = tmp_path / "forecast_lstm"
    cfg.forecast.target_signals = ["price", "load", "pv"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.env.future_horizon = 6
    cfg.forecast.history_window = 12
    cfg.forecast.lstm_hidden_size = 4
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0
    cfg.runtime.device = torch.device("cpu")

    for signal_name in ["price", "load", "pv"]:
        model = LSTMPricePredictor(hidden_size=4, num_layers=1, dropout=0.0, pred_len=6)
        signal_dir = cfg.forecast.lstm_artifact_root / "h6" / signal_name
        signal_dir.mkdir(parents=True, exist_ok=True)
        save_lstm_forecaster_artifacts(
            model_path=signal_dir / f"{signal_name}_lstm_h6.pt",
            state_dict=model.state_dict(),
            scaler=scaler,
            seq_len=12,
            pred_len=6,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            signal_name=signal_name,
            future_horizon=6,
        )

    forecaster = build_forecaster(cfg)

    assert isinstance(forecaster, LSTMForecaster)
    assert forecaster.available_signals() == ["load", "price", "pv"]


def test_collect_available_lstm_artifacts_is_horizon_specific(tmp_path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [1.0]], dtype=np.float32))
    model = LSTMPricePredictor(hidden_size=4, num_layers=1, dropout=0.0, pred_len=24)

    cfg = ExperimentConfig()
    cfg.forecast.lstm_artifact_root = tmp_path / "forecast_lstm"
    cfg.env.future_horizon = 24
    cfg.forecast.history_window = 96
    cfg.forecast.lstm_hidden_size = 4
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0

    artifact_dir = cfg.forecast.lstm_artifact_root / "h24" / "price"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_lstm_forecaster_artifacts(
        model_path=artifact_dir / "price_lstm_h24.pt",
        state_dict=model.state_dict(),
        scaler=scaler,
        seq_len=96,
        pred_len=24,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        signal_name="price",
        future_horizon=24,
    )

    artifact_map = collect_available_lstm_artifacts(cfg)
    assert "price" in artifact_map

    cfg.env.future_horizon = 36
    assert collect_available_lstm_artifacts(cfg) == {}


def test_compare_lstm_artifact_meta_reports_mismatches():
    cfg = ExperimentConfig()
    cfg.env.future_horizon = 24
    cfg.forecast.history_window = 192
    cfg.forecast.lstm_hidden_size = 64
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0

    expected_meta = expected_lstm_artifact_meta(cfg, "price")
    actual_meta = dict(expected_meta)
    actual_meta["hidden_size"] = 32
    actual_meta["dropout"] = 0.1

    comparison = compare_lstm_artifact_meta(actual_meta, expected_meta)

    assert comparison["compatible"] is False
    assert comparison["mismatches"]["hidden_size"] == {"expected": 64, "actual": 32}
    assert comparison["mismatches"]["dropout"] == {"expected": 0.0, "actual": 0.1}


def test_validate_lstm_artifact_accepts_compatible_meta(tmp_path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [1.0]], dtype=np.float32))
    cfg = ExperimentConfig()
    cfg.forecast.lstm_artifact_root = tmp_path / "forecast_lstm"
    cfg.env.future_horizon = 24
    cfg.forecast.history_window = 96
    cfg.forecast.lstm_hidden_size = 4
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0

    artifact_dir = cfg.forecast.lstm_artifact_root / "h24" / "price"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = LSTMPricePredictor(hidden_size=4, num_layers=1, dropout=0.0, pred_len=24)
    saved_paths = save_lstm_forecaster_artifacts(
        model_path=artifact_dir / "price_lstm_h24.pt",
        state_dict=model.state_dict(),
        scaler=scaler,
        seq_len=96,
        pred_len=24,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        signal_name="price",
        future_horizon=24,
    )

    validation = validate_lstm_artifact(
        cfg,
        "price",
        {
            "artifact_dir": artifact_dir,
            "model_path": saved_paths["model_path"],
            "meta_path": saved_paths["meta_path"],
            "scaler_path": saved_paths["scaler_path"],
        },
    )

    assert validation["compatible"] is True
    assert validation["mismatches"] == {}


def test_build_forecaster_rejects_incompatible_managed_artifact_when_auto_train_disabled(tmp_path):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [1.0]], dtype=np.float32))
    cfg = ExperimentConfig()
    cfg.forecast.type = "lstm"
    cfg.forecast.auto_train_missing = False
    cfg.forecast.lstm_artifact_root = tmp_path / "forecast_lstm"
    cfg.forecast.target_signals = ["price"]
    cfg.obs.sequence_features = ["price"]
    cfg.env.future_horizon = 24
    cfg.forecast.history_window = 192
    cfg.forecast.lstm_hidden_size = 64
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0
    cfg.runtime.device = "cpu"

    artifact_dir = cfg.forecast.lstm_artifact_root / "h24" / "price"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = LSTMPricePredictor(hidden_size=32, num_layers=1, dropout=0.1, pred_len=24)
    save_lstm_forecaster_artifacts(
        model_path=artifact_dir / "price_lstm_h24.pt",
        state_dict=model.state_dict(),
        scaler=scaler,
        seq_len=192,
        pred_len=24,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        signal_name="price",
        future_horizon=24,
    )

    with pytest.raises(ValueError, match=r"(?s)signal='price'.*(hidden_size|dropout)"):
        build_forecaster(cfg)


def test_ensure_lstm_artifacts_retrains_incompatible_artifact(tmp_path, monkeypatch):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [1.0]], dtype=np.float32))
    cfg = ExperimentConfig()
    cfg.forecast.auto_train_missing = True
    cfg.forecast.lstm_artifact_root = tmp_path / "forecast_lstm"
    cfg.forecast.target_signals = ["price"]
    cfg.obs.sequence_features = ["price"]
    cfg.env.future_horizon = 24
    cfg.forecast.history_window = 192
    cfg.forecast.lstm_hidden_size = 64
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0
    cfg.runtime.device = "cpu"

    artifact_dir = cfg.forecast.lstm_artifact_root / "h24" / "price"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stale_model = LSTMPricePredictor(hidden_size=32, num_layers=1, dropout=0.1, pred_len=24)
    save_lstm_forecaster_artifacts(
        model_path=artifact_dir / "price_lstm_h24.pt",
        state_dict=stale_model.state_dict(),
        scaler=scaler,
        seq_len=192,
        pred_len=24,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        signal_name="price",
        future_horizon=24,
    )

    import forecast.training as training

    def fake_train_signal_lstm(local_cfg, signal_name, *, device=None, overrides=None, show_progress=False):
        fresh_model = LSTMPricePredictor(hidden_size=64, num_layers=1, dropout=0.0, pred_len=24)
        saved_paths = save_lstm_forecaster_artifacts(
            model_path=artifact_dir / "price_lstm_h24.pt",
            state_dict=fresh_model.state_dict(),
            scaler=scaler,
            seq_len=192,
            pred_len=24,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
            signal_name="price",
            future_horizon=24,
        )
        evaluation = type(
            "Eval",
            (),
            {
                "signal_name": "price",
                "evaluation_mode": "online_aligned",
                "timestamps": np.arange(3),
                "target": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "prediction": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "metrics": {"rmse": 0.0, "mae": 0.0, "mape": 0.0},
                "source_columns": ("price",),
            },
        )()
        return {
            "signal_name": signal_name,
            "artifact_paths": saved_paths,
            "evaluation": evaluation,
        }

    monkeypatch.setattr(training, "train_signal_lstm", fake_train_signal_lstm)
    monkeypatch.setattr(training, "plot_weekly_forecasts", lambda evaluations, save_path: save_path)

    result = ensure_lstm_artifacts(cfg, device="cpu")

    assert result["retrained_signals"] == ["price"]
    assert result["trained_signals"] == []
    assert "price" in result["artifacts"]
    validation = validate_lstm_artifact(
        cfg,
        "price",
        {
            "artifact_dir": artifact_dir,
            "model_path": artifact_dir / "price_lstm_h24.pt",
            "meta_path": artifact_dir / "price_lstm_h24_meta.json",
            "scaler_path": artifact_dir / "price_lstm_h24_scaler.pkl",
        },
    )
    assert validation["compatible"] is True


def test_plotting_functions_emit_no_glyph_warnings(tmp_path):
    evaluation = SignalForecastEvaluation(
        signal_name="price",
        evaluation_mode="online_aligned",
        timestamps=np.arange(4),
        target=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        prediction=np.array([1.1, 1.9, 3.1, 4.0], dtype=np.float32),
        metrics={"rmse": 0.1, "mae": 0.1, "mape": 1.0},
        source_columns=("price",),
    )
    open_loop_evaluation = SignalForecastEvaluation(
        signal_name="price",
        evaluation_mode="open_loop",
        timestamps=np.arange(4),
        target=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        prediction=np.array([1.2, 2.3, 2.8, 3.7], dtype=np.float32),
        metrics={"rmse": 0.2, "mae": 0.2, "mape": 2.0},
        source_columns=("price",),
    )
    result = {
        "signal_name": "price",
        "training": {"history": {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}},
        "evaluation": evaluation,
        "open_loop_evaluation": open_loop_evaluation,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = plot_signal_training_report(result)
        figure.savefig(tmp_path / "report.png")
        plt.close(figure)
        plot_weekly_forecasts([evaluation, open_loop_evaluation], save_path=tmp_path / "weekly.png")

    assert [str(item.message) for item in caught] == []


def test_lstm_artifact_loader_fails_when_sidecars_are_missing(tmp_path):
    model_path = tmp_path / "best_lstm.pt"
    torch.save({}, model_path)

    with pytest.raises(FileNotFoundError, match="Missing LSTM meta artifact"):
        load_lstm_forecaster_artifacts(model_path=model_path)


def test_lstm_artifact_loader_fails_when_meta_fields_are_missing(tmp_path):
    model_path = tmp_path / "best_lstm.pt"
    meta_path = tmp_path / "best_lstm_meta.json"
    scaler_path = tmp_path / "best_lstm_scaler.pkl"

    torch.save({}, model_path)
    meta_path.write_text('{"seq_len": 6, "pred_len": 2}', encoding="utf-8")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0.0], [1.0]], dtype=np.float32))
    with scaler_path.open("wb") as f:
        import pickle

        pickle.dump(scaler, f)

    with pytest.raises(ValueError, match="missing required fields"):
        load_lstm_forecaster_artifacts(model_path=model_path)
