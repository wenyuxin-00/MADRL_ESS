import numpy as np
import pytest
import torch
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


def test_perfect_forecaster_predict_length():
    forecaster = PerfectForecaster(np.array([1, 2, 3, 4], dtype=np.float32))
    pred = forecaster.predict(np.array([1, 2], dtype=np.float32), horizon=5)

    assert pred.shape == (5,)
    assert pred.dtype == np.float32
    assert pred[0] == np.float32(2.0)


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
