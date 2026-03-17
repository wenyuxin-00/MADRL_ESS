import numpy as np

from forecast import notebook_utils


class _FakeForecaster:
    def predict(self, history, horizon):
        history = np.asarray(history, dtype=np.float32)
        start = float(history[-1])
        return np.asarray([start + step for step in range(horizon)], dtype=np.float32)


def test_prepare_lstm_data_bundle_shapes():
    price_series = np.arange(20, dtype=np.float32)

    bundle = notebook_utils.prepare_lstm_data_bundle(
        price_series,
        seq_len=4,
        pred_len=2,
        train_ratio=0.6,
        val_ratio=0.2,
        batch_size=8,
    )

    assert set(bundle["splits"]) == {"train", "val", "test"}
    assert bundle["train_loader"] is not None
    assert bundle["val_loader"] is not None
    assert bundle["test_history_seed"].shape[0] == bundle["val_end"]


def test_evaluate_one_step_forecast_matches_fake_runtime(monkeypatch):
    monkeypatch.setattr(notebook_utils, "_build_runtime_forecaster", lambda *args, **kwargs: _FakeForecaster())

    result = notebook_utils.evaluate_one_step_forecast(
        model=object(),
        history_seed=np.array([1.0, 2.0], dtype=np.float32),
        target_series=np.array([3.0, 4.0], dtype=np.float32),
        scaler=None,
        seq_len=4,
        pred_len=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        device="cpu",
    )

    assert np.allclose(result["predictions"], np.array([3.0, 4.0], dtype=np.float32))
    assert result["metrics"]["mae"] == 0.0


def test_evaluate_block_forecast_summarizes_runtime_like_windows(monkeypatch):
    monkeypatch.setattr(notebook_utils, "_build_runtime_forecaster", lambda *args, **kwargs: _FakeForecaster())

    result = notebook_utils.evaluate_block_forecast(
        model=object(),
        history_seed=np.array([1.0, 2.0], dtype=np.float32),
        target_series=np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        scaler=None,
        seq_len=4,
        pred_len=2,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        device="cpu",
        forecast_horizon=2,
        block_stride=2,
    )

    assert result["metrics"]["n_blocks"] == 2
    assert result["metrics"]["mean_block_mae"] == 0.0
    assert np.allclose(result["blocks"][0]["prediction"], np.array([3.0, 4.0], dtype=np.float32))