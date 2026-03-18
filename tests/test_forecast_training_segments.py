import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from configs.experiment_config import ExperimentConfig
from datasets.csv_prosumer import CsvProsumerDataset
from forecast.training import (
    SignalCsvSource,
    evaluate_signal_online_one_week,
    load_signal_segments,
    train_signal_lstm,
)


def _write_segmented_simbench_exports(data_dir: Path, *, n_agents: int = 3) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, object]] = []
    for segment_id in range(2):
        train_start = pd.Timestamp("2016-01-01") + pd.Timedelta(days=segment_id * 14)
        for step in range(96):
            row = {
                "timestamp": str(train_start + pd.Timedelta(minutes=15 * step)),
                "price": 0.02 + 0.002 * np.sin(step / 6.0) + 0.0005 * segment_id,
                "segment_id": segment_id,
                "is_warmup": False,
                "week_id": -1,
                "quarter": segment_id + 1,
                "split_role": "train",
            }
            for agent in range(n_agents):
                row[f"load{agent + 1}"] = 1.0 + 0.1 * agent + 0.01 * step + 0.05 * segment_id
                row[f"pv{agent + 1}"] = 0.2 + 0.03 * agent + 0.004 * step
            train_rows.append(row)
    pd.DataFrame(train_rows).to_csv(data_dir / "simbench_2016_train.csv", index=False)

    test_rows: list[dict[str, object]] = []
    for segment_id in range(2):
        warmup_start = pd.Timestamp("2016-03-01") + pd.Timedelta(days=segment_id * 21)
        for step in range(12):
            row = {
                "timestamp": str(warmup_start + pd.Timedelta(minutes=15 * step)),
                "price": 0.018 + 0.001 * step + 0.0003 * segment_id,
                "segment_id": segment_id,
                "is_warmup": True,
                "week_id": segment_id * 3,
                "quarter": segment_id + 1,
                "split_role": "test_warmup",
            }
            for agent in range(n_agents):
                row[f"load{agent + 1}"] = 0.8 + 0.1 * agent + 0.02 * step
                row[f"pv{agent + 1}"] = 0.1 + 0.02 * agent + 0.003 * step
            test_rows.append(row)
        for week_offset in range(2):
            target_start = warmup_start + pd.Timedelta(days=7 * (week_offset + 1))
            for step in range(12):
                row = {
                    "timestamp": str(target_start + pd.Timedelta(minutes=15 * step)),
                    "price": 0.021 + 0.0015 * step + 0.0003 * segment_id,
                    "segment_id": segment_id,
                    "is_warmup": False,
                    "week_id": segment_id * 3 + week_offset + 1,
                    "quarter": segment_id + 1,
                    "split_role": "test_target",
                }
                for agent in range(n_agents):
                    row[f"load{agent + 1}"] = 1.2 + 0.1 * agent + 0.025 * step
                    row[f"pv{agent + 1}"] = 0.15 + 0.02 * agent + 0.004 * step
                test_rows.append(row)
    pd.DataFrame(test_rows).to_csv(data_dir / "simbench_2016_test.csv", index=False)

    (data_dir / "simbench_2016_metadata.json").write_text(
        json.dumps(
            {
                "selected_buses": [10, 6, 12],
                "pv_peak_kw": [4.0, 3.0, 2.0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_csv_prosumer_dataset_respects_segment_boundaries_and_skips_warmup(tmp_path):
    data_dir = tmp_path / "data"
    _write_segmented_simbench_exports(data_dir)

    dataset = CsvProsumerDataset(
        data_path=data_dir / "simbench_2016_test.csv",
        episode_length=12,
        n_agents=3,
        metadata_path=data_dir / "simbench_2016_metadata.json",
    )

    assert dataset.num_episodes() == 4
    first_episode = dataset.get_episode(0)
    third_episode = dataset.get_episode(2)

    assert first_episode["meta"]["segment_id"] == 0
    assert third_episode["meta"]["segment_id"] == 1
    assert np.isclose(first_episode["signals"]["price"][0], np.float32(0.021))
    assert np.isclose(third_episode["signals"]["price"][0], np.float32(0.0213))


def test_load_signal_segments_preserves_two_train_segments(tmp_path):
    data_dir = tmp_path / "data"
    _write_segmented_simbench_exports(data_dir)

    _, segments, columns = load_signal_segments(data_dir / "simbench_2016_train.csv", "load", drop_warmup=True)

    assert columns == ("load1", "load2", "load3")
    assert len(segments) == 2
    assert all(segment.shape == (96, 3) for segment in segments)


@pytest.mark.parametrize("signal_name", ["price", "load"])
def test_train_signal_lstm_supports_small_values_and_horizon_12(tmp_path, signal_name):
    data_dir = tmp_path / "data"
    _write_segmented_simbench_exports(data_dir)

    cfg = ExperimentConfig()
    cfg.data.data_dir = data_dir
    cfg.env.future_horizon = 12
    cfg.forecast.history_window = 12
    cfg.forecast.lstm_epochs = 1
    cfg.forecast.lstm_batch_size = 16
    cfg.forecast.lstm_hidden_size = 8
    cfg.forecast.lstm_num_layers = 1
    cfg.forecast.lstm_dropout = 0.0
    cfg.forecast.target_signals = [signal_name]
    cfg.forecast.lstm_artifact_root = tmp_path / "artifacts"
    cfg.runtime.device = "cpu"

    result = train_signal_lstm(cfg, signal_name, device="cpu")

    assert Path(result["artifact_paths"]["model_path"]).exists()
    assert Path(result["artifact_paths"]["meta_path"]).exists()
    assert Path(result["artifact_paths"]["scaler_path"]).exists()
    assert result["training"]["history"]["val_loss"]
    assert result["train_stats"]["max"] > result["train_stats"]["min"]
    assert result["evaluation"].evaluation_mode == "online_aligned"
    assert set(result["evaluation"].metrics) >= {"rmse", "mae"}
    assert result["open_loop_evaluation"].evaluation_mode == "open_loop"
    assert set(result["open_loop_evaluation"].metrics) >= {"rmse", "mae"}

    prediction = result["evaluation"].prediction
    if signal_name == "price":
        assert prediction.ndim == 1
        assert prediction.shape[0] == 12
    else:
        assert prediction.ndim == 2
        assert prediction.shape == (3, 12)


def test_evaluate_signal_online_one_week_uses_growing_real_history(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    rows = []
    for step, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], start=0):
        rows.append(
            {
                "timestamp": str(pd.Timestamp("2016-01-01") + pd.Timedelta(minutes=15 * step)),
                "price": value,
                "segment_id": 0,
                "is_warmup": step < 3,
                "week_id": 0 if step < 3 else 1,
            }
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "simbench_2016_test.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    source = SignalCsvSource(
        signal_name="price",
        train_path=csv_path,
        test_path=csv_path,
        value_columns=("price",),
    )

    calls: list[tuple[np.ndarray, int]] = []

    class MockForecaster:
        def predict(self, history, horizon, signal_name=None):
            history_array = np.asarray(history, dtype=np.float32).reshape(-1)
            calls.append((history_array.copy(), int(horizon)))
            return np.array([history_array[-1], history_array[-1] + 1.0], dtype=np.float32)

    import forecast.training as training

    monkeypatch.setattr(training, "build_runtime_forecaster_from_model", lambda *args, **kwargs: MockForecaster())

    evaluation = evaluate_signal_online_one_week(
        source,
        model=object(),
        signal_name="price",
        scaler=None,
        seq_len=3,
        pred_len=1,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        device="cpu",
    )

    assert len(calls) == 4
    assert [call[0].tolist() for call in calls] == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]
    assert [call[1] for call in calls] == [2, 2, 2, 2]
    assert evaluation.evaluation_mode == "online_aligned"
    assert np.allclose(evaluation.target, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
    assert np.allclose(evaluation.prediction, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
