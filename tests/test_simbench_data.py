import numpy as np
import pandas as pd

from data.loaders.simbench_export import (
    STEPS_PER_WEEK,
    align_price_series_to_reference,
    build_quarterly_simbench_split,
)


# Note: legacy import 'from forecast.simbench_data import ...' was removed
# during project restructuring. The canonical import above now lives in
# data.loaders.simbench_export.


def test_build_quarterly_simbench_split_creates_expected_windows():
    timestamps = pd.date_range("2016-01-01 00:00:00", "2016-12-31 23:45:00", freq="15min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": np.linspace(0.01, 0.06, len(timestamps), dtype=np.float32),
            "load1": np.linspace(1.0, 2.0, len(timestamps), dtype=np.float32),
            "pv1": np.linspace(0.0, 0.6, len(timestamps), dtype=np.float32),
            "load2": np.linspace(1.2, 2.2, len(timestamps), dtype=np.float32),
            "pv2": np.linspace(0.0, 0.5, len(timestamps), dtype=np.float32),
            "load3": np.linspace(1.4, 2.4, len(timestamps), dtype=np.float32),
            "pv3": np.linspace(0.0, 0.4, len(timestamps), dtype=np.float32),
        }
    )

    full_frame, train_frame, test_frame, metadata = build_quarterly_simbench_split(frame, {"selected_buses": [1, 2, 3]})

    assert len(full_frame) == len(frame)
    assert metadata["test_target_rows"] == 4 * 2 * STEPS_PER_WEEK
    assert metadata["test_warmup_rows"] == 4 * STEPS_PER_WEEK
    assert metadata["train_segments"] == 5
    assert metadata["test_segments"] == 4
    assert set(test_frame["split_role"].unique()) == {"test_warmup", "test_target"}
    assert train_frame["split_role"].eq("train").all()
    assert test_frame["segment_id"].nunique() == 4
    assert train_frame["segment_id"].nunique() == 5

    expected_test_starts = [
        "2016-02-08 00:00:00",
        "2016-05-09 00:00:00",
        "2016-08-08 00:00:00",
        "2016-11-07 00:00:00",
    ]
    actual_test_starts = [window["test_start"] for window in metadata["quarter_test_windows"]]
    assert actual_test_starts == expected_test_starts


def test_align_price_series_to_reference_handles_dst_duplicate_rows():
    reference = pd.Series(
        pd.to_datetime(
            [
                "2016-10-30 01:45:00",
                "2016-10-30 02:00:00",
                "2016-10-30 02:15:00",
                "2016-10-30 02:00:00",
                "2016-10-30 02:15:00",
                "2016-10-30 02:30:00",
            ]
        )
    )
    price_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2016-10-30 01:45:00",
                    "2016-10-30 02:00:00",
                    "2016-10-30 02:00:00",
                    "2016-10-30 02:15:00",
                    "2016-10-30 02:15:00",
                    "2016-10-30 02:30:00",
                ]
            ),
            "price": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        }
    )

    aligned = align_price_series_to_reference(reference, price_frame)

    assert np.allclose(aligned, np.array([1, 2, 4, 3, 5, 6], dtype=np.float32))

