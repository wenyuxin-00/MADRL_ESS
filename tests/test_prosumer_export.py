from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.loaders.constants import PROSUMER_PROFILE_ALLOWLIST, TZ_LOCAL
from data.loaders.prosumer_export import export_prosumer_dataset


def _write_power_table(path: Path, key: str, timestamps_utc: pd.DatetimeIndex, watts: np.ndarray) -> None:
    frame = pd.DataFrame(
        {
            "VP_TOT": np.asarray(watts, dtype=np.float32),
        },
        index=(timestamps_utc.asi8 // 10**9).astype("int64"),
    )
    frame.to_hdf(path, key=key, format="table")


def _build_consumer_watts(base_watts: float, steps: int) -> np.ndarray:
    return np.asarray([base_watts + 10.0 * step for step in range(steps)], dtype=np.float32)


def _build_pv_watts(local_timestamps: pd.DatetimeIndex, orientation: str, *, missing_south_at_noon: bool) -> np.ndarray:
    values: list[float] = []
    for timestamp in local_timestamps:
        if timestamp.hour == 12:
            east_kw = 2.0 + 0.1 * timestamp.minute / 15.0
            west_kw = 4.0 + 0.1 * timestamp.minute / 15.0
            south_kw = 0.5 * (east_kw + west_kw)
        else:
            east_kw = 0.6
            west_kw = 0.4
            south_kw = 0.5

        if orientation == "east":
            values.append(east_kw * 1000.0)
        elif orientation == "west":
            values.append(west_kw * 1000.0)
        elif orientation == "south":
            if missing_south_at_noon and timestamp.hour == 12 and timestamp.minute == 0:
                values.append(np.nan)
            else:
                values.append(south_kw * 1000.0)
        else:  # pragma: no cover
            raise ValueError(orientation)
    return np.asarray(values, dtype=np.float32)


def _write_price_csv(path: Path, local_index: pd.DatetimeIndex) -> None:
    rows = ["Start date;Germany/Luxembourg [€/MWh] Original resolutions"]
    for idx, timestamp in enumerate(local_index):
        price = "-" if idx in {48, 49} else str(50 + idx)
        rows.append(f"{timestamp.tz_localize(None):%Y-%m-%d %H:%M};{price}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_export_prosumer_dataset_cleans_and_exports_expected_outputs(tmp_path):
    raw_dir = Path(tmp_path) / "raw"
    output_dir = Path(tmp_path) / "processed" / "prosumer"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf_2019 = raw_dir / "2019_data_15min.hdf5"
    hdf_2020 = raw_dir / "2020_data_15min.hdf5"
    price_csv = raw_dir / "Germany_price_15_2018-2020.csv"

    local_2019 = pd.date_range(
        start="2019-12-31 12:00:00",
        end="2019-12-31 23:45:00",
        freq="15min",
        tz=TZ_LOCAL,
    )
    local_2020 = pd.date_range(
        start="2020-01-01 01:00:00",
        end="2020-01-01 12:45:00",
        freq="15min",
        tz=TZ_LOCAL,
    )
    utc_2019 = local_2019.tz_convert("UTC")
    utc_2020 = local_2020.tz_convert("UTC")

    for user_idx, user in enumerate(PROSUMER_PROFILE_ALLOWLIST):
        _write_power_table(
            hdf_2019,
            f"/NO_PV/{user}/HOUSEHOLD",
            utc_2019,
            _build_consumer_watts(1000.0 + 100.0 * user_idx, len(utc_2019)),
        )
        _write_power_table(
            hdf_2019,
            f"/NO_PV/{user}/HEATPUMP",
            utc_2019,
            _build_consumer_watts(200.0 + 10.0 * user_idx, len(utc_2019)),
        )
        _write_power_table(
            hdf_2020,
            f"/NO_PV/{user}/HOUSEHOLD",
            utc_2020,
            _build_consumer_watts(2000.0 + 100.0 * user_idx, len(utc_2020)),
        )
        _write_power_table(
            hdf_2020,
            f"/NO_PV/{user}/HEATPUMP",
            utc_2020,
            _build_consumer_watts(500.0 + 10.0 * user_idx, len(utc_2020)),
        )

    for orientation in ("EAST", "SOUTH", "WEST"):
        _write_power_table(
            hdf_2019,
            f"/MISC/PV1/PV/INVERTER/{orientation}",
            utc_2019,
            _build_pv_watts(local_2019, orientation.lower(), missing_south_at_noon=False),
        )
        _write_power_table(
            hdf_2020,
            f"/MISC/PV1/PV/INVERTER/{orientation}",
            utc_2020,
            _build_pv_watts(
                local_2020,
                orientation.lower(),
                missing_south_at_noon=(orientation == "SOUTH"),
            ),
        )

    consumer_local_index = pd.date_range(
        start="2019-12-31 12:00:00",
        end="2020-01-01 12:45:00",
        freq="15min",
        tz=TZ_LOCAL,
    )
    _write_price_csv(price_csv, consumer_local_index)

    result = export_prosumer_dataset(
        hdf5_paths={2019: hdf_2019, 2020: hdf_2020},
        price_path=price_csv,
        output_dir=output_dir,
        consumer_local_start=consumer_local_index[0],
        consumer_local_end=consumer_local_index[-1],
        pv_local_start=consumer_local_index[0],
        pv_local_end=consumer_local_index[-1],
    )

    assert result.household_path.exists()
    assert result.heatpump_path.exists()
    assert result.pv_reference_path.exists()
    assert result.price_path.exists()
    assert result.metadata_path.exists()

    assert result.household_frame.columns.tolist() == ["timestamp", *PROSUMER_PROFILE_ALLOWLIST]
    assert result.heatpump_frame.columns.tolist() == ["timestamp", *PROSUMER_PROFILE_ALLOWLIST]
    assert result.pv_reference_frame.columns.tolist() == ["timestamp", "ref_east", "ref_south", "ref_west"]
    assert result.price_frame.columns.tolist() == ["timestamp", "price"]

    assert len(result.household_frame) == 100
    assert len(result.heatpump_frame) == 100
    assert len(result.pv_reference_frame) == 100
    assert len(result.price_frame) == 100

    assert not result.household_frame.isna().any().any()
    assert not result.heatpump_frame.isna().any().any()
    assert not result.pv_reference_frame.isna().any().any()
    assert not result.price_frame.isna().any().any()

    household_series = result.household_frame.set_index("timestamp")["SFH3"]
    price_series = result.price_frame.set_index("timestamp")["price"]
    pv_frame = result.pv_reference_frame.set_index("timestamp")

    assert np.isclose(household_series.loc[pd.Timestamp("2019-12-31 12:00:00", tz=TZ_LOCAL)], 1.0)
    assert np.isclose(household_series.loc[pd.Timestamp("2020-01-01 00:00:00", tz=TZ_LOCAL)], 2.0)
    assert np.isclose(price_series.loc[pd.Timestamp("2020-01-01 00:00:00", tz=TZ_LOCAL)], 0.098)

    night_timestamp = pd.Timestamp("2019-12-31 23:00:00", tz=TZ_LOCAL)
    assert np.isclose(pv_frame.loc[night_timestamp, "ref_east"], 0.0)
    assert np.isclose(pv_frame.loc[night_timestamp, "ref_south"], 0.0)
    assert np.isclose(pv_frame.loc[night_timestamp, "ref_west"], 0.0)

    noon_timestamp = pd.Timestamp("2020-01-01 12:00:00", tz=TZ_LOCAL)
    assert np.isclose(pv_frame.loc[noon_timestamp, "ref_south"], 3.0)

    assert result.metadata["allowlist_profiles"] == list(PROSUMER_PROFILE_ALLOWLIST)
    assert result.metadata["consumer_year_row_counts"] == {2019: 48, 2020: 52}
    assert result.metadata["price_year_row_counts"] == {2019: 48, 2020: 52}
    assert result.metadata["pv_year_row_counts"] == {2019: 48, 2020: 52}
    assert "SFH3" in result.metadata["user_stats_kw"]
    assert result.metadata["pv_reference_peak_kw"]["south"] >= 3.0
    assert result.metadata["sources"]["price_path"] == str(price_csv)

    metadata_json = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata_json["allowlist_profiles"] == list(PROSUMER_PROFILE_ALLOWLIST)
    assert metadata_json["consumer_year_row_counts"] == {"2019": 48, "2020": 52}
    assert metadata_json["cleaning_rules"]["pv_south_gap_fill"] == (
        "estimate from east+west empirical ratio by month/slot"
    )
