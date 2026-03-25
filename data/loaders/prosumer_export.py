"""Export raw prosumer source files into the processed training dataset."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from data.loaders.constants import (
    DATA_FREQUENCY,
    DEFAULT_PROCESSED_SUBDIR,
    DEFAULT_PROSUMER_PROFILES,
    PROSUMER_PROFILE_ALLOWLIST,
    TZ_LOCAL,
    TZ_UTC,
)

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency for raw HDF5 export
    h5py = None

FREQ = DATA_FREQUENCY
MAX_INTERP_STEPS = 8

DEFAULT_2019_HDF5 = Path("data/raw/2019_data_15min.hdf5")
DEFAULT_2020_HDF5 = Path("data/raw/2020_data_15min.hdf5")
DEFAULT_PRICE_CSV = Path("data/raw/Germany_price_15_2018-2020.csv")
DEFAULT_OUTPUT_DIR = Path("data") / DEFAULT_PROCESSED_SUBDIR

# Consumer and price rows start at 01:00 local time because the earliest
# quarter-hours on 2019-01-01 fall into the missing 2018 UTC range.
DEFAULT_CONSUMER_LOCAL_START = pd.Timestamp("2019-01-01 01:00:00", tz=TZ_LOCAL)
DEFAULT_CONSUMER_LOCAL_END = pd.Timestamp("2020-12-31 23:45:00", tz=TZ_LOCAL)

# PV references keep the full local-day timeline for later alignment.
DEFAULT_PV_LOCAL_START = pd.Timestamp("2019-01-01 00:00:00", tz=TZ_LOCAL)
DEFAULT_PV_LOCAL_END = pd.Timestamp("2020-12-31 23:45:00", tz=TZ_LOCAL)


@dataclass(frozen=True)
class ProsumerExportResult:
    """Processed prosumer export outputs and file paths."""

    household_frame: pd.DataFrame
    heatpump_frame: pd.DataFrame
    pv_reference_frame: pd.DataFrame
    price_frame: pd.DataFrame
    metadata: dict[str, object]
    household_path: Path
    heatpump_path: Path
    pv_reference_path: Path
    price_path: Path
    metadata_path: Path


def _natural_profile_key(profile: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(profile))
    number = int(match.group(1)) if match else 10**9
    return number, str(profile)


def _normalize_profiles(users: Iterable[str] | None) -> list[str]:
    selected = list(PROSUMER_PROFILE_ALLOWLIST if users is None else users)
    unknown = [user for user in selected if user not in PROSUMER_PROFILE_ALLOWLIST]
    if unknown:
        raise ValueError(f"Unknown prosumer profiles: {unknown}. Allowed: {list(PROSUMER_PROFILE_ALLOWLIST)}")
    return sorted(dict.fromkeys(selected), key=_natural_profile_key)


def _normalize_hdf5_paths(hdf5_paths: Mapping[int, str | Path] | Iterable[str | Path]) -> dict[int, Path]:
    if isinstance(hdf5_paths, Mapping):
        normalized = {int(year): Path(path) for year, path in hdf5_paths.items()}
    else:
        normalized: dict[int, Path] = {}
        for raw_path in hdf5_paths:
            path = Path(raw_path)
            match = re.search(r"(20\d{2})", path.name)
            if match is None:
                raise ValueError(f"Could not infer year from HDF5 path '{path}'.")
            normalized[int(match.group(1))] = path
    if not normalized:
        raise ValueError("At least one raw HDF5 path is required.")
    return dict(sorted(normalized.items()))


def _build_local_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=FREQ)


def _clip_window_to_year(
    start: pd.Timestamp,
    end: pd.Timestamp,
    year: int,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    year_start = pd.Timestamp(f"{int(year)}-01-01 00:00:00", tz=TZ_LOCAL)
    year_end = pd.Timestamp(f"{int(year)}-12-31 23:45:00", tz=TZ_LOCAL)
    clipped_start = max(start, year_start)
    clipped_end = min(end, year_end)
    if clipped_start > clipped_end:
        return None
    return clipped_start, clipped_end


def _standardize_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """将 WPuQ 的 ``VP_TOT`` 等列标准化为 ``P_TOT``。"""
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        column_name = str(column)
        if re.match(r"^V[A-Z0-9_]+$", column_name):
            rename_map[column_name] = column_name[1:]
        else:
            rename_map[column_name] = column_name
    return frame.rename(columns=rename_map)


def _ensure_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """将 HDF 内部 unix 索引转为 UTC / 本地时间列。"""
    out = frame.copy()
    if "index" in out.columns:
        unix_ts = pd.to_numeric(out["index"], errors="coerce")
        out = out.drop(columns=["index"])
    else:
        unix_ts = pd.to_numeric(pd.Index(out.index), errors="coerce")
        out = out.reset_index(drop=True)

    out.insert(0, "unix_ts", unix_ts.astype("int64"))
    out.insert(1, "timestamp_utc", pd.to_datetime(out["unix_ts"], unit="s", utc=True))
    out.insert(2, "timestamp_local", out["timestamp_utc"].dt.tz_convert(TZ_LOCAL))
    return out


def _add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """补齐 TOT 列并做单位换算。"""
    out = frame.copy()
    if "P_TOT" not in out.columns and {"P_1", "P_2", "P_3"}.issubset(out.columns):
        out["P_TOT"] = out[["P_1", "P_2", "P_3"]].sum(axis=1, min_count=1)
    if "Q_TOT" not in out.columns and {"Q_1", "Q_2", "Q_3"}.issubset(out.columns):
        out["Q_TOT"] = out[["Q_1", "Q_2", "Q_3"]].sum(axis=1, min_count=1)
    if "S_TOT" not in out.columns and {"S_1", "S_2", "S_3"}.issubset(out.columns):
        out["S_TOT"] = out[["S_1", "S_2", "S_3"]].sum(axis=1, min_count=1)
    if "PF_TOT" not in out.columns and {"PF_1", "PF_2", "PF_3"}.issubset(out.columns):
        out["PF_TOT"] = out[["PF_1", "PF_2", "PF_3"]].mean(axis=1)

    if "P_TOT" in out.columns:
        out["p_kw"] = pd.to_numeric(out["P_TOT"], errors="coerce") / 1000.0
    if "Q_TOT" in out.columns:
        out["q_kvar"] = pd.to_numeric(out["Q_TOT"], errors="coerce") / 1000.0
    if "S_TOT" in out.columns:
        out["s_kva"] = pd.to_numeric(out["S_TOT"], errors="coerce") / 1000.0
    if "PF_TOT" in out.columns:
        out["pf"] = pd.to_numeric(out["PF_TOT"], errors="coerce")

    u_cols = [column for column in ["U_1", "U_2", "U_3"] if column in out.columns]
    i_cols = [column for column in ["I_1", "I_2", "I_3"] if column in out.columns]
    if u_cols:
        out["u_v_mean"] = out[u_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if i_cols:
        out["i_a_mean"] = out[i_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return out


def _read_raw_node(hdf5_path: str | Path, node_path: str) -> pd.DataFrame:
    if h5py is None:
        raise ModuleNotFoundError("prosumer_export requires 'h5py' to scan raw HDF5 files.")
    frame = pd.read_hdf(Path(hdf5_path), key=node_path)
    frame = _standardize_raw_columns(frame)
    frame = _ensure_time_columns(frame)
    frame = _add_derived_columns(frame)
    return frame


def _time_filter_local(
    frame: pd.DataFrame,
    *,
    local_start: pd.Timestamp,
    local_end: pd.Timestamp,
) -> pd.DataFrame:
    return frame.loc[
        (frame["timestamp_local"] >= local_start) & (frame["timestamp_local"] <= local_end)
    ].copy()


def _reindex_and_interpolate(
    series: pd.Series,
    full_index_utc: pd.DatetimeIndex,
    *,
    max_interp_steps: int = MAX_INTERP_STEPS,
) -> pd.Series:
    aligned = pd.to_numeric(series, errors="coerce").sort_index().reindex(full_index_utc)
    aligned = aligned.interpolate(
        method="time",
        limit=max_interp_steps,
        limit_direction="both",
    )
    return aligned.astype(np.float32)


def _list_hdf_leaf_nodes(hdf5_path: Path) -> list[str]:
    if h5py is None:
        raise ModuleNotFoundError("prosumer_export requires 'h5py' to scan raw HDF5 files.")
    leaf_nodes: list[str] = []
    with h5py.File(hdf5_path, "r") as handle:
        def visitor(name, obj):
            if isinstance(obj, h5py.Group) and "table" in obj.keys():
                leaf_nodes.append("/" + name)
        handle.visititems(visitor)
    return sorted(leaf_nodes)


def _validate_node_path(hdf5_path: Path, node_path: str) -> None:
    available_nodes = _list_hdf_leaf_nodes(hdf5_path)
    if node_path not in available_nodes:
        raise KeyError(f"Node '{node_path}' not found in '{hdf5_path}'.")


def _load_consumer_signal(
    hdf5_path: str | Path,
    group: str,
    users: Iterable[str],
    field: str = "P_TOT",
    *,
    consumer_local_start: pd.Timestamp = DEFAULT_CONSUMER_LOCAL_START,
    consumer_local_end: pd.Timestamp = DEFAULT_CONSUMER_LOCAL_END,
    max_interp_steps: int = MAX_INTERP_STEPS,
) -> pd.DataFrame:
    """读取单个年份的 household / heatpump 曲线并对齐到统一时间轴。"""
    hdf_path = Path(hdf5_path)
    full_index_local = _build_local_index(consumer_local_start, consumer_local_end)
    full_index_utc = full_index_local.tz_convert(TZ_UTC)
    asset_name = str(group).strip().upper()

    payload: dict[str, pd.Series] = {}
    for user in _normalize_profiles(users):
        node_path = f"/NO_PV/{user}/{asset_name}"
        _validate_node_path(hdf_path, node_path)
        frame = _read_raw_node(hdf_path, node_path)
        frame = _time_filter_local(
            frame,
            local_start=consumer_local_start,
            local_end=consumer_local_end,
        )
        if field == "P_TOT":
            signal = pd.to_numeric(frame["P_TOT"], errors="coerce") / 1000.0
        else:
            signal = pd.to_numeric(frame[field], errors="coerce")
        signal.index = frame["timestamp_utc"]
        aligned = _reindex_and_interpolate(
            signal,
            full_index_utc,
            max_interp_steps=max_interp_steps,
        )
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(f"{node_path} still contains {missing} NaN values after interpolation.")
        payload[user] = aligned

    out = pd.DataFrame({"timestamp": full_index_local})
    for user in _normalize_profiles(users):
        out[user] = payload[user].to_numpy(dtype=np.float32)
    return out


def _load_pv_reference(
    hdf5_path: str | Path,
    *,
    pv_local_start: pd.Timestamp = DEFAULT_PV_LOCAL_START,
    pv_local_end: pd.Timestamp = DEFAULT_PV_LOCAL_END,
) -> pd.DataFrame:
    """读取单个年份的三条 PV 参考曲线。"""
    hdf_path = Path(hdf5_path)
    payload: dict[str, pd.Series] = {}
    for orientation in ("east", "south", "west"):
        node_path = f"/MISC/PV1/PV/INVERTER/{orientation.upper()}"
        _validate_node_path(hdf_path, node_path)
        frame = _read_raw_node(hdf_path, node_path)
        frame = frame.loc[
            (frame["timestamp_local"] >= pv_local_start) & (frame["timestamp_local"] <= pv_local_end)
        ].copy()
        series = pd.to_numeric(frame["p_kw"], errors="coerce").astype(np.float32)
        series.index = frame["timestamp_utc"]
        payload[orientation] = series

    out = pd.concat(payload.values(), axis=1).reset_index()
    out.columns = ["timestamp_utc", "east", "south", "west"]
    out["timestamp_local"] = pd.to_datetime(out["timestamp_utc"], utc=True).dt.tz_convert(TZ_LOCAL)
    return out[["timestamp_utc", "timestamp_local", "east", "south", "west"]]


def _build_pv_daylight_mask(timestamps_local: pd.Series) -> pd.Series:
    """用近似日照窗约束 PV 夜间出力。"""
    day_of_year = timestamps_local.dt.dayofyear.astype(float)
    is_dst = timestamps_local.apply(lambda value: bool(value.dst()))
    solar_noon = np.where(is_dst.to_numpy(), 13.5, 12.5)
    daylight_hours = 12.25 + 4.25 * np.cos(2 * np.pi * (day_of_year.to_numpy() - 172.0) / 365.25)
    sunrise = solar_noon - daylight_hours / 2.0
    sunset = solar_noon + daylight_hours / 2.0
    hour_float = timestamps_local.dt.hour + timestamps_local.dt.minute / 60.0
    return pd.Series(
        (hour_float.to_numpy() >= sunrise) & (hour_float.to_numpy() <= sunset),
        index=timestamps_local.index,
    )


def _clean_pv_reference(
    raw_frames: Iterable[pd.DataFrame],
    *,
    pv_local_start: pd.Timestamp = DEFAULT_PV_LOCAL_START,
    pv_local_end: pd.Timestamp = DEFAULT_PV_LOCAL_END,
) -> tuple[pd.DataFrame, dict[str, object]]:
    full_index_local = _build_local_index(pv_local_start, pv_local_end)
    full_index_utc = full_index_local.tz_convert(TZ_UTC)

    raw = pd.concat(list(raw_frames), ignore_index=True)
    raw = raw.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")

    out = pd.DataFrame({"timestamp": full_index_local, "timestamp_utc": full_index_utc})
    out = out.merge(raw, on=["timestamp_utc"], how="left")

    raw_cols = ["east", "south", "west"]
    original_values = out[raw_cols].apply(pd.to_numeric, errors="coerce")
    out["is_daylight_window"] = _build_pv_daylight_mask(out["timestamp"])
    out["missing_any_before"] = original_values.isna().any(axis=1)

    for column in ("east", "west"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out.loc[~out["is_daylight_window"], column] = 0.0
        out[column] = out[column].fillna(0.0).clip(lower=0.0)

    out["south"] = pd.to_numeric(out["south"], errors="coerce")
    out.loc[~out["is_daylight_window"], "south"] = 0.0

    out["month"] = out["timestamp"].dt.month.astype(int)
    out["slot"] = (out["timestamp"].dt.hour * 4 + out["timestamp"].dt.minute // 15).astype(int)

    support_mask = (
        out["is_daylight_window"]
        & out["south"].notna()
        & ((out["east"] + out["west"]) > 1e-6)
    )
    ratio_df = out.loc[support_mask, ["month", "slot", "south", "east", "west"]].copy()
    ratio_df["ew_sum"] = ratio_df["east"] + ratio_df["west"]
    ratio_df["ratio"] = ratio_df["south"] / ratio_df["ew_sum"]

    ratio_by_month_slot = ratio_df.groupby(["month", "slot"])["ratio"].median() if not ratio_df.empty else pd.Series(dtype=float)
    ratio_by_slot = ratio_df.groupby("slot")["ratio"].median() if not ratio_df.empty else pd.Series(dtype=float)
    ratio_global = float(ratio_df["ratio"].median()) if not ratio_df.empty else 1.0

    def estimate_south(row: pd.Series) -> float:
        ew_sum = float(row["east"] + row["west"])
        if ew_sum <= 1e-9:
            return 0.0
        key = (int(row["month"]), int(row["slot"]))
        ratio = ratio_by_month_slot.get(key, np.nan)
        if pd.isna(ratio):
            ratio = ratio_by_slot.get(int(row["slot"]), np.nan)
        if pd.isna(ratio):
            ratio = ratio_global
        return max(0.0, ew_sum * float(ratio))

    south_missing_daylight = out["south"].isna() & out["is_daylight_window"]
    out["south_estimated"] = south_missing_daylight
    if south_missing_daylight.any():
        estimated_south = out.loc[south_missing_daylight].apply(estimate_south, axis=1).to_numpy(dtype=np.float32)
        out.loc[south_missing_daylight, "south"] = estimated_south
    out["south"] = out["south"].fillna(0.0).clip(lower=0.0)

    night_zero_matrix = original_values.fillna(0.0).gt(1e-9) & (~out["is_daylight_window"].to_numpy()[:, None])
    out["night_zero_enforced"] = night_zero_matrix.any(axis=1)
    out.loc[~out["is_daylight_window"], raw_cols] = 0.0

    cleaned = out.loc[:, ["timestamp", "east", "south", "west"]].rename(
        columns={
            "east": "ref_east",
            "south": "ref_south",
            "west": "ref_west",
        }
    )
    cleaned["ref_east"] = cleaned["ref_east"].astype(np.float32)
    cleaned["ref_south"] = cleaned["ref_south"].astype(np.float32)
    cleaned["ref_west"] = cleaned["ref_west"].astype(np.float32)

    stats = {
        "quality_flag_counts": {
            "night_zero_enforced": int(out["night_zero_enforced"].sum()),
            "south_gap_filled": int(out["south_estimated"].sum()),
            "missing_any_before": int(out["missing_any_before"].sum()),
        },
        "peak_kw": {
            "east": float(cleaned["ref_east"].max()),
            "south": float(cleaned["ref_south"].max()),
            "west": float(cleaned["ref_west"].max()),
            "total": float((cleaned[["ref_east", "ref_south", "ref_west"]].sum(axis=1)).max()),
        },
    }
    return cleaned, stats


def _load_price_series(
    price_path: str | Path,
    years: Iterable[int],
    *,
    consumer_local_start: pd.Timestamp = DEFAULT_CONSUMER_LOCAL_START,
    consumer_local_end: pd.Timestamp = DEFAULT_CONSUMER_LOCAL_END,
    max_interp_steps: int = MAX_INTERP_STEPS,
) -> pd.DataFrame:
    """读取 SMARD 电价并对齐到统一 consumer 时间轴。"""
    raw = pd.read_csv(Path(price_path), sep=";", encoding="utf-8-sig", low_memory=False)
    raw.columns = [str(column).strip() for column in raw.columns]

    start_column = "Start date"
    germany_columns = [column for column in raw.columns if ("Germany/Luxembourg" in column and "€/MWh" in column)]
    if not germany_columns:
        germany_columns = [column for column in raw.columns if column == "DE/AT/LU [€/MWh] Original resolutions"]
    if not germany_columns:
        raise ValueError("Could not find a Germany/Luxembourg price column in SMARD CSV.")

    out = raw[[start_column, germany_columns[0]]].copy()
    out = out.rename(
        columns={
            start_column: "start_local_naive",
            germany_columns[0]: "price_eur_per_mwh",
        }
    )
    parsed_start = pd.to_datetime(
        out["start_local_naive"],
        format="%b %d, %Y %I:%M %p",
        errors="coerce",
    )
    missing_start = parsed_start.isna()
    if missing_start.any():
        parsed_start.loc[missing_start] = pd.to_datetime(
            out.loc[missing_start, "start_local_naive"],
            errors="coerce",
        )
    out["start_local_naive"] = parsed_start
    out["price_eur_per_mwh"] = pd.to_numeric(
        out["price_eur_per_mwh"].replace("-", np.nan),
        errors="coerce",
    )
    out = out.dropna(subset=["start_local_naive"]).sort_values("start_local_naive").reset_index(drop=True)

    local_start_naive = consumer_local_start.tz_localize(None)
    local_end_naive = consumer_local_end.tz_localize(None)
    out = out.loc[
        (out["start_local_naive"] >= local_start_naive)
        & (out["start_local_naive"] <= local_end_naive)
    ].copy()

    full_index_local = _build_local_index(consumer_local_start, consumer_local_end)
    full_index_utc = full_index_local.tz_convert(TZ_UTC)
    canonical = pd.DataFrame({"timestamp": full_index_local, "timestamp_utc": full_index_utc})
    out = out.reset_index(drop=True)
    if len(out) != len(canonical):
        raise ValueError(
            "Price series length does not match the canonical 15-minute axis: "
            f"expected {len(canonical)} rows, got {len(out)}."
        )

    merged = canonical.copy()
    merged["price_eur_per_mwh"] = out["price_eur_per_mwh"].to_numpy()

    merged["price_eur_per_mwh"] = merged["price_eur_per_mwh"].interpolate(
        method="linear",
        limit=max_interp_steps,
        limit_direction="both",
    )
    merged["price"] = (merged["price_eur_per_mwh"] / 1000.0).astype(np.float32)
    if merged["price"].isna().any():
        missing = int(merged["price"].isna().sum())
        raise ValueError(f"Price series still contains {missing} NaN values after interpolation.")

    merged["year"] = merged["timestamp"].dt.year.astype(int)
    target_years = {int(year) for year in years}
    year_counts = merged["year"].value_counts().to_dict()
    if not target_years.issubset(year_counts.keys()):
        raise ValueError(f"Price file does not cover all requested years {sorted(target_years)}.")
    return merged[["timestamp", "price"]].copy()


def _resolve_output_paths(output_dir: str | Path) -> dict[str, Path]:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "household": base_dir / "household.csv",
        "heatpump": base_dir / "heatpump.csv",
        "pv_reference": base_dir / "pv_reference.csv",
        "price": base_dir / "price.csv",
        "metadata": base_dir / "metadata.json",
    }


def _build_user_stats(household_frame: pd.DataFrame, heatpump_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for user in [column for column in household_frame.columns if column != "timestamp"]:
        household = pd.to_numeric(household_frame[user], errors="coerce")
        heatpump = pd.to_numeric(heatpump_frame[user], errors="coerce")
        total = household + heatpump
        stats[user] = {
            "household_peak_kw": float(household.max()),
            "household_mean_kw": float(household.mean()),
            "heatpump_peak_kw": float(heatpump.max()),
            "heatpump_mean_kw": float(heatpump.mean()),
            "load_peak_kw": float(total.max()),
            "load_mean_kw": float(total.mean()),
        }
    return stats


def export_prosumer_dataset(
    *,
    hdf5_paths,
    price_path,
    output_dir,
    users: Iterable[str] | None = None,
    consumer_local_start: pd.Timestamp = DEFAULT_CONSUMER_LOCAL_START,
    consumer_local_end: pd.Timestamp = DEFAULT_CONSUMER_LOCAL_END,
    pv_local_start: pd.Timestamp = DEFAULT_PV_LOCAL_START,
    pv_local_end: pd.Timestamp = DEFAULT_PV_LOCAL_END,
    max_interp_steps: int = MAX_INTERP_STEPS,
) -> ProsumerExportResult:
    """Export the processed prosumer CSV dataset."""
    normalized_hdf5_paths = _normalize_hdf5_paths(hdf5_paths)
    selected_users = _normalize_profiles(users)
    output_paths = _resolve_output_paths(output_dir)

    household_frames: list[pd.DataFrame] = []
    heatpump_frames: list[pd.DataFrame] = []
    pv_raw_frames: list[pd.DataFrame] = []

    for year, hdf5_path in normalized_hdf5_paths.items():
        consumer_window = _clip_window_to_year(consumer_local_start, consumer_local_end, year)
        if consumer_window is not None:
            household_frames.append(
                _load_consumer_signal(
                    hdf5_path,
                    "household",
                    selected_users,
                    field="P_TOT",
                    consumer_local_start=consumer_window[0],
                    consumer_local_end=consumer_window[1],
                    max_interp_steps=max_interp_steps,
                )
            )
            heatpump_frames.append(
                _load_consumer_signal(
                    hdf5_path,
                    "heatpump",
                    selected_users,
                    field="P_TOT",
                    consumer_local_start=consumer_window[0],
                    consumer_local_end=consumer_window[1],
                    max_interp_steps=max_interp_steps,
                )
            )

        pv_window = _clip_window_to_year(pv_local_start, pv_local_end, year)
        if pv_window is not None:
            pv_raw_frames.append(
                _load_pv_reference(
                    hdf5_path,
                    pv_local_start=pv_window[0],
                    pv_local_end=pv_window[1],
                )
            )

    if not household_frames or not heatpump_frames:
        raise ValueError("No raw prosumer consumer data was loaded for the requested export window.")
    if not pv_raw_frames:
        raise ValueError("No raw prosumer PV reference data was loaded for the requested export window.")

    household_frame = pd.concat(household_frames, ignore_index=True)
    household_frame = household_frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    heatpump_frame = pd.concat(heatpump_frames, ignore_index=True)
    heatpump_frame = heatpump_frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    pv_reference_frame, pv_stats = _clean_pv_reference(
        pv_raw_frames,
        pv_local_start=pv_local_start,
        pv_local_end=pv_local_end,
    )
    price_frame = _load_price_series(
        price_path,
        years=normalized_hdf5_paths.keys(),
        consumer_local_start=consumer_local_start,
        consumer_local_end=consumer_local_end,
        max_interp_steps=max_interp_steps,
    )

    household_frame = household_frame.reset_index(drop=True)
    heatpump_frame = heatpump_frame.reset_index(drop=True)
    pv_reference_frame = pv_reference_frame.reset_index(drop=True)
    price_frame = price_frame.reset_index(drop=True)

    household_frame.to_csv(output_paths["household"], index=False)
    heatpump_frame.to_csv(output_paths["heatpump"], index=False)
    pv_reference_frame.to_csv(output_paths["pv_reference"], index=False)
    price_frame.to_csv(output_paths["price"], index=False)

    consumer_year_row_counts = (
        pd.to_datetime(household_frame["timestamp"], utc=True)
        .dt.tz_convert(TZ_LOCAL)
        .dt.year.value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    price_year_row_counts = (
        pd.to_datetime(price_frame["timestamp"], utc=True)
        .dt.tz_convert(TZ_LOCAL)
        .dt.year.value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    pv_year_row_counts = (
        pd.to_datetime(pv_reference_frame["timestamp"], utc=True)
        .dt.tz_convert(TZ_LOCAL)
        .dt.year.value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    metadata = {
        "dataset_name": "prosumer_processed_2019_2020",
        "timezone_display": TZ_LOCAL,
        "timezone_storage": TZ_UTC,
        "frequency": FREQ,
        "allowlist_profiles": selected_users,
        "consumer_profiles": selected_users,
        "default_profiles": list(DEFAULT_PROSUMER_PROFILES),
        "consumer_row_count": int(len(household_frame)),
        "price_row_count": int(len(price_frame)),
        "pv_row_count": int(len(pv_reference_frame)),
        "consumer_year_row_counts": consumer_year_row_counts,
        "price_year_row_counts": price_year_row_counts,
        "pv_year_row_counts": pv_year_row_counts,
        "user_stats_kw": _build_user_stats(household_frame, heatpump_frame),
        "pv_reference_peak_kw": pv_stats["peak_kw"],
        "pv_reference_quality": pv_stats["quality_flag_counts"],
        "sources": {
            "hdf5_paths": {str(year): str(path) for year, path in normalized_hdf5_paths.items()},
            "price_path": str(price_path),
        },
        "time_ranges": {
            "consumer_start_local": str(consumer_local_start),
            "consumer_end_local": str(consumer_local_end),
            "pv_start_local": str(pv_local_start),
            "pv_end_local": str(pv_local_end),
        },
        "cleaning_rules": {
            "max_interp_steps": int(max_interp_steps),
            "consumer_interpolation": "time interpolation on UTC axis, up to 8 steps by default",
            "price_interpolation": "linear interpolation on canonical local axis, up to 8 steps by default",
            "pv_night_zero": True,
            "pv_south_gap_fill": "estimate from east+west empirical ratio by month/slot",
        },
    }
    output_paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ProsumerExportResult(
        household_frame=household_frame,
        heatpump_frame=heatpump_frame,
        pv_reference_frame=pv_reference_frame,
        price_frame=price_frame,
        metadata=metadata,
        household_path=output_paths["household"],
        heatpump_path=output_paths["heatpump"],
        pv_reference_path=output_paths["pv_reference"],
        price_path=output_paths["price"],
        metadata_path=output_paths["metadata"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed prosumer CSV dataset.")
    parser.add_argument("--hdf2019", type=Path, default=DEFAULT_2019_HDF5)
    parser.add_argument("--hdf2020", type=Path, default=DEFAULT_2020_HDF5)
    parser.add_argument("--price", type=Path, default=DEFAULT_PRICE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--users", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = export_prosumer_dataset(
        hdf5_paths={2019: args.hdf2019, 2020: args.hdf2020},
        price_path=args.price,
        output_dir=args.output_dir,
        users=args.users,
    )
    print(f"Exported household CSV: {result.household_path}")
    print(f"Exported heatpump CSV: {result.heatpump_path}")
    print(f"Exported PV CSV: {result.pv_reference_path}")
    print(f"Exported price CSV: {result.price_path}")
    print(f"Exported metadata JSON: {result.metadata_path}")


if __name__ == "__main__":
    main()

__all__ = [
    'DEFAULT_2019_HDF5',
    'DEFAULT_2020_HDF5',
    'DEFAULT_CONSUMER_LOCAL_END',
    'DEFAULT_CONSUMER_LOCAL_START',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_PRICE_CSV',
    'DEFAULT_PV_LOCAL_END',
    'DEFAULT_PV_LOCAL_START',
    'MAX_INTERP_STEPS',
    'ProsumerExportResult',
    'export_prosumer_dataset',
]
