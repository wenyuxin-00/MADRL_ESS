"""Utilities for building the 2016 SimBench prosumer dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import simbench as sb

DEFAULT_SB_CODE = "1-LV-rural1--0-sw"
STEPS_PER_DAY = 96
STEPS_PER_WEEK = STEPS_PER_DAY * 7
DEFAULT_TEST_WEEKS = 2
DEFAULT_WARMUP_WEEKS = 1
DEFAULT_QUARTER_OFFSET_WEEKS = 5
LOAD_PROFILE_SUFFIX = "_pload"
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


@dataclass(frozen=True)
class SimbenchExportResult:
    """Resolved data frames, metadata, and export paths."""

    full_frame: pd.DataFrame
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    metadata: dict[str, object]
    full_path: Path
    train_path: Path
    test_path: Path
    metadata_path: Path


def _read_xlsx_first_sheet(path: str | Path) -> pd.DataFrame:
    """Read the first xlsx sheet without requiring openpyxl."""
    path = Path(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        main_ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}

        first_sheet = workbook_root.find("a:sheets/a:sheet", main_ns)
        if first_sheet is None:
            raise ValueError(f"Workbook '{path}' does not contain any sheets.")
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if rel_id is None:
            raise ValueError(f"Workbook '{path}' is missing the first-sheet relationship id.")

        target = None
        for rel in rel_root.findall("r:Relationship", rel_ns):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if target is None:
            raise ValueError(f"Workbook '{path}' is missing the worksheet target for '{rel_id}'.")

        sheet_path = f"xl/{target.lstrip('/')}"
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", main_ns):
                texts = [node.text or "" for node in item.findall(".//a:t", main_ns)]
                shared_strings.append("".join(texts))

        sheet_root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[object]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", main_ns):
            values: list[object] = []
            for cell in row.findall("a:c", main_ns):
                raw_value = cell.find("a:v", main_ns)
                value = raw_value.text if raw_value is not None else ""
                cell_type = cell.attrib.get("t")
                if cell_type == "s" and value != "":
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)

    if not rows:
        raise ValueError(f"Workbook '{path}' is empty.")

    header = [str(value).strip() for value in rows[0]]
    body = rows[1:]
    return pd.DataFrame(body, columns=header)


def _parse_price_timestamp(series: pd.Series) -> pd.Series:
    """Parse a timestamp column from strings or Excel serial values."""
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().all():
        parsed = EXCEL_EPOCH + pd.to_timedelta(numeric_values, unit="D")
        return parsed.round("15min")

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Price timestamp column contains unparsable values.")
    return parsed.round("15min")


def load_germany_price_frame(price_path: str | Path) -> tuple[pd.DataFrame, str]:
    """Load Germany power prices and normalize them to EUR/kWh."""
    price_path = Path(price_path)
    suffix = price_path.suffix.lower()

    if suffix == ".xlsx":
        raw = _read_xlsx_first_sheet(price_path)
        if raw.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in '{price_path}', got {list(raw.columns)}")
        time_column = raw.columns[0]
        price_column = raw.columns[1]
        frame = pd.DataFrame(
            {
                "timestamp": _parse_price_timestamp(raw.iloc[:, 0]),
                "price": pd.to_numeric(raw.iloc[:, 1], errors="coerce") / 1000.0,
            }
        )
        source_column = str(price_column)
    elif suffix == ".csv":
        raw = pd.read_csv(price_path, sep=";")
        source_column = "DE/AT/LU [€/MWh] Original resolutions"
        if source_column not in raw.columns:
            numeric_candidates = [column for column in raw.columns if column != "Start date"]
            if not numeric_candidates:
                raise ValueError(f"Could not find a numeric price column in '{price_path}'.")
            source_column = numeric_candidates[0]
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw["Start date"], format="%b %d, %Y %I:%M %p"),
                "price": pd.to_numeric(raw[source_column], errors="coerce") / 1000.0,
            }
        )
    else:
        raise ValueError(f"Unsupported price file '{price_path}'. Expected .csv or .xlsx.")

    if frame["price"].isna().any():
        raise ValueError(f"Price column '{source_column}' in '{price_path}' contains NaN after parsing.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.round("15min")
    frame["price"] = frame["price"].astype(np.float32)
    return frame, source_column


def _resolve_profile_column(profile_name, available_arrays, fallback_column, suffix=""):
    raw_name = "" if pd.isna(profile_name) else str(profile_name).strip()
    candidate = f"{raw_name}{suffix}" if raw_name else ""
    if candidate and candidate in available_arrays:
        return candidate
    return fallback_column


def align_price_series_to_reference(
    reference_timestamps: pd.Series,
    price_frame: pd.DataFrame,
) -> np.ndarray:
    """Align prices to a reference timestamp series, including duplicated DST rows."""
    reference = pd.DataFrame({"timestamp": pd.to_datetime(reference_timestamps)}).reset_index(names="row_idx")
    reference["_timestamp_occurrence"] = reference.groupby("timestamp").cumcount()

    price_aligned = price_frame.copy().reset_index(drop=True)
    price_aligned["timestamp"] = pd.to_datetime(price_aligned["timestamp"])
    price_aligned["_timestamp_occurrence"] = price_aligned.groupby("timestamp").cumcount()

    if len(reference) != len(price_aligned):
        raise ValueError(
            "SimBench and price sources do not contain the same number of rows: "
            f"{len(reference)} vs {len(price_aligned)}."
        )

    aligned = reference.merge(
        price_aligned.loc[:, ["timestamp", "_timestamp_occurrence", "price"]],
        on=["timestamp", "_timestamp_occurrence"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if aligned["price"].isna().any():
        missing = aligned.loc[aligned["price"].isna(), "timestamp"].head(8).tolist()
        raise ValueError(f"Failed to align price rows for timestamps: {missing}")
    return aligned.sort_values("row_idx")["price"].to_numpy(dtype=np.float32)


def build_simbench_absolute_frame(net) -> tuple[pd.DataFrame, list[int], list[int], list[int]]:
    """Expand SimBench normalized profiles into absolute kW time series."""
    load_buses = set(net["load"]["bus"].astype(int).tolist())
    sgen_buses = set(net["sgen"]["bus"].astype(int).tolist())
    prosumers = sorted(load_buses & sgen_buses)

    load_profile_table = net["profiles"]["load"].copy()
    renewable_profile_table = net["profiles"]["renewables"].copy()
    profile_time = pd.to_datetime(load_profile_table["time"], format="%d.%m.%Y %H:%M")

    load_profile_columns = [col for col in load_profile_table.columns if col.endswith(LOAD_PROFILE_SUFFIX)]
    renewable_profile_columns = [col for col in renewable_profile_table.columns if col != "time"]
    if not load_profile_columns:
        raise ValueError("No load profile columns were found in net['profiles']['load'].")
    if not renewable_profile_columns:
        raise ValueError("No renewable profile columns were found in net['profiles']['renewables'].")

    load_profile_arrays = {
        col: load_profile_table[col].to_numpy(dtype=np.float32, copy=True)
        for col in load_profile_columns
    }
    renewable_profile_arrays = {
        col: renewable_profile_table[col].to_numpy(dtype=np.float32, copy=True)
        for col in renewable_profile_columns
    }

    n_steps = len(profile_time)
    bus_ids = sorted(load_buses | sgen_buses)
    load_bus_arrays = {f"load_bus_{bus_id}": np.zeros(n_steps, dtype=np.float32) for bus_id in bus_ids}
    pv_bus_arrays = {f"pv_bus_{bus_id}": np.zeros(n_steps, dtype=np.float32) for bus_id in bus_ids}

    load_fallback_column = load_profile_columns[0]
    renewable_fallback_column = renewable_profile_columns[0]

    for row in net["load"][["bus", "p_mw", "profile"]].itertuples(index=False):
        profile_column = _resolve_profile_column(
            row.profile,
            load_profile_arrays,
            load_fallback_column,
            suffix=LOAD_PROFILE_SUFFIX,
        )
        load_bus_arrays[f"load_bus_{int(row.bus)}"] += (
            load_profile_arrays[profile_column] * (float(row.p_mw) * 1000.0)
        ).astype(np.float32, copy=False)

    for row in net["sgen"][["bus", "p_mw", "profile"]].itertuples(index=False):
        profile_column = _resolve_profile_column(
            row.profile,
            renewable_profile_arrays,
            renewable_fallback_column,
        )
        pv_bus_arrays[f"pv_bus_{int(row.bus)}"] += (
            renewable_profile_arrays[profile_column] * (float(row.p_mw) * 1000.0)
        ).astype(np.float32, copy=False)

    absolute_frame = pd.DataFrame({"timestamp": profile_time})
    for bus_id in bus_ids:
        absolute_frame[f"load_bus_{bus_id}"] = load_bus_arrays[f"load_bus_{bus_id}"]
        absolute_frame[f"pv_bus_{bus_id}"] = pv_bus_arrays[f"pv_bus_{bus_id}"]

    absolute_frame["total_load_kw"] = np.sum(
        np.column_stack([load_bus_arrays[f"load_bus_{bus_id}"] for bus_id in bus_ids]),
        axis=1,
        dtype=np.float32,
    )
    absolute_frame["total_pv_kw"] = np.sum(
        np.column_stack([pv_bus_arrays[f"pv_bus_{bus_id}"] for bus_id in bus_ids]),
        axis=1,
        dtype=np.float32,
    )
    return absolute_frame, sorted(load_buses), sorted(sgen_buses), prosumers


def build_prosumer_frame(
    *,
    sb_code: str = DEFAULT_SB_CODE,
    price_path: str | Path,
    n_agents: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the aligned full-year prosumer frame from SimBench + Germany prices."""
    net = sb.get_simbench_net(sb_code)
    absolute_frame, load_buses, sgen_buses, prosumers = build_simbench_absolute_frame(net)
    price_frame, price_source_column = load_germany_price_frame(price_path)

    merged_df = absolute_frame.copy()
    merged_df["price"] = align_price_series_to_reference(merged_df["timestamp"], price_frame)

    candidate_rows = []
    for bus_id in prosumers:
        pv_col = f"pv_bus_{bus_id}"
        load_col = f"load_bus_{bus_id}"
        candidate_rows.append(
            {
                "bus_id": int(bus_id),
                "pv_peak_kw": float(merged_df[pv_col].max()),
                "load_peak_kw": float(merged_df[load_col].max()),
                "pv_energy_index": float(merged_df[pv_col].sum()),
            }
        )
    prosumer_rank_df = pd.DataFrame(candidate_rows).sort_values(
        ["pv_peak_kw", "pv_energy_index", "bus_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected_buses = prosumer_rank_df["bus_id"].head(n_agents).astype(int).tolist()
    if len(selected_buses) != n_agents:
        raise ValueError(f"Expected at least {n_agents} prosumers, got {selected_buses}")

    prosumer_frame = pd.DataFrame(
        {
            "timestamp": merged_df["timestamp"],
            "price": merged_df["price"].astype(np.float32),
        }
    )
    metadata_payload: dict[str, object] = {
        "simbench_code": sb_code,
        "selected_buses": selected_buses,
        "price_source_file": Path(price_path).name,
        "price_source_column": price_source_column,
        "price_unit": "EUR/kWh",
        "power_unit": "kW",
        "energy_unit": "kWh",
        "rows_full_year": int(len(merged_df)),
        "load_bus_count": int(len(load_buses)),
        "pv_bus_count": int(len(sgen_buses)),
        "prosumer_candidates": int(len(prosumers)),
    }
    for agent_idx, bus_id in enumerate(selected_buses, start=1):
        load_col = f"load_bus_{bus_id}"
        pv_col = f"pv_bus_{bus_id}"
        prosumer_frame[f"load{agent_idx}"] = merged_df[load_col].astype(np.float32)
        prosumer_frame[f"pv{agent_idx}"] = merged_df[pv_col].astype(np.float32)

        pv_peak_kw = float(merged_df[pv_col].max())
        ess_power_kw = pv_peak_kw * 0.5
        ess_capacity_kwh = ess_power_kw * 2.5
        metadata_payload.setdefault("pv_peak_kw", []).append(pv_peak_kw)
        metadata_payload.setdefault("ess_power_kw", []).append(ess_power_kw)
        metadata_payload.setdefault("ess_capacity_kwh", []).append(ess_capacity_kwh)

    return prosumer_frame, prosumer_rank_df, metadata_payload


def _next_monday_midnight(timestamp: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp).floor("D")
    offset_days = (7 - timestamp.weekday()) % 7
    return timestamp + pd.Timedelta(days=offset_days)


def _assign_contiguous_segment_ids(mask: pd.Series) -> pd.Series:
    """Assign monotonically increasing ids to contiguous True regions."""
    contiguous_break = mask.astype(bool) & ~mask.astype(bool).shift(fill_value=False)
    segment_ids = contiguous_break.cumsum() - 1
    segment_ids = segment_ids.where(mask.astype(bool), other=-1).astype(int)
    return segment_ids


def build_quarterly_simbench_split(
    prosumer_frame: pd.DataFrame,
    metadata: dict[str, object] | None = None,
    *,
    test_weeks: int = DEFAULT_TEST_WEEKS,
    warmup_weeks: int = DEFAULT_WARMUP_WEEKS,
    quarter_offset_weeks: int = DEFAULT_QUARTER_OFFSET_WEEKS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split one full-year frame into segment-safe train/test exports."""
    if "timestamp" not in prosumer_frame.columns:
        raise ValueError("Prosumer frame must contain a 'timestamp' column.")

    full_frame = prosumer_frame.copy()
    full_frame["timestamp"] = pd.to_datetime(full_frame["timestamp"])
    full_frame = full_frame.sort_values("timestamp").reset_index(drop=True)
    if len(full_frame) < STEPS_PER_WEEK * (test_weeks + warmup_weeks):
        raise ValueError("Prosumer frame is too short for the requested quarterly split.")

    base_metadata = dict(metadata or {})
    step_delta = full_frame["timestamp"].diff().dropna()
    if step_delta.empty:
        raise ValueError("Prosumer frame must contain at least two rows.")
    inferred_step = step_delta.mode().iloc[0]
    if inferred_step != pd.Timedelta(minutes=15):
        raise ValueError(f"Expected a 15-minute time step, got {inferred_step}.")

    full_frame["split_role"] = "train"
    full_frame["is_warmup"] = False
    full_frame["segment_id"] = -1
    full_frame["week_id"] = -1
    full_frame["quarter"] = 0

    year = int(full_frame["timestamp"].dt.year.iloc[0])
    quarter_windows: list[dict[str, object]] = []

    for quarter in range(1, 5):
        quarter_start = pd.Timestamp(year=year, month=(quarter - 1) * 3 + 1, day=1)
        if quarter < 4:
            next_quarter_start = pd.Timestamp(year=year, month=quarter * 3 + 1, day=1)
        else:
            next_quarter_start = pd.Timestamp(year=year + 1, month=1, day=1)

        anchor = quarter_start + pd.Timedelta(weeks=quarter_offset_weeks)
        test_start = _next_monday_midnight(anchor)
        warmup_start = test_start - pd.Timedelta(weeks=warmup_weeks)
        test_end_exclusive = test_start + pd.Timedelta(weeks=test_weeks)

        if warmup_start < full_frame["timestamp"].iloc[0]:
            raise ValueError(f"Warmup for quarter {quarter} starts before the dataset begins: {warmup_start}.")
        if test_end_exclusive > full_frame["timestamp"].iloc[-1] + inferred_step:
            raise ValueError(f"Test window for quarter {quarter} exceeds the dataset end: {test_end_exclusive}.")

        block_id = quarter - 1
        warmup_mask = (full_frame["timestamp"] >= warmup_start) & (full_frame["timestamp"] < test_start)
        target_mask = (full_frame["timestamp"] >= test_start) & (full_frame["timestamp"] < test_end_exclusive)
        block_mask = warmup_mask | target_mask
        if int(block_mask.sum()) != STEPS_PER_WEEK * (warmup_weeks + test_weeks):
            raise ValueError(f"Quarter {quarter} block does not align to complete weeks.")

        full_frame.loc[warmup_mask, "split_role"] = "test_warmup"
        full_frame.loc[target_mask, "split_role"] = "test_target"
        full_frame.loc[warmup_mask, "is_warmup"] = True
        full_frame.loc[block_mask, "segment_id"] = block_id
        full_frame.loc[block_mask, "quarter"] = quarter

        week_edges = pd.date_range(
            start=warmup_start,
            end=test_end_exclusive,
            freq="7D",
            inclusive="left",
        )
        for local_week_idx, week_start in enumerate(week_edges):
            week_end = week_start + pd.Timedelta(weeks=1)
            week_mask = (full_frame["timestamp"] >= week_start) & (full_frame["timestamp"] < week_end)
            full_frame.loc[week_mask, "week_id"] = block_id * (warmup_weeks + test_weeks) + local_week_idx

        quarter_windows.append(
            {
                "quarter": quarter,
                "segment_id": block_id,
                "warmup_start": str(warmup_start),
                "warmup_end": str(test_start - inferred_step),
                "test_start": str(test_start),
                "test_end": str(test_end_exclusive - inferred_step),
                "warmup_rows": int(warmup_mask.sum()),
                "test_rows": int(target_mask.sum()),
            }
        )

    train_mask = full_frame["split_role"].eq("train")
    test_mask = full_frame["split_role"].isin(["test_warmup", "test_target"])

    train_segment_ids = _assign_contiguous_segment_ids(train_mask)
    full_frame.loc[train_mask, "segment_id"] = train_segment_ids.loc[train_mask].to_numpy(dtype=int)

    train_frame = full_frame.loc[train_mask].copy().reset_index(drop=True)
    test_frame = full_frame.loc[test_mask].copy().reset_index(drop=True)

    train_frame["segment_id"] = pd.to_numeric(train_frame["segment_id"], errors="raise").astype(int)
    test_frame["segment_id"] = pd.to_numeric(test_frame["segment_id"], errors="raise").astype(int)
    test_frame["week_id"] = pd.to_numeric(test_frame["week_id"], errors="raise").astype(int)

    full_export = full_frame.reset_index(drop=True)
    export_metadata = dict(base_metadata)
    export_metadata.update(
        {
            "split_strategy": "quarterly_two_week_test_with_one_week_warmup",
            "test_weeks_per_quarter": int(test_weeks),
            "warmup_weeks_per_quarter": int(warmup_weeks),
            "quarter_anchor_offset_weeks": int(quarter_offset_weeks),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "test_target_rows": int(test_frame["split_role"].eq("test_target").sum()),
            "test_warmup_rows": int(test_frame["split_role"].eq("test_warmup").sum()),
            "train_segments": int(train_frame["segment_id"].nunique()),
            "test_segments": int(test_frame["segment_id"].nunique()),
            "train_start": str(train_frame["timestamp"].iloc[0]),
            "train_end": str(train_frame["timestamp"].iloc[-1]),
            "test_start": str(test_frame["timestamp"].iloc[0]),
            "test_end": str(test_frame["timestamp"].iloc[-1]),
            "quarter_test_windows": quarter_windows,
        }
    )
    return full_export, train_frame, test_frame, export_metadata


def export_simbench_2016_dataset(
    *,
    output_dir: str | Path,
    price_path: str | Path,
    sb_code: str = DEFAULT_SB_CODE,
    n_agents: int = 3,
    test_weeks: int = DEFAULT_TEST_WEEKS,
    warmup_weeks: int = DEFAULT_WARMUP_WEEKS,
    quarter_offset_weeks: int = DEFAULT_QUARTER_OFFSET_WEEKS,
) -> tuple[SimbenchExportResult, pd.DataFrame]:
    """Build and write the full/train/test SimBench 2016 exports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prosumer_frame, prosumer_rank_df, metadata = build_prosumer_frame(
        sb_code=sb_code,
        price_path=price_path,
        n_agents=n_agents,
    )
    full_frame, train_frame, test_frame, export_metadata = build_quarterly_simbench_split(
        prosumer_frame,
        metadata,
        test_weeks=test_weeks,
        warmup_weeks=warmup_weeks,
        quarter_offset_weeks=quarter_offset_weeks,
    )

    full_path = output_dir / "simbench_2016_full.csv"
    train_path = output_dir / "simbench_2016_train.csv"
    test_path = output_dir / "simbench_2016_test.csv"
    metadata_path = output_dir / "simbench_2016_metadata.json"

    full_frame.to_csv(full_path, index=False)
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    metadata_path.write_text(json.dumps(export_metadata, indent=2), encoding="utf-8")

    return (
        SimbenchExportResult(
            full_frame=full_frame,
            train_frame=train_frame,
            test_frame=test_frame,
            metadata=export_metadata,
            full_path=full_path,
            train_path=train_path,
            test_path=test_path,
            metadata_path=metadata_path,
        ),
        prosumer_rank_df,
    )
