"""SimBench 电网数据导出与预处理工具。

从 SimBench 电网模型中提取负荷、光伏发电等时间序列数据，与德国电力市场现货
电价对齐后，导出为标准 CSV 格式供强化学习训练和测试使用。

本模块的核心流程：
1. 从 SimBench 网络中提取归一化负荷/可再生能源配置文件，转换为绝对功率 (kW)
2. 加载并对齐德国电价数据（支持 CSV 和 XLSX 格式）
3. 按光伏峰值功率排序选取产消者节点
4. 按季度划分训练/测试集，包含预热（warmup）窗口

主要函数:
    export_simbench_2016_dataset -- 构建并导出完整的 SimBench 2016 数据集
    build_prosumer_frame -- 构建全年产消者数据帧
    build_quarterly_simbench_split -- 按季度划分训练/测试集
    build_simbench_absolute_frame -- 将归一化配置文件展开为绝对功率时间序列
    load_germany_price_frame -- 加载并归一化德国电价数据
    align_price_series_to_reference -- 将电价序列对齐到参考时间戳

主要类:
    SimbenchExportResult -- 导出结果数据类，包含数据帧、元数据和文件路径
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# SimBench 电网模型代码（低压农村场景）
DEFAULT_SB_CODE = "1-LV-rural1--0-sw"
# 每天的时间步数（15 分钟间隔，96 步 = 24 小时）
STEPS_PER_DAY = 96
# 每周的时间步数
STEPS_PER_WEEK = STEPS_PER_DAY * 7
# 每季度的测试周数
DEFAULT_TEST_WEEKS = 2
# 每季度测试集前的预热周数
DEFAULT_WARMUP_WEEKS = 1
# 季度锚点偏移周数（从季度起始日算起）
DEFAULT_QUARTER_OFFSET_WEEKS = 5
# SimBench 负荷配置文件名后缀
LOAD_PROFILE_SUFFIX = "_pload"
# Excel 序列值日期基准
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


@dataclass(frozen=True)
class SimbenchExportResult:
    """SimBench 数据导出结果，包含数据帧、元数据和导出文件路径。

    属性:
        full_frame (pd.DataFrame): 完整全年数据帧（含训练和测试数据）
        train_frame (pd.DataFrame): 训练集数据帧
        test_frame (pd.DataFrame): 测试集数据帧（含预热行）
        metadata (dict): 导出元数据（含拆分策略、行数统计等）
        full_path (Path): 完整数据 CSV 文件路径
        train_path (Path): 训练集 CSV 文件路径
        test_path (Path): 测试集 CSV 文件路径
        metadata_path (Path): 元数据 JSON 文件路径
    """

    full_frame: pd.DataFrame
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    metadata: dict[str, object]
    full_path: Path
    train_path: Path
    test_path: Path
    metadata_path: Path


def _read_xlsx_first_sheet(path: str | Path) -> pd.DataFrame:
    """读取 XLSX 文件的第一个工作表，不依赖 openpyxl 库。

    通过直接解析 XLSX 的 ZIP 内部 XML 结构来读取数据，
    适用于无法安装 openpyxl 的环境。

    Args:
        path: XLSX 文件路径

    Returns:
        pd.DataFrame: 第一个工作表的内容，首行作为列名

    Raises:
        ValueError: 工作簿为空或缺少必要的内部结构
    """
    path = Path(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        # 解析工作簿结构以找到第一个工作表
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        main_ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}

        # 定位第一个工作表的关系 ID
        first_sheet = workbook_root.find("a:sheets/a:sheet", main_ns)
        if first_sheet is None:
            raise ValueError(f"Workbook '{path}' does not contain any sheets.")
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if rel_id is None:
            raise ValueError(f"Workbook '{path}' is missing the first-sheet relationship id.")

        # 根据关系 ID 查找工作表的实际文件路径
        target = None
        for rel in rel_root.findall("r:Relationship", rel_ns):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if target is None:
            raise ValueError(f"Workbook '{path}' is missing the worksheet target for '{rel_id}'.")

        sheet_path = f"xl/{target.lstrip('/')}"
        # 加载共享字符串表（Excel 用于去重存储字符串值）
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", main_ns):
                texts = [node.text or "" for node in item.findall(".//a:t", main_ns)]
                shared_strings.append("".join(texts))

        # 逐行解析工作表单元格数据
        sheet_root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[object]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", main_ns):
            values: list[object] = []
            for cell in row.findall("a:c", main_ns):
                raw_value = cell.find("a:v", main_ns)
                value = raw_value.text if raw_value is not None else ""
                cell_type = cell.attrib.get("t")
                # 类型 "s" 表示共享字符串引用，需从共享字符串表中查找
                if cell_type == "s" and value != "":
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)

    if not rows:
        raise ValueError(f"Workbook '{path}' is empty.")

    # 首行作为列名，其余行作为数据
    header = [str(value).strip() for value in rows[0]]
    body = rows[1:]
    return pd.DataFrame(body, columns=header)


def _parse_price_timestamp(series: pd.Series) -> pd.Series:
    """解析时间戳列，支持字符串格式和 Excel 序列值两种输入。

    优先尝试将值解析为 Excel 序列日期（浮点数），若失败则按字符串日期格式解析。
    所有结果均对齐到 15 分钟精度。

    Args:
        series: 包含时间戳的 pandas Series

    Returns:
        pd.Series: 解析后的时间戳 Series（精确到 15 分钟）

    Raises:
        ValueError: 存在无法解析的时间戳值
    """
    # 尝试将全部值解析为 Excel 序列日期（从 1899-12-30 起的天数）
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().all():
        parsed = EXCEL_EPOCH + pd.to_timedelta(numeric_values, unit="D")
        return parsed.round("15min")

    # 回退方案：按字符串格式解析日期时间
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Price timestamp column contains unparsable values.")
    return parsed.round("15min")


def load_germany_price_frame(price_path: str | Path) -> tuple[pd.DataFrame, str]:
    """加载德国电力市场现货电价并归一化为 EUR/kWh。

    支持两种输入格式：
    - XLSX: 第一列为时间戳，第二列为电价（EUR/MWh），自动除以 1000 转换
    - CSV: 分号分隔，包含 "Start date" 和电价列（EUR/MWh）

    Args:
        price_path: 电价数据文件路径（.csv 或 .xlsx）

    Returns:
        tuple: (电价数据帧, 源电价列名)
            - 数据帧包含 "timestamp" 和 "price"（EUR/kWh）两列
            - 源列名用于元数据记录

    Raises:
        ValueError: 文件格式不支持、缺少必要列、或电价数据包含 NaN
    """
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
                # 从 EUR/MWh 转换为 EUR/kWh
                "price": pd.to_numeric(raw.iloc[:, 1], errors="coerce") / 1000.0,
            }
        )
        source_column = str(price_column)
    elif suffix == ".csv":
        raw = pd.read_csv(price_path, sep=";")
        # 尝试使用标准的 ENTSO-E 电价列名
        source_column = "DE/AT/LU [€/MWh] Original resolutions"
        if source_column not in raw.columns:
            # 回退：使用第一个非时间戳的数值列
            numeric_candidates = [column for column in raw.columns if column != "Start date"]
            if not numeric_candidates:
                raise ValueError(f"Could not find a numeric price column in '{price_path}'.")
            source_column = numeric_candidates[0]
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw["Start date"], format="%b %d, %Y %I:%M %p"),
                # 从 EUR/MWh 转换为 EUR/kWh
                "price": pd.to_numeric(raw[source_column], errors="coerce") / 1000.0,
            }
        )
    else:
        raise ValueError(f"Unsupported price file '{price_path}'. Expected .csv or .xlsx.")

    # 校验电价数据完整性
    if frame["price"].isna().any():
        raise ValueError(f"Price column '{source_column}' in '{price_path}' contains NaN after parsing.")

    # 统一时间戳精度为 15 分钟
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.round("15min")
    frame["price"] = frame["price"].astype(np.float32)
    return frame, source_column


def _resolve_profile_column(profile_name, available_arrays, fallback_column, suffix=""):
    """解析配置文件列名，找到对应的归一化配置文件数组。

    尝试将 profile_name 加上后缀后在可用数组中查找；若未找到则使用回退列。

    Args:
        profile_name: SimBench 负荷/发电表中的 profile 字段值
        available_arrays: 可用的配置文件数组字典（列名 -> 数组）
        fallback_column: 找不到匹配列时使用的回退列名
        suffix: 列名后缀（如负荷配置文件的 "_pload"）

    Returns:
        str: 匹配到的配置文件列名
    """
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
    import simbench as sb
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
