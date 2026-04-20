"""Shared helpers for disk-friendly rollout package caching."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY, PRICE_PROTOCOL_VERSION


ROLLOUT_PACKAGE_VERSION = 2
_CFG_FLOAT_RTOL = 1e-6
_CFG_FLOAT_ATOL = 1e-6


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def _raise_mismatch(prefix: str, field_name: str, expected: Any, actual: Any) -> None:
    raise ValueError(f"{prefix} mismatch at '{field_name}': expected={expected!r}, actual={actual!r}.")


def _assert_strict_match(prefix: str, field_name: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        _raise_mismatch(prefix, field_name, expected, actual)


def _assert_float_match(prefix: str, field_name: str, expected: float, actual: float) -> None:
    if not math.isclose(float(expected), float(actual), rel_tol=_CFG_FLOAT_RTOL, abs_tol=_CFG_FLOAT_ATOL):
        _raise_mismatch(prefix, field_name, expected, actual)


def _assert_float_sequence_match(prefix: str, field_name: str, expected: list[float], actual: list[float]) -> None:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if expected_array.shape != actual_array.shape:
        _raise_mismatch(prefix, field_name, expected, actual)
    if not np.allclose(expected_array, actual_array, rtol=_CFG_FLOAT_RTOL, atol=_CFG_FLOAT_ATOL):
        _raise_mismatch(prefix, field_name, expected, actual)


def _assert_int_sequence_match(prefix: str, field_name: str, expected: list[int], actual: list[int]) -> None:
    if [int(value) for value in expected] != [int(value) for value in actual]:
        _raise_mismatch(prefix, field_name, expected, actual)


def _assert_str_sequence_match(prefix: str, field_name: str, expected: list[str], actual: list[str]) -> None:
    if [str(value) for value in expected] != [str(value) for value in actual]:
        _raise_mismatch(prefix, field_name, expected, actual)


def _normalize_battery_controls_from_cfg(cfg: Any) -> dict[str, Any]:
    raw_capacity = getattr(getattr(cfg, "env", None), "battery_capacity", [])
    if isinstance(raw_capacity, np.ndarray):
        capacity_values = raw_capacity.reshape(-1).tolist()
    elif isinstance(raw_capacity, (list, tuple)):
        capacity_values = list(raw_capacity)
    elif raw_capacity in (None, ""):
        capacity_values = []
    else:
        capacity_values = [raw_capacity]
    return {
        "battery_capacity": [float(value) for value in capacity_values],
        "max_charge_rate": float(getattr(getattr(cfg, "env", None), "max_charge_rate", np.nan)),
        "efficiency": float(getattr(getattr(cfg, "env", None), "efficiency", np.nan)),
        "init_soc": float(getattr(getattr(cfg, "env", None), "init_soc", np.nan)),
        "soc_min": float(getattr(getattr(cfg, "env", None), "soc_min", np.nan)),
        "soc_max": float(getattr(getattr(cfg, "env", None), "soc_max", np.nan)),
        "soc_target": float(getattr(getattr(cfg, "env", None), "soc_target", np.nan)),
    }


def build_rollout_cfg_snapshot_from_cfg(cfg: Any, *, prediction_mode: str | None = None) -> dict[str, Any]:
    forecast_backend = str(getattr(getattr(cfg, "forecast", None), "type", "perfect"))
    resolved_prediction_mode = (
        grid_nb.normalize_prediction_mode(prediction_mode)
        if prediction_mode is not None
        else grid_nb.resolve_prediction_mode_from_forecast_backend(forecast_backend)
    )
    return {
        "test_start_date": str(getattr(getattr(cfg, "data", None), "test_start_date", "")),
        "test_end_date": str(getattr(getattr(cfg, "data", None), "test_end_date", "")),
        "prediction_mode": str(resolved_prediction_mode),
        "forecast_backend": forecast_backend,
        "agent_profiles": [str(value) for value in list(getattr(getattr(cfg, "data", None), "agent_profiles", []))],
        "agent_bus_ids": [int(value) for value in list(getattr(getattr(cfg, "grid", None), "agent_bus_ids", []))],
        "load_scale": [float(value) for value in list(getattr(getattr(cfg, "data", None), "load_scale", []))],
        "pv_scale": [float(value) for value in list(getattr(getattr(cfg, "data", None), "pv_scale", []))],
        "battery_controls": _normalize_battery_controls_from_cfg(cfg),
        "future_horizon": int(getattr(getattr(cfg, "env", None), "future_horizon", 0)),
        "episode_limit": int(getattr(getattr(cfg, "env", None), "episode_limit", 0)),
        "v_min_pu": float(getattr(getattr(cfg, "grid", None), "v_min_pu", np.nan)),
        "v_max_pu": float(getattr(getattr(cfg, "grid", None), "v_max_pu", np.nan)),
        "price_protocol_version": int(PRICE_PROTOCOL_VERSION),
        IMPORT_PRICE_MARKUP_KEY: float(
            getattr(getattr(cfg, "reward", None), IMPORT_PRICE_MARKUP_KEY, 0.0)
        ),
        "export_subsidy_eur_per_kwh": float(
            getattr(getattr(cfg, "reward", None), "export_subsidy_eur_per_kwh", np.nan)
        ),
    }


def assert_rollout_cfg_snapshot_matches(
    expected_snapshot: dict[str, Any],
    actual_snapshot: dict[str, Any],
    *,
    mismatch_prefix: str = "Rollout package",
) -> None:
    for field_name in (
        "test_start_date",
        "test_end_date",
        "prediction_mode",
        "forecast_backend",
        "future_horizon",
        "episode_limit",
        "price_protocol_version",
    ):
        _assert_strict_match(mismatch_prefix, field_name, expected_snapshot[field_name], actual_snapshot[field_name])
    _assert_str_sequence_match(
        mismatch_prefix,
        "agent_profiles",
        expected_snapshot["agent_profiles"],
        actual_snapshot["agent_profiles"],
    )
    _assert_int_sequence_match(
        mismatch_prefix,
        "agent_bus_ids",
        expected_snapshot["agent_bus_ids"],
        actual_snapshot["agent_bus_ids"],
    )
    _assert_float_sequence_match(
        mismatch_prefix,
        "load_scale",
        expected_snapshot["load_scale"],
        actual_snapshot["load_scale"],
    )
    _assert_float_sequence_match(
        mismatch_prefix,
        "pv_scale",
        expected_snapshot["pv_scale"],
        actual_snapshot["pv_scale"],
    )
    _assert_float_sequence_match(
        mismatch_prefix,
        "battery_controls.battery_capacity",
        expected_snapshot["battery_controls"]["battery_capacity"],
        actual_snapshot["battery_controls"]["battery_capacity"],
    )
    for field_name in ("max_charge_rate", "efficiency", "init_soc", "soc_min", "soc_max", "soc_target"):
        _assert_float_match(
            mismatch_prefix,
            f"battery_controls.{field_name}",
            float(expected_snapshot["battery_controls"][field_name]),
            float(actual_snapshot["battery_controls"][field_name]),
        )
    for field_name in (
        "v_min_pu",
        "v_max_pu",
        IMPORT_PRICE_MARKUP_KEY,
        "export_subsidy_eur_per_kwh",
    ):
        _assert_float_match(mismatch_prefix, field_name, expected_snapshot[field_name], actual_snapshot[field_name])


def assert_solver_fingerprint_matches(expected: dict[str, Any], actual: dict[str, Any], *, name: str) -> None:
    if set(expected) != set(actual):
        _raise_mismatch(name, "keys", sorted(expected), sorted(actual))
    for field_name in sorted(expected):
        expected_value = expected[field_name]
        actual_value = actual[field_name]
        if isinstance(expected_value, (float, int, np.floating, np.integer)) and isinstance(
            actual_value,
            (float, int, np.floating, np.integer),
        ):
            _assert_float_match(name, field_name, float(expected_value), float(actual_value))
            continue
        _assert_strict_match(name, field_name, expected_value, actual_value)


def table_required_files() -> tuple[str, ...]:
    return (
        "manifest.json",
        "diagnostics.json",
        "step_df.npz",
        "agent_df.npz",
        "grid_df.npz",
        "summary_df.npz",
    )


def _serialize_dataframe_for_npz(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    table = frame.copy()
    schema: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for column_index, column_name in enumerate(table.columns):
        series = table[column_name]
        key = f"col_{column_index}"
        if pd.api.types.is_datetime64_any_dtype(series):
            storage = "datetime"
            values = pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%f").to_numpy(dtype=np.str_)
        elif pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            storage = "native"
            values = np.asarray(series.to_numpy(copy=True))
        else:
            storage = "string"
            values = series.fillna("").astype(str).to_numpy(dtype=np.str_)
        arrays[key] = values
        schema.append(
            {
                "name": str(column_name),
                "dtype": str(series.dtype),
                "storage": str(storage),
                "key": key,
            }
        )
    return {"columns": schema, "row_count": int(len(table))}, arrays


def save_dataframe_npz(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    metadata, arrays = _serialize_dataframe_for_npz(frame)
    np.savez_compressed(path, **arrays)
    return metadata


def load_dataframe_npz(path: Path, *, metadata: dict[str, Any]) -> pd.DataFrame:
    column_payload = list(metadata.get("columns", []))
    if not column_payload:
        return pd.DataFrame()
    data: dict[str, Any] = {}
    ordered_columns: list[str] = []
    with np.load(path, allow_pickle=False) as archive:
        for column_meta in column_payload:
            column_name = str(column_meta["name"])
            ordered_columns.append(column_name)
            raw = np.asarray(archive[str(column_meta["key"])])
            storage = str(column_meta.get("storage", "native"))
            if storage == "datetime":
                data[column_name] = pd.to_datetime(raw.astype(str), errors="coerce")
            elif storage == "string":
                data[column_name] = raw.astype(str)
            else:
                data[column_name] = raw.astype(np.dtype(str(column_meta["dtype"])), copy=False)
    return pd.DataFrame(data, columns=ordered_columns)


def relabel_rollout_dataframe(frame: pd.DataFrame, controller_label: str) -> pd.DataFrame:
    relabeled = frame.copy()
    if "controller" in relabeled.columns:
        relabeled["controller"] = str(controller_label)
    return relabeled


__all__ = [
    "ROLLOUT_PACKAGE_VERSION",
    "assert_rollout_cfg_snapshot_matches",
    "assert_solver_fingerprint_matches",
    "build_rollout_cfg_snapshot_from_cfg",
    "json_default",
    "load_dataframe_npz",
    "relabel_rollout_dataframe",
    "save_dataframe_npz",
    "table_required_files",
]
