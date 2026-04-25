from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from scripts.utils.price_protocol import IMPORT_PRICE_COLUMN

BATTERY_POWER_COLUMN = "battery_power_kw"

STORAGE_CHARGE_COST_COLUMN = "storage_charge_cost_eur"
STORAGE_DISCHARGE_REVENUE_COLUMN = "storage_discharge_revenue_eur"
STORAGE_PROFIT_COLUMN = "storage_profit_eur"
STORAGE_OBJECTIVE_COLUMN = "storage_objective_eur"

STORAGE_CHARGE_COST_TOTAL_COLUMN = "storage_charge_cost_total_eur"
STORAGE_DISCHARGE_REVENUE_TOTAL_COLUMN = "storage_discharge_revenue_total_eur"
STORAGE_PROFIT_TOTAL_COLUMN = "storage_profit_total_eur"
STORAGE_OBJECTIVE_TOTAL_COLUMN = "storage_objective_total_eur"

STORAGE_STEP_COLUMNS = (
	STORAGE_CHARGE_COST_COLUMN,
	STORAGE_DISCHARGE_REVENUE_COLUMN,
	STORAGE_PROFIT_COLUMN,
	STORAGE_OBJECTIVE_COLUMN,
)

STORAGE_TOTAL_COLUMNS = (
	STORAGE_CHARGE_COST_TOTAL_COLUMN,
	STORAGE_DISCHARGE_REVENUE_TOTAL_COLUMN,
	STORAGE_PROFIT_TOTAL_COLUMN,
	STORAGE_OBJECTIVE_TOTAL_COLUMN,
)

LEGACY_STORAGE_PROFIT_COLUMNS = frozenset(
	{
		"storage_purchase_cost_eur",
		"storage_sale_revenue_eur",
		"storage_total_profit_eur",
		"storage_purchase_cost_eur_step",
		"storage_sale_revenue_eur_step",
		"storage_total_profit_eur_step",
	}
)


def reject_legacy_storage_profit_tables(
	tables: Mapping[str, pd.DataFrame],
	*,
	context: str,
	rerun_hint: str,
) -> None:
	legacy_hits = {
		table_name: sorted(
			str(column)
			for column in getattr(frame, "columns", [])
			if str(column) in LEGACY_STORAGE_PROFIT_COLUMNS
		)
		for table_name, frame in dict(tables).items()
	}
	legacy_hits = {table_name: columns for table_name, columns in legacy_hits.items() if columns}
	if legacy_hits:
		raise ValueError(
			f"{context} uses removed storage-profit alias column(s): {legacy_hits}. "
			"New contract expects battery_power_kw plus canonical storage_charge_cost_eur, "
			"storage_discharge_revenue_eur, storage_profit_eur, and storage_objective_eur "
			f"for step rows, with aggregate totals recomputed during compare. {rerun_hint}"
		)


def reject_legacy_storage_profit_meta(
	meta: Mapping[str, object],
	*,
	context: str,
	rerun_hint: str,
) -> None:
	legacy_keys = sorted(str(key) for key in dict(meta) if str(key) in LEGACY_STORAGE_PROFIT_COLUMNS)
	if legacy_keys:
		raise ValueError(
			f"{context} uses removed storage-profit alias metadata key(s): {legacy_keys}. "
			"New contract expects canonical storage_charge_cost_total_eur, "
			"storage_discharge_revenue_total_eur, storage_profit_total_eur, and "
			f"storage_objective_total_eur. {rerun_hint}"
		)


def _require_finite_dt_hours(dt_hours: float, *, context: str) -> float:
	value = float(dt_hours)
	if not np.isfinite(value) or value <= 0.0:
		raise ValueError(f"{context} requires a positive finite dt_hours, got {dt_hours!r}.")
	return value


def recompute_storage_profit_step_columns(
	step_df: pd.DataFrame,
	*,
	dt_hours: float,
	context: str,
) -> pd.DataFrame:
	reject_legacy_storage_profit_tables(
		{"step_df": step_df},
		context=context,
		rerun_hint="Re-run the notebook that produced this rollout record.",
	)
	result = step_df.copy()
	if result.empty:
		for column in STORAGE_STEP_COLUMNS:
			if column not in result.columns:
				result[column] = pd.Series(dtype=float)
		return result
	required = {BATTERY_POWER_COLUMN, IMPORT_PRICE_COLUMN}
	missing = sorted(required.difference(result.columns))
	if missing:
		raise ValueError(
			f"{context} requires step_df column(s) {missing} to recompute storage profit. "
			"Expected canonical rollout output from battery_power_kw and actual import price. "
			"Re-run the notebook that produced this rollout record."
		)
	dt = _require_finite_dt_hours(dt_hours, context=context)
	battery_power = result[BATTERY_POWER_COLUMN].to_numpy(dtype=np.float64)
	price = result[IMPORT_PRICE_COLUMN].to_numpy(dtype=np.float64)
	if not np.all(np.isfinite(battery_power)):
		raise ValueError(f"{context} contains non-finite battery_power_kw values.")
	if not np.all(np.isfinite(price)):
		raise ValueError(f"{context} contains non-finite {IMPORT_PRICE_COLUMN} values.")
	charge_cost = np.maximum(battery_power, 0.0) * dt * price
	discharge_revenue = np.maximum(-battery_power, 0.0) * dt * price
	profit = discharge_revenue - charge_cost
	result[STORAGE_CHARGE_COST_COLUMN] = charge_cost
	result[STORAGE_DISCHARGE_REVENUE_COLUMN] = discharge_revenue
	result[STORAGE_PROFIT_COLUMN] = profit
	result[STORAGE_OBJECTIVE_COLUMN] = -profit
	return result


def recompute_storage_profit_summary(
	step_df: pd.DataFrame,
	*,
	dt_hours: float,
	context: str,
) -> dict[str, float]:
	recomputed = recompute_storage_profit_step_columns(
		step_df,
		dt_hours=dt_hours,
		context=context,
	)
	if recomputed.empty:
		return {column: float("nan") for column in STORAGE_TOTAL_COLUMNS}
	return {
		STORAGE_CHARGE_COST_TOTAL_COLUMN: float(recomputed[STORAGE_CHARGE_COST_COLUMN].sum()),
		STORAGE_DISCHARGE_REVENUE_TOTAL_COLUMN: float(
			recomputed[STORAGE_DISCHARGE_REVENUE_COLUMN].sum()
		),
		STORAGE_PROFIT_TOTAL_COLUMN: float(recomputed[STORAGE_PROFIT_COLUMN].sum()),
		STORAGE_OBJECTIVE_TOTAL_COLUMN: float(recomputed[STORAGE_OBJECTIVE_COLUMN].sum()),
	}
