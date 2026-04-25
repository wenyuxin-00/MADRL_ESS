from __future__ import annotations

import numpy as np


def compute_storage_profit_components(
    *,
    battery_power_kw,
    price_eur_per_kwh,
    dt_hours: float,
) -> dict[str, np.ndarray]:
    battery_power = np.asarray(battery_power_kw, dtype=np.float32)
    price = np.asarray(price_eur_per_kwh, dtype=np.float32)
    charge_kw = np.maximum(battery_power, np.float32(0.0)).astype(np.float32)
    discharge_kw = np.maximum(-battery_power, np.float32(0.0)).astype(np.float32)
    charge_cost = (charge_kw * np.float32(dt_hours) * price).astype(np.float32)
    discharge_revenue = (discharge_kw * np.float32(dt_hours) * price).astype(np.float32)
    storage_profit = (discharge_revenue - charge_cost).astype(np.float32)
    return {
        "storage_charge_cost_eur": charge_cost,
        "storage_discharge_revenue_eur": discharge_revenue,
        "storage_profit_eur": storage_profit,
        "storage_objective_eur": (-storage_profit).astype(np.float32),
    }


def compute_storage_profit(
    *,
    battery_power_kw,
    price_eur_per_kwh,
    dt_hours: float,
) -> np.ndarray:
    return compute_storage_profit_components(
        battery_power_kw=battery_power_kw,
        price_eur_per_kwh=price_eur_per_kwh,
        dt_hours=dt_hours,
    )["storage_profit_eur"]
