from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def _record_dir(run_dir: str | Path, forecast_mode: str, controller: str) -> Path:
    return Path(run_dir) / "results" / str(forecast_mode) / str(controller) / "record"


def _agent_record(run_dir: str | Path, forecast_mode: str, controller: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    record = _record_dir(run_dir, forecast_mode, controller)
    agent = pd.read_parquet(record / "agent.parquet")
    step = pd.read_parquet(record / "step.parquet")
    meta = json.loads((record / "meta.json").read_text(encoding="utf-8"))
    price_cols = [c for c in ("episode_idx", "step", "timestamp", "import_price", "wholesale_price") if c in step.columns]
    agent = agent.merge(step[price_cols].drop_duplicates(["episode_idx", "step"]), on=["episode_idx", "step", "timestamp"], how="left")
    return agent, step, meta


def _x_values(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"]).to_numpy(), "timestamp"
    return np.arange(len(frame), dtype=np.int64), "time step"


def _plot_agent_battery_power_soc_price(agent: pd.DataFrame, run_dir: Path, scheme_name: str) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig_dir = run_dir / "figures" / "battery_diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for agent_id, frame in agent.sort_values(["episode_idx", "step"]).groupby("agent_id", sort=True):
        x, xlabel = _x_values(frame)
        fig, ax_power = plt.subplots(figsize=(11, 4.2))
        ax_soc = ax_power.twinx()
        ax_price = ax_power.twinx()
        ax_price.spines["right"].set_position(("axes", 1.10))
        ax_power.plot(x, frame["e_bat"].to_numpy(dtype=float), label="Battery power [kW]", color="#2563eb", linewidth=1.35)
        ax_power.axhline(0.0, color="#64748b", linewidth=0.8, alpha=0.65)
        ax_soc.plot(x, frame["soc"].to_numpy(dtype=float), label="SOC", color="#16a34a", linewidth=1.2)
        ax_soc.axhline(0.05, color="#16a34a", linestyle=":", linewidth=0.8, alpha=0.75)
        ax_soc.axhline(0.95, color="#16a34a", linestyle=":", linewidth=0.8, alpha=0.75)
        price_col = "import_price" if "import_price" in frame.columns else "wholesale_price"
        ax_price.plot(x, frame[price_col].to_numpy(dtype=float), label="Price [EUR/kWh]", color="#111827", linestyle="--", linewidth=1.1)
        ax_power.set_title(f"Agent {int(agent_id)} battery power, SOC, and price")
        ax_power.set_xlabel(xlabel)
        ax_power.set_ylabel("Battery power [kW]")
        ax_soc.set_ylabel("SOC")
        ax_price.set_ylabel("Price [EUR/kWh]")
        handles, labels = [], []
        for axis in (ax_power, ax_soc, ax_price):
            h, l = axis.get_legend_handles_labels()
            handles.extend(h); labels.extend(l)
        ax_power.legend(handles, labels, fontsize=8, loc="upper left", ncol=3)
        ax_power.grid(alpha=0.22)
        fig.autofmt_xdate()
        fig.tight_layout()
        out = fig_dir / f"{scheme_name}_agent_{int(agent_id)}_battery_power_soc_price.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths[f"agent_{int(agent_id)}"] = out
    return paths


def _plot_agent_ev_charge_soc_price_night(agent: pd.DataFrame, run_dir: Path, scheme_name: str, meta: dict[str, Any], config: dict[str, Any]) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    env_cfg = config.get("env", {})
    episode_steps = int(env_cfg.get("episode_steps", meta.get("episode_steps", 96)))
    arrival_step = int(env_cfg.get("ev_arrival_step", meta.get("ev_arrival_step", 72)))
    departure_step = int(env_cfg.get("ev_departure_step", meta.get("ev_departure_step", 28)))
    required_soc = float(env_cfg.get("ev_departure_soc_req", meta.get("ev_departure_soc_req", 0.90)))
    daily_step = agent["step"].astype(int) % max(1, episode_steps)
    if arrival_step > departure_step:
        night = (daily_step >= arrival_step) | (daily_step < departure_step)
    else:
        night = (daily_step >= arrival_step) & (daily_step < departure_step)
    night_agent = agent.loc[night].copy()
    fig_dir = run_dir / "figures" / "battery_diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if night_agent.empty:
        return paths
    price_col = "import_price" if "import_price" in night_agent.columns else "wholesale_price"
    for agent_id, frame in night_agent.sort_values(["episode_idx", "step"]).groupby("agent_id", sort=True):
        x, xlabel = _x_values(frame)
        fig, ax_power = plt.subplots(figsize=(11, 4.2))
        ax_soc = ax_power.twinx()
        ax_price = ax_power.twinx()
        ax_price.spines["right"].set_position(("axes", 1.10))
        ax_power.plot(x, frame["ev_charge_kw"].to_numpy(dtype=float), label="EV charge [kW]", color="#7c3aed", linewidth=1.35)
        ax_power.axhline(0.0, color="#64748b", linewidth=0.8, alpha=0.65)
        ax_soc.plot(x, frame["ev_soc"].to_numpy(dtype=float), label="EV SOC", color="#16a34a", linewidth=1.2)
        ax_soc.axhline(required_soc, color="#16a34a", linestyle=":", linewidth=0.9, alpha=0.85, label=f"Required SOC {required_soc:.2f}")
        ax_price.plot(x, frame[price_col].to_numpy(dtype=float), label="Price [EUR/kWh]", color="#111827", linestyle="--", linewidth=1.1)
        ax_power.set_title(f"Agent {int(agent_id)} EV charging, SOC, and price (18:00-07:00)")
        ax_power.set_xlabel(xlabel)
        ax_power.set_ylabel("EV charging power [kW]")
        ax_soc.set_ylabel("EV SOC")
        ax_price.set_ylabel("Price [EUR/kWh]")
        handles, labels = [], []
        for axis in (ax_power, ax_soc, ax_price):
            h, l = axis.get_legend_handles_labels()
            handles.extend(h); labels.extend(l)
        ax_power.legend(handles, labels, fontsize=8, loc="upper left", ncol=3)
        ax_power.grid(alpha=0.22)
        fig.autofmt_xdate()
        fig.tight_layout()
        out = fig_dir / f"{scheme_name}_agent_{int(agent_id)}_ev_charge_soc_price_night.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths[f"agent_{int(agent_id)}"] = out
    return paths


def _cost_breakdown(agent: pd.DataFrame, run_dir: Path, scheme_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cumulative = agent[["episode_idx", "step", "timestamp"]].drop_duplicates(["episode_idx", "step", "timestamp"]).sort_values(["episode_idx", "step"]).reset_index(drop=True)
    cumulative["battery_cost_eur"] = 0.0
    cumulative["ev_cost_eur"] = 0.0
    for agent_id, frame in agent.sort_values(["episode_idx", "step"]).groupby("agent_id", sort=True):
        battery_cost = -frame["storage_profit_eur"].astype(float)
        ev_cost = frame["ev_charging_cost_eur"].astype(float)
        rows.append({"component": f"battery_agent_{int(agent_id)}", "cost_eur": float(battery_cost.sum()), "mean_step_cost_eur": float(battery_cost.mean()), "min_step_cost_eur": float(battery_cost.min()), "max_step_cost_eur": float(battery_cost.max())})
        rows.append({"component": f"ev_agent_{int(agent_id)}", "cost_eur": float(ev_cost.sum()), "mean_step_cost_eur": float(ev_cost.mean()), "min_step_cost_eur": float(ev_cost.min()), "max_step_cost_eur": float(ev_cost.max())})
        part = frame[["episode_idx", "step", "timestamp"]].copy()
        part[f"battery_agent_{int(agent_id)}_cost_eur"] = battery_cost.to_numpy(dtype=float)
        part[f"ev_agent_{int(agent_id)}_cost_eur"] = ev_cost.to_numpy(dtype=float)
        cumulative = cumulative.merge(part, on=["episode_idx", "step", "timestamp"], how="left")
        cumulative["battery_cost_eur"] += part[f"battery_agent_{int(agent_id)}_cost_eur"].to_numpy(dtype=float)
        cumulative["ev_cost_eur"] += part[f"ev_agent_{int(agent_id)}_cost_eur"].to_numpy(dtype=float)
    component_cols = [c for c in cumulative.columns if c.endswith("_cost_eur") and c not in ("battery_cost_eur", "ev_cost_eur", "total_cost_eur")]
    cumulative["total_cost_eur"] = cumulative["battery_cost_eur"] + cumulative["ev_cost_eur"]
    for column in ["battery_cost_eur", "ev_cost_eur", "total_cost_eur", *component_cols]:
        cumulative[f"cumulative_{column}"] = cumulative[column].astype(float).cumsum()
    rows.extend([
        {"component": "battery_all_agents", "cost_eur": float(cumulative["battery_cost_eur"].sum()), "mean_step_cost_eur": float(cumulative["battery_cost_eur"].mean()), "min_step_cost_eur": float(cumulative["battery_cost_eur"].min()), "max_step_cost_eur": float(cumulative["battery_cost_eur"].max())},
        {"component": "ev_all_agents", "cost_eur": float(cumulative["ev_cost_eur"].sum()), "mean_step_cost_eur": float(cumulative["ev_cost_eur"].mean()), "min_step_cost_eur": float(cumulative["ev_cost_eur"].min()), "max_step_cost_eur": float(cumulative["ev_cost_eur"].max())},
        {"component": "battery_plus_ev_total", "cost_eur": float(cumulative["total_cost_eur"].sum()), "mean_step_cost_eur": float(cumulative["total_cost_eur"].mean()), "min_step_cost_eur": float(cumulative["total_cost_eur"].min()), "max_step_cost_eur": float(cumulative["total_cost_eur"].max())},
    ])
    table_dir = run_dir / "tables" / "battery_diagnostics"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / f"{scheme_name}_cumulative_cost_breakdown.csv", index=False)
    cumulative.to_csv(table_dir / f"{scheme_name}_cumulative_cost_timeseries.csv", index=False)
    return summary, cumulative


def compute_battery_soc_price_stats(
    run_dir: str | Path,
    *,
    forecast_mode: str = "lstm",
    controller: str = "MADRL_BASE",
    scheme_name: str = "madrl_base",
    soc_min: float = 0.05,
    soc_max: float = 0.95,
    boundary_epsilon: float = 0.02,
) -> pd.DataFrame:
    agent, _, _ = _agent_record(run_dir, forecast_mode, controller)
    price_col = "import_price" if "import_price" in agent.columns else "wholesale_price"
    rows = []
    for agent_id, frame in agent.groupby("agent_id", sort=True):
        battery = frame["e_bat"].astype(float)
        price = frame[price_col].astype(float)
        soc = frame["soc"].astype(float)
        charging = battery > 1e-6
        discharging = battery < -1e-6
        near_min = soc <= float(soc_min) + float(boundary_epsilon)
        near_max = soc >= float(soc_max) - float(boundary_epsilon)
        rows.append({
            "agent": int(agent_id),
            "near_soc_min_ratio": float(near_min.mean()),
            "near_soc_max_ratio": float(near_max.mean()),
            "charge_near_max_count": int((charging & near_max).sum()),
            "discharge_near_min_count": int((discharging & near_min).sum()),
            "avg_price_when_charging": float(price[charging].mean()) if charging.any() else np.nan,
            "avg_price_when_discharging": float(price[discharging].mean()) if discharging.any() else np.nan,
            "charging_minus_discharging_avg_price": float(price[charging].mean() - price[discharging].mean()) if charging.any() and discharging.any() else np.nan,
            "battery_power_price_corr": float(battery.corr(price)) if battery.nunique() > 1 and price.nunique() > 1 else np.nan,
            "storage_profit_eur": float(frame["storage_profit_eur"].astype(float).sum()),
            "battery_abs_mean_kw": float(battery.abs().mean()),
            "battery_mean_kw": float(battery.mean()),
        })
    stats = pd.DataFrame(rows)
    table_dir = Path(run_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(table_dir / f"{scheme_name}_battery_soc_price_diagnostics.csv", index=False)
    return stats


def _config_payload(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _train_summary(run_dir: Path, scheme_name: str) -> pd.DataFrame:
    path = run_dir / "tables" / f"madrl_train_summary_{scheme_name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _reward_curve(run_dir: Path, scheme_name: str) -> pd.DataFrame:
    path = run_dir / "tables" / f"madrl_reward_curves_{scheme_name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _metrics(run_dir: Path, forecast_mode: str, controller: str) -> pd.DataFrame:
    path = _record_dir(run_dir, forecast_mode, controller) / "metrics.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _stat_row(source: str, component: str, values: pd.Series) -> dict[str, Any]:
    series = pd.to_numeric(values, errors="coerce")
    return {
        "source": source, "component": component, "mean": float(series.mean()), "sum": float(series.sum()),
        "min": float(series.min()), "max": float(series.max()), "abs_mean": float(series.abs().mean()), "count": int(series.count()),
    }


def build_madrl_battery_report(
    run_dir: str | Path,
    *,
    forecast_mode: str = "lstm",
    controller: str = "MADRL_BASE",
    scheme_name: str = "madrl_base",
    experiment_name: str = "madrl_base_EV_soft",
) -> dict[str, pd.DataFrame]:
    run_path = Path(run_dir)
    agent, step, meta = _agent_record(run_path, forecast_mode, controller)
    config = _config_payload(run_path)
    train_summary = _train_summary(run_path, scheme_name)
    reward_curve = _reward_curve(run_path, scheme_name)
    metrics = _metrics(run_path, forecast_mode, controller)
    price_col = "import_price" if "import_price" in agent.columns else "wholesale_price"
    battery = agent["e_bat"].astype(float)
    price = agent[price_col].astype(float)
    ev = agent["ev_charge_kw"].astype(float)
    connected = agent["ev_available"].astype(float) > 0.5
    soc = agent["soc"].astype(float)
    charging = battery > 1e-6
    discharging = battery < -1e-6
    idle = battery.abs() <= 1e-6
    ev_charging = ev > 1e-6
    ev_idle_connected = connected & ~ev_charging
    soc_min = float(config.get("env", {}).get("soc_min", 0.05))
    soc_max = float(config.get("env", {}).get("soc_max", 0.95))
    eps = float(config.get("reward", {}).get("soc_boundary_epsilon", 0.02))
    near_min = soc <= soc_min + eps
    near_max = soc >= soc_max - eps
    boundary_push = ((battery < -1e-6) & near_min) | ((battery > 1e-6) & near_max)

    actual_episodes = int(train_summary["episodes"].iloc[-1]) if "episodes" in train_summary.columns and not train_summary.empty else int(config.get("train", {}).get("train_episodes", 0))
    train_overview = pd.DataFrame([{
        "experiment": experiment_name,
        "run_dir": str(run_path),
        "config_json_train_episodes": int(config.get("train", {}).get("train_episodes", 0)),
        "actual_train_episodes": actual_episodes,
        "train_window_days": int(config.get("env", {}).get("train_window_days", 0)),
        "episode_steps": int(config.get("env", {}).get("episode_steps", 0)),
        "train_episode_steps": int(config.get("env", {}).get("train_window_days", 0)) * int(config.get("env", {}).get("episode_steps", 0)),
        "learning_starts": int(config.get("train", {}).get("learning_starts", 0)),
        "actor_learning_starts": int(config.get("train", {}).get("actor_learning_starts", 0)),
        "updates": int(train_summary["updates"].iloc[-1]) if "updates" in train_summary.columns and not train_summary.empty else np.nan,
        "eval_episode_indices": tuple(int(x) for x in meta.get("eval_episode_indices", [])),
    }])

    rollout_effect = pd.DataFrame([{
        "storage_charge_cost_total_eur": float(metrics["storage_charge_cost_total_eur"].iloc[0]) if "storage_charge_cost_total_eur" in metrics.columns and not metrics.empty else float(np.maximum(battery, 0.0).mul(price).mul(float(meta.get("dt_hours", 0.25))).sum()),
        "storage_discharge_revenue_total_eur": float(metrics["storage_discharge_revenue_total_eur"].iloc[0]) if "storage_discharge_revenue_total_eur" in metrics.columns and not metrics.empty else float(np.maximum(-battery, 0.0).mul(price).mul(float(meta.get("dt_hours", 0.25))).sum()),
        "storage_profit_total_eur": float(metrics["storage_profit_total_eur"].iloc[0]) if "storage_profit_total_eur" in metrics.columns and not metrics.empty else float(agent["storage_profit_eur"].astype(float).sum()),
        "ev_charging_cost_total_eur": float(agent["ev_charging_cost_eur"].astype(float).sum()),
        "ev_departure_penalty_total": float(agent["ev_departure_penalty"].astype(float).sum()),
        "total_eur": float(metrics["total_eur"].iloc[0]) if "total_eur" in metrics.columns and not metrics.empty else np.nan,
        "trafo_overload_steps": int(metrics["trafo_overload_steps"].iloc[0]) if "trafo_overload_steps" in metrics.columns and not metrics.empty else np.nan,
        "battery_net_power_kw_mean_abs": float(metrics["battery_net_power_kw_mean_abs"].iloc[0]) if "battery_net_power_kw_mean_abs" in metrics.columns and not metrics.empty else float(step["battery_power_kw"].abs().mean()) if "battery_power_kw" in step.columns else np.nan,
    }])

    battery_price_dispatch = pd.DataFrame([{
        "battery_power_price_corr": float(battery.corr(price)) if battery.nunique() > 1 and price.nunique() > 1 else np.nan,
        "charging_avg_price": float(price[charging].mean()) if charging.any() else np.nan,
        "discharging_avg_price": float(price[discharging].mean()) if discharging.any() else np.nan,
        "idle_avg_price": float(price[idle].mean()) if idle.any() else np.nan,
        "charging_minus_discharging_avg_price": float(price[charging].mean() - price[discharging].mean()) if charging.any() and discharging.any() else np.nan,
        "charging_rows": int(charging.sum()),
        "discharging_rows": int(discharging.sum()),
        "idle_rows": int(idle.sum()),
        "battery_power_abs_mean_kw": float(battery.abs().mean()),
    }])

    by_agent = compute_battery_soc_price_stats(run_path, forecast_mode=forecast_mode, controller=controller, scheme_name=scheme_name, soc_min=soc_min, soc_max=soc_max, boundary_epsilon=eps)
    ev_behavior = pd.DataFrame([{
        "ev_charge_price_corr": float(ev.corr(price)) if ev.nunique() > 1 and price.nunique() > 1 else np.nan,
        "ev_charge_mean_kw": float(ev.mean()),
        "ev_charging_rows": int(ev_charging.sum()),
        "ev_idle_connected_rows": int(ev_idle_connected.sum()),
        "ev_connected_rows": int(connected.sum()),
        "ev_charging_avg_price": float(price[ev_charging].mean()) if ev_charging.any() else np.nan,
        "ev_idle_connected_avg_price": float(price[ev_idle_connected].mean()) if ev_idle_connected.any() else np.nan,
        "ev_cost_total_eur": float(agent["ev_charging_cost_eur"].astype(float).sum()),
        "ev_departure_penalty_total": float(agent["ev_departure_penalty"].astype(float).sum()),
        "ev_overcharge_gap_total": float(agent["ev_overcharge_gap"].astype(float).sum()) if "ev_overcharge_gap" in agent.columns else 0.0,
        "ev_overcharge_penalty_total": float(agent["ev_overcharge_penalty"].astype(float).sum()) if "ev_overcharge_penalty" in agent.columns else 0.0,
    }])
    dt_hours = float(meta.get("dt_hours", 0.25))
    ev_requested = agent["ev_charge_kw_requested"].astype(float) if "ev_charge_kw_requested" in agent.columns else agent["ev_charge_kw"].astype(float)
    ev_executed = agent["ev_charge_kw_executed"].astype(float) if "ev_charge_kw_executed" in agent.columns else agent["ev_charge_kw"].astype(float)
    ev_clipped = agent["ev_charge_kw_clipped"].astype(float) if "ev_charge_kw_clipped" in agent.columns else (ev_requested - ev_executed).clip(lower=0.0)
    ev_headroom = agent["ev_soc_headroom_kwh"].astype(float) if "ev_soc_headroom_kwh" in agent.columns else pd.Series(np.nan, index=agent.index, dtype=float)
    ev_overcharge_diagnostics = pd.DataFrame([{
        "total_ev_charge_requested_kwh": float((ev_requested.clip(lower=0.0) * dt_hours).sum()),
        "total_ev_charge_executed_kwh": float((ev_executed.clip(lower=0.0) * dt_hours).sum()),
        "total_ev_charge_clipped_kwh": float((ev_clipped.clip(lower=0.0) * dt_hours).sum()),
        "ev_charge_clipped_max_kw": float(ev_clipped.max()),
        "ev_charge_clipped_mean_kw": float(ev_clipped.mean()),
        "ev_charge_clipped_rows": int((ev_clipped > 1e-9).sum()),
        "ev_soc_headroom_min_kwh": float(ev_headroom.min()),
        "ev_soc_headroom_mean_kwh": float(ev_headroom.mean()),
    }])
    soc_limits = pd.DataFrame([{
        "soc_mean": float(soc.mean()),
        "soc_min_observed": float(soc.min()),
        "soc_max_observed": float(soc.max()),
        "near_soc_min_ratio": float(near_min.mean()),
        "near_soc_max_ratio": float(near_max.mean()),
        "boundary_push_ratio_inferred": float(boundary_push.mean()),
        "battery_zero_ratio": float(idle.mean()),
    }])

    reward_rows = []
    for column in ("storage_profit_eur", "ev_charging_cost_eur", "ev_departure_penalty", "ev_overcharge_penalty"):
        if column in agent.columns:
            reward_rows.append(_stat_row("rollout.agent_df", column, agent[column]))
    overcharge_penalty = agent["ev_overcharge_penalty"].astype(float) if "ev_overcharge_penalty" in agent.columns else pd.Series(0.0, index=agent.index, dtype=float)
    partial = agent["storage_profit_eur"].astype(float) - agent["ev_charging_cost_eur"].astype(float) - agent["ev_departure_penalty"].astype(float) - overcharge_penalty
    reward_rows.append(_stat_row("rollout.agent_df derived partial", "storage_profit - ev_cost - ev_departure_penalty - ev_overcharge_penalty", partial))
    for column in ("soc_penalty_total", "voltage_penalty_total", "line_penalty_total", "trafo_penalty_total", "reward_total", "storage_profit_eur", "ev_charging_cost_eur", "ev_departure_penalty_total", "ev_overcharge_penalty_total"):
        if column in step.columns:
            reward_rows.append(_stat_row("rollout.step_df", column, step[column]))
    for column in ("total_reward", "madrl_r_inc", "madrl_r_action_penalty", "madrl_r_soc_regularization", "madrl_r_throughput_bonus", "madrl_r_safe_total"):
        if column in reward_curve.columns:
            reward_rows.append(_stat_row("train reward curve per episode", column, reward_curve[column]))
    reward_scale = pd.DataFrame(reward_rows)
    cost_breakdown, cumulative_cost_timeseries = _cost_breakdown(agent, run_path, scheme_name)

    if reward_curve.empty:
        learning_windows = pd.DataFrame()
    else:
        first20 = reward_curve.head(20)
        last20 = reward_curve.tail(min(20, len(reward_curve)))
        learning_windows = pd.DataFrame([
            {"window": "first20", "reward_mean": float(first20["total_reward"].mean()), "madrl_r_inc_mean": float(first20["madrl_r_inc"].mean()), "action_penalty_mean": float(first20["madrl_r_action_penalty"].mean()), "throughput_bonus_mean": float(first20["madrl_r_throughput_bonus"].mean())},
            {"window": "last20", "reward_mean": float(last20["total_reward"].mean()), "madrl_r_inc_mean": float(last20["madrl_r_inc"].mean()), "action_penalty_mean": float(last20["madrl_r_action_penalty"].mean()), "throughput_bonus_mean": float(last20["madrl_r_throughput_bonus"].mean())},
            {"window": "all", "reward_mean": float(reward_curve["total_reward"].mean()), "madrl_r_inc_mean": float(reward_curve["madrl_r_inc"].mean()), "action_penalty_mean": float(reward_curve["madrl_r_action_penalty"].mean()), "throughput_bonus_mean": float(reward_curve["madrl_r_throughput_bonus"].mean())},
        ])

    tables = {
        "train_overview": train_overview,
        "rollout_effect": rollout_effect,
        "battery_price_dispatch": battery_price_dispatch,
        "by_agent_battery": by_agent,
        "ev_behavior": ev_behavior,
        "ev_overcharge_diagnostics": ev_overcharge_diagnostics,
        "soc_limits": soc_limits,
        "reward_scale": reward_scale,
        "cumulative_cost_breakdown": cost_breakdown,
        "cumulative_cost_timeseries": cumulative_cost_timeseries,
        "learning_windows": learning_windows,
    }
    table_dir = run_path / "tables" / "battery_diagnostics"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{scheme_name}_{name}.csv", index=False)
    return tables


def plot_madrl_training_convergence(
    run_dir: str | Path,
    *,
    scheme_name: str = "madrl_base",
    rolling_window: int = 10,
    tail_window: int = 20,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rewards = _reward_curve(run_path, scheme_name)
    if rewards.empty:
        raise FileNotFoundError(f"Expected reward curve CSV for scheme {scheme_name!r} under {run_path / 'tables'}.")
    rewards = rewards.sort_values("episode").reset_index(drop=True)
    window = max(1, min(int(rolling_window), len(rewards)))
    tail = max(1, min(int(tail_window), len(rewards)))
    first = rewards.head(tail)
    last = rewards.tail(tail)
    summary = pd.DataFrame([{
        "scheme": scheme_name,
        "episodes": int(rewards["episode"].max()),
        "rolling_window": window,
        "tail_window": tail,
        "first_reward_mean": float(first["total_reward"].mean()),
        "last_reward_mean": float(last["total_reward"].mean()),
        "reward_mean_improvement": float(last["total_reward"].mean() - first["total_reward"].mean()),
        "last_reward_std": float(last["total_reward"].std(ddof=0)),
        "last_reward_min": float(last["total_reward"].min()),
        "last_reward_max": float(last["total_reward"].max()),
        "final_reward": float(rewards["total_reward"].iloc[-1]),
        "best_reward": float(rewards["total_reward"].max()),
        "first_storage_profit_mean": float(first["madrl_r_inc"].mean()) if "madrl_r_inc" in rewards.columns else np.nan,
        "last_storage_profit_mean": float(last["madrl_r_inc"].mean()) if "madrl_r_inc" in rewards.columns else np.nan,
        "storage_profit_mean_improvement": float(last["madrl_r_inc"].mean() - first["madrl_r_inc"].mean()) if "madrl_r_inc" in rewards.columns else np.nan,
    }])

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})
    episodes = rewards["episode"].to_numpy(dtype=float)
    total = rewards["total_reward"].to_numpy(dtype=float)
    rolling = rewards["total_reward"].rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)
    axes[0].plot(episodes, total, color="#94a3b8", linewidth=0.9, alpha=0.6, label="episode reward")
    axes[0].plot(episodes, rolling, color="#111827", linewidth=1.9, label=f"rolling mean ({window})")
    axes[0].axhline(float(last["total_reward"].mean()), color="#2563eb", linestyle="--", linewidth=1.0, alpha=0.75, label=f"last {tail} mean")
    axes[0].set_title("MADRL training convergence")
    axes[0].set_ylabel("total reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3)
    component_specs = [
        ("madrl_r_inc", "storage profit", "#2563eb"),
        ("madrl_r_action_penalty", "action penalty", "#f59e0b"),
        ("madrl_r_soc_regularization", "SOC regularization", "#16a34a"),
        ("madrl_r_throughput_bonus", "throughput bonus", "#14b8a6"),
        ("madrl_r_safe_total", "safety total", "#dc2626"),
    ]
    for column, label, color in component_specs:
        if column in rewards.columns:
            values = rewards[column].rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)
            axes[1].plot(episodes, values, linewidth=1.25, alpha=0.9, color=color, label=label)
    axes[1].set_title("Rolling reward components")
    axes[1].set_ylabel("signed component")
    axes[1].set_xlabel("episode")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig_dir = run_path / "figures" / "battery_diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{scheme_name}_training_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    table_dir = run_path / "tables" / "battery_diagnostics"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_dir / f"{scheme_name}_training_convergence_summary.csv", index=False)
    return {"summary": summary, "reward_curve": rewards, "figure": out}


def _load_run_cfg(run_dir: Path):
    from utils.run_artifacts import load_experiment_context
    cfg, _ = load_experiment_context(run_dir)
    return cfg


def _try_action_projection_trace(run_dir: Path, forecast_mode: str, controller: str, scheme_name: str, meta: dict[str, Any]) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import torch
        from controllers.madrl import MADRLController
        from data.share_data import load_share_data
        from envs.grid_env import build_env
    except Exception as exc:
        return None, f"raw action trace skipped because runtime imports failed: {exc!r}"
    try:
        cfg = _load_run_cfg(run_dir)
        cfg.model.action_dim = 3
        cfg.env.ev_enabled = True
        cfg.env.ev_departure_constraint_mode = str(meta.get("ev_departure_constraint_mode", getattr(cfg.env, "ev_departure_constraint_mode", "soft")))
        cfg.env.ev_hard_projection_enabled = bool(meta.get("ev_hard_projection_enabled", getattr(cfg.env, "ev_hard_projection_enabled", False)))
        cfg.env.ev_emergency_charging_enabled = bool(meta.get("ev_emergency_charging_enabled", getattr(cfg.env, "ev_emergency_charging_enabled", False)))
        train_summary_path = run_dir / "tables" / f"madrl_train_summary_{scheme_name}.csv"
        train_summary = pd.read_csv(train_summary_path)
        model_path = Path(str(train_summary.iloc[-1]["model_path"]))
        if not model_path.is_absolute():
            model_path = run_dir / model_path.relative_to(run_dir) if str(model_path).startswith(str(run_dir)) else Path(model_path)
        share_data = load_share_data(run_dir / "share_data", cfg)
        env = build_env(cfg, "eval", forecast_mode=forecast_mode, share_data=share_data)
        policy = MADRLController.load(cfg, model_path)
        rows = []
        eval_indices = [int(x) for x in meta.get("eval_episode_indices", [0])]
        for episode_idx in eval_indices:
            obs, state = env.reset(episode_index=episode_idx)
            policy.reset(state)
            done = False
            while not done:
                obs_t = {key: torch.as_tensor(value, dtype=torch.float32, device=policy.device).unsqueeze(0) for key, value in obs.items()}
                with torch.inference_mode():
                    raw = policy._raw_actor_action(obs_t)
                    projected = policy._postprocess(obs_t, raw)
                raw_np = raw[0].detach().cpu().numpy().astype(np.float32)
                projected_np = projected[0].detach().cpu().numpy().astype(np.float32)
                step = int(env.cur_step)
                next_obs, _, done, _, _ = env.step(projected_np)
                for agent_id in range(int(cfg.env.num_agents)):
                    raw_bat = float(raw_np[agent_id, 0])
                    projected_bat = float(projected_np[agent_id, 0])
                    gap = projected_bat - raw_bat
                    rows.append({
                        "episode_idx": episode_idx,
                        "step": step,
                        "agent_id": agent_id,
                        "raw_battery_action": raw_bat,
                        "projected_battery_action": projected_bat,
                        "battery_action_gap": gap,
                        "battery_action_clipped": bool(abs(gap) > 1e-5),
                    })
                obs = next_obs
        env.close()
        trace = pd.DataFrame(rows)
        table_dir = run_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        trace.to_csv(table_dir / f"{scheme_name}_battery_action_projection_trace.csv", index=False)
        summary = trace.groupby("agent_id", as_index=False).agg(
            clipped_ratio=("battery_action_clipped", "mean"),
            clipped_count=("battery_action_clipped", "sum"),
            action_gap_abs_mean=("battery_action_gap", lambda s: float(np.abs(s).mean())),
            action_gap_min=("battery_action_gap", "min"),
            action_gap_max=("battery_action_gap", "max"),
        )
        summary.to_csv(table_dir / f"{scheme_name}_battery_action_projection_summary.csv", index=False)
        return trace, None
    except Exception as exc:
        return None, f"raw action trace skipped: {exc!r}"


def run_madrl_battery_diagnostics(
    run_dir: str | Path,
    *,
    forecast_mode: str = "lstm",
    controller: str = "MADRL_BASE",
    scheme_name: str = "madrl_base",
    include_action_trace: bool = True,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    agent, _, meta = _agent_record(run_path, forecast_mode, controller)
    config = _config_payload(run_path)
    stats = compute_battery_soc_price_stats(run_path, forecast_mode=forecast_mode, controller=controller, scheme_name=scheme_name)
    report_tables = build_madrl_battery_report(run_path, forecast_mode=forecast_mode, controller=controller, scheme_name=scheme_name)
    figures = _plot_agent_battery_power_soc_price(agent, run_path, scheme_name)
    ev_figures = _plot_agent_ev_charge_soc_price_night(agent, run_path, scheme_name, meta, config)
    action_trace, action_warning = (None, None)
    if include_action_trace:
        action_trace, action_warning = _try_action_projection_trace(run_path, forecast_mode, controller, scheme_name, meta)
    manifest = {
        "scheme_name": scheme_name,
        "forecast_mode": forecast_mode,
        "controller": controller,
        "stats_csv": str(run_path / "tables" / f"{scheme_name}_battery_soc_price_diagnostics.csv"),
        "figures": {key: str(value) for key, value in figures.items()},
        "ev_figures": {key: str(value) for key, value in ev_figures.items()},
        "action_trace_csv": str(run_path / "tables" / f"{scheme_name}_battery_action_projection_trace.csv") if action_trace is not None else None,
        "action_projection_summary_csv": str(run_path / "tables" / f"{scheme_name}_battery_action_projection_summary.csv") if action_trace is not None else None,
        "action_trace_warning": action_warning,
        "report_dir": str(run_path / "tables" / "battery_diagnostics"),
    }
    out = run_path / "tables" / f"{scheme_name}_battery_diagnostics_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stats": stats, "report_tables": report_tables, "figures": figures, "ev_figures": ev_figures, "action_trace": action_trace, "manifest": manifest}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MADRL battery/SOC/price diagnostics from an existing rollout record.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--forecast-mode", default="lstm")
    parser.add_argument("--controller", default="MADRL_BASE")
    parser.add_argument("--scheme-name", default="madrl_base")
    parser.add_argument("--plot-convergence", action="store_true")
    parser.add_argument("--no-action-trace", action="store_true")
    args = parser.parse_args()
    result = run_madrl_battery_diagnostics(args.run_dir, forecast_mode=args.forecast_mode, controller=args.controller, scheme_name=args.scheme_name, include_action_trace=not args.no_action_trace)
    print(result["stats"].to_string(index=False))
    if args.plot_convergence:
        convergence = plot_madrl_training_convergence(args.run_dir, scheme_name=args.scheme_name)
        print(convergence["summary"].to_string(index=False))
    if result["manifest"].get("action_trace_warning"):
        print(result["manifest"]["action_trace_warning"])
