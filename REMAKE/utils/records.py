from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from REMAKE.configs.cfg import Cfg
from REMAKE.data.share_data import ShareData
from REMAKE.envs.grid_env import build_env
from REMAKE.utils.price import IMPORT_PRICE_MARKUP_KEY, get_import_price_markup
from REMAKE.utils.run_artifacts import save_eval_result

RECORD_SCHEMA_VERSION = 2
WHOLESALE_PRICE_SIGNAL, WHOLESALE_PRICE_PRED_COLUMN = "wholesale_price", "wholesale_price_pred"
IMPORT_PRICE_COLUMN, IMPORT_PRICE_PRED_COLUMN = "import_price", "import_price_pred"
BATTERY_POWER_COLUMN = "battery_power_kw"
STORAGE_CHARGE_COST_COLUMN, STORAGE_DISCHARGE_REVENUE_COLUMN = "storage_charge_cost_eur", "storage_discharge_revenue_eur"
STORAGE_PROFIT_COLUMN, STORAGE_OBJECTIVE_COLUMN = "storage_profit_eur", "storage_objective_eur"
COMPARE_METRIC_COLUMNS = [
    "controller", "soc_mode", "storage_charge_cost_total_eur", "storage_discharge_revenue_total_eur",
    "storage_profit_total_eur", "storage_objective_total_eur", "system_other_cost_eur", "total_eur",
    "soc_penalty_total", "voltage_penalty_total", "trafo_penalty_total", "line_penalty_total",
    "voltage_violation_count", "price_mae", "load_mae", "pv_mae", "voltage_violation_steps",
    "voltage_violation_bus_points", "min_vm_pu", "max_vm_pu", "v_min_pu", "v_max_pu",
    "voltage_step_delta_p95_pu", "voltage_step_delta_max_pu", "voltage_spread_mean_pu",
    "voltage_spread_max_pu", "trafo_overload_steps", "trafo_loading_max_pct",
    "feeder_netload_ramp_mean_abs_kw", "feeder_netload_ramp_max_kw", "battery_net_power_kw_mean_abs",
    "returned_primary_objective_eur",
]
ECONOMIC_TABLE_COLUMNS = ["controller", "storage_discharge_revenue_total_eur", "storage_charge_cost_total_eur", "storage_profit_total_eur"]
SAFETY_TABLE_COLUMNS = [
    "controller", "voltage_violation_steps", "voltage_violation_bus_points", "voltage_step_delta_p95_pu",
    "voltage_step_delta_max_pu", "voltage_spread_mean_pu", "voltage_spread_max_pu", "trafo_overload_steps",
    "trafo_loading_max_pct", "feeder_netload_ramp_mean_abs_kw", "feeder_netload_ramp_max_kw",
]
_AGENT_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]


@dataclass(frozen=True)
class RolloutResult:
    step_df: pd.DataFrame
    agent_df: pd.DataFrame
    grid_df: pd.DataFrame
    summary: pd.DataFrame
    meta: dict[str, Any]


def record_dir(run_dir: str | Path, forecast_mode: str, controller: str) -> Path:
    return Path(run_dir) / "results" / str(forecast_mode) / str(controller) / "record"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _storage_step_columns(step_df: pd.DataFrame, dt_hours: float) -> pd.DataFrame:
    frame = step_df.copy()
    battery = frame[BATTERY_POWER_COLUMN].to_numpy(dtype=np.float64)
    price = frame[IMPORT_PRICE_COLUMN].to_numpy(dtype=np.float64)
    charge_cost = np.maximum(battery, 0.0) * float(dt_hours) * price
    discharge_revenue = np.maximum(-battery, 0.0) * float(dt_hours) * price
    frame[STORAGE_CHARGE_COST_COLUMN] = charge_cost
    frame[STORAGE_DISCHARGE_REVENUE_COLUMN] = discharge_revenue
    frame[STORAGE_PROFIT_COLUMN] = discharge_revenue - charge_cost
    frame[STORAGE_OBJECTIVE_COLUMN] = -frame[STORAGE_PROFIT_COLUMN]
    return frame


def collect_rollout(cfg: Cfg, controller, share_data: ShareData, *, forecast_mode: str, n_episodes: int | None = None, label: str | None = None) -> RolloutResult:
    env = build_env(cfg, "eval", forecast_mode=forecast_mode, share_data=share_data)
    total = int(share_data.eval["price"].shape[0])
    selected = list(range(total)) if cfg.eval.episode_indices is None else [int(value) for value in cfg.eval.episode_indices]
    selected = selected[: int(cfg.eval.n_episodes if n_episodes is None else n_episodes)]
    n, pmax = int(cfg.env.num_agents), np.asarray(cfg.env.battery_capacity_kwh, dtype=np.float32) * np.float32(cfg.env.max_charge_rate)
    markup, profiles = get_import_price_markup(cfg), tuple(cfg.data.agent_profiles)
    step_rows: list[dict[str, Any]] = []; agent_rows: list[dict[str, Any]] = []; grid_rows: list[dict[str, Any]] = []
    controller_key, controller_label = str(controller.name), str(label or controller.name)
    for episode_idx in tqdm(selected, desc=f"{controller_label}/{forecast_mode}", unit="episode", ascii=True):
        obs, state = env.reset(episode_index=episode_idx); controller.reset(state)
        for step in range(int(env.episode_length)):
            local = np.asarray(obs["local"], dtype=np.float32)
            seq = np.asarray(obs["sequence"], dtype=np.float32)
            if callable(controller):
                window = controller_window_from_obs(obs)
                raw_action, _, solve_meta = controller(env, window)
                action = np.asarray(raw_action, dtype=np.float32).reshape(n, int(cfg.model.action_dim))
            else:
                solve_meta = {}
                action = np.asarray(controller.act(obs), dtype=np.float32).reshape(n, int(cfg.model.action_dim))
            charge_kw = action[:, 0] * pmax
            pv_effective_req = np.clip(local[:, 2] * (np.float32(0.5) * (action[:, 1] + np.float32(1.0))), 0.0, local[:, 2])
            obs, reward, done, _, info = env.step(action)
            timestamp = pd.Timestamp(str(share_data.eval["timestamps"][episode_idx, step]))
            wholesale_price = float(share_data.eval["price"][episode_idx, step])
            wholesale_price_pred = float(seq[0, 0, 0])
            import_price, import_price_pred = wholesale_price + markup, wholesale_price_pred + markup
            load_total, pv_raw_total = float(np.sum(local[:, 1])), float(np.sum(local[:, 2]))
            pv_effective_total = float(np.sum(info["pv_effective"]))
            pv_curtail_total = float(np.sum(info["pv_curtail"]))
            charge_total, discharge_total = float(np.sum(np.maximum(charge_kw, 0.0))), float(np.sum(np.maximum(-charge_kw, 0.0)))
            raw_net = load_total - pv_raw_total
            effective_net = load_total - pv_effective_total
            post_net = float(np.sum(info["net_load"]))
            storage_profit = float(info["storage_profit_eur"])
            step_rows.append({
                "controller": controller_label, "controller_key": controller_key, "forecast_mode": forecast_mode,
                "prediction_mode": forecast_mode, "episode_idx": episode_idx, "step": step, "timestamp": timestamp,
                WHOLESALE_PRICE_SIGNAL: wholesale_price, WHOLESALE_PRICE_PRED_COLUMN: wholesale_price_pred,
                IMPORT_PRICE_COLUMN: import_price, IMPORT_PRICE_PRED_COLUMN: import_price_pred,
                BATTERY_POWER_COLUMN: float(np.sum(charge_kw)), "battery_charge_total": charge_total,
                "battery_discharge_total": discharge_total, "load_total": load_total, "pv_raw_total": pv_raw_total,
                "pv_effective_total": pv_effective_total, "pv_curtail_total": pv_curtail_total,
                "grid_import_total": float(max(post_net, 0.0)), "grid_export_total": float(max(-post_net, 0.0)),
                "feeder_raw_net_load_kw": raw_net, "feeder_effective_net_load_kw": effective_net,
                "feeder_post_action_net_load_kw": post_net, "root_net_exchange_kw": post_net, "pp_root_p_kw": post_net,
                STORAGE_CHARGE_COST_COLUMN: max(float(np.sum(charge_kw)), 0.0) * float(cfg.env.dt_hours) * import_price,
                STORAGE_DISCHARGE_REVENUE_COLUMN: max(-float(np.sum(charge_kw)), 0.0) * float(cfg.env.dt_hours) * import_price,
                STORAGE_PROFIT_COLUMN: storage_profit, STORAGE_OBJECTIVE_COLUMN: -storage_profit,
                "system_other_cost_eur": 0.0, "total_eur": -storage_profit, "reward_total": float(np.sum(reward)),
                "soc_penalty_total": float(np.sum(info["madrl_r_action_penalty"]) + np.sum(info["madrl_r_soc_regularization"])),
                "voltage_penalty_total": float(np.sum(info["madrl_r_safe_v"])), "line_penalty_total": float(np.sum(info["madrl_r_safe_line"])),
                "trafo_penalty_total": float(np.sum(info["madrl_r_safe_trafo"])), "voltage_violation_count": int(info["voltage_violation_count"]),
                "min_vm_pu": float(info["min_vm_pu"]), "max_vm_pu": float(info["max_vm_pu"]),
                "trafo_loading_pct_max": float(np.max(info["trafo_loading_pct"])), "n_trafo_violations": int(np.max(info["trafo_loading_pct"]) > 100.0),
                "solve_time_sec": float(solve_meta.get("solve_time_sec", 0.0)), "returned_primary_objective_eur": -storage_profit,
            })
            for agent_id in range(n):
                profile = profiles[agent_id] if agent_id < len(profiles) else f"agent_{agent_id}"
                line_values = np.asarray(info["line_loading_pct"], dtype=np.float32)
                line_value = float(line_values[agent_id]) if line_values.size == n else float(np.max(line_values)) if line_values.size else 0.0
                agent_rows.append({
                    "controller": controller_label, "forecast_mode": forecast_mode, "episode_idx": episode_idx, "step": step,
                    "timestamp": timestamp, "agent_id": agent_id, "agent_profile": profile, "reward": float(np.asarray(reward)[agent_id]),
                    "soc": float(info["soc"][agent_id]), "load": float(local[agent_id, 1]), "load_pred": float(seq[agent_id, 0, 1]),
                    "pv": float(local[agent_id, 2]), "pv_pred": float(seq[agent_id, 0, 2]), "e_bat": float(charge_kw[agent_id]),
                    "pv_effective_kw": float(info["pv_effective"][agent_id]), "pv_curtail_kw": float(info["pv_curtail"][agent_id]),
                    "net_load_kw": float(info["net_load"][agent_id]), "storage_profit_eur": float(info["madrl_r_inc"][agent_id]),
                })
                grid_rows.append({
                    "controller": controller_label, "forecast_mode": forecast_mode, "episode_idx": episode_idx, "step": step,
                    "timestamp": timestamp, "bus_id": int(cfg.grid.agent_bus_ids[agent_id]), "is_agent_bus": True,
                    "vm_pu": float(info["vm_pu"][agent_id]), "line_loading_pct": line_value,
                })
            if done:
                break
    env.close()
    step_df = _storage_step_columns(pd.DataFrame(step_rows), float(cfg.env.dt_hours))
    agent_df, grid_df = pd.DataFrame(agent_rows), pd.DataFrame(grid_rows)
    summary = agent_df.groupby(["controller", "agent_id", "agent_profile"], as_index=False).agg(reward=("reward", "sum"), storage_profit_eur=("storage_profit_eur", "sum"), mean_soc=("soc", "mean"))
    meta = {
        "controller": controller_label, "controller_key": controller_key, "forecast_mode": forecast_mode, "prediction_mode": forecast_mode,
        "cfg_hash": cfg.hash8(), "eval_episode_indices": selected, "agent_bus_ids": tuple(int(x) for x in cfg.grid.agent_bus_ids),
        "v_min_pu": 0.95, "v_max_pu": 1.05, "trafo_limit_kw": float(env.grid_core.trafo_limit_kw),
        "trafo_loading_limit_pct": 100.0, "loading_limit_pct": 100.0, "dt_hours": float(cfg.env.dt_hours),
        "soc_mode": "continuous", "objective_mode": "max_storage_profit", IMPORT_PRICE_MARKUP_KEY: markup,
    }
    return RolloutResult(step_df=step_df, agent_df=agent_df, grid_df=grid_df, summary=summary, meta=meta)


def controller_window_from_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    local = np.asarray(obs["local"], dtype=np.float32)
    seq = np.asarray(obs["sequence"], dtype=np.float32)
    load_seq = seq[:, :, 1].copy(); pv_seq = seq[:, :, 2].copy()
    load_seq[:, 0] = local[:, 1]; pv_seq[:, 0] = local[:, 2]
    return {"price_seq": seq[0, :, 0].copy(), "load_seq": load_seq, "pv_seq": pv_seq}


def _voltage_step_delta(grid_df: pd.DataFrame) -> tuple[float, float]:
    ordered = grid_df.sort_values([c for c in ("episode_idx", "bus_id", "step", "timestamp") if c in grid_df.columns])
    values = ordered.groupby(["episode_idx", "bus_id"], sort=False)["vm_pu"].diff().abs().to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 95.0)), float(np.max(values))) if values.size else (float("nan"), float("nan"))


def _voltage_spread(grid_df: pd.DataFrame) -> tuple[float, float]:
    values = grid_df.groupby(["episode_idx", "step"], sort=False)["vm_pu"].agg(lambda x: float(np.max(x) - np.min(x))).to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return (float(np.mean(values)), float(np.max(values))) if values.size else (float("nan"), float("nan"))


def _ramp(step_df: pd.DataFrame, column: str) -> tuple[float, float]:
    ordered = step_df.sort_values([c for c in ("episode_idx", "step", "timestamp") if c in step_df.columns])
    values = ordered.groupby("episode_idx", sort=False)[column].diff().abs().to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return (float(np.mean(values)), float(np.max(values))) if values.size else (float("nan"), float("nan"))


def metrics_from_rollout(rollout: RolloutResult) -> pd.DataFrame:
    step, agent, grid = rollout.step_df.copy(), rollout.agent_df.copy(), rollout.grid_df.copy()
    storage = _storage_step_columns(step, float(rollout.meta["dt_hours"]))
    v_min, v_max = float(rollout.meta["v_min_pu"]), float(rollout.meta["v_max_pu"])
    viol = (grid["vm_pu"] < v_min) | (grid["vm_pu"] > v_max)
    voltage_step_delta_p95, voltage_step_delta_max = _voltage_step_delta(grid)
    voltage_spread_mean, voltage_spread_max = _voltage_spread(grid)
    ramp_mean, ramp_max = _ramp(step, "feeder_post_action_net_load_kw")
    row = {
        "controller": str(rollout.meta["controller"]), "soc_mode": str(rollout.meta["soc_mode"]),
        "storage_charge_cost_total_eur": float(storage[STORAGE_CHARGE_COST_COLUMN].sum()),
        "storage_discharge_revenue_total_eur": float(storage[STORAGE_DISCHARGE_REVENUE_COLUMN].sum()),
        "storage_profit_total_eur": float(storage[STORAGE_PROFIT_COLUMN].sum()),
        "storage_objective_total_eur": float(storage[STORAGE_OBJECTIVE_COLUMN].sum()),
        "system_other_cost_eur": float(step["system_other_cost_eur"].sum()), "total_eur": float(step["total_eur"].sum()),
        "soc_penalty_total": float(step["soc_penalty_total"].sum()), "voltage_penalty_total": float(step["voltage_penalty_total"].sum()),
        "trafo_penalty_total": float(step["trafo_penalty_total"].sum()), "line_penalty_total": float(step["line_penalty_total"].sum()),
        "voltage_violation_count": int(viol.sum()), "price_mae": float(np.abs(step[IMPORT_PRICE_COLUMN] - step[IMPORT_PRICE_PRED_COLUMN]).mean()),
        "load_mae": float(np.abs(agent["load"] - agent["load_pred"]).mean()), "pv_mae": float(np.abs(agent["pv"] - agent["pv_pred"]).mean()),
        "voltage_violation_steps": int(grid.loc[viol, ["episode_idx", "step"]].drop_duplicates().shape[0]),
        "voltage_violation_bus_points": int(viol.sum()), "min_vm_pu": float(grid["vm_pu"].min()), "max_vm_pu": float(grid["vm_pu"].max()),
        "v_min_pu": v_min, "v_max_pu": v_max, "voltage_step_delta_p95_pu": voltage_step_delta_p95,
        "voltage_step_delta_max_pu": voltage_step_delta_max, "voltage_spread_mean_pu": voltage_spread_mean,
        "voltage_spread_max_pu": voltage_spread_max, "trafo_overload_steps": int((step["n_trafo_violations"].astype(float) > 0.0).sum()),
        "trafo_loading_max_pct": float(step["trafo_loading_pct_max"].max()), "feeder_netload_ramp_mean_abs_kw": ramp_mean,
        "feeder_netload_ramp_max_kw": ramp_max, "battery_net_power_kw_mean_abs": float(np.abs(step[BATTERY_POWER_COLUMN]).mean()),
        "returned_primary_objective_eur": float(step["returned_primary_objective_eur"].sum()),
    }
    return pd.DataFrame([row]).loc[:, COMPARE_METRIC_COLUMNS]


def compare_rollout_metrics(*rollouts: RolloutResult) -> pd.DataFrame:
    return pd.concat([metrics_from_rollout(rollout) for rollout in rollouts], ignore_index=True) if rollouts else pd.DataFrame(columns=COMPARE_METRIC_COLUMNS)


def build_compare_economic_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.loc[:, ECONOMIC_TABLE_COLUMNS].copy().reset_index(drop=True)


def build_compare_safety_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.loc[:, SAFETY_TABLE_COLUMNS].copy().reset_index(drop=True)


def price_forecast_diagnostics(*rollouts: RolloutResult) -> pd.DataFrame:
    rows = []
    for rollout in rollouts:
        step = rollout.step_df
        diff = (step[IMPORT_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64) - step[IMPORT_PRICE_COLUMN].to_numpy(dtype=np.float64))
        rows.append({
            "controller": str(rollout.meta["controller"]),
            "forecast_mode": str(rollout.meta["forecast_mode"]),
            "price_mae": float(np.mean(np.abs(diff))),
            "price_max_abs_error": float(np.max(np.abs(diff))),
            "prediction_plotted": bool(str(rollout.meta["forecast_mode"]) != "perfect" and float(np.max(np.abs(diff))) > 1e-8),
        })
    return pd.DataFrame(rows)


def eval_result_from_rollout(cfg: Cfg, rollout: RolloutResult) -> dict[str, Any]:
    metrics = metrics_from_rollout(rollout).iloc[0].to_dict()
    step = rollout.step_df.sort_values(["episode_idx", "step"])
    voltage = rollout.grid_df.pivot_table(index=["episode_idx", "step"], columns="bus_id", values="vm_pu").to_numpy(dtype=np.float32)
    return {**metrics, "controller": rollout.meta["controller_key"], "controller_label": rollout.meta["controller"], "forecast_mode": rollout.meta["forecast_mode"], "cfg_hash": cfg.hash8(), "n_episodes": len(rollout.meta["eval_episode_indices"]), "eval_episode_indices": rollout.meta["eval_episode_indices"], "episode_reward_mean": float(step["reward_total"].sum()) / max(1, len(rollout.meta["eval_episode_indices"])), "storage_profit_eur_mean": float(metrics["storage_profit_total_eur"]) / max(1, len(rollout.meta["eval_episode_indices"])), "voltage_violation_steps_mean": float(metrics["voltage_violation_steps"]) / max(1, len(rollout.meta["eval_episode_indices"])), "act_time_s_mean": float(step["solve_time_sec"].mean()), "voltage_trace": voltage, "trafo_trace": step["trafo_loading_pct_max"].to_numpy(dtype=np.float32)}


def save_rollout(cfg: Cfg, run_dir: str | Path, rollout: RolloutResult, *, scheme_name: str) -> dict[str, Any]:
    out = record_dir(run_dir, rollout.meta["forecast_mode"], rollout.meta["controller_key"]); out.mkdir(parents=True, exist_ok=True)
    metrics_df = metrics_from_rollout(rollout)
    for name, frame in {"step": rollout.step_df, "agent": rollout.agent_df, "grid": rollout.grid_df, "summary": rollout.summary, "metrics": metrics_df}.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(rollout.meta, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    manifest = {"schema_version": RECORD_SCHEMA_VERSION, "scheme_name": scheme_name, "record_dir": str(out), "files": {"step_df": "step.parquet", "agent_df": "agent.parquet", "grid_df": "grid.parquet", "summary_df": "summary.parquet", "metrics_df": "metrics.parquet", "meta": "meta.json"}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_eval_result(Path(run_dir), eval_result_from_rollout(cfg, rollout))
    return {"rollout": rollout, "metrics_df": metrics_df, "record_dir": out, "manifest": manifest}


def load_record(run_dir: str | Path, forecast_mode: str, controller: str) -> dict[str, Any]:
    out = record_dir(run_dir, forecast_mode, controller)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((out / manifest["files"]["meta"]).read_text(encoding="utf-8"))
    return {"rollout": RolloutResult(step_df=pd.read_parquet(out / manifest["files"]["step_df"]), agent_df=pd.read_parquet(out / manifest["files"]["agent_df"]), grid_df=pd.read_parquet(out / manifest["files"]["grid_df"]), summary=pd.read_parquet(out / manifest["files"]["summary_df"]), meta=meta), "metrics_df": pd.read_parquet(out / manifest["files"]["metrics_df"]), "record_dir": out, "manifest": manifest}


def _style(axis, title: str, ylabel: str, xlabel: str | None = None) -> None:
    axis.set_title(title, fontsize=16); axis.set_ylabel(ylabel, fontsize=12); axis.grid(True, alpha=0.25)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=12)


def _panel(count: int, height: float = 3.6, sharex: bool = True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(count, 1, figsize=(18, max(height * count, height + 1.4)), sharex=sharex)
    return fig, np.atleast_1d(axes)


def plot_price_prediction_comparison(*rollouts: RolloutResult):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(18, 7.0), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.2]})
    ax, err_ax = axes
    ref = rollouts[0].step_df
    ax.plot(ref["timestamp"], ref[IMPORT_PRICE_COLUMN], color="#111827", linewidth=1.8, label="Actual import price")
    plotted = 0
    plotted_predictions: list[np.ndarray] = []
    for idx, rollout in enumerate(rollouts):
        step = rollout.step_df
        diff = step[IMPORT_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64) - step[IMPORT_PRICE_COLUMN].to_numpy(dtype=np.float64)
        if str(rollout.meta["forecast_mode"]) == "perfect" or float(np.max(np.abs(diff))) <= 1e-8:
            continue
        pred = step[IMPORT_PRICE_PRED_COLUMN].to_numpy(dtype=np.float64)
        if any(np.allclose(pred, seen, rtol=0.0, atol=1e-9) for seen in plotted_predictions):
            continue
        plotted_predictions.append(pred)
        color = _AGENT_COLORS[(idx + 1) % len(_AGENT_COLORS)]
        label = "Predicted import price (LSTM)" if str(rollout.meta["forecast_mode"]) == "lstm" else f"Predicted import price ({rollout.meta['controller']})"
        ax.plot(step["timestamp"], pred, color=color, linestyle="--", linewidth=1.5, label=label)
        err_ax.plot(step["timestamp"], diff, color=color, linewidth=1.2, label=str(rollout.meta["controller"]))
        plotted += 1
    if plotted == 0:
        err_ax.axhline(0.0, color="#64748b", linewidth=1.0)
        err_ax.text(0.01, 0.72, "perfect forecast equals actual price", transform=err_ax.transAxes, color="#475569")
    _style(ax, "Import Price And Derived Forecast", "EUR/kWh")
    _style(err_ax, "Forecast Error", "pred-actual", "Timestamp")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    if plotted:
        err_ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout(); return fig


def plot_voltage_profile_comparison(*rollouts: RolloutResult):
    fig, axes = _panel(len(rollouts), 3.5)
    for idx, (ax, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        for bus_idx, bus_id in enumerate(rollout.meta["agent_bus_ids"]):
            frame = rollout.grid_df.loc[rollout.grid_df["bus_id"] == int(bus_id)]
            ax.plot(frame["timestamp"], frame["vm_pu"], color=_AGENT_COLORS[bus_idx % len(_AGENT_COLORS)], linewidth=1.8, label=f"Agent bus {bus_id}" if idx == 0 else None)
        ax.axhline(float(rollout.meta["v_min_pu"]), color="#dc2626", linestyle="--", linewidth=1.0, label="V min" if idx == 0 else None)
        ax.axhline(float(rollout.meta["v_max_pu"]), color="#ea580c", linestyle="--", linewidth=1.0, label="V max" if idx == 0 else None)
        _style(ax, str(rollout.meta["controller"]), "V [p.u.]")
        if idx == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def plot_net_load_comparison(*rollouts: RolloutResult):
    fig, axes = _panel(len(rollouts), 3.8)
    for idx, (ax, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step = rollout.step_df
        for column, color, linestyle, label in [("feeder_raw_net_load_kw", "#111827", "-", "Feeder raw net load"), ("feeder_effective_net_load_kw", "#16a34a", "-.", "Feeder post-curtail net load"), ("feeder_post_action_net_load_kw", "#2563eb", "--", "Feeder post-action net load"), ("root_net_exchange_kw", "#7c3aed", "-", "Root net exchange")]:
            ax.plot(step["timestamp"], step[column], color=color, linestyle=linestyle, linewidth=1.5, label=label if idx == 0 else None)
        limit = float(rollout.meta["trafo_limit_kw"])
        ax.axhline(limit, color="#dc2626", linestyle=":", linewidth=1.1, label="Transformer S-limit ref (+P view)" if idx == 0 else None)
        ax.axhline(-limit, color="#dc2626", linestyle=":", linewidth=1.1, label="Transformer S-limit ref (-P view)" if idx == 0 else None)
        _style(ax, f"{rollout.meta['controller']} - feeder total", "kW")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def plot_power_balance_comparison(*rollouts: RolloutResult):
    fig, axes = _panel(len(rollouts), 3.6)
    pos_specs = [("load_total", "Load", "#111827"), ("battery_charge_total", "Charge", "#dc2626"), ("grid_export_total", "Grid export", "#f59e0b"), ("pv_curtail_total", "Curtailment loss", "#fca5a5")]
    neg_specs = [("pv_raw_total", "PV raw", "#16a34a"), ("grid_import_total", "Grid import", "#2563eb"), ("battery_discharge_total", "Discharge", "#7c3aed")]
    for idx, (ax, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step = rollout.step_df; residual = np.zeros(len(step), dtype=np.float32)
        residual += sum((step[col].to_numpy(dtype=np.float32) for col, *_ in pos_specs), start=np.zeros(len(step), dtype=np.float32))
        residual -= sum((step[col].to_numpy(dtype=np.float32) for col, *_ in neg_specs), start=np.zeros(len(step), dtype=np.float32))
        for specs, sign in ((pos_specs, 1.0), (neg_specs, -1.0)):
            bottom = np.zeros(len(step), dtype=np.float32)
            for col, label, color in specs:
                values = sign * step[col].to_numpy(dtype=np.float32)
                ax.bar(step["timestamp"], values, width=0.008, bottom=bottom, color=color, alpha=0.78, label=label if idx == 0 else None)
                bottom += values
        ax.axhline(0.0, color="#111827", linewidth=0.9); _style(ax, f"{rollout.meta['controller']} - agent-only balance (residual<={float(np.max(np.abs(residual))):.3f} kW)", "kW")
        if idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def plot_battery_power_and_soc_comparison(*rollouts: RolloutResult):
    fig, axes = _panel(len(rollouts), 3.8)
    for idx, (ax, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step, agent = rollout.step_df, rollout.agent_df
        net_power = step["battery_discharge_total"].to_numpy(dtype=np.float32) - step["battery_charge_total"].to_numpy(dtype=np.float32)
        ax.bar(step["timestamp"], net_power, width=0.008, color=["#dc2626" if value >= 0.0 else "#2563eb" for value in net_power], alpha=0.82)
        ax.axhline(0.0, color="#475569", linewidth=0.9); _style(ax, str(rollout.meta["controller"]), "Battery Power [kW]")
        soc_ax = ax.twinx()
        summary = agent.groupby(["episode_idx", "step", "timestamp"], as_index=False).agg(mean=("soc", "mean"))
        for _, frame in agent.sort_values(["episode_idx", "agent_id", "step"]).groupby("agent_id", sort=True):
            soc_ax.plot(frame["timestamp"], frame["soc"], color="#94a3b8", linewidth=1.0, alpha=0.45)
        soc_ax.plot(summary["timestamp"], summary["mean"], color="#111827", linewidth=1.5, label="Mean SoC")
        soc_ax.set_ylabel("SoC"); soc_ax.set_ylim(0.0, 1.0)
        if idx == 0:
            ax.legend(handles=[], labels=[], loc="upper right")
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def plot_aligned_records(records: dict[str, RolloutResult], run_dir: str | Path, prefix: str) -> dict[str, Any]:
    out = Path(run_dir) / "figures"; out.mkdir(parents=True, exist_ok=True)
    rollouts = tuple(records.values())
    table_dir = Path(run_dir) / "tables"; table_dir.mkdir(parents=True, exist_ok=True)
    price_forecast_diagnostics(*rollouts).to_csv(table_dir / f"{prefix}_price_forecast_diagnostics.csv", index=False)
    specs = {
        f"{prefix}_power_balance": plot_power_balance_comparison(*rollouts),
        f"{prefix}_price": plot_price_prediction_comparison(*rollouts),
        f"{prefix}_battery_soc": plot_battery_power_and_soc_comparison(*rollouts),
        f"{prefix}_voltage": plot_voltage_profile_comparison(*rollouts),
        f"{prefix}_net_load": plot_net_load_comparison(*rollouts),
    }
    for name, fig in specs.items():
        fig.savefig(out / f"{name}.png", dpi=140)
    return specs


def plot_basic_records(records: dict[str, RolloutResult], run_dir: str | Path, prefix: str) -> dict[str, Any]:
    return plot_aligned_records(records, run_dir, prefix)
