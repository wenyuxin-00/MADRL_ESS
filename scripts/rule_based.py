from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from configs.cfg import Cfg
from controllers.rule_based import RuleBasedEVBatteryController, RuleBasedEVNoBatteryController
from data.share_data import ShareData
from utils.records import (
    RolloutResult,
    compute_ev_agent_cost_summary,
    compute_ev_cost_summary,
    compute_ev_session_cost_summary,
    compute_ev_session_summary,
    collect_rollout,
    plot_aligned_records,
    save_rollout,
)

RULE_BASED_NO_BATTERY = "rule_based_ev_no_battery"
RULE_BASED_WITH_BATTERY = "rule_based_ev_with_battery"
RULE_BASED_NO_BATTERY_LABEL = "Rule-Based EV without Battery"
RULE_BASED_WITH_BATTERY_LABEL = "Rule-Based EV with Battery"
_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]


def _enrich_rollout(rollout: RolloutResult, controller: Any) -> RolloutResult:
    step = rollout.step_df.copy()
    agent = rollout.agent_df.copy()
    meta = dict(rollout.meta)
    metadata = {
        "controller_type": str(controller.controller_type),
        "battery_rule": str(controller.battery_rule),
        "low_price_threshold": controller.low_price_threshold,
        "high_price_threshold": controller.high_price_threshold,
    }
    meta.update(metadata)
    for key, value in metadata.items():
        step[key] = value
        agent[key] = value
    if "battery_power_kw" not in agent.columns and "e_bat" in agent.columns:
        agent["battery_power_kw"] = agent["e_bat"].astype(float)
    if "battery_soc" not in agent.columns and "soc" in agent.columns:
        agent["battery_soc"] = agent["soc"].astype(float)
    if "net_load" not in agent.columns and "net_load_kw" in agent.columns:
        agent["net_load"] = agent["net_load_kw"].astype(float)
    if "price" not in step.columns and "import_price" in step.columns:
        step["price"] = step["import_price"].astype(float)
    if "price" not in agent.columns and {"episode_idx", "step", "import_price"}.issubset(step.columns):
        price = step[["episode_idx", "step", "import_price"]].drop_duplicates(["episode_idx", "step"])
        agent = agent.merge(price.rename(columns={"import_price": "price"}), on=["episode_idx", "step"], how="left")
    return RolloutResult(step_df=step, agent_df=agent, grid_df=rollout.grid_df.copy(), summary=rollout.summary.copy(), meta=meta)


def _save_tables(cfg: Cfg, run_dir: str | Path, scheme_name: str, rollout: RolloutResult, metrics_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    table_dir = Path(run_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    cost_ts, total_cost_summary = compute_ev_cost_summary(rollout, cfg)
    agent_cost_summary = compute_ev_agent_cost_summary(rollout, cfg)
    ev_departure_summary = compute_ev_session_summary(rollout, cfg)
    ev_session_cost_summary, ev_session_cost_totals = compute_ev_session_cost_summary(rollout, cfg)
    frames = {
        "cost_timeseries": cost_ts,
        "total_cost_summary": total_cost_summary,
        "agent_cost_summary": agent_cost_summary,
        "ev_departure_summary": ev_departure_summary,
        "ev_session_cost_summary": ev_session_cost_summary,
        "ev_session_cost_totals": ev_session_cost_totals,
        "agent_summary": rollout.summary,
        "metrics": metrics_df,
    }
    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(table_dir / f"{scheme_name}_{name}.csv", index=False)
    return frames


def _pre_ev_soc(agent: pd.DataFrame, cfg: Cfg) -> pd.Series:
    caps = np.asarray(cfg.env.ev_capacity_kwh, dtype=np.float64).reshape(-1)
    cap = agent["agent_id"].astype(int).map(lambda idx: float(caps[idx]) if idx < caps.size else float(caps[-1]))
    executed = agent.get("ev_charge_kw_executed", agent.get("ev_charge_kw", pd.Series(0.0, index=agent.index))).astype(float)
    delta = executed * float(cfg.env.dt_hours) * float(cfg.env.ev_efficiency) / cap.replace(0.0, np.nan)
    return agent["ev_soc"].astype(float) - delta.fillna(0.0)


def _sanity_checks(cfg: Cfg, rollout: RolloutResult, *, with_battery: bool) -> pd.DataFrame:
    tol = 1e-5
    agent = rollout.agent_df.copy().sort_values(["episode_idx", "agent_id", "step"]).reset_index(drop=True)
    step = rollout.step_df.copy()
    cost_ts, total_cost = compute_ev_cost_summary(rollout, cfg)
    pre_ev_soc = _pre_ev_soc(agent, cfg) if not agent.empty and "ev_soc" in agent.columns else pd.Series([], dtype=float)
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "detail": detail})

    battery_power = agent["battery_power_kw"].astype(float) if "battery_power_kw" in agent.columns else pd.Series(0.0, index=agent.index)
    battery_cost = float(total_cost["battery_total_cost_eur"].iloc[0]) if not total_cost.empty else 0.0
    if not with_battery:
        add("no_battery_power_zero", bool(np.all(np.abs(battery_power.to_numpy()) <= tol)), f"max_abs={float(np.max(np.abs(battery_power.to_numpy()))) if len(battery_power) else 0.0:.6g}")
        add("no_battery_cost_zero", abs(battery_cost) <= tol, f"battery_total_cost_eur={battery_cost:.6g}")
    unavailable = agent["ev_available"].astype(float) <= 0.0
    ev_charge = agent["ev_charge_kw"].astype(float) if "ev_charge_kw" in agent.columns else pd.Series(0.0, index=agent.index)
    add("ev_unavailable_charge_zero", bool(np.all(ev_charge.loc[unavailable].abs().to_numpy() <= tol)), f"violations={int((ev_charge.loc[unavailable].abs() > tol).sum())}")
    if len(pre_ev_soc):
        available_not_full = (agent["ev_available"].astype(float) > 0.0) & (pre_ev_soc < float(cfg.env.ev_soc_max) - 1e-4)
        requested = agent.get("ev_charge_kw_requested", ev_charge).astype(float)
        add("ev_available_not_full_requests_charge", bool(np.all(requested.loc[available_not_full].to_numpy() > tol)), f"violations={int((requested.loc[available_not_full] <= tol).sum())}")
        daily_step = agent["step"].astype(int) % int(cfg.env.episode_steps)
        prev_full = (agent.groupby("agent_id")["ev_soc"].shift(1).astype(float) >= float(cfg.env.ev_soc_max) - 1e-4) & (agent["ev_available"].astype(float) > 0.0) & (daily_step != int(cfg.env.ev_arrival_step))
        executed = agent.get("ev_charge_kw_executed", ev_charge).astype(float)
        add("ev_full_next_charge_zero", bool(np.all(executed.loc[prev_full].abs().to_numpy() <= tol)), f"violations={int((executed.loc[prev_full].abs() > tol).sum())}")
    if with_battery:
        low = float(rollout.meta["low_price_threshold"])
        high = float(rollout.meta["high_price_threshold"])
        price = agent["price"].astype(float)
        add("battery_low_price_nonnegative", bool(np.all(battery_power.loc[price <= low + tol].to_numpy() >= -tol)), f"violations={int((battery_power.loc[price <= low + tol] < -tol).sum())}")
        add("battery_high_price_nonpositive", bool(np.all(battery_power.loc[price >= high - tol].to_numpy() <= tol)), f"violations={int((battery_power.loc[price >= high - tol] > tol).sum())}")
    soc = agent["battery_soc"].astype(float) if "battery_soc" in agent.columns else agent["soc"].astype(float)
    add("battery_soc_bounds", bool(np.all((soc >= float(cfg.env.soc_min) - tol) & (soc <= float(cfg.env.soc_max) + tol))), f"min={float(soc.min()) if len(soc) else np.nan:.6g}, max={float(soc.max()) if len(soc) else np.nan:.6g}")
    ev_soc = agent["ev_soc"].astype(float) if "ev_soc" in agent.columns else pd.Series(dtype=float)
    add("ev_soc_bounds", bool(np.all((ev_soc >= float(cfg.env.ev_soc_min) - tol) & (ev_soc <= float(cfg.env.ev_soc_max) + tol))), f"min={float(ev_soc.min()) if len(ev_soc) else np.nan:.6g}, max={float(ev_soc.max()) if len(ev_soc) else np.nan:.6g}")
    if len(pre_ev_soc):
        first = agent.loc[agent["step"].astype(int) == 0, ["episode_idx", "agent_id"]].copy()
        if not first.empty:
            first["pre_ev_soc"] = pre_ev_soc.loc[first.index].to_numpy()
            last = agent.sort_values(["episode_idx", "agent_id", "step"]).groupby(["episode_idx", "agent_id"], as_index=False).tail(1)
            prev = last[["episode_idx", "agent_id", "ev_soc"]].rename(columns={"episode_idx": "prev_episode_idx", "ev_soc": "prev_final_ev_soc"})
            first["prev_episode_idx"] = first["episode_idx"].astype(int) - 1
            continuity = first.merge(prev, on=["prev_episode_idx", "agent_id"], how="inner")
            gap = np.abs(continuity["pre_ev_soc"].astype(float) - continuity["prev_final_ev_soc"].astype(float)) if not continuity.empty else pd.Series(dtype=float)
            add("ev_soc_calendar_episode_continuity", bool(gap.empty or np.all(gap.to_numpy() <= 1e-4)), f"max_gap={float(gap.max()) if len(gap) else 0.0:.6g}")
    frame = pd.DataFrame(rows)
    if not bool(frame["passed"].all()):
        failed = ", ".join(frame.loc[~frame["passed"], "check"].astype(str).tolist())
        raise AssertionError(f"Rule-based sanity checks failed: {failed}")
    return frame


def _plot_ev_soc(rollouts: dict[str, RolloutResult], run_dir: str | Path) -> Any:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.8)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        for agent_id, frame in rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"]).groupby("agent_id", sort=True):
            ax.plot(frame["timestamp"], frame["ev_soc"], color=_COLORS[int(agent_id) % len(_COLORS)], linewidth=1.6, label=f"Agent {agent_id}" if idx == 0 else None)
        ax.axhline(float(rollout.meta.get("ev_departure_soc_req", 0.90)), color="#dc2626", linestyle="--", linewidth=1.1, label="Departure req" if idx == 0 else None)
        ax.set_title(f"{rollout.meta['controller']} - EV SOC"); ax.set_ylabel("EV SOC"); ax.grid(True, alpha=0.25); ax.set_ylim(0.0, 1.0)
        if idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_agent_power(rollouts: dict[str, RolloutResult], column: str, ylabel: str, title: str, *, price_axis: bool = False) -> Any:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(4.0 * len(rollouts), 5.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        for agent_id, frame in rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"]).groupby("agent_id", sort=True):
            ax.plot(frame["timestamp"], frame[column].astype(float), color=_COLORS[int(agent_id) % len(_COLORS)], linewidth=1.5, label=f"Agent {agent_id}" if idx == 0 else None)
        ax.axhline(0.0, color="#64748b", linewidth=0.9); ax.set_title(f"{rollout.meta['controller']} - {title}"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25)
        if price_axis:
            price_ax = ax.twinx()
            price = rollout.step_df.drop_duplicates(["episode_idx", "step"])
            price_ax.plot(price["timestamp"], price["price"], color="#111827", linestyle="--", linewidth=1.2, alpha=0.82, label="Price" if idx == 0 else None)
            price_ax.set_ylabel("EUR/kWh")
            if idx == 0:
                lines, labels = ax.get_legend_handles_labels()
                price_lines, price_labels = price_ax.get_legend_handles_labels()
                ax.legend(lines + price_lines, labels + price_labels, loc="upper right", ncol=4, fontsize=9)
        elif idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_departure_check(rollouts: dict[str, RolloutResult]) -> Any:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(10, max(3.4 * len(rollouts), 4.0)))
    axes = np.atleast_1d(axes)
    for (_, rollout), ax in zip(rollouts.items(), axes, strict=False):
        departure = int(rollout.meta.get("ev_departure_step", 28))
        frame = rollout.agent_df.loc[rollout.agent_df["step"].astype(int) == departure]
        summary = frame.groupby("agent_id", as_index=False).agg(ev_departure_gap=("ev_departure_gap", "max"), ev_soc=("ev_soc", "min"))
        ax.bar(summary["agent_id"].astype(str), summary["ev_departure_gap"].astype(float), color=["#16a34a" if gap <= 1e-6 else "#dc2626" for gap in summary["ev_departure_gap"]])
        ax.set_title(f"{rollout.meta['controller']} - EV departure check"); ax.set_xlabel("Agent"); ax.set_ylabel("SOC gap"); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); return fig


def _plot_cost(rollouts: dict[str, RolloutResult], cfg: Cfg, *, components: bool, cumulative: bool) -> Any:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.8)), sharex=True)
    axes = np.atleast_1d(axes)
    specs = [("battery_cost_eur", "Battery", "#2563eb"), ("ev_cost_eur", "EV", "#0891b2"), ("pv_cost_eur", "PV", "#16a34a"), ("total_cost_eur", "Total", "#111827")]
    for idx, ((_, rollout), ax) in enumerate(zip(rollouts.items(), axes, strict=False)):
        cost_ts, total = compute_ev_cost_summary(rollout, cfg)
        x = cost_ts["timestamp"] if "timestamp" in cost_ts.columns else cost_ts["step"]
        columns = specs if components else [specs[-1]]
        for column, label, color in columns:
            values = cost_ts[column].fillna(0.0).to_numpy(dtype=np.float64)
            values = np.cumsum(values) if cumulative else values
            ax.plot(x, values, color=color, linewidth=1.7 if column == "total_cost_eur" else 1.25, linestyle="-" if column == "total_cost_eur" else "--", label=label if idx == 0 else None)
        final = float(total["total_cost_eur"].iloc[0]) if not total.empty else 0.0
        ax.set_title(f"{rollout.meta['controller']} - total cost {final:.3f} EUR"); ax.set_ylabel("EUR"); ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper left", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def plot_rule_based_outputs(records: dict[str, Any], run_dir: str | Path, cfg: Cfg) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    rollouts = {name: saved["rollout"] for name, saved in records.items()}
    figures = plot_aligned_records(rollouts, run_dir, "rule_based")
    extra = {
        "rule_based_ev_soc": _plot_ev_soc(rollouts, run_dir),
        "rule_based_ev_charge_kw": _plot_agent_power(rollouts, "ev_charge_kw", "EV charging power [kW]", "EV charging power"),
        "rule_based_ev_departure_check": _plot_departure_check(rollouts),
        "rule_based_total_cost": _plot_cost(rollouts, cfg, components=False, cumulative=True),
        "rule_based_cost_components": _plot_cost(rollouts, cfg, components=True, cumulative=False),
        "rule_based_cumulative_cost_components": _plot_cost(rollouts, cfg, components=True, cumulative=True),
        "rule_based_battery_power_price": _plot_agent_power(rollouts, "battery_power_kw", "Battery power [kW]", "Battery power and price", price_axis=True),
        "rule_based_ev_charge_price": _plot_agent_power(rollouts, "ev_charge_kw", "EV charging power [kW]", "EV charging power and price", price_axis=True),
    }
    out = Path(run_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in extra.items():
        fig.savefig(out / f"{name}.png", dpi=140)
        figures[name] = fig
    return figures


def _run_rule_based(cfg: Cfg, run_dir: str | Path, share_data: ShareData, controller: Any, *, with_battery: bool) -> dict[str, Any]:
    rollout = collect_rollout(cfg, controller, share_data, forecast_mode="perfect", label=controller.display_name)
    rollout = _enrich_rollout(rollout, controller)
    sanity = _sanity_checks(cfg, rollout, with_battery=with_battery)
    saved = save_rollout(cfg, run_dir, rollout, scheme_name=controller.name)
    saved["rollout"] = rollout
    saved["cost_tables"] = _save_tables(cfg, run_dir, controller.name, rollout, saved["metrics_df"])
    saved["sanity_checks"] = sanity
    table_dir = Path(run_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    sanity.to_csv(table_dir / f"{controller.name}_sanity_checks.csv", index=False)
    return saved


def run_rule_based_ev_no_battery(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    return _run_rule_based(cfg, run_dir, share_data, RuleBasedEVNoBatteryController(cfg), with_battery=False)


def run_rule_based_ev_with_battery(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    controller = RuleBasedEVBatteryController.from_training_prices(cfg, share_data.train["price"])
    return _run_rule_based(cfg, run_dir, share_data, controller, with_battery=True)


def run_rule_based_ev_baselines(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    records = {
        RULE_BASED_NO_BATTERY: run_rule_based_ev_no_battery(cfg, run_dir, share_data),
        RULE_BASED_WITH_BATTERY: run_rule_based_ev_with_battery(cfg, run_dir, share_data),
    }
    metrics = pd.concat([records[key]["metrics_df"].assign(scheme_name=key) for key in records], ignore_index=True)
    figures = plot_rule_based_outputs(records, run_dir, cfg)
    return {"records": records, "metrics_df": metrics, "figures": figures}
