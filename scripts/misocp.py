from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import replace

import numpy as np

from configs.cfg import Cfg
from controllers.mpc_global import MisocpController
from data.share_data import ShareData
from utils.records import collect_rollout, compute_ev_agent_cost_summary, compute_ev_cost_summary, load_record, plot_aligned_records, save_rollout


def run_global_misocp(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    rollout = collect_rollout(cfg, MisocpController(cfg), share_data, forecast_mode="perfect", label="Global MISOCP")
    saved = save_rollout(cfg, run_dir, rollout, scheme_name="global_misocp")
    return {"rollout": saved["rollout"], "metrics_df": saved["metrics_df"], "record_dir": saved["record_dir"], "manifest": saved["manifest"]}


def _ev_misocp_cfg(cfg: Cfg, ev_mode: str) -> Cfg:
    mode = str(ev_mode).lower()
    if mode not in {"soft", "hard", "emergency"}:
        raise ValueError(f"ev_mode expected 'soft', 'hard', or 'emergency', got {ev_mode!r}.")
    env = replace(
        cfg.env,
        ev_enabled=True,
        ev_departure_constraint_mode=mode,
        ev_hard_projection_enabled=(mode == "hard"),
        ev_emergency_charging_enabled=(mode == "emergency"),
        ev_emergency_window_hours=4.0 if mode == "emergency" else cfg.env.ev_emergency_window_hours,
        ev_emergency_strategy="required_power" if mode == "emergency" else cfg.env.ev_emergency_strategy,
    )
    return replace(cfg, env=env, model=replace(cfg.model, action_dim=3))


def run_global_misocp_ev(cfg: Cfg, run_dir: str | Path, share_data: ShareData, ev_mode: str) -> dict[str, Any]:
    work_cfg = _ev_misocp_cfg(cfg, ev_mode)
    labels = {
        "soft": ("global_misocp_EV_soft", "Global MISOCP + EV Soft Constraint"),
        "hard": ("global_misocp_EV_hard", "Global MISOCP + EV Hard Constraint"),
        "emergency": ("global_misocp_EV_eme", "Global MISOCP + EV Emergency Charging"),
    }
    scheme_name, label = labels[str(ev_mode).lower()]
    controller = MisocpController(work_cfg)
    controller.name = scheme_name
    rollout = collect_rollout(work_cfg, controller, share_data, forecast_mode="perfect", label=label)
    saved = save_rollout(work_cfg, run_dir, rollout, scheme_name=scheme_name)
    return {"rollout": saved["rollout"], "metrics_df": saved["metrics_df"], "record_dir": saved["record_dir"], "manifest": saved["manifest"], "cost_summary": compute_ev_cost_summary(saved["rollout"], work_cfg)[1], "agent_cost_summary": compute_ev_agent_cost_summary(saved["rollout"], work_cfg)}


def load_misocp_record(run_dir: str | Path) -> dict[str, Any]:
    return load_record(run_dir, "perfect", "MISOCP")


def plot_misocp_outputs(record: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    figures = plot_aligned_records({"global_misocp": record["rollout"]}, run_dir, "misocp")
    rollout = record["rollout"]
    if {"ev_soc", "ev_charge_kw"}.issubset(set(rollout.agent_df.columns)):
        import matplotlib.pyplot as plt
        out = Path(run_dir) / "figures"; out.mkdir(parents=True, exist_ok=True)
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        colors = ("#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2")
        for key, column, ylabel, title in [
            ("misocp_ev_soc", "ev_soc", "EV SOC", "EV SOC"),
            ("misocp_ev_charge_kw", "ev_charge_kw", "EV charging power [kW]", "EV charging power"),
        ]:
            fig, ax = plt.subplots(figsize=(18, 4.6))
            for agent_id, frame in agent.groupby("agent_id", sort=True):
                ax.plot(frame["timestamp"], frame[column], color=colors[int(agent_id) % len(colors)], linewidth=1.6, label=f"Agent {agent_id}")
            if column == "ev_soc":
                ax.axhline(float(rollout.meta.get("ev_departure_soc_req", 0.90)), color="#dc2626", linestyle="--", linewidth=1.1, label="Required departure SOC")
                ax.set_ylim(0.0, 1.0)
            ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Timestamp"); ax.grid(True, alpha=0.25); ax.legend(loc="upper right", ncol=4, fontsize=9); fig.tight_layout(); fig.savefig(out / f"{key}.png", dpi=140); figures[key] = fig
        fig, ax = plt.subplots(figsize=(10, 4.0))
        dep_step = int(rollout.meta.get("ev_departure_step", 28))
        departures = agent.loc[agent["step"].astype(int) == dep_step]
        if departures.empty:
            departures = agent.sort_values(["episode_idx", "agent_id", "step"]).groupby(["episode_idx", "agent_id"], as_index=False).tail(1)
        summary = departures.groupby("agent_id", as_index=False).agg(ev_departure_gap=("ev_departure_gap", "max"), ev_soc=("ev_soc", "min"))
        ax.bar(summary["agent_id"].astype(str), summary["ev_departure_gap"].astype(float), color=["#16a34a" if gap <= 1e-6 else "#dc2626" for gap in summary["ev_departure_gap"]])
        ax.set_title("Departure SOC requirement check"); ax.set_ylabel("SOC gap"); ax.set_xlabel("Agent"); ax.grid(True, axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(out / "misocp_ev_departure_check.png", dpi=140); figures["misocp_ev_departure_check"] = fig
        cost_ts, _ = compute_ev_cost_summary(rollout, Cfg())
        cost_specs = [
            ("misocp_total_cost", "Total cumulative cost", {"total_cost_eur": ("Total cumulative cost", "#111827")}, True),
            ("misocp_cost_components", "Cost components per step", {"battery_cost_eur": ("Battery cost / revenue", "#2563eb"), "ev_cost_eur": ("EV charging cost", "#0891b2"), "pv_cost_eur": ("PV cost", "#16a34a"), "total_cost_eur": ("Total cost", "#111827")}, False),
            ("misocp_cumulative_cost_components", "Cumulative cost components", {"battery_cost_eur": ("Battery cost / revenue", "#2563eb"), "ev_cost_eur": ("EV charging cost", "#0891b2"), "pv_cost_eur": ("PV cost", "#16a34a"), "total_cost_eur": ("Total cost", "#111827")}, True),
        ]
        for key, title, columns, cumulative in cost_specs:
            fig, ax = plt.subplots(figsize=(18, 4.6)); x = cost_ts["timestamp"] if "timestamp" in cost_ts.columns else cost_ts["step"]
            for column, (label, color) in columns.items():
                values = cost_ts[column].fillna(0.0).to_numpy(dtype=np.float64)
                ax.plot(x, np.cumsum(values) if cumulative else values, color=color, linewidth=1.7 if column == "total_cost_eur" else 1.2, label=label)
            ax.set_title(title); ax.set_ylabel("Cumulative cost [EUR]" if cumulative else "Cost per step [EUR]"); ax.set_xlabel("Timestamp"); ax.grid(True, alpha=0.25); ax.legend(loc="upper left", ncol=4, fontsize=9); fig.tight_layout(); fig.savefig(out / f"{key}.png", dpi=140); figures[key] = fig
    return figures


def _episode_start_timestamps(share_data: ShareData) -> list[str]:
    return [str(value) for value in share_data.eval["timestamps"][:, 0]]
