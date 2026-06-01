from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from configs.cfg import Cfg
from controllers.mpc_admm import AdmmMpcController
from controllers.mpc_local import LocalMPCController
from data.share_data import ShareData
from utils.records import collect_rollout, compute_ev_agent_cost_summary, compute_ev_cost_summary, load_record, plot_aligned_records, save_rollout

LOCAL_MPC_PERFECT_LABEL = "Local MPC + Perfect Forecast"
LOCAL_MPC_LSTM_LABEL = "Local MPC + LSTM Forecast"
ADMM_MPC_LSTM_LABEL = "ADMM MPC + LSTM Forecast"
EV_MODE_LABELS = {
    "soft": "EV Soft Constraint",
    "hard": "EV Hard Constraint",
    "emergency": "EV Emergency Charging",
}


def run_local_mpc(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    records = {}
    for mode in tqdm(("perfect", "lstm"), desc="local MPC modes", unit="mode", ascii=True):
        controller = LocalMPCController(cfg)
        label = LOCAL_MPC_PERFECT_LABEL if mode == "perfect" else LOCAL_MPC_LSTM_LABEL
        rollout = collect_rollout(cfg, controller, share_data, forecast_mode=mode, label=label)
        records[mode] = save_rollout(cfg, run_dir, rollout, scheme_name=f"local_mpc_{mode}")
        controller.close()
    return records


def run_admm_mpc(cfg: Cfg, run_dir: str | Path, share_data: ShareData) -> dict[str, Any]:
    controller = AdmmMpcController(cfg)
    rollout = collect_rollout(cfg, controller, share_data, forecast_mode="lstm", label=ADMM_MPC_LSTM_LABEL)
    return save_rollout(cfg, run_dir, rollout, scheme_name="admm_mpc_lstm")


def _ev_mpc_cfg(cfg: Cfg, ev_mode: str) -> Cfg:
    mode = str(ev_mode).lower()
    if mode not in EV_MODE_LABELS:
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
    reward = replace(cfg.reward, ev_departure_penalty_weight=0.0 if mode == "hard" else float(cfg.reward.ev_departure_penalty_weight))
    return replace(cfg, env=env, reward=reward, model=replace(cfg.model, action_dim=3))


def run_local_mpc_ev(cfg: Cfg, run_dir: str | Path, share_data: ShareData, ev_mode: str) -> dict[str, Any]:
    work_cfg = _ev_mpc_cfg(cfg, ev_mode)
    suffix = str(ev_mode).lower() if str(ev_mode).lower() != "emergency" else "eme"
    records = {}
    for mode in tqdm(("perfect", "lstm"), desc=f"local MPC EV {suffix} modes", unit="mode", ascii=True):
        controller = LocalMPCController(work_cfg)
        controller.name = f"local_mpc_{mode}_EV_{suffix}"
        label = f"{LOCAL_MPC_PERFECT_LABEL if mode == 'perfect' else LOCAL_MPC_LSTM_LABEL} + {EV_MODE_LABELS[str(ev_mode).lower()]}"
        rollout = collect_rollout(work_cfg, controller, share_data, forecast_mode=mode, label=label)
        saved = save_rollout(work_cfg, run_dir, rollout, scheme_name=controller.name)
        records[mode] = {**saved, "cost_summary": compute_ev_cost_summary(saved["rollout"], work_cfg)[1], "agent_cost_summary": compute_ev_agent_cost_summary(saved["rollout"], work_cfg)}
        controller.close()
    return records


def run_admm_mpc_ev(cfg: Cfg, run_dir: str | Path, share_data: ShareData, ev_mode: str) -> dict[str, Any]:
    work_cfg = _ev_mpc_cfg(cfg, ev_mode)
    suffix = str(ev_mode).lower() if str(ev_mode).lower() != "emergency" else "eme"
    controller = AdmmMpcController(work_cfg)
    controller.name = f"admm_mpc_lstm_EV_{suffix}"
    rollout = collect_rollout(work_cfg, controller, share_data, forecast_mode="lstm", label=f"{ADMM_MPC_LSTM_LABEL} + {EV_MODE_LABELS[str(ev_mode).lower()]}")
    saved = save_rollout(work_cfg, run_dir, rollout, scheme_name=controller.name)
    return {**saved, "cost_summary": compute_ev_cost_summary(saved["rollout"], work_cfg)[1], "agent_cost_summary": compute_ev_agent_cost_summary(saved["rollout"], work_cfg)}


def load_mpc_record(run_dir: str | Path, mode: str, controller: str) -> dict[str, Any]:
    return load_record(run_dir, mode, controller)


def plot_mpc_outputs(records: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    rollouts = {
        "local_perfect": records["local"]["perfect"]["rollout"],
        "local_lstm": records["local"]["lstm"]["rollout"],
        "admm_lstm": records["admm"]["rollout"],
    }
    figures = plot_aligned_records(rollouts, run_dir, "mpc")
    _add_ev_mpc_figures(figures, rollouts, run_dir)
    return figures


def _has_ev_rollout_fields(rollouts: dict[str, Any]) -> bool:
    return any({"ev_soc", "ev_charge_kw"}.issubset(set(rollout.agent_df.columns)) for rollout in rollouts.values())


def _shade_ev_connected(axis, frame) -> None:
    if "ev_available" not in frame.columns:
        return
    connected = frame.groupby(["episode_idx", "step", "timestamp"], as_index=False)["ev_available"].max()
    active = connected["ev_available"].to_numpy(dtype=np.float32) > 0.0
    if not bool(np.any(active)):
        return
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    ends = np.flatnonzero(active & np.r_[~active[1:], True])
    for start, end in zip(starts, ends, strict=False):
        axis.axvspan(connected.iloc[start]["timestamp"], connected.iloc[end]["timestamp"], color="#e5e7eb", alpha=0.22, linewidth=0)


def _plot_ev_soc(rollouts: dict[str, Any], required_soc: float):
    import matplotlib.pyplot as plt

    colors = ("#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2")
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rollout in zip(axes, rollouts.values(), strict=False):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        _shade_ev_connected(ax, agent)
        for agent_id, frame in agent.groupby("agent_id", sort=True):
            ax.plot(frame["timestamp"], frame["ev_soc"], color=colors[int(agent_id) % len(colors)], linewidth=1.7, label=f"Agent {agent_id}")
        ax.axhline(float(required_soc), color="#dc2626", linestyle="--", linewidth=1.1, label="Required departure SOC")
        ax.set_title(f"{rollout.meta['controller']} - EV SOC"); ax.set_ylabel("EV SOC"); ax.set_ylim(0.0, 1.0); ax.grid(True, alpha=0.25); ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_ev_charge(rollouts: dict[str, Any]):
    import matplotlib.pyplot as plt

    colors = ("#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2")
    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.8 * len(rollouts), 4.5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rollout in zip(axes, rollouts.values(), strict=False):
        agent = rollout.agent_df.sort_values(["episode_idx", "agent_id", "step"])
        _shade_ev_connected(ax, agent)
        for agent_id, frame in agent.groupby("agent_id", sort=True):
            color = colors[int(agent_id) % len(colors)]
            if "ev_charge_kw_rl" in frame.columns and np.max(np.abs(frame["ev_charge_kw_rl"].to_numpy(dtype=np.float32) - frame["ev_charge_kw"].to_numpy(dtype=np.float32))) > 1e-6:
                ax.plot(frame["timestamp"], np.maximum(frame["ev_charge_kw_rl"], 0.0), color=color, linestyle=":", linewidth=1.1, alpha=0.75, label=f"Agent {agent_id} RL")
                ax.plot(frame["timestamp"], np.maximum(frame["ev_charge_kw"], 0.0), color=color, linewidth=1.7, label=f"Agent {agent_id} actual")
            else:
                ax.plot(frame["timestamp"], np.maximum(frame["ev_charge_kw"], 0.0), color=color, linewidth=1.7, label=f"Agent {agent_id}")
        ax.set_title(f"{rollout.meta['controller']} - EV charging power"); ax.set_ylabel("kW"); ax.grid(True, alpha=0.25); ax.legend(loc="upper right", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _plot_ev_departure_check(rollouts: dict[str, Any], required_soc: float, departure_step: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rollouts), 1, figsize=(10, max(3.2 * len(rollouts), 4.0)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rollout in zip(axes, rollouts.values(), strict=False):
        agent = rollout.agent_df.copy()
        departures = agent.loc[agent["step"].astype(int) == int(departure_step)]
        if departures.empty:
            departures = agent.sort_values(["episode_idx", "agent_id", "step"]).groupby(["episode_idx", "agent_id"], as_index=False).tail(1)
        if "ev_departure_gap" not in departures.columns:
            departures["ev_departure_gap"] = np.maximum(0.0, float(required_soc) - departures["ev_soc"].astype(float))
        summary = departures.groupby("agent_id", as_index=False).agg(ev_departure_gap=("ev_departure_gap", "max"), ev_soc=("ev_soc", "min"))
        ax.bar(summary["agent_id"].astype(str), summary["ev_departure_gap"].astype(float), color=["#16a34a" if gap <= 1e-6 else "#dc2626" for gap in summary["ev_departure_gap"]])
        for pos, row in enumerate(summary.itertuples(index=False)):
            ax.text(pos, float(row.ev_departure_gap), f"SOC {float(row.ev_soc):.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{rollout.meta['controller']} - Departure SOC requirement check"); ax.set_ylabel("SOC gap"); ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("Agent"); fig.tight_layout(); return fig


def _plot_cost(rollouts: dict[str, Any], *, key: str, title: str, cumulative: bool):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rollouts), 1, figsize=(18, max(3.6 * len(rollouts), 4.5)), sharex=True)
    axes = np.atleast_1d(axes)
    specs = {"battery_cost_eur": ("Battery cost / revenue", "#2563eb"), "ev_cost_eur": ("EV charging cost", "#0891b2"), "pv_cost_eur": ("PV cost", "#16a34a"), "total_cost_eur": ("Total cost", "#111827")}
    columns = {"total_cost_eur": specs["total_cost_eur"]} if key == "total" else specs
    for ax, rollout in zip(axes, rollouts.values(), strict=False):
        cost_ts, _ = compute_ev_cost_summary(rollout, Cfg())
        x = cost_ts["timestamp"] if "timestamp" in cost_ts.columns else cost_ts["step"]
        for column, (label, color) in columns.items():
            values = cost_ts[column].fillna(0.0).to_numpy(dtype=np.float64)
            ax.plot(x, np.cumsum(values) if cumulative else values, color=color, linewidth=1.7 if column == "total_cost_eur" else 1.2, label=label)
        ax.set_title(f"{rollout.meta['controller']} - {title}"); ax.set_ylabel("Cumulative cost [EUR]" if cumulative else "Cost per step [EUR]"); ax.grid(True, alpha=0.25); ax.legend(loc="upper left", ncol=4, fontsize=9)
    axes[-1].set_xlabel("Timestamp"); fig.tight_layout(); return fig


def _add_ev_mpc_figures(figures: dict[str, Any], rollouts: dict[str, Any], run_dir: str | Path) -> None:
    if not _has_ev_rollout_fields(rollouts):
        return
    out = Path(run_dir) / "figures"; out.mkdir(parents=True, exist_ok=True)
    first = next(iter(rollouts.values()))
    required_soc = float(first.meta.get("ev_departure_soc_req", 0.90))
    departure_step = int(first.meta.get("ev_departure_step", 28))
    additions = {
        "mpc_ev_soc": _plot_ev_soc(rollouts, required_soc),
        "mpc_ev_charge_kw": _plot_ev_charge(rollouts),
        "mpc_ev_departure_check": _plot_ev_departure_check(rollouts, required_soc, departure_step),
        "mpc_total_cost": _plot_cost(rollouts, key="total", title="Total cumulative cost", cumulative=True),
        "mpc_cost_components": _plot_cost(rollouts, key="components", title="Cost components per step", cumulative=False),
        "mpc_cumulative_cost_components": _plot_cost(rollouts, key="components", title="Cumulative cost components", cumulative=True),
    }
    for name, fig in additions.items():
        fig.savefig(out / f"{name}.png", dpi=140)
        figures[name] = fig
