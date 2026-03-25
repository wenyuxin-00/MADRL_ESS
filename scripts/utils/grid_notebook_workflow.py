"""High-level workflow helpers for the grid training notebook."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predictors.artifacts import get_default_lstm_artifact_dir
from predictors.training import ensure_lstm_artifacts

PERFECT_PREDICTION_MODE = "perfect"
NORMAL_PREDICTION_MODE = "normal"
ORACLE_EVAL_MODE = "oracle_eval"
FORECAST_EVAL_MODE = "forecast_eval"


def normalize_prediction_mode(prediction_mode: str) -> str:
    normalized = str(prediction_mode).strip().lower()
    if normalized in {"perfect", "perfect_prediction", "perfect prediction"}:
        return PERFECT_PREDICTION_MODE
    if normalized in {"normal", "normal_prediction", "normal prediction"}:
        return NORMAL_PREDICTION_MODE
    raise ValueError(
        "prediction_mode must be one of {'perfect', 'normal'}, "
        f"got '{prediction_mode}'."
    )


def normalize_date_input(value: str | int | None) -> str | None:
    if value in (None, ""):
        return None
    text_value = str(value).strip()
    if len(text_value) == 8 and text_value.isdigit():
        return f"{text_value[:4]}-{text_value[4:6]}-{text_value[6:]}"
    return str(pd.Timestamp(text_value).date())


def normalize_agent_scale(scale: float | list[float] | tuple[float, ...], *, n_agents: int, name: str) -> list[float]:
    values = np.asarray(scale if isinstance(scale, (list, tuple, np.ndarray)) else [scale], dtype=np.float32)
    if values.size == 1:
        values = np.repeat(values, n_agents)
    if values.size != n_agents:
        raise ValueError(f"{name} should provide {n_agents} value(s), got {values.size}.")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative, got {values.tolist()}.")
    return values.astype(np.float32).tolist()


def resolve_forecast_backend(prediction_mode: str, future_horizon: int) -> str:
    mode = normalize_prediction_mode(prediction_mode)
    if mode == PERFECT_PREDICTION_MODE:
        return "perfect"
    if int(future_horizon) <= 0:
        raise ValueError("Normal prediction mode requires future_horizon > 0 for the LSTM forecaster.")
    return "lstm"


def resolve_evaluation_mode(prediction_mode: str) -> str:
    mode = normalize_prediction_mode(prediction_mode)
    return ORACLE_EVAL_MODE if mode == PERFECT_PREDICTION_MODE else FORECAST_EVAL_MODE


def resolve_prediction_mode_from_forecast_backend(forecast_backend: str) -> str:
    normalized = str(forecast_backend).strip().lower()
    return PERFECT_PREDICTION_MODE if normalized == "perfect" else NORMAL_PREDICTION_MODE


def apply_notebook_experiment_settings(
    cfg,
    *,
    prediction_mode: str,
    test_start_date: str | int | None,
    test_end_date: str | int | None,
    agent_profiles: list[str],
    load_scale: float | list[float],
    pv_scale: float | list[float],
    storage_scale: float | list[float],
    future_horizon: int,
    train_year: int | None = None,
    test_year: int | None = None,
) -> dict[str, object]:
    cfg.obs.local_features = ["calendar_time", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.forecast.target_signals = ["price", "load", "pv"]

    cfg.data.agent_profiles = list(agent_profiles)
    cfg.env.num_agents = int(len(agent_profiles))
    cfg.env.future_horizon = int(future_horizon)

    if len(cfg.grid.agent_bus_ids) < cfg.env.num_agents:
        raise ValueError(
            f"grid.agent_bus_ids only provides {len(cfg.grid.agent_bus_ids)} buses, "
            f"but {cfg.env.num_agents} agents were requested."
        )
    cfg.grid.agent_bus_ids = list(cfg.grid.agent_bus_ids[: cfg.env.num_agents])

    if train_year is not None:
        cfg.data.train_year = int(train_year)
    if test_year is not None:
        cfg.data.test_year = int(test_year)

    cfg.data.test_start_date = normalize_date_input(test_start_date)
    cfg.data.test_end_date = normalize_date_input(test_end_date)
    cfg.data.load_scale = normalize_agent_scale(load_scale, n_agents=cfg.env.num_agents, name="load_scale")
    cfg.data.pv_scale = normalize_agent_scale(pv_scale, n_agents=cfg.env.num_agents, name="pv_scale")
    cfg.data.storage_scale = normalize_agent_scale(
        storage_scale,
        n_agents=cfg.env.num_agents,
        name="storage_scale",
    )

    if cfg.data.pv_capacity_kw and len(cfg.data.pv_capacity_kw) != cfg.env.num_agents:
        cfg.data.pv_capacity_kw = []

    resolved_prediction_mode = normalize_prediction_mode(prediction_mode)
    cfg.forecast.type = resolve_forecast_backend(resolved_prediction_mode, cfg.env.future_horizon)
    cfg.runtime.observation_normalization_state = None
    if cfg.forecast.type == "lstm" and cfg.forecast.lstm_artifact_root is None:
        cfg.forecast.lstm_artifact_root = get_default_lstm_artifact_dir()

    return {
        "prediction_mode": resolved_prediction_mode,
        "evaluation_mode": resolve_evaluation_mode(resolved_prediction_mode),
        "forecast_backend": cfg.forecast.type,
        "future_horizon": int(cfg.env.future_horizon),
        "test_start_date": cfg.data.test_start_date,
        "test_end_date": cfg.data.test_end_date,
        "agent_profiles": list(cfg.data.agent_profiles),
        "load_scale": list(cfg.data.load_scale),
        "pv_scale": list(cfg.data.pv_scale),
        "storage_scale": list(cfg.data.storage_scale),
        "train_year": int(cfg.data.train_year),
        "test_year": int(cfg.data.test_year),
    }


def ensure_forecast_ready(cfg) -> dict[str, object] | None:
    if cfg.forecast.type != "lstm":
        return None
    return ensure_lstm_artifacts(cfg, device=cfg.runtime.device)


def build_comparison_cfg(cfg, *, prediction_mode: str):
    comparison_cfg = deepcopy(cfg)
    comparison_cfg.forecast.type = resolve_forecast_backend(prediction_mode, comparison_cfg.env.future_horizon)
    if comparison_cfg.forecast.type == "lstm" and comparison_cfg.forecast.lstm_artifact_root is None:
        comparison_cfg.forecast.lstm_artifact_root = get_default_lstm_artifact_dir()
    return comparison_cfg


def _aligned_prediction(previous_obs: dict | None, current_obs: dict, field_name: str):
    current_value = np.asarray(current_obs[field_name], dtype=np.float32)
    if previous_obs is None:
        if current_value.ndim == 1:
            return float(current_value[0])
        return current_value[..., 0].astype(np.float32)

    previous_value = np.asarray(previous_obs[field_name], dtype=np.float32)
    forecast_index = 1 if previous_value.shape[-1] > 1 else 0
    if previous_value.ndim == 1:
        return float(previous_value[forecast_index])
    return previous_value[..., forecast_index].astype(np.float32)


def _step_timestamp(reset_info: dict[str, object], step_idx: int) -> pd.Timestamp:
    episode_meta = dict(reset_info.get("episode_meta", {}))
    timestamps = episode_meta.get("timestamps") or []
    if step_idx < len(timestamps):
        return pd.Timestamp(timestamps[step_idx])
    return pd.Timestamp(step_idx, unit="m")


def _operating_cost_per_agent(info: dict[str, object], dt: float) -> np.ndarray:
    return (
        np.asarray(info["net_load"], dtype=np.float32)
        * np.float32(dt)
        * np.float32(info["price"])
    ).astype(np.float32)


def _power_to_normalized_action(power_kw: float, power_limit_kw: float) -> np.ndarray:
    denom = max(float(power_limit_kw), 1e-6)
    action = np.clip(float(power_kw) / denom, -1.0, 1.0)
    return np.asarray([action], dtype=np.float32)


def _solve_single_agent_mpc_action(
    *,
    price_seq: np.ndarray,
    load_seq: np.ndarray,
    pv_seq: np.ndarray,
    soc: float,
    battery_capacity_kwh: float,
    p_max_kw: float,
    dt_hours: float,
    efficiency: float,
    soc_min: float,
    soc_max: float,
    soc_target: float,
    energy_grid_size: int = 61,
    terminal_weight: float = 0.25,
) -> float:
    if battery_capacity_kwh <= 0.0 or p_max_kw <= 0.0:
        return 0.0

    prices = np.asarray(price_seq, dtype=np.float32).reshape(-1)
    net_load = (np.asarray(load_seq, dtype=np.float32) - np.asarray(pv_seq, dtype=np.float32)).reshape(-1)
    horizon = int(min(len(prices), len(net_load)))
    if horizon <= 0:
        return 0.0

    eff = max(float(efficiency), 1e-6)
    energy_min = float(soc_min) * float(battery_capacity_kwh)
    energy_max = float(soc_max) * float(battery_capacity_kwh)
    energy_now = float(np.clip(soc, soc_min, soc_max) * battery_capacity_kwh)
    target_energy = float(np.clip(soc_target, soc_min, soc_max) * battery_capacity_kwh)
    energy_grid = np.linspace(energy_min, energy_max, int(max(3, energy_grid_size)), dtype=np.float32)

    terminal = terminal_weight * (prices.max(initial=0.0) + 1e-3) * np.square(energy_grid - target_energy)
    cost_to_go = terminal.astype(np.float32)

    for step_idx in range(horizon - 1, -1, -1):
        next_cost = np.full_like(cost_to_go, np.inf, dtype=np.float32)
        for state_idx, energy in enumerate(energy_grid):
            delta = energy_grid - energy
            power = np.where(
                delta >= 0.0,
                delta / (eff * float(dt_hours)),
                delta * eff / float(dt_hours),
            ).astype(np.float32)
            feasible = np.abs(power) <= float(p_max_kw) + 1e-6
            if not np.any(feasible):
                continue
            feasible_idx = np.flatnonzero(feasible)
            stage_cost = (net_load[step_idx] + power[feasible]) * float(dt_hours) * prices[step_idx]
            total_cost = stage_cost + cost_to_go[feasible_idx]
            next_cost[state_idx] = float(np.min(total_cost))
        cost_to_go = next_cost

    delta0 = energy_grid - energy_now
    power0 = np.where(
        delta0 >= 0.0,
        delta0 / (eff * float(dt_hours)),
        delta0 * eff / float(dt_hours),
    ).astype(np.float32)
    feasible0 = np.abs(power0) <= float(p_max_kw) + 1e-6
    if not np.any(feasible0):
        return 0.0

    feasible_idx0 = np.flatnonzero(feasible0)
    stage0 = (net_load[0] + power0[feasible0]) * float(dt_hours) * prices[0]
    total0 = stage0 + cost_to_go[feasible_idx0]
    return float(power0[feasible_idx0[int(np.argmin(total0))]])


def _mpc_policy(env, obs: dict[str, np.ndarray]) -> list[np.ndarray]:
    actions: list[np.ndarray] = []
    price_seq = np.asarray(obs["price_seq"], dtype=np.float32)
    load_seq = np.asarray(obs["load_seq"], dtype=np.float32)
    pv_seq = np.asarray(obs["pv_seq"], dtype=np.float32)
    for agent_idx in range(env.n):
        power_kw = _solve_single_agent_mpc_action(
            price_seq=price_seq,
            load_seq=load_seq[agent_idx],
            pv_seq=pv_seq[agent_idx],
            soc=float(env.soc[agent_idx]),
            battery_capacity_kwh=float(env.agent_c_bat[agent_idx]),
            p_max_kw=float(env.agent_p_max[agent_idx]),
            dt_hours=float(env.dt),
            efficiency=float(env.eff),
            soc_min=float(env.soc_min),
            soc_max=float(env.soc_max),
            soc_target=float(env.soc_target),
        )
        actions.append(_power_to_normalized_action(power_kw, float(env.agent_p_max[agent_idx])))
    return actions


@dataclass
class RolloutResult:
    step_df: pd.DataFrame
    agent_df: pd.DataFrame
    grid_df: pd.DataFrame
    summary: pd.DataFrame
    meta: dict[str, object]


def collect_controller_rollout(
    cfg,
    *,
    label: str,
    controller=None,
    action_fn=None,
) -> RolloutResult:
    if (controller is None) == (action_fn is None):
        raise ValueError("Provide exactly one of controller or action_fn.")

    from scripts.builder import build_env

    env = build_env(cfg, mode="test")
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(env._grid_core, "agent_bus_ids", cfg.grid.agent_bus_ids)]
    agent_bus_set = set(agent_bus_ids)
    try:
        for episode_idx in range(env.num_available_episodes):
            obs, reset_info = env.reset(episode_idx=episode_idx)
            raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
            if controller is not None:
                controller.reset()
            previous_raw_obs = None
            done = False
            step_in_episode = 0

            while not done:
                price_pred = _aligned_prediction(previous_raw_obs, raw_obs, "price_seq")
                load_pred = _aligned_prediction(previous_raw_obs, raw_obs, "load_seq")
                pv_pred = _aligned_prediction(previous_raw_obs, raw_obs, "pv_seq")
                timestamp = _step_timestamp(reset_info, step_in_episode)

                if controller is not None:
                    actions = controller.act(obs, deterministic=True)
                else:
                    actions = action_fn(env, raw_obs)

                next_obs, reward, terminated, truncated, info = env.step(actions)
                del reward, terminated, truncated
                raw_next_obs = (
                    env.obs_builder.build_raw(env)
                    if hasattr(env.obs_builder, "build_raw") and not bool(info.get("episode_done", False))
                    else next_obs
                )

                cost_per_agent = _operating_cost_per_agent(info, float(env.dt))
                step_rows.append(
                    {
                        "controller": label,
                        "episode_idx": episode_idx,
                        "step": step_in_episode,
                        "timestamp": timestamp,
                        "price": float(info["price"]),
                        "price_pred": float(price_pred),
                        "operating_cost": float(np.sum(cost_per_agent)),
                    }
                )

                for agent_idx, profile in enumerate(cfg.data.agent_profiles):
                    agent_rows.append(
                        {
                            "controller": label,
                            "episode_idx": episode_idx,
                            "step": step_in_episode,
                            "timestamp": timestamp,
                            "agent_id": agent_idx,
                            "agent_profile": str(profile),
                            "load": float(np.asarray(info["load"], dtype=np.float32)[agent_idx]),
                            "load_pred": float(np.asarray(load_pred, dtype=np.float32)[agent_idx]),
                            "pv": float(np.asarray(info["pv"], dtype=np.float32)[agent_idx]),
                            "pv_pred": float(np.asarray(pv_pred, dtype=np.float32)[agent_idx]),
                            "e_bat": float(np.asarray(info["e_bat"], dtype=np.float32)[agent_idx]),
                            "soc": float(np.asarray(info["soc_next"], dtype=np.float32)[agent_idx]),
                            "operating_cost": float(cost_per_agent[agent_idx]),
                        }
                    )

                vm_pu = np.asarray(info.get("vm_pu", np.zeros(len(bus_ids), dtype=np.float32)), dtype=np.float32)
                if vm_pu.shape[0] == len(bus_ids):
                    for bus_id, vm_value in zip(bus_ids, vm_pu, strict=False):
                        grid_rows.append(
                            {
                                "controller": label,
                                "episode_idx": episode_idx,
                                "step": step_in_episode,
                                "timestamp": timestamp,
                                "bus_id": int(bus_id),
                                "vm_pu": float(vm_value),
                                "is_agent_bus": bool(int(bus_id) in agent_bus_set),
                            }
                        )

                done = bool(info.get("episode_done", False))
                previous_raw_obs = raw_obs
                obs = next_obs
                raw_obs = raw_next_obs
                step_in_episode += 1

        step_df = pd.DataFrame(step_rows).sort_values(["episode_idx", "step"]).reset_index(drop=True)
        agent_df = pd.DataFrame(agent_rows).sort_values(["episode_idx", "step", "agent_id"]).reset_index(drop=True)
        grid_df = pd.DataFrame(grid_rows).sort_values(["episode_idx", "step", "bus_id"]).reset_index(drop=True)
        summary = (
            agent_df.groupby(["controller", "agent_profile"], as_index=False)["operating_cost"].sum()
            if not agent_df.empty
            else pd.DataFrame(columns=["controller", "agent_profile", "operating_cost"])
        )
        return RolloutResult(
            step_df=step_df,
            agent_df=agent_df,
            grid_df=grid_df,
            summary=summary,
            meta={
                "controller": label,
                "n_agents": int(env.n),
                "agent_profiles": list(cfg.data.agent_profiles),
                "agent_bus_ids": agent_bus_ids,
                "bus_ids": bus_ids,
                "v_min_pu": float(cfg.grid.v_min_pu),
                "v_max_pu": float(cfg.grid.v_max_pu),
                "future_horizon": int(cfg.env.future_horizon),
                "prediction_mode": resolve_prediction_mode_from_forecast_backend(cfg.forecast.type),
                "evaluation_mode": resolve_evaluation_mode(
                    resolve_prediction_mode_from_forecast_backend(cfg.forecast.type)
                ),
                "dt_hours": float(cfg.env.dt),
            },
        )
    finally:
        env.close()


def collect_madrl_rollout(
    cfg,
    *,
    model_root=None,
    algorithm: str | None = None,
    episode_tag: int | None = None,
    experiment_name: str = "grid_mainline",
    checkpoint_root=None,
) -> RolloutResult:
    from scripts.utils.experiment_notebook_utils import load_madrl_controller

    prediction_mode = resolve_prediction_mode_from_forecast_backend(cfg.forecast.type)
    loaded = load_madrl_controller(
        cfg,
        model_root,
        algorithm=algorithm,
        episode_tag=episode_tag,
        device=cfg.runtime.device,
        prediction_mode=prediction_mode,
        experiment_name=experiment_name,
        checkpoint_root=checkpoint_root,
    )
    return collect_controller_rollout(
        loaded["cfg"],
        label=f"DRL ({resolve_evaluation_mode(prediction_mode)})",
        controller=loaded["controller"],
    )


def collect_mpc_rollout(cfg, *, prediction_mode: str, label: str | None = None) -> RolloutResult:
    comparison_cfg = build_comparison_cfg(cfg, prediction_mode=prediction_mode)
    resolved_mode = normalize_prediction_mode(prediction_mode)
    rollout_label = label or f"MPC ({resolve_evaluation_mode(resolved_mode)})"
    return collect_controller_rollout(
        comparison_cfg,
        label=rollout_label,
        action_fn=_mpc_policy,
    )


def compare_operating_costs(*rollouts: RolloutResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rollout in rollouts:
        total_cost = float(rollout.agent_df["operating_cost"].sum()) if not rollout.agent_df.empty else 0.0
        rows.append(
            {
                "controller": rollout.meta["controller"],
                "total_operating_cost": total_cost,
            }
        )
    return pd.DataFrame(rows).sort_values("total_operating_cost").reset_index(drop=True)


def plot_test_rollout(rollout: RolloutResult, *, figsize: tuple[float, float] | None = None):
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    if step_df.empty or agent_df.empty:
        raise ValueError("Rollout is empty; nothing to plot.")

    agent_profiles = list(rollout.meta["agent_profiles"])
    n_agents = len(agent_profiles)
    if figsize is None:
        figsize = (18.0, 2.8 * (3 + n_agents))

    figure, axes = plt.subplots(3 + n_agents, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    palette = ["#0f172a", "#2563eb", "#16a34a", "#ea580c", "#dc2626", "#7c3aed"]

    axes[0].plot(step_df["timestamp"], step_df["price"], color="#111827", linewidth=1.6, label="Actual")
    axes[0].plot(
        step_df["timestamp"],
        step_df["price_pred"],
        color="#dc2626",
        linewidth=1.4,
        linestyle="--",
        label="Forecast",
    )
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"Test Rollout - {rollout.meta['controller']}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    for axis_idx, signal_name in enumerate(["pv", "load"], start=1):
        axis = axes[axis_idx]
        for agent_idx, profile in enumerate(agent_profiles):
            agent_frame = agent_df.loc[agent_df["agent_profile"] == profile]
            color = palette[agent_idx % len(palette)]
            axis.plot(
                agent_frame["timestamp"],
                agent_frame[signal_name],
                color=color,
                linewidth=1.4,
                label=f"{profile} actual",
            )
            axis.plot(
                agent_frame["timestamp"],
                agent_frame[f"{signal_name}_pred"],
                color=color,
                linewidth=1.2,
                linestyle="--",
                label=f"{profile} forecast",
            )
        axis.set_ylabel(signal_name.upper())
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=2)

    for agent_offset, profile in enumerate(agent_profiles, start=3):
        axis = axes[agent_offset]
        agent_frame = agent_df.loc[agent_df["agent_profile"] == profile]
        charge = np.clip(agent_frame["e_bat"].to_numpy(dtype=np.float32), 0.0, None)
        discharge = np.clip(agent_frame["e_bat"].to_numpy(dtype=np.float32), None, 0.0)
        axis.bar(agent_frame["timestamp"], charge, width=0.008, color="#dc2626", alpha=0.7, label="Charge")
        axis.bar(agent_frame["timestamp"], discharge, width=0.008, color="#2563eb", alpha=0.7, label="Discharge")
        axis.set_ylabel(f"{profile}\nP_bat")
        axis.grid(True, alpha=0.25)

        soc_axis = axis.twinx()
        soc_axis.plot(agent_frame["timestamp"], agent_frame["soc"], color="#111827", linewidth=1.2, label="SoC")
        soc_axis.set_ylabel("SoC")
        soc_axis.set_ylim(0.0, 1.0)

        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = soc_axis.get_legend_handles_labels()
        axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    axes[-1].set_xlabel("Timestamp")
    figure.tight_layout()
    return figure


def plot_test_voltage_profile(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] = (18.0, 4.8),
):
    grid_df = rollout.grid_df.copy()
    if grid_df.empty:
        raise ValueError("Rollout does not contain full-grid voltage traces.")

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    muted_color = "#cbd5e1"
    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]

    for bus_id, frame in grid_df.loc[~grid_df["is_agent_bus"]].groupby("bus_id"):
        axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=muted_color,
            linewidth=0.9,
            alpha=0.35,
            zorder=1,
        )

    for color_idx, bus_id in enumerate(rollout.meta["agent_bus_ids"]):
        frame = grid_df.loc[grid_df["bus_id"] == int(bus_id)]
        if frame.empty:
            continue
        axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=highlight_palette[color_idx % len(highlight_palette)],
            linewidth=2.1,
            alpha=0.95,
            label=f"Agent bus {bus_id}",
            zorder=3,
        )

    axis.axhline(float(rollout.meta["v_min_pu"]), color="#dc2626", linestyle="--", linewidth=1.1, label="V min")
    axis.axhline(float(rollout.meta["v_max_pu"]), color="#ea580c", linestyle="--", linewidth=1.1, label="V max")
    axis.set_title(f"Node Voltage Profile - {rollout.meta['controller']}")
    axis.set_ylabel("Voltage [p.u.]")
    axis.set_xlabel("Timestamp")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", ncol=2)
    figure.tight_layout()
    return figure


def plot_operating_cost_comparison(cost_df: pd.DataFrame, *, figsize: tuple[float, float] = (10.0, 4.5)):
    if cost_df.empty:
        raise ValueError("cost_df is empty; nothing to plot.")
    figure, axis = plt.subplots(1, 1, figsize=figsize)
    ordered = cost_df.sort_values("total_operating_cost", ascending=True)
    axis.bar(ordered["controller"], ordered["total_operating_cost"], color=["#2563eb", "#16a34a", "#dc2626"])
    axis.set_ylabel("Real Operating Cost")
    axis.set_title("Operating Cost Comparison")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


__all__ = [
    "FORECAST_EVAL_MODE",
    "NORMAL_PREDICTION_MODE",
    "ORACLE_EVAL_MODE",
    "PERFECT_PREDICTION_MODE",
    "RolloutResult",
    "apply_notebook_experiment_settings",
    "build_comparison_cfg",
    "collect_madrl_rollout",
    "collect_mpc_rollout",
    "collect_controller_rollout",
    "compare_operating_costs",
    "ensure_forecast_ready",
    "normalize_date_input",
    "normalize_prediction_mode",
    "plot_operating_cost_comparison",
    "plot_test_rollout",
    "plot_test_voltage_profile",
    "resolve_evaluation_mode",
    "resolve_forecast_backend",
    "resolve_prediction_mode_from_forecast_backend",
]
