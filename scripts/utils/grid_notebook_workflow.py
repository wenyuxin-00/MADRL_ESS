"""High-level workflow helpers for the grid training notebook."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers.action_feasibility import (
    build_safety_local_numpy,
    compute_action_gap_metrics_numpy,
    merge_action_info_into_step_info,
)
from controllers.mpc import solve_single_agent_gurobi_mpc_action
from envs.grid.deployments import resolve_fixed_battery_spec
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


def resolve_battery_controls(cfg, battery_controls: Mapping[str, object] | None = None) -> dict[str, object]:
    controls = dict(battery_controls or {})
    capacity_kwh, c_rate, p_max_kw = resolve_fixed_battery_spec(
        controls.get("battery_capacity", cfg.env.battery_capacity),
        controls.get("max_charge_rate", cfg.env.max_charge_rate),
        n_agents=int(cfg.env.num_agents),
    )
    resolved = {
        "mode": "fixed",
        "battery_capacity": list(capacity_kwh),
        "max_charge_rate": float(c_rate),
        "p_max_kw": list(p_max_kw),
        "efficiency": float(controls.get("efficiency", cfg.env.efficiency)),
        "init_soc": float(controls.get("init_soc", cfg.env.init_soc)),
        "soc_min": float(controls.get("soc_min", cfg.env.soc_min)),
        "soc_max": float(controls.get("soc_max", cfg.env.soc_max)),
        "soc_target": float(controls.get("soc_target", cfg.env.soc_target)),
    }
    if resolved["efficiency"] <= 0.0 or resolved["efficiency"] > 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {resolved['efficiency']}.")
    if not 0.0 <= resolved["soc_min"] <= resolved["soc_max"] <= 1.0:
        raise ValueError(
            f"Invalid SoC range: soc_min={resolved['soc_min']}, soc_max={resolved['soc_max']}."
        )
    if not 0.0 <= resolved["init_soc"] <= 1.0:
        raise ValueError(f"init_soc must be in [0, 1], got {resolved['init_soc']}.")
    if not 0.0 <= resolved["soc_target"] <= 1.0:
        raise ValueError(f"soc_target must be in [0, 1], got {resolved['soc_target']}.")
    return resolved


def _normalize_signal_training_overrides(
    overrides: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for signal_name, signal_overrides in dict(overrides or {}).items():
        if not isinstance(signal_overrides, Mapping):
            raise ValueError(
                f"signal_training_overrides['{signal_name}'] must be a mapping, got {signal_overrides!r}."
            )
        normalized[str(signal_name).strip().lower()] = {
            str(field_name): value for field_name, value in dict(signal_overrides).items()
        }
    return normalized


def resolve_forecast_controls(cfg, forecast_controls: Mapping[str, object] | None = None) -> dict[str, object]:
    controls = dict(forecast_controls or {})
    target_signals = [str(value) for value in controls.get("target_signals", cfg.forecast.target_signals)]
    history_window = int(controls.get("history_window", cfg.forecast.history_window))
    configured_future_horizon = controls.get("future_horizon")
    if configured_future_horizon is not None and int(configured_future_horizon) != int(cfg.env.future_horizon):
        raise ValueError(
            "forecast_controls.future_horizon must match cfg.env.future_horizon, "
            f"got {configured_future_horizon} vs {cfg.env.future_horizon}."
        )

    artifact_root = controls.get("artifact_root", cfg.forecast.lstm_artifact_root)
    resolved_artifact_root = None if artifact_root in (None, "") else str(Path(artifact_root).resolve())
    signal_training_overrides = _normalize_signal_training_overrides(
        controls.get("signal_training_overrides", getattr(cfg.forecast, "signal_training_overrides", {}))
    )

    resolved = {
        "target_signals": target_signals,
        "history_window": history_window,
        "load_model_mode": str(controls.get("load_model_mode", cfg.forecast.load_model_mode)),
        "load_component_split": bool(controls.get("load_component_split", cfg.forecast.load_component_split)),
        "load_scaler_type": str(controls.get("load_scaler_type", cfg.forecast.load_scaler_type)),
        "load_time_feature_mode": str(
            controls.get("load_time_feature_mode", cfg.forecast.load_time_feature_mode)
        ),
        "pv_time_feature_mode": str(controls.get("pv_time_feature_mode", cfg.forecast.pv_time_feature_mode)),
        "load_hybrid_mode": str(controls.get("load_hybrid_mode", cfg.forecast.load_hybrid_mode)),
        "load_baseline_mode": str(controls.get("load_baseline_mode", cfg.forecast.load_baseline_mode)),
        "pv_postprocess_mode": str(controls.get("pv_postprocess_mode", cfg.forecast.pv_postprocess_mode)),
        "auto_train_missing": bool(controls.get("auto_train_missing", cfg.forecast.auto_train_missing)),
        "artifact_root": resolved_artifact_root,
        "signal_training_overrides": signal_training_overrides,
    }
    return resolved


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
    agent_profiles: list[str] | None = None,
    agent_bus_ids: list[int] | None = None,
    load_scale: float | list[float] | None = None,
    pv_scale: float | list[float] | None = None,
    future_horizon: int | None = None,
    battery_controls: Mapping[str, object] | None = None,
    forecast_controls: Mapping[str, object] | None = None,
    train_year: int | None = None,
    test_year: int | None = None,
) -> dict[str, object]:
    cfg.obs.local_features = ["calendar_time", "soc"]
    cfg.obs.sequence_features = ["price", "load", "pv"]
    cfg.forecast.target_signals = ["price", "load", "pv"]

    cfg.data.agent_profiles = list(cfg.data.agent_profiles if agent_profiles is None else agent_profiles)
    cfg.env.num_agents = int(len(cfg.data.agent_profiles))
    cfg.env.future_horizon = int(cfg.env.future_horizon if future_horizon is None else future_horizon)

    resolved_agent_bus_ids = list(cfg.grid.agent_bus_ids if agent_bus_ids is None else agent_bus_ids)
    if len(resolved_agent_bus_ids) < cfg.env.num_agents:
        raise ValueError(
            f"grid.agent_bus_ids only provides {len(resolved_agent_bus_ids)} buses, "
            f"but {cfg.env.num_agents} agents were requested."
        )
    cfg.grid.agent_bus_ids = [int(bus_id) for bus_id in resolved_agent_bus_ids[: cfg.env.num_agents]]

    if train_year is not None:
        cfg.data.train_year = int(train_year)
    if test_year is not None:
        cfg.data.test_year = int(test_year)

    cfg.data.test_start_date = normalize_date_input(test_start_date)
    cfg.data.test_end_date = normalize_date_input(test_end_date)
    resolved_load_scale = (
        cfg.data.resolved_load_scale(cfg.env.num_agents) if load_scale is None else load_scale
    )
    resolved_pv_scale = (
        cfg.data.resolved_pv_scale(cfg.env.num_agents) if pv_scale is None else pv_scale
    )
    cfg.data.load_scale = normalize_agent_scale(
        resolved_load_scale,
        n_agents=cfg.env.num_agents,
        name="load_scale",
    )
    cfg.data.pv_scale = normalize_agent_scale(
        resolved_pv_scale,
        n_agents=cfg.env.num_agents,
        name="pv_scale",
    )
    resolved_battery_controls = resolve_battery_controls(cfg, battery_controls)
    cfg.env.battery_capacity = list(resolved_battery_controls["battery_capacity"])
    cfg.env.max_charge_rate = resolved_battery_controls["max_charge_rate"]
    cfg.env.efficiency = resolved_battery_controls["efficiency"]
    cfg.env.init_soc = resolved_battery_controls["init_soc"]
    cfg.env.soc_min = resolved_battery_controls["soc_min"]
    cfg.env.soc_max = resolved_battery_controls["soc_max"]
    cfg.env.soc_target = resolved_battery_controls["soc_target"]

    if cfg.data.pv_capacity_kw and len(cfg.data.pv_capacity_kw) != cfg.env.num_agents:
        cfg.data.pv_capacity_kw = []

    resolved_prediction_mode = normalize_prediction_mode(prediction_mode)
    cfg.forecast.type = resolve_forecast_backend(resolved_prediction_mode, cfg.env.future_horizon)
    resolved_forecast_controls = resolve_forecast_controls(cfg, forecast_controls)
    cfg.forecast.target_signals = list(resolved_forecast_controls["target_signals"])
    cfg.forecast.history_window = int(resolved_forecast_controls["history_window"])
    cfg.forecast.load_model_mode = str(resolved_forecast_controls["load_model_mode"])
    cfg.forecast.load_component_split = bool(resolved_forecast_controls["load_component_split"])
    cfg.forecast.load_scaler_type = str(resolved_forecast_controls["load_scaler_type"])
    cfg.forecast.load_time_feature_mode = str(resolved_forecast_controls["load_time_feature_mode"])
    cfg.forecast.pv_time_feature_mode = str(resolved_forecast_controls["pv_time_feature_mode"])
    cfg.forecast.load_hybrid_mode = str(resolved_forecast_controls["load_hybrid_mode"])
    cfg.forecast.load_baseline_mode = str(resolved_forecast_controls["load_baseline_mode"])
    cfg.forecast.pv_postprocess_mode = str(resolved_forecast_controls["pv_postprocess_mode"])
    cfg.forecast.auto_train_missing = bool(resolved_forecast_controls["auto_train_missing"])
    cfg.forecast.signal_training_overrides = dict(resolved_forecast_controls["signal_training_overrides"])
    cfg.runtime.observation_normalization_state = None
    if cfg.forecast.type == "lstm":
        cfg.forecast.lstm_artifact_root = (
            resolved_forecast_controls["artifact_root"] or str(get_default_lstm_artifact_dir())
        )
    else:
        cfg.forecast.lstm_artifact_root = resolved_forecast_controls["artifact_root"]

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
        "battery": dict(resolved_battery_controls),
        "forecast": {
            "artifact_root": cfg.forecast.lstm_artifact_root,
            "target_signals": list(cfg.forecast.target_signals),
            "history_window": int(cfg.forecast.history_window),
            "auto_train_missing": bool(cfg.forecast.auto_train_missing),
            "signal_training_overrides": dict(cfg.forecast.signal_training_overrides),
            "load_component_split": bool(cfg.forecast.load_component_split),
            "load_scaler_type": str(cfg.forecast.load_scaler_type),
        },
        "train_year": int(cfg.data.train_year),
        "test_year": int(cfg.data.test_year),
        "agent_bus_ids": list(cfg.grid.agent_bus_ids),
    }


def ensure_forecast_ready(cfg) -> dict[str, object] | None:
    if cfg.forecast.type != "lstm":
        return None
    return ensure_lstm_artifacts(cfg, device=cfg.runtime.device)


def cache_only_forecast_enabled(cfg) -> bool:
    shared_data_dir = getattr(getattr(cfg, "runtime", None), "shared_data_dir", None)
    return shared_data_dir not in (None, "")


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


def _purchase_cost_per_agent(info: dict[str, object], dt: float) -> np.ndarray:
    grid_import = np.asarray(
        info.get("grid_import_kw", np.maximum(np.asarray(info["net_load"], dtype=np.float32), 0.0)),
        dtype=np.float32,
    )
    return (grid_import * np.float32(dt) * np.float32(info["price"])).astype(np.float32)


def _export_subsidy_per_agent(info: dict[str, object], dt: float, subsidy_rate: float) -> np.ndarray:
    grid_export = np.asarray(
        info.get("grid_export_kw", np.maximum(-np.asarray(info["net_load"], dtype=np.float32), 0.0)),
        dtype=np.float32,
    )
    return (grid_export * np.float32(dt) * np.float32(subsidy_rate)).astype(np.float32)


def _mean_component_total(info: dict[str, object], component_key: str, n_agents: int) -> float:
    values = np.asarray(info.get(component_key, np.zeros((n_agents,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _approx_trafo_limit_kw(env) -> float | None:
    net = getattr(getattr(env, "_grid_core", None), "net", None)
    trafo_table = getattr(net, "trafo", None)
    if trafo_table is None or len(trafo_table) == 0:
        return None
    try:
        if "sn_mva" not in trafo_table:
            return None
        sn_mva = np.asarray(trafo_table["sn_mva"], dtype=np.float64).reshape(-1)
        if sn_mva.size == 0:
            return None
        return float(np.sum(sn_mva) * 1000.0)
    except Exception:
        return None


def _load_json_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonicalize_compare_value(value):
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_compare_value(sub_value)
            for key, sub_value in sorted(dict(value).items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_canonicalize_compare_value(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 10)
    return value


def load_training_run_bundle(model_root) -> dict[str, object]:
    resolved_model_root = Path(model_root).resolve()
    meta_dir = resolved_model_root / "_meta"
    required_files = (
        "train_result.json",
        "experiment_controls.json",
        "data_controls.json",
        "battery_controls.json",
        "train_controls.json",
        "checkpoint_controls.json",
    )
    missing = [filename for filename in required_files if not (meta_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"Training metadata is incomplete for '{resolved_model_root}'. Missing: {missing}"
        )
    bundle = {
        "model_root": str(resolved_model_root),
        "meta_dir": str(meta_dir),
    }
    for filename in required_files:
        bundle[filename.removesuffix(".json")] = _load_json_payload(meta_dir / filename)
    return bundle


def _extract_compare_signature(bundle: Mapping[str, object]) -> dict[str, object]:
    experiment_controls = dict(bundle["experiment_controls"])
    data_controls = dict(bundle["data_controls"])
    battery_controls = dict(bundle["battery_controls"])
    train_controls = dict(bundle["train_controls"])
    reward_controls = dict(experiment_controls.get("reward_controls", {}))
    forecast_controls = dict(experiment_controls.get("forecast_controls", {}))
    model_controls = dict(experiment_controls.get("model_controls", {}))
    return {
        "prediction_mode": str(data_controls.get("prediction_mode", "perfect")),
        "seed": int(experiment_controls.get("seed", 0)),
        "export_subsidy_eur_per_kwh": float(reward_controls.get("export_subsidy_eur_per_kwh", 0.079)),
        "w_soc_pen": float(reward_controls.get("w_soc_pen", reward_controls.get("w_action_pen", 0.0))),
        "agent_profiles": list(data_controls.get("agent_profiles", [])),
        "agent_bus_ids": list(data_controls.get("agent_bus_ids", [])),
        "load_scale": list(data_controls.get("load_scale", [])),
        "pv_scale": list(data_controls.get("pv_scale", [])),
        "future_horizon": int(data_controls.get("future_horizon", 0)),
        "train_year": data_controls.get("train_year"),
        "test_year": data_controls.get("test_year"),
        "test_start_date": data_controls.get("test_start_date"),
        "test_end_date": data_controls.get("test_end_date"),
        "shared_data_signature": bundle.get("train_result", {}).get("shared_data_signature"),
        "battery_controls": battery_controls,
        "forecast_controls": forecast_controls,
        "model_controls": model_controls,
        "train_controls": {
            key: train_controls.get(key)
            for key in (
                "profile",
                "model_family",
                "train_episodes",
                "max_train_steps",
                "num_envs",
                "vec_env_type",
                "batch_size",
                "buffer_size",
                "update_interval",
                "updates_per_step",
                "policy_update_freq",
                "actor_lr",
                "critic_lr",
                "noise_std_init",
                "noise_std_min",
                "noise_decay_steps",
                "use_noise_decay",
            )
        },
    }


def validate_compare_model_bundles(
    model_roots: Mapping[str, str | Path],
    *,
    expected_count: int = 3,
) -> dict[str, dict[str, object]]:
    if len(model_roots) != int(expected_count):
        raise ValueError(
            f"Compare requires exactly {expected_count} DRL model roots, got {len(model_roots)}."
        )

    bundles: dict[str, dict[str, object]] = {}
    signatures: dict[str, dict[str, object]] = {}
    for label, model_root in dict(model_roots).items():
        bundle = load_training_run_bundle(model_root)
        bundles[str(label)] = bundle
        signatures[str(label)] = _extract_compare_signature(bundle)

    labels = list(signatures)
    reference_label = labels[0]
    reference_signature = signatures[reference_label]
    mismatch_lines: list[str] = []
    for label in labels[1:]:
        current_signature = signatures[label]
        for field_name, reference_value in reference_signature.items():
            current_value = current_signature[field_name]
            if _canonicalize_compare_value(current_value) != _canonicalize_compare_value(reference_value):
                mismatch_lines.append(
                    f"{field_name}: {reference_label}={reference_value!r} vs {label}={current_value!r}"
                )
    if mismatch_lines:
        joined = "\n".join(mismatch_lines)
        raise ValueError(
            "Compare model metadata mismatch detected. All DRL runs must share the same "
            f"subsidy, hyperparameters, forecast controls, battery controls, dates, and seed.\n{joined}"
        )
    return bundles


def _power_to_normalized_action(power_kw: float, power_limit_kw: float) -> np.ndarray:
    denom = max(float(power_limit_kw), 1e-6)
    action = np.clip(float(power_kw) / denom, -1.0, 1.0)
    return np.asarray([action], dtype=np.float32)


def _power_to_full_action(power_kw: float, power_limit_kw: float, *, pv_action: float = 1.0) -> np.ndarray:
    battery_action = _power_to_normalized_action(power_kw, power_limit_kw)[0]
    return np.asarray([battery_action, float(pv_action)], dtype=np.float32)


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
) -> float:
    return solve_single_agent_gurobi_mpc_action(
        price_seq=price_seq,
        load_seq=load_seq,
        pv_seq=pv_seq,
        soc=soc,
        battery_capacity_kwh=battery_capacity_kwh,
        p_max_kw=p_max_kw,
        dt_hours=dt_hours,
        efficiency=efficiency,
        soc_min=soc_min,
        soc_max=soc_max,
    )


def _mpc_policy(env, obs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
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
        )
        actions.append(_power_to_full_action(power_kw, float(env.agent_p_max[agent_idx]), pv_action=1.0))
    load_raw = (
        np.asarray(env.get_signal_step("load"), dtype=np.float32)
        if hasattr(env, "get_signal_step")
        else np.asarray(load_seq[:, 0], dtype=np.float32)
    )
    pv_raw = (
        np.asarray(env.get_signal_step("pv"), dtype=np.float32)
        if hasattr(env, "get_signal_step")
        else np.asarray(pv_seq[:, 0], dtype=np.float32)
    )
    action_array = np.asarray(actions, dtype=np.float32)
    action_info = compute_action_gap_metrics_numpy(
        build_safety_local_numpy(
            soc=np.asarray(env.soc, dtype=np.float32),
            load_raw=load_raw,
            pv_raw=pv_raw,
            battery_capacity_kwh=np.asarray(env.agent_c_bat, dtype=np.float32),
            p_max_kw=np.asarray(env.agent_p_max, dtype=np.float32),
        ),
        action_array,
        action_array,
    )
    return actions, action_info


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

    if cache_only_forecast_enabled(cfg):
        cfg.runtime.forecast_ready = None
    else:
        cfg.runtime.forecast_ready = ensure_forecast_ready(cfg)
    env = build_env(cfg, mode="test")
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(env._grid_core, "agent_bus_ids", cfg.grid.agent_bus_ids)]
    agent_bus_set = set(agent_bus_ids)
    trafo_limit_kw = _approx_trafo_limit_kw(env)
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
                    action_info = getattr(controller, "last_action_info", None)
                else:
                    action_result = action_fn(env, raw_obs)
                    if isinstance(action_result, tuple) and len(action_result) == 2:
                        actions, action_info = action_result
                    else:
                        actions = action_result
                        action_info = None

                next_obs, reward, terminated, truncated, info = env.step(actions)
                reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
                apply_action_penalty = bool(getattr(controller, "apply_action_penalty", False))
                info, action_penalty = merge_action_info_into_step_info(
                    info,
                    action_info,
                    soc_pen_weight=float(cfg.reward.w_soc_pen),
                    apply_action_penalty=apply_action_penalty,
                )
                reward_array = reward_array - np.asarray(action_penalty, dtype=np.float32)
                info["reward"] = reward_array.astype(np.float32)
                del terminated, truncated
                raw_next_obs = (
                    env.obs_builder.build_raw(env)
                    if hasattr(env.obs_builder, "build_raw") and not bool(info.get("episode_done", False))
                    else next_obs
                )

                purchase_cost_per_agent = _purchase_cost_per_agent(info, float(env.dt))
                base_net_load = np.asarray(
                    info.get(
                        "base_net_load",
                        np.asarray(info["load"], dtype=np.float32) - np.asarray(info["pv"], dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
                base_net_load_effective = np.asarray(
                    info.get("base_net_load_effective", base_net_load),
                    dtype=np.float32,
                )
                net_load = np.asarray(
                    info.get(
                        "net_load",
                        base_net_load_effective + np.asarray(info["e_bat"], dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
                pv_raw = np.asarray(info.get("pv_raw", info["pv"]), dtype=np.float32)
                pv_effective = np.asarray(info.get("pv_effective", pv_raw), dtype=np.float32)
                pv_curtail = np.asarray(info.get("pv_curtail", pv_raw - pv_effective), dtype=np.float32)
                pv_utilization = np.asarray(
                    info.get("pv_utilization", np.ones_like(pv_raw, dtype=np.float32)),
                    dtype=np.float32,
                )
                grid_import = np.asarray(
                    info.get("grid_import_kw", np.maximum(net_load, 0.0)),
                    dtype=np.float32,
                )
                grid_export = np.asarray(
                    info.get("grid_export_kw", np.maximum(-net_load, 0.0)),
                    dtype=np.float32,
                )
                controller_action_gap = np.asarray(
                    info.get("controller_action_gap", np.zeros(env.n, dtype=np.float32)),
                    dtype=np.float32,
                )
                battery_action_req = np.asarray(
                    info.get("battery_action_req", np.zeros(env.n, dtype=np.float32)),
                    dtype=np.float32,
                )
                battery_action_exec = np.asarray(
                    info.get("battery_action_exec", np.zeros(env.n, dtype=np.float32)),
                    dtype=np.float32,
                )
                pv_action_req = np.asarray(
                    info.get("pv_action_req", np.ones(env.n, dtype=np.float32)),
                    dtype=np.float32,
                )
                pv_action_exec = np.asarray(
                    info.get("pv_action_exec", info.get("pv_action", np.ones(env.n, dtype=np.float32))),
                    dtype=np.float32,
                )
                soc_penalty_unweighted = np.asarray(
                    info.get("soc_penalty_unweighted", info.get("action_penalty_unweighted", np.zeros(env.n, dtype=np.float32))),
                    dtype=np.float32,
                )
                soc_penalty = np.asarray(info.get("r_soc_pen", info.get("r_action_pen", np.zeros(env.n, dtype=np.float32))), dtype=np.float32)
                battery_power = np.asarray(info["e_bat"], dtype=np.float32)
                battery_power_req = np.asarray(
                    info.get("e_bat_req", battery_power),
                    dtype=np.float32,
                )
                battery_charge = np.clip(battery_power, 0.0, None).astype(np.float32)
                battery_discharge = np.maximum(-battery_power, 0.0).astype(np.float32)
                export_subsidy_per_agent = _export_subsidy_per_agent(
                    info,
                    float(env.dt),
                    float(getattr(cfg.reward, "export_subsidy_eur_per_kwh", 0.079)),
                )
                voltage_penalty_per_agent = np.asarray(
                    info.get("r_safe_v", np.zeros(env.n, dtype=np.float32)),
                    dtype=np.float32,
                )
                line_penalty_per_agent = np.asarray(
                    info.get("r_safe_line", np.zeros(env.n, dtype=np.float32)),
                    dtype=np.float32,
                )
                trafo_penalty_per_agent = np.asarray(
                    info.get("r_safe_trafo", np.zeros(env.n, dtype=np.float32)),
                    dtype=np.float32,
                )
                objective_per_agent = (
                    purchase_cost_per_agent
                    - export_subsidy_per_agent
                    + soc_penalty
                    + voltage_penalty_per_agent
                    + line_penalty_per_agent
                    + trafo_penalty_per_agent
                ).astype(np.float32)
                voltage_penalty_total = _mean_component_total(info, "r_safe_v", env.n)
                line_penalty_total = _mean_component_total(info, "r_safe_line", env.n)
                trafo_penalty_total = _mean_component_total(info, "r_safe_trafo", env.n)
                soc_penalty_total = float(np.sum(soc_penalty))
                purchase_cost_total = float(np.sum(purchase_cost_per_agent))
                export_subsidy_total = float(np.sum(export_subsidy_per_agent))
                step_rows.append(
                    {
                        "controller": label,
                        "episode_idx": episode_idx,
                        "step": step_in_episode,
                        "timestamp": timestamp,
                        "price": float(info["price"]),
                        "price_pred": float(price_pred),
                        "base_net_load_total": float(np.sum(base_net_load)),
                        "base_net_load_effective_total": float(np.sum(base_net_load_effective)),
                        "net_load_total": float(np.sum(net_load)),
                        "load_total": float(np.sum(np.asarray(info["load"], dtype=np.float32))),
                        "pv_raw_total": float(np.sum(pv_raw)),
                        "pv_effective_total": float(np.sum(pv_effective)),
                        "pv_curtail_total": float(np.sum(pv_curtail)),
                        "grid_import_total": float(np.sum(grid_import)),
                        "grid_export_total": float(np.sum(grid_export)),
                        "battery_charge_total": float(np.sum(battery_charge)),
                        "battery_discharge_total": float(np.sum(battery_discharge)),
                        "purchase_cost_total": purchase_cost_total,
                        "export_subsidy_total": export_subsidy_total,
                        "soc_penalty_total": soc_penalty_total,
                        "voltage_penalty_total": voltage_penalty_total,
                        "line_penalty_total": line_penalty_total,
                        "trafo_penalty_total": trafo_penalty_total,
                        "objective_total": (
                            purchase_cost_total
                            - export_subsidy_total
                            + soc_penalty_total
                            + voltage_penalty_total
                            + line_penalty_total
                            + trafo_penalty_total
                        ),
                        "voltage_violation_count": int(
                            ((np.asarray(info.get("vm_pu", []), dtype=np.float32) < float(cfg.grid.v_min_pu))
                            | (np.asarray(info.get("vm_pu", []), dtype=np.float32) > float(cfg.grid.v_max_pu))).sum()
                        ),
                        "controller_action_gap_total": float(np.sum(controller_action_gap)),
                        "soc_penalty_step_total": soc_penalty_total,
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
                            "pv_raw": float(pv_raw[agent_idx]),
                            "pv_effective": float(pv_effective[agent_idx]),
                            "pv_curtail": float(pv_curtail[agent_idx]),
                            "pv_utilization": float(pv_utilization[agent_idx]),
                            "pv_pred": float(np.asarray(pv_pred, dtype=np.float32)[agent_idx]),
                            "base_net_load": float(base_net_load[agent_idx]),
                            "base_net_load_effective": float(base_net_load_effective[agent_idx]),
                            "net_load": float(net_load[agent_idx]),
                            "grid_import_kw": float(grid_import[agent_idx]),
                            "grid_export_kw": float(grid_export[agent_idx]),
                            "e_bat": float(battery_power[agent_idx]),
                            "e_bat_req": float(battery_power_req[agent_idx]),
                            "battery_action_req": float(battery_action_req[agent_idx]),
                            "battery_action_exec": float(battery_action_exec[agent_idx]),
                            "pv_action_req": float(pv_action_req[agent_idx]),
                            "pv_action_exec": float(pv_action_exec[agent_idx]),
                            "controller_action_gap": float(controller_action_gap[agent_idx]),
                            "soc_penalty_unweighted": float(soc_penalty_unweighted[agent_idx]),
                            "r_soc_pen": float(soc_penalty[agent_idx]),
                            "r_safe_v": float(voltage_penalty_per_agent[agent_idx]),
                            "r_safe_line": float(line_penalty_per_agent[agent_idx]),
                            "r_safe_trafo": float(trafo_penalty_per_agent[agent_idx]),
                            "soc": float(np.asarray(info["soc_next"], dtype=np.float32)[agent_idx]),
                            "purchase_cost": float(purchase_cost_per_agent[agent_idx]),
                            "export_subsidy": float(export_subsidy_per_agent[agent_idx]),
                            "objective_total": float(objective_per_agent[agent_idx]),
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
            agent_df.groupby(["controller", "agent_profile"], as_index=False)[
                ["purchase_cost", "export_subsidy", "objective_total"]
            ].sum()
            if not agent_df.empty
            else pd.DataFrame(columns=["controller", "agent_profile", "purchase_cost", "export_subsidy", "objective_total"])
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
                "trafo_limit_kw": trafo_limit_kw,
                "export_subsidy_eur_per_kwh": float(getattr(cfg.reward, "export_subsidy_eur_per_kwh", 0.079)),
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
    label: str | None = None,
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
        label=label or f"DRL ({resolve_evaluation_mode(prediction_mode)})",
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


def compare_purchase_costs(*rollouts: RolloutResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rollout in rollouts:
        total_cost = float(rollout.agent_df["purchase_cost"].sum()) if not rollout.agent_df.empty else 0.0
        rows.append(
            {
                "controller": rollout.meta["controller"],
                "purchase_cost_total": total_cost,
            }
        )
    return pd.DataFrame(rows).sort_values("purchase_cost_total").reset_index(drop=True)


def summarize_rollout_metrics(rollout: RolloutResult) -> dict[str, object]:
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    summary_df = rollout.summary.copy()

    purchase_cost_total = float(step_df["purchase_cost_total"].sum()) if "purchase_cost_total" in step_df.columns else 0.0
    export_subsidy_total = (
        float(step_df["export_subsidy_total"].sum()) if "export_subsidy_total" in step_df.columns else 0.0
    )
    objective_total = float(step_df["objective_total"].sum()) if "objective_total" in step_df.columns else 0.0
    soc_penalty_total = (
        float(step_df["soc_penalty_total"].sum()) if "soc_penalty_total" in step_df.columns else 0.0
    )
    voltage_penalty_total = (
        float(step_df["voltage_penalty_total"].sum()) if "voltage_penalty_total" in step_df.columns else 0.0
    )
    line_penalty_total = float(step_df["line_penalty_total"].sum()) if "line_penalty_total" in step_df.columns else 0.0
    trafo_penalty_total = (
        float(step_df["trafo_penalty_total"].sum()) if "trafo_penalty_total" in step_df.columns else 0.0
    )
    price_mae = (
        float(np.abs(step_df["price"] - step_df["price_pred"]).mean())
        if not step_df.empty
        else float("nan")
    )
    load_mae = (
        float(np.abs(agent_df["load"] - agent_df["load_pred"]).mean())
        if not agent_df.empty
        else float("nan")
    )
    pv_mae = (
        float(np.abs(agent_df["pv"] - agent_df["pv_pred"]).mean())
        if not agent_df.empty
        else float("nan")
    )

    v_min = float(rollout.meta.get("v_min_pu", np.nan))
    v_max = float(rollout.meta.get("v_max_pu", np.nan))
    if grid_df.empty or not np.isfinite(v_min) or not np.isfinite(v_max):
        voltage_violation_bus_points = 0
        voltage_violation_steps = 0
        min_vm_pu = float("nan")
        max_vm_pu = float("nan")
    else:
        violation_mask = (grid_df["vm_pu"] < v_min) | (grid_df["vm_pu"] > v_max)
        voltage_violation_bus_points = int(violation_mask.sum())
        voltage_violation_steps = int(
            grid_df.loc[violation_mask, ["episode_idx", "step"]].drop_duplicates().shape[0]
        )
        min_vm_pu = float(grid_df["vm_pu"].min())
        max_vm_pu = float(grid_df["vm_pu"].max())

    metrics = {
        "controller": str(rollout.meta.get("controller", "unknown")),
        "purchase_cost_total": purchase_cost_total,
        "export_subsidy_total": export_subsidy_total,
        "objective_total": objective_total,
        "soc_penalty_total": soc_penalty_total,
        "voltage_penalty_total": voltage_penalty_total,
        "trafo_penalty_total": trafo_penalty_total,
        "line_penalty_total": line_penalty_total,
        "voltage_violation_count": voltage_violation_bus_points,
        "price_mae": price_mae,
        "load_mae": load_mae,
        "pv_mae": pv_mae,
        "voltage_violation_steps": voltage_violation_steps,
        "voltage_violation_bus_points": voltage_violation_bus_points,
        "min_vm_pu": min_vm_pu,
        "max_vm_pu": max_vm_pu,
        "v_min_pu": v_min,
        "v_max_pu": v_max,
    }
    if not summary_df.empty:
        for row in summary_df.itertuples(index=False):
            profile = str(getattr(row, "agent_profile", "unknown")).strip().lower()
            safe_profile = "".join(ch if ch.isalnum() else "_" for ch in profile).strip("_") or "unknown"
            metrics[f"purchase_cost_{safe_profile}"] = float(getattr(row, "purchase_cost"))
    return metrics


def compare_rollout_metrics(*rollouts: RolloutResult) -> pd.DataFrame:
    rows = [summarize_rollout_metrics(rollout) for rollout in rollouts]
    if not rows:
        return pd.DataFrame(
            columns=[
                "controller",
                "purchase_cost_total",
                "export_subsidy_total",
                "objective_total",
                "soc_penalty_total",
                "voltage_penalty_total",
                "trafo_penalty_total",
                "line_penalty_total",
                "voltage_violation_count",
                "price_mae",
                "load_mae",
                "pv_mae",
                "voltage_violation_steps",
                "voltage_violation_bus_points",
                "min_vm_pu",
                "max_vm_pu",
                "v_min_pu",
                "v_max_pu",
            ]
        )
    return pd.DataFrame(rows)


def plot_rollout_dashboard(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] | None = None,
):
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    if step_df.empty or agent_df.empty:
        raise ValueError("Rollout is empty; nothing to plot.")

    agent_profiles = list(rollout.meta["agent_profiles"])
    n_agents = len(agent_profiles)
    main_axis_count = 6 + n_agents
    if figsize is None:
        figsize = (18.0, 2.8 * main_axis_count)

    figure, axes = plt.subplots(main_axis_count, 1, figsize=figsize, sharex=True)
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
    axes[0].set_title(f"Test Rollout Dashboard - {rollout.meta['controller']}")
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

    voltage_axis = axes[3]
    if grid_df.empty:
        raise ValueError("Rollout does not contain full-grid voltage traces.")

    muted_color = "#cbd5e1"
    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]

    for bus_id, frame in grid_df.loc[~grid_df["is_agent_bus"]].groupby("bus_id"):
        voltage_axis.plot(
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
        voltage_axis.plot(
            frame["timestamp"],
            frame["vm_pu"],
            color=highlight_palette[color_idx % len(highlight_palette)],
            linewidth=2.1,
            alpha=0.95,
            label=f"Agent bus {bus_id}",
            zorder=3,
        )

    voltage_axis.axhline(
        float(rollout.meta["v_min_pu"]),
        color="#dc2626",
        linestyle="--",
        linewidth=1.1,
        label="V min",
    )
    voltage_axis.axhline(
        float(rollout.meta["v_max_pu"]),
        color="#ea580c",
        linestyle="--",
        linewidth=1.1,
        label="V max",
    )
    voltage_axis.set_ylabel("Voltage [p.u.]")
    voltage_axis.grid(True, alpha=0.25)
    voltage_axis.legend(loc="upper right", ncol=2)

    net_load_axis = axes[4]
    net_load_axis.plot(
        step_df["timestamp"],
        step_df["base_net_load_total"],
        color="#111827",
        linewidth=1.6,
        label="Raw net load",
    )
    if "base_net_load_effective_total" in step_df.columns:
        net_load_axis.plot(
            step_df["timestamp"],
            step_df["base_net_load_effective_total"],
            color="#16a34a",
            linewidth=1.4,
            linestyle="-.",
            label="Post-curtail net load",
        )
    net_load_axis.plot(
        step_df["timestamp"],
        step_df["net_load_total"],
        color="#2563eb",
        linewidth=1.5,
        linestyle="--",
        label="Post-action net load",
    )
    trafo_limit_kw = rollout.meta.get("trafo_limit_kw")
    if trafo_limit_kw is not None:
        trafo_limit_value = float(trafo_limit_kw)
        if np.isfinite(trafo_limit_value) and trafo_limit_value > 0.0:
            net_load_axis.axhline(
                trafo_limit_value,
                color="#dc2626",
                linestyle=":",
                linewidth=1.2,
                label=f"Approx trafo +limit ({trafo_limit_value:.1f} kW)",
            )
            net_load_axis.axhline(
                -trafo_limit_value,
                color="#dc2626",
                linestyle=":",
                linewidth=1.2,
                label=f"Approx trafo -limit ({trafo_limit_value:.1f} kW)",
            )
    net_load_axis.set_ylabel("Net load")
    net_load_axis.grid(True, alpha=0.25)
    net_load_axis.legend(loc="upper right")

    pv_axis = axes[5]
    if "pv_effective_total" in step_df.columns:
        pv_axis.plot(
            step_df["timestamp"],
            step_df["pv_raw_total"],
            color="#ea580c",
            linewidth=1.5,
            label="Raw PV",
        )
        pv_axis.plot(
            step_df["timestamp"],
            step_df["pv_effective_total"],
            color="#16a34a",
            linewidth=1.5,
            linestyle="--",
            label="Effective PV",
        )
        pv_axis.bar(
            step_df["timestamp"],
            step_df["pv_curtail_total"],
            width=0.008,
            color="#dc2626",
            alpha=0.35,
            label="Curtailment",
        )
        pv_axis.set_ylabel("PV")
        pv_axis.grid(True, alpha=0.25)
        pv_axis.legend(loc="upper right")

    for agent_offset, profile in enumerate(agent_profiles, start=6):
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
    figure._dashboard_main_axes = list(axes)
    return figure


def plot_power_balance_bars(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] = (18.0, 4.8),
):
    step_df = rollout.step_df.copy()
    if step_df.empty:
        raise ValueError("Rollout is empty; nothing to plot.")

    required_columns = {
        "load_total",
        "battery_charge_total",
        "pv_effective_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
        "battery_discharge_total",
    }
    missing = sorted(required_columns.difference(step_df.columns))
    if missing:
        raise ValueError(f"Rollout step_df is missing required power-balance columns: {missing}")

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    timestamps = step_df["timestamp"]
    width = 0.008

    positive_specs = [
        ("load_total", "Load", "#111827"),
        ("battery_charge_total", "Charge", "#dc2626"),
        ("grid_export_total", "Grid export", "#f59e0b"),
    ]
    negative_specs = [
        ("pv_effective_total", "PV", "#16a34a"),
        ("grid_import_total", "Grid import", "#2563eb"),
        ("battery_discharge_total", "Discharge", "#7c3aed"),
    ]

    positive_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, label, color in positive_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        axis.bar(timestamps, values, width=width, bottom=positive_bottom, color=color, alpha=0.78, label=label)
        positive_bottom = positive_bottom + values

    negative_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, label, color in negative_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        axis.bar(timestamps, -values, width=width, bottom=negative_bottom, color=color, alpha=0.78, label=label)
        negative_bottom = negative_bottom - values

    curtailment = step_df["pv_curtail_total"].to_numpy(dtype=np.float32)
    axis.bar(
        timestamps,
        curtailment,
        width=width,
        color="none",
        edgecolor="#dc2626",
        linewidth=1.0,
        hatch="//",
        label="Curtailment loss",
    )

    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_title(f"Power Balance - {rollout.meta['controller']}")
    axis.set_ylabel("kW")
    axis.set_xlabel("Timestamp")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="upper right", ncol=3)
    figure.tight_layout()
    return figure


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


def plot_voltage_profile_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (18.0, max(3.2 * n_rows, 4.8)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    muted_color = "#cbd5e1"
    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]

    for axis, rollout in zip(axes, rollouts, strict=False):
        grid_df = rollout.grid_df.copy()
        if grid_df.empty:
            raise ValueError(f"Rollout '{rollout.meta.get('controller', 'unknown')}' has no grid_df.")
        for _, frame in grid_df.loc[~grid_df["is_agent_bus"]].groupby("bus_id"):
            axis.plot(frame["timestamp"], frame["vm_pu"], color=muted_color, linewidth=0.9, alpha=0.35, zorder=1)
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
        axis.set_ylabel("V [p.u.]")
        axis.set_title(str(rollout.meta["controller"]))
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", ncol=2)

    axes[-1].set_xlabel("Timestamp")
    figure.tight_layout()
    return figure


def plot_net_load_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (18.0, max(3.2 * n_rows, 4.8)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for axis, rollout in zip(axes, rollouts, strict=False):
        step_df = rollout.step_df.copy()
        if step_df.empty:
            raise ValueError(f"Rollout '{rollout.meta.get('controller', 'unknown')}' has no step_df.")
        axis.plot(step_df["timestamp"], step_df["base_net_load_total"], color="#111827", linewidth=1.6, label="Raw net load")
        if "base_net_load_effective_total" in step_df.columns:
            axis.plot(
                step_df["timestamp"],
                step_df["base_net_load_effective_total"],
                color="#16a34a",
                linewidth=1.4,
                linestyle="-.",
                label="Post-curtail net load",
            )
        axis.plot(
            step_df["timestamp"],
            step_df["net_load_total"],
            color="#2563eb",
            linewidth=1.5,
            linestyle="--",
            label="Post-action net load",
        )
        trafo_limit_kw = rollout.meta.get("trafo_limit_kw")
        if trafo_limit_kw is not None:
            trafo_limit_value = float(trafo_limit_kw)
            if np.isfinite(trafo_limit_value) and trafo_limit_value > 0.0:
                axis.axhline(
                    trafo_limit_value,
                    color="#dc2626",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"Approx trafo +limit ({trafo_limit_value:.1f} kW)",
                )
                axis.axhline(
                    -trafo_limit_value,
                    color="#dc2626",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"Approx trafo -limit ({trafo_limit_value:.1f} kW)",
                )
        axis.set_ylabel("kW")
        axis.set_title(str(rollout.meta["controller"]))
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Timestamp")
    figure.tight_layout()
    return figure


def plot_power_balance_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (18.0, max(3.2 * n_rows, 4.8)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    width = 0.008
    positive_specs = [
        ("load_total", "Load", "#111827"),
        ("battery_charge_total", "Charge", "#dc2626"),
        ("grid_export_total", "Grid export", "#f59e0b"),
    ]
    negative_specs = [
        ("pv_effective_total", "PV", "#16a34a"),
        ("grid_import_total", "Grid import", "#2563eb"),
        ("battery_discharge_total", "Discharge", "#7c3aed"),
    ]

    for axis_idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df = rollout.step_df.copy()
        if step_df.empty:
            raise ValueError(f"Rollout '{rollout.meta.get('controller', 'unknown')}' has no step_df.")

        positive_bottom = np.zeros(len(step_df), dtype=np.float32)
        for column, label, color in positive_specs:
            values = step_df[column].to_numpy(dtype=np.float32)
            axis.bar(
                step_df["timestamp"],
                values,
                width=width,
                bottom=positive_bottom,
                color=color,
                alpha=0.78,
                label=label if axis_idx == 0 else None,
            )
            positive_bottom = positive_bottom + values

        negative_bottom = np.zeros(len(step_df), dtype=np.float32)
        for column, label, color in negative_specs:
            values = step_df[column].to_numpy(dtype=np.float32)
            axis.bar(
                step_df["timestamp"],
                -values,
                width=width,
                bottom=negative_bottom,
                color=color,
                alpha=0.78,
                label=label if axis_idx == 0 else None,
            )
            negative_bottom = negative_bottom - values

        axis.bar(
            step_df["timestamp"],
            step_df["pv_curtail_total"].to_numpy(dtype=np.float32),
            width=width,
            color="none",
            edgecolor="#dc2626",
            linewidth=1.0,
            hatch="//",
            label="Curtailment loss" if axis_idx == 0 else None,
        )
        axis.axhline(0.0, color="#111827", linewidth=1.0)
        axis.set_ylabel("kW")
        axis.set_title(str(rollout.meta["controller"]))
        axis.grid(True, axis="y", alpha=0.25)
        if axis_idx == 0:
            axis.legend(loc="upper right", ncol=4)

    axes[-1].set_xlabel("Timestamp")
    figure.tight_layout()
    return figure


def plot_purchase_cost_comparison(cost_df: pd.DataFrame, *, figsize: tuple[float, float] = (10.0, 4.5)):
    if cost_df.empty:
        raise ValueError("cost_df is empty; nothing to plot.")
    figure, axis = plt.subplots(1, 1, figsize=figsize)
    ordered = cost_df.sort_values("purchase_cost_total", ascending=True)
    axis.bar(ordered["controller"], ordered["purchase_cost_total"], color=["#2563eb", "#16a34a", "#dc2626"])
    axis.set_ylabel("Purchase Cost")
    axis.set_title("Purchase Cost Comparison")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_operating_cost_comparison(cost_df: pd.DataFrame, *, figsize: tuple[float, float] = (10.0, 4.5)):
    return plot_purchase_cost_comparison(cost_df, figsize=figsize)


def plot_rollout_comparison_dashboard(
    metrics_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (16.0, 9.0),
):
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; nothing to plot.")

    figure, axes = plt.subplots(2, 3, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    controllers = metrics_df["controller"].astype(str).tolist()
    x = np.arange(len(controllers))
    bar_palette = ["#2563eb", "#16a34a", "#dc2626", "#ea580c", "#7c3aed", "#0891b2"]
    colors = [bar_palette[idx % len(bar_palette)] for idx in range(len(controllers))]

    metric_specs = [
        ("purchase_cost_total", "Purchase Cost"),
        ("export_subsidy_total", "Export Subsidy"),
        ("objective_total", "Objective Total"),
        ("voltage_violation_count", "Voltage Violation Count"),
        ("trafo_penalty_total", "Transformer Penalty"),
        ("line_penalty_total", "Line Penalty"),
    ]
    for axis, (column, title) in zip(axes, metric_specs, strict=False):
        axis.bar(x, metrics_df[column].astype(float).to_numpy(), color=colors)
        axis.set_xticks(x)
        axis.set_xticklabels(controllers, rotation=15, ha="right")
        axis.set_title(title)
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
    "compare_purchase_costs",
    "compare_rollout_metrics",
    "ensure_forecast_ready",
    "normalize_date_input",
    "normalize_prediction_mode",
    "load_training_run_bundle",
    "plot_purchase_cost_comparison",
    "plot_operating_cost_comparison",
    "plot_net_load_comparison",
    "plot_power_balance_bars",
    "plot_power_balance_comparison",
    "plot_rollout_comparison_dashboard",
    "plot_rollout_dashboard",
    "plot_test_rollout",
    "plot_test_voltage_profile",
    "plot_voltage_profile_comparison",
    "resolve_evaluation_mode",
    "resolve_forecast_backend",
    "resolve_prediction_mode_from_forecast_backend",
    "summarize_rollout_metrics",
    "validate_compare_model_bundles",
]
