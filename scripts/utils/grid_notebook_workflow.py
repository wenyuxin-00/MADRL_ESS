"""High-level workflow helpers for the grid training notebook."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from controllers.action_feasibility import (
    build_safety_local_numpy,
    compute_action_gap_metrics_numpy,
    merge_action_info_into_step_info,
)
from controllers.mpc import gurobi_agent_mpc as single_agent_mpc_module
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
    resolved_type = resolve_forecast_backend(prediction_mode, comparison_cfg.env.future_horizon)
    if resolved_type != str(cfg.forecast.type):
        comparison_cfg.runtime.shared_data_dir = None
        comparison_cfg.runtime.shared_data_signature = None
        comparison_cfg.runtime.forecast_ready = None
    comparison_cfg.forecast.type = resolved_type
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


def _approx_trafo_limit_kw(env, *, loading_limit_pct: float | None = None) -> float | None:
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
        limit_scale = float(
            (getattr(getattr(env, "_grid_cfg", None), "line_max_loading_pct", 100.0) if loading_limit_pct is None else loading_limit_pct)
        ) / 100.0
        return float(np.sum(sn_mva) * max(limit_scale, 0.0) * 1000.0)
    except Exception:
        return None


def _fixed_feeder_components_kw(env) -> tuple[float, float]:
    net = getattr(getattr(env, "_grid_core", None), "net", None)
    if net is None:
        return 0.0, 0.0
    agent_bus_set = set(int(bus_id) for bus_id in getattr(getattr(env, "_grid_core", None), "agent_bus_ids", []))
    fixed_load_kw = 0.0
    fixed_generation_kw = 0.0
    try:
        load_table = getattr(net, "load", None)
        if load_table is not None and not load_table.empty and "bus" in load_table and "p_mw" in load_table:
            fixed_load_kw += float(
                np.asarray(load_table.loc[~load_table["bus"].isin(agent_bus_set), "p_mw"], dtype=np.float64).sum() * 1000.0
            )
        sgen_table = getattr(net, "sgen", None)
        if sgen_table is not None and not sgen_table.empty and "bus" in sgen_table and "p_mw" in sgen_table:
            fixed_generation_kw += float(
                np.asarray(sgen_table.loc[~sgen_table["bus"].isin(agent_bus_set), "p_mw"], dtype=np.float64).sum() * 1000.0
            )
    except Exception:
        return 0.0, 0.0
    return max(fixed_load_kw, 0.0), max(fixed_generation_kw, 0.0)


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


def _extract_safety_summary(train_result: Mapping[str, object] | None) -> dict[str, object]:
    payload = dict(train_result or {})
    if isinstance(payload.get("safety_summary"), Mapping):
        return dict(payload["safety_summary"])
    nested_result = payload.get("train_result")
    if isinstance(nested_result, Mapping) and isinstance(nested_result.get("safety_summary"), Mapping):
        return dict(nested_result["safety_summary"])
    return {}


def _extract_trafo_penalty_weight(train_result: Mapping[str, object] | None) -> float | None:
    payload = dict(train_result or {})
    if isinstance(payload.get("experiment_controls"), Mapping):
        reward_controls = dict(payload["experiment_controls"].get("reward_controls", {}))
        if "w_trafo_pen" in reward_controls:
            return float(reward_controls["w_trafo_pen"])
    nested_result = payload.get("train_result")
    if isinstance(nested_result, Mapping) and isinstance(nested_result.get("experiment_controls"), Mapping):
        reward_controls = dict(nested_result["experiment_controls"].get("reward_controls", {}))
        if "w_trafo_pen" in reward_controls:
            return float(reward_controls["w_trafo_pen"])
    return None


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
                "parallel_episode_sampling",
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
    export_subsidy_eur_per_kwh: float,
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
        export_subsidy_eur_per_kwh=export_subsidy_eur_per_kwh,
    )


def _ensure_single_agent_mpc_cache_state(env) -> tuple[dict[tuple[object, ...], object], dict[str, float]]:
    cache = getattr(env, "_single_agent_mpc_solver_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, "_single_agent_mpc_solver_cache", cache)
    stats = getattr(env, "_single_agent_mpc_stats", None)
    if not isinstance(stats, dict):
        stats = {}
        setattr(env, "_single_agent_mpc_stats", stats)
    stats.setdefault("solver_build_count", 0.0)
    stats.setdefault("solver_reuse_count", 0.0)
    stats.setdefault("solve_count", 0.0)
    stats.setdefault("solve_time_sec_total", 0.0)
    stats.setdefault("guarded_fallback_count", 0.0)
    return cache, stats


def _get_single_agent_mpc_solver(
    env,
    *,
    agent_idx: int,
    price_seq: np.ndarray,
    battery_capacity_kwh: float,
    p_max_kw: float,
    dt_hours: float,
    efficiency: float,
    soc_min: float,
    soc_max: float,
    export_subsidy_eur_per_kwh: float,
):
    cache, stats = _ensure_single_agent_mpc_cache_state(env)
    use_guarded_fallback = False
    horizon = int(np.asarray(price_seq, dtype=np.float32).reshape(-1).size)
    cache_key = (
        int(agent_idx),
        int(horizon),
        float(battery_capacity_kwh),
        float(p_max_kw),
        float(dt_hours),
        float(efficiency),
        float(soc_min),
        float(soc_max),
        float(export_subsidy_eur_per_kwh),
        bool(use_guarded_fallback),
    )
    solver = cache.get(cache_key)
    if solver is None:
        solver = single_agent_mpc_module._ReusableSingleAgentMPCSolver(
            horizon=horizon,
            battery_capacity_kwh=float(battery_capacity_kwh),
            p_max_kw=float(p_max_kw),
            dt_hours=float(dt_hours),
            efficiency=float(efficiency),
            soc_min=float(soc_min),
            soc_max=float(soc_max),
            export_subsidy_eur_per_kwh=float(export_subsidy_eur_per_kwh),
            use_guarded_fallback=bool(use_guarded_fallback),
        )
        cache[cache_key] = solver
        stats["solver_build_count"] = float(stats["solver_build_count"]) + 1.0
    else:
        stats["solver_reuse_count"] = float(stats["solver_reuse_count"]) + 1.0
    return solver, stats


def _mpc_policy(env, obs: dict[str, np.ndarray]) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    requested_battery_kw = np.zeros((env.n,), dtype=np.float32)
    price_seq = np.asarray(obs["price_seq"], dtype=np.float32)
    load_seq = np.asarray(obs["load_seq"], dtype=np.float32)
    pv_seq = np.asarray(obs["pv_seq"], dtype=np.float32)
    export_subsidy_eur_per_kwh = float(getattr(env.reward_fn, "export_subsidy_eur_per_kwh", 0.079))
    for agent_idx in range(env.n):
        solver, solver_stats = _get_single_agent_mpc_solver(
            env,
            agent_idx=agent_idx,
            price_seq=price_seq,
            battery_capacity_kwh=float(env.agent_c_bat[agent_idx]),
            p_max_kw=float(env.agent_p_max[agent_idx]),
            dt_hours=float(env.dt),
            efficiency=float(env.eff),
            soc_min=float(env.soc_min),
            soc_max=float(env.soc_max),
            export_subsidy_eur_per_kwh=export_subsidy_eur_per_kwh,
        )
        solve_result = solver.solve_full_horizon(
            price_seq=price_seq,
            load_seq=load_seq[agent_idx],
            pv_seq=pv_seq[agent_idx],
            soc=float(env.soc[agent_idx]),
            pv_curtail_upper_kw=np.zeros_like(pv_seq[agent_idx], dtype=np.float32),
        )
        if not bool(solve_result.feasible):
            raise RuntimeError(
                "Local MPC returned an infeasible or unbounded full-horizon solution "
                f"for agent {agent_idx}. This usually means the relaxed continuous "
                "local MPC formulation is not properly bounded on the current price window."
            )
        solver_stats["solve_count"] = float(solver_stats["solve_count"]) + 1.0
        solver_stats["solve_time_sec_total"] = (
            float(solver_stats["solve_time_sec_total"]) + float(solve_result.solve_time_sec)
        )
        if solve_result.used_guarded_fallback:
            solver_stats["guarded_fallback_count"] = (
                float(solver_stats["guarded_fallback_count"]) + 1.0
            )
        requested_battery_kw[agent_idx] = (
            np.float32(solve_result.signed_battery_kw[0]) if solve_result.signed_battery_kw.size else np.float32(0.0)
        )
    if hasattr(env, "get_signal_step"):
        actions, action_array = _assemble_global_oracle_actions(
            env,
            battery_power_kw=requested_battery_kw,
            pv_curtail_kw=np.zeros((env.n,), dtype=np.float32),
        )
    else:
        clipped_battery_kw = _clip_global_oracle_battery_power_kw(env, requested_battery_kw)
        p_max_kw = np.maximum(np.asarray(env.agent_p_max, dtype=np.float32), 1e-6)
        battery_action = np.clip(clipped_battery_kw / p_max_kw, -1.0, 1.0).astype(np.float32)
        action_array = np.stack(
            [battery_action, np.ones((env.n,), dtype=np.float32)],
            axis=-1,
        ).astype(np.float32)
        actions = [action_array[agent_idx].copy() for agent_idx in range(env.n)]
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


def _clip_global_oracle_battery_power_kw(env, battery_power_kw: np.ndarray) -> np.ndarray:
    battery_power_kw = np.asarray(battery_power_kw, dtype=np.float32)
    e_t = np.asarray(env.soc, dtype=np.float32) * np.asarray(env.agent_c_bat, dtype=np.float32)
    e_min = float(env.soc_min) * np.asarray(env.agent_c_bat, dtype=np.float32)
    e_max = float(env.soc_max) * np.asarray(env.agent_c_bat, dtype=np.float32)
    p_max = np.asarray(env.agent_p_max, dtype=np.float32)
    eff = max(float(env.eff), 1e-6)
    p_max_charge = np.minimum(p_max, np.maximum(0.0, (e_max - e_t) / (eff * float(env.dt))))
    p_max_discharge = np.minimum(p_max, np.maximum(0.0, (e_t - e_min) * eff / float(env.dt)))
    return np.clip(battery_power_kw, -p_max_discharge, p_max_charge).astype(np.float32)


def _assemble_global_oracle_actions(
    env,
    battery_power_kw: np.ndarray,
    pv_curtail_kw: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    clipped_battery_kw = _clip_global_oracle_battery_power_kw(env, battery_power_kw)
    pv_raw_kw = np.maximum(np.asarray(env.get_signal_step("pv"), dtype=np.float32), 0.0)
    clipped_curtail_kw = np.minimum(np.maximum(np.asarray(pv_curtail_kw, dtype=np.float32), 0.0), pv_raw_kw)
    p_max_kw = np.maximum(np.asarray(env.agent_p_max, dtype=np.float32), 1e-6)

    battery_action = np.clip(clipped_battery_kw / p_max_kw, -1.0, 1.0).astype(np.float32)
    pv_utilization = np.ones_like(pv_raw_kw, dtype=np.float32)
    valid_mask = pv_raw_kw > 1e-6
    pv_utilization[valid_mask] = 1.0 - (clipped_curtail_kw[valid_mask] / pv_raw_kw[valid_mask])
    pv_action = np.clip(2.0 * pv_utilization - 1.0, -1.0, 1.0).astype(np.float32)

    action_array = np.stack([battery_action, pv_action], axis=-1).astype(np.float32)
    return [action_array[agent_idx].copy() for agent_idx in range(env.n)], action_array


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
    controller_builder=None,
    action_fn=None,
) -> RolloutResult:
    provided = int(controller is not None) + int(controller_builder is not None) + int(action_fn is not None)
    if provided != 1:
        raise ValueError("Provide exactly one of controller, controller_builder, or action_fn.")

    from scripts.builder import build_env

    if cache_only_forecast_enabled(cfg):
        cfg.runtime.forecast_ready = None
    else:
        cfg.runtime.forecast_ready = ensure_forecast_ready(cfg)
    env = build_env(cfg, mode="test")
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(env._grid_core, "agent_bus_ids", cfg.grid.agent_bus_ids)]
    agent_bus_set = set(agent_bus_ids)
    loading_limit_pct = float(cfg.grid.line_max_loading_pct)
    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_kw(env)
    trafo_limit_kw = _approx_trafo_limit_kw(env, loading_limit_pct=loading_limit_pct)
    try:
        active_controller = controller_builder(env) if controller_builder is not None else controller
        for episode_idx in range(env.num_available_episodes):
            obs, reset_info = env.reset(episode_idx=episode_idx)
            raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
            if active_controller is not None:
                active_controller.reset()
            previous_raw_obs = None
            done = False
            step_in_episode = 0

            while not done:
                price_pred = _aligned_prediction(previous_raw_obs, raw_obs, "price_seq")
                load_pred = _aligned_prediction(previous_raw_obs, raw_obs, "load_seq")
                pv_pred = _aligned_prediction(previous_raw_obs, raw_obs, "pv_seq")
                timestamp = _step_timestamp(reset_info, step_in_episode)

                if active_controller is not None:
                    actions = active_controller.act(obs, deterministic=True)
                    action_info = getattr(active_controller, "last_action_info", None)
                else:
                    action_result = action_fn(env, raw_obs)
                    if isinstance(action_result, tuple) and len(action_result) == 2:
                        actions, action_info = action_result
                    else:
                        actions = action_result
                        action_info = None

                next_obs, reward, terminated, truncated, info = env.step(actions)
                reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
                apply_action_penalty = bool(getattr(active_controller, "apply_action_penalty", False))
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
                battery_charge_req = np.clip(battery_power_req, 0.0, None).astype(np.float32)
                battery_discharge_req = np.maximum(-battery_power_req, 0.0).astype(np.float32)
                pv_curtail_req = np.asarray(
                    info.get("pv_curtail_req", pv_curtail),
                    dtype=np.float32,
                )
                battery_request_gap_kw = np.abs(battery_power_req - battery_power).astype(np.float32)
                pv_curtail_request_gap_kw = np.abs(pv_curtail_req - pv_curtail).astype(np.float32)
                line_loading_pct = np.asarray(info.get("line_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
                trafo_loading_pct = np.asarray(info.get("trafo_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
                misocp_fallback = float(info.get("misocp_fallback", 0.0))
                misocp_time_limit_feasible = float(info.get("misocp_time_limit_feasible", 0.0))
                solve_time_sec = float(info.get("solve_time_sec", np.nan))
                root_import_kw = float(info.get("root_import_kw", np.nan))
                root_export_kw = float(info.get("root_export_kw", np.nan))
                simultaneous_charge_discharge_kw_total = float(info.get("simultaneous_charge_discharge_kw_total", 0.0))
                simultaneous_agent_count = int(float(info.get("simultaneous_agent_count", 0.0)))
                simultaneous_step_flag = float(info.get("simultaneous_step_flag", 0.0))
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
                pp_root_p_kw = float(
                    np.asarray(info.get("trafo_p_signed_kw", np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1).sum()
                )
                agent_raw_net_load_kw = float(np.sum(base_net_load))
                agent_effective_net_load_kw = float(np.sum(base_net_load_effective))
                agent_post_action_net_load_kw = float(np.sum(net_load))
                feeder_raw_net_load_kw = float(agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_effective_net_load_kw = float(agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_post_action_net_load_kw = float(agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw)
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
                        "agent_raw_net_load_kw": agent_raw_net_load_kw,
                        "agent_effective_net_load_kw": agent_effective_net_load_kw,
                        "agent_post_action_net_load_kw": agent_post_action_net_load_kw,
                        "fixed_load_kw": fixed_load_kw,
                        "fixed_generation_kw": fixed_generation_kw,
                        "feeder_raw_net_load_kw": feeder_raw_net_load_kw,
                        "feeder_effective_net_load_kw": feeder_effective_net_load_kw,
                        "feeder_post_action_net_load_kw": feeder_post_action_net_load_kw,
                        "load_total": float(np.sum(np.asarray(info["load"], dtype=np.float32))),
                        "pv_raw_total": float(np.sum(pv_raw)),
                        "pv_effective_total": float(np.sum(pv_effective)),
                        "pv_curtail_total": float(np.sum(pv_curtail)),
                        "grid_import_total": float(np.sum(grid_import)),
                        "grid_export_total": float(np.sum(grid_export)),
                        "battery_charge_total": float(np.sum(battery_charge)),
                        "battery_charge_req_total": float(np.sum(battery_charge_req)),
                        "battery_discharge_total": float(np.sum(battery_discharge)),
                        "battery_discharge_req_total": float(np.sum(battery_discharge_req)),
                        "pv_curtail_req_total": float(np.sum(pv_curtail_req)),
                        "battery_request_gap_kw_total": float(np.sum(battery_request_gap_kw)),
                        "pv_curtail_request_gap_kw_total": float(np.sum(pv_curtail_request_gap_kw)),
                        "projector_adjustment_kw_total": float(
                            np.sum(battery_request_gap_kw) + np.sum(pv_curtail_request_gap_kw)
                        ),
                        "purchase_cost_total": purchase_cost_total,
                        "export_subsidy_total": export_subsidy_total,
                        "soc_penalty_total": soc_penalty_total,
                        "voltage_penalty_total": voltage_penalty_total,
                        "line_penalty_total": line_penalty_total,
                        "trafo_penalty_total": trafo_penalty_total,
                        "psi_v_raw": float(info.get("psi_v_raw", 0.0)),
                        "psi_line_raw": float(info.get("psi_line_raw", 0.0)),
                        "psi_trafo_raw": float(info.get("psi_trafo_raw", 0.0)),
                        "line_loading_pct_max": float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0,
                        "trafo_loading_pct_max": float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0,
                        "line_violation": float(info.get("line_violation", info.get("l_violation", 0.0))),
                        "trafo_violation": float(info.get("trafo_violation", 0.0)),
                        "n_line_violations": int(
                            info.get("n_line_violations", info.get("n_l_violations", int(np.any(line_loading_pct > loading_limit_pct))))
                        ),
                        "n_trafo_violations": int(
                            info.get("n_trafo_violations", info.get("n_t_violations", int(np.any(trafo_loading_pct > loading_limit_pct))))
                        ),
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
                        "misocp_fallback": misocp_fallback,
                        "misocp_time_limit_feasible": misocp_time_limit_feasible,
                        "solve_time_sec": solve_time_sec,
                        "root_import_kw": root_import_kw,
                        "root_export_kw": root_export_kw,
                        "simultaneous_charge_discharge_kw_total": simultaneous_charge_discharge_kw_total,
                        "simultaneous_agent_count": simultaneous_agent_count,
                        "simultaneous_step_flag": simultaneous_step_flag,
                        "trafo_limit_reference_kw": trafo_limit_kw,
                    }
                )

                last_diagnostic = getattr(active_controller, "last_diagnostic", None) if active_controller is not None else None
                if isinstance(last_diagnostic, dict) and last_diagnostic:
                    diagnostic_row = dict(last_diagnostic)
                    diagnostic_row["controller"] = label
                    diagnostic_row["episode_idx"] = episode_idx
                    diagnostic_row["step"] = step_in_episode
                    diagnostic_row["timestamp"] = timestamp
                    diagnostic_row["pp_vm_pu"] = np.asarray(info.get("vm_pu", np.zeros(0, dtype=np.float32)), dtype=np.float32).copy()
                    diagnostic_row["pp_line_loading_pct"] = line_loading_pct.copy()
                    diagnostic_row["pp_trafo_loading_pct"] = trafo_loading_pct.copy()
                    diagnostic_row["pp_root_p_kw"] = float(
                        np.asarray(info.get("trafo_p_signed_kw", np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)[0]
                    )
                    diagnostic_rows.append(diagnostic_row)

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
                            "pv_curtail_req": float(pv_curtail_req[agent_idx]),
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
                            "battery_request_gap_kw": float(battery_request_gap_kw[agent_idx]),
                            "pv_curtail_request_gap_kw": float(pv_curtail_request_gap_kw[agent_idx]),
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
                "trafo_limit_note": "Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.",
                "loading_limit_pct": loading_limit_pct,
                "trafo_loading_limit_pct": loading_limit_pct,
                "export_subsidy_eur_per_kwh": float(getattr(cfg.reward, "export_subsidy_eur_per_kwh", 0.079)),
                "import_price_adder_eur_per_kwh": float(
                    getattr(getattr(cfg, "reward", None), "import_price_adder_eur_per_kwh", 0.0)
                ),
                "single_agent_mpc_solver_build_count": int(
                    float(getattr(env, "_single_agent_mpc_stats", {}).get("solver_build_count", 0.0))
                ),
                "single_agent_mpc_solver_reuse_count": int(
                    float(getattr(env, "_single_agent_mpc_stats", {}).get("solver_reuse_count", 0.0))
                ),
                "single_agent_mpc_solve_count": int(
                    float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_count", 0.0))
                ),
                "single_agent_mpc_total_solve_time_sec": float(
                    getattr(env, "_single_agent_mpc_stats", {}).get("solve_time_sec_total", 0.0)
                ),
                "single_agent_mpc_avg_solve_time_sec": float(
                    (
                        float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_time_sec_total", 0.0))
                        / max(float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_count", 0.0)), 1.0)
                    )
                    if float(getattr(env, "_single_agent_mpc_stats", {}).get("solve_count", 0.0)) > 0.0
                    else 0.0
                ),
                "single_agent_mpc_guarded_fallback_count": int(
                    float(getattr(env, "_single_agent_mpc_stats", {}).get("guarded_fallback_count", 0.0))
                ),
                "controller_diagnostic_log": diagnostic_rows,
                "soc_mode": "reset",
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


def collect_global_mpc_rollout(cfg, *, prediction_mode: str, label: str | None = None) -> RolloutResult:
    from controllers.mpc import GlobalSOCPMPCController

    resolved_mode = normalize_prediction_mode(prediction_mode)
    if resolved_mode != PERFECT_PREDICTION_MODE:
        raise ValueError("Global SOCP-MPC currently supports only prediction_mode='perfect'.")
    comparison_cfg = build_comparison_cfg(cfg, prediction_mode=prediction_mode)
    rollout_label = label or f"Global SOCP-MPC ({resolve_evaluation_mode(resolved_mode)})"
    rollout = collect_controller_rollout(
        comparison_cfg,
        label=rollout_label,
        controller_builder=lambda env: GlobalSOCPMPCController(env, comparison_cfg),
    )
    return _attach_misocp_validation_meta(rollout)


def collect_global_full_horizon_rollout(
    cfg,
    *,
    label: str | None = None,
    time_limit_sec: float = 600.0,
) -> RolloutResult:
    from controllers.mpc.global_socp_mpc import (
        GurobiSolveConfig,
        GlobalMISOCPProblem,
        _FULL_HORIZON_TIME_LIMIT_SEC,
        _SIMULTANEOUS_THRESHOLD_RATIO,
        default_primary_solve_config,
        default_retry_solve_config,
    )
    from scripts.builder import build_env

    comparison_cfg = build_comparison_cfg(cfg, prediction_mode=PERFECT_PREDICTION_MODE)
    rollout_label = str(label) if label is not None else ""
    resolved_time_limit = float(time_limit_sec if time_limit_sec is not None else _FULL_HORIZON_TIME_LIMIT_SEC)

    if cache_only_forecast_enabled(comparison_cfg):
        comparison_cfg.runtime.forecast_ready = None
    else:
        comparison_cfg.runtime.forecast_ready = ensure_forecast_ready(comparison_cfg)

    env = build_env(comparison_cfg, mode="test")
    step_rows: list[dict[str, object]] = []
    agent_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    bus_ids = [int(bus_id) for bus_id in env._grid_core.net.bus.index.tolist()]
    agent_bus_ids = [int(bus_id) for bus_id in getattr(env._grid_core, "agent_bus_ids", comparison_cfg.grid.agent_bus_ids)]
    agent_bus_set = set(agent_bus_ids)
    loading_limit_pct = float(comparison_cfg.grid.line_max_loading_pct)
    fixed_load_kw, fixed_generation_kw = _fixed_feeder_components_kw(env)
    trafo_limit_kw = _approx_trafo_limit_kw(env, loading_limit_pct=loading_limit_pct)

    try:
        problem = GlobalMISOCPProblem.from_env(env, comparison_cfg)
        full_input = problem.build_full_horizon_input(env)
        export_subsidy = float(
            getattr(comparison_cfg.reward, "export_subsidy_eur_per_kwh", problem.export_subsidy_default)
        )
        primary_config = default_primary_solve_config()
        primary_config = GurobiSolveConfig(
            time_limit_sec=resolved_time_limit,
            mip_gap=float(primary_config.mip_gap),
            threads=primary_config.threads,
            presolve=primary_config.presolve,
            cuts=primary_config.cuts,
            heuristics=primary_config.heuristics,
            mip_focus=primary_config.mip_focus,
        )
        retry_config = default_retry_solve_config(primary_config)
        result = problem.solve_adaptive_full_horizon(
            full_input,
            export_subsidy=export_subsidy,
            verbose=False,
            primary_window_steps=int(full_input.horizon_steps),
            fallback_window_steps=96,
            solve_config=primary_config,
            retry_solve_config=retry_config,
            export_debug=True,
            debug_tag="adaptive_global_misocp",
        )
        solve_mode = str(result.solve_mode)
        if not result.has_solution:
            raise RuntimeError(
                "Global MISOCP failed to produce a feasible incumbent in both single_window and chunked_window modes. "
                f"Final status={result.status_label!r}, debug_artifacts={result.debug_artifacts}."
            )
        if not rollout_label:
            rollout_label = (
                "Global MISOCP (chunked, near-optimal)"
                if solve_mode == "chunked_window"
                else "Global MISOCP (single_window)"
            )

        carried_soc = np.asarray(full_input.soc_init, dtype=np.float32).copy()
        threshold_kw = (_SIMULTANEOUS_THRESHOLD_RATIO * np.asarray(env.agent_p_max, dtype=np.float32)).astype(np.float32)

        for episode_list_idx, episode_idx in enumerate(full_input.episode_indices.tolist()):
            obs, reset_info = env.reset(episode_idx=int(episode_idx))
            if episode_list_idx > 0:
                env.soc = carried_soc.copy()
                obs = env.obs_builder.build(env)
            raw_obs = env.obs_builder.build_raw(env) if hasattr(env.obs_builder, "build_raw") else obs
            previous_raw_obs = None
            episode_offset = int(full_input.episode_offsets[episode_list_idx])
            episode_length = int(full_input.episode_lengths[episode_list_idx])

            for step_in_episode in range(episode_length):
                global_step = episode_offset + step_in_episode
                timestamp = _step_timestamp(reset_info, step_in_episode)
                price_pred = _aligned_prediction(previous_raw_obs, raw_obs, "price_seq")
                load_pred = _aligned_prediction(previous_raw_obs, raw_obs, "load_seq")
                pv_pred = _aligned_prediction(previous_raw_obs, raw_obs, "pv_seq")

                battery_power_kw = (
                    (
                        np.asarray(result.battery_charge_mw[:, global_step], dtype=np.float32)
                        - np.asarray(result.battery_discharge_mw[:, global_step], dtype=np.float32)
                    )
                    * 1000.0
                ).astype(np.float32)
                pv_curtail_kw = (
                    np.asarray(result.pv_curtail_mw[:, global_step], dtype=np.float32) * 1000.0
                ).astype(np.float32)
                actions, action_array = _assemble_global_oracle_actions(env, battery_power_kw, pv_curtail_kw)

                action_info = compute_action_gap_metrics_numpy(
                    build_safety_local_numpy(
                        soc=np.asarray(env.soc, dtype=np.float32),
                        load_raw=np.asarray(env.get_signal_step("load"), dtype=np.float32),
                        pv_raw=np.asarray(env.get_signal_step("pv"), dtype=np.float32),
                        battery_capacity_kwh=np.asarray(env.agent_c_bat, dtype=np.float32),
                        p_max_kw=np.asarray(env.agent_p_max, dtype=np.float32),
                    ),
                    action_array,
                    action_array,
                )
                simultaneous_kw = np.asarray(result.simultaneous_charge_discharge_kw[:, global_step], dtype=np.float32)
                simultaneous_mask = simultaneous_kw > threshold_kw
                action_info.update(
                    {
                        "misocp_fallback": np.asarray(0.0, dtype=np.float32),
                        "misocp_time_limit_feasible": np.asarray(float(result.time_limit_feasible), dtype=np.float32),
                        "solve_time_sec": np.asarray(float(result.solve_time_sec), dtype=np.float32),
                        "root_import_kw": np.asarray(float(result.root_import_mw[global_step] * 1000.0), dtype=np.float32),
                        "root_export_kw": np.asarray(float(result.root_export_mw[global_step] * 1000.0), dtype=np.float32),
                        "simultaneous_charge_discharge_kw_total": np.asarray(float(np.sum(simultaneous_kw)), dtype=np.float32),
                        "simultaneous_agent_count": np.asarray(float(np.sum(simultaneous_mask)), dtype=np.float32),
                        "simultaneous_step_flag": np.asarray(float(np.any(simultaneous_mask)), dtype=np.float32),
                    }
                )

                next_obs, reward, terminated, truncated, info = env.step(actions)
                reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
                info, action_penalty = merge_action_info_into_step_info(
                    info,
                    action_info,
                    soc_pen_weight=float(comparison_cfg.reward.w_soc_pen),
                    apply_action_penalty=False,
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
                soc_penalty = np.asarray(
                    info.get("r_soc_pen", info.get("r_action_pen", np.zeros(env.n, dtype=np.float32))),
                    dtype=np.float32,
                )
                battery_power = np.asarray(info["e_bat"], dtype=np.float32)
                battery_power_req = np.asarray(info.get("e_bat_req", battery_power), dtype=np.float32)
                battery_charge = np.clip(battery_power, 0.0, None).astype(np.float32)
                battery_discharge = np.maximum(-battery_power, 0.0).astype(np.float32)
                battery_charge_req = np.clip(battery_power_req, 0.0, None).astype(np.float32)
                battery_discharge_req = np.maximum(-battery_power_req, 0.0).astype(np.float32)
                pv_curtail_req = np.asarray(info.get("pv_curtail_req", pv_curtail), dtype=np.float32)
                battery_request_gap_kw = np.abs(battery_power_req - battery_power).astype(np.float32)
                pv_curtail_request_gap_kw = np.abs(pv_curtail_req - pv_curtail).astype(np.float32)
                line_loading_pct = np.asarray(info.get("line_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
                trafo_loading_pct = np.asarray(info.get("trafo_loading_pct", np.zeros(0, dtype=np.float32)), dtype=np.float32)
                export_subsidy_per_agent = _export_subsidy_per_agent(info, float(env.dt), export_subsidy)
                voltage_penalty_per_agent = np.asarray(info.get("r_safe_v", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                line_penalty_per_agent = np.asarray(info.get("r_safe_line", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
                trafo_penalty_per_agent = np.asarray(info.get("r_safe_trafo", np.zeros(env.n, dtype=np.float32)), dtype=np.float32)
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
                pp_root_p_kw = float(
                    np.asarray(info.get("trafo_p_signed_kw", np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1).sum()
                )
                agent_raw_net_load_kw = float(np.sum(base_net_load))
                agent_effective_net_load_kw = float(np.sum(base_net_load_effective))
                agent_post_action_net_load_kw = float(np.sum(net_load))
                feeder_raw_net_load_kw = float(agent_raw_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_effective_net_load_kw = float(agent_effective_net_load_kw + fixed_load_kw - fixed_generation_kw)
                feeder_post_action_net_load_kw = float(agent_post_action_net_load_kw + fixed_load_kw - fixed_generation_kw)

                step_rows.append(
                    {
                        "controller": rollout_label,
                        "episode_idx": int(episode_idx),
                        "step": step_in_episode,
                        "timestamp": timestamp,
                        "price": float(info["price"]),
                        "price_pred": float(price_pred),
                        "base_net_load_total": agent_raw_net_load_kw,
                        "base_net_load_effective_total": agent_effective_net_load_kw,
                        "net_load_total": agent_post_action_net_load_kw,
                        "agent_raw_net_load_kw": agent_raw_net_load_kw,
                        "agent_effective_net_load_kw": agent_effective_net_load_kw,
                        "agent_post_action_net_load_kw": agent_post_action_net_load_kw,
                        "fixed_load_kw": fixed_load_kw,
                        "fixed_generation_kw": fixed_generation_kw,
                        "feeder_raw_net_load_kw": feeder_raw_net_load_kw,
                        "feeder_effective_net_load_kw": feeder_effective_net_load_kw,
                        "feeder_post_action_net_load_kw": feeder_post_action_net_load_kw,
                        "root_net_exchange_kw": pp_root_p_kw,
                        "pp_root_p_kw": pp_root_p_kw,
                        "load_total": float(np.sum(np.asarray(info["load"], dtype=np.float32))),
                        "pv_raw_total": float(np.sum(pv_raw)),
                        "pv_effective_total": float(np.sum(pv_effective)),
                        "pv_curtail_total": float(np.sum(pv_curtail)),
                        "grid_import_total": float(np.sum(grid_import)),
                        "grid_export_total": float(np.sum(grid_export)),
                        "battery_charge_total": float(np.sum(battery_charge)),
                        "battery_charge_req_total": float(np.sum(battery_charge_req)),
                        "battery_discharge_total": float(np.sum(battery_discharge)),
                        "battery_discharge_req_total": float(np.sum(battery_discharge_req)),
                        "pv_curtail_req_total": float(np.sum(pv_curtail_req)),
                        "battery_request_gap_kw_total": float(np.sum(battery_request_gap_kw)),
                        "pv_curtail_request_gap_kw_total": float(np.sum(pv_curtail_request_gap_kw)),
                        "projector_adjustment_kw_total": float(
                            np.sum(battery_request_gap_kw) + np.sum(pv_curtail_request_gap_kw)
                        ),
                        "purchase_cost_total": purchase_cost_total,
                        "export_subsidy_total": export_subsidy_total,
                        "soc_penalty_total": soc_penalty_total,
                        "voltage_penalty_total": voltage_penalty_total,
                        "line_penalty_total": line_penalty_total,
                        "trafo_penalty_total": trafo_penalty_total,
                        "psi_v_raw": float(info.get("psi_v_raw", 0.0)),
                        "psi_line_raw": float(info.get("psi_line_raw", 0.0)),
                        "psi_trafo_raw": float(info.get("psi_trafo_raw", 0.0)),
                        "line_loading_pct_max": float(np.max(line_loading_pct)) if line_loading_pct.size else 0.0,
                        "trafo_loading_pct_max": float(np.max(trafo_loading_pct)) if trafo_loading_pct.size else 0.0,
                        "line_violation": float(info.get("line_violation", info.get("l_violation", 0.0))),
                        "trafo_violation": float(info.get("trafo_violation", 0.0)),
                        "n_line_violations": int(
                            info.get("n_line_violations", info.get("n_l_violations", int(np.any(line_loading_pct > loading_limit_pct))))
                        ),
                        "n_trafo_violations": int(
                            info.get("n_trafo_violations", info.get("n_t_violations", int(np.any(trafo_loading_pct > loading_limit_pct))))
                        ),
                        "objective_total": (
                            purchase_cost_total
                            - export_subsidy_total
                            + soc_penalty_total
                            + voltage_penalty_total
                            + line_penalty_total
                            + trafo_penalty_total
                        ),
                        "voltage_violation_count": int(
                            ((np.asarray(info.get("vm_pu", []), dtype=np.float32) < float(comparison_cfg.grid.v_min_pu))
                            | (np.asarray(info.get("vm_pu", []), dtype=np.float32) > float(comparison_cfg.grid.v_max_pu))).sum()
                        ),
                        "controller_action_gap_total": float(np.sum(controller_action_gap)),
                        "soc_penalty_step_total": soc_penalty_total,
                        "misocp_fallback": 0.0,
                        "misocp_time_limit_feasible": float(result.time_limit_feasible),
                        "solve_time_sec": float(result.solve_time_sec),
                        "root_import_kw": float(result.root_import_mw[global_step] * 1000.0),
                        "root_export_kw": float(result.root_export_mw[global_step] * 1000.0),
                        "simultaneous_charge_discharge_kw_total": float(np.sum(simultaneous_kw)),
                        "simultaneous_agent_count": int(np.sum(simultaneous_mask)),
                        "simultaneous_step_flag": float(np.any(simultaneous_mask)),
                        "trafo_limit_reference_kw": trafo_limit_kw,
                    }
                )

                diagnostic_row = {
                    "solver_type": "gurobi_misocp",
                    "status_code": int(result.status_code),
                    "status_label": str(result.status_label),
                    "solve_mode": str(solve_mode),
                    "solve_time_sec": float(result.solve_time_sec),
                    "misocp_fallback": 0.0,
                    "misocp_time_limit_feasible": float(result.time_limit_feasible),
                    "mip_gap": float(result.mip_gap),
                    "best_bound": float(result.best_bound),
                    "objective_value": float(result.objective_value),
                    "agent_purchase_cost_eur": float(result.agent_purchase_cost_eur),
                    "agent_export_subsidy_eur": float(result.agent_export_subsidy_eur),
                    "agent_net_cost_eur": float(result.agent_net_cost_eur),
                    "feeder_purchase_cost_eur": float(result.feeder_purchase_cost_eur),
                    "feeder_export_subsidy_eur": float(result.feeder_export_subsidy_eur),
                    "feeder_net_cost_eur": float(result.feeder_net_cost_eur),
                    "throughput_regularization_eur": float(result.throughput_regularization_eur),
                    "throughput_regularization_weight": float(result.throughput_regularization_weight),
                    "root_import_kw": float(result.root_import_mw[global_step] * 1000.0),
                    "root_export_kw": float(result.root_export_mw[global_step] * 1000.0),
                    "root_p_kw": float(result.root_p_kw[global_step]),
                    "root_q_kvar": float(result.root_q_kvar[global_step]),
                    "misocp_vm_pu": np.asarray(result.bus_vm_pu[:, global_step], dtype=np.float32).copy(),
                    "misocp_line_loading_pct": np.asarray(result.line_loading_pct[:, global_step], dtype=np.float32).copy(),
                    "misocp_trafo_loading_pct": np.asarray(result.trafo_loading_pct[:, global_step], dtype=np.float32).copy(),
                    "model_size_num_vars": float(result.model_size.num_vars),
                    "model_size_num_binary_vars": float(result.model_size.num_binary_vars),
                    "model_size_num_linear_constraints": float(result.model_size.num_linear_constraints),
                    "model_size_num_quadratic_constraints": float(result.model_size.num_quadratic_constraints),
                    "simultaneous_step_ratio": float(result.simultaneous_step_ratio),
                    "max_simultaneous_kw": float(result.max_simultaneous_kw),
                    "simultaneous_agent_steps": float(result.simultaneous_agent_steps),
                    "debug_artifacts": dict(result.debug_artifacts),
                    "sanity_warning": str(result.sanity_warning or ""),
                    "controller": rollout_label,
                    "episode_idx": int(episode_idx),
                    "step": step_in_episode,
                    "timestamp": timestamp,
                    "pp_vm_pu": np.asarray(info.get("vm_pu", np.zeros(0, dtype=np.float32)), dtype=np.float32).copy(),
                    "pp_line_loading_pct": line_loading_pct.copy(),
                    "pp_trafo_loading_pct": trafo_loading_pct.copy(),
                    "pp_root_p_kw": float(
                        np.asarray(info.get("trafo_p_signed_kw", np.zeros(1, dtype=np.float32)), dtype=np.float32).reshape(-1)[0]
                    ),
                }
                diagnostic_rows.append(diagnostic_row)

                for agent_idx, profile in enumerate(comparison_cfg.data.agent_profiles):
                    agent_rows.append(
                        {
                            "controller": rollout_label,
                            "episode_idx": int(episode_idx),
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
                            "pv_curtail_req": float(pv_curtail_req[agent_idx]),
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
                            "battery_request_gap_kw": float(battery_request_gap_kw[agent_idx]),
                            "pv_curtail_request_gap_kw": float(pv_curtail_request_gap_kw[agent_idx]),
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
                                "controller": rollout_label,
                                "episode_idx": int(episode_idx),
                                "step": step_in_episode,
                                "timestamp": timestamp,
                                "bus_id": int(bus_id),
                                "vm_pu": float(vm_value),
                                "is_agent_bus": bool(int(bus_id) in agent_bus_set),
                            }
                        )

                previous_raw_obs = raw_obs
                obs = next_obs
                raw_obs = raw_next_obs

            carried_soc = np.asarray(env.soc, dtype=np.float32).copy()

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
        rollout = RolloutResult(
            step_df=step_df,
            agent_df=agent_df,
            grid_df=grid_df,
            summary=summary,
            meta={
                "controller": rollout_label,
                "n_agents": int(env.n),
                "agent_profiles": list(comparison_cfg.data.agent_profiles),
                "agent_bus_ids": agent_bus_ids,
                "bus_ids": bus_ids,
                "v_min_pu": float(comparison_cfg.grid.v_min_pu),
                "v_max_pu": float(comparison_cfg.grid.v_max_pu),
                "future_horizon": int(comparison_cfg.env.future_horizon),
                "prediction_mode": PERFECT_PREDICTION_MODE,
                "evaluation_mode": resolve_evaluation_mode(PERFECT_PREDICTION_MODE),
                "dt_hours": float(comparison_cfg.env.dt),
                "trafo_limit_kw": trafo_limit_kw,
                "trafo_limit_note": "Transformer apparent-power limit shown as an active-power-view reference; not a strict P bound when Q != 0.",
                "loading_limit_pct": loading_limit_pct,
                "trafo_loading_limit_pct": loading_limit_pct,
                "export_subsidy_eur_per_kwh": export_subsidy,
                "import_price_adder_eur_per_kwh": float(
                    getattr(getattr(comparison_cfg, "reward", None), "import_price_adder_eur_per_kwh", 0.0)
                ),
                "controller_diagnostic_log": diagnostic_rows,
                "soc_mode": "continuous",
                "solve_mode": str(solve_mode),
                "global_oracle_time_limit_sec": resolved_time_limit,
                "global_oracle_runtime_sec": float(result.solve_time_sec),
                "global_oracle_gap": float(result.mip_gap),
                "global_misocp_gap": float(result.mip_gap),
                "global_oracle_status_label": str(result.status_label),
                "global_oracle_debug_artifacts": dict(result.debug_artifacts),
                "is_near_optimal": bool(str(solve_mode) == "chunked_window"),
                "economics_scope": "agent_only",
                "agent_purchase_cost_eur": float(result.agent_purchase_cost_eur),
                "agent_export_subsidy_eur": float(result.agent_export_subsidy_eur),
                "agent_net_cost_eur": float(result.agent_net_cost_eur),
                "feeder_purchase_cost_eur": float(result.feeder_purchase_cost_eur),
                "feeder_export_subsidy_eur": float(result.feeder_export_subsidy_eur),
                "feeder_net_cost_eur": float(result.feeder_net_cost_eur),
                "episode_offsets": np.asarray(full_input.episode_offsets, dtype=np.int32).copy(),
                "episode_lengths": np.asarray(full_input.episode_lengths, dtype=np.int32).copy(),
            },
        )
        return _attach_misocp_validation_meta(rollout)
    finally:
        env.close()


def _attach_misocp_validation_meta(rollout: RolloutResult) -> RolloutResult:
    validation_df = _build_misocp_validation_df(rollout.meta.get("controller_diagnostic_log", []))
    fallback_ratio = (
        float(rollout.step_df["misocp_fallback"].mean())
        if not rollout.step_df.empty and "misocp_fallback" in rollout.step_df.columns
        else 0.0
    )
    simultaneous_step_ratio = (
        float(rollout.step_df["simultaneous_step_flag"].mean())
        if not rollout.step_df.empty and "simultaneous_step_flag" in rollout.step_df.columns
        else 0.0
    )
    health_warning = ""
    if fallback_ratio > 0.05:
        health_warning = (
            "Global SOCP-MPC fallback ratio exceeds 5%; hard-constrained steps were often infeasible "
            "or timed out without an incumbent."
        )
    if simultaneous_step_ratio > 0.05:
        suffix = (
            "Simultaneous charge/discharge exceeded 5% of steps; check subsidy-driven arbitrage and the tiny "
            "throughput regularization weight."
        )
        health_warning = f"{health_warning} {suffix}".strip()
    if not validation_df.empty and bool((~validation_df["within_tolerance"]).any()):
        suffix = "MISOCP-vs-pandapower validation exceeded at least one tolerance."
        health_warning = f"{health_warning} {suffix}".strip()
    rollout.meta["misocp_validation_df"] = validation_df
    rollout.meta["misocp_fallback_ratio"] = fallback_ratio
    rollout.meta["misocp_simultaneous_step_ratio"] = simultaneous_step_ratio
    rollout.meta["misocp_health_warning"] = health_warning
    return rollout


def _build_misocp_validation_df(diagnostic_rows: list[dict[str, object]]) -> pd.DataFrame:
    from controllers.mpc.global_socp_mpc import (
        _LINE_LOADING_ERR_TOL_PCT,
        _ROOT_POWER_ERR_TOL_KW,
        _TRAFO_LOADING_ERR_TOL_PCT,
        _VOLTAGE_ERR_TOL_PU,
    )

    rows: list[dict[str, object]] = []
    for entry in diagnostic_rows:
        misocp_vm = entry.get("misocp_vm_pu")
        pp_vm = entry.get("pp_vm_pu")
        misocp_line = entry.get("misocp_line_loading_pct")
        pp_line = entry.get("pp_line_loading_pct")
        misocp_trafo = entry.get("misocp_trafo_loading_pct")
        pp_trafo = entry.get("pp_trafo_loading_pct")
        if misocp_vm is None or misocp_line is None or misocp_trafo is None:
            max_vm_abs_err = np.nan
            max_line_abs_err = np.nan
            trafo_abs_err = np.nan
            root_p_abs_err = np.nan
            within_tolerance = False
        else:
            max_vm_abs_err = float(
                np.max(np.abs(np.asarray(misocp_vm, dtype=np.float32) - np.asarray(pp_vm, dtype=np.float32)))
            )
            max_line_abs_err = float(
                np.max(np.abs(np.asarray(misocp_line, dtype=np.float32) - np.asarray(pp_line, dtype=np.float32)))
            )
            trafo_abs_err = float(
                np.max(np.abs(np.asarray(misocp_trafo, dtype=np.float32) - np.asarray(pp_trafo, dtype=np.float32)))
            )
            root_p_abs_err = float(
                abs(float(entry.get("root_p_kw", np.nan)) - float(entry.get("pp_root_p_kw", np.nan)))
            )
            within_tolerance = bool(
                max_vm_abs_err < _VOLTAGE_ERR_TOL_PU
                and max_line_abs_err < _LINE_LOADING_ERR_TOL_PCT
                and trafo_abs_err < _TRAFO_LOADING_ERR_TOL_PCT
                and root_p_abs_err < _ROOT_POWER_ERR_TOL_KW
            )

        rows.append(
            {
                "controller": entry.get("controller", "Global SOCP-MPC"),
                "episode_idx": int(entry.get("episode_idx", 0)),
                "step": int(entry.get("step", 0)),
                "timestamp": entry.get("timestamp"),
                "misocp_fallback": float(entry.get("misocp_fallback", 0.0)),
                "misocp_time_limit_feasible": float(entry.get("misocp_time_limit_feasible", 0.0)),
                "solve_time_sec": float(entry.get("solve_time_sec", np.nan)),
                "max_vm_abs_err_pu": max_vm_abs_err,
                "max_line_loading_abs_err_pct": max_line_abs_err,
                "trafo_loading_abs_err_pct": trafo_abs_err,
                "root_p_abs_err_kw": root_p_abs_err,
                "within_tolerance": within_tolerance,
            }
        )
    return pd.DataFrame(rows)


def plot_global_misocp_validation(
    rollout: RolloutResult,
    *,
    figsize: tuple[float, float] = (11.0, 8.0),
):
    validation_df = rollout.meta.get("misocp_validation_df")
    if not isinstance(validation_df, pd.DataFrame) or validation_df.empty:
        raise ValueError("Rollout does not contain MISOCP-vs-pandapower validation data.")

    from controllers.mpc.global_socp_mpc import (
        _LINE_LOADING_ERR_TOL_PCT,
        _ROOT_POWER_ERR_TOL_KW,
        _TRAFO_LOADING_ERR_TOL_PCT,
        _VOLTAGE_ERR_TOL_PU,
    )

    figure, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
    series = [
        ("max_vm_abs_err_pu", _VOLTAGE_ERR_TOL_PU, "Max |V_misocp - V_pp| [p.u.]"),
        ("max_line_loading_abs_err_pct", _LINE_LOADING_ERR_TOL_PCT, "Max |Line Loading| Error [pct-point]"),
        ("trafo_loading_abs_err_pct", _TRAFO_LOADING_ERR_TOL_PCT, "|Trafo Loading| Error [pct-point]"),
        ("root_p_abs_err_kw", _ROOT_POWER_ERR_TOL_KW, "|P_root| Error [kW]"),
    ]
    timestamps = validation_df["timestamp"]
    for axis, (column, threshold, ylabel) in zip(axes, series, strict=False):
        axis.plot(timestamps, validation_df[column], color="#1d4ed8", linewidth=1.8)
        axis.axhline(float(threshold), color="#dc2626", linestyle="--", linewidth=1.2)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    warning = str(rollout.meta.get("misocp_health_warning", "") or "")
    title = f"{rollout.meta.get('controller', 'Global SOCP-MPC')} Validation"
    if warning:
        title = f"{title}\n{warning}"
    axes[0].set_title(title)
    axes[-1].set_xlabel("Timestamp")
    return figure


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


def _step_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ["episode_idx", "step"] if column in frame.columns]
    if columns:
        return columns
    if "timestamp" in frame.columns:
        return ["timestamp"]
    return []


def _sort_columns(frame: pd.DataFrame, *preferred: str) -> list[str]:
    return [column for column in preferred if column in frame.columns]


def _prepare_agent_power_balance_frame(
    step_df: pd.DataFrame,
    *,
    controller_label: str,
    tolerance_kw: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    if step_df.empty:
        raise ValueError(f"Rollout '{controller_label}' has no step_df.")
    prepared = step_df.copy()
    required_columns = {
        "load_total",
        "battery_charge_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
        "battery_discharge_total",
    }
    missing = sorted(required_columns.difference(prepared.columns))
    if missing:
        raise ValueError(
            f"Rollout '{controller_label}' is missing required agent power-balance columns: {missing}"
        )
    if "pv_raw_total" not in prepared.columns:
        if "pv_effective_total" not in prepared.columns:
            raise ValueError(
                f"Rollout '{controller_label}' must contain either pv_raw_total or pv_effective_total."
            )
        prepared["pv_raw_total"] = (
            prepared["pv_effective_total"].to_numpy(dtype=np.float32)
            + prepared["pv_curtail_total"].to_numpy(dtype=np.float32)
        )

    positive_total = (
        prepared["load_total"].to_numpy(dtype=np.float32)
        + prepared["battery_charge_total"].to_numpy(dtype=np.float32)
        + prepared["grid_export_total"].to_numpy(dtype=np.float32)
        + prepared["pv_curtail_total"].to_numpy(dtype=np.float32)
    )
    negative_total = (
        prepared["pv_raw_total"].to_numpy(dtype=np.float32)
        + prepared["grid_import_total"].to_numpy(dtype=np.float32)
        + prepared["battery_discharge_total"].to_numpy(dtype=np.float32)
    )
    balance_residual_kw = positive_total - negative_total
    max_abs_balance_residual_kw = float(np.max(np.abs(balance_residual_kw))) if balance_residual_kw.size else 0.0
    if max_abs_balance_residual_kw > float(tolerance_kw):
        raise ValueError(
            f"Rollout '{controller_label}' violates agent-only power balance by "
            f"{max_abs_balance_residual_kw:.3f} kW (tolerance={float(tolerance_kw):.3f} kW)."
        )
    prepared["agent_power_balance_residual_kw"] = balance_residual_kw.astype(np.float32)
    return prepared, max_abs_balance_residual_kw


def _prepare_compare_net_load_frame(
    step_df: pd.DataFrame,
    *,
    controller_label: str,
) -> pd.DataFrame:
    if step_df.empty:
        raise ValueError(f"Rollout '{controller_label}' has no step_df.")
    prepared = step_df.copy()

    if not {"agent_raw_net_load_kw", "agent_effective_net_load_kw", "agent_post_action_net_load_kw"}.issubset(
        prepared.columns
    ):
        if not {"base_net_load_total", "net_load_total"}.issubset(prepared.columns):
            raise ValueError(
                f"Rollout '{controller_label}' is missing agent net-load columns required for compare plotting."
            )
        prepared["agent_raw_net_load_kw"] = prepared["base_net_load_total"].to_numpy(dtype=np.float32)
        prepared["agent_effective_net_load_kw"] = np.asarray(
            prepared.get("base_net_load_effective_total", prepared["base_net_load_total"]),
            dtype=np.float32,
        )
        prepared["agent_post_action_net_load_kw"] = prepared["net_load_total"].to_numpy(dtype=np.float32)

    if not {"feeder_raw_net_load_kw", "feeder_effective_net_load_kw", "feeder_post_action_net_load_kw"}.issubset(
        prepared.columns
    ):
        if not {"fixed_load_kw", "fixed_generation_kw"}.issubset(prepared.columns):
            raise ValueError(
                f"Rollout '{controller_label}' is missing feeder-total net-load columns required for compare plotting."
            )
        fixed_load = prepared["fixed_load_kw"].to_numpy(dtype=np.float32)
        fixed_generation = prepared["fixed_generation_kw"].to_numpy(dtype=np.float32)
        prepared["feeder_raw_net_load_kw"] = (
            prepared["agent_raw_net_load_kw"].to_numpy(dtype=np.float32) + fixed_load - fixed_generation
        )
        prepared["feeder_effective_net_load_kw"] = (
            prepared["agent_effective_net_load_kw"].to_numpy(dtype=np.float32) + fixed_load - fixed_generation
        )
        prepared["feeder_post_action_net_load_kw"] = (
            prepared["agent_post_action_net_load_kw"].to_numpy(dtype=np.float32) + fixed_load - fixed_generation
        )

    return prepared


def _summarize_voltage_step_delta(grid_df: pd.DataFrame) -> tuple[float, float]:
    if grid_df.empty or "vm_pu" not in grid_df.columns or "bus_id" not in grid_df.columns:
        return float("nan"), float("nan")
    sort_columns = _sort_columns(grid_df, "episode_idx", "bus_id", "step", "timestamp")
    group_columns = [column for column in ["episode_idx", "bus_id"] if column in grid_df.columns]
    if "bus_id" not in group_columns:
        group_columns.append("bus_id")
    ordered = grid_df.sort_values(sort_columns) if sort_columns else grid_df.copy()
    step_delta = ordered.groupby(group_columns, sort=False)["vm_pu"].diff().abs()
    finite = step_delta.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 95.0)), float(np.max(finite))


def _summarize_voltage_spread(grid_df: pd.DataFrame) -> tuple[float, float]:
    if grid_df.empty or "vm_pu" not in grid_df.columns:
        return float("nan"), float("nan")
    group_columns = _step_group_columns(grid_df)
    if not group_columns:
        return float("nan"), float("nan")
    spread = grid_df.groupby(group_columns, sort=False)["vm_pu"].agg(lambda values: float(np.max(values) - np.min(values)))
    values = spread.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.max(values))


def _summarize_grouped_ramp(step_df: pd.DataFrame, column: str) -> tuple[float, float]:
    if step_df.empty or column not in step_df.columns:
        return float("nan"), float("nan")
    sort_columns = _sort_columns(step_df, "episode_idx", "step", "timestamp")
    ordered = step_df.sort_values(sort_columns) if sort_columns else step_df.copy()
    group_columns = [column_name for column_name in ["episode_idx"] if column_name in ordered.columns]
    if group_columns:
        ramp = ordered.groupby(group_columns, sort=False)[column].diff().abs()
    else:
        ramp = ordered[column].diff().abs()
    values = ramp.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.max(values))


def _collect_compare_optional_missing_fields(rollout: RolloutResult) -> list[str]:
    missing: list[str] = []
    if "trafo_loading_pct_max" not in rollout.step_df.columns:
        missing.append("step_df.trafo_loading_pct_max")
    if "n_trafo_violations" not in rollout.step_df.columns:
        missing.append("step_df.n_trafo_violations")
    if "soc" not in rollout.agent_df.columns:
        missing.append("agent_df.soc")
    return missing


def _safe_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_series = numerator.astype(np.float32)
    denominator_series = denominator.astype(np.float32)
    safe_denominator = denominator_series.where(denominator_series.abs() > 1e-6, np.nan)
    return numerator_series / safe_denominator


def _build_trafo_cause_hint(row: pd.Series) -> str:
    regime = str(row.get("dominant_regime", "mixed_or_balanced"))
    raw_export = max(float(row.get("raw_export_total", 0.0)), 0.0)
    raw_import = max(float(row.get("raw_import_total", 0.0)), 0.0)
    grid_export = max(float(row.get("grid_export_total", 0.0)), 0.0)
    grid_import = max(float(row.get("grid_import_total", 0.0)), 0.0)
    battery_charge = max(float(row.get("battery_charge_total", 0.0)), 0.0)
    battery_discharge = max(float(row.get("battery_discharge_total", 0.0)), 0.0)
    curtailment_ratio = row.get("curtailment_ratio", np.nan)
    projector_adjustment = row.get("projector_adjustment_kw_total", np.nan)

    if regime == "reverse_flow_export":
        stress_kw = max(raw_export, 1.0)
        adjustment_ratio = float(projector_adjustment / stress_kw) if pd.notna(projector_adjustment) else np.nan
        if pd.notna(projector_adjustment) and adjustment_ratio >= 0.20 and grid_export > 0.25 * raw_export:
            return "PV reverse flow dominates; projection changed actions but residual export stayed high."
        if (
            (pd.isna(projector_adjustment) or adjustment_ratio < 0.05)
            and (pd.isna(curtailment_ratio) or float(curtailment_ratio) < 0.10)
            and battery_charge < 0.10 * stress_kw
        ):
            return "PV reverse flow dominates; projection barely added charging or curtailment."
        return "PV reverse flow dominates; check reverse-flow linearization and available flexibility."

    if regime == "grid_import_overload":
        stress_kw = max(raw_import, 1.0)
        adjustment_ratio = float(projector_adjustment / stress_kw) if pd.notna(projector_adjustment) else np.nan
        if pd.notna(projector_adjustment) and adjustment_ratio >= 0.20 and grid_import > 0.25 * raw_import:
            return "Load-driven import dominates; projection changed actions but residual import stayed high."
        if (pd.isna(projector_adjustment) or adjustment_ratio < 0.05) and battery_discharge < 0.10 * stress_kw:
            return "Load-driven import dominates; projection barely added discharge support."
        return "Load-driven import dominates; likely limited discharge headroom or linearization error."

    return "Transformer stress is mixed; inspect export/import swings and action gaps together."


def _prepare_trafo_diagnostic_frame(
    rollout: RolloutResult,
    *,
    train_result: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, str]:
    step_df = rollout.step_df.copy()
    if step_df.empty:
        raise ValueError("Rollout is empty; no trafo diagnostics are available.")

    required_columns = {
        "episode_idx",
        "step",
        "timestamp",
        "base_net_load_total",
        "net_load_total",
        "load_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
        "battery_charge_total",
        "battery_discharge_total",
        "trafo_penalty_total",
    }
    missing = sorted(required_columns.difference(step_df.columns))
    if missing:
        raise ValueError(f"Rollout step_df is missing required trafo diagnostic columns: {missing}")

    if "pv_raw_total" not in step_df.columns:
        if "pv_effective_total" not in step_df.columns:
            raise ValueError("Rollout step_df must contain either pv_raw_total or pv_effective_total.")
        step_df["pv_raw_total"] = (
            step_df["pv_effective_total"].to_numpy(dtype=np.float32)
            + step_df["pv_curtail_total"].to_numpy(dtype=np.float32)
        )

    if "psi_trafo_raw" not in step_df.columns:
        trafo_weight = _extract_trafo_penalty_weight(train_result)
        if trafo_weight is not None and abs(float(trafo_weight)) > 1e-6:
            step_df["psi_trafo_raw"] = step_df["trafo_penalty_total"].astype(np.float32) / float(trafo_weight)

    step_df["raw_export_total"] = np.clip(
        -step_df["base_net_load_total"].to_numpy(dtype=np.float32),
        0.0,
        None,
    )
    step_df["raw_import_total"] = np.clip(
        step_df["base_net_load_total"].to_numpy(dtype=np.float32),
        0.0,
        None,
    )
    step_df["net_load_shift_from_raw_kw"] = (
        step_df["net_load_total"].astype(np.float32) - step_df["base_net_load_total"].astype(np.float32)
    )
    step_df["curtailment_ratio"] = _safe_ratio_series(step_df["pv_curtail_total"], step_df["pv_raw_total"])
    step_df["charge_to_raw_export_ratio"] = _safe_ratio_series(step_df["battery_charge_total"], step_df["raw_export_total"])
    step_df["discharge_to_raw_import_ratio"] = _safe_ratio_series(
        step_df["battery_discharge_total"],
        step_df["raw_import_total"],
    )

    reverse_flow_mask = (
        (step_df["raw_export_total"] > step_df["raw_import_total"])
        & (step_df["grid_export_total"] > 1e-6)
        & (step_df["net_load_total"] < 0.0)
    )
    import_mask = (
        (step_df["raw_import_total"] >= step_df["raw_export_total"])
        & (step_df["grid_import_total"] > 1e-6)
        & (step_df["net_load_total"] >= 0.0)
    )
    step_df["dominant_regime"] = np.select(
        [reverse_flow_mask, import_mask],
        ["reverse_flow_export", "grid_import_overload"],
        default="mixed_or_balanced",
    )

    step_df["mitigation_toward_safe_direction_kw"] = np.where(
        reverse_flow_mask,
        step_df["pv_curtail_total"].astype(np.float32) + step_df["battery_charge_total"].astype(np.float32),
        np.where(
            import_mask,
            step_df["battery_discharge_total"].astype(np.float32),
            np.nan,
        ),
    )
    step_df["mitigation_against_safe_direction_kw"] = np.where(
        reverse_flow_mask,
        step_df["battery_discharge_total"].astype(np.float32),
        np.where(
            import_mask,
            step_df["battery_charge_total"].astype(np.float32) + step_df["pv_curtail_total"].astype(np.float32),
            np.nan,
        ),
    )

    loading_limit_pct = rollout.meta.get("trafo_loading_limit_pct", rollout.meta.get("loading_limit_pct", np.nan))
    if "trafo_loading_pct_max" not in step_df.columns:
        step_df["trafo_loading_pct_max"] = np.nan
    if pd.notna(loading_limit_pct):
        step_df["trafo_over_limit_pct"] = step_df["trafo_loading_pct_max"].astype(np.float32) - float(loading_limit_pct)
    else:
        step_df["trafo_over_limit_pct"] = np.nan

    rank_metric = "psi_trafo_raw" if "psi_trafo_raw" in step_df.columns else "trafo_penalty_total"
    step_df["rank_score"] = step_df[rank_metric].astype(np.float32)
    step_df["cause_hint"] = step_df.apply(_build_trafo_cause_hint, axis=1)
    ordered = step_df.sort_values(
        ["rank_score", "episode_idx", "step"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return ordered, rank_metric


def build_trafo_diagnostic_table(
    rollout: RolloutResult,
    *,
    train_result: Mapping[str, object] | None = None,
    top_k: int = 12,
) -> pd.DataFrame:
    ordered, rank_metric = _prepare_trafo_diagnostic_frame(rollout, train_result=train_result)
    top_n = max(1, int(top_k))
    selected = ordered.head(top_n).copy()
    selected["rank_metric"] = rank_metric

    preferred_columns = [
        "episode_idx",
        "step",
        "timestamp",
        "rank_metric",
        "rank_score",
        "trafo_penalty_total",
        "psi_trafo_raw",
        "trafo_loading_pct_max",
        "trafo_over_limit_pct",
        "n_trafo_violations",
        "dominant_regime",
        "raw_export_total",
        "raw_import_total",
        "net_load_total",
        "grid_export_total",
        "grid_import_total",
        "pv_raw_total",
        "pv_curtail_total",
        "curtailment_ratio",
        "battery_charge_total",
        "battery_discharge_total",
        "mitigation_toward_safe_direction_kw",
        "mitigation_against_safe_direction_kw",
        "projector_adjustment_kw_total",
        "battery_request_gap_kw_total",
        "pv_curtail_request_gap_kw_total",
        "controller_action_gap_total",
        "voltage_penalty_total",
        "line_penalty_total",
        "cause_hint",
    ]
    existing_columns = [column for column in preferred_columns if column in selected.columns]
    return selected.loc[:, existing_columns]


def summarize_trafo_diagnostics(
    rollout: RolloutResult,
    *,
    train_result: Mapping[str, object] | None = None,
    top_k: int = 12,
) -> pd.Series:
    ordered, rank_metric = _prepare_trafo_diagnostic_frame(rollout, train_result=train_result)
    top_n = max(1, int(top_k))
    selected = ordered.head(top_n).copy()
    if selected.empty:
        raise ValueError("Rollout is empty; no trafo diagnostics are available.")

    dominant_regime = (
        str(selected["dominant_regime"].value_counts().idxmax())
        if "dominant_regime" in selected.columns and not selected["dominant_regime"].empty
        else "mixed_or_balanced"
    )
    safety_summary = _extract_safety_summary(train_result)
    pre_violation = safety_summary.get("mean_pre_projection_violation", np.nan)
    post_violation = safety_summary.get("mean_post_projection_violation", np.nan)
    if pd.notna(pre_violation) and abs(float(pre_violation)) > 1e-6 and pd.notna(post_violation):
        violation_reduction_pct = 100.0 * (1.0 - float(post_violation) / float(pre_violation))
    else:
        violation_reduction_pct = np.nan

    projected_fraction = safety_summary.get("projected_fraction", np.nan)
    mean_adjustment_kw = (
        float(selected["projector_adjustment_kw_total"].mean())
        if "projector_adjustment_kw_total" in selected.columns
        else float("nan")
    )
    export_share = float((selected["dominant_regime"] == "reverse_flow_export").mean())
    import_share = float((selected["dominant_regime"] == "grid_import_overload").mean())

    if dominant_regime == "reverse_flow_export":
        if pd.notna(projected_fraction) and float(projected_fraction) < 0.10:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven, and the projector rarely changes the action."
        elif pd.notna(violation_reduction_pct) and float(violation_reduction_pct) >= 50.0:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven; the projector reduces linearized violation, but real trafo stress remains high."
        elif pd.notna(mean_adjustment_kw) and float(mean_adjustment_kw) < 0.25:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven, but the projector response stays small."
        else:
            diagnosis = "Top trafo steps are mostly reverse-flow/export driven; inspect reverse-flow sensitivity quality and flexibility limits."
    elif dominant_regime == "grid_import_overload":
        if pd.notna(projected_fraction) and float(projected_fraction) < 0.10:
            diagnosis = "Top trafo steps are mostly import driven, and the projector rarely changes the action."
        elif pd.notna(violation_reduction_pct) and float(violation_reduction_pct) >= 50.0:
            diagnosis = "Top trafo steps are mostly import driven; the projector reduces linearized violation, but real trafo stress remains high."
        else:
            diagnosis = "Top trafo steps are mostly import driven; inspect discharge headroom and linearization quality."
    else:
        diagnosis = "Top trafo steps mix export and import stress; inspect the detailed rows for regime switching."

    summary = {
        "controller": str(rollout.meta.get("controller", "unknown")),
        "rank_metric": rank_metric,
        "analyzed_steps": int(len(ordered)),
        "top_k_steps": int(len(selected)),
        "steps_with_trafo_penalty": int((ordered["trafo_penalty_total"] > 0.0).sum()),
        "trafo_penalty_total": float(ordered["trafo_penalty_total"].sum()),
        "top_k_mean_trafo_penalty": float(selected["trafo_penalty_total"].mean()),
        "top_k_max_trafo_penalty": float(selected["trafo_penalty_total"].max()),
        "dominant_regime_top_k": dominant_regime,
        "reverse_flow_export_share_top_k": export_share,
        "grid_import_share_top_k": import_share,
        "top_k_mean_raw_export_kw": float(selected["raw_export_total"].mean()),
        "top_k_mean_raw_import_kw": float(selected["raw_import_total"].mean()),
        "top_k_mean_grid_export_kw": float(selected["grid_export_total"].mean()),
        "top_k_mean_grid_import_kw": float(selected["grid_import_total"].mean()),
        "top_k_mean_pv_curtail_total": float(selected["pv_curtail_total"].mean()),
        "top_k_mean_battery_charge_total": float(selected["battery_charge_total"].mean()),
        "top_k_mean_battery_discharge_total": float(selected["battery_discharge_total"].mean()),
        "top_k_mean_projector_adjustment_kw_total": mean_adjustment_kw,
        "top_k_mean_trafo_loading_pct_max": (
            float(selected["trafo_loading_pct_max"].mean())
            if "trafo_loading_pct_max" in selected.columns
            else float("nan")
        ),
        "projector_enabled": bool(safety_summary.get("enabled", False)) if safety_summary else False,
        "projected_fraction": float(projected_fraction) if pd.notna(projected_fraction) else float("nan"),
        "mean_pre_projection_violation": float(pre_violation) if pd.notna(pre_violation) else float("nan"),
        "mean_post_projection_violation": float(post_violation) if pd.notna(post_violation) else float("nan"),
        "projector_violation_reduction_pct": float(violation_reduction_pct)
        if pd.notna(violation_reduction_pct)
        else float("nan"),
        "diagnosis": diagnosis,
    }
    return pd.Series(summary)


def summarize_rollout_metrics(rollout: RolloutResult) -> dict[str, object]:
    step_df = rollout.step_df.copy()
    agent_df = rollout.agent_df.copy()
    grid_df = rollout.grid_df.copy()
    summary_df = rollout.summary.copy()

    purchase_cost_total = (
        float(step_df["purchase_cost_total"].sum())
        if "purchase_cost_total" in step_df.columns
        else float(rollout.meta.get("agent_purchase_cost_eur", float("nan")))
    )
    export_subsidy_total = (
        float(step_df["export_subsidy_total"].sum())
        if "export_subsidy_total" in step_df.columns
        else float(rollout.meta.get("agent_export_subsidy_eur", float("nan")))
    )
    total_cost_eur = (
        float(purchase_cost_total - export_subsidy_total)
        if np.isfinite(purchase_cost_total) and np.isfinite(export_subsidy_total)
        else float("nan")
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
        voltage_violation_bus_points = float("nan")
        voltage_violation_steps = float("nan")
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

    voltage_step_delta_p95_pu, voltage_step_delta_max_pu = _summarize_voltage_step_delta(grid_df)
    voltage_spread_mean_pu, voltage_spread_max_pu = _summarize_voltage_spread(grid_df)
    feeder_netload_ramp_mean_abs_kw, feeder_netload_ramp_max_kw = _summarize_grouped_ramp(
        step_df,
        "feeder_post_action_net_load_kw",
    )
    trafo_loading_limit_pct = float(
        rollout.meta.get(
            "trafo_loading_limit_pct",
            rollout.meta.get("loading_limit_pct", np.nan),
        )
    )
    trafo_loading_max_pct = (
        float(step_df["trafo_loading_pct_max"].max()) if "trafo_loading_pct_max" in step_df.columns else float("nan")
    )
    if "n_trafo_violations" in step_df.columns:
        trafo_overload_steps = int((step_df["n_trafo_violations"].astype(float) > 0.0).sum())
    elif np.isfinite(trafo_loading_max_pct) and np.isfinite(trafo_loading_limit_pct) and "trafo_loading_pct_max" in step_df.columns:
        trafo_overload_steps = int((step_df["trafo_loading_pct_max"].astype(float) > trafo_loading_limit_pct).sum())
    else:
        trafo_overload_steps = float("nan")
    battery_net_power_kw_mean_abs = (
        float(np.mean(np.abs(step_df["battery_discharge_total"].astype(float) - step_df["battery_charge_total"].astype(float))))
        if {"battery_charge_total", "battery_discharge_total"}.issubset(step_df.columns)
        else float("nan")
    )
    missing_optional_fields = _collect_compare_optional_missing_fields(rollout)

    metrics = {
        "controller": str(rollout.meta.get("controller", "unknown")),
        "soc_mode": str(rollout.meta.get("soc_mode", "reset")),
        "purchase_cost_total": purchase_cost_total,
        "purchase_cost_total_eur": purchase_cost_total,
        "export_subsidy_total": export_subsidy_total,
        "export_subsidy_total_eur": export_subsidy_total,
        "total_cost_eur": total_cost_eur,
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
        "voltage_step_delta_p95_pu": voltage_step_delta_p95_pu,
        "voltage_step_delta_max_pu": voltage_step_delta_max_pu,
        "voltage_spread_mean_pu": voltage_spread_mean_pu,
        "voltage_spread_max_pu": voltage_spread_max_pu,
        "trafo_overload_steps": trafo_overload_steps,
        "trafo_loading_max_pct": trafo_loading_max_pct,
        "feeder_netload_ramp_mean_abs_kw": feeder_netload_ramp_mean_abs_kw,
        "feeder_netload_ramp_max_kw": feeder_netload_ramp_max_kw,
        "battery_net_power_kw_mean_abs": battery_net_power_kw_mean_abs,
        "returned_primary_objective_eur": float(rollout.meta.get("returned_primary_objective_eur", np.nan)),
        "high_budget_refinement_warn": bool(rollout.meta.get("high_budget_refinement_warn", False)),
        "formulation_tightening_required": bool(rollout.meta.get("formulation_tightening_required", False)),
        "physics_refinement_status": str(rollout.meta.get("physics_refinement_status", "")),
        "missing_optional_compare_fields": ", ".join(missing_optional_fields),
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
                "soc_mode",
                "purchase_cost_total",
                "purchase_cost_total_eur",
                "export_subsidy_total",
                "export_subsidy_total_eur",
                "total_cost_eur",
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
                "voltage_step_delta_p95_pu",
                "voltage_step_delta_max_pu",
                "voltage_spread_mean_pu",
                "voltage_spread_max_pu",
                "trafo_overload_steps",
                "trafo_loading_max_pct",
                "feeder_netload_ramp_mean_abs_kw",
                "feeder_netload_ramp_max_kw",
                "battery_net_power_kw_mean_abs",
                "returned_primary_objective_eur",
                "high_budget_refinement_warn",
                "formulation_tightening_required",
                "physics_refinement_status",
                "missing_optional_compare_fields",
            ]
        )
    return pd.DataFrame(rows)


def build_compare_economic_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "controller",
                "purchase_cost_total_eur",
                "export_subsidy_total_eur",
                "total_cost_eur",
            ]
        )
    required_columns = [
        "purchase_cost_total_eur",
        "export_subsidy_total_eur",
        "total_cost_eur",
    ]
    if metrics_df.loc[:, required_columns].isna().any().any():
        incomplete = metrics_df.loc[
            metrics_df.loc[:, required_columns].isna().any(axis=1),
            "controller",
        ].astype(str).tolist()
        raise ValueError(
            "Compare economic table requires final-dispatch purchase/export/total cost fields. "
            f"Missing values detected for: {incomplete}"
        )
    return metrics_df.loc[
        :,
        [
            "controller",
            "purchase_cost_total_eur",
            "export_subsidy_total_eur",
            "total_cost_eur",
        ],
    ].copy()


def build_compare_safety_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "controller",
                "voltage_violation_steps",
                "voltage_violation_bus_points",
                "voltage_step_delta_p95_pu",
                "voltage_step_delta_max_pu",
                "voltage_spread_mean_pu",
                "voltage_spread_max_pu",
                "trafo_overload_steps",
                "trafo_loading_max_pct",
                "feeder_netload_ramp_mean_abs_kw",
                "feeder_netload_ramp_max_kw",
            ]
        )
    return metrics_df.loc[
        :,
        [
            "controller",
            "voltage_violation_steps",
            "voltage_violation_bus_points",
            "voltage_step_delta_p95_pu",
            "voltage_step_delta_max_pu",
            "voltage_spread_mean_pu",
            "voltage_spread_max_pu",
            "trafo_overload_steps",
            "trafo_loading_max_pct",
            "feeder_netload_ramp_mean_abs_kw",
            "feeder_netload_ramp_max_kw",
        ],
    ].copy()


def build_compare_warning_banner(*rollouts: RolloutResult):
    try:
        from IPython.display import HTML as _HTML  # type: ignore
    except ModuleNotFoundError:
        class _HTML(str):
            @property
            def data(self) -> str:
                return str(self)

            def _repr_html_(self) -> str:
                return str(self)

    items: list[str] = []
    for rollout in rollouts:
        controller = escape(str(rollout.meta.get("controller", "unknown")))
        if bool(rollout.meta.get("high_budget_refinement_warn", False)):
            items.append(
                f"<li><strong>{controller}</strong>: high-budget MISOCP refinement was used; "
                "compare economics are based on the returned/final dispatch, so the economic comparison should be interpreted with care.</li>"
            )
        if bool(rollout.meta.get("formulation_tightening_required", False)):
            items.append(
                f"<li><strong>{controller}</strong>: formulation tightening is still required; "
                "safety and economic conclusions may be unstable.</li>"
            )
        missing_optional_fields = _collect_compare_optional_missing_fields(rollout)
        if missing_optional_fields:
            joined = ", ".join(escape(field_name) for field_name in missing_optional_fields)
            items.append(
                f"<li><strong>{controller}</strong>: optional compare fields missing "
                f"({joined}); affected table cells will show N/A.</li>"
            )

    if not items:
        html = (
            "<div style='padding:10px 12px;border:1px solid #86efac;background:#f0fdf4;color:#166534;"
            "border-radius:8px;'>Compare warnings: none.</div>"
        )
        return _HTML(html)

    html = (
        "<div style='padding:10px 12px;border:1px solid #fdba74;background:#fff7ed;color:#9a3412;"
        "border-radius:8px;'><strong>Compare warnings</strong><ul style='margin:8px 0 0 18px;'>"
        + "".join(items)
        + "</ul></div>"
    )
    return _HTML(html)


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
    step_df, _ = _prepare_agent_power_balance_frame(
        rollout.step_df,
        controller_label=str(rollout.meta.get("controller", "unknown")),
    )

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    timestamps = step_df["timestamp"]
    width = 0.008

    positive_specs = [
        ("load_total", "Load", "#111827"),
        ("battery_charge_total", "Charge", "#dc2626"),
        ("grid_export_total", "Grid export", "#f59e0b"),
        ("pv_curtail_total", "Curtailment loss", "#fca5a5"),
    ]
    negative_specs = [
        ("pv_raw_total", "PV raw", "#16a34a"),
        ("grid_import_total", "Grid import", "#2563eb"),
        ("battery_discharge_total", "Discharge", "#7c3aed"),
    ]

    positive_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, label, color in positive_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        extra_kwargs = {"hatch": "//", "edgecolor": "#dc2626", "linewidth": 1.0} if column == "pv_curtail_total" else {}
        axis.bar(
            timestamps,
            values,
            width=width,
            bottom=positive_bottom,
            color=color,
            alpha=0.78,
            label=label,
            **extra_kwargs,
        )
        positive_bottom = positive_bottom + values

    negative_bottom = np.zeros(len(step_df), dtype=np.float32)
    for column, label, color in negative_specs:
        values = step_df[column].to_numpy(dtype=np.float32)
        axis.bar(timestamps, -values, width=width, bottom=negative_bottom, color=color, alpha=0.78, label=label)
        negative_bottom = negative_bottom - values

    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_title(f"Power Balance - {rollout.meta['controller']}")
    axis.set_ylabel("kW")
    axis.set_xlabel("Timestamp")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="upper right", ncol=4)
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
        figsize=figsize or (20.0, max(4.0 * n_rows, 5.6)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    muted_color = "#cbd5e1"
    highlight_palette = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#0891b2"]
    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 11
    legend_fontsize = 11

    for axis_idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        grid_df = rollout.grid_df
        if grid_df.empty:
            raise ValueError(f"Rollout '{rollout.meta.get('controller', 'unknown')}' has no grid_df.")
        background_df = grid_df.loc[~grid_df["is_agent_bus"], ["timestamp", "bus_id", "vm_pu"]].sort_values(
            ["timestamp", "bus_id"]
        )
        if not background_df.empty:
            background_wide = background_df.pivot_table(
                index="timestamp",
                columns="bus_id",
                values="vm_pu",
                aggfunc="first",
            ).sort_index()
            if background_wide.shape[1] > 0:
                axis.plot(
                    background_wide.index.to_numpy(),
                    background_wide.to_numpy(dtype=np.float32),
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
                label=f"Agent bus {bus_id}" if axis_idx == 0 else None,
                zorder=3,
            )
        axis.axhline(
            float(rollout.meta["v_min_pu"]),
            color="#dc2626",
            linestyle="--",
            linewidth=1.1,
            label="V min" if axis_idx == 0 else None,
        )
        axis.axhline(
            float(rollout.meta["v_max_pu"]),
            color="#ea580c",
            linestyle="--",
            linewidth=1.1,
            label="V max" if axis_idx == 0 else None,
        )
        axis.set_ylabel("V [p.u.]", fontsize=label_fontsize)
        axis.set_title(str(rollout.meta["controller"]), fontsize=title_fontsize)
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        if axis_idx == 0:
            axis.legend(loc="upper right", ncol=2, fontsize=legend_fontsize)

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
    figure.tight_layout()
    return figure


def plot_price_prediction_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] = (20.0, 5.0),
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 11
    legend_fontsize = 11

    reference_rollout = next(
        (rollout for rollout in rollouts if str(rollout.meta.get("prediction_mode", "")) == NORMAL_PREDICTION_MODE),
        rollouts[0],
    )
    reference_step_df = reference_rollout.step_df.copy()
    if reference_step_df.empty or not {"timestamp", "price", "price_pred"}.issubset(reference_step_df.columns):
        raise ValueError("Rollouts must contain timestamp, price, and price_pred columns.")

    figure, axis = plt.subplots(1, 1, figsize=figsize)
    axis.plot(
        reference_step_df["timestamp"],
        reference_step_df["price"],
        color="#111827",
        linewidth=1.8,
        label="Price",
    )

    reference_price_adder = float(reference_rollout.meta.get("import_price_adder_eur_per_kwh", 0.0))
    reference_timestamps = pd.to_datetime(reference_step_df["timestamp"]).to_numpy()
    reference_pred = reference_step_df["price_pred"].to_numpy(dtype=np.float64) + reference_price_adder
    for rollout in rollouts:
        if str(rollout.meta.get("prediction_mode", "")) != NORMAL_PREDICTION_MODE:
            continue
        step_df = rollout.step_df.copy()
        if step_df.empty or "price_pred" not in step_df.columns:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing price_pred for compare plotting."
            )
        price_adder = float(rollout.meta.get("import_price_adder_eur_per_kwh", 0.0))
        candidate_timestamps = pd.to_datetime(step_df["timestamp"]).to_numpy()
        candidate_pred = step_df["price_pred"].to_numpy(dtype=np.float64) + price_adder
        if (
            candidate_timestamps.shape != reference_timestamps.shape
            or not np.array_equal(candidate_timestamps, reference_timestamps)
            or candidate_pred.shape != reference_pred.shape
            or not np.allclose(candidate_pred, reference_pred, equal_nan=True)
        ):
            raise ValueError(
                "Compare price plotting expects all forecast/LSTM rollouts to share the same adjusted price prediction."
            )

    axis.plot(
        reference_step_df["timestamp"],
        reference_pred,
        color="#dc2626",
        linewidth=1.8,
        linestyle="--",
        label="Predicted price",
    )

    axis.set_title("Price And Adjusted Forecast", fontsize=title_fontsize)
    axis.set_ylabel("EUR/kWh", fontsize=label_fontsize)
    axis.set_xlabel("Timestamp", fontsize=label_fontsize)
    axis.grid(True, alpha=0.25)
    axis.tick_params(axis="both", labelsize=tick_fontsize)
    axis.legend(loc="upper right", ncol=2, fontsize=legend_fontsize)
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
        figsize=figsize or (20.0, max(4.2 * n_rows, 5.8)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 11
    legend_fontsize = 11

    for axis_idx, (feeder_axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df = _prepare_compare_net_load_frame(
            rollout.step_df,
            controller_label=str(rollout.meta.get("controller", "unknown")),
        )
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["feeder_raw_net_load_kw"],
            color="#111827",
            linewidth=1.6,
            label="Feeder raw net load",
        )
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["feeder_effective_net_load_kw"],
            color="#16a34a",
            linewidth=1.4,
            linestyle="-.",
            label="Feeder post-curtail net load",
        )
        feeder_axis.plot(
            step_df["timestamp"],
            step_df["feeder_post_action_net_load_kw"],
            color="#2563eb",
            linewidth=1.5,
            linestyle="--",
            label="Feeder post-action net load",
        )
        if "root_net_exchange_kw" in step_df.columns:
            feeder_axis.plot(
                step_df["timestamp"],
                step_df["root_net_exchange_kw"],
                color="#7c3aed",
                linewidth=1.7,
                label="Root net exchange",
            )
        elif "pp_root_p_kw" in step_df.columns:
            feeder_axis.plot(
                step_df["timestamp"],
                step_df["pp_root_p_kw"],
                color="#7c3aed",
                linewidth=1.7,
                label="Pandapower root exchange",
            )
        trafo_limit_kw = rollout.meta.get("trafo_limit_kw")
        if trafo_limit_kw is not None:
            trafo_limit_value = float(trafo_limit_kw)
            if np.isfinite(trafo_limit_value) and trafo_limit_value > 0.0:
                feeder_axis.axhline(
                    trafo_limit_value,
                    color="#dc2626",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"Transformer S-limit ref (+P view) ({trafo_limit_value:.1f} kW)" if axis_idx == 0 else None,
                )
                feeder_axis.axhline(
                    -trafo_limit_value,
                    color="#dc2626",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"Transformer S-limit ref (-P view) ({trafo_limit_value:.1f} kW)" if axis_idx == 0 else None,
                )
        feeder_axis.set_ylabel("kW", fontsize=label_fontsize)
        feeder_axis.set_title(f"{rollout.meta['controller']} - feeder total", fontsize=title_fontsize)
        feeder_axis.grid(True, alpha=0.25)
        feeder_axis.tick_params(axis="both", labelsize=tick_fontsize)
        if axis_idx == 0:
            feeder_axis.legend(loc="upper right", fontsize=legend_fontsize)

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
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
        figsize=figsize or (20.0, max(4.0 * n_rows, 5.6)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 11
    legend_fontsize = 11
    positive_specs = [
        ("load_total", "Load", "#111827"),
        ("battery_charge_total", "Charge", "#dc2626"),
        ("grid_export_total", "Grid export", "#f59e0b"),
        ("pv_curtail_total", "Curtailment loss", "#fca5a5"),
    ]
    negative_specs = [
        ("pv_raw_total", "PV raw", "#16a34a"),
        ("grid_import_total", "Grid import", "#2563eb"),
        ("battery_discharge_total", "Discharge", "#7c3aed"),
    ]

    for axis_idx, (axis, rollout) in enumerate(zip(axes, rollouts, strict=False)):
        step_df, max_abs_balance_residual_kw = _prepare_agent_power_balance_frame(
            rollout.step_df,
            controller_label=str(rollout.meta.get("controller", "unknown")),
        )

        timestamps = step_df["timestamp"].to_numpy()
        positive_bottom = np.zeros(len(step_df), dtype=np.float32)
        for column, label, color in positive_specs:
            values = step_df[column].to_numpy(dtype=np.float32)
            extra_kwargs = (
                {"hatch": "//", "edgecolor": "#dc2626", "linewidth": 1.0}
                if column == "pv_curtail_total"
                else {}
            )
            axis.bar(
                timestamps,
                values,
                width=0.008,
                bottom=positive_bottom,
                color=color,
                alpha=0.78,
                label=label if axis_idx == 0 else None,
                **extra_kwargs,
            )
            positive_bottom = positive_bottom + values

        negative_bottom = np.zeros(len(step_df), dtype=np.float32)
        for column, label, color in negative_specs:
            values = step_df[column].to_numpy(dtype=np.float32)
            axis.bar(
                timestamps,
                -values,
                width=0.008,
                bottom=negative_bottom,
                color=color,
                alpha=0.78,
                label=label if axis_idx == 0 else None,
            )
            negative_bottom = negative_bottom - values
        axis.axhline(0.0, color="#111827", linewidth=1.0)
        axis.set_ylabel("kW", fontsize=label_fontsize)
        axis.set_title(
            f"{rollout.meta['controller']} - agent-only balance (residual<={max_abs_balance_residual_kw:.3f} kW)",
            fontsize=title_fontsize,
        )
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        if axis_idx == 0:
            axis.legend(loc="upper right", ncol=4, fontsize=legend_fontsize)

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
    figure.tight_layout()
    return figure


def plot_battery_power_and_soc_comparison(
    *rollouts: RolloutResult,
    figsize: tuple[float, float] | None = None,
):
    if not rollouts:
        raise ValueError("At least one rollout is required.")

    n_rows = len(rollouts)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize or (20.0, max(4.2 * n_rows, 6.5)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 11
    legend_fontsize = 11
    discharge_color = "#dc2626"
    charge_color = "#2563eb"
    soc_agent_color = "#94a3b8"
    soc_mean_color = "#111827"
    zero_line_color = "#475569"

    for rollout_idx, rollout in enumerate(rollouts):
        power_axis = axes[rollout_idx]
        soc_axis = power_axis.twinx()
        step_df = rollout.step_df.copy()
        agent_df = rollout.agent_df.copy()
        if step_df.empty:
            raise ValueError(f"Rollout '{rollout.meta.get('controller', 'unknown')}' has no step_df.")
        if not {"battery_charge_total", "battery_discharge_total", "timestamp"}.issubset(step_df.columns):
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing battery power columns required for compare plotting."
            )
        step_df = step_df.copy()
        step_df["battery_net_power_kw"] = (
            step_df["battery_discharge_total"].to_numpy(dtype=np.float32)
            - step_df["battery_charge_total"].to_numpy(dtype=np.float32)
        )
        net_power = step_df["battery_net_power_kw"].to_numpy(dtype=np.float32)
        bar_colors = [discharge_color if value >= 0.0 else charge_color for value in net_power]
        power_axis.bar(
            step_df["timestamp"],
            net_power,
            width=0.008,
            color=bar_colors,
            alpha=0.82,
        )
        power_axis.axhline(0.0, color=zero_line_color, linewidth=1.0)
        power_axis.set_ylabel("Battery Power [kW]", fontsize=label_fontsize)
        power_axis.set_title(str(rollout.meta["controller"]), fontsize=title_fontsize)
        power_axis.grid(True, alpha=0.25)
        power_axis.tick_params(axis="both", labelsize=tick_fontsize)

        if agent_df.empty or "soc" not in agent_df.columns:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing agent_df.soc required for compare plotting."
            )
        group_columns = [column for column in ["episode_idx", "step", "timestamp"] if column in agent_df.columns]
        if not group_columns:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing grouping columns required to aggregate SoC."
            )
        agent_id_column = "agent_id" if "agent_id" in agent_df.columns else None
        if agent_id_column is None:
            raise ValueError(
                f"Rollout '{rollout.meta.get('controller', 'unknown')}' is missing agent_df.agent_id required for compare plotting."
            )
        soc_summary = (
            agent_df.groupby(group_columns, as_index=False)
            .agg(
                mean=("soc", "mean"),
            )
        )
        first_agent = True
        for _, agent_frame in agent_df.sort_values(group_columns + [agent_id_column]).groupby(agent_id_column, sort=True):
            soc_axis.plot(
                agent_frame["timestamp"],
                agent_frame["soc"],
                color=soc_agent_color,
                linewidth=1.1,
                alpha=0.45,
                label="Agent SoC" if first_agent and rollout_idx == 0 else None,
            )
            first_agent = False
        soc_axis.plot(
            soc_summary["timestamp"],
            soc_summary["mean"],
            color=soc_mean_color,
            linewidth=1.7,
            label="Mean SoC" if rollout_idx == 0 else None,
        )
        soc_axis.set_ylabel("SoC", fontsize=label_fontsize)
        soc_axis.set_ylim(0.0, 1.0)
        soc_axis.grid(True, alpha=0.25)
        soc_axis.tick_params(axis="both", labelsize=tick_fontsize)
        if rollout_idx == 0:
            legend_handles = [
                Patch(facecolor=discharge_color, alpha=0.82, label="Discharge (+)"),
                Patch(facecolor=charge_color, alpha=0.82, label="Charge (-)"),
            ]
            handles_2, labels_2 = soc_axis.get_legend_handles_labels()
            power_axis.legend(
                legend_handles + handles_2,
                ["Discharge (+)", "Charge (-)"] + labels_2,
                loc="upper right",
                ncol=2,
                fontsize=legend_fontsize,
            )

    axes[-1].set_xlabel("Timestamp", fontsize=label_fontsize)
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
    "build_compare_economic_table",
    "build_compare_safety_table",
    "build_compare_warning_banner",
    "collect_madrl_rollout",
    "collect_controller_rollout",
    "collect_global_full_horizon_rollout",
    "collect_global_mpc_rollout",
    "collect_mpc_rollout",
    "compare_purchase_costs",
    "compare_rollout_metrics",
    "build_trafo_diagnostic_table",
    "ensure_forecast_ready",
    "normalize_date_input",
    "normalize_prediction_mode",
    "load_training_run_bundle",
    "plot_purchase_cost_comparison",
    "plot_operating_cost_comparison",
    "plot_net_load_comparison",
    "plot_price_prediction_comparison",
    "plot_power_balance_bars",
    "plot_power_balance_comparison",
    "plot_battery_power_and_soc_comparison",
    "plot_global_misocp_validation",
    "plot_rollout_comparison_dashboard",
    "plot_rollout_dashboard",
    "plot_test_rollout",
    "plot_test_voltage_profile",
    "plot_voltage_profile_comparison",
    "resolve_evaluation_mode",
    "resolve_forecast_backend",
    "resolve_prediction_mode_from_forecast_backend",
    "summarize_trafo_diagnostics",
    "summarize_rollout_metrics",
    "validate_compare_model_bundles",
]
