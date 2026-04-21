"""Managed LSTM artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from predictors.artifacts import (
    get_default_lstm_artifact_dir,
    get_default_lstm_artifact_paths,
    get_weekly_forecast_plot_path,
)
from predictors.lstm_forecaster import (
    LSTM_LOAD_HYBRID_ARTIFACT_FORMAT,
    POSTPROCESS_MODE_BASELINE_BLEND,
    POSTPROCESS_MODE_NONE,
)


def _training_module():
    from predictors import training as training_module

    return training_module


def _normalize_signal_name(signal_name: str) -> str:
    return _training_module()._normalize_signal_name(signal_name)


def _resolve_signal_training_settings(
    cfg,
    signal_name: str,
    overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    return _training_module().resolve_signal_training_settings(cfg, signal_name, overrides=overrides)


def _resolve_signal_artifact_format(signal_name: str, settings: dict[str, object]) -> str:
    return _training_module().resolve_signal_artifact_format(signal_name, settings)


def _resolve_signal_physical_normalization_mode(signal_name: str) -> str:
    return _training_module().resolve_signal_physical_normalization_mode(signal_name)


def _build_lstm_source_signature(cfg, signal_name: str) -> dict[str, object]:
    return _training_module().build_lstm_source_signature(cfg, signal_name)


def _resolve_signal_model_mode(cfg, signal_name: str) -> str:
    return _training_module().resolve_signal_model_mode(cfg, signal_name)


def _resolve_signal_component_split(cfg, signal_name: str) -> bool:
    return _training_module().resolve_signal_component_split(cfg, signal_name)


def _configured_forecast_signals(cfg) -> list[str]:
    return _training_module().configured_forecast_signals(cfg)


def _resolve_signal_csv_source(data_dir: str | Path, signal_name: str, cfg=None):
    return _training_module().resolve_signal_csv_source(data_dir, signal_name, cfg=cfg)


def _train_signal_lstm(
    cfg,
    signal_name: str,
    *,
    device: str | torch.device | None = None,
    overrides: dict[str, object] | None = None,
):
    return _training_module().train_signal_lstm(
        cfg,
        signal_name,
        device=device,
        overrides=overrides,
    )


def _plot_weekly_forecasts(evaluations, *, save_path: str | Path) -> Path:
    return _training_module().plot_weekly_forecasts(evaluations, save_path=save_path)


def forecast_artifact_root(cfg) -> Path:
    """Return the artifact root for the current forecast config."""
    root = cfg.forecast.lstm_artifact_root
    if root is None:
        return get_default_lstm_artifact_dir()
    return Path(root)


def _expected_signal_optimized_metric(
    signal_name: str,
    *,
    postprocess_mode: str,
    component: str | None = None,
) -> str | None:
    normalized_signal = _normalize_signal_name(signal_name)
    if normalized_signal != "load" or str(postprocess_mode) != POSTPROCESS_MODE_BASELINE_BLEND:
        return None
    if str(component or "").strip().lower() == "heatpump":
        return "blocked_bias_guard_step1"
    return "mae_step1"


def expected_lstm_artifact_meta(
    cfg,
    signal_name: str,
    overrides: dict[str, object] | None = None,
    *,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    component: str | None = None,
) -> dict[str, object]:
    """Build the expected artifact metadata for one signal."""
    local_cfg, settings = _resolve_signal_training_settings(cfg, signal_name, overrides=overrides)
    future_horizon = int(settings["future_horizon"])
    return {
        "artifact_format": _resolve_signal_artifact_format(signal_name, settings),
        "signal_name": str(settings["signal_name"]),
        "future_horizon": future_horizon,
        "pred_len": future_horizon,
        "seq_len": int(settings["history_window"]),
        "hidden_size": int(settings["hidden_size"]),
        "num_layers": int(settings["num_layers"]),
        "dropout": float(settings["dropout"]),
        "input_size": int(settings["input_size"]),
        "time_feature_mode": str(settings["time_feature_mode"]),
        "model_mode": str(settings["model_mode"]),
        "normalization_mode": _resolve_signal_physical_normalization_mode(signal_name),
        "source_signature": _build_lstm_source_signature(local_cfg, signal_name),
        "agent_index": None if agent_index is None else int(agent_index),
        "agent_profile": None if agent_profile is None else str(agent_profile),
        "component": None if component is None else str(component),
        **(
            {
                "postprocess_mode": str(settings["postprocess_mode"]),
                "baseline_mode": str(settings["baseline_mode"]),
                "blend_weight": None,
                "optimized_metric": _expected_signal_optimized_metric(
                    signal_name,
                    postprocess_mode=str(settings["postprocess_mode"]),
                    component=component,
                ),
            }
            if (
                _resolve_signal_artifact_format(signal_name, settings) == LSTM_LOAD_HYBRID_ARTIFACT_FORMAT
                or str(settings["postprocess_mode"]) != POSTPROCESS_MODE_NONE
            )
            else {}
        ),
    }


def compare_lstm_artifact_meta(
    actual_meta: dict[str, object] | None,
    expected_meta: dict[str, object],
) -> dict[str, object]:
    """Compare saved artifact metadata with the current expected configuration."""
    comparable_fields = tuple(expected_meta.keys())
    actual_meta = dict(actual_meta or {})
    actual = {field: actual_meta.get(field) for field in comparable_fields}
    expected = {field: expected_meta.get(field) for field in comparable_fields}
    mismatches: dict[str, dict[str, object]] = {}

    for field in comparable_fields:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if field == "dropout":
            matches = actual_value is not None and bool(np.isclose(float(actual_value), float(expected_value)))
        elif field == "blend_weight" and expected_value is None:
            matches = field in actual_meta and (
                actual_value is None or 0.0 <= float(actual_value) <= 1.0
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches[field] = {
                "expected": expected_value,
                "actual": actual_value,
            }

    return {
        "compatible": not mismatches,
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }


def validate_lstm_artifact(
    cfg,
    signal_name: str,
    paths: dict[str, str | Path],
    *,
    overrides: dict[str, object] | None = None,
    agent_index: int | None = None,
    agent_profile: str | None = None,
    component: str | None = None,
) -> dict[str, object]:
    """Validate one artifact against the current config."""
    normalized_signal = _normalize_signal_name(signal_name)
    expected = expected_lstm_artifact_meta(
        cfg,
        normalized_signal,
        overrides=overrides,
        agent_index=agent_index,
        agent_profile=agent_profile,
        component=component,
    )
    resolved_paths = {name: Path(path) for name, path in paths.items()}
    required_files = {
        key: resolved_paths[key]
        for key in ("model_path", "meta_path", "scaler_path")
        if key in resolved_paths
    }
    existing_files = {key: path.exists() for key, path in required_files.items()}
    missing_files = [key for key, exists in existing_files.items() if not exists]
    any_existing = any(existing_files.values())
    result: dict[str, object] = {
        "signal_name": normalized_signal,
        "artifact_path": str(resolved_paths.get("model_path", "")),
        "paths": {key: str(path) for key, path in required_files.items()},
        "expected": expected,
        "actual": {},
        "mismatches": {},
        "missing_files": missing_files,
        "compatible": False,
        "issue_type": None,
        "agent_index": None if agent_index is None else int(agent_index),
        "agent_profile": None if agent_profile is None else str(agent_profile),
    }

    if missing_files:
        result["issue_type"] = "incomplete" if any_existing else "missing"
        return result

    try:
        actual_meta = json.loads(required_files["meta_path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["issue_type"] = "invalid_meta"
        result["error"] = f"Failed to read meta: {exc}"
        return result

    comparison = compare_lstm_artifact_meta(actual_meta, expected)
    result.update(comparison)
    result["issue_type"] = None if comparison["compatible"] else "mismatch"
    return result


def _managed_lstm_artifact_paths(
    cfg,
    signal_name: str,
    *,
    agent_index: int | None = None,
    component: str | None = None,
) -> dict[str, Path]:
    """Return the managed artifact paths for one signal."""
    return get_default_lstm_artifact_paths(
        root=forecast_artifact_root(cfg),
        signal_name=_normalize_signal_name(signal_name),
        future_horizon=int(cfg.env.future_horizon),
        agent_index=agent_index,
        component=component,
    )


def _artifact_tuple_from_paths(paths: dict[str, Path]) -> tuple[str, str, str]:
    """Convert a managed path mapping into the legacy tuple shape."""
    return (
        str(paths["model_path"]),
        str(paths["meta_path"]),
        str(paths["scaler_path"]),
    )


def _format_lstm_artifact_mismatches(mismatches: dict[str, dict[str, object]]) -> str:
    """Render mismatch details as a compact log string."""
    return ", ".join(
        f"{field}(expected={detail['expected']}, actual={detail['actual']})"
        for field, detail in sorted(mismatches.items())
    )


def _format_lstm_artifact_issue(validation: dict[str, object]) -> str:
    """Build a user-facing artifact problem summary."""
    signal_name = validation["signal_name"]
    artifact_path = validation["artifact_path"]
    issue_type = validation.get("issue_type")
    if issue_type == "missing":
        missing_files = ", ".join(validation.get("missing_files", []))
        return (
            f"signal='{signal_name}', artifact='{artifact_path}', missing files: {missing_files}. "
            "Delete stale artifacts or enable cfg.forecast.auto_train_missing=True to rebuild them automatically."
        )
    if issue_type == "incomplete":
        missing_files = ", ".join(validation.get("missing_files", []))
        return (
            f"signal='{signal_name}', artifact='{artifact_path}', artifact is incomplete: {missing_files}. "
            "Delete the stale artifact and retry, or enable cfg.forecast.auto_train_missing=True to rebuild it automatically."
        )
    if issue_type == "invalid_meta":
        return (
            f"signal='{signal_name}', artifact='{artifact_path}', {validation.get('error', 'meta is unreadable')}. "
            "Delete the stale artifact or enable cfg.forecast.auto_train_missing=True to rebuild it automatically."
        )
    mismatch_text = _format_lstm_artifact_mismatches(validation.get("mismatches", {}))
    return (
        f"signal='{signal_name}', artifact='{artifact_path}', config mismatch: {mismatch_text}. "
        "Delete the stale artifact or enable cfg.forecast.auto_train_missing=True to rebuild it automatically."
    )


def _signal_artifact_specs(cfg, signal_name: str) -> list[dict[str, object]]:
    normalized_signal = _normalize_signal_name(signal_name)
    if normalized_signal == "load" and _resolve_signal_model_mode(cfg, normalized_signal) == "per_agent":
        profiles = [str(profile) for profile in cfg.data.agent_profiles][: int(cfg.env.num_agents)]
        use_component_split = _resolve_signal_component_split(cfg, normalized_signal)
        components = list(cfg.data.load_components) if use_component_split else [None]
        specs = []
        for agent_index, profile in enumerate(profiles):
            for component in components:
                specs.append({
                    "signal_name": normalized_signal,
                    "agent_index": int(agent_index),
                    "agent_profile": profile,
                    "component": component,
                    "paths": _managed_lstm_artifact_paths(
                        cfg, normalized_signal, agent_index=agent_index, component=component,
                    ),
                })
        return specs
    return [
        {
            "signal_name": normalized_signal,
            "agent_index": None,
            "agent_profile": None,
            "paths": _managed_lstm_artifact_paths(cfg, normalized_signal),
        }
    ]


def _collect_lstm_artifact_inventory(
    cfg,
    *,
    overrides_by_signal: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Scan every signal and classify compatible, missing, and invalid artifacts."""
    resolved_overrides_by_signal = dict(
        getattr(cfg.forecast, "signal_training_overrides", {}) if overrides_by_signal is None else overrides_by_signal
    )
    artifacts: dict[str, object] = {}
    missing_artifacts: dict[str, list[dict[str, object]]] = {}
    invalid_artifacts: dict[str, list[dict[str, object]]] = {}

    for signal_name in _configured_forecast_signals(cfg):
        compatible_artifacts: list[tuple[str, str, str]] = []
        signal_missing: list[dict[str, object]] = []
        signal_invalid: list[dict[str, object]] = []
        signal_overrides = dict(resolved_overrides_by_signal.get(signal_name) or {})

        for spec in _signal_artifact_specs(cfg, signal_name):
            validation = validate_lstm_artifact(
                cfg,
                signal_name,
                spec["paths"],
                overrides=signal_overrides or None,
                agent_index=spec["agent_index"],
                agent_profile=spec["agent_profile"],
                component=spec.get("component"),
            )
            if validation["compatible"]:
                compatible_artifacts.append(_artifact_tuple_from_paths(spec["paths"]))
                continue
            if validation.get("issue_type") == "missing":
                signal_missing.append(validation)
                continue
            signal_invalid.append(validation)

        if not signal_missing and not signal_invalid and compatible_artifacts:
            artifacts[signal_name] = (
                compatible_artifacts if len(compatible_artifacts) > 1 else compatible_artifacts[0]
            )
        if signal_missing:
            missing_artifacts[signal_name] = signal_missing
        if signal_invalid:
            invalid_artifacts[signal_name] = signal_invalid

    return {
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "missing_signals": sorted(missing_artifacts),
        "invalid_artifacts": invalid_artifacts,
        "mismatched_signals": sorted(invalid_artifacts),
    }


def _raise_lstm_artifact_requirements_error(cfg, inventory: dict[str, object]) -> None:
    """Raise a detailed artifact error when auto-retrain is disabled."""
    problem_lines: list[str] = []
    for validations in inventory.get("invalid_artifacts", {}).values():
        for validation in validations:
            problem_lines.append(f"- {_format_lstm_artifact_issue(validation)}")

    for validations in inventory.get("missing_artifacts", {}).values():
        for validation in validations:
            problem_lines.append(f"- {_format_lstm_artifact_issue(validation)}")

    if not problem_lines:
        return

    mismatch_count = sum(len(validations) for validations in inventory.get("invalid_artifacts", {}).values())
    missing_count = sum(len(validations) for validations in inventory.get("missing_artifacts", {}).values())
    if mismatch_count and missing_count:
        exc_type = RuntimeError
    elif mismatch_count:
        exc_type = ValueError
    else:
        exc_type = FileNotFoundError

    raise exc_type(
        "Managed LSTM forecast artifacts are missing or incompatible:\n"
        + "\n".join(problem_lines)
    )


def collect_available_lstm_artifacts(
    cfg,
    *,
    overrides_by_signal: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return only artifacts compatible with the current config."""
    resolved_overrides_by_signal = (
        getattr(cfg.forecast, "signal_training_overrides", {}) if overrides_by_signal is None else overrides_by_signal
    )
    return dict(
        _collect_lstm_artifact_inventory(cfg, overrides_by_signal=resolved_overrides_by_signal)["artifacts"]
    )


def ensure_lstm_artifacts(
    cfg,
    *,
    device: str | torch.device | None = None,
    overrides_by_signal: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Ensure the managed LSTM artifacts required by the current config exist."""
    artifact_root = forecast_artifact_root(cfg)
    artifact_root.mkdir(parents=True, exist_ok=True)
    resolved_overrides_by_signal = (
        getattr(cfg.forecast, "signal_training_overrides", {}) if overrides_by_signal is None else overrides_by_signal
    )

    inventory_before = _collect_lstm_artifact_inventory(cfg, overrides_by_signal=resolved_overrides_by_signal)
    if not bool(cfg.forecast.auto_train_missing) and (
        inventory_before["missing_signals"] or inventory_before["invalid_artifacts"]
    ):
        _raise_lstm_artifact_requirements_error(cfg, inventory_before)

    trained_results: list[dict[str, object]] = []
    trained_signals: list[str] = []
    retrained_signals: list[str] = []
    data_dir = Path(cfg.data.data_dir or (Path(__file__).resolve().parent.parent / "data"))

    for signal_name in _configured_forecast_signals(cfg):
        if signal_name in inventory_before["artifacts"]:
            continue

        invalid_validations = inventory_before["invalid_artifacts"].get(signal_name, [])
        for invalid_validation in invalid_validations:
            print(f"[forecast] artifact mismatch detected: {_format_lstm_artifact_issue(invalid_validation)}")

        source = _resolve_signal_csv_source(data_dir, signal_name, cfg=cfg)
        if source is None:
            print(f"[forecast] skip '{signal_name}': no matching train/test source found.")
            continue

        if not bool(cfg.forecast.auto_train_missing):
            continue

        print(f"[forecast] retraining signal={signal_name}")
        trained_result = _train_signal_lstm(
            cfg,
            signal_name,
            device=device,
            overrides=dict(resolved_overrides_by_signal.get(signal_name) or {}) or None,
        )
        trained_results.append(trained_result)
        if invalid_validations:
            retrained_signals.append(signal_name)
        else:
            trained_signals.append(signal_name)

    inventory_after = _collect_lstm_artifact_inventory(cfg, overrides_by_signal=resolved_overrides_by_signal)
    if not bool(cfg.forecast.auto_train_missing) and (
        inventory_after["missing_signals"] or inventory_after["invalid_artifacts"]
    ):
        _raise_lstm_artifact_requirements_error(cfg, inventory_after)

    evaluations = [result["evaluation"] for result in trained_results]
    plot_path = get_weekly_forecast_plot_path(
        future_horizon=int(cfg.env.future_horizon),
        root=artifact_root,
    )
    if evaluations:
        plot_path = _plot_weekly_forecasts(
            evaluations,
            save_path=plot_path,
        )
    elif not plot_path.exists():
        plot_path = None

    return {
        "mode": "managed_multi_signal",
        "artifacts": dict(inventory_after["artifacts"]),
        "trained_signals": trained_signals,
        "retrained_signals": retrained_signals,
        "mismatched_signals": list(inventory_before["mismatched_signals"]),
        "invalid_artifacts": dict(inventory_before["invalid_artifacts"]),
        "missing_signals": list(inventory_after["missing_signals"]),
        "plot_path": str(plot_path) if plot_path is not None else None,
    }


__all__ = [
    "_artifact_tuple_from_paths",
    "_collect_lstm_artifact_inventory",
    "_format_lstm_artifact_issue",
    "_format_lstm_artifact_mismatches",
    "_managed_lstm_artifact_paths",
    "_raise_lstm_artifact_requirements_error",
    "_signal_artifact_specs",
    "collect_available_lstm_artifacts",
    "compare_lstm_artifact_meta",
    "ensure_lstm_artifacts",
    "expected_lstm_artifact_meta",
    "forecast_artifact_root",
    "validate_lstm_artifact",
]
