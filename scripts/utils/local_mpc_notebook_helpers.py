"""Cached rollout helpers for the local_MPC notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils.price_protocol import IMPORT_PRICE_MARKUP_KEY
from scripts.utils.rollout_package_utils import (
    ROLLOUT_PACKAGE_VERSION,
    assert_rollout_cfg_snapshot_matches,
    assert_solver_fingerprint_matches,
    build_rollout_cfg_snapshot_from_cfg,
    json_default,
    load_dataframe_npz,
    relabel_rollout_dataframe,
    save_dataframe_npz,
    table_required_files,
)


DEFAULT_LOCAL_MPC_PRICE_MODE = "import_adjusted"
DEFAULT_LOCAL_MPC_OBJECTIVE_MODE = "economic_only"
LOCAL_MPC_PERFECT_LABEL = "Local MPC + Perfect Forecast"
LOCAL_MPC_LSTM_LABEL = "Local MPC + LSTM Forecast"


def _build_local_mpc_solver_fingerprint(
    *,
    price_mode: str = DEFAULT_LOCAL_MPC_PRICE_MODE,
    objective_mode: str = DEFAULT_LOCAL_MPC_OBJECTIVE_MODE,
) -> dict[str, Any]:
    return {
        "price_mode": str(price_mode),
        "objective_mode": str(objective_mode),
    }


def _expected_rollout_contract(
    cfg: Any,
    *,
    prediction_mode: str | None = None,
    price_mode: str = DEFAULT_LOCAL_MPC_PRICE_MODE,
    objective_mode: str = DEFAULT_LOCAL_MPC_OBJECTIVE_MODE,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    resolved_prediction_mode = (
        grid_nb.normalize_prediction_mode(prediction_mode)
        if prediction_mode is not None
        else grid_nb.resolve_prediction_mode_from_forecast_backend(str(getattr(cfg.forecast, "type", "perfect")))
    )
    comparison_cfg = grid_nb.build_comparison_cfg(cfg, prediction_mode=resolved_prediction_mode)
    cfg_snapshot = build_rollout_cfg_snapshot_from_cfg(comparison_cfg, prediction_mode=resolved_prediction_mode)
    solver_fingerprint = _build_local_mpc_solver_fingerprint(
        price_mode=price_mode,
        objective_mode=objective_mode,
    )
    return comparison_cfg, cfg_snapshot, solver_fingerprint


def build_local_mpc_rollout_package(
    rollout: grid_nb.RolloutResult,
    *,
    controller_label: str,
    cfg: Any | None = None,
    cfg_snapshot: dict[str, Any] | None = None,
    prediction_mode: str | None = None,
    price_mode: str = DEFAULT_LOCAL_MPC_PRICE_MODE,
    objective_mode: str = DEFAULT_LOCAL_MPC_OBJECTIVE_MODE,
    extra_meta: dict[str, Any] | None = None,
    diagnostic_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a disk-friendly local MPC rollout package from compare-ready tables."""

    if cfg_snapshot is None:
        if cfg is None:
            raise ValueError("Provide either cfg or cfg_snapshot when building a local MPC rollout package.")
        _, cfg_snapshot, _ = _expected_rollout_contract(
            cfg,
            prediction_mode=prediction_mode,
            price_mode=price_mode,
            objective_mode=objective_mode,
        )
    resolved_prediction_mode = str(
        grid_nb.normalize_prediction_mode(
            prediction_mode
            if prediction_mode is not None
            else cfg_snapshot.get("prediction_mode", rollout.meta.get("prediction_mode", "normal"))
        )
    )
    forecast_backend = str(cfg_snapshot.get("forecast_backend", rollout.meta.get("forecast_backend", "perfect")))
    manifest = {
        "rollout_package_version": int(ROLLOUT_PACKAGE_VERSION),
        "controller_label": str(controller_label),
        "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
        "prediction_mode": resolved_prediction_mode,
        "forecast_backend": forecast_backend,
        "cfg_snapshot": dict(cfg_snapshot),
        "local_mpc_solver_fingerprint": _build_local_mpc_solver_fingerprint(
            price_mode=price_mode,
            objective_mode=objective_mode,
        ),
        "extra_meta": dict(extra_meta or {}),
    }
    return {
        "manifest": manifest,
        "diagnostics": dict(diagnostic_summary or {}),
        "step_df": rollout.step_df.copy(),
        "agent_df": rollout.agent_df.copy(),
        "grid_df": rollout.grid_df.copy(),
        "summary_df": rollout.summary.copy(),
    }


def save_local_mpc_rollout_package(package: dict[str, Any], target_dir: str | Path) -> Path:
    """Persist a local MPC rollout package to a deterministic directory layout."""

    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    manifest = dict(package["manifest"])
    diagnostics = dict(package.get("diagnostics", {}))
    step_meta = save_dataframe_npz(pd.DataFrame(package["step_df"]).copy(), target_path / "step_df.npz")
    agent_meta = save_dataframe_npz(pd.DataFrame(package["agent_df"]).copy(), target_path / "agent_df.npz")
    grid_meta = save_dataframe_npz(pd.DataFrame(package["grid_df"]).copy(), target_path / "grid_df.npz")
    summary_meta = save_dataframe_npz(pd.DataFrame(package["summary_df"]).copy(), target_path / "summary_df.npz")
    manifest["tables"] = {
        "step_df": step_meta,
        "agent_df": agent_meta,
        "grid_df": grid_meta,
        "summary_df": summary_meta,
    }

    (target_path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    (target_path / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=json_default), encoding="utf-8")
    return target_path


def load_local_mpc_rollout_package(target_dir: str | Path) -> dict[str, Any]:
    """Load a saved local MPC rollout package without applying cfg compatibility checks."""

    target_path = Path(target_dir).resolve()
    manifest_path = target_path / "manifest.json"
    diagnostics_path = target_path / "diagnostics.json"
    step_path = target_path / "step_df.npz"
    agent_path = target_path / "agent_df.npz"
    grid_path = target_path / "grid_df.npz"
    summary_path = target_path / "summary_df.npz"

    missing_files = [
        path.name
        for path in (manifest_path, diagnostics_path, step_path, agent_path, grid_path, summary_path)
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Local MPC rollout package is incomplete at {target_path}: missing {', '.join(missing_files)}."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("rollout_package_version", -1)) != int(ROLLOUT_PACKAGE_VERSION):
        raise ValueError(
            "Unsupported local MPC rollout package version at "
            f"{target_path}: expected={ROLLOUT_PACKAGE_VERSION}, actual={manifest.get('rollout_package_version')!r}. "
            "Re-run notebooks/madrl/local_MPC.ipynb to regenerate it."
        )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    tables = dict(manifest.get("tables", {}))
    required_table_names = {"step_df", "agent_df", "grid_df", "summary_df"}
    missing_tables = sorted(required_table_names.difference(tables))
    if missing_tables:
        raise ValueError(
            f"Local MPC rollout package manifest at {target_path} is missing table metadata for {', '.join(missing_tables)}."
        )

    return {
        "manifest": manifest,
        "diagnostics": diagnostics,
        "step_df": load_dataframe_npz(step_path, metadata=tables["step_df"]),
        "agent_df": load_dataframe_npz(agent_path, metadata=tables["agent_df"]),
        "grid_df": load_dataframe_npz(grid_path, metadata=tables["grid_df"]),
        "summary_df": load_dataframe_npz(summary_path, metadata=tables["summary_df"]),
    }


def resolve_latest_compatible_local_mpc_rollout_package_dir(
    target_dir: str | Path,
    *,
    cfg: Any | None = None,
    prediction_mode: str | None = None,
    price_mode: str = DEFAULT_LOCAL_MPC_PRICE_MODE,
    objective_mode: str = DEFAULT_LOCAL_MPC_OBJECTIVE_MODE,
) -> Path:
    """Resolve the newest compatible cached local MPC rollout directory sharing the requested prefix."""

    target_path = Path(target_dir).expanduser().resolve()
    parent_dir = target_path.parent
    prefix = target_path.name
    required_names = table_required_files()

    expected_cfg_snapshot = None
    expected_solver_fingerprint = None
    if cfg is not None:
        _, expected_cfg_snapshot, expected_solver_fingerprint = _expected_rollout_contract(
            cfg,
            prediction_mode=prediction_mode,
            price_mode=price_mode,
            objective_mode=objective_mode,
        )

    if not parent_dir.exists():
        raise FileNotFoundError(
            "Cannot resolve cached local MPC rollout prefix "
            f"'{prefix}' because the scan root does not exist: {parent_dir}. "
            "Please run notebooks/madrl/local_MPC.ipynb first or set the local rollout input directory."
        )

    compatible_candidates: list[dict[str, Any]] = []
    discovered_candidates: list[str] = []
    for candidate_dir in parent_dir.iterdir():
        if not candidate_dir.is_dir():
            continue
        if candidate_dir.name != prefix and not candidate_dir.name.startswith(f"{prefix}_"):
            continue

        missing_required = [name for name in required_names if not (candidate_dir / name).exists()]
        if missing_required:
            discovered_candidates.append(f"{candidate_dir.name} (missing={','.join(missing_required)})")
            continue

        manifest_path = candidate_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            discovered_candidates.append(f"{candidate_dir.name} (invalid manifest)")
            continue

        version = int(manifest.get("rollout_package_version", -1))
        if version != int(ROLLOUT_PACKAGE_VERSION):
            discovered_candidates.append(
                f"{candidate_dir.name} (version={version}, expected={ROLLOUT_PACKAGE_VERSION})"
            )
            continue
        try:
            if expected_cfg_snapshot is not None:
                assert_rollout_cfg_snapshot_matches(
                    expected_cfg_snapshot,
                    manifest["cfg_snapshot"],
                    mismatch_prefix="Local MPC rollout package",
                )
            if expected_solver_fingerprint is not None:
                assert_solver_fingerprint_matches(
                    expected_solver_fingerprint,
                    manifest["local_mpc_solver_fingerprint"],
                    name="local_mpc_solver_fingerprint",
                )
        except Exception as exc:
            discovered_candidates.append(f"{candidate_dir.name} ({exc})")
            continue

        saved_at_raw = manifest.get("saved_at_utc")
        try:
            saved_at = pd.Timestamp(saved_at_raw)
            if saved_at.tzinfo is None:
                saved_at = saved_at.tz_localize("UTC")
            else:
                saved_at = saved_at.tz_convert("UTC")
            saved_at_key = int(saved_at.value)
        except Exception:
            saved_at_key = -1
        compatible_candidates.append(
            {
                "path": candidate_dir,
                "saved_at_key": saved_at_key,
                "mtime_ns": int(candidate_dir.stat().st_mtime_ns),
            }
        )

    if compatible_candidates:
        compatible_candidates.sort(
            key=lambda item: (int(item["saved_at_key"]), int(item["mtime_ns"]), str(item["path"].name)),
            reverse=True,
        )
        return Path(compatible_candidates[0]["path"])

    discovered_text = ", ".join(discovered_candidates) if discovered_candidates else "none"
    raise FileNotFoundError(
        "No compatible cached local MPC rollout package was found. "
        f"Expected prefix='{prefix}', scan_root='{parent_dir}', discovered_candidates={discovered_text}. "
        "Please run notebooks/madrl/local_MPC.ipynb to generate a compatible cache."
    )


def replay_local_mpc_rollout_package(
    cfg: Any,
    target_dir: str | Path,
    *,
    label: str | None = None,
    prediction_mode: str | None = None,
) -> dict[str, Any]:
    """Load a cached local MPC rollout package and reconstruct a compare-ready rollout result."""

    package = load_local_mpc_rollout_package(target_dir)
    comparison_cfg, current_cfg_snapshot, current_solver_fingerprint = _expected_rollout_contract(
        cfg,
        prediction_mode=prediction_mode or package["manifest"].get("prediction_mode"),
    )
    assert_rollout_cfg_snapshot_matches(
        current_cfg_snapshot,
        package["manifest"]["cfg_snapshot"],
        mismatch_prefix="Local MPC rollout package",
    )
    assert_solver_fingerprint_matches(
        current_solver_fingerprint,
        package["manifest"]["local_mpc_solver_fingerprint"],
        name="local_mpc_solver_fingerprint",
    )

    controller_label = str(label or package["manifest"]["controller_label"])
    step_df = relabel_rollout_dataframe(package["step_df"], controller_label)
    agent_df = relabel_rollout_dataframe(package["agent_df"], controller_label)
    grid_df = relabel_rollout_dataframe(package["grid_df"], controller_label)
    summary_df = relabel_rollout_dataframe(package["summary_df"], controller_label)
    meta = dict(package["manifest"].get("extra_meta", {}).get("rollout_meta", {}))
    meta.update(
        {
            "controller": controller_label,
            "prediction_mode": str(current_cfg_snapshot["prediction_mode"]),
            "forecast_backend": str(current_cfg_snapshot["forecast_backend"]),
            "future_horizon": int(current_cfg_snapshot["future_horizon"]),
            "agent_profiles": list(current_cfg_snapshot["agent_profiles"]),
            "agent_bus_ids": list(current_cfg_snapshot["agent_bus_ids"]),
            "v_min_pu": float(current_cfg_snapshot["v_min_pu"]),
            "v_max_pu": float(current_cfg_snapshot["v_max_pu"]),
            IMPORT_PRICE_MARKUP_KEY: float(current_cfg_snapshot[IMPORT_PRICE_MARKUP_KEY]),
            "export_subsidy_eur_per_kwh": float(current_cfg_snapshot["export_subsidy_eur_per_kwh"]),
            "loaded_from_cached_rollout": True,
            "rollout_package_dir": str(Path(target_dir).resolve()),
            "rollout_package_version": int(package["manifest"]["rollout_package_version"]),
            "cached_rollout_diagnostics": dict(package["diagnostics"]),
            "cached_local_mpc_solver_fingerprint": dict(package["manifest"]["local_mpc_solver_fingerprint"]),
            "cached_rollout_prediction_mode": str(package["manifest"]["prediction_mode"]),
            "cached_rollout_forecast_backend": str(package["manifest"]["forecast_backend"]),
            "comparison_forecast_backend": str(comparison_cfg.forecast.type),
        }
    )
    rollout = grid_nb.RolloutResult(
        step_df=step_df,
        agent_df=agent_df,
        grid_df=grid_df,
        summary=summary_df,
        meta=meta,
    )
    return {
        "manifest": package["manifest"],
        "diagnostics": package["diagnostics"],
        "rollout": rollout,
    }


__all__ = [
    "DEFAULT_LOCAL_MPC_OBJECTIVE_MODE",
    "DEFAULT_LOCAL_MPC_PRICE_MODE",
    "LOCAL_MPC_LSTM_LABEL",
    "LOCAL_MPC_PERFECT_LABEL",
    "build_local_mpc_rollout_package",
    "load_local_mpc_rollout_package",
    "replay_local_mpc_rollout_package",
    "resolve_latest_compatible_local_mpc_rollout_package_dir",
    "save_local_mpc_rollout_package",
]
