from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.utils.cleanup_shared_data import (
    _build_cli,
    build_mainline_artifact_cleanup_plan,
    cleanup_mainline_artifacts,
    cleanup_shared_data_root,
)


def _shared_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts" / "training" / "shared_data" / "mainline"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_mainline_cleanup_tree(tmp_path: Path) -> dict[str, Path]:
    shared_root = _shared_root(tmp_path)
    keep_shared = shared_root / "current_schema_dir"
    old_shared = shared_root / "legacy_schema_dir"
    lock_dir = shared_root / "abc123.lock"
    tmp_dir = shared_root / "abc123.tmp.worker"

    protected_plan = tmp_path / "artifacts" / "misocp_cached_plan" / "latest_plan"
    backup_plan = tmp_path / "artifacts" / "misocp_cached_plan" / "recent_backup"
    old_plan = tmp_path / "artifacts" / "misocp_cached_plan" / "legacy_plan"
    debug_dir = tmp_path / "artifacts" / "misocp_debug" / "debug_run_1"

    active_horizon = tmp_path / "artifacts" / "forecast" / "lstm" / "h24"
    wholesale_dir = active_horizon / "wholesale_price"
    load_dir = active_horizon / "load"
    pv_dir = active_horizon / "pv"
    price_dir = active_horizon / "price"
    plots_dir = active_horizon / "plots"
    exports_dir = active_horizon / "exports"

    for path in (
        keep_shared,
        old_shared,
        lock_dir,
        tmp_dir,
        protected_plan,
        backup_plan,
        old_plan,
        debug_dir,
        wholesale_dir,
        load_dir,
        pv_dir,
        price_dir,
        plots_dir,
        exports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    _write_json(keep_shared / "manifest.json", {"schema_version": 3})
    _write_json(old_shared / "manifest.json", {"schema_version": 0})
    _write_json(protected_plan / "manifest.json", {"plan_package_version": 12})
    _write_json(backup_plan / "manifest.json", {"plan_package_version": 12})
    _write_json(old_plan / "manifest.json", {"plan_package_version": 1})
    _write_json(pv_dir / "pv_lstm_h24_meta.json", {"signal_name": "pv"})

    protection_manifest = {
        "protected_outputs": {
            "forecast": {
                "artifact_root": "artifacts/forecast/lstm",
                "active_horizon_dir": "artifacts/forecast/lstm/h24",
                "integrity_checks": [
                    {"path": "artifacts/forecast/lstm/h24/pv/pv_lstm_h24_meta.json"}
                ],
            },
            "shared_data": {
                "dir": "artifacts/training/shared_data/mainline/current_schema_dir",
                "excluded_runtime_dirs": [
                    "artifacts/training/shared_data/mainline/abc123.lock",
                    "artifacts/training/shared_data/mainline/abc123.tmp.worker",
                ],
            },
            "misocp_cached_plans": [
                {"dir": "artifacts/misocp_cached_plan/latest_plan"},
                {"dir": "artifacts/misocp_cached_plan/recent_backup"},
            ],
        }
    }
    _write_json(tmp_path / "artifacts" / "_protection" / "latest_results_manifest.json", protection_manifest)

    return {
        "keep_shared": keep_shared,
        "old_shared": old_shared,
        "lock_dir": lock_dir,
        "tmp_dir": tmp_dir,
        "protected_plan": protected_plan,
        "backup_plan": backup_plan,
        "old_plan": old_plan,
        "debug_dir": debug_dir,
        "wholesale_dir": wholesale_dir,
        "load_dir": load_dir,
        "pv_dir": pv_dir,
        "price_dir": price_dir,
        "plots_dir": plots_dir,
        "exports_dir": exports_dir,
    }


def test_cleanup_shared_data_root_removes_only_runtime_dirs(tmp_path: Path) -> None:
    shared_root = _shared_root(tmp_path)
    lock_dir = shared_root / "abc123.lock"
    tmp_dir = shared_root / "abc123.tmp.worker"
    legacy_dir = shared_root / "legacy_schema_dir"
    current_dir = shared_root / "current_schema_dir"

    for path in (lock_dir, tmp_dir, legacy_dir, current_dir):
        path.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "manifest.json").write_text('{"schema_version": 0}', encoding="utf-8")
    (current_dir / "manifest.json").write_text('{"schema_version": 999}', encoding="utf-8")

    result = cleanup_shared_data_root(tmp_path)

    assert result["root"] == str(shared_root)
    assert sorted(result["removed"]) == sorted([str(lock_dir), str(tmp_dir)])
    assert not lock_dir.exists()
    assert not tmp_dir.exists()
    assert legacy_dir.exists()
    assert current_dir.exists()


def test_cleanup_shared_data_root_returns_empty_when_root_missing(tmp_path: Path) -> None:
    result = cleanup_shared_data_root(tmp_path)

    assert result == {
        "root": str(_shared_root(tmp_path)),
        "removed": [],
    }


def test_build_mainline_artifact_cleanup_plan_respects_protection_manifest(tmp_path: Path) -> None:
    paths = _make_mainline_cleanup_tree(tmp_path)

    result = build_mainline_artifact_cleanup_plan(tmp_path)

    assert result["root"] == str(tmp_path.resolve())
    assert str(paths["keep_shared"]) not in result["candidates"]["shared_data_old_dirs"]
    assert str(paths["old_shared"]) in result["candidates"]["shared_data_old_dirs"]
    assert str(paths["lock_dir"]) in result["candidates"]["shared_data_runtime_dirs"]
    assert str(paths["tmp_dir"]) in result["candidates"]["shared_data_runtime_dirs"]
    assert str(paths["protected_plan"]) not in result["candidates"]["misocp_cached_plan_old_dirs"]
    assert str(paths["backup_plan"]) not in result["candidates"]["misocp_cached_plan_old_dirs"]
    assert str(paths["old_plan"]) in result["candidates"]["misocp_cached_plan_old_dirs"]
    assert str(paths["debug_dir"]) in result["candidates"]["misocp_debug_dirs"]
    assert str(paths["price_dir"]) in result["candidates"]["forecast_legacy_dirs"]
    assert str(paths["wholesale_dir"]) not in result["candidates"]["forecast_legacy_dirs"]
    assert str(paths["load_dir"]) not in result["candidates"]["forecast_legacy_dirs"]
    assert str(paths["pv_dir"]) not in result["candidates"]["forecast_legacy_dirs"]
    assert str(paths["plots_dir"]) not in result["candidates"]["forecast_legacy_dirs"]
    assert str(paths["exports_dir"]) not in result["candidates"]["forecast_legacy_dirs"]


def test_cleanup_mainline_artifacts_dry_run_does_not_delete_candidates(tmp_path: Path) -> None:
    paths = _make_mainline_cleanup_tree(tmp_path)

    result = cleanup_mainline_artifacts(tmp_path, dry_run=True)

    assert result["dry_run"] is True
    assert result["removed_count"] == 0
    assert paths["old_shared"].exists()
    assert paths["old_plan"].exists()
    assert paths["debug_dir"].exists()
    assert paths["price_dir"].exists()


def test_cleanup_mainline_artifacts_removes_only_unprotected_candidates(tmp_path: Path) -> None:
    paths = _make_mainline_cleanup_tree(tmp_path)

    result = cleanup_mainline_artifacts(tmp_path, dry_run=False)

    assert result["dry_run"] is False
    assert result["removed_count"] == 6
    assert not paths["lock_dir"].exists()
    assert not paths["tmp_dir"].exists()
    assert not paths["old_shared"].exists()
    assert not paths["old_plan"].exists()
    assert not paths["debug_dir"].exists()
    assert not paths["price_dir"].exists()
    assert paths["keep_shared"].exists()
    assert paths["protected_plan"].exists()
    assert paths["backup_plan"].exists()
    assert paths["wholesale_dir"].exists()
    assert paths["load_dir"].exists()
    assert paths["pv_dir"].exists()


def test_cleanup_shared_data_cli_rejects_removed_legacy_schema_flag() -> None:
    parser = _build_cli()

    with pytest.raises(SystemExit):
        parser.parse_args(["--purge-legacy-schema"])
