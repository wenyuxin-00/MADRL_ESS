"""Cleanup helpers for protected MADRL artifact directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

from scripts.utils.madrl_shared_data import _shared_data_root
from scripts.utils.project_paths import get_artifact_root, project_root

_PROTECTION_MANIFEST_RELATIVE_PATH = Path("artifacts") / "_protection" / "latest_results_manifest.json"
_SUPPORTED_FORECAST_SIGNAL_DIRS = frozenset({"wholesale_price", "load", "pv"})
_FORECAST_KEEP_DIRS = frozenset({"exports", "plots"})
_LEGACY_TEMP_ARTIFACT_DIR_NAMES = ("tmp_oracle_smoke", "tmp_oracle_smoke_2")


def _iter_cleanup_targets(root: Path):
    for entry in root.iterdir():
        name = entry.name
        if name.endswith(".lock") or ".tmp." in name:
            yield entry


def _resolve_repo_root(root: str | Path | None = None) -> Path:
    return (Path(root) if root is not None else project_root()).resolve()


def cleanup_shared_data_root(
    root: str | Path | None = None,
) -> dict[str, object]:
    shared_root = _shared_data_root(root)
    if not shared_root.exists():
        return {
            "root": str(shared_root),
            "removed": [],
        }

    removed: list[str] = []
    for target in _iter_cleanup_targets(shared_root):
        shutil.rmtree(target, ignore_errors=True)
        removed.append(str(target))
    return {
        "root": str(shared_root),
        "removed": removed,
    }


def _manifest_path_for_repo(repo_root: Path, manifest_path: str | Path | None = None) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    return (repo_root / _PROTECTION_MANIFEST_RELATIVE_PATH).resolve()


def _load_protection_manifest(
    repo_root: Path,
    manifest_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    target_path = _manifest_path_for_repo(repo_root, manifest_path)
    if not target_path.exists():
        raise FileNotFoundError(
            "Artifact cleanup requires the latest-results protection manifest. "
            f"Expected '{target_path}'."
        )
    return target_path, json.loads(target_path.read_text(encoding="utf-8"))


def _resolve_manifest_entry(repo_root: Path, entry: str | None) -> Path | None:
    if not entry:
        return None
    return (repo_root / Path(entry)).resolve()


def _iter_manifest_artifact_paths(payload: Any) -> Iterator[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "excluded_runtime_dirs":
                continue
            if key in {"dir", "artifact_root", "active_horizon_dir", "integrity_path", "path"} and isinstance(value, str):
                if value.startswith("artifacts/"):
                    yield value
                continue
            yield from _iter_manifest_artifact_paths(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_manifest_artifact_paths(item)


def _stringify_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in sorted(paths)]


def build_mainline_artifact_cleanup_plan(
    root: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    repo_root = _resolve_repo_root(root)
    artifact_root = get_artifact_root(repo_root).resolve()
    loaded_manifest_path, protection_manifest = _load_protection_manifest(repo_root, manifest_path)
    protected_outputs = dict(protection_manifest.get("protected_outputs") or {})

    protected_shared_data_dir = _resolve_manifest_entry(repo_root, dict(protected_outputs.get("shared_data") or {}).get("dir"))
    protected_plan_dirs = [
        resolved
        for resolved in (
            _resolve_manifest_entry(repo_root, dict(entry).get("dir"))
            for entry in list(protected_outputs.get("misocp_cached_plans") or [])
        )
        if resolved is not None
    ]
    forecast_info = dict(protected_outputs.get("forecast") or {})
    active_horizon_dir = _resolve_manifest_entry(repo_root, forecast_info.get("active_horizon_dir"))

    shared_runtime_dirs: list[Path] = []
    shared_data_old_dirs: list[Path] = []
    shared_root = _shared_data_root(repo_root)
    if shared_root.exists():
        shared_runtime_dirs.extend(path.resolve() for path in _iter_cleanup_targets(shared_root))
        for entry in shared_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.endswith(".lock") or ".tmp." in entry.name:
                continue
            if protected_shared_data_dir is not None and entry.resolve() == protected_shared_data_dir:
                continue
            shared_data_old_dirs.append(entry.resolve())

    misocp_cached_plan_old_dirs: list[Path] = []
    cached_plan_root = artifact_root / "misocp_cached_plan"
    protected_plan_dir_set = {path.resolve() for path in protected_plan_dirs}
    if cached_plan_root.exists():
        for entry in cached_plan_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.resolve() in protected_plan_dir_set:
                continue
            misocp_cached_plan_old_dirs.append(entry.resolve())

    misocp_debug_dirs: list[Path] = []
    debug_root = artifact_root / "misocp_debug"
    if debug_root.exists():
        for entry in debug_root.iterdir():
            misocp_debug_dirs.append(entry.resolve())

    forecast_legacy_dirs: list[Path] = []
    if active_horizon_dir is not None and active_horizon_dir.exists():
        for entry in active_horizon_dir.iterdir():
            if not entry.is_dir():
                continue
            if entry.name in _SUPPORTED_FORECAST_SIGNAL_DIRS or entry.name in _FORECAST_KEEP_DIRS:
                continue
            forecast_legacy_dirs.append(entry.resolve())

    temp_artifact_dirs: list[Path] = []
    for dir_name in _LEGACY_TEMP_ARTIFACT_DIR_NAMES:
        candidate = (artifact_root / dir_name).resolve()
        if candidate.exists():
            temp_artifact_dirs.append(candidate)

    candidate_paths = {
        "shared_data_runtime_dirs": _stringify_paths(shared_runtime_dirs),
        "shared_data_old_dirs": _stringify_paths(shared_data_old_dirs),
        "misocp_cached_plan_old_dirs": _stringify_paths(misocp_cached_plan_old_dirs),
        "misocp_debug_dirs": _stringify_paths(misocp_debug_dirs),
        "forecast_legacy_dirs": _stringify_paths(forecast_legacy_dirs),
        "temp_artifact_dirs": _stringify_paths(temp_artifact_dirs),
    }
    protected_paths = sorted(
        {
            str((repo_root / Path(relative_path)).resolve())
            for relative_path in _iter_manifest_artifact_paths(protection_manifest)
        }
    )
    return {
        "root": str(repo_root),
        "artifact_root": str(artifact_root),
        "manifest_path": str(loaded_manifest_path),
        "protected": protected_paths,
        "candidates": candidate_paths,
        "candidate_counts": {
            category: len(paths)
            for category, paths in candidate_paths.items()
        },
    }


def _assert_cleanup_target(target: str | Path, artifact_root: Path) -> Path:
    resolved = Path(target).expanduser().resolve()
    if resolved == artifact_root or artifact_root not in resolved.parents:
        raise ValueError(
            "Refusing to remove a path outside the artifact root: "
            f"target='{resolved}', artifact_root='{artifact_root}'."
        )
    return resolved


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def cleanup_mainline_artifacts(
    root: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, object]:
    plan = build_mainline_artifact_cleanup_plan(root, manifest_path=manifest_path)
    artifact_root = Path(plan["artifact_root"]).resolve()
    removed: dict[str, list[str]] = {category: [] for category in dict(plan["candidates"]).keys()}

    if not dry_run:
        for category, paths in dict(plan["candidates"]).items():
            for path in list(paths):
                target = _assert_cleanup_target(path, artifact_root)
                if not target.exists():
                    continue
                _remove_path(target)
                removed[str(category)].append(str(target))

    return {
        **plan,
        "dry_run": bool(dry_run),
        "removed": removed,
        "removed_count": sum(len(paths) for paths in removed.values()),
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean up shared MADRL data artifacts.")
    parser.add_argument("--root")
    parser.add_argument("--mainline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.mainline:
        result = cleanup_mainline_artifacts(args.root, dry_run=bool(args.dry_run))
    else:
        result = cleanup_shared_data_root(args.root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
