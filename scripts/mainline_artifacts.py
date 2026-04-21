from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Any, Iterator
from scripts.utils.madrl_shared_data import _shared_data_root
from scripts.utils.project_paths import get_artifact_root, project_root
_PROTECTION_MANIFEST_RELATIVE_PATH = Path('artifacts') / '_protection' / 'latest_results_manifest.json'
_SUPPORTED_FORECAST_SIGNAL_DIRS = frozenset({'wholesale_price', 'load', 'pv'})
_FORECAST_KEEP_DIRS = frozenset({'exports', 'plots'})
_LEGACY_TEMP_ARTIFACT_DIR_NAMES = ('tmp_oracle_smoke', 'tmp_oracle_smoke_2')
_TRASH_BATCH_PREFIX = 'batch-'
_TRASH_MANIFEST_FILENAME = 'manifest.json'
_TRASH_PAYLOAD_DIRNAME = 'payload'

def _iter_cleanup_targets(root: Path):
    for entry in root.iterdir():
        name = entry.name
        if name.endswith('.lock') or '.tmp.' in name:
            yield entry

def _resolve_repo_root(root: str | Path | None=None) -> Path:
    return (Path(root) if root is not None else project_root()).resolve()

def cleanup_shared_data_root(root: str | Path | None=None) -> dict[str, object]:
    shared_root = _shared_data_root(root)
    if not shared_root.exists():
        return {'root': str(shared_root), 'removed': []}
    removed: list[str] = []
    for target in _iter_cleanup_targets(shared_root):
        shutil.rmtree(target, ignore_errors=True)
        removed.append(str(target))
    return {'root': str(shared_root), 'removed': removed}

def _manifest_path_for_repo(repo_root: Path, manifest_path: str | Path | None=None) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    return (repo_root / _PROTECTION_MANIFEST_RELATIVE_PATH).resolve()

def _load_protection_manifest(repo_root: Path, manifest_path: str | Path | None=None) -> tuple[Path, dict[str, Any]]:
    target_path = _manifest_path_for_repo(repo_root, manifest_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Artifact cleanup requires the latest-results protection manifest. Expected '{target_path}'.")
    return (target_path, json.loads(target_path.read_text(encoding='utf-8')))

def _resolve_manifest_entry(repo_root: Path, entry: str | None) -> Path | None:
    if not entry:
        return None
    return (repo_root / Path(entry)).resolve()

def _iter_manifest_artifact_paths(payload: Any) -> Iterator[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == 'excluded_runtime_dirs':
                continue
            if key in {'dir', 'artifact_root', 'active_horizon_dir', 'integrity_path', 'path'} and isinstance(value, str):
                if value.startswith('artifacts/'):
                    yield value
                continue
            yield from _iter_manifest_artifact_paths(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_manifest_artifact_paths(item)

def _stringify_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in sorted(paths)]

def build_mainline_artifact_cleanup_plan(root: str | Path | None=None, *, manifest_path: str | Path | None=None) -> dict[str, object]:
    repo_root = _resolve_repo_root(root)
    artifact_root = get_artifact_root(repo_root).resolve()
    loaded_manifest_path, protection_manifest = _load_protection_manifest(repo_root, manifest_path)
    protected_outputs = dict(protection_manifest.get('protected_outputs') or {})
    protected_shared_data_dir = _resolve_manifest_entry(repo_root, dict(protected_outputs.get('shared_data') or {}).get('dir'))
    protected_plan_dirs = [resolved for resolved in (_resolve_manifest_entry(repo_root, dict(entry).get('dir')) for entry in list(protected_outputs.get('misocp_cached_plans') or [])) if resolved is not None]
    forecast_info = dict(protected_outputs.get('forecast') or {})
    active_horizon_dir = _resolve_manifest_entry(repo_root, forecast_info.get('active_horizon_dir'))
    shared_runtime_dirs: list[Path] = []
    shared_data_old_dirs: list[Path] = []
    shared_root = _shared_data_root(repo_root)
    if shared_root.exists():
        shared_runtime_dirs.extend((path.resolve() for path in _iter_cleanup_targets(shared_root)))
        for entry in shared_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.endswith('.lock') or '.tmp.' in entry.name:
                continue
            if protected_shared_data_dir is not None and entry.resolve() == protected_shared_data_dir:
                continue
            shared_data_old_dirs.append(entry.resolve())
    misocp_cached_plan_old_dirs: list[Path] = []
    cached_plan_root = artifact_root / 'misocp_cached_plan'
    protected_plan_dir_set = {path.resolve() for path in protected_plan_dirs}
    if cached_plan_root.exists():
        for entry in cached_plan_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.resolve() in protected_plan_dir_set:
                continue
            misocp_cached_plan_old_dirs.append(entry.resolve())
    misocp_debug_dirs: list[Path] = []
    debug_root = artifact_root / 'misocp_debug'
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
    candidate_paths = {'shared_data_runtime_dirs': _stringify_paths(shared_runtime_dirs), 'shared_data_old_dirs': _stringify_paths(shared_data_old_dirs), 'misocp_cached_plan_old_dirs': _stringify_paths(misocp_cached_plan_old_dirs), 'misocp_debug_dirs': _stringify_paths(misocp_debug_dirs), 'forecast_legacy_dirs': _stringify_paths(forecast_legacy_dirs), 'temp_artifact_dirs': _stringify_paths(temp_artifact_dirs)}
    protected_paths = sorted({str((repo_root / Path(relative_path)).resolve()) for relative_path in _iter_manifest_artifact_paths(protection_manifest)})
    return {'root': str(repo_root), 'artifact_root': str(artifact_root), 'manifest_path': str(loaded_manifest_path), 'protected': protected_paths, 'candidates': candidate_paths, 'candidate_counts': {category: len(paths) for category, paths in candidate_paths.items()}}

def _assert_cleanup_target(target: str | Path, artifact_root: Path) -> Path:
    resolved = Path(target).expanduser().resolve()
    if resolved == artifact_root or artifact_root not in resolved.parents:
        raise ValueError(f"Refusing to remove a path outside the artifact root: target='{resolved}', artifact_root='{artifact_root}'.")
    return resolved

def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()

def _trash_root(artifact_root: Path) -> Path:
    return (artifact_root / '_trash').resolve()

def _batch_root(trash_root: Path, batch_id: str) -> Path:
    return (trash_root / batch_id).resolve()

def _batch_manifest_path(batch_root: Path) -> Path:
    return (batch_root / _TRASH_MANIFEST_FILENAME).resolve()

def _next_trash_batch_id(trash_root: Path) -> str:
    max_index = 0
    if trash_root.exists():
        for entry in trash_root.iterdir():
            if not entry.is_dir() or not entry.name.startswith(_TRASH_BATCH_PREFIX):
                continue
            suffix = entry.name.removeprefix(_TRASH_BATCH_PREFIX)
            try:
                max_index = max(max_index, int(suffix))
            except ValueError:
                continue
    return f'{_TRASH_BATCH_PREFIX}{max_index + 1:04d}'

def _latest_trash_batch_manifest_path(trash_root: Path) -> Path | None:
    if not trash_root.exists():
        return None
    manifests = sorted(((entry / _TRASH_MANIFEST_FILENAME).resolve() for entry in trash_root.iterdir() if entry.is_dir() and entry.name.startswith(_TRASH_BATCH_PREFIX) and (entry / _TRASH_MANIFEST_FILENAME).exists()))
    return manifests[-1] if manifests else None

def _load_trash_batch_manifest(artifact_root: Path, batch_manifest_path: str | Path | None=None) -> tuple[Path, dict[str, Any]] | None:
    if batch_manifest_path is not None:
        target_path = Path(batch_manifest_path).expanduser().resolve()
    else:
        target_path = _latest_trash_batch_manifest_path(_trash_root(artifact_root))
    if target_path is None or not target_path.exists():
        return None
    return (target_path, json.loads(target_path.read_text(encoding='utf-8')))

def _stage_cleanup_batch(plan: dict[str, object]) -> tuple[str | None, Path | None, dict[str, list[str]]]:
    artifact_root = Path(plan['artifact_root']).resolve()
    candidates = {str(category): list(paths) for category, paths in dict(plan['candidates']).items()}
    staged: dict[str, list[str]] = {category: [] for category in candidates}
    existing_targets: list[tuple[str, Path]] = []
    for category, paths in candidates.items():
        for path in paths:
            target = _assert_cleanup_target(path, artifact_root)
            if target.exists():
                existing_targets.append((category, target))
    if not existing_targets:
        return (None, None, staged)
    trash_root = _trash_root(artifact_root)
    trash_root.mkdir(parents=True, exist_ok=True)
    batch_id = _next_trash_batch_id(trash_root)
    batch_root = _batch_root(trash_root, batch_id)
    payload_root = (batch_root / _TRASH_PAYLOAD_DIRNAME).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for category, target in existing_targets:
        relative_target = target.relative_to(artifact_root)
        trash_path = (payload_root / relative_target).resolve()
        trash_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(trash_path))
        staged[category].append(str(target))
        entries.append({'category': category, 'source': str(target), 'relative_path': relative_target.as_posix(), 'trash_path': str(trash_path)})
    batch_manifest = _batch_manifest_path(batch_root)
    batch_manifest.write_text(json.dumps({'batch_id': batch_id, 'artifact_root': str(artifact_root), 'trash_root': str(trash_root), 'protection_manifest_path': str(plan['manifest_path']), 'entries': entries}, indent=2), encoding='utf-8')
    return (batch_id, batch_manifest, staged)

def restore_mainline_artifact_trash(root: str | Path | None=None, *, batch_manifest_path: str | Path | None=None, dry_run: bool=True) -> dict[str, object]:
    repo_root = _resolve_repo_root(root)
    artifact_root = get_artifact_root(repo_root).resolve()
    trash_root = _trash_root(artifact_root)
    loaded_manifest = _load_trash_batch_manifest(artifact_root, batch_manifest_path)
    if loaded_manifest is None:
        return {'root': str(repo_root), 'artifact_root': str(artifact_root), 'trash_root': str(trash_root), 'batch_id': None, 'batch_manifest_path': None, 'dry_run': bool(dry_run), 'restored': [], 'restored_count': 0}
    manifest_path, batch_manifest = loaded_manifest
    restored: list[str] = []
    if not dry_run:
        for entry in list(batch_manifest.get('entries') or []):
            source = _assert_cleanup_target(str(entry['source']), artifact_root)
            trash_path = Path(str(entry['trash_path'])).expanduser().resolve()
            if not trash_path.exists():
                continue
            if source.exists():
                raise FileExistsError(f"Refusing to restore over an existing artifact path: source='{source}'.")
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trash_path), str(source))
            restored.append(str(source))
        _remove_path(manifest_path.parent)
        trash_root.mkdir(parents=True, exist_ok=True)
    return {'root': str(repo_root), 'artifact_root': str(artifact_root), 'trash_root': str(trash_root), 'batch_id': batch_manifest.get('batch_id'), 'batch_manifest_path': str(manifest_path), 'dry_run': bool(dry_run), 'restored': restored, 'restored_count': len(restored)}

def purge_mainline_artifact_trash(root: str | Path | None=None, *, batch_manifest_path: str | Path | None=None, dry_run: bool=True) -> dict[str, object]:
    repo_root = _resolve_repo_root(root)
    artifact_root = get_artifact_root(repo_root).resolve()
    trash_root = _trash_root(artifact_root)
    loaded_manifest = _load_trash_batch_manifest(artifact_root, batch_manifest_path)
    if loaded_manifest is None:
        return {'root': str(repo_root), 'artifact_root': str(artifact_root), 'trash_root': str(trash_root), 'batch_id': None, 'batch_manifest_path': None, 'dry_run': bool(dry_run), 'purged': [], 'purged_count': 0}
    manifest_path, batch_manifest = loaded_manifest
    purged = [str(entry['trash_path']) for entry in list(batch_manifest.get('entries') or [])]
    if not dry_run:
        _remove_path(manifest_path.parent)
        trash_root.mkdir(parents=True, exist_ok=True)
    return {'root': str(repo_root), 'artifact_root': str(artifact_root), 'trash_root': str(trash_root), 'batch_id': batch_manifest.get('batch_id'), 'batch_manifest_path': str(manifest_path), 'dry_run': bool(dry_run), 'purged': purged if not dry_run else [], 'purged_count': len(purged) if not dry_run else 0}

def cleanup_mainline_artifacts(root: str | Path | None=None, *, manifest_path: str | Path | None=None, dry_run: bool=True) -> dict[str, object]:
    plan = build_mainline_artifact_cleanup_plan(root, manifest_path=manifest_path)
    artifact_root = Path(plan['artifact_root']).resolve()
    staged: dict[str, list[str]] = {str(category): [] for category in dict(plan['candidates']).keys()}
    batch_id: str | None = None
    batch_manifest: Path | None = None
    if not dry_run:
        batch_id, batch_manifest, staged = _stage_cleanup_batch(plan)
    return {**plan, 'dry_run': bool(dry_run), 'trash_root': str(_trash_root(artifact_root)), 'batch_id': batch_id, 'batch_manifest_path': str(batch_manifest) if batch_manifest is not None else None, 'staged': staged, 'staged_count': sum((len(paths) for paths in staged.values()))}
