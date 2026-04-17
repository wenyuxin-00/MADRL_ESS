"""Cleanup helper for shared MADRL data directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.utils.madrl_shared_data import _SCHEMA_VERSION, _shared_data_root


def _iter_cleanup_targets(root: Path):
    for entry in root.iterdir():
        name = entry.name
        if name.endswith(".lock") or ".tmp." in name:
            yield entry


def _iter_legacy_schema_dirs(root: Path):
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(manifest.get("schema_version", 0)) < int(_SCHEMA_VERSION):
            yield entry


def cleanup_shared_data_root(
    root: str | Path | None = None,
    *,
    purge_legacy_schema: bool = False,
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
    if purge_legacy_schema:
        for target in _iter_legacy_schema_dirs(shared_root):
            shutil.rmtree(target, ignore_errors=True)
            removed.append(str(target))
    return {
        "root": str(shared_root),
        "removed": removed,
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean up shared MADRL data artifacts.")
    parser.add_argument("--root")
    parser.add_argument("--mainline", action="store_true")
    parser.add_argument("--purge-legacy-schema", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    result = cleanup_shared_data_root(
        None if args.mainline else args.root,
        purge_legacy_schema=bool(args.purge_legacy_schema),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
