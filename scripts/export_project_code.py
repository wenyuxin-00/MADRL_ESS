"""Export the repository source snapshot to `full_project_code.txt`.

This script is a developer utility and is intentionally kept outside the
training/runtime path. The project root is resolved from the parent directory
of `scripts/` so the export still covers the whole repository after the move.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
    ".pytest_tmp",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    "tests",
    ".env",
    ".conda",
    ".tox",
    ".nox",
    "data",
    "runs",
    "saved_models",
    "checkpoints",
    "logs",
    "results",
    "outputs",
    "tests_runtime",
    "htmlcov",
    "dist",
    "build",
    "node_modules",
    ".coverage",
}

ALLOWED_EXTENSIONS = {
    ".ipynb",
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".toml",
    ".sh",
    ".bat",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "full_project_code.txt"


def is_environment_dir(path: Path) -> bool:
    """Skip virtual environments and generated caches."""
    if path.name in IGNORE_DIRS:
        return True

    return (
        (path / "pyvenv.cfg").exists()
        or (path / "conda-meta").is_dir()
        or (path / "Lib" / "site-packages").is_dir()
        or (path / "Scripts" / "python.exe").exists()
        or (path / "bin" / "python").exists()
    )


def should_export_file(path: Path) -> bool:
    if path.name == OUTPUT_FILE.name:
        return False
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def normalize_source(source: str | list[str]) -> str:
    if isinstance(source, list):
        return "".join(source)
    return source


def read_export_content(path: Path) -> str:
    if path.suffix.lower() != ".ipynb":
        return path.read_text(encoding="utf-8")

    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    sections: list[str] = []

    for index, cell in enumerate(cells, start=1):
        source = normalize_source(cell.get("source", ""))
        if not source.strip():
            continue

        cell_type = cell.get("cell_type", "unknown").upper()
        sections.append(f"[{cell_type} CELL {index}]\n")
        sections.append(source.rstrip())
        sections.append("\n\n")

    if not sections:
        return "Notebook has no non-empty cells.\n"
    return "".join(sections).rstrip() + "\n"


def export_project_code() -> None:
    """Export repository source files and notebook cell contents."""
    with OUTPUT_FILE.open("w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)
            dirs[:] = sorted(
                [directory for directory in dirs if not is_environment_dir(root_path / directory)]
            )

            for file_name in sorted(files):
                file_path = root_path / file_name
                if not should_export_file(file_path):
                    continue

                relative_path = file_path.relative_to(PROJECT_ROOT)
                outfile.write(f"\n\n{'=' * 60}\n")
                outfile.write(f"File: {relative_path}\n")
                outfile.write(f"{'=' * 60}\n\n")

                try:
                    outfile.write(read_export_content(file_path))
                except Exception as exc:  # pragma: no cover - developer utility fallback
                    outfile.write(f"Error reading file: {exc}\n")

    message = f"All project code has been exported to {OUTPUT_FILE.name}"
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((message + "\n").encode("utf-8"))


if __name__ == "__main__":
    export_project_code()
