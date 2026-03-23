"""项目代码导出工具。

将项目所有源代码文件合并导出为单个文本文件，
方便代码审查或提交。
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
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".txt",
    ".cfg",
    ".toml",
    ".sh",
    ".bat",
}

ALLOWED_EXTENSIONS = {
    ".ipynb",
    ".py",

}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "full_project_code.txt"


def is_environment_dir(path: Path) -> bool:
    """Skip virtual environments and generated caches."""
    if path.name in IGNORE_DIRS:
        return True

    # 仅按目录名过滤不够稳妥，所以再看几个“像虚拟环境/依赖目录”的标志文件。
    return (
        (path / "pyvenv.cfg").exists()
        or (path / "conda-meta").is_dir()
        or (path / "Lib" / "site-packages").is_dir()
        or (path / "Scripts" / "python.exe").exists()
        or (path / "bin" / "python").exists()
    )


def should_export_file(path: Path) -> bool:
    # 避免把上一次导出的总文件再次读进来，造成内容无限膨胀。
    if path.name == OUTPUT_FILE.name:
        return False
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def normalize_source(source: str | list[str]) -> str:
    # Jupyter 的 `source` 可能是字符串，也可能是按行拆开的列表，这里统一成字符串。
    if isinstance(source, list):
        return "".join(source)
    return source


def read_export_content(path: Path) -> str:
    # 普通文本文件直接读；Notebook 需要把每个 cell 展平成可阅读的文本结构。
    if path.suffix.lower() != ".ipynb":
        return path.read_text(encoding="utf-8")

    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    sections: list[str] = []

    for index, cell in enumerate(cells, start=1):
        source = normalize_source(cell.get("source", ""))
        if not source.strip():
            # 空 cell 没有信息量，跳过后导出的总文件会更紧凑。
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
            # 原地修改 `dirs` 是 `os.walk` 官方支持的剪枝方式。
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
                    # 这个脚本面向开发辅助，个别文件失败时优先继续导出剩余内容。
                    outfile.write(f"Error reading file: {exc}\n")

    message = f"All project code has been exported to {OUTPUT_FILE.name}"
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((message + "\n").encode("utf-8"))


if __name__ == "__main__":
    export_project_code()

