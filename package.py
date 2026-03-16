import json
import os
import sys
from pathlib import Path

# 要忽略的文件夹
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

# 允许导出的程序文件类型
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

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "full_project_code.txt"


def is_environment_dir(path: Path) -> bool:
    """识别虚拟环境或 Conda 环境目录，避免把底层库代码打进去。"""
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
    sections = []

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
    with OUTPUT_FILE.open("w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(BASE_DIR):
            root_path = Path(root)

            # 提前剪枝，避免继续深入无关目录
            dirs[:] = sorted(
                [
                    directory
                    for directory in dirs
                    if not is_environment_dir(root_path / directory)
                ]
            )

            for file_name in sorted(files):
                file_path = root_path / file_name

                if not should_export_file(file_path):
                    continue

                relative_path = file_path.relative_to(BASE_DIR)
                outfile.write(f"\n\n{'=' * 60}\n")
                outfile.write(f"File: {relative_path}\n")
                outfile.write(f"{'=' * 60}\n\n")

                try:
                    outfile.write(read_export_content(file_path))
                except Exception as exc:
                    outfile.write(f"Error reading file: {exc}\n")

    message = f"所有程序代码已成功导出到 {OUTPUT_FILE.name}"

    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((message + "\n").encode("utf-8"))


if __name__ == "__main__":
    export_project_code()
