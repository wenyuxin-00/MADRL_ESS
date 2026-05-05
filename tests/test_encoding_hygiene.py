from __future__ import annotations

import json
from pathlib import Path


TEXT_ROOTS = ("configs", "controllers", "data", "envs", "models", "notebooks", "predictors", "scripts", "tests", "utils")
TEXT_SUFFIXES = {".py", ".md", ".ipynb"}


def _text_paths() -> list[Path]:
    repo = Path(__file__).resolve().parents[1]
    paths: list[Path] = [repo / "README.md", repo / "remake_test.md"]
    for root_name in TEXT_ROOTS:
        root = repo / root_name
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in TEXT_SUFFIXES)
    return sorted(set(paths))


def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook.get("cells", [])
    )


def test_text_files_are_utf8_without_bom() -> None:
    offenders = [str(path) for path in _text_paths() if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert offenders == []


def test_text_files_do_not_contain_mojibake_markers() -> None:
    offenders: list[str] = []
    placeholder = "?" * 3
    for path in _text_paths():
        text = path.read_text(encoding="utf-8")
        if "\ufffd" in text or placeholder in text:
            offenders.append(str(path))
    assert offenders == []


def test_notebooks_parse_as_json_and_have_text_cells() -> None:
    notebooks = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
    assert {path.name for path in notebooks} == {"compare.ipynb", "madrl.ipynb", "madrl_base.ipynb", "madrl_base_safe.ipynb", "madrl_projection_safe.ipynb", "misocp.ipynb", "mpc.ipynb", "predict.ipynb"}
    for path in notebooks:
        assert _notebook_text(path).strip()
