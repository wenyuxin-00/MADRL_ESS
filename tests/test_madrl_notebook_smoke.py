from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


@pytest.mark.parametrize(
    "notebook_name",
    [
        "train_base.ipynb",
        "train_base_safe.ipynb",
        "train_projection_safe.ipynb",
    ],
)
def test_madrl_training_notebook_configuration_cells_compile(notebook_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / notebook_name
    code_cells = _load_code_cells(notebook_path)

    for cell_index, source in enumerate(code_cells[:4], start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


@pytest.mark.parametrize(
    ("notebook_name", "expected_num_envs"),
    [
        ("train_base.ipynb", 4),
        ("train_base_safe.ipynb", 8),
        ("train_projection_safe.ipynb", 8),
    ],
)
def test_madrl_training_notebooks_expose_expected_defaults(notebook_name: str, expected_num_envs: int) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / notebook_name
    code_cells = _load_code_cells(notebook_path)
    notebook_source = "\n".join(code_cells)

    assert "ROLLOUT_TEST_START_DATE" not in notebook_source
    assert "ROLLOUT_TEST_END_DATE" not in notebook_source
    assert 'test_start_date = "2020-06-01"' in notebook_source
    assert 'test_end_date = "2020-06-05"' in notebook_source
    assert '"runtime_controls": {}' in notebook_source
    assert "policy_update_freq" in notebook_source
    assert f"num_envs = {expected_num_envs}" in notebook_source
