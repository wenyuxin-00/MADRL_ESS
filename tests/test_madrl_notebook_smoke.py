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
    "notebook_name",
    [
        "train_base.ipynb",
        "train_base_safe.ipynb",
        "train_projection_safe.ipynb",
    ],
)
def test_madrl_training_notebooks_expose_expected_defaults(notebook_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / notebook_name
    code_cells = _load_code_cells(notebook_path)
    parameter_cell = code_cells[1]

    assert "ROLLOUT_TEST_START_DATE" not in parameter_cell
    assert "ROLLOUT_TEST_END_DATE" not in parameter_cell
    assert 'test_start_date = None' in parameter_cell
    assert 'test_end_date = None' in parameter_cell
    assert '"runtime_controls": {}' in parameter_cell
    assert "policy_update_freq" in parameter_cell
    assert "num_envs = recommended_gpu_fast_num_envs()" in parameter_cell
