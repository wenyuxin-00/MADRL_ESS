from __future__ import annotations

import json
import re
from pathlib import Path


def _load_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def test_admm_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "ADMM.ipynb"
    assert not (repo_root / "notebooks" / "madrl" / "Dual_test.ipynb").exists()
    assert not (repo_root / "notebooks" / "madrl" / "Dual_mpc.ipynb").exists()
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 6
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_admm_notebook_contains_direct_day_calls_and_string_dates():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "ADMM.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    assert "TEST_START_DATE = \"2020-06-01\"" in joined_source
    assert "TEST_END_DATE = \"2020-06-05\"" in joined_source
    assert re.search(r'TEST_START_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
    assert re.search(r'TEST_END_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
    assert "admm_direct_notebook_helpers as direct_nb" in joined_source
    assert "dual_direct_notebook_helpers" not in joined_source
    assert "build_direct_day_problem_data(" in joined_source
    assert "build_direct_day_trafo_surrogate(" in joined_source
    assert "solve_direct_day_centralized(" in joined_source
    assert "solve_direct_day_admm(" in joined_source
    assert "plot_direct_day_power_energy_balance(" in joined_source
    assert "plot_direct_day_voltage_profile(" in joined_source
    assert "plot_direct_day_net_load(" in joined_source
    assert "plot_direct_day_root_p_comparison(" in joined_source
