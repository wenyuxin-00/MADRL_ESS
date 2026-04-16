from __future__ import annotations

import json
from pathlib import Path


def _load_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def test_dual_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "Dual.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 4
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_dual_notebook_contains_compare_and_dual_rollout_calls():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "Dual.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    assert "collect_mpc_rollout(" in joined_source
    assert "collect_dual_two_pass_rollout(" in joined_source
    assert "show_progress=True" in joined_source
    assert "compare_rollout_metrics(" in joined_source
    assert "build_compare_economic_table(" in joined_source
    assert "build_compare_safety_table(" in joined_source
    assert "plot_dual_resource_diagnostics(" in joined_source
    assert "dual_delta_curtail_headroom_first_step_total_kw" in joined_source
    assert "dual_round2_achieved_netload_lift_first_step_total_kw" in joined_source
