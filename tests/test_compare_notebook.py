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


def test_compare_notebook_code_cells_compile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "compare.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 2
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_compare_notebook_reads_saved_records_instead_of_recomputing() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "compare.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    required_tokens = [
        "COMPARE_SCHEME_ORDER",
        "RECORD_SCHEME_CATEGORIES",
        "load_rollout_record(",
        "compare_rollout_metrics(*rollouts)",
        "build_compare_economic_table",
        "build_compare_safety_table",
    ]
    forbidden_tokens = [
        "collect_global_full_horizon_rollout",
        "collect_local_mpc_rollout",
        "collect_admm_mpc_rollout",
        "collect_madrl_rollout",
        "run_external_train_mainline",
        "ensure_madrl_shared_data",
        "resolve_madrl_model_root",
    ]

    for token in required_tokens:
        assert token in joined_source
    for token in forbidden_tokens:
        assert token not in joined_source
