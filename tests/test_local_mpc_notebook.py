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


def test_local_mpc_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "local_MPC.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 6
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_local_mpc_notebook_contains_config_and_rollout_calls():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "local_MPC.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    assert "TEST_START_DATE = \"2020-06-01\"" in joined_source
    assert "TEST_END_DATE = \"2020-06-04\"" in joined_source
    assert "BASE_PREDICTION_MODE = \"normal\"" in joined_source
    assert "SAVE_LOCAL_MPC_ROLLOUT" not in joined_source
    assert "LOCAL_MPC_PERFECT_TAG_OVERRIDE" not in joined_source
    assert "LOCAL_MPC_LSTM_TAG_OVERRIDE" not in joined_source
    assert "grid_nb.apply_notebook_experiment_settings(" in joined_source
    assert "grid_nb.ensure_forecast_ready(normal_comparison_cfg)" in joined_source
    assert "collect_local_mpc_rollout(" in joined_source
    assert "LOCAL_MPC_PERFECT_LABEL" in joined_source
    assert "LOCAL_MPC_LSTM_LABEL" in joined_source
    assert "build_local_mpc_rollout_package(" not in joined_source
    assert "save_local_mpc_rollout_package(" not in joined_source
    assert "local_mpc_cached_rollout" not in joined_source
    assert "local_mpc_rollout_summary" in joined_source
    assert "compare_rollout_metrics(local_mpc_perfect, local_mpc_lstm)" in joined_source
    assert re.search(r'TEST_START_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
    assert re.search(r'TEST_END_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
