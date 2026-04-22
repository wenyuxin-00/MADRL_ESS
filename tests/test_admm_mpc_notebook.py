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


def test_admm_mpc_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "ADMM_mpc.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 6
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_admm_mpc_notebook_contains_config_and_rollout_calls():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "ADMM_mpc.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    assert "TEST_START_DATE = \"2020-06-01\"" in joined_source
    assert "TEST_END_DATE = \"2020-06-30\"" in joined_source
    assert "PREDICTION_MODE = \"normal\"" in joined_source
    assert "grid_nb.apply_notebook_experiment_settings(" in joined_source
    assert "collect_admm_mpc_rollout(" in joined_source
    assert "SHOW_PROGRESS = True" in joined_source
    assert "SAVE_ADMM_MPC_ROLLOUT" not in joined_source
    assert "ADMM_MPC_ROLLOUT_TAG_OVERRIDE" not in joined_source
    assert "show_progress=SHOW_PROGRESS" in joined_source
    assert "build_admm_mpc_rollout_package(" not in joined_source
    assert "save_admm_mpc_rollout_package(" not in joined_source
    assert "admm_mpc_rollout_package_dir" not in joined_source
    assert "admm_mpc_rollout_summary" in joined_source
    assert "plot_rollout_dashboard(admm_mpc_rollout)" in joined_source
    assert "plot_voltage_profile_comparison(admm_mpc_rollout)" in joined_source
    assert "plot_net_load_comparison(admm_mpc_rollout)" in joined_source
    assert "plot_battery_power_and_soc_comparison(admm_mpc_rollout)" in joined_source
    assert re.search(r'TEST_START_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
    assert re.search(r'TEST_END_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
