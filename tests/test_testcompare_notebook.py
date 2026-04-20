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


def test_testcompare_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "TestCompare.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 5
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_testcompare_notebook_contains_compare_calls():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "TestCompare.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    assert "TEST_START_DATE = \"2020-06-01\"" in joined_source
    assert "TEST_END_DATE = \"2020-06-02\"" in joined_source
    assert "PREDICTION_MODE = \"normal\"" in joined_source
    assert (
        "SELECTED_SCHEMES = [\"admm_mpc\", \"drl_no_safety\", \"drl_safety_penalty\", \"drl_safety_projection\"]"
        in joined_source
    )
    assert "DRL_MODEL_ROOT_OVERRIDES = {}" in joined_source
    assert "REFERENCE_SCHEME = \"drl_safety_projection\"" in joined_source
    assert "SCHEME_SPECS = {" in joined_source
    assert "\"local_mpc\"" in joined_source
    assert "\"admm_mpc\"" in joined_source
    assert "\"drl_no_safety\"" in joined_source
    assert "\"drl_safety_penalty\"" in joined_source
    assert "\"drl_safety_projection\"" in joined_source
    assert "DRL_PRESET =" not in joined_source
    assert "DRL_MODEL_ROOT = None" not in joined_source
    assert "REFERENCE_DRL_PRESET =" not in joined_source
    assert "DRL_PRESETS = {" not in joined_source
    assert "load_training_run_bundle(" in joined_source
    assert "resolve_madrl_model_root(" in joined_source
    assert "collect_madrl_rollout(" in joined_source
    assert "ensure_madrl_shared_data(" in joined_source
    assert "selected_drl_schemes" in joined_source
    assert "len(selected_drl_schemes) >= 2" in joined_source
    assert "grid_nb.validate_compare_model_bundles(" in joined_source
    assert "expected_count=len(selected_drl_schemes)" in joined_source
    assert "experiment_controls" in joined_source
    assert "data_controls" in joined_source
    assert "battery_controls" in joined_source
    assert "train_controls" in joined_source
    assert "reference_experiment.get(" in joined_source
    assert "reference_data.get(" in joined_source
    assert "compose_experiment_config(" in joined_source
    assert "grid_nb.apply_notebook_experiment_settings(" in joined_source
    assert "base_cfg.env.episode_limit = int(round(24.0 / base_cfg.env.dt))" in joined_source
    assert "W_SOC_PEN =" not in joined_source
    assert "base_cfg.reward.w_soc_pen =" not in joined_source
    assert "base_cfg.reward.w_voltage_pen =" in joined_source
    assert "base_cfg.reward.w_line_pen =" in joined_source
    assert "base_cfg.reward.w_trafo_pen =" in joined_source
    assert "collect_admm_mpc_rollout(" in joined_source
    assert "collect_local_mpc_rollout(" in joined_source
    assert "admm_cfg.runtime.shared_data_dir = None" in joined_source
    assert "admm_cfg.runtime.shared_data_signature = None" in joined_source
    assert "admm_cfg.runtime.forecast_ready = None" in joined_source
    assert "effective_steps" in joined_source
    assert "admm_cfg.data.test_end_date" in joined_source
    assert "compare_rollout_metrics(" in joined_source
    assert "build_compare_economic_table(" in joined_source
    assert "build_compare_safety_table(" in joined_source
    assert "build_compare_warning_banner(" in joined_source
    assert "voltage_safety_table" in joined_source
    assert "transformer_safety_table" in joined_source
    assert "plot_shared_forecast_vs_actual(" in joined_source
    assert "plot_power_balance_comparison(" in joined_source
    assert "plot_voltage_profile_comparison(" in joined_source
    assert "plot_net_load_comparison(" in joined_source
    assert "plot_battery_power_and_soc_comparison(" in joined_source
    assert re.search(r'TEST_START_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
    assert re.search(r'TEST_END_DATE\s*=\s*"\d{4}-\d{2}-\d{2}"', joined_source)
