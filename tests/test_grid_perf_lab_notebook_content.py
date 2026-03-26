from __future__ import annotations

import ast
import json
from pathlib import Path



def test_grid_perf_lab_notebook_declares_staged_sweep_and_rollout_evaluation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "train_madrl_grid_perf_lab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    code_cells = [
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for source in code_cells:
        ast.parse(source)

    notebook_text = "\n".join(code_cells)

    assert 'experiment_name_base = "grid_mainline_perf_lab"' in notebook_text
    assert 'train_episodes_short = 50' in notebook_text
    assert 'sweep_candidates = {' in notebook_text
    assert '"batch_size": [4096, 8192, 12288]' in notebook_text
    assert '"updates_per_step": [2, 3, 4]' in notebook_text
    assert '"num_envs": [16, 20, 24]' in notebook_text
    assert 'enable_compile_candidates = [False, True]' in notebook_text
    assert 'evaluate_after_train = True' in notebook_text
    assert 'run_external_train_mainline(' in notebook_text
    assert 'collect_madrl_rollout(' in notebook_text
    assert 'build_perf_leaderboard(' in notebook_text
    assert 'recommend_perf_candidate(' in notebook_text
    assert 'summarize_rollout_metrics(' in notebook_text
    assert 'plot_test_rollout(recommended_result["rollout"])' in notebook_text
    assert 'plot_test_voltage_profile(recommended_result["rollout"])' in notebook_text
    assert 'model_controls' in notebook_text
    assert 'runtime_controls' in notebook_text
    assert 'checkpoint_controls_local["experiment_name"] = build_candidate_experiment_name(' in notebook_text
