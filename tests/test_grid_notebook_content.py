from __future__ import annotations

import json
from pathlib import Path


def test_grid_notebook_exposes_mainline_controls_and_helper_workflow():
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_root / "notebooks" / "madrl" / "train_madrl_grid.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "compose_experiment_config(" in notebook_text
    assert "apply_notebook_experiment_settings(" in notebook_text
    assert "ensure_forecast_ready(" in notebook_text
    assert "collect_madrl_rollout(" in notebook_text
    assert "collect_mpc_rollout(" in notebook_text
    assert "plot_test_rollout(" in notebook_text
    assert "plot_operating_cost_comparison(" in notebook_text
    assert "experiment_controls = {" in notebook_text
    assert "data_controls = {" in notebook_text
    assert "train_controls = {" in notebook_text
    assert '"prediction_mode": "perfect"' in notebook_text
    assert '"agent_profiles": ["SFH12", "SFH14", "SFH16"]' in notebook_text
    assert '"future_horizon": 24' in notebook_text
    assert '"num_envs": 16' in notebook_text
    assert '"vec_env_type": "subproc"' in notebook_text
    assert '"batch_size": 4096' in notebook_text
    assert '"updates_per_step": 8' in notebook_text
    assert '"require_cuda": False' in notebook_text
    assert "runner = build_runner(" in notebook_text
    assert "episodes_completed = runner.run()" in notebook_text
    assert "plot_reward_decomposition(" in notebook_text
    assert "runner.save_model(" in notebook_text

    assert "apply_grid_profile(cfg" not in notebook_text
    assert "dataset_type" not in notebook_text
    assert "cfg.env.env_type" not in notebook_text
    assert "grid_profile" not in notebook_text
    assert "obs_builder_type" not in notebook_text
    assert "wpuq_" not in notebook_text
    assert "csv_prosumer" not in notebook_text
    assert "grid_pf" not in notebook_text

