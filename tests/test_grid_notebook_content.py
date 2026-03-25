from __future__ import annotations

import json
from pathlib import Path


def test_grid_notebook_exposes_external_mainline_and_reloadable_test_paths():
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
    assert "plot_test_voltage_profile(" in notebook_text
    assert "plot_operating_cost_comparison(" in notebook_text
    assert "recommended_gpu_fast_num_envs()" in notebook_text
    assert "run_external_train_mainline(" in notebook_text
    assert '"launch_mode": "external"' in notebook_text
    assert '"profile": "gpu_fast"' in notebook_text
    assert 'battery_controls = {' in notebook_text
    assert '"mode": "from_pv"' in notebook_text
    assert '"from_pv_power_ratio": 0.5' in notebook_text
    assert '"algorithm": "MATD3"' in notebook_text
    assert '"vec_env_type": "subproc"' in notebook_text
    assert '"batch_size": 4096' in notebook_text
    assert '"buffer_size": 200000' in notebook_text
    assert '"update_interval": 1' in notebook_text
    assert '"updates_per_step": 2' in notebook_text
    assert '"policy_update_freq": 2' in notebook_text
    assert 'checkpoint_controls = {' in notebook_text
    assert 'test_controls = {' in notebook_text
    assert "prepare_madrl_run_paths(" in notebook_text
    assert "resolve_madrl_model_root(" in notebook_text
    assert 'reward_summary_path = Path(train_result["reward_summary_path"])' in notebook_text
    assert "plot_reward_decomposition(" in notebook_text
    assert "reward_summary=reward_summary_path" in notebook_text
    assert "cfg.algo.name = algorithm" in notebook_text

    assert "write_notebook_run_metadata(" not in notebook_text
    assert 'if launch_mode == "external":' not in notebook_text
    assert 'if launch_mode == "inline":' not in notebook_text
    assert "runner = build_runner(" not in notebook_text
    assert "runner.save_model(" not in notebook_text

    assert "apply_grid_profile(cfg" not in notebook_text
    assert "dataset_type" not in notebook_text
    assert "cfg.env.env_type" not in notebook_text
    assert "grid_profile" not in notebook_text
    assert "obs_builder_type" not in notebook_text
    assert '"storage_scale"' not in notebook_text
    assert "wpuq_" not in notebook_text
    assert "csv_prosumer" not in notebook_text
    assert "grid_pf" not in notebook_text
