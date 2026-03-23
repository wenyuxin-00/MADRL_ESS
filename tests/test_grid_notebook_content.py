from __future__ import annotations

import json
from pathlib import Path


def test_grid_notebook_uses_test_split_and_parallel_training_defaults():
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
    assert "apply_grid_profile(cfg" not in notebook_text
    assert "run_perf_probe(" not in notebook_text
    assert "estimate_static_grid_sensitivity" not in notebook_text
    assert "ZeroController" not in notebook_text
    assert "evaluate_controller(" not in notebook_text
    assert "import gymnasium as gym" in notebook_text
    assert "sys.executable" in notebook_text
    assert "cfg.train.show_progress = True" in notebook_text
    assert "cfg.train.progress_postfix_interval = 10" in notebook_text
    assert "cfg.train.num_envs = 32" in notebook_text
    assert "cfg.train.vec_env_type = 'subproc'" in notebook_text
    assert "cfg.train.batch_size = 4096" in notebook_text
    assert "cfg.train.updates_per_step = 8" in notebook_text
    assert "require_cuda = False" in notebook_text
    assert "runner = build_runner(" in notebook_text
    assert "episodes_completed = runner.run()" in notebook_text
    assert "plot_reward_decomposition(" in notebook_text
    assert "runner.save_model(" in notebook_text
    assert "plot_per_agent_reward_components(" not in notebook_text
    assert "plot_operating_cost_comparison(" not in notebook_text
    assert "base_net_load" not in notebook_text
    assert "test episode index" not in notebook_text
    assert "test day offset" not in notebook_text
    assert "plot_last_k_episodes_price_action_soc" not in notebook_text
    assert "w_l_pen" not in notebook_text
    assert "getattr(cfg.grid, 'w_line_pen'" not in notebook_text
