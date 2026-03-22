from __future__ import annotations

import json
from pathlib import Path


def test_grid_notebook_uses_test_split_and_parallel_training_defaults():
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_root / "notebooks" / "madrl" / "train_madrl_grid.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "simbench_2016_test.csv" in notebook_text
    assert "cfg.train.num_envs = 8" in notebook_text
    assert "cfg.train.vec_env_type = 'subproc'" in notebook_text
    assert "cfg.train.batch_size = 4096" in notebook_text
    assert "cfg.train.updates_per_step = 1" in notebook_text
    assert "plot_reward_decomposition(" in notebook_text
    assert "plot_per_agent_reward_components(" in notebook_text
    assert "plot_operating_cost_comparison(" in notebook_text
    assert "base_net_load" in notebook_text
    assert "test episode index" in notebook_text
    assert "test day offset" in notebook_text
    assert "plot_last_k_episodes_price_action_soc" not in notebook_text
