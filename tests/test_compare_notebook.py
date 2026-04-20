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


def test_compare_notebook_code_cells_compile():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "compare.ipynb"
    code_cells = _load_code_cells(notebook_path)
    joined_source = "\n".join(code_cells)

    assert len(code_cells) >= 6
    assert "from scripts.utils.local_mpc_notebook_helpers import (" in joined_source
    assert "replay_local_mpc_rollout_package" in joined_source
    assert "resolve_latest_compatible_local_mpc_rollout_package_dir" in joined_source
    assert "USE_CACHED_LOCAL_MPC_ROLLOUT = True" in joined_source
    assert "LOCAL_MPC_PERFECT_INPUT_DIR = None" in joined_source
    assert "LOCAL_MPC_LSTM_INPUT_DIR = None" in joined_source
    assert "local_rollout_prefix = PROJECT_ROOT / \"artifacts\" / \"local_mpc_cached_rollout\"" in joined_source
    assert "_resolve_local_mpc_cached_rollout_dir(" in joined_source
    assert "local_mpc_perfect_bundle = replay_local_mpc_rollout_package(" in joined_source
    assert "local_mpc_lstm_bundle = replay_local_mpc_rollout_package(" in joined_source
    assert "notebooks/madrl/local_MPC.ipynb" in joined_source
    assert "collect_local_mpc_rollout(" not in joined_source
    assert "from scripts.utils.admm_mpc_notebook_helpers import (" in joined_source
    assert "replay_admm_mpc_rollout_package" in joined_source
    assert "resolve_latest_compatible_admm_mpc_rollout_package_dir" in joined_source
    assert "USE_CACHED_ADMM_ROLLOUT = True" in joined_source
    assert "ADMM_ROLLOUT_INPUT_DIR = None" in joined_source
    assert "admm_cfg.runtime.shared_data_dir = None" in joined_source
    assert "admm_cfg.runtime.shared_data_signature = None" in joined_source
    assert "admm_cfg.runtime.forecast_ready = None" in joined_source
    assert "effective_steps =" in joined_source
    assert "admm_cfg.data.test_end_date" in joined_source
    assert "admm_rollout_prefix = PROJECT_ROOT / \"artifacts\" / \"admm_mpc_cached_rollout\"" in joined_source
    assert "resolve_latest_compatible_admm_mpc_rollout_package_dir(" in joined_source
    assert "admm_mpc_lstm_bundle = replay_admm_mpc_rollout_package(" in joined_source
    assert "loaded_from_cached_rollout" in joined_source
    assert "rollout_package_dir" in joined_source
    assert "notebooks/madrl/ADMM_mpc.ipynb" in joined_source
    assert "collect_admm_mpc_rollout(" not in joined_source
    assert """rollouts = [
    global_oracle,
    local_mpc_perfect,
    local_mpc_lstm,
    admm_mpc_lstm,
    madrl_base,
    madrl_safe,
    madrl_projection,
]""" in joined_source
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")
