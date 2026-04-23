from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


@pytest.mark.parametrize(
    "notebook_name",
    [
        "train_base.ipynb",
        "train_base_safe.ipynb",
        "train_projection_safe.ipynb",
    ],
)
def test_madrl_training_notebook_code_cells_compile(notebook_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / notebook_name
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 4
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


@pytest.mark.parametrize(
    "notebook_name",
    [
        "train_base.ipynb",
        "train_base_safe.ipynb",
        "train_projection_safe.ipynb",
    ],
)
def test_madrl_training_notebooks_keep_single_parameter_and_record_flow(notebook_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / notebook_name
    notebook_source = "\n".join(_load_code_cells(notebook_path))

    required_tokens = [
        "force_retrain_madrl = False",
        "spec = MADRL_NOTEBOOK_SPECS[NOTEBOOK_KEY]",
        "bootstrap_madrl_notebook_shared_data(",
        "resolve_madrl_notebook_training(",
        "shared_data_record.json",
        "save_rollout_record(",
        "plot_power_balance_comparison",
        "plot_price_prediction_comparison",
        "plot_battery_power_and_soc_comparison",
        "plot_voltage_profile_comparison",
        "plot_net_load_comparison",
        "_plot_specs = [",
        "display(figure)",
        "'shared_data_signature': shared_data_signature",
        "'episodes_completed': int(train_result.get('episodes_completed', 0))",
    ]
    forbidden_tokens = [
        "test_start_date =",
        "test_end_date =",
        "REUSE_MODEL_ROOT",
        "CHECKPOINT_ROOT =",
        "force_retrain_forecast",
    ]

    for token in required_tokens:
        assert token in notebook_source
    for token in forbidden_tokens:
        assert token not in notebook_source
