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


def _first_token_location(code_cells: list[str], token: str) -> tuple[int, int]:
    for cell_index, source in enumerate(code_cells):
        token_index = source.find(token)
        if token_index >= 0:
            return cell_index, token_index
    raise AssertionError(f"Token not found: {token}")


def test_admm_mpc_notebook_code_cells_compile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "ADMM_mpc.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 2
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_admm_mpc_notebook_uses_canonical_record_flow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "ADMM_mpc.ipynb"
    code_cells = _load_code_cells(notebook_path)
    joined_source = "\n".join(code_cells)

    required_tokens = [
        "ADMM_NOTEBOOK_SPEC",
        "get_mainline_forecast_controls(auto_train_missing=False)",
        "cfg.forecast.auto_train_missing = False",
        "bootstrap_madrl_notebook_shared_data(",
        "collect_admm_mpc_rollout(",
        "save_rollout_record(",
        "scheme_name=str(ADMM_NOTEBOOK_SPEC['scheme_name'])",
        "plot_power_balance_comparison",
        "plot_price_prediction_comparison",
        "plot_battery_power_and_soc_comparison",
        "plot_voltage_profile_comparison",
        "plot_net_load_comparison",
    ]
    forbidden_tokens = [
        "TEST_START_DATE",
        "TEST_END_DATE",
        "PREDICTION_MODE =",
        "SHOW_PROGRESS =",
        "SAVE_ADMM_MPC_ROLLOUT",
        "ADMM_MPC_ROLLOUT_TAG_OVERRIDE",
        "shared_data_record.json",
        "terminal_cost_multiplier",
    ]
    ordered_tokens = [
        "forecast_controls = get_mainline_forecast_controls(auto_train_missing=False)",
        "cfg.forecast.auto_train_missing = False",
        "bootstrap_madrl_notebook_shared_data(",
        "runtime_state = configure_torch_runtime(cfg, seed=cfg.runtime.seed)",
        "rollout = collect_admm_mpc_rollout(",
        "save_rollout_record(rollout",
        "metrics_df = compare_rollout_metrics(rollout)",
        "for title, plot_fn in _plot_specs:",
    ]

    for token in required_tokens:
        assert token in joined_source
    for token in forbidden_tokens:
        assert token not in joined_source
    ordered_locations = [_first_token_location(code_cells, token) for token in ordered_tokens]
    assert ordered_locations == sorted(ordered_locations)
