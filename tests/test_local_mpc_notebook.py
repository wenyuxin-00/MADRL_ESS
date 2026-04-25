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


def test_local_mpc_notebook_code_cells_compile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "local_MPC.ipynb"
    code_cells = _load_code_cells(notebook_path)

    assert len(code_cells) >= 2
    for cell_index, source in enumerate(code_cells, start=1):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")


def test_local_mpc_notebook_uses_canonical_record_flow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "madrl" / "local_MPC.ipynb"
    code_cells = _load_code_cells(notebook_path)
    joined_source = "\n".join(code_cells)

    required_tokens = [
        "get_mainline_forecast_controls(auto_train_missing=False)",
        "cfg.forecast.auto_train_missing = False",
        "bootstrap_madrl_notebook_shared_data(",
        "resolve_comparison_episode_window(cfg)",
        "resolve_episode_indices_for_start_timestamps(perfect_cfg, episode_start_timestamps)",
        "resolve_episode_indices_for_start_timestamps(cfg, episode_start_timestamps)",
        "collect_local_mpc_rollout",
        "episode_indices=perfect_episode_indices",
        "episode_indices=lstm_episode_indices",
        "comparison_window_contract",
        "plot_power_balance_comparison",
        "plot_price_prediction_comparison",
        "plot_battery_power_and_soc_comparison",
        "plot_voltage_profile_comparison",
        "plot_net_load_comparison",
        "plot_fn(local_mpc_perfect, local_mpc_lstm)",
        "save_rollout_record(",
        "scheme_name='local_mpc_perfect'",
        "scheme_name='local_mpc_lstm'",
        "compare_rollout_metrics(local_mpc_perfect, local_mpc_lstm)",
    ]
    forbidden_tokens = [
        "TEST_START_DATE",
        "TEST_END_DATE",
        "BASE_PREDICTION_MODE",
        "SAVE_LOCAL_MPC_ROLLOUT",
        "LOCAL_MPC_PERFECT_TAG_OVERRIDE",
        "LOCAL_MPC_LSTM_TAG_OVERRIDE",
        "shared_data_record.json",
        "MPC_COMPARE_EPISODE_INDICES",
        "range(89, 103)",
    ]
    ordered_tokens = [
        "forecast_controls = get_mainline_forecast_controls(auto_train_missing=False)",
        "cfg.forecast.auto_train_missing = False",
        "bootstrap_madrl_notebook_shared_data(",
        "runtime_state = configure_torch_runtime(cfg, seed=cfg.runtime.seed)",
        "comparison_window = resolve_comparison_episode_window(cfg)",
        "perfect_episode_indices = resolve_episode_indices_for_start_timestamps(perfect_cfg, episode_start_timestamps)",
        "local_mpc_perfect = collect_local_mpc_rollout(",
        "local_mpc_lstm = collect_local_mpc_rollout(",
        "save_rollout_record(local_mpc_perfect",
        "save_rollout_record(local_mpc_lstm",
        "metrics_df = compare_rollout_metrics(local_mpc_perfect, local_mpc_lstm)",
        "for title, plot_fn in _plot_specs:",
    ]

    for token in required_tokens:
        assert token in joined_source
    for token in forbidden_tokens:
        assert token not in joined_source
    ordered_locations = [_first_token_location(code_cells, token) for token in ordered_tokens]
    assert ordered_locations == sorted(ordered_locations)
