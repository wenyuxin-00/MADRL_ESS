from __future__ import annotations

import json
from pathlib import Path


def _load_notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
    )


def test_python_files_are_utf8_without_bom() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders = [path for path in repo_root.rglob("*.py") if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert offenders == []


def test_key_notebooks_are_utf8_without_bom() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb",
        repo_root / "notebooks" / "madrl" / "train_base.ipynb",
        repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb",
    ]
    offenders = [path for path in notebook_paths if path.read_bytes().startswith(b"\xef\xbb\xbf")]
    assert offenders == []


def test_key_notebooks_do_not_contain_placeholder_text() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb",
        repo_root / "notebooks" / "madrl" / "train_base.ipynb",
        repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb",
    ]
    for path in notebook_paths:
        text = path.read_text(encoding="utf-8")
        assert "???" not in text
        assert "?? notebook" not in text


def test_notebook_defaults_stay_portable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forecast_text = _load_notebook_text(repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb")
    madrl_text = _load_notebook_text(repo_root / "notebooks" / "madrl" / "train_base.ipynb")
    grid_text = _load_notebook_text(repo_root / "notebooks" / "madrl" / "grid_network_analysis.ipynb")

    required_forecast_tokens = [
        "force_retrain_forecast = False",
        "ensure_lstm_artifacts",
        "ensure_madrl_shared_data",
        "predictions.parquet",
        "metrics.parquet",
        "shared_data_record.json",
    ]
    forbidden_forecast_tokens = [
        'cfg.data.agent_profiles = ["SFH12", "SFH14", "SFH16", "SFH18", "SFH20"]',
        "reuse_saved_artifacts",
        "delete_stale_load_artifacts",
        "legacy_aliases",
    ]

    for token in required_forecast_tokens:
        assert token in forecast_text
    for token in forbidden_forecast_tokens:
        assert token not in forecast_text

    required_madrl_tokens = [
        "force_retrain_madrl = False",
        "load_madrl_training_result",
        "run_external_train_mainline",
        "save_rollout_record(",
    ]
    forbidden_madrl_tokens = [
        "REUSE_MODEL_ROOT",
        "CHECKPOINT_ROOT =",
        'test_start_date = "',
        'test_end_date = "',
    ]

    for token in required_madrl_tokens:
        assert token in madrl_text
    for token in forbidden_madrl_tokens:
        assert token not in madrl_text

    assert "load_rollout_record(" in grid_text
    assert "plot_voltage_profile_comparison" in grid_text
    assert "ProsumerDataset(" not in grid_text


def test_forecast_record_contract_tokens_present() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forecast_text = _load_notebook_text(repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb")

    required_tokens = [
        "predictions.parquet",
        "metrics.parquet",
        "shared_data_record.json",
        "artifact_map",
        "shared_data_signature",
    ]
    for token in required_tokens:
        assert token in forecast_text
