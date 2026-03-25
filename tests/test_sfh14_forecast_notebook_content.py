from __future__ import annotations

import json
from pathlib import Path


def test_sfh14_forecast_notebook_declares_isolated_repair_workflow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_root / "notebooks" / "forecast" / "sfh14_forecast.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'focus_profile = "SFH14"' in notebook_text
    assert 'control_profiles = ["SFH12", "SFH16"]' in notebook_text
    assert 'agent_profiles = ["SFH12", "SFH14", "SFH16"]' in notebook_text
    assert "retrain = True" in notebook_text
    assert 'artifact_root = project_root / "artifacts" / "forecast" / "experiments" / "sfh14" / "lstm"' in notebook_text
    assert (
        'analysis_output_dir = project_root / "artifacts" / "forecast" / "experiments" / "sfh14" / "analysis" / '
        '"sfh14_load"' in notebook_text
    )
    assert 'cfg.forecast.target_signals = ["load"]' in notebook_text
    assert 'cfg.obs.sequence_features = ["load"]' in notebook_text
    assert "cfg.data.agent_profiles = agent_profiles" in notebook_text
    assert "run_sfh14_load_repair_report(" in notebook_text
