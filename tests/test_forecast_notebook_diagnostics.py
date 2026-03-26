from __future__ import annotations

import json
from pathlib import Path


def test_forecast_notebook_exposes_sfh14_repair_report() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'analysis_output_dir = project_root / "artifacts" / "forecast" / "analysis" / "sfh14_load"' not in notebook_text
    assert 'load_result=results.get("load")' not in notebook_text
    assert 'load_overrides=signal_training_overrides["load"]' not in notebook_text
    assert 'display(load_analysis_report[table_name])' not in notebook_text
