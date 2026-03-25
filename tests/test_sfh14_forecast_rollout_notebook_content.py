from __future__ import annotations

import json
from pathlib import Path


def test_sfh14_forecast_notebook_displays_rollout_tables() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_root / "notebooks" / "forecast" / "sfh14_forecast.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert '"horizon_metrics"' in notebook_text
    assert '"sfh14_month_horizon_metrics"' in notebook_text
    assert '"rollout_experiment_summary"' in notebook_text
    assert 'load_analysis_report["recommendation"]' in notebook_text
