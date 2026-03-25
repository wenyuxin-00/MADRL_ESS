from __future__ import annotations

import json
from pathlib import Path


def test_forecast_notebook_declares_load_hybrid_configuration() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'load_model_mode = "per_agent"' in notebook_text
    assert 'load_time_feature_mode = "hour_week_year"' in notebook_text
    assert 'load_hybrid_mode = "baseline_blend"' in notebook_text
    assert 'load_baseline_mode = "last_value"' in notebook_text
    assert "cfg.forecast.load_model_mode = load_model_mode" in notebook_text
    assert "cfg.forecast.load_time_feature_mode = load_time_feature_mode" in notebook_text
    assert "cfg.forecast.load_hybrid_mode = load_hybrid_mode" in notebook_text
    assert "cfg.forecast.load_baseline_mode = load_baseline_mode" in notebook_text
    assert "if result.get(\"agent_results\"):" in notebook_text
    assert "blend_weight={agent_result['hybrid']['blend_weight']}" in notebook_text
    assert "collect_available_lstm_artifacts(cfg, overrides_by_signal=signal_training_overrides)" in notebook_text
    assert "LSTMForecaster.from_signal_artifacts(" in notebook_text
    assert "history_timestamps=history_timestamps" in notebook_text
