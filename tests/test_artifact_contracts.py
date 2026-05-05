from __future__ import annotations

import json
from pathlib import Path


RUN_ID = "20260504_234339_01e73bd1"
PRUNED_RUN_ID = "20260504_234328_01e73bd1"


def test_only_complete_mainline_run_is_kept() -> None:
    runs = Path("artifacts/runs")
    assert (runs / RUN_ID).is_dir()
    assert not (runs / PRUNED_RUN_ID).exists()


def test_complete_run_has_required_records_and_tables() -> None:
    run = Path("artifacts/runs") / RUN_ID
    required = [
        "config.json",
        "share_data/manifest.json",
        "forecast/artifacts",
        "results/perfect/MISOCP/record/manifest.json",
        "results/perfect/LOCAL_MPC/record/manifest.json",
        "results/lstm/LOCAL_MPC/record/manifest.json",
        "results/lstm/ADMM_MPC/record/manifest.json",
        "results/lstm/MADRL_BASE/record/manifest.json",
        "results/lstm/MADRL_PENALTY/record/manifest.json",
        "results/lstm/MADRL_PROJECTION/record/manifest.json",
        "models/madrl/madrl_base/meta.json",
        "models/madrl/madrl_base_safe/meta.json",
        "models/madrl/madrl_projection_safe/meta.json",
        "tables/compare_metrics.csv",
        "tables/economic_table.csv",
        "tables/safety_table.csv",
        "tables/compare.xlsx",
    ]
    missing = [item for item in required if not (run / item).exists()]
    assert missing == []


def test_complete_run_madrl_models_are_500_episode_records() -> None:
    run = Path("artifacts/runs") / RUN_ID
    for scheme in ("madrl_base", "madrl_base_safe", "madrl_projection_safe"):
        meta = json.loads((run / "models" / "madrl" / scheme / "meta.json").read_text(encoding="utf-8"))
        assert int(meta["episodes"]) >= 500
