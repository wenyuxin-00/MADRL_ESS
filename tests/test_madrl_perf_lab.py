from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from configs import compose_experiment_config
from scripts.run_train_mainline import _apply_model_controls, _apply_runtime_controls
from scripts.utils.madrl_perf_lab import (
    build_candidate_experiment_name,
    build_perf_leaderboard,
    merge_control_overrides,
    recommend_perf_candidate,
    summarize_rollout_metrics,
)


def test_merge_control_overrides_preserves_base_and_merges_nested_dicts() -> None:
    base = {
        "experiment_name": "grid_mainline_perf_lab",
        "runtime_controls": {"enable_compile": True, "amp_dtype": "bfloat16"},
    }
    merged = merge_control_overrides(base, {"runtime_controls": {"enable_compile": False}})

    assert base["runtime_controls"]["enable_compile"] is True
    assert merged["runtime_controls"]["enable_compile"] is False
    assert merged["runtime_controls"]["amp_dtype"] == "bfloat16"



def test_build_candidate_experiment_name_appends_slugified_candidate_token() -> None:
    assert build_candidate_experiment_name("grid_mainline_perf_lab", "batch 8192") == "grid_mainline_perf_lab_batch_8192"



def test_summarize_rollout_metrics_extracts_error_cost_and_voltage_fields() -> None:
    rollout = SimpleNamespace(
        step_df=pd.DataFrame(
            {
                "episode_idx": [0, 0],
                "step": [0, 1],
                "price": [1.0, 3.0],
                "price_pred": [1.5, 2.0],
            }
        ),
        agent_df=pd.DataFrame(
            {
                "episode_idx": [0, 0],
                "step": [0, 1],
                "agent_profile": ["SFH12", "SFH12"],
                "load": [2.0, 4.0],
                "load_pred": [1.0, 5.0],
                "pv": [0.0, 2.0],
                "pv_pred": [0.5, 1.5],
                "operating_cost": [10.0, 12.0],
            }
        ),
        grid_df=pd.DataFrame(
            {
                "episode_idx": [0, 0, 0],
                "step": [0, 0, 1],
                "bus_id": [1, 2, 1],
                "vm_pu": [0.97, 1.08, 0.93],
            }
        ),
        summary=pd.DataFrame(
            {
                "controller": ["DRL (forecast_eval)"],
                "agent_profile": ["SFH12"],
                "operating_cost": [22.0],
            }
        ),
        meta={"v_min_pu": 0.95, "v_max_pu": 1.05},
    )

    metrics = summarize_rollout_metrics(rollout)

    assert metrics["total_operating_cost"] == pytest.approx(22.0)
    assert metrics["price_mae"] == pytest.approx(0.75)
    assert metrics["load_mae"] == pytest.approx(1.0)
    assert metrics["pv_mae"] == pytest.approx(0.5)
    assert metrics["voltage_violation_bus_points"] == 2
    assert metrics["voltage_violation_steps"] == 2
    assert metrics["operating_cost_sfh12"] == pytest.approx(22.0)



def test_build_perf_leaderboard_and_recommendation_prefer_cost_then_time() -> None:
    leaderboard = build_perf_leaderboard(
        [
            {
                "candidate_name": "baseline",
                "status": "ok",
                "total_operating_cost": 120.0,
                "total_wall_time_s": 100.0,
                "steps_per_sec": 80.0,
                "is_baseline": True,
            },
            {
                "candidate_name": "batch_8192",
                "status": "ok",
                "total_operating_cost": 110.0,
                "total_wall_time_s": 130.0,
                "steps_per_sec": 90.0,
                "is_baseline": False,
            },
            {
                "candidate_name": "env_24",
                "status": "ok",
                "total_operating_cost": 110.0,
                "total_wall_time_s": 95.0,
                "steps_per_sec": 92.0,
                "is_baseline": False,
            },
        ]
    )

    recommendation = recommend_perf_candidate(leaderboard)

    assert recommendation["recommended_candidate"] == "env_24"
    assert recommendation["improves_over_baseline"] is True
    assert leaderboard.loc[leaderboard["candidate_name"] == "env_24", "delta_operating_cost_vs_baseline"].item() == pytest.approx(-10.0)



def test_apply_model_and_runtime_controls_accept_optional_perf_lab_overrides(tmp_path) -> None:
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    _apply_model_controls(cfg, {"hidden_dim": 384})
    _apply_runtime_controls(
        cfg,
        {
            "pin_memory": True,
            "non_blocking_transfers": True,
            "enable_amp": False,
            "enable_compile": False,
            "amp_dtype": "float16",
            "compile_mode": "default",
        },
    )

    assert cfg.model.hidden_dim == 384
    assert cfg.runtime.pin_memory is True
    assert cfg.runtime.non_blocking_transfers is True
    assert cfg.runtime.enable_amp is False
    assert cfg.runtime.enable_compile is False
    assert cfg.runtime.amp_dtype == "float16"
    assert cfg.runtime.compile_mode == "default"



def test_apply_model_controls_rejects_non_positive_hidden_dim(tmp_path) -> None:
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    with pytest.raises(ValueError, match="hidden_dim"):
        _apply_model_controls(cfg, {"hidden_dim": 0})
