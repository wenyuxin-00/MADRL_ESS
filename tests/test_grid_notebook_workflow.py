import numpy as np
import pytest

from scripts.utils.grid_notebook_workflow import (
    NORMAL_PREDICTION_MODE,
    PERFECT_PREDICTION_MODE,
    apply_notebook_experiment_settings,
    normalize_date_input,
    resolve_forecast_backend,
)
from tests.support.helpers import make_case_dir, make_smoke_config


def test_resolve_forecast_backend_handles_mainline_modes():
    assert resolve_forecast_backend(PERFECT_PREDICTION_MODE, 24) == "perfect"
    assert resolve_forecast_backend(NORMAL_PREDICTION_MODE, 24) == "lstm"
    with pytest.raises(ValueError, match="future_horizon > 0"):
        resolve_forecast_backend(NORMAL_PREDICTION_MODE, 0)


def test_apply_notebook_experiment_settings_updates_cfg_for_user_controls(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    summary = apply_notebook_experiment_settings(
        cfg,
        prediction_mode="normal",
        test_start_date=20190101,
        test_end_date=20190130,
        agent_profiles=["SFH12", "SFH14"],
        load_scale=[1.2, 0.8],
        pv_scale=0.5,
        storage_scale=[1.0, 1.5],
        future_horizon=24,
        train_year=2019,
        test_year=2019,
    )

    assert cfg.obs.local_features == ["time", "soc"]
    assert cfg.obs.sequence_features == ["price", "load", "pv"]
    assert cfg.env.future_horizon == 24
    assert cfg.forecast.type == "lstm"
    assert cfg.data.test_start_date == "2019-01-01"
    assert cfg.data.test_end_date == "2019-01-30"
    assert np.allclose(cfg.data.load_scale, [1.2, 0.8])
    assert np.allclose(cfg.data.pv_scale, [0.5, 0.5])
    assert np.allclose(cfg.data.storage_scale, [1.0, 1.5])
    assert summary["prediction_mode"] == "normal"
    assert summary["forecast_backend"] == "lstm"
    assert summary["agent_profiles"] == ["SFH12", "SFH14"]


def test_normalize_date_input_accepts_compact_dates():
    assert normalize_date_input(20190101) == "2019-01-01"
    assert normalize_date_input("2019-01-30") == "2019-01-30"
    assert normalize_date_input(None) is None
