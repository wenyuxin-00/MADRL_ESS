import matplotlib
import json
import importlib.util
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from types import SimpleNamespace

matplotlib.use("Agg")

HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


def _has_working_gurobi_license() -> bool:
    if not HAS_GUROBI:
        return False
    try:
        import gurobipy as gp

        model = gp.Model("grid_nb_test")
        model.Params.OutputFlag = 0
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
        return True
    except Exception:
        return False


HAS_WORKING_GUROBI_LICENSE = _has_working_gurobi_license()

from configs.profiles import summarize_experiment
from configs.experiment_config import ExperimentConfig
from predictors.shared_data import PRICE_OBSERVATION_CONTRACT
from scripts.checkpoints import ACTOR_ACTION_MAPPING_CONTRACT, TRAINING_HEALTH_CONTRACT
from scripts.mainline_compare import (
    _get_local_mpc_solver,
    build_compare_economic_table,
    build_compare_safety_table,
    collect_global_full_horizon_rollout,
    collect_local_mpc_rollout,
    compare_rollout_metrics,
    validate_compare_model_bundles,
)

_CURRENT_COMPARE_TRAINING_CONTRACT = {
    "actor_action_mapping_contract": ACTOR_ACTION_MAPPING_CONTRACT,
    "training_health_contract": TRAINING_HEALTH_CONTRACT,
    "price_observation_contract": PRICE_OBSERVATION_CONTRACT,
    "discount_gamma": 0.999,
}
from scripts.utils.grid_notebook_workflow import (
    FORECAST_EVAL_MODE,
    NORMAL_PREDICTION_MODE,
    ORACLE_EVAL_MODE,
    PERFECT_PREDICTION_MODE,
    RolloutResult,
    assert_rollout_timestamp_alignment,
    apply_notebook_experiment_settings,
    build_comparison_cfg,
    bootstrap_madrl_notebook_shared_data,
    collect_controller_rollout,
    load_rollout_record,
    normalize_date_input,
    plot_battery_power_and_soc_comparison,
    plot_price_prediction_comparison,
    plot_net_load_comparison,
    plot_global_misocp_validation,
    plot_power_balance_comparison,
    plot_voltage_profile_comparison,
    resolve_madrl_notebook_training,
    resolve_evaluation_mode,
    resolve_comparison_episode_window,
    resolve_episode_indices_for_start_timestamps,
    resolve_forecast_backend,
    save_rollout_record,
)
from predictors.mainline_forecast import (
    get_mainline_forecast_controls,
)
from predictors.shared_data import ensure_madrl_shared_data
from tests.support.helpers import make_case_dir, make_smoke_config, write_prosumer_processed_dataset


def _load_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _expand_cfg_to_multiday(cfg, *, evaluation_days: int = 5) -> None:
    cfg.env.episode_limit = 96
    cfg.env.train_window_days = 1
    cfg.env.window_stride_days = 1
    cfg.env.future_horizon = 1
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    write_prosumer_processed_dataset(
        cfg.data.data_dir,
        agent_profiles=list(cfg.data.agent_profiles),
        train_year=2019,
        test_year=2020,
        train_steps=96 * evaluation_days,
        test_steps=96 * evaluation_days,
    )


def _enable_normal_comparison_contract(cfg) -> None:
    shared_data_dir = (Path(cfg.data.data_dir).resolve().parent / "_test_shared_data_contract").resolve()
    shared_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.forecast.type = "lstm"
    cfg.forecast.auto_train_missing = False
    cfg.runtime.shared_data_dir = str(shared_data_dir)
    cfg.runtime.shared_data_signature = "sig-normal-contract"


def test_resolve_forecast_backend_handles_mainline_modes():
    assert resolve_forecast_backend(PERFECT_PREDICTION_MODE, 24) == "perfect"
    assert resolve_forecast_backend(NORMAL_PREDICTION_MODE, 24) == "lstm"
    with pytest.raises(ValueError, match="future_horizon > 0"):
        resolve_forecast_backend(NORMAL_PREDICTION_MODE, 0)


def test_resolve_evaluation_mode_maps_prediction_modes():
    assert resolve_evaluation_mode(PERFECT_PREDICTION_MODE) == ORACLE_EVAL_MODE
    assert resolve_evaluation_mode(NORMAL_PREDICTION_MODE) == FORECAST_EVAL_MODE


@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_fragment"),
    [
        ("shared_data_dir", None, "cfg.runtime.shared_data_dir"),
        ("shared_data_signature", None, "cfg.runtime.shared_data_signature"),
        ("auto_train_missing", True, "cfg.forecast.auto_train_missing=False"),
    ],
)
def test_build_comparison_cfg_rejects_missing_normal_runtime_contract(tmp_path, field_name, field_value, expected_fragment):
    case_dir = make_case_dir(tmp_path, f"grid_nb_normal_contract_{field_name}")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    _enable_normal_comparison_contract(cfg)

    if field_name == "auto_train_missing":
        cfg.forecast.auto_train_missing = field_value
    else:
        setattr(cfg.runtime, field_name, field_value)

    with pytest.raises(ValueError) as exc_info:
        build_comparison_cfg(cfg, prediction_mode="normal")

    assert expected_fragment in str(exc_info.value)


def test_build_comparison_cfg_allows_perfect_mode_without_shared_data_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_nb_perfect_without_shared_data")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    comparison_cfg = build_comparison_cfg(cfg, prediction_mode="perfect")

    assert comparison_cfg.forecast.type == "perfect"
    assert comparison_cfg.runtime.shared_data_dir is None
    assert comparison_cfg.runtime.shared_data_signature is None


def test_build_comparison_cfg_preserves_normal_shared_data_contract_when_cfg_is_already_lstm(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_nb_normal_with_shared_data")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    _enable_normal_comparison_contract(cfg)

    comparison_cfg = build_comparison_cfg(cfg, prediction_mode="normal")

    assert comparison_cfg.forecast.type == "lstm"
    assert comparison_cfg.runtime.shared_data_dir == cfg.runtime.shared_data_dir
    assert comparison_cfg.runtime.shared_data_signature == cfg.runtime.shared_data_signature
    assert comparison_cfg.runtime.forecast_ready == cfg.runtime.forecast_ready


def test_build_comparison_cfg_clears_only_the_comparison_copy_for_perfect_mode(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_nb_perfect_clears_copy_only")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    _enable_normal_comparison_contract(cfg)

    comparison_cfg = build_comparison_cfg(cfg, prediction_mode="perfect")

    assert comparison_cfg.forecast.type == "perfect"
    assert comparison_cfg.runtime.shared_data_dir is None
    assert comparison_cfg.runtime.shared_data_signature is None
    assert comparison_cfg.runtime.forecast_ready is None
    assert cfg.runtime.shared_data_dir is not None
    assert cfg.runtime.shared_data_signature == "sig-normal-contract"


def test_resolve_comparison_episode_window_uses_shared_data_selected_timestamps(monkeypatch):
    cfg = SimpleNamespace(
        forecast=SimpleNamespace(type="lstm"),
        runtime=SimpleNamespace(
            shared_data_dir="C:/shared-data",
            shared_data_signature="sig",
            selected_episode_indices=None,
        ),
    )

    class _FakeEnv:
        num_available_episodes = 4

        def reset(self, *, episode_idx):
            timestamps = pd.date_range(
                f"2020-04-{int(episode_idx) + 1:02d} 01:00:00",
                periods=3,
                freq="15min",
                tz="Europe/Berlin",
            )
            return {}, {"episode_meta": {"timestamps": [timestamp.isoformat() for timestamp in timestamps]}}

        def close(self):
            pass

    def _fake_build_env(local_cfg, mode):
        assert mode == "test"
        local_cfg.runtime.selected_episode_indices = [1, 2]
        return _FakeEnv()

    monkeypatch.setattr("scripts.builder.build_env", _fake_build_env)

    window = resolve_comparison_episode_window(cfg)

    assert window["comparison_window_contract"] == "episode_start_timestamps_v1"
    assert window["source_episode_indices"] == [1, 2]
    assert window["episode_start_timestamps"] == [
        "2020-04-02T01:00:00+02:00",
        "2020-04-03T01:00:00+02:00",
    ]


def test_resolve_episode_indices_for_start_timestamps_maps_backend_by_exact_start(monkeypatch):
    cfg = SimpleNamespace(runtime=SimpleNamespace(selected_episode_indices=None))

    class _FakeEnv:
        num_available_episodes = 5

        def reset(self, *, episode_idx):
            start = pd.Timestamp("2020-03-30T01:00:00", tz="Europe/Berlin") + pd.Timedelta(days=int(episode_idx))
            timestamps = pd.date_range(
                start,
                periods=3,
                freq="15min",
            )
            return {}, {"episode_meta": {"timestamps": [timestamp.isoformat() for timestamp in timestamps]}}

        def close(self):
            pass

    monkeypatch.setattr("scripts.builder.build_env", lambda local_cfg, mode: _FakeEnv())

    indices = resolve_episode_indices_for_start_timestamps(
        cfg,
        ["2020-04-01T01:00:00+02:00", "2020-04-03T01:00:00+02:00"],
    )

    assert indices == [2, 4]


def test_assert_rollout_timestamp_alignment_rejects_shifted_records():
    first = RolloutResult(
        step_df=pd.DataFrame({"timestamp": ["2020-04-01T01:00:00+02:00", "2020-04-01T01:15:00+02:00"]}),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "reference"},
    )
    shifted = RolloutResult(
        step_df=pd.DataFrame({"timestamp": ["2020-03-30T01:00:00+02:00", "2020-03-30T01:15:00+02:00"]}),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "shifted"},
    )

    with pytest.raises(ValueError, match="Compare timestamp alignment failed"):
        assert_rollout_timestamp_alignment(first, shifted)


def test_apply_notebook_experiment_settings_updates_cfg_for_user_controls(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    summary = apply_notebook_experiment_settings(
        cfg,
        prediction_mode="normal",
        test_start_date=20190101,
        test_end_date=20190130,
        agent_profiles=["SFH12", "SFH14"],
        agent_bus_ids=[12, 4],
        load_scale=[1.2, 0.8],
        pv_scale=0.5,
        battery_controls={
            "battery_capacity": 6.0,
            "max_charge_rate": 0.5,
            "efficiency": 0.9,
            "init_soc": 0.4,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "soc_target": 0.6,
        },
        future_horizon=24,
        train_year=2019,
        test_year=2019,
    )

    assert cfg.obs.local_features == ["calendar_time", "soc"]
    assert cfg.obs.sequence_features == ["wholesale_price_relative", "wholesale_price_spread", "load", "pv"]
    assert cfg.env.future_horizon == 24
    assert cfg.forecast.type == "lstm"
    assert cfg.data.test_start_date == "2019-01-01"
    assert cfg.data.test_end_date == "2019-01-30"
    assert cfg.grid.agent_bus_ids == [12, 4]
    assert np.allclose(cfg.data.load_scale, [1.2, 0.8])
    assert np.allclose(cfg.data.pv_scale, [0.5, 0.5])
    assert cfg.env.battery_capacity == [6.0, 6.0]
    assert np.isclose(cfg.env.max_charge_rate, 0.5)
    assert summary["prediction_mode"] == "normal"
    assert summary["evaluation_mode"] == FORECAST_EVAL_MODE
    assert summary["forecast_backend"] == "lstm"
    assert summary["agent_profiles"] == ["SFH12", "SFH14"]
    assert summary["agent_bus_ids"] == [12, 4]
    assert summary["battery"]["mode"] == "fixed"
    assert summary["battery"]["battery_capacity"] == [6.0, 6.0]
    assert summary["battery"]["p_max_kw"] == [3.0, 3.0]


def test_apply_notebook_experiment_settings_supports_fixed_battery_vectors(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_fixed")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    summary = apply_notebook_experiment_settings(
        cfg,
        prediction_mode="perfect",
        test_start_date=20190101,
        test_end_date=20190130,
        agent_profiles=["SFH12", "SFH14"],
        agent_bus_ids=[10, 6],
        load_scale=1.0,
        pv_scale=1.0,
        battery_controls={
            "battery_capacity": [10.0, 12.0],
            "max_charge_rate": 0.5,
        },
        future_horizon=1,
        train_year=2019,
        test_year=2019,
    )

    assert cfg.env.battery_capacity == [10.0, 12.0]
    assert np.isclose(cfg.env.max_charge_rate, 0.5)
    assert summary["battery"]["battery_capacity"] == [10.0, 12.0]
    assert summary["battery"]["p_max_kw"] == [5.0, 6.0]


def test_apply_notebook_experiment_settings_applies_forecast_controls(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_forecast_controls")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    forecast_controls = get_mainline_forecast_controls(
        artifact_root=case_dir / "artifacts" / "forecast" / "lstm",
        auto_train_missing=False,
    )

    summary = apply_notebook_experiment_settings(
        cfg,
        prediction_mode="normal",
        test_start_date=20190101,
        test_end_date=20190130,
        agent_profiles=["SFH12", "SFH14"],
        agent_bus_ids=[10, 6],
        load_scale=1.0,
        pv_scale=1.0,
        battery_controls={
            "battery_capacity": [25.0, 25.0],
            "max_charge_rate": 0.4,
        },
        forecast_controls=forecast_controls,
        future_horizon=int(forecast_controls["future_horizon"]),
        train_year=2019,
        test_year=2019,
    )

    assert cfg.forecast.type == "lstm"
    assert cfg.forecast.lstm_artifact_root == str((case_dir / "artifacts" / "forecast" / "lstm").resolve())
    assert cfg.forecast.history_window == int(forecast_controls["history_window"])
    assert cfg.forecast.auto_train_missing is False
    assert cfg.forecast.load_component_split is True
    assert cfg.forecast.load_scaler_type == "robust"
    assert cfg.forecast.signal_training_overrides == forecast_controls["signal_training_overrides"]
    assert summary["forecast"]["artifact_root"] == cfg.forecast.lstm_artifact_root
    assert summary["forecast"]["auto_train_missing"] is False


def test_get_mainline_forecast_controls_matches_canonical_config_defaults():
    cfg = ExperimentConfig()

    controls = get_mainline_forecast_controls(auto_train_missing=False)

    assert controls["future_horizon"] == cfg.env.future_horizon
    assert controls["history_window"] == cfg.forecast.history_window
    assert controls["load_component_split"] is cfg.forecast.load_component_split
    assert controls["load_scaler_type"] == cfg.forecast.load_scaler_type
    assert controls["signal_training_overrides"] == cfg.forecast.signal_training_overrides


def test_apply_notebook_experiment_settings_backfills_partial_forecast_controls_from_canonical_defaults(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_forecast_partial")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    apply_notebook_experiment_settings(
        cfg,
        prediction_mode="normal",
        test_start_date=20190101,
        test_end_date=20190130,
        agent_profiles=["SFH12", "SFH14"],
        agent_bus_ids=[10, 6],
        load_scale=1.0,
        pv_scale=1.0,
        forecast_controls={
            "auto_train_missing": False,
            "signal_training_overrides": {
                    "wholesale_price": {
                    "epochs": 99,
                }
            },
        },
        future_horizon=24,
        train_year=2019,
        test_year=2019,
    )

    assert cfg.forecast.auto_train_missing is False
    assert cfg.forecast.load_component_split is True
    assert cfg.forecast.load_scaler_type == "robust"
    assert cfg.forecast.signal_training_overrides["wholesale_price"]["epochs"] == 99


def test_resolve_madrl_notebook_training_builds_external_launch_payloads(tmp_path, monkeypatch):
    import scripts.mainline_madrl as mainline_madrl

    case_dir = make_case_dir(tmp_path, "madrl_notebook_training_external")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    cfg.forecast.type = "lstm"
    (case_dir / "shared_data").mkdir(parents=True, exist_ok=True)
    cfg.runtime.shared_data_dir = str((case_dir / "shared_data").resolve())
    cfg.runtime.shared_data_signature = "sig-123"
    spec = {"algorithm": "MATD3", "experiment_name": "train_base", "env_name": "GridTrainBase"}
    captured: dict[str, object] = {}

    def _fake_run_external_train_mainline(**kwargs):
        captured.update(kwargs)
        return {
            "result": {
                "model_root": str(case_dir / "models" / "run_a"),
                "shared_data_signature": "sig-123",
                "episodes_completed": 12,
            },
            "launch_info": {"result_json_path": str(case_dir / "models" / "run_a" / "_meta" / "train_result.json")},
        }

    monkeypatch.setattr(mainline_madrl, "run_external_train_mainline", _fake_run_external_train_mainline)

    resolved = resolve_madrl_notebook_training(
        cfg,
        spec=spec,
        force_retrain_madrl=True,
        root=case_dir,
        notebook_path="notebooks/madrl/train_base.ipynb",
    )

    assert captured["env_name"] == "GridTrainBase"
    assert captured["project_root"] == case_dir.resolve()
    assert captured["train_controls"]["show_progress"] is True
    assert captured["train_controls"]["progress_episode_interval"] == int(cfg.train.progress_episode_interval)
    assert captured["train_controls"]["train_window_days"] == int(cfg.env.train_window_days)
    assert captured["train_controls"]["window_stride_days"] == int(cfg.env.window_stride_days)
    assert captured["train_controls"]["discount_gamma"] == pytest.approx(cfg.algo.gamma)
    assert captured["experiment_controls"]["runtime_controls"]["shared_data_dir"] == str((case_dir / "shared_data").resolve())
    assert captured["experiment_controls"]["runtime_controls"]["shared_data_signature"] == "sig-123"
    assert resolved["model_root"].endswith("run_a")
    assert resolved["result_json_path"].endswith("train_result.json")
    assert resolved["train_result"]["episodes_completed"] == 12


def test_resolve_madrl_notebook_training_loads_and_validates_saved_run(tmp_path, monkeypatch):
    import scripts.mainline_madrl as mainline_madrl

    case_dir = make_case_dir(tmp_path, "madrl_notebook_training_load")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    cfg.forecast.type = "lstm"
    (case_dir / "shared_data").mkdir(parents=True, exist_ok=True)
    cfg.runtime.shared_data_dir = str((case_dir / "shared_data").resolve())
    cfg.runtime.shared_data_signature = "sig-456"
    stored_test_start_date = "2020-04-01"
    stored_test_end_date = "2020-04-05"
    cfg.data.test_start_date = "2020-06-01"
    cfg.data.test_end_date = "2020-06-03"
    spec = {"algorithm": "MATD3", "experiment_name": "train_base", "env_name": "GridTrainBase"}

    def _fake_load_madrl_training_result(**kwargs):
        assert kwargs["experiment_name"] == "train_base"
        return {
            "model_root": str(case_dir / "models" / "run_b"),
            "result_json_path": str(case_dir / "models" / "run_b" / "_meta" / "train_result.json"),
            "result": {
                "prediction_mode": "normal",
                "shared_data_signature": "sig-456",
                "data_controls": {
                    "agent_profiles": list(cfg.data.agent_profiles),
                    "agent_bus_ids": list(cfg.grid.agent_bus_ids),
                    "load_scale": list(cfg.data.load_scale),
                    "pv_scale": list(cfg.data.pv_scale),
                    "future_horizon": int(cfg.env.future_horizon),
                    "test_start_date": stored_test_start_date,
                    "test_end_date": stored_test_end_date,
                },
                "experiment_controls": {
                    "reward_controls": {
                        "action_boundary_penalty_weight": float(cfg.reward.action_boundary_penalty_weight),
                        "soc_boundary_regularization_weight": float(cfg.reward.soc_boundary_regularization_weight),
                        "throughput_bonus_eur_per_kwh_max": float(cfg.reward.throughput_bonus_eur_per_kwh_max),
                        "soc_boundary_epsilon": float(cfg.reward.soc_boundary_epsilon),
                        "soc_boundary_margin": float(cfg.reward.soc_boundary_margin),
                        "w_voltage_pen": float(cfg.reward.w_voltage_pen),
                        "w_line_pen": float(cfg.reward.w_line_pen),
                        "w_trafo_pen": float(cfg.reward.w_trafo_pen),
                        "export_subsidy_eur_per_kwh": float(cfg.reward.export_subsidy_eur_per_kwh),
                        "import_price_markup_eur_per_kwh": float(cfg.reward.import_price_markup_eur_per_kwh),
                        "storage_objective_mode": str(cfg.reward.storage_objective_mode),
                        "storage_price_mode": str(cfg.reward.storage_price_mode),
                        "storage_profit_weight": float(cfg.reward.storage_profit_weight),
                    },
                },
                "train_controls": {
                    "train_window_days": int(cfg.env.train_window_days),
                    "window_stride_days": int(cfg.env.window_stride_days),
                    "learning_starts_transitions": cfg.train.learning_starts_transitions,
                    "actor_learning_starts_transitions": cfg.train.actor_learning_starts_transitions,
                    "n_step_return": int(cfg.train.n_step_return),
                    "feasible_random_exploration_start": float(cfg.train.feasible_random_exploration_start),
                    "feasible_random_exploration_end": float(cfg.train.feasible_random_exploration_end),
                    "feasible_random_exploration_decay_steps": int(cfg.train.feasible_random_exploration_decay_steps),
                    "discount_gamma": float(cfg.algo.gamma),
                },
            },
        }

    monkeypatch.setattr(mainline_madrl, "load_madrl_training_result", _fake_load_madrl_training_result)

    resolved = resolve_madrl_notebook_training(
        cfg,
        spec=spec,
        force_retrain_madrl=False,
        root=case_dir,
        notebook_path="notebooks/madrl/train_base.ipynb",
    )

    assert resolved["model_root"].endswith("run_b")
    assert resolved["result_json_path"].endswith("train_result.json")
    assert resolved["train_result"]["shared_data_signature"] == "sig-456"


def test_resolve_madrl_notebook_training_rejects_mismatched_saved_run(tmp_path, monkeypatch):
    import scripts.mainline_madrl as mainline_madrl

    case_dir = make_case_dir(tmp_path, "madrl_notebook_training_mismatch")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    cfg.forecast.type = "lstm"
    (case_dir / "shared_data").mkdir(parents=True, exist_ok=True)
    cfg.runtime.shared_data_dir = str((case_dir / "shared_data").resolve())
    cfg.runtime.shared_data_signature = "sig-789"
    spec = {"algorithm": "MATD3", "experiment_name": "train_base", "env_name": "GridTrainBase"}

    monkeypatch.setattr(
        mainline_madrl,
        "load_madrl_training_result",
        lambda **kwargs: {
            "model_root": str(case_dir / "models" / "run_c"),
            "result_json_path": str(case_dir / "models" / "run_c" / "_meta" / "train_result.json"),
            "result": {
                "prediction_mode": "normal",
                "data_controls": {
                    "agent_profiles": ["SFH12", "SFH99"],
                    "agent_bus_ids": list(cfg.grid.agent_bus_ids),
                    "load_scale": list(cfg.data.load_scale),
                    "pv_scale": list(cfg.data.pv_scale),
                    "future_horizon": int(cfg.env.future_horizon),
                    "test_start_date": str(cfg.data.test_start_date),
                    "test_end_date": str(cfg.data.test_end_date),
                },
            },
        },
    )

    with pytest.raises(ValueError, match="agent_profiles"):
        resolve_madrl_notebook_training(
            cfg,
            spec=spec,
            force_retrain_madrl=False,
            root=case_dir,
            notebook_path="notebooks/madrl/train_base.ipynb",
        )
    assert cfg.forecast.signal_training_overrides["wholesale_price"]["hidden_size"] == 128
    assert cfg.forecast.signal_training_overrides["load"]["hidden_size"] == 64


def test_resolve_madrl_notebook_training_rejects_mismatched_window_contract(tmp_path, monkeypatch):
    import scripts.mainline_madrl as mainline_madrl

    case_dir = make_case_dir(tmp_path, "madrl_notebook_training_window_mismatch")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    cfg.forecast.type = "lstm"
    cfg.env.train_window_days = 7
    (case_dir / "shared_data").mkdir(parents=True, exist_ok=True)
    cfg.runtime.shared_data_dir = str((case_dir / "shared_data").resolve())
    cfg.runtime.shared_data_signature = "sig-window"
    spec = {"algorithm": "MATD3", "experiment_name": "train_base", "env_name": "GridTrainBase"}

    monkeypatch.setattr(
        mainline_madrl,
        "load_madrl_training_result",
        lambda **kwargs: {
            "model_root": str(case_dir / "models" / "run_window"),
            "result_json_path": str(case_dir / "models" / "run_window" / "_meta" / "train_result.json"),
            "result": {
                "prediction_mode": "normal",
                "shared_data_signature": "sig-window",
                "data_controls": {
                    "agent_profiles": list(cfg.data.agent_profiles),
                    "agent_bus_ids": list(cfg.grid.agent_bus_ids),
                    "load_scale": list(cfg.data.load_scale),
                    "pv_scale": list(cfg.data.pv_scale),
                    "future_horizon": int(cfg.env.future_horizon),
                    "test_start_date": str(cfg.data.test_start_date),
                    "test_end_date": str(cfg.data.test_end_date),
                },
                "experiment_controls": {
                    "reward_controls": {
                        "action_boundary_penalty_weight": float(cfg.reward.action_boundary_penalty_weight),
                        "soc_boundary_regularization_weight": float(cfg.reward.soc_boundary_regularization_weight),
                        "throughput_bonus_eur_per_kwh_max": float(cfg.reward.throughput_bonus_eur_per_kwh_max),
                        "soc_boundary_epsilon": float(cfg.reward.soc_boundary_epsilon),
                        "soc_boundary_margin": float(cfg.reward.soc_boundary_margin),
                        "w_voltage_pen": float(cfg.reward.w_voltage_pen),
                        "w_line_pen": float(cfg.reward.w_line_pen),
                        "w_trafo_pen": float(cfg.reward.w_trafo_pen),
                        "export_subsidy_eur_per_kwh": float(cfg.reward.export_subsidy_eur_per_kwh),
                        "import_price_markup_eur_per_kwh": float(cfg.reward.import_price_markup_eur_per_kwh),
                        "storage_objective_mode": str(cfg.reward.storage_objective_mode),
                        "storage_price_mode": str(cfg.reward.storage_price_mode),
                        "storage_profit_weight": float(cfg.reward.storage_profit_weight),
                    },
                },
                "train_controls": {
                    "train_window_days": 1,
                    "window_stride_days": int(cfg.env.window_stride_days),
                    "learning_starts_transitions": cfg.train.learning_starts_transitions,
                    "actor_learning_starts_transitions": cfg.train.actor_learning_starts_transitions,
                    "n_step_return": int(cfg.train.n_step_return),
                    "feasible_random_exploration_start": float(cfg.train.feasible_random_exploration_start),
                    "feasible_random_exploration_end": float(cfg.train.feasible_random_exploration_end),
                    "feasible_random_exploration_decay_steps": int(cfg.train.feasible_random_exploration_decay_steps),
                    "discount_gamma": float(cfg.algo.gamma),
                },
            },
        },
    )

    with pytest.raises(ValueError, match="train_controls\\.train_window_days"):
        resolve_madrl_notebook_training(
            cfg,
            spec=spec,
            force_retrain_madrl=False,
            root=case_dir,
            notebook_path="notebooks/madrl/train_base.ipynb",
        )


def test_bootstrap_madrl_notebook_shared_data_reads_record_and_updates_cfg(tmp_path):
    case_dir = make_case_dir(tmp_path, "madrl_notebook_shared_data_bootstrap")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    shared_data_dir = (case_dir / "shared_data" / "pkg_a").resolve()
    shared_data_dir.mkdir(parents=True, exist_ok=True)
    record_path = case_dir / "notebooks" / "record" / "forecast" / "lstm" / "shared_data_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "shared_data_dir": str(shared_data_dir),
                "signature_hash": "sig-bootstrap",
            }
        ),
        encoding="utf-8",
    )

    resolved = bootstrap_madrl_notebook_shared_data(
        cfg,
        root=case_dir,
        notebook_path="notebooks/madrl/train_base.ipynb",
    )

    assert resolved["shared_data_record_path"] == str(record_path.resolve())
    assert resolved["shared_data_dir"] == str(shared_data_dir)
    assert resolved["shared_data_signature"] == "sig-bootstrap"
    assert cfg.runtime.shared_data_dir == str(shared_data_dir)
    assert cfg.runtime.shared_data_signature == "sig-bootstrap"


def test_bootstrap_madrl_notebook_shared_data_rejects_missing_record(tmp_path):
    case_dir = make_case_dir(tmp_path, "madrl_notebook_shared_data_missing_record")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")

    with pytest.raises(FileNotFoundError, match="shared_data_record.json"):
        bootstrap_madrl_notebook_shared_data(
            cfg,
            root=case_dir,
            notebook_path="notebooks/madrl/train_base_safe.ipynb",
        )


def test_bootstrap_madrl_notebook_shared_data_rejects_missing_shared_data_dir(tmp_path):
    case_dir = make_case_dir(tmp_path, "madrl_notebook_shared_data_missing_dir")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    record_path = case_dir / "notebooks" / "record" / "forecast" / "lstm" / "shared_data_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    missing_shared_dir = (case_dir / "shared_data" / "missing_pkg").resolve()
    record_path.write_text(
        json.dumps(
            {
                "shared_data_dir": str(missing_shared_dir),
                "signature_hash": "sig-missing-dir",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match=str(missing_shared_dir).replace("\\", "\\\\")):
        bootstrap_madrl_notebook_shared_data(
            cfg,
            root=case_dir,
            notebook_path="notebooks/madrl/train_base_safe.ipynb",
        )


def test_bootstrap_madrl_notebook_shared_data_rejects_missing_signature(tmp_path):
    case_dir = make_case_dir(tmp_path, "madrl_notebook_shared_data_missing_signature")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    shared_data_dir = (case_dir / "shared_data" / "pkg_b").resolve()
    shared_data_dir.mkdir(parents=True, exist_ok=True)
    record_path = case_dir / "notebooks" / "record" / "forecast" / "lstm" / "shared_data_record.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"shared_data_dir": str(shared_data_dir)}), encoding="utf-8")

    with pytest.raises(ValueError, match="signature_hash"):
        bootstrap_madrl_notebook_shared_data(
            cfg,
            root=case_dir,
            notebook_path="notebooks/madrl/train_projection_safe.ipynb",
        )


def test_resolve_madrl_notebook_training_requires_shared_data_runtime_contract(tmp_path):
    case_dir = make_case_dir(tmp_path, "madrl_notebook_training_requires_shared_data")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    cfg.forecast.type = "lstm"
    spec = {"algorithm": "MATD3", "experiment_name": "train_base", "env_name": "GridTrainBase"}

    with pytest.raises(ValueError, match="shared_data_record.json"):
        resolve_madrl_notebook_training(
            cfg,
            spec=spec,
            force_retrain_madrl=True,
            root=case_dir,
            notebook_path="notebooks/madrl/train_base.ipynb",
        )


def test_resolve_madrl_notebook_training_allows_perfect_without_shared_data(tmp_path, monkeypatch):
    import scripts.mainline_madrl as mainline_madrl

    case_dir = make_case_dir(tmp_path, "madrl_notebook_training_perfect")
    cfg = make_smoke_config(case_dir, algorithm="MATD3")
    cfg.forecast.type = "perfect"
    spec = {"algorithm": "MATD3", "experiment_name": "train_base", "env_name": "GridTrainBase"}
    captured: dict[str, object] = {}

    def _fake_run_external_train_mainline(**kwargs):
        captured.update(kwargs)
        return {
            "result": {"model_root": str(case_dir / "models" / "run_perfect"), "episodes_completed": 1},
            "launch_info": {
                "result_json_path": str(case_dir / "models" / "run_perfect" / "_meta" / "train_result.json")
            },
        }

    monkeypatch.setattr(mainline_madrl, "run_external_train_mainline", _fake_run_external_train_mainline)

    resolved = resolve_madrl_notebook_training(
        cfg,
        spec=spec,
        force_retrain_madrl=True,
        root=case_dir,
        notebook_path="notebooks/madrl/train_base.ipynb",
    )

    assert captured["data_controls"]["prediction_mode"] == "perfect"
    assert "runtime_controls" not in captured["experiment_controls"]
    assert resolved["model_root"].endswith("run_perfect")


def test_resolve_madrl_notebook_training_keeps_base_and_safe_payloads_aligned(tmp_path, monkeypatch):
    import scripts.mainline_madrl as mainline_madrl

    case_dir = make_case_dir(tmp_path, "madrl_notebook_training_payload_alignment")
    shared_data_dir = (case_dir / "shared_data" / "pkg_c").resolve()
    shared_data_dir.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, object]] = []

    def _fake_run_external_train_mainline(**kwargs):
        captured.append(dict(kwargs))
        experiment_name = str(dict(kwargs["checkpoint_controls"])["experiment_name"])
        return {
            "result": {
                "model_root": str(case_dir / "models" / experiment_name),
                "shared_data_signature": "sig-aligned",
                "episodes_completed": 1,
            },
            "launch_info": {
                "result_json_path": str(case_dir / "models" / experiment_name / "_meta" / "train_result.json")
            },
        }

    monkeypatch.setattr(mainline_madrl, "run_external_train_mainline", _fake_run_external_train_mainline)

    base_cfg = make_smoke_config(case_dir / "base_case", algorithm="MATD3")
    base_cfg.forecast.type = "lstm"
    base_cfg.runtime.shared_data_dir = str(shared_data_dir)
    base_cfg.runtime.shared_data_signature = "sig-aligned"
    base_cfg.reward.action_boundary_penalty_weight = 0.5
    base_cfg.reward.soc_boundary_regularization_weight = 0.25
    base_cfg.reward.throughput_bonus_eur_per_kwh_max = 0.002
    base_cfg.reward.soc_boundary_epsilon = 0.01
    base_cfg.reward.soc_boundary_margin = 0.07
    base_cfg.reward.w_voltage_pen = 0.0
    base_cfg.reward.w_line_pen = 0.0
    base_cfg.reward.w_trafo_pen = 0.0

    safe_cfg = make_smoke_config(case_dir / "safe_case", algorithm="MATD3")
    safe_cfg.forecast.type = "lstm"
    safe_cfg.runtime.shared_data_dir = str(shared_data_dir)
    safe_cfg.runtime.shared_data_signature = "sig-aligned"
    safe_cfg.reward.action_boundary_penalty_weight = 2.0
    safe_cfg.reward.soc_boundary_regularization_weight = 0.75
    safe_cfg.reward.throughput_bonus_eur_per_kwh_max = 0.001
    safe_cfg.reward.soc_boundary_epsilon = 0.03
    safe_cfg.reward.soc_boundary_margin = 0.08
    safe_cfg.reward.w_voltage_pen = 400.0
    safe_cfg.reward.w_line_pen = 0.0
    safe_cfg.reward.w_trafo_pen = 10.0

    resolve_madrl_notebook_training(
        base_cfg,
        spec={"algorithm": "MATD3", "experiment_name": "train_base", "env_name": "GridTrainBase"},
        force_retrain_madrl=True,
        root=case_dir,
        notebook_path="notebooks/madrl/train_base.ipynb",
    )
    resolve_madrl_notebook_training(
        safe_cfg,
        spec={"algorithm": "MATD3", "experiment_name": "train_base_safe", "env_name": "GridTrainBaseSafe"},
        force_retrain_madrl=True,
        root=case_dir,
        notebook_path="notebooks/madrl/train_base_safe.ipynb",
    )

    assert len(captured) == 2
    base_payload, safe_payload = captured
    assert base_payload["data_controls"] == safe_payload["data_controls"]
    assert base_payload["train_controls"] == safe_payload["train_controls"]
    assert base_payload["battery_controls"] == safe_payload["battery_controls"]
    assert base_payload["experiment_controls"]["runtime_controls"] == safe_payload["experiment_controls"]["runtime_controls"]
    assert base_payload["experiment_controls"]["reward_controls"] == {
        "action_boundary_penalty_weight": 0.5,
        "soc_boundary_regularization_weight": 0.25,
        "throughput_bonus_eur_per_kwh_max": 0.002,
        "soc_boundary_epsilon": 0.01,
        "soc_boundary_margin": 0.07,
        "w_voltage_pen": 0.0,
        "w_line_pen": 0.0,
        "w_trafo_pen": 0.0,
        "export_subsidy_eur_per_kwh": 0.0,
        "import_price_markup_eur_per_kwh": 0.0,
        "storage_objective_mode": "max_storage_profit",
        "storage_price_mode": "real_time_price",
        "storage_profit_weight": 1.0,
    }
    assert safe_payload["experiment_controls"]["reward_controls"] == {
        "action_boundary_penalty_weight": 2.0,
        "soc_boundary_regularization_weight": 0.75,
        "throughput_bonus_eur_per_kwh_max": 0.001,
        "soc_boundary_epsilon": 0.03,
        "soc_boundary_margin": 0.08,
        "w_voltage_pen": 400.0,
        "w_line_pen": 0.0,
        "w_trafo_pen": 10.0,
        "export_subsidy_eur_per_kwh": 0.0,
        "import_price_markup_eur_per_kwh": 0.0,
        "storage_objective_mode": "max_storage_profit",
        "storage_price_mode": "real_time_price",
        "storage_profit_weight": 1.0,
    }
    assert base_payload["checkpoint_controls"]["experiment_name"] == "train_base"
    assert safe_payload["checkpoint_controls"]["experiment_name"] == "train_base_safe"
    assert base_payload["env_name"] == "GridTrainBase"
    assert safe_payload["env_name"] == "GridTrainBaseSafe"


def test_forecast_lstm_notebook_uses_shared_forecast_preset():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "forecast" / "forecast_lstm.ipynb"
    joined_source = "\n".join(_load_code_cells(notebook_path))

    assert "force_retrain_forecast = False" in joined_source
    assert "get_mainline_forecast_controls(auto_train_missing=False)" in joined_source
    assert "ensure_lstm_artifacts" in joined_source
    assert "ensure_madrl_shared_data" in joined_source
    assert "predictions.parquet" in joined_source
    assert "shared_data_record.json" in joined_source
    assert "reuse_saved_artifacts" not in joined_source
    assert "legacy_aliases" not in joined_source
    assert "delete_stale_load_artifacts" not in joined_source


def test_forecast_test_notebook_is_read_only_lstm_diagnostic():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "forecast" / "forecast_test.ipynb"
    code_cells = _load_code_cells(notebook_path)
    joined_source = "\n".join(code_cells)

    for cell_index, source in enumerate(code_cells):
        compile(source, f"{notebook_path.name}:cell{cell_index}", "exec")

    assert "ensure_lstm_artifacts" in joined_source
    assert "auto_train_missing = False" in joined_source
    assert "shared_data_record.json" in joined_source
    assert "wholesale_price_seq.npy" in joined_source
    assert "load_seq.npy" in joined_source
    assert "pv_seq.npy" in joined_source
    assert "history_date = str(config_snapshot['test_start_date'])" in joined_source
    assert "forecast_steps = int(config_snapshot['future_horizon'])" in joined_source
    assert "generated_lstm" in joined_source
    assert "shared_data" in joined_source
    assert "train_signal_lstm" not in joined_source
    assert "ensure_madrl_shared_data" not in joined_source
    assert ".to_parquet(" not in joined_source
    assert ".write_text(" not in joined_source


def test_summarize_cfg_supports_fixed_battery_vectors(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_fixed_summary")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    apply_notebook_experiment_settings(
        cfg,
        prediction_mode="perfect",
        test_start_date=20190101,
        test_end_date=20190130,
        agent_profiles=["SFH12", "SFH14"],
        agent_bus_ids=[10, 6],
        load_scale=1.0,
        pv_scale=1.0,
        battery_controls={
            "battery_capacity": [10.0, 12.0],
            "max_charge_rate": 0.5,
        },
        future_horizon=1,
        train_year=2019,
        test_year=2019,
    )

    summary = summarize_experiment(cfg)

    assert summary["battery"]["mode"] == "fixed"
    assert summary["battery"]["battery_capacity"] == [10.0, 12.0]
    assert summary["battery"]["max_charge_rate"] == 0.5
    assert summary["battery"]["p_max_kw"] == [5.0, 6.0]


def test_apply_notebook_experiment_settings_broadcasts_scalar_fixed_battery_capacity(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_fixed_broadcast")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    summary = apply_notebook_experiment_settings(
        cfg,
        prediction_mode="perfect",
        test_start_date=20190101,
        test_end_date=20190130,
        agent_profiles=["SFH12", "SFH14"],
        agent_bus_ids=[10, 6],
        load_scale=1.0,
        pv_scale=1.0,
        battery_controls={
            "battery_capacity": 8.0,
            "max_charge_rate": 0.5,
        },
        future_horizon=1,
        train_year=2019,
        test_year=2019,
    )

    assert cfg.env.battery_capacity == [8.0, 8.0]
    assert summary["battery"]["battery_capacity"] == [8.0, 8.0]
    assert summary["battery"]["p_max_kw"] == [4.0, 4.0]


def test_apply_notebook_experiment_settings_rejects_too_few_agent_bus_ids(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_bus_error")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    with pytest.raises(ValueError, match="only provides 2 buses"):
        apply_notebook_experiment_settings(
            cfg,
            prediction_mode="perfect",
            test_start_date=20190101,
            test_end_date=20190130,
            agent_profiles=["SFH12", "SFH14", "SFH16"],
            agent_bus_ids=[10, 6],
            load_scale=1.0,
            pv_scale=1.0,
            future_horizon=1,
        )


def test_apply_notebook_experiment_settings_rejects_fixed_battery_vector_length_mismatch(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_fixed_capacity_error")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    with pytest.raises(ValueError, match="battery_capacity should provide 2 value\\(s\\)"):
        apply_notebook_experiment_settings(
            cfg,
            prediction_mode="perfect",
            test_start_date=20190101,
            test_end_date=20190130,
            agent_profiles=["SFH12", "SFH14"],
            agent_bus_ids=[10, 6],
            load_scale=1.0,
            pv_scale=1.0,
            battery_controls={
                "battery_capacity": [10.0, 12.0, 8.0],
                "max_charge_rate": 0.5,
            },
            future_horizon=1,
        )


def test_apply_notebook_experiment_settings_rejects_fixed_battery_nonpositive_c_rate(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_fixed_c_rate_error")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    with pytest.raises(ValueError, match="max_charge_rate must be positive"):
        apply_notebook_experiment_settings(
            cfg,
            prediction_mode="perfect",
            test_start_date=20190101,
            test_end_date=20190130,
            agent_profiles=["SFH12", "SFH14"],
            agent_bus_ids=[10, 6],
            load_scale=1.0,
            pv_scale=1.0,
            battery_controls={
                "battery_capacity": [10.0, 12.0],
                "max_charge_rate": 0.0,
            },
            future_horizon=1,
        )


def test_apply_notebook_experiment_settings_rejects_fixed_battery_vector_c_rate(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_notebook_workflow_fixed_c_rate_vector_error")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    with pytest.raises(ValueError, match="positive scalar C-rate"):
        apply_notebook_experiment_settings(
            cfg,
            prediction_mode="perfect",
            test_start_date=20190101,
            test_end_date=20190130,
            agent_profiles=["SFH12", "SFH14"],
            agent_bus_ids=[10, 6],
            load_scale=1.0,
            pv_scale=1.0,
            battery_controls={
                "battery_capacity": [10.0, 12.0],
                "max_charge_rate": [0.5, 0.5],
            },
            future_horizon=1,
        )


def test_normalize_date_input_accepts_compact_dates():
    assert normalize_date_input(20190101) == "2019-01-01"
    assert normalize_date_input("2019-01-30") == "2019-01-30"
    assert normalize_date_input(None) is None


def test_collect_controller_rollout_tracks_full_grid_voltage(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_rollout_voltage")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.data.test_start_date = "2020-01-01"
    cfg.data.test_end_date = "2020-01-01"

    rollout = collect_controller_rollout(
        cfg,
        label="ZeroPolicy",
        action_fn=lambda env, obs: [np.array([0.0, 1.0], dtype=np.float32) for _ in range(env.n)],
    )

    assert not rollout.grid_df.empty
    assert {"bus_id", "vm_pu", "is_agent_bus"}.issubset(rollout.grid_df.columns)
    assert {
        "base_net_load_total",
        "base_net_load_effective_total",
        "net_load_total",
        "agent_raw_net_load_kw",
        "agent_effective_net_load_kw",
        "agent_post_action_net_load_kw",
        "fixed_load_kw",
        "fixed_generation_kw",
        "feeder_raw_net_load_kw",
        "feeder_effective_net_load_kw",
        "feeder_post_action_net_load_kw",
        "pv_raw_total",
        "pv_effective_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
        "trafo_loading_pct_max",
        "n_trafo_violations",
    }.issubset(rollout.step_df.columns)
    assert {
        "agent_id",
        "agent_profile",
        "load",
        "load_pred",
        "pv",
        "pv_pred",
        "e_bat",
        "e_bat_req",
        "soc",
        "purchase_cost",
        "export_subsidy",
        "objective_total",
    }.issubset(rollout.agent_df.columns)
    assert rollout.meta["agent_bus_ids"] == cfg.grid.agent_bus_ids
    assert rollout.meta["v_min_pu"] == cfg.grid.v_min_pu
    assert rollout.meta["v_max_pu"] == cfg.grid.v_max_pu
    assert rollout.meta["trafo_loading_limit_pct"] == cfg.grid.line_max_loading_pct
    assert rollout.meta["import_price_markup_eur_per_kwh"] == pytest.approx(
        cfg.reward.import_price_markup_eur_per_kwh
    )

    agent_power = rollout.agent_df.assign(
        base_net_load=rollout.agent_df["load"].astype(float) - rollout.agent_df["pv"].astype(float),
        net_load=rollout.agent_df["load"].astype(float) - rollout.agent_df["pv"].astype(float) + rollout.agent_df["e_bat"].astype(float),
    )
    aggregated = (
        agent_power.groupby(["episode_idx", "step"], as_index=False)[["base_net_load", "net_load"]]
        .sum()
        .rename(
            columns={
                "base_net_load": "base_net_load_total_from_agents",
                "net_load": "net_load_total_from_agents",
            }
        )
    )
    step_totals = rollout.step_df.loc[
        :, ["episode_idx", "step", "base_net_load_total", "net_load_total"]
    ].copy()
    merged = step_totals.merge(aggregated, on=["episode_idx", "step"], how="inner")
    assert not merged.empty
    assert np.allclose(
        merged["base_net_load_total"],
        merged["base_net_load_total_from_agents"],
    )
    assert np.allclose(
        merged["net_load_total"],
        merged["net_load_total_from_agents"],
    )
    assert np.allclose(rollout.step_df["fixed_load_kw"].astype(float), 0.0)
    assert np.allclose(rollout.step_df["fixed_generation_kw"].astype(float), 0.0)
    assert np.allclose(
        rollout.step_df["feeder_raw_net_load_kw"].astype(float),
        rollout.step_df["agent_raw_net_load_kw"].astype(float),
    )
    assert np.allclose(
        rollout.step_df["feeder_effective_net_load_kw"].astype(float),
        rollout.step_df["agent_effective_net_load_kw"].astype(float),
    )
    assert np.allclose(
        rollout.step_df["feeder_post_action_net_load_kw"].astype(float),
        rollout.step_df["agent_post_action_net_load_kw"].astype(float),
    )

    assert np.allclose(
        rollout.step_df["pv_raw_total"].astype(float),
        rollout.step_df["pv_effective_total"].astype(float) + rollout.step_df["pv_curtail_total"].astype(float),
    )


def test_collect_controller_rollout_skips_forecast_preflight_in_shared_data_mode(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "grid_rollout_shared_data")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.data.test_start_date = "2020-01-01"
    cfg.data.test_end_date = "2020-01-01"
    shared_data = ensure_madrl_shared_data(cfg, root=case_dir / "artifacts" / "training" / "shared_data")
    cfg.runtime.shared_data_dir = str(shared_data.shared_data_dir)
    cfg.runtime.shared_data_signature = str(shared_data.signature_hash)

    def _unexpected_forecast_ready(_cfg):
        raise AssertionError("ensure_forecast_ready should be skipped in shared-data mode")

    monkeypatch.setattr("scripts.utils.grid_notebook_workflow.ensure_forecast_ready", _unexpected_forecast_ready)

    rollout = collect_controller_rollout(
        cfg,
        label="ZeroPolicy",
        action_fn=lambda env, obs: [np.array([0.0, 1.0], dtype=np.float32) for _ in range(env.n)],
    )

    assert not rollout.step_df.empty


def test_collect_controller_rollout_respects_shared_data_selected_episode_indices(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_rollout_shared_data_subset")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    _expand_cfg_to_multiday(cfg, evaluation_days=5)
    shared_data = ensure_madrl_shared_data(cfg, root=case_dir / "artifacts" / "training" / "shared_data")
    cfg.runtime.shared_data_dir = str(shared_data.shared_data_dir)
    cfg.runtime.shared_data_signature = str(shared_data.signature_hash)
    cfg.data.test_start_date = "2020-01-02"
    cfg.data.test_end_date = "2020-01-04"

    rollout = collect_controller_rollout(
        cfg,
        label="ZeroPolicy",
        action_fn=lambda env, obs: [np.array([0.0, 1.0], dtype=np.float32) for _ in range(env.n)],
    )

    assert sorted(rollout.step_df["episode_idx"].unique().tolist()) == [1, 2, 3]
    assert rollout.meta["selected_episode_indices"] == [1, 2, 3]


def test_collect_controller_rollout_can_carry_soc_across_episodes(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_rollout_continuous_soc")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.runtime.selected_episode_indices = [0, 1]

    rollout = collect_controller_rollout(
        cfg,
        label="ChargePolicy",
        action_fn=lambda env, obs: [np.array([0.1, 1.0], dtype=np.float32) for _ in range(env.n)],
        soc_mode="continuous",
    )

    assert rollout.meta["soc_mode"] == "continuous"
    assert {"soc_start", "soc_end", "global_step"}.issubset(rollout.agent_df.columns)
    first_episode_end = (
        rollout.agent_df.loc[
            (rollout.agent_df["episode_idx"] == 0) & (rollout.agent_df["step"] == cfg.env.episode_limit - 1)
        ]
        .sort_values("agent_id")["soc_end"]
        .to_numpy(dtype=np.float32)
    )
    second_episode_start = (
        rollout.agent_df.loc[(rollout.agent_df["episode_idx"] == 1) & (rollout.agent_df["step"] == 0)]
        .sort_values("agent_id")["soc_start"]
        .to_numpy(dtype=np.float32)
    )
    assert np.allclose(second_episode_start, first_episode_end)
    assert not np.allclose(second_episode_start, np.full_like(second_episode_start, cfg.env.init_soc))
    assert rollout.step_df["global_step"].tolist() == list(range(len(rollout.step_df)))


def test_collect_local_mpc_rollout_preserves_interface_for_both_prediction_modes(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "grid_rollout_mpc")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    _enable_normal_comparison_contract(cfg)

    recorded_modes: list[str] = []
    recorded_subsidies: list[float] = []
    recorded_agent_indices: list[int] = []
    recorded_solve_calls: list[dict[str, np.ndarray | float | int]] = []
    recorded_actions: list[np.ndarray] = []

    def _fake_collect_controller_rollout(local_cfg, *, label: str, controller=None, controller_builder=None, action_fn=None, soc_mode="reset", episode_indices=None):
        assert controller is None
        assert controller_builder is None
        assert action_fn is not None
        assert soc_mode == "continuous"
        assert episode_indices is None
        recorded_modes.append(str(local_cfg.forecast.type))

        class _DummyGridNet:
            bus = type("_DummyBus", (), {"index": np.array([10, 6], dtype=np.int32)})()

        class _DummyGridCore:
            net = _DummyGridNet()
            agent_bus_ids = [10, 6]

        class _DummyEnv:
            n = 2
            dt = 1.0
            eff = 1.0
            soc_min = 0.1
            soc_max = 0.9
            soc = np.array([0.5, 0.5], dtype=np.float32)
            agent_c_bat = np.array([4.0, 4.0], dtype=np.float32)
            agent_p_max = np.array([2.0, 2.0], dtype=np.float32)
            reward_fn = type("_DummyReward", (), {"export_subsidy_eur_per_kwh": 0.079})()
            import_price_markup_eur_per_kwh = 0.2
            _grid_core = _DummyGridCore()
            _local_mpc_solver_cache = {}
            _local_mpc_stats = {}

            @staticmethod
            def get_signal_step(signal_name: str):
                if signal_name == "load":
                    return np.array([1.0, 1.2], dtype=np.float32)
                if signal_name == "pv":
                    return np.array([0.6, 0.4], dtype=np.float32)
                raise KeyError(signal_name)

        action_result = action_fn(
            _DummyEnv(),
            {
                    "wholesale_price_seq": np.array([0.2, 0.2], dtype=np.float32),
                "load_seq": np.array([[1.0, 1.0], [1.2, 1.2]], dtype=np.float32),
                "pv_seq": np.zeros((2, 2), dtype=np.float32),
            },
        )
        actions = action_result[0] if isinstance(action_result, tuple) else action_result
        assert len(actions) == 2
        action_array = np.stack([np.asarray(action, dtype=np.float32) for action in actions], axis=0)
        recorded_actions.append(action_array.copy())
        for action in actions:
            value = float(np.asarray(action, dtype=np.float32)[0])
            assert -1.0 <= value <= 1.0

        empty = RolloutResult(
            step_df=pd.DataFrame(),
            agent_df=pd.DataFrame(),
            grid_df=pd.DataFrame(),
            summary=pd.DataFrame(),
            meta={"controller": label},
        )
        return empty

    monkeypatch.setattr(
        "scripts.utils.grid_notebook_workflow.collect_controller_rollout",
        _fake_collect_controller_rollout,
    )

    class _FakeSolver:
        def __init__(self, agent_idx: int) -> None:
            self.agent_idx = agent_idx

        def solve_full_horizon(self, *, import_price_seq, load_seq, pv_seq, soc, pv_curtail_upper_kw=None):
            recorded_solve_calls.append(
                {
                    "agent_idx": int(self.agent_idx),
                    "import_price_seq": np.asarray(import_price_seq, dtype=np.float32).copy(),
                    "load_seq": np.asarray(load_seq, dtype=np.float32).copy(),
                    "pv_seq": np.asarray(pv_seq, dtype=np.float32).copy(),
                    "soc": float(soc),
                    "pv_curtail_upper_kw": np.asarray(pv_curtail_upper_kw, dtype=np.float32).copy(),
                }
            )
            return type(
                "_SolveResult",
                (),
                {
                    "signed_battery_kw": np.array([0.5, 0.0], dtype=np.float32),
                    "pv_curtail_kw": np.array([0.25, 0.0], dtype=np.float32),
                    "feasible": True,
                    "solve_time_sec": 0.01,
                },
            )()

    def _fake_get_solver(
        env,
        *,
        agent_idx,
        import_price_seq,
        battery_capacity_kwh,
        p_max_kw,
        dt_hours,
        efficiency,
        soc_min,
        soc_max,
        export_subsidy_eur_per_kwh,
    ):
        del env, import_price_seq, battery_capacity_kwh, p_max_kw, dt_hours, efficiency, soc_min, soc_max
        recorded_subsidies.append(float(export_subsidy_eur_per_kwh))
        recorded_agent_indices.append(int(agent_idx))
        return _FakeSolver(int(agent_idx)), {
            "solve_count": 0.0,
            "solve_time_sec_total": 0.0,
        }

    monkeypatch.setattr("scripts.mainline_compare._get_local_mpc_solver", _fake_get_solver)

    perfect_rollout = collect_local_mpc_rollout(cfg, prediction_mode="perfect", label="Local MPC (oracle_eval)")
    normal_rollout = collect_local_mpc_rollout(cfg, prediction_mode="normal", label="Local MPC (forecast_eval)")

    assert perfect_rollout.meta["controller"] == "Local MPC (oracle_eval)"
    assert normal_rollout.meta["controller"] == "Local MPC (forecast_eval)"
    assert perfect_rollout.meta["local_mpc_price_mode"] == "real_time_price"
    assert normal_rollout.meta["local_mpc_price_mode"] == "real_time_price"
    assert perfect_rollout.meta["local_mpc_objective_mode"] == "max_storage_profit"
    assert normal_rollout.meta["local_mpc_objective_mode"] == "max_storage_profit"
    assert recorded_modes == ["perfect", "lstm"]
    assert recorded_subsidies == [0.079, 0.079, 0.079, 0.079]
    assert recorded_agent_indices == [0, 1, 0, 1]
    assert len(recorded_solve_calls) == 4
    for call in recorded_solve_calls:
        np.testing.assert_allclose(call["import_price_seq"], np.array([0.4, 0.4], dtype=np.float32))
        np.testing.assert_allclose(call["pv_curtail_upper_kw"], call["pv_seq"])
    assert len(recorded_actions) == 2
    for action_array in recorded_actions:
        np.testing.assert_allclose(action_array[:, 1], np.array([0.16666663, -0.25], dtype=np.float32), atol=1e-5)


def test_collect_local_mpc_rollout_declares_storage_profit_objective(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "grid_rollout_mpc_objective")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    _enable_normal_comparison_contract(cfg)

    def _fake_collect_controller_rollout(local_cfg, *, label: str, controller=None, controller_builder=None, action_fn=None, soc_mode="reset", episode_indices=None):
        del local_cfg, controller, controller_builder, action_fn, episode_indices
        assert soc_mode == "continuous"
        step_df = pd.DataFrame(
            [
                {
                    "controller": label,
                    "purchase_cost_total": 3.0,
                    "export_subsidy_total": 5.0,
                    "objective_total": 9.0,
                    "storage_objective_eur": -0.7,
                }
            ]
        )
        agent_df = pd.DataFrame(
            [
                {
                    "controller": label,
                    "agent_profile": "agent_0",
                    "purchase_cost": 1.0,
                    "export_subsidy": 2.5,
                    "objective_total": 4.0,
                    "storage_objective_eur": -0.4,
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "controller": label,
                    "agent_profile": "agent_0",
                    "purchase_cost": 1.0,
                    "export_subsidy": 2.5,
                    "objective_total": 4.0,
                    "storage_objective_eur": -0.4,
                }
            ]
        )
        return RolloutResult(
            step_df=step_df,
            agent_df=agent_df,
            grid_df=pd.DataFrame(),
            summary=summary,
            meta={"controller": label},
        )

    monkeypatch.setattr(
        "scripts.utils.grid_notebook_workflow.collect_controller_rollout",
        _fake_collect_controller_rollout,
    )

    rollout = collect_local_mpc_rollout(cfg, prediction_mode="normal", label="Local MPC (forecast_eval)")

    assert rollout.meta["local_mpc_objective_mode"] == "max_storage_profit"
    assert float(rollout.step_df.loc[0, "objective_total"]) == pytest.approx(-0.7)
    assert float(rollout.agent_df.loc[0, "objective_total"]) == pytest.approx(-0.4)
    assert float(rollout.summary.loc[0, "objective_total"]) == pytest.approx(-0.4)


def test_get_local_mpc_solver_reuses_solver_per_agent_only(monkeypatch):
    class _DummySolver:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    created_solvers: list[_DummySolver] = []

    def _fake_solver_factory(**kwargs):
        solver = _DummySolver(**kwargs)
        created_solvers.append(solver)
        return solver

    monkeypatch.setattr(
        "controllers.mpc.gurobi_agent_mpc._ReusableLocalMPCSolver",
        _fake_solver_factory,
    )

    class _DummyEnv:
        _local_mpc_solver_cache = {}
        _local_mpc_stats = {}

    env = _DummyEnv()
    solver_a_1, stats = _get_local_mpc_solver(
        env,
        agent_idx=0,
        import_price_seq=np.asarray([0.2, 0.2], dtype=np.float32),
        battery_capacity_kwh=4.0,
        p_max_kw=2.0,
        dt_hours=1.0,
        efficiency=1.0,
        soc_min=0.1,
        soc_max=0.9,
        export_subsidy_eur_per_kwh=0.079,
    )
    solver_a_2, stats = _get_local_mpc_solver(
        env,
        agent_idx=0,
        import_price_seq=np.asarray([0.2, 0.2], dtype=np.float32),
        battery_capacity_kwh=4.0,
        p_max_kw=2.0,
        dt_hours=1.0,
        efficiency=1.0,
        soc_min=0.1,
        soc_max=0.9,
        export_subsidy_eur_per_kwh=0.079,
    )
    solver_b_1, stats = _get_local_mpc_solver(
        env,
        agent_idx=1,
        import_price_seq=np.asarray([0.2, 0.2], dtype=np.float32),
        battery_capacity_kwh=4.0,
        p_max_kw=2.0,
        dt_hours=1.0,
        efficiency=1.0,
        soc_min=0.1,
        soc_max=0.9,
        export_subsidy_eur_per_kwh=0.079,
    )

    assert solver_a_1 is solver_a_2
    assert solver_a_1 is not solver_b_1
    assert len(created_solvers) == 2
    assert stats["solver_build_count"] == pytest.approx(2.0)
    assert stats["solver_reuse_count"] == pytest.approx(1.0)


@pytest.mark.skipif(not HAS_WORKING_GUROBI_LICENSE, reason="requires a working Gurobi installation/license")
def test_collect_global_full_horizon_rollout_reports_continuous_soc_mode(tmp_path):
    case_dir = make_case_dir(tmp_path, "grid_rollout_global_oracle")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.data.test_start_date = "2020-01-01"
    cfg.data.test_end_date = "2020-01-01"

    try:
        rollout = collect_global_full_horizon_rollout(
            cfg,
            label="Global MISOCP Oracle (continuous SoC)",
            time_limit_sec=30.0,
        )
    except RuntimeError as exc:
        if "inf_or_unbd" in str(exc):
            pytest.xfail("Current smoke MISOCP instance can return inf_or_unbd under the storage-profit baseline.")
        raise

    assert rollout.meta["controller"] == "Global MISOCP Oracle (continuous SoC)"
    assert rollout.meta["soc_mode"] == "continuous"
    assert rollout.meta["solve_mode"] in {"single_window", "chunked_window"}
    assert "is_near_optimal" in rollout.meta
    assert rollout.meta["global_oracle_gap"] <= rollout.meta["global_oracle_target_gap"]
    assert "misocp_validation_df" in rollout.meta
    assert isinstance(rollout.meta["misocp_validation_df"], pd.DataFrame)
    assert not rollout.step_df.empty
    assert not rollout.agent_df.empty
    assert not rollout.grid_df.empty


def test_plot_global_misocp_validation_builds_figure():
    validation_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"),
            "soc_slack_max": [0.00003, 0.00004],
            "soc_slack_mean": [0.00001, 0.00002],
            "soc_slack_p95_global": [0.00004, 0.00004],
            "solver_feeder_gap_kw": [0.01, -0.02],
            "replay_feeder_gap_kw": [0.03, -0.04],
            "max_vm_abs_err_pu": [0.0002, 0.0003],
            "max_line_loading_abs_err_pct": [0.1, 0.2],
            "trafo_loading_abs_err_pct": [0.2, 0.3],
            "root_p_abs_err_kw": [0.01, 0.02],
            "within_tolerance": [True, True],
        }
    )
    rollout = RolloutResult(
        step_df=pd.DataFrame(),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={
            "controller": "Global SOCP-MPC + Perfect Forecast",
            "misocp_validation_df": validation_df,
            "misocp_health_warning": "",
        },
    )

    figure = plot_global_misocp_validation(rollout)

    assert len(figure.axes) == 5


def test_plot_global_misocp_validation_requires_tightness_columns():
    validation_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"),
            "max_vm_abs_err_pu": [0.0002, 0.0003],
            "max_line_loading_abs_err_pct": [0.1, 0.2],
            "trafo_loading_abs_err_pct": [0.2, 0.3],
            "root_p_abs_err_kw": [0.01, 0.02],
        }
    )
    rollout = RolloutResult(
        step_df=pd.DataFrame(),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "Global MISOCP", "misocp_validation_df": validation_df},
    )

    with pytest.raises(ValueError, match="soc_slack_max"):
        plot_global_misocp_validation(rollout)


def test_compare_rollout_metrics_returns_expected_columns():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")

    def _make_rollout(controller: str, vm_values: list[float]) -> RolloutResult:
        step_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "wholesale_price": [0.10, 0.20],
                "import_price": [0.30, 0.40],
                "wholesale_price_pred": [0.11, 0.18],
                "import_price_pred": [0.31, 0.38],
                "purchase_cost_total": [1.0, 1.1],
                "export_subsidy_total": [0.1, 0.1],
                "battery_power_kw": [1.0, -1.0],
                "storage_charge_cost_eur": [0.30, 0.0],
                "storage_discharge_revenue_eur": [0.0, 0.40],
                "storage_profit_eur": [-0.30, 0.40],
                "storage_objective_eur": [0.30, -0.40],
                "system_other_cost_eur": [0.10, 0.20],
                "total_eur": [0.10, 0.00],
                "voltage_penalty_total": [0.0, 0.0],
                "line_penalty_total": [0.0, 0.0],
                "trafo_penalty_total": [0.0, 0.0],
                "objective_total": [0.9, 1.0],
                "feeder_post_action_net_load_kw": [1.2, 1.8],
                "trafo_loading_pct_max": [80.0, 105.0],
                "n_trafo_violations": [0, 1],
                "episode_idx": [0, 0],
                "step": [0, 1],
            }
        )
        agent_df = pd.DataFrame(
            {
                "timestamp": list(timestamps) * 2,
                "agent_profile": ["A", "A", "B", "B"],
                "agent_id": [0, 0, 1, 1],
                "episode_idx": [0, 0, 0, 0],
                "step": [0, 1, 0, 1],
                "e_bat": [0.1, -0.1, 0.0, 0.1],
                "soc": [0.5, 0.55, 0.45, 0.5],
                "load": [1.0, 1.1, 0.9, 1.0],
                "load_pred": [0.95, 1.05, 0.88, 0.98],
                "pv": [0.2, 0.25, 0.1, 0.12],
                "pv_pred": [0.18, 0.23, 0.09, 0.10],
                "purchase_cost": [0.4, 0.5, 0.3, 0.4],
                "export_subsidy": [0.0, 0.1, 0.0, 0.0],
                "objective_total": [0.4, 0.4, 0.3, 0.4],
                "controller": [controller] * 4,
            }
        )
        grid_df = pd.DataFrame(
            {
                "controller": [controller] * 4,
                "episode_idx": [0, 0, 0, 0],
                "step": [0, 0, 1, 1],
                "timestamp": list(timestamps.repeat(2)),
                "bus_id": [1, 2, 1, 2],
                "vm_pu": vm_values,
                "is_agent_bus": [False, True, False, True],
            }
        )
        return RolloutResult(
            step_df=step_df,
            agent_df=agent_df,
            grid_df=grid_df,
            summary=pd.DataFrame(
                {
                    "controller": [controller],
                    "agent_profile": ["A"],
                    "purchase_cost": [1.6],
                    "export_subsidy": [0.1],
                    "objective_total": [1.5],
                }
            ),
            meta={
                "controller": controller,
                "agent_profiles": ["A", "B"],
                "agent_bus_ids": [2],
                "v_min_pu": 0.95,
                "v_max_pu": 1.05,
                "trafo_loading_limit_pct": 100.0,
                "import_price_markup_eur_per_kwh": 0.2,
                "dt_hours": 1.0,
                "high_budget_refinement_warn": controller == "Local MPC (forecast_eval)",
                "returned_primary_objective_eur": 1.9,
            },
        )

    metrics_df = compare_rollout_metrics(
        _make_rollout("Local MPC (oracle_eval)", [0.99, 1.01, 1.00, 1.02]),
        _make_rollout("Local MPC (forecast_eval)", [0.94, 1.02, 0.96, 1.06]),
        _make_rollout("DRL (forecast_eval)", [0.98, 1.00, 0.99, 1.01]),
    )

    assert list(metrics_df["controller"]) == [
        "Local MPC (oracle_eval)",
        "Local MPC (forecast_eval)",
        "DRL (forecast_eval)",
    ]
    assert {
        "controller",
        "soc_mode",
        "storage_charge_cost_total_eur",
        "storage_discharge_revenue_total_eur",
        "storage_profit_total_eur",
        "storage_objective_total_eur",
        "system_other_cost_eur",
        "total_eur",
        "voltage_penalty_total",
        "trafo_penalty_total",
        "line_penalty_total",
        "voltage_violation_count",
        "price_mae",
        "load_mae",
        "pv_mae",
        "voltage_violation_steps",
        "voltage_violation_bus_points",
        "min_vm_pu",
        "max_vm_pu",
        "voltage_step_delta_p95_pu",
        "voltage_step_delta_max_pu",
        "voltage_spread_mean_pu",
        "voltage_spread_max_pu",
        "trafo_overload_steps",
        "trafo_loading_max_pct",
        "feeder_netload_ramp_mean_abs_kw",
        "feeder_netload_ramp_max_kw",
        "returned_primary_objective_eur",
    }.issubset(metrics_df.columns)
    assert "purchase_cost_total_eur" not in metrics_df.columns
    assert "export_subsidy_total_eur" not in metrics_df.columns
    assert "total_cost_eur" not in metrics_df.columns
    assert metrics_df.loc[metrics_df["controller"] == "Local MPC (forecast_eval)", "voltage_violation_steps"].item() == 2
    assert metrics_df.loc[metrics_df["controller"] == "Local MPC (oracle_eval)", "total_eur"].item() == pytest.approx(0.1)


def test_compare_rollout_metrics_derives_missing_import_price_from_wholesale_only():
    rollout = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"),
                "wholesale_price": [0.10, 0.20],
                "wholesale_price_pred": [0.12, 0.19],
                "purchase_cost_total": [1.0, 1.1],
                "export_subsidy_total": [0.1, 0.1],
                "battery_power_kw": [1.0, -1.0],
                "objective_total": [0.9, 1.0],
                "feeder_post_action_net_load_kw": [1.0, 1.2],
                "trafo_loading_pct_max": [60.0, 65.0],
                "n_trafo_violations": [0, 0],
                "episode_idx": [0, 0],
                "step": [0, 1],
            }
        ),
        agent_df=pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"),
                "agent_profile": ["A", "A"],
                "agent_id": [0, 0],
                "episode_idx": [0, 0],
                "step": [0, 1],
                "load": [1.0, 1.1],
                "load_pred": [1.0, 1.1],
                "pv": [0.2, 0.3],
                "pv_pred": [0.2, 0.3],
            }
        ),
        grid_df=pd.DataFrame(
            {
                "controller": ["Derived price"] * 2,
                "episode_idx": [0, 0],
                "step": [0, 1],
                "timestamp": pd.date_range("2020-01-01", periods=2, freq="15min"),
                "bus_id": [1, 1],
                "vm_pu": [1.0, 1.0],
                "is_agent_bus": [True, True],
            }
        ),
        summary=pd.DataFrame(),
        meta={
            "controller": "Derived price",
            "agent_profiles": ["A"],
            "agent_bus_ids": [1],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
            "trafo_loading_limit_pct": 100.0,
            "import_price_markup_eur_per_kwh": 0.2,
            "dt_hours": 1.0,
        },
    )

    metrics_df = compare_rollout_metrics(rollout)

    assert metrics_df.loc[0, "price_mae"] == pytest.approx(0.015)


def test_save_and_load_rollout_record_materializes_import_price_from_wholesale(tmp_path):
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")
    rollout = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "episode_idx": [0, 0],
                "step": [0, 1],
                "wholesale_price": [0.10, 0.20],
                "wholesale_price_pred": [0.12, 0.19],
                "battery_power_kw": [0.0, 0.0],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={
            "controller": "Saved rollout",
            "agent_profiles": ["A"],
            "agent_bus_ids": [1],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
            "import_price_markup_eur_per_kwh": 0.2,
            "dt_hours": 1.0,
        },
    )

    save_rollout_record(
        rollout,
        category="forecast",
        scheme_name="price_protocol_roundtrip",
        root=tmp_path,
    )
    loaded = load_rollout_record(
        category="forecast",
        scheme_name="price_protocol_roundtrip",
        root=tmp_path,
    )

    np.testing.assert_allclose(loaded.step_df["import_price"].to_numpy(dtype=np.float32), np.array([0.3, 0.4], dtype=np.float32))
    np.testing.assert_allclose(loaded.step_df["import_price_pred"].to_numpy(dtype=np.float32), np.array([0.32, 0.39], dtype=np.float32))


def test_save_rollout_record_serializes_meta_table_array_columns(tmp_path):
    timestamps = pd.date_range("2020-01-01", periods=1, freq="15min")
    rollout = RolloutResult(
        step_df=pd.DataFrame({"timestamp": timestamps, "episode_idx": [0], "step": [0], "wholesale_price": [0.10], "battery_power_kw": [0.0]}),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={
            "controller": "MISOCP",
            "agent_profiles": ["A"],
            "agent_bus_ids": [1],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
            "import_price_markup_eur_per_kwh": 0.2,
            "dt_hours": 1.0,
            "misocp_validation_df": pd.DataFrame({"timestamp": timestamps, "misocp_vm_pu": [np.asarray([1.0, 0.99], dtype=np.float32)], "max_line_loading_abs_err_pct": [0.1]}),
        },
    )

    save_rollout_record(rollout, category="mpc", scheme_name="meta_array_roundtrip", root=tmp_path)
    loaded = load_rollout_record(category="mpc", scheme_name="meta_array_roundtrip", root=tmp_path)

    encoded = loaded.meta["misocp_validation_df"].loc[0, "misocp_vm_pu"]
    assert json.loads(encoded) == pytest.approx([1.0, 0.99])


def test_build_compare_tables_use_final_dispatch_costs():
    metrics_df = pd.DataFrame(
        {
            "controller": ["Global MISOCP", "MADRL"],
            "storage_charge_cost_total_eur": [2.0, 1.5],
            "storage_discharge_revenue_total_eur": [2.4, 1.9],
            "storage_profit_total_eur": [0.4, 0.4],
            "system_other_cost_eur": [0.1, 0.2],
            "total_eur": [0.3, 0.2],
            "voltage_violation_steps": [1, 0],
            "voltage_violation_bus_points": [2, 0],
            "voltage_step_delta_p95_pu": [0.01, 0.02],
            "voltage_step_delta_max_pu": [0.03, 0.04],
            "voltage_spread_mean_pu": [0.02, 0.03],
            "voltage_spread_max_pu": [0.05, 0.06],
            "trafo_overload_steps": [1, 0],
            "trafo_loading_max_pct": [101.0, 92.0],
            "feeder_netload_ramp_mean_abs_kw": [3.0, 2.5],
            "feeder_netload_ramp_max_kw": [8.0, 6.5],
        }
    )
    economic_df = build_compare_economic_table(metrics_df)
    safety_df = build_compare_safety_table(metrics_df)

    assert list(economic_df.columns) == [
        "controller",
        "storage_discharge_revenue_total_eur",
        "storage_charge_cost_total_eur",
        "storage_profit_total_eur",
    ]
    assert list(safety_df.columns) == [
        "controller",
        "voltage_violation_steps",
        "voltage_violation_bus_points",
        "voltage_step_delta_p95_pu",
        "voltage_step_delta_max_pu",
        "voltage_spread_mean_pu",
        "voltage_spread_max_pu",
        "trafo_overload_steps",
        "trafo_loading_max_pct",
        "feeder_netload_ramp_mean_abs_kw",
        "feeder_netload_ramp_max_kw",
    ]

    rollout = RolloutResult(
        step_df=pd.DataFrame({"trafo_loading_pct_max": [100.0], "n_trafo_violations": [0]}),
        agent_df=pd.DataFrame({"soc": [0.5]}),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={
            "controller": "Global MISOCP",
            "high_budget_refinement_warn": True,
            "formulation_tightening_required": False,
        },
    )


def test_multi_rollout_compare_helpers_render_expected_row_counts():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")

    def _rollout(label: str, offset: float) -> RolloutResult:
        return RolloutResult(
            step_df=pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "wholesale_price": [0.10, 0.20],
                    "import_price": [0.30, 0.40],
                    "wholesale_price_pred": [0.11, 0.21],
                    "import_price_pred": [0.31, 0.41],
                    "base_net_load_total": [2.0 + offset, 2.1 + offset],
                    "base_net_load_effective_total": [1.8 + offset, 1.9 + offset],
                    "net_load_total": [1.7 + offset, 1.8 + offset],
                    "agent_raw_net_load_kw": [2.0 + offset, 2.1 + offset],
                    "agent_effective_net_load_kw": [1.8 + offset, 1.9 + offset],
                    "agent_post_action_net_load_kw": [1.7 + offset, 1.8 + offset],
                    "fixed_load_kw": [0.5, 0.5],
                    "fixed_generation_kw": [0.1, 0.1],
                    "feeder_raw_net_load_kw": [2.4 + offset, 2.5 + offset],
                    "feeder_effective_net_load_kw": [2.2 + offset, 2.3 + offset],
                    "feeder_post_action_net_load_kw": [2.1 + offset, 2.2 + offset],
                    "root_net_exchange_kw": [2.0 + offset, 2.15 + offset],
                    "load_total": [2.4 + offset, 2.5 + offset],
                    "pv_raw_total": [1.2, 1.1],
                    "battery_charge_total": [0.3, 0.1],
                    "pv_effective_total": [1.1, 1.0],
                    "pv_curtail_total": [0.1, 0.1],
                    "grid_import_total": [1.6 + offset, 1.4 + offset],
                    "grid_export_total": [0.0, 0.0],
                    "battery_discharge_total": [0.0, 0.2],
                }
            ),
            agent_df=pd.DataFrame(
                {
                    "controller": [label] * 4,
                    "episode_idx": [0, 0, 0, 0],
                    "step": [0, 0, 1, 1],
                    "timestamp": list(timestamps.repeat(2)),
                    "agent_id": [0, 1, 0, 1],
                    "agent_profile": ["A", "B", "A", "B"],
                    "soc": [0.4, 0.5, 0.45, 0.55],
                }
            ),
            grid_df=pd.DataFrame(
                {
                    "controller": [label] * 4,
                    "episode_idx": [0, 0, 0, 0],
                    "step": [0, 0, 1, 1],
                    "timestamp": list(timestamps.repeat(2)),
                    "bus_id": [1, 2, 1, 2],
                    "vm_pu": [0.99, 1.01, 0.98, 1.02],
                    "is_agent_bus": [False, True, False, True],
                }
            ),
            summary=pd.DataFrame(),
            meta={
                "controller": label,
                "agent_bus_ids": [2],
                "v_min_pu": 0.95,
                "v_max_pu": 1.05,
                "trafo_limit_kw": 4.0,
            },
        )

    rollouts = (_rollout("A", 0.0), _rollout("B", 0.1), _rollout("C", 0.2))
    price_fig = plot_price_prediction_comparison(*rollouts)
    voltage_fig = plot_voltage_profile_comparison(*rollouts)
    net_load_fig = plot_net_load_comparison(*rollouts)
    power_fig = plot_power_balance_comparison(*rollouts)
    battery_fig = plot_battery_power_and_soc_comparison(*rollouts)

    assert len(price_fig.axes) == 1
    assert len(price_fig.axes[0].lines) == 4
    assert price_fig.axes[0].title.get_fontsize() == pytest.approx(20)
    assert price_fig.axes[0].xaxis.label.get_size() == pytest.approx(18)
    assert price_fig.axes[0].yaxis.label.get_size() == pytest.approx(18)
    assert any(label.get_fontsize() == pytest.approx(16) for label in price_fig.axes[0].get_xticklabels())
    assert any(label.get_fontsize() == pytest.approx(16) for label in price_fig.axes[0].get_yticklabels())
    assert len(voltage_fig.axes) == 3
    assert voltage_fig.axes[0].get_legend() is not None
    assert all(axis.get_legend() is None for axis in voltage_fig.axes[1:])
    assert voltage_fig.axes[0].get_shared_y_axes().joined(voltage_fig.axes[0], voltage_fig.axes[1])
    assert voltage_fig.axes[0].title.get_fontsize() == pytest.approx(20)
    assert voltage_fig.axes[0].yaxis.label.get_size() == pytest.approx(18)
    assert all(text.get_fontsize() == pytest.approx(16) for text in voltage_fig.axes[0].get_legend().get_texts())
    assert len(net_load_fig.axes) == 3
    assert net_load_fig.axes[0].get_shared_y_axes().joined(net_load_fig.axes[0], net_load_fig.axes[1])
    assert len(power_fig.axes) == 3
    assert all(len(axis.patches) > 0 for axis in power_fig.axes)
    assert power_fig.axes[0].get_shared_y_axes().joined(power_fig.axes[0], power_fig.axes[1])
    assert all(text.get_fontsize() == pytest.approx(16) for text in power_fig.axes[0].get_legend().get_texts())
    soc_axes = [axis for axis in battery_fig.axes if axis.get_ylabel() == "SoC"]
    power_axes = [axis for axis in battery_fig.axes if axis.get_ylabel() == "Battery Power [kW]"]
    assert len(power_axes) == 3
    assert len(soc_axes) == 3
    assert all(len(axis.patches) > 0 for axis in power_axes)
    assert all(len(axis.lines) >= 3 for axis in soc_axes)
    assert power_axes[0].get_shared_y_axes().joined(power_axes[0], power_axes[1])
    assert all(axis.get_ylim() == (0.0, 1.0) for axis in soc_axes)


def test_plot_net_load_comparison_requires_canonical_feeder_columns():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")
    rollout = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "feeder_raw_net_load_kw": [2.4, 2.6],
                "feeder_effective_net_load_kw": [2.2, 2.4],
                "feeder_post_action_net_load_kw": [1.9, 2.1],
                "root_net_exchange_kw": [1.9, 2.0],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "Legacy MPC", "trafo_limit_kw": 4.0},
    )

    figure = plot_net_load_comparison(rollout)

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) >= 3


def test_plot_net_load_comparison_raises_when_feeder_columns_are_missing():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")
    rollout = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "base_net_load_total": [2.0, 2.2],
                "net_load_total": [1.5, 1.7],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "Broken rollout"},
    )

    with pytest.raises(ValueError, match="feeder net-load columns required for compare plotting"):
        plot_net_load_comparison(rollout)


def test_plot_price_prediction_comparison_requires_canonical_import_price_predictions():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")

    def _rollout(label: str) -> RolloutResult:
        return RolloutResult(
            step_df=pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "import_price": [0.30, 0.40],
                    "import_price_pred": [0.30, 0.40],
                }
            ),
            agent_df=pd.DataFrame(),
            grid_df=pd.DataFrame(),
            summary=pd.DataFrame(),
            meta={"controller": label},
        )

    figure = plot_price_prediction_comparison(_rollout("A"), _rollout("B"))

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 3
    predicted_line = figure.axes[0].lines[1]
    assert np.allclose(predicted_line.get_ydata(), np.asarray([0.30, 0.40], dtype=np.float64))


def test_plot_price_prediction_comparison_overlays_one_line_per_rollout():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")

    def _rollout(label: str, predicted: list[float]) -> RolloutResult:
        return RolloutResult(
            step_df=pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "import_price": [0.30, 0.40],
                    "import_price_pred": predicted,
                }
            ),
            agent_df=pd.DataFrame(),
            grid_df=pd.DataFrame(),
            summary=pd.DataFrame(),
            meta={"controller": label, "prediction_mode": "normal"},
        )

    figure = plot_price_prediction_comparison(
        _rollout("A", [0.10, 0.20]),
        _rollout("B", [0.11, 0.21]),
    )

    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 3
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert labels[0] == "Actual import price"
    assert "Predicted import price (A)" in labels
    assert "Predicted import price (B)" in labels


def test_plot_price_prediction_comparison_reads_each_rollout_import_price_pred():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")

    rollout_a = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "import_price": [0.30, 0.40],
                "import_price_pred": [0.30, 0.40],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={
            "controller": "ADMM MPC",
            "prediction_mode": "normal",
        },
    )
    rollout_b = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "import_price": [0.30, 0.40],
                "import_price_pred": [0.35, 0.45],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={
            "controller": "Local MPC",
            "prediction_mode": "normal",
        },
    )

    figure = plot_price_prediction_comparison(rollout_a, rollout_b)

    assert np.allclose(figure.axes[0].lines[1].get_ydata(), np.asarray([0.30, 0.40], dtype=np.float64))
    assert np.allclose(figure.axes[0].lines[2].get_ydata(), np.asarray([0.35, 0.45], dtype=np.float64))


def test_validate_compare_model_bundles_rejects_missing_or_mismatched_models(tmp_path):
    with pytest.raises(ValueError, match="exactly 3 DRL model roots"):
        validate_compare_model_bundles({"base": tmp_path / "missing"})

    def _write_bundle(run_dir, *, subsidy=0.079, seed=0):
        meta_dir = run_dir / "_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        experiment_controls = {
            "seed": seed,
            "reward_controls": {
                "action_boundary_penalty_weight": 0.05,
                "soc_boundary_regularization_weight": 0.005,
                "throughput_bonus_eur_per_kwh_max": 0.002,
                "soc_boundary_epsilon": 0.02,
                "soc_boundary_margin": 0.02,
                "export_subsidy_eur_per_kwh": subsidy,
                "import_price_markup_eur_per_kwh": 0.2,
                "storage_objective_mode": "max_storage_profit",
                "storage_price_mode": "real_time_price",
                "storage_profit_weight": 1.0,
            },
            "forecast_controls": {"history_window": 96},
            "model_controls": {"hidden_dim": 256},
        }
        data_controls = {
            "prediction_mode": "normal",
            "agent_profiles": ["SFH12", "SFH14"],
            "agent_bus_ids": [6, 10],
            "load_scale": [1.0, 1.0],
            "pv_scale": [1.0, 1.0],
            "future_horizon": 24,
            "train_year": 2019,
            "test_year": 2020,
            "test_start_date": 20200101,
            "test_end_date": 20200103,
        }
        battery_controls = {"battery_capacity": [5.0, 6.0], "max_charge_rate": 0.5}
        train_controls = {
            "profile": "gpu_fast",
            "model_family": "mlp",
            "train_episodes": 10,
            "max_train_steps": 100,
            "num_envs": 2,
            "vec_env_type": "subproc",
            "batch_size": 256,
            "buffer_size": 4096,
            "update_interval": 1,
            "updates_per_step": 1,
            "policy_update_freq": 2,
            "use_noise_decay": True,
            "noise_std_init": 0.2,
            "noise_std_min": 0.05,
        }
        payloads = {
            "train_result.json": {
                "model_root": str(run_dir),
                "training_contract": dict(_CURRENT_COMPARE_TRAINING_CONTRACT),
            },
            "experiment_controls.json": experiment_controls,
            "data_controls.json": data_controls,
            "battery_controls.json": battery_controls,
            "train_controls.json": train_controls,
            "checkpoint_controls.json": {"experiment_name": run_dir.name},
        }
        for filename, payload in payloads.items():
            (meta_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    base_dir = tmp_path / "train_base"
    safe_dir = tmp_path / "train_base_safe"
    proj_dir = tmp_path / "train_projection_safe"
    _write_bundle(base_dir)
    _write_bundle(safe_dir)
    _write_bundle(proj_dir, subsidy=0.081)

    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_compare_model_bundles(
            {
                "train_base": base_dir,
                "train_base_safe": safe_dir,
                "train_projection_safe": proj_dir,
            }
        )


def test_validate_compare_model_bundles_accepts_matching_triplet_with_different_test_windows(tmp_path):
    def _write_bundle(run_dir, *, test_start_date, test_end_date):
        meta_dir = run_dir / "_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        shared_experiment = {
            "seed": 0,
            "reward_controls": {
                "action_boundary_penalty_weight": 0.05,
                "soc_boundary_regularization_weight": 0.005,
                "throughput_bonus_eur_per_kwh_max": 0.002,
                "soc_boundary_epsilon": 0.02,
                "soc_boundary_margin": 0.02,
                "export_subsidy_eur_per_kwh": 0.079,
                "import_price_markup_eur_per_kwh": 0.2,
                "storage_objective_mode": "max_storage_profit",
                "storage_price_mode": "real_time_price",
                "storage_profit_weight": 1.0,
            },
            "forecast_controls": {"history_window": 96},
            "model_controls": {"hidden_dim": 256},
        }
        shared_data = {
            "prediction_mode": "normal",
            "agent_profiles": ["SFH12", "SFH14"],
            "agent_bus_ids": [6, 10],
            "load_scale": [1.0, 1.0],
            "pv_scale": [1.0, 1.0],
            "future_horizon": 24,
            "train_year": 2019,
            "test_year": 2020,
            "test_start_date": int(test_start_date),
            "test_end_date": int(test_end_date),
        }
        shared_battery = {"battery_capacity": [5.0, 6.0], "max_charge_rate": 0.5}
        shared_train = {
            "profile": "gpu_fast",
            "model_family": "mlp",
            "train_episodes": 10,
            "max_train_steps": 100,
            "num_envs": 2,
            "vec_env_type": "subproc",
            "batch_size": 256,
            "buffer_size": 4096,
            "update_interval": 1,
            "updates_per_step": 1,
            "policy_update_freq": 2,
            "use_noise_decay": True,
            "noise_std_init": 0.2,
            "noise_std_min": 0.05,
        }
        payloads = {
            "train_result.json": {
                "model_root": str(run_dir),
                "training_contract": dict(_CURRENT_COMPARE_TRAINING_CONTRACT),
            },
            "experiment_controls.json": shared_experiment,
            "data_controls.json": shared_data,
            "battery_controls.json": shared_battery,
            "train_controls.json": shared_train,
            "checkpoint_controls.json": {"experiment_name": run_dir.name},
        }
        for filename, payload in payloads.items():
            (meta_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    model_roots = {}
    for name, test_start_date, test_end_date in (
        ("train_base", 20200101, 20200103),
        ("train_base_safe", 20200401, 20200415),
        ("train_projection_safe", 20200601, 20200610),
    ):
        run_dir = tmp_path / name
        _write_bundle(run_dir, test_start_date=test_start_date, test_end_date=test_end_date)
        model_roots[name] = run_dir

    bundles = validate_compare_model_bundles(model_roots)
    assert set(bundles) == set(model_roots)


@pytest.mark.parametrize("legacy_reward_key", ["w_action_pen", "action_feasibility_regularization_weight"])
def test_validate_compare_model_bundles_rejects_removed_legacy_reward_keys(tmp_path, legacy_reward_key):
    meta_dir = (tmp_path / "train_base" / "_meta")
    meta_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train_result.json": {
            "model_root": str((tmp_path / "train_base").resolve()),
            "training_contract": dict(_CURRENT_COMPARE_TRAINING_CONTRACT),
        },
        "experiment_controls.json": {
            "seed": 0,
            "reward_controls": {
                legacy_reward_key: 2.0,
            },
            "forecast_controls": {"history_window": 96},
            "model_controls": {"hidden_dim": 256},
        },
        "data_controls.json": {
            "prediction_mode": "normal",
            "agent_profiles": ["SFH12", "SFH14"],
            "agent_bus_ids": [6, 10],
            "load_scale": [1.0, 1.0],
            "pv_scale": [1.0, 1.0],
            "future_horizon": 24,
            "train_year": 2019,
            "test_year": 2020,
            "test_start_date": 20200101,
            "test_end_date": 20200103,
        },
        "battery_controls.json": {"battery_capacity": [5.0, 6.0], "max_charge_rate": 0.5},
        "train_controls.json": {
            "profile": "gpu_fast",
            "model_family": "mlp",
            "train_episodes": 10,
            "max_train_steps": 100,
            "num_envs": 2,
            "vec_env_type": "subproc",
            "batch_size": 256,
            "buffer_size": 4096,
            "update_interval": 1,
            "updates_per_step": 1,
            "policy_update_freq": 2,
            "use_noise_decay": True,
            "noise_std_init": 0.2,
            "noise_std_min": 0.05,
        },
        "checkpoint_controls.json": {"experiment_name": "train_base"},
    }
    for filename, payload in payloads.items():
        (meta_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=legacy_reward_key):
        validate_compare_model_bundles(
            {
                "train_base": tmp_path / "train_base",
                "train_base_safe": tmp_path / "train_base",
                "train_projection_safe": tmp_path / "train_base",
            }
        )
