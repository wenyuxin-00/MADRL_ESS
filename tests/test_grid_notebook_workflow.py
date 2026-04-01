import matplotlib
import json
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from scripts.utils.grid_notebook_workflow import (
    FORECAST_EVAL_MODE,
    NORMAL_PREDICTION_MODE,
    ORACLE_EVAL_MODE,
    PERFECT_PREDICTION_MODE,
    RolloutResult,
    apply_notebook_experiment_settings,
    compare_rollout_metrics,
    collect_mpc_rollout,
    collect_controller_rollout,
    load_training_run_bundle,
    normalize_date_input,
    plot_net_load_comparison,
    plot_power_balance_bars,
    plot_power_balance_comparison,
    plot_rollout_comparison_dashboard,
    plot_voltage_profile_comparison,
    resolve_evaluation_mode,
    resolve_forecast_backend,
    validate_compare_model_bundles,
)
from scripts.utils.forecast_shared_preset import get_managed_lstm_forecast_controls
from scripts.utils.experiment_notebook_utils import summarize_cfg
from scripts.utils.madrl_shared_data import ensure_madrl_shared_data
from tests.support.helpers import make_case_dir, make_smoke_config


def test_resolve_forecast_backend_handles_mainline_modes():
    assert resolve_forecast_backend(PERFECT_PREDICTION_MODE, 24) == "perfect"
    assert resolve_forecast_backend(NORMAL_PREDICTION_MODE, 24) == "lstm"
    with pytest.raises(ValueError, match="future_horizon > 0"):
        resolve_forecast_backend(NORMAL_PREDICTION_MODE, 0)


def test_resolve_evaluation_mode_maps_prediction_modes():
    assert resolve_evaluation_mode(PERFECT_PREDICTION_MODE) == ORACLE_EVAL_MODE
    assert resolve_evaluation_mode(NORMAL_PREDICTION_MODE) == FORECAST_EVAL_MODE


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
    assert cfg.obs.sequence_features == ["price", "load", "pv"]
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
    forecast_controls = get_managed_lstm_forecast_controls(
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

    summary = summarize_cfg(cfg)

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
        "pv_raw_total",
        "pv_effective_total",
        "pv_curtail_total",
        "grid_import_total",
        "grid_export_total",
    }.issubset(rollout.step_df.columns)
    assert {
        "base_net_load",
        "base_net_load_effective",
        "net_load",
        "pv_raw",
        "pv_effective",
        "pv_curtail",
        "pv_utilization",
        "grid_import_kw",
        "grid_export_kw",
        "battery_action_req",
        "battery_action_exec",
        "pv_action_req",
        "pv_action_exec",
        "controller_action_gap",
    }.issubset(rollout.agent_df.columns)
    assert rollout.meta["agent_bus_ids"] == cfg.grid.agent_bus_ids
    assert rollout.meta["v_min_pu"] == cfg.grid.v_min_pu
    assert rollout.meta["v_max_pu"] == cfg.grid.v_max_pu

    aggregated = (
        rollout.agent_df.groupby(["episode_idx", "step"], as_index=False)[["base_net_load", "net_load"]]
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

    pv_merged = (
        rollout.agent_df.groupby(["episode_idx", "step"], as_index=False)[["pv_raw", "pv_effective", "pv_curtail"]]
        .sum()
        .rename(
            columns={
                "pv_raw": "pv_raw_total_from_agents",
                "pv_effective": "pv_effective_total_from_agents",
                "pv_curtail": "pv_curtail_total_from_agents",
            }
        )
    )
    merged = rollout.step_df.merge(pv_merged, on=["episode_idx", "step"], how="inner")
    assert not merged.empty
    assert np.allclose(merged["pv_raw_total"], merged["pv_raw_total_from_agents"])
    assert np.allclose(merged["pv_effective_total"], merged["pv_effective_total_from_agents"])
    assert np.allclose(merged["pv_curtail_total"], merged["pv_curtail_total_from_agents"])


def test_collect_controller_rollout_skips_forecast_preflight_in_shared_data_mode(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "grid_rollout_shared_data")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
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


def test_collect_mpc_rollout_preserves_interface_for_both_prediction_modes(tmp_path, monkeypatch):
    case_dir = make_case_dir(tmp_path, "grid_rollout_mpc")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")

    recorded_modes: list[str] = []

    def _fake_collect_controller_rollout(local_cfg, *, label: str, controller=None, action_fn=None):
        assert controller is None
        assert action_fn is not None
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
            _grid_core = _DummyGridCore()

        action_result = action_fn(
            _DummyEnv(),
            {
                "price_seq": np.array([0.2, 0.2], dtype=np.float32),
                "load_seq": np.array([[1.0, 1.0], [1.2, 1.2]], dtype=np.float32),
                "pv_seq": np.zeros((2, 2), dtype=np.float32),
            },
        )
        actions = action_result[0] if isinstance(action_result, tuple) else action_result
        assert len(actions) == 2
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
    monkeypatch.setattr(
        "scripts.utils.grid_notebook_workflow.solve_single_agent_gurobi_mpc_action",
        lambda **kwargs: 0.5,
    )

    perfect_rollout = collect_mpc_rollout(cfg, prediction_mode="perfect", label="MPC (oracle_eval)")
    normal_rollout = collect_mpc_rollout(cfg, prediction_mode="normal", label="MPC (forecast_eval)")

    assert perfect_rollout.meta["controller"] == "MPC (oracle_eval)"
    assert normal_rollout.meta["controller"] == "MPC (forecast_eval)"
    assert recorded_modes == ["perfect", "lstm"]


def test_compare_rollout_metrics_returns_expected_columns():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")

    def _make_rollout(controller: str, vm_values: list[float]) -> RolloutResult:
        step_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": [0.10, 0.20],
                "price_pred": [0.11, 0.18],
                "purchase_cost_total": [1.0, 1.1],
                "export_subsidy_total": [0.1, 0.1],
                "voltage_penalty_total": [0.0, 0.0],
                "line_penalty_total": [0.0, 0.0],
                "trafo_penalty_total": [0.0, 0.0],
                "objective_total": [0.9, 1.0],
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
            },
        )

    metrics_df = compare_rollout_metrics(
        _make_rollout("MPC (oracle_eval)", [0.99, 1.01, 1.00, 1.02]),
        _make_rollout("MPC (forecast_eval)", [0.94, 1.02, 0.96, 1.06]),
        _make_rollout("DRL (forecast_eval)", [0.98, 1.00, 0.99, 1.01]),
    )

    assert list(metrics_df["controller"]) == [
        "MPC (oracle_eval)",
        "MPC (forecast_eval)",
        "DRL (forecast_eval)",
    ]
    assert {
        "controller",
        "purchase_cost_total",
        "export_subsidy_total",
        "objective_total",
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
    }.issubset(metrics_df.columns)
    assert metrics_df.loc[metrics_df["controller"] == "MPC (forecast_eval)", "voltage_violation_steps"].item() == 2


def test_plot_rollout_comparison_dashboard_accepts_three_rollouts():
    metrics_df = pd.DataFrame(
        {
            "controller": ["MPC (oracle_eval)", "MPC (forecast_eval)", "DRL (forecast_eval)"],
            "purchase_cost_total": [1.0, 1.2, 0.9],
            "export_subsidy_total": [0.1, 0.1, 0.2],
            "objective_total": [0.9, 1.3, 0.7],
            "voltage_violation_count": [0, 3, 1],
            "trafo_penalty_total": [0.0, 0.4, 0.1],
            "line_penalty_total": [0.0, 0.2, 0.0],
        }
    )

    figure = plot_rollout_comparison_dashboard(metrics_df)
    assert len(figure.axes) == 6


def test_plot_power_balance_bars_accepts_rollout_with_balance_columns():
    timestamps = pd.date_range("2020-01-01", periods=3, freq="15min")
    rollout = RolloutResult(
        step_df=pd.DataFrame(
            {
                "timestamp": timestamps,
                "load_total": [2.4, 2.5, 2.6],
                "battery_charge_total": [0.3, 0.1, 0.0],
                "pv_effective_total": [1.2, 1.0, 0.8],
                "pv_curtail_total": [0.2, 0.1, 0.0],
                "grid_import_total": [0.9, 1.2, 1.4],
                "grid_export_total": [0.0, 0.0, 0.0],
                "battery_discharge_total": [0.0, 0.2, 0.4],
            }
        ),
        agent_df=pd.DataFrame(),
        grid_df=pd.DataFrame(),
        summary=pd.DataFrame(),
        meta={"controller": "DRL (forecast_eval)"},
    )

    figure = plot_power_balance_bars(rollout)
    assert len(figure.axes) == 1


def test_multi_rollout_compare_helpers_render_expected_row_counts():
    timestamps = pd.date_range("2020-01-01", periods=2, freq="15min")

    def _rollout(label: str, offset: float) -> RolloutResult:
        return RolloutResult(
            step_df=pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "base_net_load_total": [2.0 + offset, 2.1 + offset],
                    "base_net_load_effective_total": [1.8 + offset, 1.9 + offset],
                    "net_load_total": [1.7 + offset, 1.8 + offset],
                    "load_total": [2.4 + offset, 2.5 + offset],
                    "battery_charge_total": [0.3, 0.1],
                    "pv_effective_total": [1.1, 1.0],
                    "pv_curtail_total": [0.1, 0.1],
                    "grid_import_total": [0.9 + offset, 1.0 + offset],
                    "grid_export_total": [0.0, 0.0],
                    "battery_discharge_total": [0.0, 0.2],
                }
            ),
            agent_df=pd.DataFrame(),
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
    voltage_fig = plot_voltage_profile_comparison(*rollouts)
    net_load_fig = plot_net_load_comparison(*rollouts)
    power_fig = plot_power_balance_comparison(*rollouts)

    assert len(voltage_fig.axes) == 3
    assert len(net_load_fig.axes) == 3
    assert len(power_fig.axes) == 3


def test_validate_compare_model_bundles_rejects_missing_or_mismatched_models(tmp_path):
    with pytest.raises(ValueError, match="exactly 3 DRL model roots"):
        validate_compare_model_bundles({"base": tmp_path / "missing"})

    def _write_bundle(run_dir, *, subsidy=0.079, seed=0):
        meta_dir = run_dir / "_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        experiment_controls = {
            "seed": seed,
            "reward_controls": {
                "export_subsidy_eur_per_kwh": subsidy,
                "lambda_throughput": 0.0,
                "w_action_pen": 0.0,
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
            "train_result.json": {"model_root": str(run_dir)},
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

    assert load_training_run_bundle(base_dir)["model_root"] == str(base_dir.resolve())
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_compare_model_bundles(
            {
                "train_base": base_dir,
                "train_base_safe": safe_dir,
                "train_projection_safe": proj_dir,
            }
        )


def test_validate_compare_model_bundles_accepts_matching_triplet(tmp_path):
    def _write_bundle(run_dir):
        meta_dir = run_dir / "_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        shared_experiment = {
            "seed": 0,
            "reward_controls": {
                "export_subsidy_eur_per_kwh": 0.079,
                "lambda_throughput": 0.0,
                "w_action_pen": 0.0,
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
            "test_start_date": 20200101,
            "test_end_date": 20200103,
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
            "train_result.json": {"model_root": str(run_dir)},
            "experiment_controls.json": shared_experiment,
            "data_controls.json": shared_data,
            "battery_controls.json": shared_battery,
            "train_controls.json": shared_train,
            "checkpoint_controls.json": {"experiment_name": run_dir.name},
        }
        for filename, payload in payloads.items():
            (meta_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    model_roots = {}
    for name in ("train_base", "train_base_safe", "train_projection_safe"):
        run_dir = tmp_path / name
        _write_bundle(run_dir)
        model_roots[name] = run_dir

    bundles = validate_compare_model_bundles(model_roots)
    assert set(bundles) == set(model_roots)
