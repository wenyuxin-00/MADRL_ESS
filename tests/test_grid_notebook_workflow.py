import numpy as np
import pandas as pd
import pytest

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
    normalize_date_input,
    plot_rollout_comparison_dashboard,
    resolve_evaluation_mode,
    resolve_forecast_backend,
)
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
        load_scale=[1.2, 0.8],
        pv_scale=0.5,
        battery_controls={
            "mode": "from_pv",
            "from_pv_power_ratio": 0.75,
            "from_pv_duration_hours": 3.0,
            "battery_capacity": 6.0,
            "max_charge_rate": 2.0,
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
    assert np.allclose(cfg.data.load_scale, [1.2, 0.8])
    assert np.allclose(cfg.data.pv_scale, [0.5, 0.5])
    assert cfg.env.battery_mode == "from_pv"
    assert np.isclose(cfg.env.from_pv_power_ratio, 0.75)
    assert np.isclose(cfg.env.from_pv_duration_hours, 3.0)
    assert np.isclose(cfg.env.battery_capacity, 6.0)
    assert np.isclose(cfg.env.max_charge_rate, 2.0)
    assert summary["prediction_mode"] == "normal"
    assert summary["evaluation_mode"] == FORECAST_EVAL_MODE
    assert summary["forecast_backend"] == "lstm"
    assert summary["agent_profiles"] == ["SFH12", "SFH14"]
    assert summary["battery"]["mode"] == "from_pv"
    assert np.isclose(summary["battery"]["from_pv_power_ratio"], 0.75)


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
        action_fn=lambda env, obs: [np.zeros(1, dtype=np.float32) for _ in range(env.n)],
    )

    assert not rollout.grid_df.empty
    assert {"bus_id", "vm_pu", "is_agent_bus"}.issubset(rollout.grid_df.columns)
    assert rollout.meta["agent_bus_ids"] == cfg.grid.agent_bus_ids
    assert rollout.meta["v_min_pu"] == cfg.grid.v_min_pu
    assert rollout.meta["v_max_pu"] == cfg.grid.v_max_pu


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

        actions = action_fn(
            _DummyEnv(),
            {
                "price_seq": np.array([0.2, 0.2], dtype=np.float32),
                "load_seq": np.array([[1.0, 1.0], [1.2, 1.2]], dtype=np.float32),
                "pv_seq": np.zeros((2, 2), dtype=np.float32),
            },
        )
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
                "operating_cost": [1.0, 1.1],
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
                "operating_cost": [0.4, 0.5, 0.3, 0.4],
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
            summary=pd.DataFrame({"controller": [controller], "agent_profile": ["A"], "operating_cost": [1.6]}),
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
        "total_operating_cost",
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
            "total_operating_cost": [1.0, 1.2, 0.9],
            "price_mae": [0.0, 0.1, 0.1],
            "load_mae": [0.0, 0.2, 0.2],
            "pv_mae": [0.0, 0.3, 0.3],
            "voltage_violation_steps": [0, 2, 1],
            "voltage_violation_bus_points": [0, 3, 1],
            "min_vm_pu": [0.97, 0.94, 0.96],
            "max_vm_pu": [1.02, 1.06, 1.03],
            "v_min_pu": [0.95, 0.95, 0.95],
            "v_max_pu": [1.05, 1.05, 1.05],
        }
    )

    figure = plot_rollout_comparison_dashboard(metrics_df)
    assert len(figure.axes) == 6
