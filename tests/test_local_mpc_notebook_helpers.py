from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.utils import grid_notebook_workflow as grid_nb
from scripts.utils import local_mpc_notebook_helpers as local_mpc_nb


def _make_cfg(
    *,
    future_horizon: int = 4,
    n_agents: int = 2,
    test_start_date: str = "2020-06-01",
    test_end_date: str = "2020-06-05",
) -> SimpleNamespace:
    return SimpleNamespace(
        env=SimpleNamespace(
            dt=0.25,
            episode_limit=192,
            future_horizon=int(future_horizon),
            num_agents=int(n_agents),
            battery_capacity=[4.0] * int(n_agents),
            max_charge_rate=0.5,
            efficiency=0.95,
            init_soc=0.5,
            soc_min=0.0,
            soc_max=1.0,
            soc_target=0.5,
        ),
        reward=SimpleNamespace(
            import_price_markup_eur_per_kwh=0.2,
            export_subsidy_eur_per_kwh=0.079,
            w_soc_pen=10.0,
            w_voltage_pen=10.0,
            w_line_pen=10.0,
            w_trafo_pen=10.0,
        ),
        grid=SimpleNamespace(
            line_max_loading_pct=100.0,
            v_min_pu=0.95,
            v_max_pu=1.05,
            agent_bus_ids=[idx + 1 for idx in range(int(n_agents))],
        ),
        forecast=SimpleNamespace(type="perfect", lstm_artifact_root=None),
        runtime=SimpleNamespace(shared_data_dir=None, shared_data_signature=None, forecast_ready=None),
        data=SimpleNamespace(
            test_start_date=str(test_start_date),
            test_end_date=str(test_end_date),
            agent_profiles=[f"agent_{idx}" for idx in range(int(n_agents))],
            load_scale=[1.0] * int(n_agents),
            pv_scale=[1.0] * int(n_agents),
        ),
    )


def _make_cached_rollout(
    *,
    controller: str = local_mpc_nb.LOCAL_MPC_LSTM_LABEL,
    forecast_backend: str = "lstm",
) -> grid_nb.RolloutResult:
    timestamp = pd.Timestamp("2020-06-01 00:00:00")
    step_df = pd.DataFrame(
        [
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "wholesale_price": 0.1,
                "import_price": 0.3,
                "wholesale_price_pred": 0.3,
                "import_price_pred": 0.5,
                "purchase_cost_total": 0.33,
                "export_subsidy_total": 0.0,
                "objective_total": 0.33,
            }
        ]
    )
    agent_df = pd.DataFrame(
        [
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "agent_id": 0,
                "agent_profile": "agent_0",
                "load": 1.2,
                "load_pred": 1.2,
                "pv": 0.5,
                "pv_pred": 0.5,
                "e_bat": 0.4,
                "soc": 0.525,
                "purchase_cost": 0.195,
                "export_subsidy": 0.0,
                "objective_total": 0.195,
            },
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "agent_id": 1,
                "agent_profile": "agent_1",
                "load": 1.2,
                "load_pred": 1.2,
                "pv": 0.0,
                "pv_pred": 0.0,
                "e_bat": 0.2,
                "soc": 0.5125,
                "purchase_cost": 0.135,
                "export_subsidy": 0.0,
                "objective_total": 0.135,
            },
        ]
    )
    grid_df = pd.DataFrame(
        [
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "bus_id": 0,
                "vm_pu": 1.0,
                "is_agent_bus": False,
            },
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "bus_id": 1,
                "vm_pu": 0.99,
                "is_agent_bus": True,
            },
            {
                "controller": controller,
                "episode_idx": 0,
                "step": 0,
                "global_step": 0,
                "timestamp": timestamp,
                "bus_id": 2,
                "vm_pu": 1.01,
                "is_agent_bus": True,
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "controller": controller,
                "agent_profile": "agent_0",
                "purchase_cost": 0.195,
                "export_subsidy": 0.0,
                "objective_total": 0.195,
            },
            {
                "controller": controller,
                "agent_profile": "agent_1",
                "purchase_cost": 0.135,
                "export_subsidy": 0.0,
                "objective_total": 0.135,
            },
        ]
    )
    return grid_nb.RolloutResult(
        step_df=step_df,
        agent_df=agent_df,
        grid_df=grid_df,
        summary=summary,
        meta={
            "controller": controller,
            "agent_profiles": ["agent_0", "agent_1"],
            "agent_bus_ids": [1, 2],
            "v_min_pu": 0.95,
            "v_max_pu": 1.05,
            "prediction_mode": "normal" if forecast_backend == "lstm" else "perfect",
            "forecast_backend": forecast_backend,
            "local_mpc_price_mode": local_mpc_nb.DEFAULT_LOCAL_MPC_PRICE_MODE,
            "local_mpc_objective_mode": local_mpc_nb.DEFAULT_LOCAL_MPC_OBJECTIVE_MODE,
            "import_price_markup_eur_per_kwh": 0.2,
            "export_subsidy_eur_per_kwh": 0.079,
        },
    )


def test_local_mpc_rollout_package_round_trip(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=local_mpc_nb.LOCAL_MPC_LSTM_LABEL, forecast_backend="lstm")

    package = local_mpc_nb.build_local_mpc_rollout_package(
        rollout,
        controller_label=rollout.meta["controller"],
        cfg=cfg,
        prediction_mode="normal",
        extra_meta={"rollout_meta": dict(rollout.meta)},
        diagnostic_summary={"avg_local_mpc_solve_time_sec": 0.01},
    )
    saved_dir = local_mpc_nb.save_local_mpc_rollout_package(package, tmp_path / "cached_rollout")
    loaded = local_mpc_nb.load_local_mpc_rollout_package(saved_dir)

    assert int(loaded["manifest"]["rollout_package_version"]) == 2
    assert loaded["diagnostics"]["avg_local_mpc_solve_time_sec"] == pytest.approx(0.01)
    assert pd.api.types.is_datetime64_any_dtype(loaded["step_df"]["timestamp"])
    assert list(loaded["agent_df"]["agent_profile"]) == ["agent_0", "agent_1"]
    assert float(loaded["summary_df"]["objective_total"].sum()) == pytest.approx(0.33)


def test_replay_local_mpc_rollout_package_returns_compare_ready_rollout(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=local_mpc_nb.LOCAL_MPC_LSTM_LABEL, forecast_backend="lstm")
    saved_dir = local_mpc_nb.save_local_mpc_rollout_package(
        local_mpc_nb.build_local_mpc_rollout_package(
            rollout,
            controller_label=rollout.meta["controller"],
            cfg=cfg,
            prediction_mode="normal",
            extra_meta={"rollout_meta": dict(rollout.meta)},
            diagnostic_summary={"avg_local_mpc_solve_time_sec": 0.01},
        ),
        tmp_path / "cached_rollout",
    )

    replay = local_mpc_nb.replay_local_mpc_rollout_package(
        cfg,
        saved_dir,
        label="Replayed Local MPC",
        prediction_mode="normal",
    )

    replay_rollout = replay["rollout"]
    assert replay_rollout.meta["loaded_from_cached_rollout"] is True
    assert replay_rollout.meta["rollout_package_dir"] == str(saved_dir.resolve())
    assert replay_rollout.meta["cached_rollout_diagnostics"]["avg_local_mpc_solve_time_sec"] == pytest.approx(0.01)
    assert replay_rollout.meta["cached_local_mpc_solver_fingerprint"]["price_mode"] == local_mpc_nb.DEFAULT_LOCAL_MPC_PRICE_MODE
    assert replay_rollout.meta["controller"] == "Replayed Local MPC"
    assert replay_rollout.meta["local_mpc_price_mode"] == local_mpc_nb.DEFAULT_LOCAL_MPC_PRICE_MODE
    assert replay_rollout.meta["local_mpc_objective_mode"] == local_mpc_nb.DEFAULT_LOCAL_MPC_OBJECTIVE_MODE
    assert replay_rollout.step_df["controller"].unique().tolist() == ["Replayed Local MPC"]
    assert replay_rollout.summary["controller"].unique().tolist() == ["Replayed Local MPC"]


def test_replay_local_mpc_rollout_package_rejects_cfg_snapshot_mismatch(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=local_mpc_nb.LOCAL_MPC_LSTM_LABEL, forecast_backend="lstm")
    saved_dir = local_mpc_nb.save_local_mpc_rollout_package(
        local_mpc_nb.build_local_mpc_rollout_package(
            rollout,
            controller_label=rollout.meta["controller"],
            cfg=cfg,
            prediction_mode="normal",
            extra_meta={"rollout_meta": dict(rollout.meta)},
        ),
        tmp_path / "cached_rollout",
    )

    mismatched_cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-03")
    mismatched_cfg.env.episode_limit = 96
    mismatched_cfg.forecast.type = "lstm"
    with pytest.raises(ValueError, match="Local MPC rollout package mismatch"):
        local_mpc_nb.replay_local_mpc_rollout_package(
            mismatched_cfg,
            saved_dir,
            prediction_mode="normal",
        )


def test_replay_local_mpc_rollout_package_rejects_solver_fingerprint_mismatch(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=local_mpc_nb.LOCAL_MPC_LSTM_LABEL, forecast_backend="lstm")
    saved_dir = local_mpc_nb.save_local_mpc_rollout_package(
        local_mpc_nb.build_local_mpc_rollout_package(
            rollout,
            controller_label=rollout.meta["controller"],
            cfg=cfg,
            prediction_mode="normal",
            extra_meta={"rollout_meta": dict(rollout.meta)},
        ),
        tmp_path / "cached_rollout",
    )
    manifest_path = saved_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["local_mpc_solver_fingerprint"]["objective_mode"] = "different_objective"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="local_mpc_solver_fingerprint mismatch at 'objective_mode'"):
        local_mpc_nb.replay_local_mpc_rollout_package(
            cfg,
            saved_dir,
            prediction_mode="normal",
        )


def test_resolve_latest_compatible_local_mpc_rollout_package_dir_returns_exact_dir(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=local_mpc_nb.LOCAL_MPC_LSTM_LABEL, forecast_backend="lstm")
    package = local_mpc_nb.build_local_mpc_rollout_package(
        rollout,
        controller_label=rollout.meta["controller"],
        cfg=cfg,
        prediction_mode="normal",
        extra_meta={"rollout_meta": dict(rollout.meta)},
    )

    prefix = tmp_path / "2020-06-01_2020-06-02_agents2_normal_lstm"
    base_dir = local_mpc_nb.save_local_mpc_rollout_package(package, prefix)

    resolved = local_mpc_nb.resolve_latest_compatible_local_mpc_rollout_package_dir(
        prefix,
        cfg=cfg,
        prediction_mode="normal",
    )

    assert base_dir.exists()
    assert resolved == base_dir.resolve()


def test_resolve_latest_compatible_local_mpc_rollout_package_dir_requires_exact_dir(tmp_path):
    cfg = _make_cfg(future_horizon=4, n_agents=2, test_end_date="2020-06-02")
    cfg.env.episode_limit = 96
    cfg.forecast.type = "lstm"
    rollout = _make_cached_rollout(controller=local_mpc_nb.LOCAL_MPC_LSTM_LABEL, forecast_backend="lstm")
    package = local_mpc_nb.build_local_mpc_rollout_package(
        rollout,
        controller_label=rollout.meta["controller"],
        cfg=cfg,
        prediction_mode="normal",
        objective_mode="different_objective",
        extra_meta={"rollout_meta": dict(rollout.meta)},
    )

    prefix = tmp_path / "2020-06-01_2020-06-02_agents2_normal_lstm"
    local_mpc_nb.save_local_mpc_rollout_package(package, tmp_path / f"{prefix.name}_badsolver")

    with pytest.raises(FileNotFoundError, match="requires one exact package directory") as exc_info:
        local_mpc_nb.resolve_latest_compatible_local_mpc_rollout_package_dir(
            prefix,
            cfg=cfg,
            prediction_mode="normal",
        )

    message = str(exc_info.value)
    assert "2020-06-01_2020-06-02_agents2_normal_lstm_badsolver" in message
    assert "exact rollout package directory" in message
