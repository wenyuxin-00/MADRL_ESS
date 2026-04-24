from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from configs.profiles import compose_experiment_config, recommended_gpu_fast_num_envs
from envs.rewards.NormalReward import NormalReward
from scripts.mainline_madrl import _apply_reward_controls, _apply_runtime_controls, _apply_train_controls
from scripts.train import apply_controller_action_postprocessing
from scripts.utils.grid_notebook_workflow import apply_notebook_experiment_settings
from predictors.shared_data import ensure_madrl_shared_data
from tests.support.helpers import write_prosumer_processed_dataset


def test_gpu_fast_profile_defaults_are_predictable_with_28_threads(tmp_path):
    with patch("configs.profiles.os.cpu_count", return_value=28):
        assert recommended_gpu_fast_num_envs() == 20
        cfg = compose_experiment_config(
            profile="gpu_fast",
            algorithm=None,
            data_dir=tmp_path / "data",
            device="cpu",
        )

    assert cfg.algo.name == "MATD3"
    assert cfg.train.num_envs == 20
    assert cfg.train.vec_env_type == "subproc"
    assert cfg.train.batch_size == 4096
    assert cfg.train.buffer_size == 200000
    assert cfg.train.update_interval == 1
    assert cfg.train.updates_per_step == 2
    assert cfg.train.use_noise_decay is True


@pytest.mark.parametrize(
    ("deprecated_key", "value"),
    [
        ("forecast_" "data_source", "precomputed_" "observation_cache"),
        ("observation_" "cache_root", "C:/tmp/cache"),
        ("observation_" "cache_batch_size", 8192),
        ("refresh_" "observation_cache", True),
    ],
)
def test_apply_runtime_controls_rejects_legacy_cache_keys(tmp_path, deprecated_key, value):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    with pytest.raises(ValueError, match=deprecated_key):
        _apply_runtime_controls(cfg, {deprecated_key: value})


def test_apply_reward_controls_supports_step3_reward_fields(tmp_path):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    _apply_reward_controls(
        cfg,
        {
            "action_boundary_penalty_weight": 0.12,
            "soc_boundary_regularization_weight": 0.34,
            "throughput_bonus_eur_per_kwh_max": 0.005,
            "soc_boundary_epsilon": 0.03,
            "soc_boundary_margin": 0.08,
        },
    )
    assert cfg.reward.action_boundary_penalty_weight == pytest.approx(0.12)
    assert cfg.reward.soc_boundary_regularization_weight == pytest.approx(0.34)
    assert cfg.reward.throughput_bonus_eur_per_kwh_max == pytest.approx(0.005)
    assert cfg.reward.soc_boundary_epsilon == pytest.approx(0.03)
    assert cfg.reward.soc_boundary_margin == pytest.approx(0.08)

    _apply_reward_controls(
        cfg,
        {
            "action_boundary_penalty_weight": 0.5,
            "soc_boundary_regularization_weight": 0.25,
            "throughput_bonus_eur_per_kwh_max": 0.001,
            "soc_boundary_epsilon": 0.01,
            "soc_boundary_margin": 0.07,
            "export_subsidy_eur_per_kwh": 0.081,
            "import_price_markup_eur_per_kwh": 0.205,
            "storage_objective_mode": "max_storage_profit",
            "storage_price_mode": "real_time_price",
            "storage_profit_weight": 1.0,
        },
    )

    assert cfg.reward.action_boundary_penalty_weight == pytest.approx(0.5)
    assert cfg.reward.soc_boundary_regularization_weight == pytest.approx(0.25)
    assert cfg.reward.throughput_bonus_eur_per_kwh_max == pytest.approx(0.001)
    assert cfg.reward.soc_boundary_epsilon == pytest.approx(0.01)
    assert cfg.reward.soc_boundary_margin == pytest.approx(0.07)
    assert cfg.reward.export_subsidy_eur_per_kwh == pytest.approx(0.081)
    assert cfg.reward.import_price_markup_eur_per_kwh == pytest.approx(0.205)
    assert cfg.reward.storage_objective_mode == "max_storage_profit"
    assert cfg.reward.storage_price_mode == "real_time_price"
    assert cfg.reward.storage_profit_weight == pytest.approx(1.0)


@pytest.mark.parametrize("legacy_key", ["w_action_pen", "w_soc_pen", "lambda_throughput", "local_action_penalty_mode", "local_action_penalty_weight", "terminal_soc_value_weight", "action_feasibility_regularization_weight"])
def test_apply_reward_controls_rejects_removed_legacy_keys(tmp_path, legacy_key):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    with pytest.raises(ValueError, match=legacy_key):
        _apply_reward_controls(cfg, {legacy_key: 1.0})


def test_apply_reward_controls_rejects_unknown_keys(tmp_path):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    with pytest.raises(ValueError, match="Unknown reward_controls"):
        _apply_reward_controls(cfg, {"action_boundary_penalty_weight": 0.1, "bogus_key": 1.0})


def test_apply_controller_action_postprocessing_keeps_reward_and_merges_action_info(tmp_path):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")
    runner = SimpleNamespace(env_evaluate=SimpleNamespace(reward_fn=NormalReward(cfg)))
    reward = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    base_reward = reward[0, :, 0].copy()
    info_list = [
        {
            "madrl_r_action_penalty": np.zeros(3, dtype=np.float32),
            "madrl_r_total_internal": base_reward.copy(),
        }
    ]
    action_info = {"soc_penalty_unweighted": np.array([[0.0, 0.5, 1.0]], dtype=np.float32)}

    processed_reward, processed_info = apply_controller_action_postprocessing(
        runner,
        reward,
        info_list,
        action_info,
    )

    np.testing.assert_allclose(processed_reward[0, :, 0], base_reward, rtol=1e-6)
    np.testing.assert_allclose(processed_info[0]["reward"], base_reward, rtol=1e-6)
    np.testing.assert_allclose(processed_info[0]["madrl_r_action_penalty"], 0.0, rtol=1e-6)
    np.testing.assert_allclose(processed_info[0]["madrl_r_total_internal"], base_reward, rtol=1e-6)
    np.testing.assert_allclose(processed_info[0]["soc_penalty_unweighted"], action_info["soc_penalty_unweighted"][0], rtol=1e-6)


def test_apply_train_controls_sets_progress_episode_interval(tmp_path):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    _apply_train_controls(cfg, {"progress_episode_interval": 7, "train_window_days": 3, "window_stride_days": 2})

    assert cfg.train.progress_episode_interval == 7
    assert cfg.env.train_window_days == 3
    assert cfg.env.window_stride_days == 2


def test_run_train_mainline_cli_smoke_with_subproc(tmp_path):
    data_dir = tmp_path / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14"],
        train_steps=288,
        test_steps=288,
    )

    controls_dir = tmp_path / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    shared_cfg = compose_experiment_config(
        profile="gpu_fast",
        algorithm="MATD3",
        data_dir=data_dir,
        device="cpu",
        runtime_mode="performance",
        seed=0,
        require_cuda=False,
    )
    shared_cfg.env.train_window_days = 1
    shared_cfg.env.window_stride_days = 1
    apply_notebook_experiment_settings(
        shared_cfg,
        prediction_mode="perfect",
        test_start_date=20200101,
        test_end_date=20200103,
        agent_profiles=["SFH12", "SFH14"],
        agent_bus_ids=[6, 10],
        load_scale=[1.0, 1.0],
        pv_scale=[1.0, 1.0],
        future_horizon=1,
        train_year=2019,
        test_year=2020,
    )
    shared_data = ensure_madrl_shared_data(shared_cfg, root=tmp_path / "shared_data")
    experiment_controls = {
        "algorithm": "MATD3",
        "seed": 0,
        "runtime_mode": "performance",
        "device_request": "cpu",
        "require_cuda": False,
        "runtime_controls": {
            "shared_data_dir": str(shared_data.shared_data_dir),
            "shared_data_signature": str(shared_data.signature_hash),
        },
        "reward_controls": {
            "action_boundary_penalty_weight": 0.05,
            "soc_boundary_regularization_weight": 0.005,
            "throughput_bonus_eur_per_kwh_max": 0.002,
            "soc_boundary_epsilon": 0.02,
            "soc_boundary_margin": 0.02,
            "export_subsidy_eur_per_kwh": 0.079,
            "w_voltage_pen": 0.0,
            "w_line_pen": 0.0,
            "w_trafo_pen": 0.0,
        },
    }
    data_controls = {
        "prediction_mode": "perfect",
        "test_start_date": 20200101,
        "test_end_date": 20200103,
        "agent_profiles": ["SFH12", "SFH14"],
        "agent_bus_ids": [6, 10],
        "load_scale": [1.0, 1.0],
        "pv_scale": [1.0, 1.0],
        "future_horizon": 1,
        "train_year": 2019,
        "test_year": 2020,
    }
    battery_controls = {
        "mode": "fixed",
        "battery_capacity": [5.0, 6.0],
        "max_charge_rate": 0.5,
        "efficiency": 0.95,
        "init_soc": 0.5,
        "train_init_soc_low": 0.2,
        "train_init_soc_high": 0.8,
        "soc_min": 0.05,
        "soc_max": 0.95,
        "soc_target": 0.5,
    }
    train_controls = {
        "launch_mode": "external",
        "profile": "gpu_fast",
        "model_family": "mlp",
        "num_envs": 2,
        "vec_env_type": "subproc",
        "train_episodes": 2,
        "max_train_steps": None,
        "batch_size": 2,
        "buffer_size": 32,
        "update_interval": 1,
        "updates_per_step": 1,
        "policy_update_freq": 2,
        "use_noise_decay": True,
        "show_progress": True,
        "progress_episode_interval": 1,
        "progress_postfix_interval": 2,
        "noise_std_init": 0.2,
        "noise_std_min": 0.05,
        "train_window_days": 1,
        "window_stride_days": 1,
    }
    checkpoint_controls = {
        "experiment_name": "grid_mainline",
        "checkpoint_root": str(tmp_path / "checkpoints"),
    }

    experiment_path = (controls_dir / "experiment_controls.json").resolve()
    data_path = (controls_dir / "data_controls.json").resolve()
    battery_path = (controls_dir / "battery_controls.json").resolve()
    train_path = (controls_dir / "train_controls.json").resolve()
    checkpoint_path = (controls_dir / "checkpoint_controls.json").resolve()
    result_path = (controls_dir / "result.json").resolve()

    experiment_path.write_text(json.dumps(experiment_controls), encoding="utf-8")
    data_path.write_text(json.dumps(data_controls), encoding="utf-8")
    battery_path.write_text(json.dumps(battery_controls), encoding="utf-8")
    train_path.write_text(json.dumps(train_controls), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint_controls), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "scripts.mainline_madrl",
        "--experiment-controls",
        str(experiment_path),
        "--data-controls",
        str(data_path),
        "--battery-controls",
        str(battery_path),
        "--train-controls",
        str(train_path),
        "--checkpoint-controls",
        str(checkpoint_path),
        "--data-dir",
        str(data_dir),
        "--result-json",
        str(result_path),
        "--env-name",
        "GridTrainMainline",
        "--run-number",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "Training:" in combined_output
    assert "avg_reward=" in combined_output
    for token in ("Training log:", "\"episode_idx\"", "\"shared_data_metadata\"", "\"agent_profiles\""):
        assert token not in combined_output
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert f"0/{int(result['perf_summary']['target_total_steps'])}" in combined_output
    reward_summary_path = Path(result["reward_summary_path"])
    reward_summary = json.loads(reward_summary_path.read_text(encoding="utf-8"))
    assert result["algorithm"] == "MATD3"
    assert result["prediction_mode"] == "perfect"
    assert result["evaluation_mode"] == "oracle_eval"
    assert result["battery_controls"]["mode"] == "fixed"
    assert result["summary"]["applied_controls"]["battery"]["battery_capacity"] == [5.0, 6.0]
    assert result["summary"]["applied_controls"]["battery"]["p_max_kw"] == [2.5, 3.0]
    assert result["summary"]["applied_controls"]["agent_bus_ids"] == [6, 10]
    assert result["vec_env"] == "SubprocVecEnv"
    assert result["device"] == "cpu"
    assert result["started_at"]
    assert result["finished_at"]
    assert result["estimated_end_time"]
    assert result["elapsed_seconds"] >= 0.0
    assert Path(result["model_root"]).exists()
    assert Path(result["meta_dir"]).exists()
    assert Path(result["log_path"]).parent == Path(result["meta_dir"])
    assert reward_summary_path.exists()
    assert reward_summary_path.parent == Path(result["meta_dir"])
    assert len(reward_summary["episodes"]) == result["episodes_completed"]
    assert len(reward_summary["episode_total_reward"]) == result["episodes_completed"]
    assert "components" in reward_summary
    assert {
        "madrl_r_inc",
        "madrl_r_action_penalty",
        "madrl_r_soc_regularization",
        "madrl_r_throughput_bonus",
        "madrl_r_safe_v",
        "madrl_r_safe_line",
        "madrl_r_safe_trafo",
        "madrl_r_safe_total",
        "madrl_r_total_internal",
    } == set(reward_summary["components"])
    assert result["experiment_controls"]["reward_controls"]["export_subsidy_eur_per_kwh"] == 0.079
    assert "terminal_soc_value_weight" not in result["experiment_controls"]["reward_controls"]
    assert "training_contract_signature" in result
    assert "training_contract" in result
    assert "action_mapping_contract" not in result["training_contract"]
    assert result["training_contract"]["action_boundary_penalty_weight"] == pytest.approx(0.05)
    assert result["checkpoint_info"]["training_contract_signature"] == result["training_contract_signature"]
    assert "steps_per_sec" in result["perf_summary"]
    assert "avg_env_ms_per_iter" in result["perf_summary"]
    assert "avg_update_ms_per_call" in result["perf_summary"]
    assert "sample_time_s" in result["perf_summary"]
    assert "history_time_s" in result["perf_summary"]
    assert "agent_update_time_s" in result["perf_summary"]
    assert result["perf_summary"]["shared_data_enabled"] is True
    assert result["perf_summary"]["shared_data_signature"] == str(shared_data.signature_hash)
    assert result["shared_data_signature"] == str(shared_data.signature_hash)
    assert result["shared_data_dir"] == str(shared_data.shared_data_dir)
    assert result["safety_summary"]["enabled"] is False
    assert result["safety_controls"]["enabled"] is False


def test_run_train_mainline_cli_supports_matd3_safe_poc(tmp_path):
    data_dir = tmp_path / "data"
    write_prosumer_processed_dataset(
        data_dir,
        agent_profiles=["SFH12", "SFH14"],
        train_steps=288,
        test_steps=288,
    )

    controls_dir = tmp_path / "cs"
    controls_dir.mkdir(parents=True, exist_ok=True)
    experiment_controls = {
        "algorithm": "MATD3_SAFE_POC",
        "seed": 0,
        "runtime_mode": "performance",
        "device_request": "cpu",
        "require_cuda": False,
        "reward_controls": {
            "action_boundary_penalty_weight": 0.05,
            "soc_boundary_regularization_weight": 0.005,
            "throughput_bonus_eur_per_kwh_max": 0.002,
            "soc_boundary_epsilon": 0.02,
            "soc_boundary_margin": 0.02,
            "export_subsidy_eur_per_kwh": 0.079,
        },
        "safety_controls": {
            "enabled": True,
            "projection_iters": 4,
            "voltage_margin_pu": 0.005,
            "line_margin_pct": 5.0,
            "trafo_margin_pct": 5.0,
            "record_diagnostics": True,
        },
    }
    data_controls = {
        "prediction_mode": "perfect",
        "test_start_date": 20200101,
        "test_end_date": 20200103,
        "agent_profiles": ["SFH12", "SFH14"],
        "agent_bus_ids": [6, 10],
        "load_scale": [1.0, 1.0],
        "pv_scale": [1.0, 1.0],
        "future_horizon": 1,
        "train_year": 2019,
        "test_year": 2020,
    }
    battery_controls = {
        "mode": "fixed",
        "battery_capacity": [5.0, 6.0],
        "max_charge_rate": 0.5,
        "efficiency": 0.95,
        "init_soc": 0.5,
        "train_init_soc_low": 0.2,
        "train_init_soc_high": 0.8,
        "soc_min": 0.05,
        "soc_max": 0.95,
        "soc_target": 0.5,
    }
    train_controls = {
        "launch_mode": "external",
        "profile": "base",
        "model_family": "mlp",
        "num_envs": 1,
        "vec_env_type": "dummy",
        "train_episodes": 1,
        "max_train_steps": 4,
        "batch_size": 2,
        "buffer_size": 32,
        "update_interval": 1,
        "updates_per_step": 1,
        "policy_update_freq": 2,
        "use_noise_decay": False,
        "show_progress": False,
        "progress_postfix_interval": 2,
        "noise_std_init": 0.2,
        "noise_std_min": 0.05,
        "train_window_days": 1,
        "window_stride_days": 1,
    }
    checkpoint_controls = {
        "experiment_name": "grid_mainline_safe_poc",
        "checkpoint_root": str(tmp_path / "ck"),
    }

    experiment_path = (controls_dir / "experiment_controls.json").resolve()
    data_path = (controls_dir / "data_controls.json").resolve()
    battery_path = (controls_dir / "battery_controls.json").resolve()
    train_path = (controls_dir / "train_controls.json").resolve()
    checkpoint_path = (controls_dir / "checkpoint_controls.json").resolve()
    result_path = (controls_dir / "result.json").resolve()

    experiment_path.write_text(json.dumps(experiment_controls), encoding="utf-8")
    data_path.write_text(json.dumps(data_controls), encoding="utf-8")
    battery_path.write_text(json.dumps(battery_controls), encoding="utf-8")
    train_path.write_text(json.dumps(train_controls), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint_controls), encoding="utf-8")
    assert experiment_path.exists()
    assert data_path.exists()
    assert battery_path.exists()
    assert train_path.exists()
    assert checkpoint_path.exists()
    assert experiment_path.exists()
    assert data_path.exists()
    assert battery_path.exists()
    assert train_path.exists()
    assert checkpoint_path.exists()

    command = [
        sys.executable,
        "-m",
        "scripts.mainline_madrl",
        "--experiment-controls",
        str(experiment_path),
        "--data-controls",
        str(data_path),
        "--battery-controls",
        str(battery_path),
        "--train-controls",
        str(train_path),
        "--checkpoint-controls",
        str(checkpoint_path),
        "--data-dir",
        str(data_dir),
        "--result-json",
        str(result_path),
        "--env-name",
        "GridTrainMainlineSafePOC",
        "--run-number",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    for token in ("Training log:", "\"episode_idx\"", "\"shared_data_metadata\"", "\"agent_profiles\""):
        assert token not in combined_output
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reward_summary = json.loads(Path(result["reward_summary_path"]).read_text(encoding="utf-8"))

    assert result["algorithm"] == "MATD3_SAFE_POC"
    assert result["experiment_controls"]["reward_controls"]["export_subsidy_eur_per_kwh"] == 0.079
    assert "terminal_soc_value_weight" not in result["experiment_controls"]["reward_controls"]
    assert result["safety_controls"]["enabled"] is True
    assert result["safety_summary"]["enabled"] is True
    assert result["safety_summary"]["projection_batches"] > 0
    assert result["safety_summary"]["rollout_projection_calls"] > 0
    assert result["safety_summary"]["target_projection_calls"] > 0
    assert result["safety_summary"]["actor_projection_calls"] >= 0
    assert result["safety_summary"]["projection_time_s"] >= 0.0
    assert result["safety_summary"]["projector_local_infeasible_count"] >= 0
    assert result["safety_summary"]["mean_pre_trafo_import_violation_kw"] >= 0.0
    assert result["safety_summary"]["mean_pre_trafo_export_violation_kw"] >= 0.0
    assert result["safety_summary"]["mean_post_trafo_import_violation_kw"] >= 0.0
    assert result["safety_summary"]["mean_post_trafo_export_violation_kw"] >= 0.0
    assert result["perf_summary"]["projection_time_s"] >= 0.0
    assert "target_projection_time_s" in result["perf_summary"]
    assert "actor_projection_time_s" in result["perf_summary"]
    assert "madrl_r_soc_regularization" in reward_summary["components"]
    assert "r_terminal_soc_value" not in reward_summary["components"]
    assert Path(result["model_root"]).parts[-4:-1] == ("MATD3_SAFE_POC", "perfect", "grid_mainline_safe_poc")
