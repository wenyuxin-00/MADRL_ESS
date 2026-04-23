from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from configs.profiles import compose_experiment_config, recommended_gpu_fast_num_envs
from scripts.mainline_madrl import _apply_reward_controls, _apply_runtime_controls, _apply_train_controls
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


def test_apply_reward_controls_supports_w_soc_pen_only(tmp_path):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    _apply_reward_controls(cfg, {"w_soc_pen": 5.0})
    assert cfg.reward.w_soc_pen == pytest.approx(5.0)

    _apply_reward_controls(
        cfg,
        {
            "export_subsidy_eur_per_kwh": 0.081,
            "import_price_markup_eur_per_kwh": 0.205,
            "w_soc_pen": 0.0,
        },
    )

    assert cfg.reward.export_subsidy_eur_per_kwh == pytest.approx(0.081)
    assert cfg.reward.import_price_markup_eur_per_kwh == pytest.approx(0.205)
    assert cfg.reward.w_soc_pen == pytest.approx(0.0)


@pytest.mark.parametrize("legacy_key", ["w_action_pen", "lambda_throughput"])
def test_apply_reward_controls_rejects_removed_legacy_keys(tmp_path, legacy_key):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    with pytest.raises(ValueError, match=legacy_key):
        _apply_reward_controls(cfg, {legacy_key: 1.0})


def test_apply_reward_controls_rejects_unknown_keys(tmp_path):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    with pytest.raises(ValueError, match="Unknown reward_controls"):
        _apply_reward_controls(cfg, {"w_soc_pen": 5.0, "bogus_key": 1.0})


def test_apply_train_controls_sets_progress_episode_interval(tmp_path):
    cfg = compose_experiment_config(profile="base", algorithm="MATD3", data_dir=tmp_path / "data", device="cpu")

    _apply_train_controls(cfg, {"progress_episode_interval": 7})

    assert cfg.train.progress_episode_interval == 7


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
            "export_subsidy_eur_per_kwh": 0.079,
            "w_soc_pen": 0.0,
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
    assert "0/96" in combined_output
    assert "avg_reward=" in combined_output
    for token in ("Training log:", "\"episode_idx\"", "\"shared_data_metadata\"", "\"agent_profiles\""):
        assert token not in combined_output
    result = json.loads(result_path.read_text(encoding="utf-8"))
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
    assert {"r_purchase_cost", "r_export_subsidy", "r_soc_pen", "r_safe_v", "r_safe_line", "r_safe_trafo"} == set(
        reward_summary["components"]
    )
    assert result["experiment_controls"]["reward_controls"]["export_subsidy_eur_per_kwh"] == 0.079
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
            "export_subsidy_eur_per_kwh": 0.079,
            "w_soc_pen": 0.0,
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
    assert "r_soc_pen" in reward_summary["components"]
    assert Path(result["model_root"]).parts[-4:-1] == ("MATD3_SAFE_POC", "perfect", "grid_mainline_safe_poc")
