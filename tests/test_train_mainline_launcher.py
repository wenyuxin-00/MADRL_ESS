from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from configs import compose_experiment_config, recommended_gpu_fast_num_envs
from scripts.utils.train_mainline_launcher import (
    build_train_mainline_command,
    prepare_train_mainline_launch,
)
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


def test_prepare_train_mainline_launch_builds_expected_command(tmp_path):
    launch = prepare_train_mainline_launch(
        project_root=tmp_path,
        experiment_controls={"seed": 7, "algorithm": "MATD3"},
        data_controls={"prediction_mode": "perfect"},
        battery_controls={"mode": "from_pv", "from_pv_power_ratio": 0.5, "from_pv_duration_hours": 2.5},
        train_controls={"profile": "gpu_fast", "launch_mode": "external", "train_episodes": 100},
        checkpoint_controls={"experiment_name": "grid_mainline"},
        env_name="GridTrainMainline",
        run_number=3,
    )
    command = build_train_mainline_command(launch, python_executable="python")

    assert Path(launch["experiment_controls_path"]).exists()
    assert Path(launch["data_controls_path"]).exists()
    assert Path(launch["battery_controls_path"]).exists()
    assert Path(launch["train_controls_path"]).exists()
    assert Path(launch["checkpoint_controls_path"]).exists()
    assert Path(launch["meta_dir"]).exists()
    assert Path(launch["model_root"]).parts[-4:-1] == ("MATD3", "perfect", "grid_mainline")
    assert command[:3] == ["python", "-m", "scripts.run_train_mainline"]
    assert "--battery-controls" in command
    assert "--checkpoint-controls" in command
    assert "--env-name" in command
    assert "GridTrainMainline" in command
    assert "--run-number" in command
    assert "3" in command


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
    experiment_controls = {
        "algorithm": "MATD3",
        "seed": 0,
        "runtime_mode": "performance",
        "device_request": "cpu",
        "require_cuda": False,
        "observation_cache_root": str(tmp_path / "cache"),
    }
    data_controls = {
        "prediction_mode": "perfect",
        "test_start_date": 20200101,
        "test_end_date": 20200103,
        "agent_profiles": ["SFH12", "SFH14"],
        "load_scale": [1.0, 1.0],
        "pv_scale": [1.0, 1.0],
        "future_horizon": 1,
        "train_year": 2019,
        "test_year": 2020,
    }
    battery_controls = {
        "mode": "from_pv",
        "from_pv_power_ratio": 0.5,
        "from_pv_duration_hours": 2.5,
        "battery_capacity": 5.0,
        "max_charge_rate": 2.5,
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
        "train_episodes": 1,
        "max_train_steps": 4,
        "batch_size": 2,
        "buffer_size": 32,
        "update_interval": 1,
        "updates_per_step": 1,
        "policy_update_freq": 2,
        "use_noise_decay": True,
        "show_progress": False,
        "progress_postfix_interval": 2,
        "noise_std_init": 0.2,
        "noise_std_min": 0.05,
    }
    checkpoint_controls = {
        "experiment_name": "grid_mainline",
        "checkpoint_root": str(tmp_path / "checkpoints"),
    }

    experiment_path = controls_dir / "experiment_controls.json"
    data_path = controls_dir / "data_controls.json"
    battery_path = controls_dir / "battery_controls.json"
    train_path = controls_dir / "train_controls.json"
    checkpoint_path = controls_dir / "checkpoint_controls.json"
    result_path = controls_dir / "result.json"

    experiment_path.write_text(json.dumps(experiment_controls), encoding="utf-8")
    data_path.write_text(json.dumps(data_controls), encoding="utf-8")
    battery_path.write_text(json.dumps(battery_controls), encoding="utf-8")
    train_path.write_text(json.dumps(train_controls), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(checkpoint_controls), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "scripts.run_train_mainline",
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
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reward_summary_path = Path(result["reward_summary_path"])
    reward_summary = json.loads(reward_summary_path.read_text(encoding="utf-8"))
    progress_payload = json.loads((Path(result["meta_dir"]) / "progress.json").read_text(encoding="utf-8"))

    assert result["algorithm"] == "MATD3"
    assert result["prediction_mode"] == "perfect"
    assert result["evaluation_mode"] == "oracle_eval"
    assert result["battery_controls"]["mode"] == "from_pv"
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
    assert progress_payload["estimated_end_time"]
    assert progress_payload["remaining_seconds"] == 0.0
    assert "steps_per_sec" in result["perf_summary"]
    assert "avg_env_ms_per_iter" in result["perf_summary"]
    assert "avg_update_ms_per_call" in result["perf_summary"]
    assert "cache_build_time_s" in result["perf_summary"]
    assert "cache_hit" in result["perf_summary"]
