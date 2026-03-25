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
        train_controls={"profile": "gpu_fast", "launch_mode": "external"},
        env_name="GridTrainMainline",
        run_number=3,
    )
    command = build_train_mainline_command(launch, python_executable="python")

    assert Path(launch["experiment_controls_path"]).exists()
    assert Path(launch["data_controls_path"]).exists()
    assert Path(launch["train_controls_path"]).exists()
    assert command[:3] == ["python", "-m", "scripts.run_train_mainline"]
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
    }
    data_controls = {
        "prediction_mode": "perfect",
        "test_start_date": 20200101,
        "test_end_date": 20200103,
        "agent_profiles": ["SFH12", "SFH14"],
        "load_scale": [1.0, 1.0],
        "pv_scale": [1.0, 1.0],
        "storage_scale": [1.0, 1.0],
        "future_horizon": 1,
        "train_year": 2019,
        "test_year": 2020,
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
        "progress_postfix_interval": 1,
        "noise_std_init": 0.2,
        "noise_std_min": 0.05,
    }

    experiment_path = controls_dir / "experiment_controls.json"
    data_path = controls_dir / "data_controls.json"
    train_path = controls_dir / "train_controls.json"
    result_path = controls_dir / "result.json"
    save_dir = tmp_path / "checkpoints"

    experiment_path.write_text(json.dumps(experiment_controls), encoding="utf-8")
    data_path.write_text(json.dumps(data_controls), encoding="utf-8")
    train_path.write_text(json.dumps(train_controls), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "scripts.run_train_mainline",
        "--experiment-controls",
        str(experiment_path),
        "--data-controls",
        str(data_path),
        "--train-controls",
        str(train_path),
        "--data-dir",
        str(data_dir),
        "--save-dir",
        str(save_dir),
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
    assert result["algorithm"] == "MATD3"
    assert result["vec_env"] == "SubprocVecEnv"
    assert result["device"] == "cpu"
    assert "steps_per_sec" in result["perf_summary"]
    assert "avg_env_ms_per_iter" in result["perf_summary"]
    assert "avg_update_ms_per_call" in result["perf_summary"]
