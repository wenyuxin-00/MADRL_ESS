from __future__ import annotations

import json
from pathlib import Path

from scripts.run_train_mainline_fastlab import main as fastlab_main
from tests.support.helpers import write_prosumer_processed_dataset


def test_run_train_mainline_fastlab_cli_smoke_with_subproc(tmp_path):
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
        "observation_cache_mode": "precomputed_exact",
        "refresh_observation_cache": False,
        "train_info_mode": "minimal",
        "fast_grid_core": True,
        "observation_cache_batch_size": 8,
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
        "experiment_name": "grid_mainline_fastlab",
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

    exit_code = fastlab_main(
        [
            "--experiment-controls",
            str(experiment_path.resolve()),
            "--data-controls",
            str(data_path.resolve()),
            "--battery-controls",
            str(battery_path.resolve()),
            "--train-controls",
            str(train_path.resolve()),
            "--checkpoint-controls",
            str(checkpoint_path.resolve()),
            "--data-dir",
            str(data_dir.resolve()),
            "--result-json",
            str(result_path.resolve()),
            "--env-name",
            "GridTrainMainlineFastLab",
            "--run-number",
            "1",
        ]
    )

    assert exit_code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reward_summary_path = Path(result["reward_summary_path"])
    reward_summary = json.loads(reward_summary_path.read_text(encoding="utf-8"))
    progress_payload = json.loads((Path(result["meta_dir"]) / "progress.json").read_text(encoding="utf-8"))

    assert result["algorithm"] == "MATD3"
    assert result["prediction_mode"] == "perfect"
    assert result["vec_env"] == "SubprocVecEnvFastLab"
    assert result["device"] == "cpu"
    assert reward_summary_path.exists()
    assert len(reward_summary["episodes"]) == result["episodes_completed"]
    assert progress_payload["remaining_seconds"] == 0.0
    assert result["perf_summary"]["observation_cache_mode"] == "precomputed_exact"
    assert result["perf_summary"]["train_info_mode"] == "minimal"
    assert result["perf_summary"]["observation_cache_batch_size"] == 8
    assert "cache_build_time_s" in result["perf_summary"]
    assert "cache_hit" in result["perf_summary"]
