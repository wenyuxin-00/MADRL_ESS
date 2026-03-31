from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmarks.run_mainline_ab import main as benchmark_main
from tests.support.helpers import write_prosumer_processed_dataset


def test_run_mainline_ab_smoke_outputs_cold_and_warm_mainline_rows(tmp_path):
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
        "seed": 3,
        "runtime_mode": "strict_reproducibility",
        "device_request": "cpu",
        "require_cuda": False,
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
        "battery_capacity": [5.0, 5.0],
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
        "num_envs": 1,
        "vec_env_type": "dummy",
        "train_episodes": 1,
        "max_train_steps": 3,
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
        "experiment_name": "grid_mainline_perf_ab_smoke",
        "checkpoint_root": str(tmp_path / "checkpoints"),
    }

    for name, payload in {
        "experiment_controls.json": experiment_controls,
        "data_controls.json": data_controls,
        "battery_controls.json": battery_controls,
        "train_controls.json": train_controls,
        "checkpoint_controls.json": checkpoint_controls,
    }.items():
        (controls_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    output_dir = tmp_path / "benchmark_output"
    exit_code = benchmark_main(
        [
            "--experiment-controls",
            str((controls_dir / "experiment_controls.json").resolve()),
            "--data-controls",
            str((controls_dir / "data_controls.json").resolve()),
            "--battery-controls",
            str((controls_dir / "battery_controls.json").resolve()),
            "--train-controls",
            str((controls_dir / "train_controls.json").resolve()),
            "--checkpoint-controls",
            str((controls_dir / "checkpoint_controls.json").resolve()),
            "--data-dir",
            str(data_dir.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--episode-budgets",
            "1",
            "--run-number",
            "1",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert Path(summary["leaderboard_path"]).exists()
    assert len(summary["rows"]) == 2

    candidate_names = {row["candidate_name"] for row in summary["rows"]}
    assert candidate_names == {"mainline_cold", "mainline_warm"}

    warm_row = next(row for row in summary["rows"] if row["candidate_name"] == "mainline_warm")
    assert warm_row["cache_hit"] is True
    for row in summary["rows"]:
        assert Path(row["reward_summary_path"]).exists()
        assert Path(row["rollout_summary_path"]).exists()
        assert row["episodes_completed"] == 1
        rollout_summary = json.loads(Path(row["rollout_summary_path"]).read_text(encoding="utf-8"))
        assert rollout_summary["meta"]["agent_bus_ids"] == [6, 10]
