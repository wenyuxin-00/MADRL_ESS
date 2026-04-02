"""Helpers for launching the training mainline outside notebooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.checkpoints import build_training_run_paths, slugify_checkpoint_token
from scripts.utils.project_paths import get_checkpoint_root


def _default_algorithm(experiment_controls: dict[str, Any], train_controls: dict[str, Any]) -> str:
    algorithm = experiment_controls.get("algorithm")
    if algorithm:
        return str(algorithm)
    return "MATD3" if str(train_controls.get("profile", "base")) == "gpu_fast" else "MADDPG"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def prepare_train_mainline_launch(
    *,
    project_root,
    experiment_controls: dict[str, Any],
    data_controls: dict[str, Any],
    train_controls: dict[str, Any],
    battery_controls: dict[str, Any] | None = None,
    checkpoint_controls: dict[str, Any] | None = None,
    data_dir=None,
    save_dir=None,
    env_name: str = "GridTrainMainline",
    run_number: int = 1,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    data_dir = Path(data_dir).resolve() if data_dir is not None else (project_root / "data").resolve()
    checkpoint_controls = dict(checkpoint_controls or {})

    algorithm = _default_algorithm(experiment_controls, train_controls)
    prediction_mode = str(data_controls.get("prediction_mode", "perfect")).strip().lower()
    checkpoint_root = Path(
        checkpoint_controls.get("checkpoint_root") or get_checkpoint_root(project_root)
    ).resolve()
    experiment_name = slugify_checkpoint_token(
        checkpoint_controls.get("experiment_name", "grid_mainline"),
        default="grid_mainline",
    )

    if save_dir is None:
        run_paths = build_training_run_paths(
            checkpoint_root,
            algorithm=algorithm,
            prediction_mode=prediction_mode,
            experiment_name=experiment_name,
            train_episodes=train_controls.get("train_episodes"),
            max_train_steps=train_controls.get("max_train_steps"),
        )
        model_root = Path(run_paths["model_root"]).resolve()
        meta_dir = Path(run_paths["meta_dir"]).resolve()
        run_label = str(run_paths["run_label"])
        prediction_mode = str(run_paths["prediction_mode"])
        experiment_name = str(run_paths["experiment_name"])
    else:
        model_root = Path(save_dir).resolve()
        meta_dir = (model_root / "_meta").resolve()
        run_label = model_root.name

    meta_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "experiment_name": experiment_name,
        "checkpoint_root": str(checkpoint_root),
        "prediction_mode": prediction_mode,
        "run_label": run_label,
        "model_root": str(model_root),
    }
    battery_controls = dict(battery_controls or {})

    experiment_controls_path = _write_json(meta_dir / "experiment_controls.json", experiment_controls)
    data_controls_path = _write_json(meta_dir / "data_controls.json", data_controls)
    battery_controls_path = _write_json(meta_dir / "battery_controls.json", battery_controls)
    train_controls_path = _write_json(meta_dir / "train_controls.json", train_controls)
    checkpoint_controls_path = _write_json(meta_dir / "checkpoint_controls.json", checkpoint_payload)
    result_json_path = meta_dir / "train_result.json"
    progress_json_path = meta_dir / "progress.json"
    log_path = meta_dir / "train.log"

    return {
        "project_root": str(project_root),
        "experiment_controls_path": str(experiment_controls_path),
        "data_controls_path": str(data_controls_path),
        "battery_controls_path": str(battery_controls_path),
        "train_controls_path": str(train_controls_path),
        "checkpoint_controls_path": str(checkpoint_controls_path),
        "result_json_path": str(result_json_path),
        "progress_json_path": str(progress_json_path),
        "log_path": str(log_path),
        "data_dir": str(data_dir),
        "save_dir": str(model_root),
        "model_root": str(model_root),
        "meta_dir": str(meta_dir),
        "checkpoint_root": str(checkpoint_root),
        "prediction_mode": str(prediction_mode),
        "experiment_name": str(experiment_name),
        "run_label": str(run_label),
        "env_name": str(env_name),
        "run_number": int(run_number),
    }


def build_train_mainline_command(
    launch_info: dict[str, Any],
    *,
    python_executable: str | None = None,
) -> list[str]:
    python_executable = python_executable or sys.executable
    return [
        str(python_executable),
        "-m",
        "scripts.run_train_mainline",
        "--experiment-controls",
        str(launch_info["experiment_controls_path"]),
        "--data-controls",
        str(launch_info["data_controls_path"]),
        "--battery-controls",
        str(launch_info["battery_controls_path"]),
        "--train-controls",
        str(launch_info["train_controls_path"]),
        "--checkpoint-controls",
        str(launch_info["checkpoint_controls_path"]),
        "--data-dir",
        str(launch_info["data_dir"]),
        "--save-dir",
        str(launch_info["save_dir"]),
        "--result-json",
        str(launch_info["result_json_path"]),
        "--env-name",
        str(launch_info["env_name"]),
        "--run-number",
        str(launch_info["run_number"]),
    ]


def load_train_mainline_result(result_json_path) -> dict[str, Any]:
    return json.loads(Path(result_json_path).read_text(encoding="utf-8"))


def _load_progress_payload(progress_json_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(progress_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _format_progress_summary(payload: dict[str, Any]) -> str:
    interaction_step = int(payload.get("interaction_step", 0))
    target_interactions = max(int(payload.get("target_interactions", 0)), 1)
    episodes_completed = int(payload.get("episodes_completed", 0))
    avg_reward = float(payload.get("avg_reward", 0.0))
    steps_per_sec = float(payload.get("steps_per_sec", 0.0))
    percent = 100.0 * interaction_step / target_interactions
    status = str(payload.get("status", "running"))
    estimated_end_time = payload.get("estimated_end_time")
    remaining_seconds = payload.get("remaining_seconds")
    if estimated_end_time:
        if remaining_seconds is None:
            eta_segment = f" | ETA={estimated_end_time}"
        else:
            eta_segment = f" | ETA={estimated_end_time} ({float(remaining_seconds):.1f}s)"
    else:
        eta_segment = " | ETA=--"
    return (
        f"[train:{status}] {interaction_step}/{target_interactions} iters ({percent:5.1f}%) | "
        f"episodes={episodes_completed} | avg_reward={avg_reward:8.3f} | steps/s={steps_per_sec:7.1f}"
        f"{eta_segment}"
    )


def _monitor_process_progress(
    process: subprocess.Popen[str],
    *,
    progress_json_path: Path,
    summary_interval_s: float,
    progress_episode_interval: int,
) -> dict[str, Any] | None:
    last_mtime_ns = -1
    last_print_time = 0.0
    last_summary = ""
    last_payload: dict[str, Any] | None = None
    next_episode_report = max(int(progress_episode_interval), 1)

    while process.poll() is None:
        payload = _load_progress_payload(progress_json_path)
        if payload is not None and progress_json_path.exists():
            last_payload = payload
            stat = progress_json_path.stat()
            summary = _format_progress_summary(payload)
            now = time.monotonic()
            episodes_completed = int(payload.get("episodes_completed", 0))
            status = str(payload.get("status", "running"))
            should_print = False
            if status != "running":
                should_print = True
            elif episodes_completed > 0 and episodes_completed >= next_episode_report:
                should_print = True

            if (
                should_print
                and stat.st_mtime_ns != last_mtime_ns
                and (now - last_print_time >= summary_interval_s)
            ):
                print(summary)
                last_summary = summary
                last_mtime_ns = stat.st_mtime_ns
                last_print_time = now
                while next_episode_report <= episodes_completed:
                    next_episode_report += max(int(progress_episode_interval), 1)
        time.sleep(0.5)

    payload = _load_progress_payload(progress_json_path)
    if payload is not None:
        last_payload = payload
        summary = _format_progress_summary(payload)
        if summary != last_summary:
            print(summary)
    return last_payload


def run_external_train_mainline(
    *,
    project_root,
    experiment_controls: dict[str, Any],
    data_controls: dict[str, Any],
    train_controls: dict[str, Any],
    battery_controls: dict[str, Any] | None = None,
    checkpoint_controls: dict[str, Any] | None = None,
    data_dir=None,
    save_dir=None,
    env_name: str = "GridTrainMainline",
    run_number: int = 1,
    python_executable: str | None = None,
    stream_output: bool = True,
    summary_interval_s: float = 2.0,
) -> dict[str, Any]:
    launch_info = prepare_train_mainline_launch(
        project_root=project_root,
        experiment_controls=experiment_controls,
        data_controls=data_controls,
        battery_controls=battery_controls,
        train_controls=train_controls,
        checkpoint_controls=checkpoint_controls,
        data_dir=data_dir,
        save_dir=save_dir,
        env_name=env_name,
        run_number=run_number,
    )
    command = build_train_mainline_command(launch_info, python_executable=python_executable)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_path = Path(launch_info["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_progress = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if stream_output:
            print(f"Training log: {log_path}")
            last_progress = _monitor_process_progress(
                process,
                progress_json_path=Path(launch_info["progress_json_path"]),
                summary_interval_s=float(summary_interval_s),
                progress_episode_interval=int(train_controls.get("progress_episode_interval", 10)),
            )
        returncode = process.wait()

    if last_progress is None:
        last_progress = _load_progress_payload(Path(launch_info["progress_json_path"]))
    result = (
        load_train_mainline_result(launch_info["result_json_path"])
        if Path(launch_info["result_json_path"]).exists()
        else None
    )
    launch_metadata = {
        "pid": int(process.pid),
        "returncode": int(returncode),
        "command": command,
        "launch_info": launch_info,
        "result": result,
        "last_progress": last_progress,
        "stream_output": bool(stream_output),
    }

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return launch_metadata
