"""Helpers for launching the training mainline outside notebooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.utils.project_paths import get_checkpoint_root, get_training_artifact_root


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
    data_dir=None,
    save_dir=None,
    env_name: str = "GridTrainMainline",
    run_number: int = 1,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    data_dir = Path(data_dir).resolve() if data_dir is not None else (project_root / "data").resolve()

    algorithm = _default_algorithm(experiment_controls, train_controls)
    default_save_dir = get_checkpoint_root(project_root) / f"{algorithm}_Grid_Mainline"
    save_dir = Path(save_dir).resolve() if save_dir is not None else default_save_dir.resolve()

    launch_dir = (
        get_training_artifact_root(project_root)
        / "launches"
        / f"mainline_seed_{int(experiment_controls.get('seed', 0))}_{int(time.time() * 1000)}"
    )
    launch_dir.mkdir(parents=True, exist_ok=True)

    experiment_controls_path = _write_json(launch_dir / "experiment_controls.json", experiment_controls)
    data_controls_path = _write_json(launch_dir / "data_controls.json", data_controls)
    train_controls_path = _write_json(launch_dir / "train_controls.json", train_controls)
    result_json_path = launch_dir / "train_result.json"

    return {
        "project_root": str(project_root),
        "experiment_controls_path": str(experiment_controls_path),
        "data_controls_path": str(data_controls_path),
        "train_controls_path": str(train_controls_path),
        "result_json_path": str(result_json_path),
        "data_dir": str(data_dir),
        "save_dir": str(save_dir),
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
        "--train-controls",
        str(launch_info["train_controls_path"]),
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


def _relay_process_output(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    try:
        while True:
            chunk = process.stdout.read(1)
            if chunk == "" and process.poll() is not None:
                break
            if not chunk:
                continue
            sys.stdout.write(chunk)
            sys.stdout.flush()
    finally:
        process.stdout.close()


def run_external_train_mainline(
    *,
    project_root,
    experiment_controls: dict[str, Any],
    data_controls: dict[str, Any],
    train_controls: dict[str, Any],
    data_dir=None,
    save_dir=None,
    env_name: str = "GridTrainMainline",
    run_number: int = 1,
    python_executable: str | None = None,
    stream_output: bool = True,
) -> dict[str, Any]:
    launch_info = prepare_train_mainline_launch(
        project_root=project_root,
        experiment_controls=experiment_controls,
        data_controls=data_controls,
        train_controls=train_controls,
        data_dir=data_dir,
        save_dir=save_dir,
        env_name=env_name,
        run_number=run_number,
    )
    command = build_train_mainline_command(launch_info, python_executable=python_executable)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    popen_kwargs: dict[str, Any] = {
        "cwd": str(project_root),
        "env": env,
    }
    if stream_output:
        popen_kwargs.update(
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 0,
            }
        )

    process = subprocess.Popen(command, **popen_kwargs)
    if stream_output:
        _relay_process_output(process)
    returncode = process.wait()

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
        "stream_output": bool(stream_output),
    }

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return launch_metadata
