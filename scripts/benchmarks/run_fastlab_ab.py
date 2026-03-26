"""Run cold-cache vs warm-cache benchmarks for the fast-lab-backed MADRL mainline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from configs import compose_experiment_config
from scripts.run_train_mainline import (
    _apply_model_controls,
    _apply_runtime_controls,
    _apply_train_controls,
)
from scripts.utils.grid_notebook_workflow import (
    apply_notebook_experiment_settings,
    collect_madrl_rollout,
    ensure_forecast_ready,
)
from scripts.utils.madrl_perf_lab import (
    build_candidate_experiment_name,
    build_perf_leaderboard,
    merge_control_overrides,
    recommend_perf_candidate,
    summarize_rollout_metrics,
)
from scripts.utils.project_paths import project_root
from scripts.utils.torch_runtime import configure_torch_runtime, describe_device


BENCHMARK_CASES = (
    ("mainline_cold", "scripts.run_train_mainline"),
    ("mainline_warm", "scripts.run_train_mainline"),
)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return target


def _slug_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_dir() -> Path:
    return (project_root() / "artifacts" / "benchmarks" / "fastlab_ab" / _slug_timestamp()).resolve()


def _build_candidate_controls(
    *,
    experiment_controls: Mapping[str, Any],
    train_controls: Mapping[str, Any],
    checkpoint_controls: Mapping[str, Any],
    candidate_name: str,
    episode_budget: int,
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_train = merge_control_overrides(
        train_controls,
        {
            "train_episodes": int(episode_budget),
            "max_train_steps": None,
        },
    )
    experiment_name_base = str(checkpoint_controls.get("experiment_name", "grid_mainline"))
    experiment_name = build_candidate_experiment_name(
        f"{experiment_name_base}_ab_ep{int(episode_budget)}",
        candidate_name,
    )
    candidate_checkpoint = merge_control_overrides(
        checkpoint_controls,
        {
            "experiment_name": experiment_name,
        },
    )

    refresh = candidate_name.endswith("_cold")
    candidate_experiment = merge_control_overrides(
        experiment_controls,
        {
            "observation_cache_mode": "precomputed_exact",
            "refresh_observation_cache": refresh,
            "train_info_mode": "minimal",
            "fast_grid_core": True,
            "observation_cache_root": str(cache_root),
        },
    )
    return candidate_experiment, candidate_train, candidate_checkpoint


def _write_controls_bundle(
    *,
    output_dir: Path,
    candidate_name: str,
    episode_budget: int,
    experiment_controls: Mapping[str, Any],
    data_controls: Mapping[str, Any],
    battery_controls: Mapping[str, Any],
    train_controls: Mapping[str, Any],
    checkpoint_controls: Mapping[str, Any],
) -> dict[str, Path]:
    controls_dir = output_dir / f"controls_ep{int(episode_budget)}_{candidate_name}"
    controls_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "runs" / f"ep{int(episode_budget)}_{candidate_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return {
        "experiment_controls": _write_json(controls_dir / "experiment_controls.json", experiment_controls),
        "data_controls": _write_json(controls_dir / "data_controls.json", data_controls),
        "battery_controls": _write_json(controls_dir / "battery_controls.json", battery_controls),
        "train_controls": _write_json(controls_dir / "train_controls.json", train_controls),
        "checkpoint_controls": _write_json(controls_dir / "checkpoint_controls.json", checkpoint_controls),
        "result_json": controls_dir / "train_result.json",
        "save_dir": save_dir,
    }


def _build_command(
    *,
    module_name: str,
    python_executable: str,
    control_paths: Mapping[str, Path],
    data_dir: Path | None,
    env_name: str,
    run_number: int,
) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        module_name,
        "--experiment-controls",
        str(control_paths["experiment_controls"]),
        "--data-controls",
        str(control_paths["data_controls"]),
        "--train-controls",
        str(control_paths["train_controls"]),
        "--save-dir",
        str(control_paths["save_dir"]),
        "--result-json",
        str(control_paths["result_json"]),
        "--env-name",
        env_name,
        "--run-number",
        str(int(run_number)),
    ]
    if data_dir is not None:
        command.extend(["--data-dir", str(data_dir)])
    if control_paths.get("battery_controls") is not None:
        command.extend(["--battery-controls", str(control_paths["battery_controls"])])
    if control_paths.get("checkpoint_controls") is not None:
        command.extend(["--checkpoint-controls", str(control_paths["checkpoint_controls"])])
    return command


def _run_training_case(
    *,
    module_name: str,
    python_executable: str,
    control_paths: Mapping[str, Path],
    data_dir: Path | None,
    env_name: str,
    run_number: int,
) -> dict[str, Any]:
    command = _build_command(
        module_name=module_name,
        python_executable=python_executable,
        control_paths=control_paths,
        data_dir=data_dir,
        env_name=env_name,
        run_number=run_number,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root(),
        check=False,
        text=True,
    )
    wall_time_s = float(time.perf_counter() - started)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    result = _load_json(control_paths["result_json"])
    return {
        "command": command,
        "wall_time_s": wall_time_s,
        "result": result,
    }


def _compose_cfg_for_rollout(
    *,
    experiment_controls: Mapping[str, Any],
    data_controls: Mapping[str, Any],
    battery_controls: Mapping[str, Any],
    train_controls: Mapping[str, Any],
    data_dir: Path | None,
) -> Any:
    cfg = compose_experiment_config(
        profile=str(train_controls.get("profile", "base")),
        algorithm=experiment_controls.get("algorithm"),
        model_family=str(train_controls.get("model_family", "mlp")),
        vec_env_type=train_controls.get("vec_env_type"),
        data_dir=data_dir if data_dir is not None else (project_root() / "data"),
        device=experiment_controls.get("device_request"),
        runtime_mode=str(experiment_controls.get("runtime_mode", "performance")),
        seed=int(experiment_controls.get("seed", 0)),
        require_cuda=experiment_controls.get("require_cuda"),
    )
    _apply_model_controls(cfg, experiment_controls.get("model_controls"))
    _apply_runtime_controls(cfg, experiment_controls.get("runtime_controls"))
    apply_notebook_experiment_settings(
        cfg,
        prediction_mode=str(data_controls.get("prediction_mode", "perfect")),
        test_start_date=data_controls.get("test_start_date"),
        test_end_date=data_controls.get("test_end_date"),
        agent_profiles=list(data_controls.get("agent_profiles", cfg.data.agent_profiles)),
        load_scale=data_controls.get("load_scale", cfg.data.load_scale or [1.0] * cfg.env.num_agents),
        pv_scale=data_controls.get("pv_scale", cfg.data.pv_scale or [1.0] * cfg.env.num_agents),
        battery_controls=battery_controls,
        future_horizon=int(data_controls.get("future_horizon", cfg.env.future_horizon)),
        train_year=data_controls.get("train_year"),
        test_year=data_controls.get("test_year"),
    )
    _apply_train_controls(cfg, train_controls)
    runtime_state = configure_torch_runtime(
        cfg,
        device=experiment_controls.get("device_request"),
        seed=int(experiment_controls.get("seed", 0)),
        require_cuda=experiment_controls.get("require_cuda"),
    )
    ensure_forecast_ready(cfg)
    cfg.runtime.device_info = describe_device(runtime_state)
    return cfg


def _write_rollout_outputs(
    *,
    output_dir: Path,
    candidate_name: str,
    episode_budget: int,
    rollout,
    rollout_metrics: Mapping[str, Any],
) -> Path:
    rollout_dir = output_dir / f"rollout_ep{int(episode_budget)}_{candidate_name}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    step_path = rollout_dir / "step.csv"
    agent_path = rollout_dir / "agent.csv"
    grid_path = rollout_dir / "grid.csv"
    summary_path = rollout_dir / "summary.csv"
    metrics_path = rollout_dir / "metrics.json"
    rollout.step_df.to_csv(step_path, index=False)
    rollout.agent_df.to_csv(agent_path, index=False)
    rollout.grid_df.to_csv(grid_path, index=False)
    rollout.summary.to_csv(summary_path, index=False)
    _write_json(
        metrics_path,
        {
            "candidate_name": candidate_name,
            "episode_budget": int(episode_budget),
            "meta": rollout.meta,
            "metrics": dict(rollout_metrics),
            "step_path": step_path,
            "agent_path": agent_path,
            "grid_path": grid_path,
            "summary_path": summary_path,
        },
    )
    return metrics_path


def _build_benchmark_row(
    *,
    candidate_name: str,
    episode_budget: int,
    training_result: Mapping[str, Any],
    train_run: Mapping[str, Any],
    rollout_metrics: Mapping[str, Any],
    rollout_summary_path: Path,
) -> dict[str, Any]:
    perf_summary = dict(training_result.get("perf_summary", {}))
    reward_summary_path = Path(training_result["reward_summary_path"])
    return {
        "candidate_name": candidate_name,
        "episode_budget": int(episode_budget),
        "status": "ok",
        "algorithm": str(training_result["algorithm"]),
        "prediction_mode": str(training_result["prediction_mode"]),
        "evaluation_mode": str(training_result["evaluation_mode"]),
        "experiment_name": str(training_result["experiment_name"]),
        "run_label": str(training_result["run_label"]),
        "result_json_path": str(train_run["result_json_path"]),
        "reward_summary_path": str(reward_summary_path),
        "rollout_summary_path": str(rollout_summary_path),
        "elapsed_seconds": float(training_result.get("elapsed_seconds", train_run["wall_time_s"])),
        "total_wall_time_s": float(train_run["wall_time_s"]),
        "steps_per_sec": float(perf_summary.get("steps_per_sec", float("nan"))),
        "avg_env_ms_per_iter": float(perf_summary.get("avg_env_ms_per_iter", float("nan"))),
        "avg_update_ms_per_call": float(perf_summary.get("avg_update_ms_per_call", float("nan"))),
        "cache_build_time_s": float(perf_summary.get("cache_build_time_s", 0.0)),
        "cache_hit": bool(perf_summary.get("cache_hit", False)),
        "episodes_completed": int(training_result["episodes_completed"]),
        **{key: value for key, value in rollout_metrics.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cold-cache vs warm-cache performance for the fast-lab-backed MADRL mainline.")
    parser.add_argument("--experiment-controls", required=True)
    parser.add_argument("--data-controls", required=True)
    parser.add_argument("--battery-controls")
    parser.add_argument("--train-controls", required=True)
    parser.add_argument("--checkpoint-controls")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--episode-budgets", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--env-name-prefix", default="GridFastLabAB")
    parser.add_argument("--run-number", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    experiment_controls = _load_json(args.experiment_controls)
    data_controls = _load_json(args.data_controls)
    battery_controls = _load_json(args.battery_controls)
    train_controls = _load_json(args.train_controls)
    checkpoint_controls = _load_json(args.checkpoint_controls)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "cache"
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None

    rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    for episode_budget in [int(value) for value in args.episode_budgets]:
        for candidate_name, module_name in BENCHMARK_CASES:
            candidate_experiment, candidate_train, candidate_checkpoint = _build_candidate_controls(
                experiment_controls=experiment_controls,
                train_controls=train_controls,
                checkpoint_controls=checkpoint_controls,
                candidate_name=candidate_name,
                episode_budget=episode_budget,
                cache_root=cache_root,
            )
            control_paths = _write_controls_bundle(
                output_dir=output_dir,
                candidate_name=candidate_name,
                episode_budget=episode_budget,
                experiment_controls=candidate_experiment,
                data_controls=data_controls,
                battery_controls=battery_controls,
                train_controls=candidate_train,
                checkpoint_controls=candidate_checkpoint,
            )
            refresh_cache = candidate_name.endswith("_cold")
            print(
                f"[fastlab_ab] running {candidate_name} ep{episode_budget} via {module_name} "
                f"(refresh_cache={refresh_cache})"
            )
            train_run = _run_training_case(
                module_name=module_name,
                python_executable=args.python_executable,
                control_paths=control_paths,
                data_dir=data_dir,
                env_name=f"{args.env_name_prefix}_{candidate_name}_ep{episode_budget}",
                run_number=args.run_number,
            )
            training_result = dict(train_run["result"])
            cfg = _compose_cfg_for_rollout(
                experiment_controls=candidate_experiment,
                data_controls=data_controls,
                battery_controls=battery_controls,
                train_controls=candidate_train,
                data_dir=data_dir,
            )
            rollout = collect_madrl_rollout(
                cfg,
                model_root=training_result["model_root"],
                algorithm=training_result["algorithm"],
                episode_tag=int(training_result["saved_episode_tag"]),
            )
            rollout_metrics = summarize_rollout_metrics(rollout)
            rollout_summary_path = _write_rollout_outputs(
                output_dir=output_dir,
                candidate_name=candidate_name,
                episode_budget=episode_budget,
                rollout=rollout,
                rollout_metrics=rollout_metrics,
            )
            row = _build_benchmark_row(
                candidate_name=candidate_name,
                episode_budget=episode_budget,
                training_result=training_result,
                train_run={
                    "wall_time_s": train_run["wall_time_s"],
                    "result_json_path": control_paths["result_json"],
                },
                rollout_metrics=rollout_metrics,
                rollout_summary_path=rollout_summary_path,
            )
            rows.append(row)
            run_manifest.append(
                {
                    "episode_budget": int(episode_budget),
                    "candidate_name": candidate_name,
                    "module_name": module_name,
                    "command": train_run["command"],
                    "result_json_path": str(control_paths["result_json"]),
                    "reward_summary_path": str(training_result["reward_summary_path"]),
                    "rollout_summary_path": str(rollout_summary_path),
                }
            )

    leaderboard = build_perf_leaderboard(rows)
    recommendation_by_budget: dict[str, Any] = {}
    for episode_budget, frame in leaderboard.groupby("episode_budget", sort=True):
        recommendation_by_budget[str(int(episode_budget))] = recommend_perf_candidate(
            frame.reset_index(drop=True),
            baseline_name="mainline_cold",
        )

    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    summary_payload = {
        "output_dir": str(output_dir),
        "episode_budgets": [int(value) for value in args.episode_budgets],
        "leaderboard_path": str(leaderboard_path),
        "rows": rows,
        "recommendation_by_budget": recommendation_by_budget,
        "run_manifest": run_manifest,
    }
    summary_path = _write_json(output_dir / "benchmark_summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2, default=_json_default))
    print(f"[fastlab_ab] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

