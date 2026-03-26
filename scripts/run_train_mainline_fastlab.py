"""CLI entrypoint for the exact-equivalence MADRL fast-lab training path."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)

import torch

from configs import compose_experiment_config, print_experiment_summary
from scripts.builder_fastlab import build_train_runner_fastlab
from scripts.run_train_mainline import (
    _apply_model_controls,
    _apply_runtime_controls,
    _apply_train_controls,
    _build_result_payload,
    _json_default,
    _load_json,
    _resolve_save_dir,
    _write_json,
)
from scripts.utils.grid_notebook_workflow import apply_notebook_experiment_settings, ensure_forecast_ready
from scripts.utils.madrl_observation_cache_lab import build_or_load_observation_cache
from scripts.utils.project_paths import project_root
from scripts.utils.torch_runtime import configure_torch_runtime, describe_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exact-equivalence Grid MADRL fast-lab.")
    parser.add_argument("--experiment-controls", required=True)
    parser.add_argument("--data-controls", required=True)
    parser.add_argument("--battery-controls")
    parser.add_argument("--train-controls", required=True)
    parser.add_argument("--checkpoint-controls")
    parser.add_argument("--data-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--result-json")
    parser.add_argument("--env-name", default="GridTrainMainlineFastLab")
    parser.add_argument("--run-number", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except (AttributeError, RuntimeError):
        pass

    experiment_controls = _load_json(args.experiment_controls)
    data_controls = _load_json(args.data_controls)
    battery_controls = _load_json(args.battery_controls) if args.battery_controls else {}
    train_controls = _load_json(args.train_controls)
    checkpoint_controls = _load_json(args.checkpoint_controls) if args.checkpoint_controls else {}

    seed = int(experiment_controls.get("seed", 0))
    data_dir = Path(args.data_dir).resolve() if args.data_dir is not None else (project_root() / "data")
    save_dir = _resolve_save_dir(
        args=args,
        experiment_controls=experiment_controls,
        data_controls=data_controls,
        train_controls=train_controls,
        checkpoint_controls=checkpoint_controls,
    )
    meta_dir = (save_dir / "_meta").resolve()
    result_json = Path(args.result_json).resolve() if args.result_json is not None else meta_dir / "train_result.json"
    log_path = meta_dir / "train.log"
    progress_json = meta_dir / "progress.json"
    reward_summary_path = meta_dir / "train_reward_summary.json"
    meta_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_experiment_config(
        profile=str(train_controls.get("profile", "base")),
        algorithm=experiment_controls.get("algorithm"),
        model_family=str(train_controls.get("model_family", "mlp")),
        vec_env_type=train_controls.get("vec_env_type"),
        data_dir=data_dir,
        device=experiment_controls.get("device_request"),
        runtime_mode=str(experiment_controls.get("runtime_mode", "performance")),
        seed=seed,
        require_cuda=experiment_controls.get("require_cuda"),
    )
    _apply_model_controls(cfg, experiment_controls.get("model_controls"))
    _apply_runtime_controls(cfg, experiment_controls.get("runtime_controls"))
    applied_controls = apply_notebook_experiment_settings(
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
    cfg.train.noise_decay_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.runtime.progress_state_path = str(progress_json)

    runtime_state = configure_torch_runtime(
        cfg,
        device=experiment_controls.get("device_request"),
        seed=seed,
        require_cuda=experiment_controls.get("require_cuda"),
    )
    forecast_ready = ensure_forecast_ready(cfg)

    observation_cache_mode = str(experiment_controls.get("observation_cache_mode", "precomputed_exact"))
    refresh_observation_cache = bool(experiment_controls.get("refresh_observation_cache", False))
    train_info_mode = str(experiment_controls.get("train_info_mode", "minimal"))
    fast_grid_core = bool(experiment_controls.get("fast_grid_core", True))
    observation_cache_root = experiment_controls.get("observation_cache_root")
    if observation_cache_mode != "precomputed_exact":
        raise ValueError(
            f"Unsupported observation_cache_mode '{observation_cache_mode}', expected 'precomputed_exact'."
        )

    cache_result = build_or_load_observation_cache(
        cfg,
        split="train",
        forecast_ready=forecast_ready,
        refresh=refresh_observation_cache,
        root=observation_cache_root,
    )
    cfg.runtime.fastlab_observation_cache_dir = str(cache_result.cache_dir)
    cfg.runtime.fastlab_train_info_mode = train_info_mode
    cfg.runtime.fastlab_fast_grid_core = fast_grid_core

    summary = print_experiment_summary(cfg)
    summary["applied_controls"] = applied_controls
    summary["device_info"] = describe_device(runtime_state)
    summary["battery_controls"] = dict(battery_controls)
    summary["checkpoint_controls"] = dict(checkpoint_controls)
    summary["model_root"] = str(save_dir)
    summary["meta_dir"] = str(meta_dir)
    summary["log_path"] = str(log_path)
    summary["run_label"] = str(save_dir.name)
    summary["observation_cache_mode"] = observation_cache_mode
    summary["train_info_mode"] = train_info_mode
    summary["fast_grid_core"] = fast_grid_core
    summary["observation_cache_dir"] = str(cache_result.cache_dir)

    runner = build_train_runner_fastlab(cfg, seed=seed, env_name=args.env_name, number=args.run_number)
    episodes_completed = 0
    try:
        episodes_completed = runner.run()
        save_dir.mkdir(parents=True, exist_ok=True)
        runner.save_model(str(save_dir), episode=episodes_completed)
        reward_summary_payload = runner.build_reward_summary()
        _write_json(reward_summary_path, reward_summary_payload)
        result_payload = _build_result_payload(
            cfg=cfg,
            experiment_controls=experiment_controls,
            data_controls=data_controls,
            battery_controls=battery_controls,
            train_controls=train_controls,
            checkpoint_controls=checkpoint_controls,
            runtime_state=runtime_state,
            forecast_ready=forecast_ready,
            summary=summary,
            runner=runner,
            episodes_completed=episodes_completed,
            save_dir=save_dir,
            meta_dir=meta_dir,
            log_path=log_path,
            reward_summary_path=reward_summary_path,
            env_name=args.env_name,
            run_number=int(args.run_number),
            seed=seed,
            applied_controls=applied_controls,
        )
        result_payload["perf_summary"].update(
            {
                "cache_build_time_s": float(cache_result.cache_build_time_s),
                "cache_hit": bool(cache_result.cache_hit),
                "train_info_mode": train_info_mode,
                "observation_cache_mode": observation_cache_mode,
            }
        )
        _write_json(result_json, result_payload)
        print(json.dumps(result_payload, indent=2, default=_json_default))
        return 0
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
