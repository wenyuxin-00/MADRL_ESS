"""CLI entrypoint for the high-throughput training mainline."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
    category=FutureWarning,
)

import torch

from configs import compose_experiment_config, print_experiment_summary
from scripts.builder import build_train_runner
from scripts.checkpoints import resolve_checkpoint_to_load
from scripts.utils.grid_notebook_workflow import apply_notebook_experiment_settings, ensure_forecast_ready
from scripts.utils.project_paths import get_checkpoint_root, get_tensorboard_run_dir, project_root
from scripts.utils.torch_runtime import configure_torch_runtime, describe_device


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _apply_train_controls(cfg, train_controls: dict[str, Any]) -> None:
    field_names = (
        "train_episodes",
        "max_train_steps",
        "num_envs",
        "vec_env_type",
        "batch_size",
        "buffer_size",
        "update_interval",
        "updates_per_step",
        "actor_lr",
        "critic_lr",
        "noise_std_init",
        "noise_std_min",
        "noise_decay_steps",
        "use_noise_decay",
        "show_progress",
        "progress_postfix_interval",
    )
    for field_name in field_names:
        if field_name in train_controls:
            setattr(cfg.train, field_name, train_controls[field_name])
    if "policy_update_freq" in train_controls:
        cfg.algo.policy_update_freq = int(train_controls["policy_update_freq"])


def _build_result_payload(
    *,
    cfg,
    experiment_controls: dict[str, Any],
    data_controls: dict[str, Any],
    train_controls: dict[str, Any],
    runtime_state,
    forecast_ready,
    summary: dict[str, Any],
    runner,
    episodes_completed: int,
    save_dir: Path,
    env_name: str,
    run_number: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "algorithm": str(cfg.algo.name),
        "episodes_completed": int(episodes_completed),
        "saved_episode_tag": int(episodes_completed),
        "save_dir": str(save_dir),
        "checkpoint_info": resolve_checkpoint_to_load(save_dir, cfg.algo.name, episode_tag=episodes_completed),
        "tensorboard_dir": str(
            get_tensorboard_run_dir(
                cfg.algo.name,
                env_name,
                run_number=run_number,
                seed=seed,
                root=project_root(),
            )
        ),
        "device": str(cfg.runtime.device),
        "vec_env": type(runner.env).__name__,
        "perf_summary": dict(runner.perf_summary),
        "summary": summary,
        "device_info": describe_device(runtime_state),
        "experiment_controls": dict(experiment_controls),
        "data_controls": dict(data_controls),
        "train_controls": dict(train_controls),
        "forecast_ready": forecast_ready,
        "pid": int(os.getpid()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the accelerated Grid MADRL training mainline.")
    parser.add_argument("--experiment-controls", required=True)
    parser.add_argument("--data-controls", required=True)
    parser.add_argument("--train-controls", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--result-json")
    parser.add_argument("--env-name", default="GridTrainMainline")
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
    train_controls = _load_json(args.train_controls)

    seed = int(experiment_controls.get("seed", 0))
    data_dir = Path(args.data_dir).resolve() if args.data_dir is not None else (project_root() / "data")
    save_dir = (
        Path(args.save_dir).resolve()
        if args.save_dir is not None
        else get_checkpoint_root(project_root()) / f"{experiment_controls.get('algorithm') or 'MATD3'}_Grid_Mainline"
    )
    result_json = (
        Path(args.result_json).resolve()
        if args.result_json is not None
        else save_dir / "train_mainline_result.json"
    )

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
    applied_controls = apply_notebook_experiment_settings(
        cfg,
        prediction_mode=str(data_controls.get("prediction_mode", "perfect")),
        test_start_date=data_controls.get("test_start_date"),
        test_end_date=data_controls.get("test_end_date"),
        agent_profiles=list(data_controls.get("agent_profiles", cfg.data.agent_profiles)),
        load_scale=data_controls.get("load_scale", cfg.data.load_scale or [1.0] * cfg.env.num_agents),
        pv_scale=data_controls.get("pv_scale", cfg.data.pv_scale or [1.0] * cfg.env.num_agents),
        storage_scale=data_controls.get(
            "storage_scale",
            cfg.data.storage_scale or [1.0] * cfg.env.num_agents,
        ),
        future_horizon=int(data_controls.get("future_horizon", cfg.env.future_horizon)),
        train_year=data_controls.get("train_year"),
        test_year=data_controls.get("test_year"),
    )
    _apply_train_controls(cfg, train_controls)
    cfg.train.noise_decay_steps = cfg.train.train_episodes * cfg.env.episode_limit

    runtime_state = configure_torch_runtime(
        cfg,
        device=experiment_controls.get("device_request"),
        seed=seed,
        require_cuda=experiment_controls.get("require_cuda"),
    )
    forecast_ready = ensure_forecast_ready(cfg)
    summary = print_experiment_summary(cfg)
    summary["applied_controls"] = applied_controls
    summary["device_info"] = describe_device(runtime_state)

    runner = build_train_runner(cfg, seed=seed, env_name=args.env_name, number=args.run_number)
    episodes_completed = 0
    try:
        episodes_completed = runner.run()
        save_dir.mkdir(parents=True, exist_ok=True)
        runner.save_model(str(save_dir), episode=episodes_completed)
        result_payload = _build_result_payload(
            cfg=cfg,
            experiment_controls=experiment_controls,
            data_controls=data_controls,
            train_controls=train_controls,
            runtime_state=runtime_state,
            forecast_ready=forecast_ready,
            summary=summary,
            runner=runner,
            episodes_completed=episodes_completed,
            save_dir=save_dir,
            env_name=args.env_name,
            run_number=int(args.run_number),
            seed=seed,
        )
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(
            json.dumps(result_payload, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(json.dumps(result_payload, indent=2, default=_json_default))
        return 0
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
